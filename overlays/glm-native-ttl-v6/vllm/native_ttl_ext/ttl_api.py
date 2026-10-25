# SparkFleet reference overlay: modified deployment/latency behavior; see overlays/README.md and NOTICE.
# SPDX-License-Identifier: Apache-2.0
"""Request-local Anthropic TTL rendering and eager engine admission.

Conservatively lease only whole native blocks in a verified common token prefix.
Never estimate boundaries from characters or modify the full inference prompt.
"""
import asyncio
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import time

from .native_ttl import TTLRejected
from .ttl_protocol import markers, CacheProtocolError
from .ttl_runtime import PLAN_KEY, config, reject_external, token_digest

state_var = ContextVar('native_ttl_request', default=None)


@dataclass
class State:
    payload: dict
    started_at: float
    receipt: dict | None = None
    engine: object = None
    internal_id: str | None = None


def prefix_payload(payload, marker):
    p = deepcopy(payload)
    path = marker.path
    p.pop('cache_control', None)
    if path[0] == 'tools':
        p['tools'] = p['tools'][:path[1] + 1]
        p['system'], p['messages'] = None, []
        p['tool_choice'] = None
    elif path[0] == 'system':
        p['system'] = p['system'][:path[1] + 1]
        p['messages'] = []
    else:
        p['messages'] = p['messages'][:path[1] + 1]
        if len(path) == 4:
            p['messages'][-1]['content'] = p['messages'][-1]['content'][:path[3] + 1]
    return p


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


async def map_plan(payload, full_ids, render_prefix, *, started_at, identity):
    ends = []
    for marker in markers(payload):
        ids = await render_prefix(prefix_payload(payload, marker))
        end = common_prefix(full_ids, ids)
        if not end:
            raise CacheProtocolError('Template did not preserve the marked token prefix')
        if ends and end < ends[-1][0]:
            raise CacheProtocolError('Template reordered cache breakpoints')
        if ends and end == ends[-1][0]:
            # Two schema markers may map to the same token boundary.
            ends[-1][1] = max(ends[-1][1], marker.ttl_seconds)
        else:
            ends.append([end, marker.ttl_seconds])
    return {'schema': 2, 'ends': ends, 'digest': token_digest(full_ids),
            'identity': identity, 'started_at': started_at,
            'strict': any('ttl' in _marker_control(payload, m) for m in markers(payload))}


async def render_with_ttl(serving, request, original_render):
    result = await original_render(request)
    state = state_var.get()
    if state is None or not isinstance(result, tuple):
        return result
    _, inputs = result
    if len(inputs) != 1:
        raise TTLRejected('TTL requires exactly one rendered prompt')
    full_ids = serving._extract_prompt_components(inputs[0]).token_ids
    if full_ids is None:
        raise TTLRejected('TTL requires tokenized text input')
    # Run through exactly the same converter, renderer and tokenizer. Turning
    # off generation suffixes prevents caching an assistant marker beyond the
    # selected content; token equality protects template/BPE boundary changes.
    async def render_prefix(payload):
        from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
        from vllm.entrypoints.anthropic.serving import _ContextCountRequest
        prefix = AnthropicMessagesRequest.model_validate(payload)
        chat = serving.to_chat_completion_request(prefix,
                merge_inline_system=serving._merge_inline_system,
                disabled_thinking_effort=getattr(state, 'disabled_effort', 'low'))
        chat = _ContextCountRequest.model_validate(chat.model_dump())
        chat.add_generation_prompt = False
        rendered = await serving.online_renderer.render_chat(chat)
        if not isinstance(rendered, tuple) or len(rendered[1]) != 1:
            raise TTLRejected('Cannot render selected cache prefix')
        return serving._extract_prompt_components(rendered[1][0]).token_ids
    cfg = config()
    plan = await map_plan(state.payload, full_ids, render_prefix,
                          started_at=state.started_at, identity=cfg['identity'])
    request.vllm_xargs = dict(request.vllm_xargs or {}, **{PLAN_KEY: plan})
    return result


async def serve_with_ttl(serving, request, raw_request, original):
    reject_external(request.vllm_xargs)
    payload = request.model_dump(exclude_none=True)
    found = markers(payload)
    if not found:
        return await original(request, raw_request)
    if config() is None:
        # Legacy ephemeral without explicit ttl retains its original APC path.
        if any('ttl' in _marker_control(payload, m) for m in found):
            raise TTLRejected('Explicit TTL requires enabled native TTL candidate')
        return await original(request, raw_request)
    state = State(payload, time.monotonic())
    token = state_var.set(state)
    try:
        result = await original(request, raw_request)
    except BaseException:
        await cleanup(state)
        raise
    finally:
        state_var.reset(token)
    if hasattr(result, '__aiter__'):
        return OwnedStream(result, state)
    await cleanup(state)
    return result


class OwnedStream:
    """Cleanup works even when closed before the first iteration.

    Do not keep a ContextVar token across a yield: ASGI may close a stream from
    a different task than the one that read its last event.
    """
    def __init__(self, source, state):
        self.source, self.state, self.closed = source, state, False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        token = state_var.set(self.state)
        try:
            return await anext(self.source)
        except BaseException:
            await self.aclose()
            raise
        finally:
            state_var.reset(token)

    async def aclose(self):
        if self.closed:
            return
        token = state_var.set(self.state)
        try:
            try:
                await self.source.aclose()
            finally:
                await cleanup(self.state)
            self.closed = True
        finally:
            state_var.reset(token)


def streaming_response(generator):
    from starlette.responses import StreamingResponse
    if not isinstance(generator, OwnedStream):
        return StreamingResponse(content=generator, media_type='text/event-stream')

    class ClosingStreamingResponse(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                # This also covers send errors before streaming starts.
                import anyio
                with anyio.CancelScope(shield=True):
                    await generator.aclose()
    return ClosingStreamingResponse(content=generator, media_type='text/event-stream')


def _marker_control(payload, marker):
    if marker.automatic:
        return payload['cache_control']
    value = payload
    for key in marker.path:
        value = value[key]
    return value['cache_control']


async def cleanup(state):
    if state.engine is not None and state.internal_id is not None:
        await state.engine.abort(state.internal_id, internal=True)
        state.internal_id = None


async def start_generate(engine, prompt, params, request_id, **kwargs):
    if PLAN_KEY not in (params.extra_args or {}):
        return engine.generate(prompt, params, request_id, **kwargs)
    state = state_var.get()
    if state is None or params.n != 1:
        raise TTLRejected('TTL requires the authenticated Anthropic single-output path')
    queue = await engine.add_request(request_id, prompt, params, **kwargs)
    state.engine, state.internal_id = engine, queue.request_id
    generator = engine.generate(prompt, params, request_id,
                                _ttl_admitted_queue=queue, **kwargs)
    async def consume():
        try:
            async for output in generator:
                if output.finished:
                    state.receipt = await engine.engine_core.call_utility_async(
                        'native_ttl_receipt', queue.request_id)
                yield output
        finally:
            await generator.aclose()
            await cleanup(state)
    return consume()


def usage_fields(usage, fields):
    state = state_var.get()
    if state is None or usage is None:
        return fields
    # No synthetic hit counter: vLLM supplies the actual prefix-cache read count.
    details = usage.prompt_tokens_details
    read = getattr(details, 'cached_tokens', None) if details is not None else None
    if read is None or state.receipt is None:
        # Missing receipt / disabled upstream detail: do not label generic APC
        # creation as a successfully retained 5m/1h prefix.
        result = dict(fields)
        result.pop('cache_creation', None)
        result.pop('cache_creation_input_tokens', None)
        result['input_tokens'] = usage.prompt_tokens - (read or 0)
        return result
    if not 0 <= read <= usage.prompt_tokens:
        raise TTLRejected('Invalid engine cache-read accounting')
    counts = {300: 0, 3600: 0}
    size = state.receipt['block_size']
    for end, seconds in state.receipt['published']:
        counts[seconds] += max(0, min(end, usage.prompt_tokens) - max(end - size, read))
    created = sum(counts.values())
    if read + created > usage.prompt_tokens:
        raise TTLRejected('Invalid TTL usage accounting')
    return dict(fields, input_tokens=usage.prompt_tokens-read-created,
                cache_read_input_tokens=read, cache_creation_input_tokens=created,
                cache_creation={'ephemeral_5m_input_tokens': counts[300],
                                'ephemeral_1h_input_tokens': counts[3600]})
