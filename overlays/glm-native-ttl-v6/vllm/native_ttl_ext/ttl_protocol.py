# SparkFleet reference overlay: modified deployment/latency behavior; see overlays/README.md and NOTICE.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only Anthropic cache-marker validation; no tokenization or model calls.

Positions are schema paths, NOT token offsets. A serving integration must map
them through the exact renderer and verified token boundaries before admission.
This module deliberately refuses unsupported marker placements.
"""
from dataclasses import dataclass


class CacheProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Marker:
    path: tuple
    ttl_seconds: int
    automatic: bool = False


def ttl_seconds(cache):
    if not isinstance(cache, dict) or cache.get('type') != 'ephemeral':
        raise CacheProtocolError('cache_control.type must be ephemeral')
    if set(cache) - {'type', 'ttl'}:
        raise CacheProtocolError('Unsupported cache_control field')
    ttl = cache.get('ttl', '5m')
    if ttl not in ('5m', '1h'):
        raise CacheProtocolError('cache_control.ttl must be 5m or 1h')
    return 300 if ttl == '5m' else 3600


def markers(payload):
    if not isinstance(payload, dict):
        raise CacheProtocolError('Object request required')
    found = []
    last_cacheable = None

    def visit(block, path, allowed):
        nonlocal last_cacheable
        if not isinstance(block, dict):
            return
        if allowed:
            last_cacheable = path
        if block.get('cache_control') is not None:
            if not allowed:
                raise CacheProtocolError('Unsupported cache marker placement')
            found.append(Marker(path, ttl_seconds(block['cache_control'])))
        # Arbitrary tool input/schema/JSON text is never traversed.
        if block.get('type') == 'tool_result' and isinstance(block.get('content'), list):
            if any(isinstance(x, dict) and x.get('cache_control') is not None for x in block['content']):
                raise CacheProtocolError('Nested tool-result markers require renderer support')

    tools, system, messages = (payload.get(k) for k in ('tools', 'system', 'messages'))
    if tools is not None and not isinstance(tools, list):
        raise CacheProtocolError('tools must be a list')
    if system is not None and not isinstance(system, (list, str)):
        raise CacheProtocolError('system must be text or blocks')
    if messages is not None and not isinstance(messages, list):
        raise CacheProtocolError('messages must be a list')
    for i, tool in enumerate(tools or []):
        visit(tool, ('tools', i), True)
    for i, block in enumerate(system or []):
        if isinstance(system, str):
            break
        visit(block, ('system', i), isinstance(block, dict) and block.get('type') == 'text')
    for i, message in enumerate(messages or []):
        if not isinstance(message, dict):
            raise CacheProtocolError('Message must be an object')
        content = message.get('content')
        if isinstance(content, str):
            last_cacheable = ('messages', i, 'content')
        elif isinstance(content, list):
            for j, block in enumerate(content):
                allowed = isinstance(block, dict) and block.get('type') in ('text', 'tool_use', 'tool_result')
                visit(block, ('messages', i, 'content', j), allowed)
    if payload.get('cache_control') is not None:
        # Combining automatic and explicit controls is deferred until rendering
        # and marker-overlap behavior have native integration coverage.
        if found:
            raise CacheProtocolError('Combined automatic and explicit cache controls not yet qualified')
        if last_cacheable is None:
            raise CacheProtocolError('No cacheable prefix')
        found.append(Marker(last_cacheable, ttl_seconds(payload['cache_control']), True))
    if len(found) > 4:
        raise CacheProtocolError('At most four cache breakpoints')
    if any(a.ttl_seconds < b.ttl_seconds for a, b in zip(found, found[1:])):
        raise CacheProtocolError('1h cache breakpoints must precede 5m breakpoints')
    return tuple(found)


def expand_boundaries(markers_with_token_ends, engine_hashes, block_size, *, prompt_tokens):
    """Convert verified token endpoints to full-block leases, longest TTL wins.

    The trailing partial block is not promised as a physical cache hit. Caller
    must report true full-block cache counts and any uncached remainder.
    """
    if type(block_size) is not int or block_size <= 0:
        raise CacheProtocolError('Positive block size required')
    if type(prompt_tokens) is not int or prompt_tokens < 0 or len(engine_hashes) != prompt_tokens // block_size:
        raise CacheProtocolError('Hash count must match the complete prompt blocks')
    out = {}
    previous_end, previous_ttl = 0, 3600
    for marker, end in markers_with_token_ends:
        if type(end) is not int or end <= previous_end or end > prompt_tokens:
            raise CacheProtocolError('Strictly increasing verified full-prompt token endpoints required')
        if marker.ttl_seconds not in (300, 3600) or marker.ttl_seconds > previous_ttl:
            raise CacheProtocolError('Invalid mixed TTL order')
        for key in engine_hashes[:end // block_size]:
            out[key] = max(out.get(key, 0), marker.ttl_seconds)
        previous_end, previous_ttl = end, marker.ttl_seconds
    return tuple(out.items())
