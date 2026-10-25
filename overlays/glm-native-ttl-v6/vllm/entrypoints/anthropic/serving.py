# SparkFleet reference overlay: modified deployment/latency behavior; see overlays/README.md and NOTICE.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/serving.py

"""Anthropic Messages API serving handler"""

import json
import logging
import time
import uuid
from dataclasses import replace
from collections.abc import AsyncGenerator
from typing import Any, get_args

import jinja2
from vllm.entrypoints.anthropic.compat import (
    context_candidate, glm_effort, prepare_messages, stream_messages,
    text_block, tool_arguments, has_hidden_history, history_key,
    restore_hidden_thinking, seal_thinking,
)
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicContextManagement,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicDelta,
    AnthropicDisabledThinkingEffort,
    AnthropicDisabledThinkingEffortOption,
    AnthropicEffort,
    AnthropicError,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicOutputConfig,
    AnthropicStreamEvent,
    AnthropicThinkingConfig,
    AnthropicTool,
    AnthropicToolChangeDefinition,
    AnthropicUsage,
)
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.generate.base.protocol import (
    JsonSchemaResponseFormat,
    ResponseFormat,
    StreamOptions,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.serve.exception_handling.utils import sanitize_message
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.renderers.online_renderer import OnlineRenderer

logger = logging.getLogger(__name__)


class _ContextCountRequest(ChatCompletionRequest):
    """Count the original input before editing; never used for generation."""

    def build_tok_params(self, model_config):
        return replace(super().build_tok_params(model_config),
                       max_total_tokens=None, max_output_tokens=0)


def _build_anthropic_usage(
    usage: UsageInfo | None,
) -> AnthropicUsage:
    """Build an AnthropicUsage from UsageInfo.

    Anthropic defines ``total_input == input_tokens + cache_read +
    cache_creation``.  vLLM's ``prompt_tokens`` is the total, so
    ``input_tokens = prompt_tokens - cache_read - cache_creation``.

    Cache fields are taken from ``UsageInfo.prompt_tokens_details``.
    When cache info is absent (e.g. ``--enable-prompt-tokens-details``
    off, or a streaming chunk that hasn't carried it yet), cache fields
    are left **unset** so ``exclude_unset=True`` serialization omits them
    entirely.

    ``completion_tokens`` follows ``UsageInfo`` and may be ``None`` on
    intermediate stream chunks; we coerce to ``0`` for the wire format.
    """
    kwargs = {}
    if usage is None:
        kwargs["input_tokens"] = 0
        kwargs["output_tokens"] = 0
    else:
        kwargs["output_tokens"] = usage.completion_tokens or 0
        input_tokens = usage.prompt_tokens

        if (details := usage.prompt_tokens_details) is not None:
            if (cache_read := details.cached_tokens) is not None:
                input_tokens -= cache_read
                kwargs["cache_read_input_tokens"] = cache_read

            if (cache_creation := details.created_cache_tokens) is not None:
                input_tokens -= cache_creation
                kwargs["cache_creation_input_tokens"] = cache_creation

        kwargs["input_tokens"] = max(0, input_tokens)
    from vllm.native_ttl_ext.ttl_api import usage_fields
    return AnthropicUsage(**usage_fields(usage, kwargs))


def wrap_data_with_event(data: str, event: str):
    return f"event: {event}\ndata: {data}\n\n"


class AnthropicServingMessages(OpenAIServingChat):
    """Handler for Anthropic Messages API requests"""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        online_renderer: "OnlineRenderer",
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        disabled_thinking_effort: AnthropicDisabledThinkingEffortOption = "auto",
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            response_role=response_role,
            online_renderer=online_renderer,
            request_logger=request_logger,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            reasoning_parser=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            tool_parser=tool_parser,
            enable_prompt_tokens_details=True,
            enable_force_include_usage=enable_force_include_usage,
            default_chat_template_kwargs=default_chat_template_kwargs,
        )
        self.stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }
        resolved_template = chat_template
        if resolved_template is None:
            resolved_template = getattr(online_renderer.renderer.get_tokenizer(), 'chat_template', None)
        self._merge_inline_system = self._detect_merge_inline_system(resolved_template)
        self._glm_content_stops = bool(resolved_template and '[gMASK]' in resolved_template and 'effective_reasoning_effort' in resolved_template)
        # Resolved lazily from the renderer when "auto".
        self._disabled_thinking_effort: AnthropicDisabledThinkingEffort | None = (
            None if disabled_thinking_effort == "auto" else disabled_thinking_effort
        )

    async def _get_disabled_thinking_effort(self) -> AnthropicDisabledThinkingEffort:
        if self._disabled_thinking_effort is None:
            self._disabled_thinking_effort = (
                await self._probe_disabled_thinking_effort()
            )
            logger.info(
                "Anthropic thinking.type=disabled maps to reasoning_effort=%r",
                self._disabled_thinking_effort,
            )
        return self._disabled_thinking_effort

    async def _probe_disabled_thinking_effort(self) -> AnthropicDisabledThinkingEffort:
        """Use ``low`` if the renderer rejects or ignores ``none``.

        ``none`` is ignored when it renders the same prompt as a thinking
        effort, e.g. GLM-5.3 treats unknown efforts as ``max``.
        """
        none_prompt = await self._render_probe_prompt("none")
        if none_prompt is None:
            return "low"
        for effort in get_args(AnthropicEffort):
            if await self._render_probe_prompt(effort) == none_prompt:
                return "low"
        return "none"

    async def _render_probe_prompt(
        self, effort: AnthropicDisabledThinkingEffort
    ) -> tuple[list[int] | None, str | None] | None:
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hi"}],
            reasoning_effort=effort,
        )
        try:
            result = await self.online_renderer.render_chat(request)
        except Exception:
            logger.debug("Rendering reasoning_effort=%r failed", effort, exc_info=True)
            return None
        if isinstance(result, ErrorResponse):
            return None
        _, (engine_input,) = result
        components = self._extract_prompt_components(engine_input)
        return components.token_ids, components.text

    @staticmethod
    def _detect_merge_inline_system(chat_template: str | None) -> bool:
        """Auto-detect whether the chat template requires system-first ordering.

        Renders a [system, user, system, user] conversation against the
        template; if it raises (e.g. Qwen's ``loop.first`` guard), the
        model needs inline system messages merged into the leading block.
        """
        if not chat_template:
            return True
        try:
            env = jinja2.sandbox.ImmutableSandboxedEnvironment(
                trim_blocks=True,
                lstrip_blocks=True,
                extensions=[jinja2.ext.loopcontrols],
            )
            env.from_string(chat_template).render(
                messages=[
                    {"role": "system", "content": "t"},
                    {"role": "user", "content": "t"},
                    {"role": "system", "content": "t"},
                    {"role": "user", "content": "t"},
                ],
                add_generation_prompt=False,
            )
            return False
        except jinja2.TemplateError:
            return True

    @staticmethod
    def _convert_image_source_to_url(source: dict[str, Any]) -> str:
        """Convert an Anthropic image source to an OpenAI-compatible URL.

        Anthropic supports two image source types:
        - base64: {"type": "base64", "media_type": "image/jpeg", "data": "..."}
        - url: {"type": "url", "url": "https://..."}

        For base64 sources, this constructs a proper data URI that
        downstream processors (e.g. vLLM's media connector) can handle.
        """
        source_type = source.get("type")
        if source_type == "url":
            return source.get("url", "")
        # Default to base64 processing if type is "base64"
        # or missing, ensuring a proper data URI is always
        # constructed for non-URL sources.
        media_type = source.get("media_type", "image/jpeg")
        data = source.get("data", "")
        return f"data:{media_type};base64,{data}"

    @classmethod
    def to_chat_completion_request(
        cls,
        anthropic_request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
        *,
        merge_inline_system: bool = False,
        disabled_thinking_effort: AnthropicDisabledThinkingEffort = "none",
    ) -> ChatCompletionRequest:
        """Convert Anthropic message format to OpenAI format"""
        anthropic_request = prepare_messages(anthropic_request, cls._resolve_tools(anthropic_request))
        openai_messages: list[dict[str, Any]] = []

        cls._convert_system_message(
            anthropic_request,
            openai_messages,
            merge_inline_system=merge_inline_system,
        )
        cls._convert_messages(
            anthropic_request.messages,
            openai_messages,
            merge_inline_system=merge_inline_system,
        )
        req = cls._build_base_request(anthropic_request, openai_messages)
        cls._handle_streaming_options(req, anthropic_request)
        cls._handle_output_config(req, anthropic_request)
        cls._handle_thinking(req, anthropic_request, disabled_thinking_effort)
        cls._convert_tool_choice(anthropic_request, req)
        cls._convert_tools(anthropic_request, req)
        return req

    @classmethod
    def _convert_system_message(
        cls,
        anthropic_request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
        openai_messages: list[dict[str, Any]],
        *,
        merge_inline_system: bool = False,
    ) -> None:
        """Convert Anthropic system message to OpenAI format"""
        system_parts: list[str] = []

        # Top-level system field
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                system_parts.append(anthropic_request.system)
            else:
                for block in anthropic_request.system:
                    if block.type == "text" and block.text:
                        # Strip Claude Code's attribution header which contains
                        # a per-request hash that defeats prefix caching.
                        if block.text.startswith("x-anthropic-billing-header"):
                            continue
                        system_parts.append(block.text)

        # When the template requires system-first ordering, extract inline
        # system messages from the messages array and merge them into the
        # top-level block so the template doesn't reject them.
        if merge_inline_system:
            for msg in anthropic_request.messages:
                if msg.role != "system":
                    continue
                text = cls._extract_system_text(msg)
                if text:
                    system_parts.append(text)

        if system_parts:
            openai_messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    @classmethod
    def _extract_system_text(cls, msg) -> str | None:
        """Extract text from a system message, stripping billing headers."""
        if isinstance(msg.content, str):
            text = msg.content
            if text.startswith("x-anthropic-billing-header"):
                return None
            return text
        parts: list[str] = []
        for block in msg.content:
            if block.type == "text" and block.text:
                if block.text.startswith("x-anthropic-billing-header"):
                    continue
                parts.append(block.text)
        return "\n\n".join(parts) if parts else None

    @classmethod
    def _convert_messages(
        cls,
        messages: list,
        openai_messages: list[dict[str, Any]],
        *,
        merge_inline_system: bool = False,
    ) -> None:
        """Convert Anthropic messages to OpenAI format"""
        for msg in messages:
            # Handle system messages in-place: extract text, strip billing
            # headers, and only emit if there is real content.  This avoids
            # going through _convert_block / _convert_message_content which
            # doesn't strip billing headers and may produce messages with
            # no "content" key.
            if msg.role == "system":
                if merge_inline_system:
                    continue  # already merged into top-level by _convert_system_message
                text = cls._extract_system_text(msg)
                if text:
                    openai_messages.append({"role": "system", "content": text})
                continue

            openai_msg: dict[str, Any] = {"role": msg.role}  # type: ignore

            if isinstance(msg.content, str):
                openai_msg["content"] = msg.content
            else:
                cls._convert_message_content(msg, openai_msg, openai_messages)

            if not (msg.role == "user" and "content" not in openai_msg):
                openai_messages.append(openai_msg)

    @classmethod
    def _convert_message_content(
        cls,
        msg,
        openai_msg: dict[str, Any],
        openai_messages: list[dict[str, Any]],
    ) -> None:
        """Convert complex message content blocks"""
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []

        for block in msg.content:
            cls._convert_block(
                block,
                msg.role,
                content_parts,
                tool_calls,
                reasoning_parts,
                openai_messages,
            )

        if reasoning_parts:
            openai_msg["reasoning"] = "".join(reasoning_parts)

        if tool_calls:
            openai_msg["tool_calls"] = tool_calls  # type: ignore

        if content_parts:
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                openai_msg["content"] = content_parts[0]["text"]
            else:
                openai_msg["content"] = content_parts  # type: ignore
        elif not tool_calls and not reasoning_parts:
            return

    @classmethod
    def _convert_block(
        cls,
        block,
        role: str,
        content_parts: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        reasoning_parts: list[str],
        openai_messages: list[dict[str, Any]],
    ) -> None:
        """Convert individual content block"""
        if block.type == "text" and block.text:
            content_parts.append({"type": "text", "text": block.text})
        elif block.type in {"document", "search_result"}:
            content_parts.append({"type": "text", "text": text_block(block)})
        elif block.type == "image" and block.source:
            image_url = cls._convert_image_source_to_url(block.source)
            content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
        elif block.type == "thinking" and block.thinking is not None:
            reasoning_parts.append(block.thinking)
        elif block.type == "redacted_thinking":
            # Redacted thinking blocks contain safety-filtered reasoning.
            # We skip them as the content is opaque (base64 'data' field),
            # but accepting the block prevents a validation error when the
            # client echoes back the full assistant message.
            pass
        elif block.type == "tool_use":
            cls._convert_tool_use_block(block, tool_calls)
        elif block.type == "tool_result":
            cls._convert_tool_result_block(block, role, openai_messages, content_parts)
        elif block.type == "tool_reference":
            # Tool references are expanded during tool_result processing
            # when they appear inside tool_result content.
            pass

    @classmethod
    def _convert_tool_use_block(cls, block, tool_calls: list[dict[str, Any]]) -> None:
        """Convert tool_use block to OpenAI function call format"""
        tool_call = {
            "id": block.id or f"call_{int(time.time())}",
            "type": "function",
            "function": {
                "name": block.name or "",
                "arguments": json.dumps(block.input or {}),
            },
        }
        tool_calls.append(tool_call)

    @classmethod
    def _convert_tool_result_block(
        cls,
        block,
        role: str,
        openai_messages: list[dict[str, Any]],
        content_parts: list[dict[str, Any]],
    ) -> None:
        """Convert tool_result block to OpenAI format"""
        if role == "user":
            cls._convert_user_tool_result(block, openai_messages)
        else:
            tool_result_text = str(block.content) if block.content else ""
            content_parts.append(
                {"type": "text", "text": f"Tool result: {tool_result_text}"}
            )

    @classmethod
    def _convert_user_tool_result(
        cls, block, openai_messages: list[dict[str, Any]]
    ) -> None:
        """Convert user tool_result with text and image support"""
        tool_text = ""
        tool_image_urls: list[str] = []
        tool_reference: list[dict[str, Any]] = []

        if isinstance(block.content, str):
            tool_text = block.content
        elif isinstance(block.content, list):
            text_parts: list[str] = []
            for item in block.content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    text_parts.append(item.get("text", ""))
                elif item_type == "image":
                    source = item.get("source", {})
                    url = cls._convert_image_source_to_url(source)
                    if url:
                        tool_image_urls.append(url)
                elif item_type == "tool_reference":
                    ref_name = item.get("tool_name") or item.get("name")
                    if ref_name:
                        tool_reference.append(
                            {"type": "tool_reference", "name": ref_name}
                        )
            tool_text = "\n".join(text_parts)

        if block.is_error:
            tool_text = "[Tool execution failed]\n" + tool_text
        openai_messages.append(
            {
                "role": "tool",
                "tool_call_id": block.tool_use_id or "",
                "content": tool_text or "",
            }
        )

        if tool_image_urls:
            openai_messages.append(
                {
                    "role": "user",
                    "content": [  # type: ignore[dict-item]
                        {"type": "image_url", "image_url": {"url": img}}
                        for img in tool_image_urls
                    ],
                }
            )

        if tool_reference:
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.tool_use_id or "",
                    "content": tool_reference,  # type: ignore[dict-item]
                }
            )

    @classmethod
    def _build_base_request(
        cls,
        anthropic_request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
        openai_messages: list[dict[str, Any]],
    ) -> ChatCompletionRequest:
        """Build base ChatCompletionRequest"""
        if isinstance(anthropic_request, AnthropicCountTokensRequest):
            return ChatCompletionRequest(
                model=anthropic_request.model,
                messages=openai_messages,
                chat_template_kwargs=anthropic_request.chat_template_kwargs,
            )

        return ChatCompletionRequest(
            model=anthropic_request.model,
            messages=openai_messages,
            max_tokens=anthropic_request.max_tokens,
            max_completion_tokens=anthropic_request.max_tokens,
            stop=anthropic_request.stop_sequences,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            top_k=anthropic_request.top_k,
            cache_salt=anthropic_request.cache_salt,
            kv_transfer_params=anthropic_request.kv_transfer_params,
            ec_transfer_params=anthropic_request.ec_transfer_params,
            vllm_xargs=anthropic_request.vllm_xargs,
            chat_template_kwargs=anthropic_request.chat_template_kwargs,
        )

    @classmethod
    def _handle_thinking(
        cls,
        req: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
        disabled_thinking_effort: AnthropicDisabledThinkingEffort = "none",
    ) -> None:
        """Handle extended-thinking configuration.

        GLM returns its actual reasoning. Unsupported display transformations
        fail explicitly rather than silently returning a different format.
        """
        thinking: AnthropicThinkingConfig | None = anthropic_request.thinking
        if thinking is None:
            return

        if getattr(thinking, 'display', None) not in (None, 'omitted'):
            raise ValueError('GLM-5.3 supports omitted or actual reasoning display; summarized/updates require unsupported model capabilities')

        if thinking.type == "disabled":
            # "none" clears enable_thinking for templates that honor it; models
            # that cannot disable thinking are configured with a low effort.
            req.reasoning_effort = disabled_thinking_effort
        elif thinking.type == "enabled":
            if req.max_completion_tokens is not None and thinking.budget_tokens >= req.max_completion_tokens:
                raise ValueError('thinking.budget_tokens must be less than the remaining task_budget')
            req.thinking_token_budget = thinking.budget_tokens
        # "adaptive" pins nothing: the model chooses depth beneath the ceiling
        # already set from output_config.effort.

    @classmethod
    def _handle_output_config(
        cls,
        req: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
    ) -> None:
        """Handle output configuration such as output format and effort"""
        output_config: AnthropicOutputConfig | None = anthropic_request.output_config
        if output_config and output_config.format and output_config.format.json_schema:
            req.response_format = ResponseFormat(
                type=output_config.format.type,
                json_schema=JsonSchemaResponseFormat(
                    schema=output_config.format.json_schema,
                    name=output_config.format.type,
                ),
            )
        if output_config and output_config.effort is not None:
            req.reasoning_effort = (glm_effort(output_config.effort)
                                    if 'glm-5.3' in anthropic_request.model.lower()
                                    else output_config.effort)
        if output_config and output_config.task_budget and not isinstance(anthropic_request, AnthropicCountTokensRequest):
            budget = output_config.task_budget
            remaining = budget.get('remaining', budget['total'])
            if remaining == 0:
                raise ValueError('output_config.task_budget is exhausted')
            req.max_tokens = req.max_completion_tokens = min(anthropic_request.max_tokens, remaining)
            req.messages.append({'role': 'system', 'content':
                f'Task output token budget: {remaining} remaining of {budget["total"]}. Complete the task within this budget.'})

    @classmethod
    def _handle_streaming_options(
        cls,
        req: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
    ) -> None:
        """Handle streaming configuration"""
        if isinstance(anthropic_request, AnthropicCountTokensRequest):
            return
        if anthropic_request.stream:
            req.stream = anthropic_request.stream
            req.stream_options = StreamOptions.model_validate(
                {"include_usage": True, "continuous_usage_stats": True}
            )

    @classmethod
    def _convert_tool_choice(
        cls,
        anthropic_request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
        req: ChatCompletionRequest,
    ) -> None:
        """Convert Anthropic tool_choice to OpenAI format"""
        if anthropic_request.tool_choice is None:
            req.tool_choice = None
            return

        req.parallel_tool_calls = (
            not anthropic_request.tool_choice.disable_parallel_tool_use
        )
        tool_choice_type = anthropic_request.tool_choice.type
        if tool_choice_type == "auto":
            req.tool_choice = "auto"
        elif tool_choice_type == "any":
            req.tool_choice = "required"
        elif tool_choice_type == "none":
            req.tool_choice = "none"
        elif tool_choice_type == "tool":
            req.tool_choice = ChatCompletionNamedToolChoiceParam.model_validate(
                {
                    "type": "function",
                    "function": {"name": anthropic_request.tool_choice.name},
                }
            )

    @classmethod
    def _resolve_tools(
        cls,
        anthropic_request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
    ) -> list[AnthropicTool] | None:
        """Apply tool_addition/tool_removal blocks in ``messages`` to ``tools``.

        Chat templates see a single tool list, so this resolves the tools in
        effect at the end of the conversation: a tool added by reference is no
        longer deferred, a tool added by value is declared, and a removed tool
        is dropped.
        """
        changes = [
            block
            for msg in anthropic_request.messages
            if not isinstance(msg.content, str)
            for block in msg.content
            if block.type in ("tool_addition", "tool_removal")
        ]
        if not changes:
            return anthropic_request.tools

        tools = {tool.name: tool for tool in anthropic_request.tools or []}
        removed: set[str] = set()
        for block in changes:
            change = block.tool
            assert change is not None
            if isinstance(change, AnthropicToolChangeDefinition):
                name = change.definition.name
                tools[name] = change.definition
            else:
                name = change.name
                if name not in tools:
                    raise ValueError(
                        f"{block.type} references tool {name!r}, "
                        "which is not declared in tools"
                    )
                if block.type == "tool_addition":
                    tools[name] = tools[name].model_copy(update={"defer_loading": None})
            if block.type == "tool_removal":
                removed.add(name)
            else:
                removed.discard(name)
        return [tool for name, tool in tools.items() if name not in removed]


    @classmethod
    def _convert_tools(
        cls,
        anthropic_request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
        req: ChatCompletionRequest,
    ) -> None:
        """Convert Anthropic tools to OpenAI format"""
        anthropic_tools = cls._resolve_tools(anthropic_request)
        if not anthropic_tools:
            # Anthropic accepts tools=[] as no tools. Normalize before the TTL
            # prefix renderer revalidates the converted OpenAI request.
            req.tools = None
            if req.tool_choice == "auto":
                req.tool_choice = None
            return

        tools = []
        for tool in anthropic_tools:
            tools.append(
                ChatCompletionToolsParam.model_validate(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                            "strict": tool.strict,
                            "defer_loading": tool.defer_loading,
                        },
                    }
                )
            )

        if req.tool_choice is None:
            req.tool_choice = "auto"
        req.tools = tools

    async def render_chat_request(self, request):
        from vllm.native_ttl_ext.ttl_api import render_with_ttl
        return await render_with_ttl(self, request, super().render_chat_request)

    async def create_messages(self, request, raw_request=None):
        from vllm.native_ttl_ext.ttl_api import serve_with_ttl
        try:
            return await serve_with_ttl(self, request, raw_request, self._create_messages_native_ttl)
        except ValueError as exc:
            from vllm.exceptions import VLLMValidationError
            raise VLLMValidationError(str(exc)) from exc

    async def _create_messages_native_ttl(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | AnthropicMessagesResponse | ErrorResponse:
        """
        Messages API similar to Anthropic's API.

        See https://docs.anthropic.com/en/api/messages
        for the API specification. This API mimics the Anthropic messages API.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Received messages request %s", request.model_dump_json())
        omitted = request.thinking is not None and getattr(request.thinking, 'display', None) == 'omitted'
        key = history_key(raw_request) if omitted or has_hidden_history(request) else None
        request = restore_hidden_thinking(request, key)
        disabled_thinking_effort: AnthropicDisabledThinkingEffort = "none"
        if request.thinking is not None and request.thinking.type == "disabled":
            disabled_thinking_effort = await self._get_disabled_thinking_effort()
        request, applied_edits, _ = await self._prepare_context(request)
        from vllm.native_ttl_ext.ttl_api import state_var
        if (ttl_state := state_var.get()) is not None:
            ttl_state.payload = request.model_dump(exclude_none=True)
            ttl_state.disabled_effort = disabled_thinking_effort
        chat_req = self.to_chat_completion_request(
            request,
            merge_inline_system=self._merge_inline_system,
            disabled_thinking_effort=disabled_thinking_effort,
        )
        if self._glm_content_stops and chat_req.stop:
            tokenizer = self.online_renderer.renderer.get_tokenizer()
            vocab = tokenizer.get_vocab()
            if '<think>' not in vocab or '</think>' not in vocab:
                raise ValueError('GLM content-only stopping requires actual reasoning token ids')
            chat_req.vllm_xargs = dict(chat_req.vllm_xargs or {},
                anthropic_content_stops=1,
                anthropic_think_start=vocab['<think>'],
                anthropic_think_end=vocab['</think>'])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Convert to OpenAI request %s", chat_req.model_dump_json())
        generator = await self.create_chat_completion(chat_req, raw_request)

        if isinstance(generator, ErrorResponse):
            return generator

        elif isinstance(generator, ChatCompletionResponse):
            result = self.messages_full_converter(generator)
            result.model = request.model
            result.context_management = applied_edits
            if omitted:
                for block in result.content:
                    if block.type == 'thinking':
                        block.signature = seal_thinking(block.thinking or '', key, request.model)
                        block.thinking = ''
            return result

        return self.message_stream_converter(generator, model=request.model, context_management=applied_edits,
                                             omitted_key=key if omitted else None)

    def messages_full_converter(
        self,
        generator: ChatCompletionResponse,
    ) -> AnthropicMessagesResponse:
        result = AnthropicMessagesResponse(
            id=generator.id,
            content=[],
            model=generator.model,
            usage=_build_anthropic_usage(generator.usage),
            kv_transfer_params=generator.kv_transfer_params,
            ec_transfer_params=generator.ec_transfer_params,
        )
        choice = generator.choices[0]
        if choice.finish_reason == "stop":
            # vLLM reports the matched stop string in stop_reason (a str);
            # an int stop-token-id or None (natural EOS) maps to end_turn.
            if isinstance(choice.stop_reason, str):
                result.stop_reason = "stop_sequence"
                result.stop_sequence = choice.stop_reason
            elif choice.message.tool_calls:
                result.stop_reason = "tool_use"
            else:
                result.stop_reason = "end_turn"
        elif choice.finish_reason == "length":
            result.stop_reason = "max_tokens"
        elif choice.finish_reason == "tool_calls":
            result.stop_reason = "tool_use"

        content: list[AnthropicContentBlock] = []
        if choice.message.reasoning:
            content.append(
                AnthropicContentBlock(
                    type="thinking",
                    thinking=choice.message.reasoning,
                    signature=uuid.uuid4().hex,
                )
            )
        if choice.message.content:
            content.append(
                AnthropicContentBlock(
                    type="text",
                    text=choice.message.content,
                )
            )

        for tool_call in choice.message.tool_calls:
            anthropic_tool_call = AnthropicContentBlock(
                type="tool_use",
                id=tool_call.id,
                name=tool_call.function.name,
                input=tool_arguments(tool_call.function.arguments),
            )
            content += [anthropic_tool_call]

        # Anthropic's canonical shape for an empty completion is a single
        # empty text block, not []. Some strict clients assume content[0]
        # exists, so emit one here.
        if not content:
            content.append(AnthropicContentBlock(type="text", text=""))

        result.content = content

        return result

    async def message_stream_converter(self, generator, *, model=None, context_management=None, omitted_key=None):
        stream = stream_messages(generator, model, _build_anthropic_usage,
                                 self.stop_reason_map, context_management, omitted_key=omitted_key)
        try:
            async for frame in stream:
                yield frame
        finally:
            await stream.aclose()

    async def _context_token_count(self, request):
        chat_req = self.to_chat_completion_request(request,
            merge_inline_system=self._merge_inline_system,
            disabled_thinking_effort='low' if self._glm_content_stops else 'none')
        # Editing counts input only, without reserving the generation allowance.
        chat_req.max_tokens = chat_req.max_completion_tokens = None
        chat_req = _ContextCountRequest.model_validate(chat_req.model_dump())
        result = await self.online_renderer.render_chat(chat_req)
        if isinstance(result, ErrorResponse):
            raise ValueError(result.error.message)
        _, (engine_input,) = result
        ids = self._extract_prompt_components(engine_input).token_ids
        if ids is None:
            raise ValueError('Tokenized input is required for context_management')
        return len(ids)

    async def _prepare_context(self, request):
        config = request.context_management
        if not config or not config.get('edits'):
            return request, None, None
        edits = config['edits']
        if all(e['type'] == 'clear_thinking_20251015' and e.get('keep', 'all') == 'all' for e in edits):
            return request, {'applied_edits': []}, None
        original = current_tokens = await self._context_token_count(request)
        applied = []
        for edit in edits:
            candidate, counts = context_candidate(request, edit, current_tokens)
            if candidate is None:
                continue
            after = await self._context_token_count(candidate)
            cleared = current_tokens - after
            minimum = (edit.get('clear_at_least') or {}).get('value', 0)
            if cleared <= 0 or cleared < minimum:
                continue
            counts['cleared_input_tokens'] = cleared
            applied.append(counts)
            request, current_tokens = candidate, after
        return request, {'applied_edits': applied}, original

    async def count_tokens(
        self,
        request: AnthropicCountTokensRequest,
        raw_request: Request | None = None,
    ) -> AnthropicCountTokensResponse | ErrorResponse:
        """Implements Anthropic's messages.count_tokens endpoint."""
        if has_hidden_history(request):
            request = restore_hidden_thinking(request, history_key(raw_request))
        request, applied_edits, original_tokens = await self._prepare_context(request)
        chat_req = self.to_chat_completion_request(
            request,
            merge_inline_system=self._merge_inline_system,
            disabled_thinking_effort='low' if self._glm_content_stops else 'none',
        )
        result = await self.render_chat_request(chat_req)
        if isinstance(result, ErrorResponse):
            return result

        _, engine_inputs = result

        input_tokens = sum(  # type: ignore
            len(engine_input["prompt_token_ids"])  # type: ignore[typeddict-item, misc]
            for engine_input in engine_inputs
            if "prompt_token_ids" in engine_input
        )

        response = AnthropicCountTokensResponse(
            input_tokens=input_tokens,
            context_management=AnthropicContextManagement(
                original_input_tokens=original_tokens if original_tokens is not None else input_tokens
            ),
        )

        return response
