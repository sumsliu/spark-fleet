# SparkFleet reference overlay: modified deployment/latency behavior; see overlays/README.md and NOTICE.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pydantic models for Anthropic API protocol"""

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

import vllm.envs as envs


class AnthropicError(BaseModel):
    """Error structure for Anthropic API"""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API"""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    """Token usage information"""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation: dict[str, int] | None = None


class AnthropicContentBlock(BaseModel):
    """Content block in message"""

    type: Literal[
        "text",
        "image",
        "tool_use",
        "tool_result",
        "tool_reference",
        "thinking",
        "redacted_thinking",
        "tool_addition",
        "tool_removal",
        "document",
        "search_result",
    ]
    text: str | None = None
    # For image content
    source: dict[str, Any] | str | None = None
    title: str | None = None
    context: str | None = None
    citations: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None
    # For tool use/result
    id: str | None = None
    tool_use_id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None
    # For tool_reference content
    tool_name: str | None = None
    # For thinking content
    thinking: str | None = None
    signature: str | None = None
    # For redacted thinking content (safety-filtered by the API)
    data: str | None = None
    # For tool_addition/tool_removal content
    tool: "AnthropicToolChange | None" = None

    @model_validator(mode="after")
    def validate_tool_change(self) -> "AnthropicContentBlock":
        if self.type == "tool_use":
            if not self.id or not self.name or self.input is None:
                raise ValueError("tool_use requires id, name and input")
        elif self.type == "tool_result" and not self.tool_use_id:
            raise ValueError("tool_result requires tool_use_id")
        elif self.type == "image" and not isinstance(self.source, dict):
            raise ValueError("image.source must be an object")
        elif self.type == "document" and not isinstance(self.source, dict):
            raise ValueError("document.source must be an object")
        elif self.type == "search_result" and not isinstance(self.source, str):
            raise ValueError("search_result.source must be a string")
        if self.citations and self.citations.get("enabled"):
            raise ValueError("GLM-5.3 does not support generated Anthropic citation blocks")
        _validate_cache_control(self.cache_control)
        if self.type not in ("tool_addition", "tool_removal"):
            return self
        if self.tool is None:
            raise ValueError(f"tool is required for {self.type} blocks")
        if self.type == "tool_removal" and self.tool.type != "tool_reference":
            raise ValueError("tool_removal only accepts a tool_reference")
        return self


def _validate_cache_control(value) -> None:
    if value is None:
        return
    if value.get("type") != "ephemeral":
        raise ValueError("cache_control.type must be ephemeral")
    from vllm.native_ttl_ext.ttl_protocol import ttl_seconds
    from vllm.native_ttl_ext.ttl_runtime import config
    ttl_seconds(value)
    if "ttl" in value and config() is None:
        raise ValueError("Explicit TTL requires enabled native TTL candidate")


def _validate_context_management(value):
    if value is None:
        return value
    if not isinstance(value.get('edits', []), list):
        raise ValueError('context_management.edits must be an array')
    for edit in value.get("edits", []):
        if not isinstance(edit, dict):
            raise ValueError('context_management.edits entries must be objects')
        kind = edit.get("type")
        if kind not in {"clear_tool_uses_20250919", "clear_thinking_20251015"}:
            raise ValueError(f"context_management strategy {kind!r} is unsupported")
        if kind == "clear_thinking_20251015":
            keep = edit.get("keep", "all")
            if keep != "all":
                if not isinstance(keep, dict) or keep.get("type") != "thinking_turns" or not isinstance(keep.get("value"), int) or isinstance(keep["value"], bool) or keep["value"] < 1:
                    raise ValueError("context_management.keep requires all or positive thinking_turns")
        else:
            for key, choices, default in [("trigger", {"input_tokens", "tool_uses"}, 100000),
                                          ("keep", {"tool_uses"}, 3),
                                          ("clear_at_least", {"input_tokens"}, None)]:
                option = edit.get(key)
                if option is not None and (not isinstance(option, dict) or option.get("type") not in choices or not isinstance(option.get("value"), int) or isinstance(option["value"], bool) or option["value"] < 0):
                    raise ValueError(f"context_management.{key} has an invalid type or value")
            excluded = edit.get('exclude_tools', [])
            if not isinstance(excluded, list) or any(not isinstance(x, str) for x in excluded):
                raise ValueError('context_management.exclude_tools must be an array of tool names')
            inputs = edit.get('clear_tool_inputs', False)
            if type(inputs) is not bool and (not isinstance(inputs, list) or any(not isinstance(x, str) for x in inputs)):
                raise ValueError('context_management.clear_tool_inputs must be boolean or an array of tool names')
    return value


def _reject_tool_changes(content: str | list[AnthropicContentBlock] | None) -> None:
    if content is None or isinstance(content, str):
        return
    for block in content:
        if block.type in ("tool_addition", "tool_removal"):
            raise ValueError(
                f'{block.type} blocks are only allowed in messages with role "system"'
            )


class AnthropicMessage(BaseModel):
    """Message structure"""

    role: Literal["user", "assistant", "system"]
    content: str | list[AnthropicContentBlock]

    @model_validator(mode="after")
    def validate_tool_change_role(self) -> "AnthropicMessage":
        if self.role != "system":
            _reject_tool_changes(self.content)
        return self


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]
    strict: bool | None = None
    defer_loading: bool | None = None
    cache_control: dict[str, Any] | None = None

    @field_validator("cache_control")
    @classmethod
    def validate_cache_control(cls, value):
        _validate_cache_control(value)
        return value

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"  # Default to object type
        return v


class AnthropicToolChangeReference(BaseModel):
    """A tool named by a tool_addition or tool_removal block."""

    type: Literal["tool_reference"]
    name: str


class AnthropicToolChangeDefinition(BaseModel):
    """A tool defined by value in a tool_addition block."""

    type: Literal["tool_definition"]
    definition: AnthropicTool


AnthropicToolChange = Annotated[
    AnthropicToolChangeReference | AnthropicToolChangeDefinition,
    Field(discriminator="type"),
]


class AnthropicToolChoice(BaseModel):
    """Tool Choice definition"""

    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None
    disable_parallel_tool_use: bool | None = None

    @model_validator(mode="after")
    def validate_name_required_for_tool(self) -> "AnthropicToolChoice":
        if self.type == "tool" and not self.name:
            raise ValueError("tool_choice.name is required when type is 'tool'")
        return self


class AnthropicJsonOutputFormat(BaseModel):
    """JSON output format configuration"""

    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    type: Literal["json_schema"] = "json_schema"


AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
# reasoning_effort sent to the chat template for thinking.type=disabled.
AnthropicDisabledThinkingEffort = Literal["none", AnthropicEffort]
AnthropicDisabledThinkingEffortOption = Literal["auto", AnthropicDisabledThinkingEffort]
AnthropicThinkingDisplay = Literal["summarized", "omitted", "updates"]


class AnthropicOutputConfig(BaseModel):
    """Configuration options for the model's output, such as the output format."""

    effort: AnthropicEffort | None = None
    format: AnthropicJsonOutputFormat | None = None
    task_budget: dict[str, Any] | None = None

    @field_validator("task_budget")
    @classmethod
    def validate_task_budget(cls, value):
        if value is None:
            return value
        if value.get("type") != "tokens" or not isinstance(value.get("total"), int) or isinstance(value["total"], bool) or value["total"] <= 0:
            raise ValueError("output_config.task_budget requires positive total tokens")
        remaining = value.get("remaining", value["total"])
        if not isinstance(remaining, int) or isinstance(remaining, bool) or not 0 <= remaining <= value["total"]:
            raise ValueError("output_config.task_budget.remaining must be between zero and total")
        return value


class AnthropicThinkingConfigEnabled(BaseModel):
    """Extended thinking with a fixed token budget.

    Omitted display preserves encrypted history; unsupported summaries and
    model progress updates are rejected by the serving handler.
    """

    type: Literal["enabled"]
    budget_tokens: int = Field(ge=1024)
    display: AnthropicThinkingDisplay | None = None


class AnthropicThinkingConfigDisabled(BaseModel):
    type: Literal["disabled"]


class AnthropicThinkingConfigAdaptive(BaseModel):
    """Extended thinking whose depth the model chooses.

    Omitted display preserves encrypted history; unsupported summaries and
    model progress updates are rejected by the serving handler.
    """

    type: Literal["adaptive"]
    display: AnthropicThinkingDisplay | None = None


AnthropicThinkingConfig = Annotated[
    AnthropicThinkingConfigEnabled
    | AnthropicThinkingConfigDisabled
    | AnthropicThinkingConfigAdaptive,
    Field(discriminator="type"),
]


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request"""
    cache_control: dict[str, Any] | None = None
    @model_validator(mode="before")
    @classmethod
    def normalize_local_thinking_budget(cls, value):
        from copy import deepcopy
        from vllm.native_ttl_ext.thinking_budget import normalize_thinking_budget
        value = deepcopy(value)
        normalize_thinking_budget(value)
        return value
    @field_validator("cache_control")
    @classmethod
    def validate_top_cache_control(cls, value):
        _validate_cache_control(value)
        return value

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: dict[str, Any] | None = None
    output_config: AnthropicOutputConfig | None = None
    thinking: AnthropicThinkingConfig | None = None
    context_management: dict[str, Any] | None = None

    @field_validator("context_management")
    @classmethod
    def validate_context_management(cls, value):
        return _validate_context_management(value)
    stop_sequences: (
        Annotated[list[str], Field(max_length=envs.VLLM_MAX_STOP_STRINGS)] | None
    ) = None
    stream: bool | None = False
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None
    top_k: int | None = None
    top_p: float | None = None

    # vLLM-specific fields that are not in Anthropic spec
    cache_salt: str | None = Field(
        default=None,
        min_length=1,
        max_length=1024,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit)."
        ),
    )
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )
    ec_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "ECTransfer parameters used for encoder-cache disaggregated serving."
        ),
    )
    vllm_xargs: dict[str, str | int | float | list[str | int | float]] | None = Field(
        default=None,
        description=(
            "Additional request parameters with (list of) string or "
            "numeric values, used by custom extensions."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the chat template renderer. "
            "Will be accessible by the template."
        ),
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    @field_validator("system")
    @classmethod
    def validate_system(cls, v):
        _reject_tool_changes(v)
        return v

    @model_validator(mode="after")
    def validate_thinking_budget(self) -> "AnthropicMessagesRequest":
        if (
            isinstance(self.thinking, AnthropicThinkingConfigEnabled)
            and self.thinking.budget_tokens >= self.max_tokens
        ):
            raise ValueError("thinking.budget_tokens must be less than max_tokens")
        return self


class AnthropicDelta(BaseModel):
    """Delta for streaming responses"""

    type: (
        Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"]
        | None
    ) = None
    text: str | None = None
    thinking: str | None = None
    partial_json: str | None = None
    signature: str | None = None

    # Message delta
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None


class AnthropicStreamEvent(BaseModel):
    """Streaming event"""

    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: "AnthropicMessagesResponse | None" = None
    delta: AnthropicDelta | None = None
    content_block: AnthropicContentBlock | None = None
    index: int | None = None
    error: AnthropicError | None = None
    usage: AnthropicUsage | None = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response"""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: AnthropicUsage | None = None
    context_management: dict[str, Any] | None = None

    # vLLM-specific fields that are not in Anthropic spec
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None, description="KVTransfer parameters."
    )
    ec_transfer_params: dict[str, Any] | None = Field(
        default=None, description="ECTransfer parameters."
    )

    def model_post_init(self, __context):
        if not self.id:
            self.id = f"msg_{int(time.time() * 1000)}"


class AnthropicContextManagement(BaseModel):
    """Context management information for token counting."""

    original_input_tokens: int


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic messages.count_tokens request"""
    @model_validator(mode="before")
    @classmethod
    def normalize_local_thinking_budget(cls, value):
        from copy import deepcopy
        from vllm.native_ttl_ext.thinking_budget import normalize_thinking_budget
        value = deepcopy(value)
        normalize_thinking_budget(value)
        return value

    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicContentBlock] | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None
    thinking: AnthropicThinkingConfig | None = None
    output_config: AnthropicOutputConfig | None = None
    context_management: dict[str, Any] | None = None

    @field_validator("context_management")
    @classmethod
    def validate_context_management(cls, value):
        return _validate_context_management(value)

    # vLLM-specific fields that are not in Anthropic spec
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the chat template renderer. "
            "Will be accessible by the template."
        ),
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("system")
    @classmethod
    def validate_system(cls, v):
        _reject_tool_changes(v)
        return v


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic messages.count_tokens response"""

    input_tokens: int
    context_management: AnthropicContextManagement | None = None
