"""Envelope-only compatibility helpers. Opaque tool inputs are never rewritten."""

import copy

DS = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
GLM = "glm-5.3"
DS_EFFORT = {"low": 25, "medium": 38, "high": 50, "xhigh": 75, "max": 100}
GLM_EFFORT = {
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


def normalize_messages(body):
    p = copy.deepcopy(body)
    if p.get("model") not in (DS, GLM):
        raise ValueError("Explicit supported model ID required")
    cfg = p.get("output_config") or {}
    existing = p.get("chat_template_kwargs", {})
    if not isinstance(existing, dict):
        raise ValueError("Invalid chat template envelope")
    default_thinking = (
        "enabled"
        if existing.get("thinking") is True or existing.get("enable_thinking") is True
        else "disabled"
    )
    thinking = p.get("thinking") or {"type": default_thinking}
    if not isinstance(cfg, dict) or not isinstance(thinking, dict):
        raise ValueError("Invalid thinking envelope")
    effort = cfg.get("effort", "low")
    if not isinstance(effort, str) or effort not in DS_EFFORT:
        raise ValueError("Unknown reasoning effort")
    if thinking.get("type") not in ("enabled", "adaptive", "disabled"):
        raise ValueError("Invalid thinking type")
    if p["model"] == GLM:
        p.setdefault("output_config", {})["effort"] = GLM_EFFORT[effort]
        return p
    p.pop("thinking", None)
    kwargs = p.setdefault("chat_template_kwargs", {})
    enabled = thinking["type"] != "disabled"
    mapped = DS_EFFORT[effort] if enabled else "none"
    if "thinking" not in body and "output_config" not in body and enabled:
        mapped = existing.get("reasoning_effort", mapped)
    kwargs.update(thinking=enabled, enable_thinking=enabled, reasoning_effort=mapped)
    if "output_config" in p:
        p["output_config"].pop("effort", None)
        if not p["output_config"]:
            p.pop("output_config")
    referenced = set()

    def blocks(content):
        if not isinstance(content, list):
            return
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_reference":
                name = item.get("tool_name", item.get("name"))
                if not isinstance(name, str) or not name:
                    raise ValueError("Invalid tool reference")
                referenced.add(name)
                value = {
                    "type": "text",
                    "text": "Tool discovered: "
                    + name
                    + ". Use its declared schema to call it.",
                }
                if "cache_control" in item:
                    value["cache_control"] = copy.deepcopy(item["cache_control"])
                content[i] = value
            elif item.get("type") == "tool_result":
                blocks(item.get("content"))

    for message in p.get("messages", []):
        if isinstance(message, dict):
            blocks(message.get("content"))
    choice = p.get("tool_choice", {})
    if isinstance(choice, dict) and choice.get("type") == "tool":
        referenced.add(choice.get("name"))
    if isinstance(p.get("tools"), list):
        selected = []
        for tool in p["tools"]:
            if not isinstance(tool, dict):
                raise ValueError("Invalid tool declaration")
            if tool.get("defer_loading") and tool.get("name") not in referenced:
                continue
            tool.pop("defer_loading", None)
            selected.append(tool)
        p["tools"] = selected
    return p


def responses_input(body):
    """Normalize only known Responses message/input_text parts for V4.1."""
    p = copy.deepcopy(body)
    if p.get("model") != DS:
        return p
    for item in p.get("input", []) if isinstance(p.get("input"), list) else []:
        if not isinstance(item, dict):
            continue
        if "role" in item and "type" not in item:
            item["type"] = "message"
        if item.get("type") == "message" and item.get("role") in (
            "user",
            "system",
            "developer",
        ):
            if item["role"] == "developer":
                item["role"] = "system"
            field = "content"
        elif item.get("type") == "function_call_output":
            field = "output"
        else:
            continue
        parts = item.get(field)
        if not isinstance(parts, list) or not parts:
            continue
        if all(
            isinstance(x, dict)
            and x.get("type") == "input_text"
            and isinstance(x.get("text"), str)
            for x in parts
        ):
            item[field] = "\n\n".join(x["text"] for x in parts)
        else:
            for x in parts:
                if isinstance(x, dict) and x.get("type") == "input_image":
                    x.setdefault("detail", "auto")
    return p
