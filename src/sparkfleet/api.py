"""Standard-library API qualification. Keys stay in environment, reports omit bodies."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from .common import save_json
from .adapters import normalize_messages


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Client:
    def __init__(self, base, token, timeout=900):
        parsed = urllib.parse.urlsplit(base)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Use an HTTP(S) API base without embedded credentials/query/fragment"
            )
        self.base = base.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.token = token
        self.timeout = timeout
        context = ssl.create_default_context(
            cafile=os.environ.get("SPARKFLEET_CA_FILE")
        )
        cert = os.environ.get("SPARKFLEET_CERT_FILE")
        key = os.environ.get("SPARKFLEET_KEY_FILE")
        if bool(cert) != bool(key):
            raise ValueError("Set both SPARKFLEET_CERT_FILE and SPARKFLEET_KEY_FILE")
        if cert:
            context.load_cert_chain(cert, key)
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            NoRedirect(),
            urllib.request.HTTPSHandler(context=context),
        )

    def request(self, path, payload=None, stream=False):
        req = urllib.request.Request(
            self.base + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "Accept-Encoding": "identity",
            },
        )
        started = time.monotonic()
        try:
            r = self.opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            data = e.read(1024 * 1024)
            # Retain the error type only; a server may echo secrets or private prompts.
            try:
                kind = json.loads(data).get("error", {}).get("type", "http_error")
            except (ValueError, AttributeError):
                kind = "http_error"
            return e.code, {"error_type": kind}, time.monotonic() - started
        if stream:
            return r, started
        with r:
            data = r.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError("Response exceeds qualification bound")
            return r.status, json.loads(data), time.monotonic() - started


def chat(model, text, **extra):
    return dict(
        model=model,
        max_tokens=128,
        temperature=0,
        chat_template_kwargs={"thinking": False, "enable_thinking": False},
        messages=[{"role": "user", "content": text}],
        **extra,
    )


def qualify(client, model, out, *, anthropic=True, long_tokens=0):
    rows = []

    def check(name, fn):
        start = time.monotonic()
        try:
            detail = fn()
            row = dict(
                name=name,
                passed=True,
                seconds=round(time.monotonic() - start, 4),
                detail=detail,
            )
        except Exception as e:
            row = dict(
                name=name,
                passed=False,
                seconds=round(time.monotonic() - start, 4),
                error_type=type(e).__name__,
            )
        rows.append(row)
        save_json(
            out,
            dict(
                passed=all(x["passed"] for x in rows),
                complete=False,
                model=model,
                tests=rows,
            ),
        )
        print(name + ": " + ("PASS" if row["passed"] else "FAIL"), flush=True)

    def ask(path, p):
        status, d, _ = client.request(path, p)
        if status != 200:
            raise ValueError("HTTP " + str(status))
        return d

    def catalog():
        status, d, _ = client.request("/v1/models")
        if status != 200 or model not in [x["id"] for x in d["data"]]:
            raise ValueError("Model catalog mismatch")
        return {"model_id_exact": True}

    def basic():
        d = ask(
            "/v1/chat/completions", chat(model, "Reply only with the integer: 17 + 25.")
        )
        if d["model"] != model or d["choices"][0]["message"]["content"].strip() != "42":
            raise ValueError("Incorrect arithmetic/model")
        return {"correct": True, "usage": d.get("usage")}

    def json_mode():
        d = ask(
            "/v1/chat/completions",
            chat(
                model,
                'Return only JSON: {"ok":true,"sum":42}',
                response_format={"type": "json_object"},
            ),
        )
        if json.loads(d["choices"][0]["message"]["content"]) != {"ok": True, "sum": 42}:
            raise ValueError("JSON mismatch")

    def tools():
        tool = {
            "type": "function",
            "function": {
                "name": "add_fixture",
                "description": "Return a+b.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        }
        p = chat(
            model,
            "Call add_fixture for a=17 and b=25, then return its result.",
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "add_fixture"}},
        )
        d = ask("/v1/chat/completions", p)
        a = d["choices"][0]["message"]
        b = a["tool_calls"]
        if (
            len(b) != 1
            or b[0]["function"]["name"] != "add_fixture"
            or json.loads(b[0]["function"]["arguments"]) != {"a": 17, "b": 25}
        ):
            raise ValueError("Tool arguments/correlation mismatch")
        p["messages"] += [
            a,
            {"role": "tool", "tool_call_id": b[0]["id"], "content": "42"},
        ]
        p.pop("tool_choice")
        d = ask("/v1/chat/completions", p)
        if "42" not in d["choices"][0]["message"]["content"]:
            raise ValueError("Tool round trip failed")
        return {"tool_and_result_correlated": True}

    def stream(cancel=False):
        p = chat(
            model,
            "Count from 1 to 1000, one number per line."
            if cancel
            else "Reply exactly STREAM_READY.",
            stream=True,
        )
        p["max_tokens"] = 1024 if cancel else 128
        p["stream_options"] = {"include_usage": True}
        value = client.request("/v1/chat/completions", p, stream=True)
        if len(value) != 2:
            raise ValueError("Streaming HTTP error")
        r, start = value
        first = None
        text = ""
        done = False
        usage = None
        with r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    done = True
                    break
                d = json.loads(raw)
                usage = d.get("usage") or usage
                for choice in d.get("choices", []):
                    part = choice.get("delta", {}).get("content") or ""
                    if part and first is None:
                        first = time.monotonic() - start
                    text += part
                if cancel and first is not None:
                    break
        if first is None or (not cancel and (not done or "STREAM_READY" not in text)):
            raise ValueError("Incomplete text stream")
        if cancel:
            basic()
        return {
            "first_text_seconds": first,
            "done": done,
            "cancelled_after_real_text": cancel,
            "usage": usage,
        }

    def concurrent():
        def one(i):
            d = ask(
                "/v1/chat/completions", chat(model, "Reply exactly PARALLEL_" + str(i))
            )
            if "PARALLEL_" + str(i) not in d["choices"][0]["message"]["content"]:
                raise ValueError("Cross-request response")

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, [1, 2]))

    def overflow():
        p = chat(model, "Hello")
        p["max_tokens"] = 262145
        if client.request("/v1/chat/completions", p)[0] != 400:
            raise ValueError("Context overflow not rejected")

    def messages():
        p = normalize_messages(
            dict(
                model=model,
                max_tokens=256,
                thinking={"type": "disabled"},
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": "Reply exactly MESSAGE_READY."}],
            )
        )
        d = ask("/v1/messages", p)
        if d["model"] != model or "MESSAGE_READY" not in "".join(
            x.get("text", "") for x in d["content"]
        ):
            raise ValueError("Messages mismatch")
        c = ask("/v1/messages/count_tokens", dict(model=model, messages=p["messages"]))
        if c["input_tokens"] <= 0:
            raise ValueError("Token count missing")

    def thinking():
        p = chat(
            model, "Find the LCM of 6, 10, 15; think briefly then give the answer."
        )
        p["max_tokens"] = 1024
        p["reasoning_effort"] = "low"
        p["chat_template_kwargs"] = {"thinking": True, "enable_thinking": True}
        if model.startswith("deepseek-ai/"):
            p.pop("reasoning_effort", None)
            p["chat_template_kwargs"]["reasoning_effort"] = 25
        d = ask("/v1/chat/completions", p)
        a = d["choices"][0]["message"]
        reason = a.get("reasoning") or a.get("reasoning_content")
        if not reason or "30" not in (a.get("content") or ""):
            raise ValueError("Independent reasoning/final content missing")
        return {"independent_reasoning": True, "reasoning_characters": len(reason)}

    def modality():
        # A known synthetic PNG: text GLM must reject, DS must produce a real response.
        png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
        p = chat(model, "Describe image")
        p["messages"][0]["content"] = [
            {"type": "text", "text": "Describe this image briefly."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}},
        ]
        status, d, _ = client.request("/v1/chat/completions", p)
        if model == "glm-5.3":
            if status != 400:
                raise ValueError("Text-only model did not reject image")
            return {"expected_text_only_rejection": True}
        if status != 200 or not d["choices"][0]["message"].get("content"):
            raise ValueError("Image request failed")
        return {
            "synthetic_image_request_completed": True,
            "semantic_vision_quality_tested": False,
        }

    for name, fn in [
        ("catalog", catalog),
        ("arithmetic", basic),
        ("json", json_mode),
        ("tools", tools),
        ("stream", stream),
        ("thinking", thinking),
        ("modality", modality),
        ("concurrency", concurrent),
        ("overflow", overflow),
        ("cancel_recovery", lambda: stream(True)),
    ]:
        check(name, fn)
    if anthropic:
        check("anthropic_messages_count", messages)
    if long_tokens:
        if not 1024 <= long_tokens <= 230000:
            raise ValueError("Long target must leave output headroom")
        if not anthropic:
            raise ValueError(
                "Long sizing uses Messages count_tokens; enable that endpoint"
            )

        def long_case():
            def content(n):
                return (
                    "REFERENCE_KEY=SPARKFLEET_LONG_72519\n"
                    + "".join(
                        f"record {i:06}: neutral synthetic filler.\n" for i in range(n)
                    )
                    + "\nReturn just REFERENCE_KEY."
                )

            n = 1000
            c = ask(
                "/v1/messages/count_tokens",
                {"model": model, "messages": [{"role": "user", "content": content(n)}]},
            )
            n = max(1, int(n * long_tokens / c["input_tokens"]))
            msgs = [{"role": "user", "content": content(n)}]
            turns = []
            counted = ask(
                "/v1/messages/count_tokens", {"model": model, "messages": msgs}
            )["input_tokens"]
            for _ in range(3):
                p = chat(model, "")
                p["messages"] = msgs
                p["max_tokens"] = 64
                start = time.monotonic()
                d = ask("/v1/chat/completions", p)
                answer = d["choices"][0]["message"]["content"]
                if "SPARKFLEET_LONG_72519" not in answer:
                    raise ValueError("Long marker lost")
                turns.append(
                    dict(seconds=time.monotonic() - start, usage=d.get("usage"))
                )
                msgs += [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": "Repeat just REFERENCE_KEY."},
                ]
            return dict(
                actual_input_tokens=counted,
                fixture="synthetic, not a private conversation",
                turns=turns,
            )

        check("long_context_3_turns", long_case)
    result = dict(
        schema=1,
        at=time.time(),
        model=model,
        passed=all(x["passed"] for x in rows),
        complete=True,
        tests=rows,
        scope="bounded API acceptance; not GUI, sustained-load or full-window certification",
        effort="low",
        long_target=long_tokens,
        endpoints=["/v1/chat/completions"]
        + (["/v1/messages", "/v1/messages/count_tokens"] if anthropic else []),
    )
    save_json(out, result)
    return result
