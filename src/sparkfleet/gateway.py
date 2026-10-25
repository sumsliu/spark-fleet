"""Optional authenticated mTLS gateway with qualification-gated model discovery.

Atomic config reload affects new requests; existing streams drain normally.
Device revocation/expiry cancels that device's existing TLS transports.
"""

import asyncio, hashlib, hmac, json, os, ssl, stat, time
from pathlib import Path
from urllib.parse import urlsplit
from aiohttp import web, ClientSession, ClientTimeout
from .adapters import normalize_messages, responses_input

PATHS = frozenset(
    [
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/chat/completions",
        "/v1/responses",
    ]
)
HOP = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "content-length",
        "authorization",
        "x-api-key",
        "host",
    ]
)
FORWARD = frozenset(["content-type", "accept", "anthropic-version", "anthropic-beta"])


def error(status, message, kind="invalid_request_error"):
    return web.json_response(
        {"type": "error", "error": {"type": kind, "message": message}},
        status=status,
        headers={"Cache-Control": "no-store"},
    )


def private_text(path):
    p = Path(path)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        if (
            not stat.S_ISREG(s.st_mode)
            or s.st_mode & 0o077
            or s.st_uid not in (0, os.geteuid())
            or s.st_size > 65536
        ):
            raise ValueError("Private credential file required")
        with os.fdopen(fd, "r") as f:
            fd = None
            return f.read().strip()
    finally:
        if fd is not None:
            os.close(fd)


def load_config(path):
    p = Path(path)
    if p.is_symlink() or p.stat().st_size > 1024 * 1024:
        raise ValueError("Invalid gateway config file")
    c = json.loads(p.read_text())
    if type(c["max_active"]) is not int or not 1 <= c["max_active"] <= 16:
        raise ValueError("Invalid gateway concurrency")
    for model, b in c["models"].items():
        u = urlsplit(b["base_url"])
        if (
            u.scheme not in ("http", "https")
            or not u.hostname
            or u.username
            or u.password
            or u.path not in ("", "/")
            or u.query
            or u.fragment
        ):
            raise ValueError("Invalid fixed backend")
        if not set(b["endpoints"]).issubset(PATHS):
            raise ValueError("Invalid route capability")
        if type(b["enabled"]) is not bool:
            raise ValueError("Model enabled must be a boolean")
        if b["enabled"]:
            q = b["qualification"]
            raw = Path(q["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != q["sha256"]:
                raise ValueError("Qualification receipt changed")
            d = json.loads(raw)
            if (
                d.get("passed") is not True
                or d.get("complete") is not True
                or d.get("model") != model
            ):
                raise ValueError(
                    "Completed exact-model acceptance required before enabling"
                )
            if not set(b["endpoints"]).issubset(d.get("endpoints", [])):
                raise ValueError("Route capabilities exceed qualification receipt")
    for row in c["clients"].values():
        for field in ["api_key_sha256", "certificate_sha256"]:
            value = row[field]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(x not in "0123456789abcdef" for x in value)
            ):
                raise ValueError("Expected credential/certificate SHA256")
        if not set(row["models"]).issubset(c["models"]):
            raise ValueError("Unknown model grant")
    return c


class Gateway:
    def __init__(self, path):
        self.path = Path(path)
        self.config = load_config(path)
        self.active = 0
        self.transports = {}
        self.session = None

    def reload(self):
        # Invalid policy fails closed; never retain a revoked device by accident.
        try:
            c = load_config(self.path)
        except (OSError, ValueError, KeyError):
            return False
        if c.get("listen") != self.config.get("listen"):
            return False
        self.config = c
        return True

    def device_valid(self, identity, pin, key_hash, expires):
        r = self.config["clients"].get(identity, {})
        return (
            r.get("enabled") is True
            and time.time() < expires
            and hmac.compare_digest(r.get("certificate_sha256", ""), pin)
            and hmac.compare_digest(r.get("api_key_sha256", ""), key_hash)
        )

    def authenticate(self, request):
        auth = request.headers.getall("Authorization", [])
        xkey = request.headers.getall("x-api-key", [])
        if len(auth) > 1 or len(xkey) > 1 or (auth and xkey):
            return None
        token = (
            auth[0][7:]
            if auth and auth[0].startswith("Bearer ")
            else xkey[0]
            if xkey
            else ""
        )
        digest = hashlib.sha256(token.encode()).hexdigest()
        transport = request.transport
        conn = transport.get_extra_info("ssl_object") if transport else None
        if not conn:
            return None
        cert = conn.getpeercert(binary_form=True)
        details = conn.getpeercert()
        if not cert:
            return None
        pin = hashlib.sha256(cert).hexdigest()
        try:
            expires = ssl.cert_time_to_seconds(details["notAfter"])
        except (KeyError, ValueError):
            return None
        for identity, row in self.config["clients"].items():
            if self.device_valid(identity, pin, digest, expires):
                if len(self.transports) >= 1024 and transport not in self.transports:
                    return None
                self.transports[transport] = (identity, pin, digest, expires)
                return row
        return None

    async def watch(self):
        while True:
            await asyncio.sleep(1)
            valid = self.reload()
            for t, info in list(self.transports.items()):
                if t.is_closing():
                    self.transports.pop(t, None)
                elif not valid or not self.device_valid(*info):
                    t.abort()
                    self.transports.pop(t, None)

    async def handle(self, request):
        if (
            request.path == "/health"
            and request.method == "GET"
            and request.remote in ("127.0.0.1", "::1")
        ):
            return web.json_response({"gateway": "ok"})
        if not self.reload():
            return error(503, "Gateway policy validation failed", "api_error")
        client = self.authenticate(request)
        if not client:
            return error(
                401,
                "Valid device certificate and API key required",
                "authentication_error",
            )
        if (
            request.path == "/v1/models"
            and request.method == "GET"
            and not request.query_string
        ):
            rows = [
                {"id": m, "object": "model"}
                for m, b in self.config["models"].items()
                if b["enabled"] and m in client["models"]
            ]
            return web.json_response(
                {"object": "list", "data": rows, "has_more": False},
                headers={"Cache-Control": "no-store"},
            )
        beta = (
            request.path in ("/v1/messages", "/v1/messages/count_tokens")
            and request.query_string == "beta=true"
        )
        if (
            request.method != "POST"
            or request.path not in PATHS
            or (request.query_string and not beta)
        ):
            return error(404, "Unsupported route", "not_found_error")
        try:
            raw = await asyncio.wait_for(request.read(), 120)
            p = json.loads(raw)
            if not isinstance(p, dict) or not isinstance(p.get("model"), str):
                raise ValueError()
        except (ValueError, UnicodeError):
            return error(400, "An explicit JSON model is required")
        except asyncio.TimeoutError:
            return error(408, "Request body timeout", "api_error")
        model = p["model"]
        b = self.config["models"].get(model)
        if not b or model not in client["models"]:
            return error(404, "Model unavailable to this device", "not_found_error")
        if not b["enabled"]:
            return error(503, "Model is disabled pending acceptance", "api_error")
        if request.path not in b["endpoints"]:
            return error(400, "Unsupported model API capability")
        if self.active >= self.config["max_active"]:
            value = error(429, "Gateway concurrency limit reached", "rate_limit_error")
            value.headers["Retry-After"] = "5"
            return value
        if b.get("compatibility") == "deepseek-v41":
            if request.path.startswith("/v1/messages"):
                try:
                    p = normalize_messages(p)
                except (ValueError, TypeError, KeyError):
                    return error(400, "Invalid model thinking/tool envelope")
            elif request.path == "/v1/responses":
                p = responses_input(p)
            raw = json.dumps(p, ensure_ascii=False).encode()
        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD}
        headers["Accept-Encoding"] = "identity"
        if b.get("upstream_key_file"):
            headers["Authorization"] = "Bearer " + private_text(b["upstream_key_file"])
        self.active += 1
        response = None
        sent = False
        try:
            async with self.session.post(
                b["base_url"].rstrip("/")
                + request.path
                + ("?beta=true" if beta else ""),
                data=raw,
                headers=headers,
                allow_redirects=False,
            ) as r:
                if 300 <= r.status < 400:
                    return error(502, "Backend redirect refused", "api_error")
                blocked = (
                    HOP
                    | {
                        x.strip().lower()
                        for x in r.headers.get("Connection", "").split(",")
                    }
                    | {"set-cookie", "server", "location"}
                )
                response = web.StreamResponse(
                    status=r.status,
                    headers={
                        k: v for k, v in r.headers.items() if k.lower() not in blocked
                    },
                )
                response.headers["Cache-Control"] = "no-store"
                await response.prepare(request)
                sent = True
                async for chunk in r.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        except asyncio.CancelledError:
            raise
        except Exception:
            if sent:
                if request.transport:
                    request.transport.abort()
                return response
            return error(502, "Backend unavailable or timed out", "api_error")
        finally:
            self.active -= 1

    def app(self):
        app = web.Application(client_max_size=32 * 1024 * 1024)
        app.router.add_route("*", "/{path:.*}", self.handle)

        async def start(_):
            self.session = ClientSession(
                trust_env=False,
                auto_decompress=False,
                timeout=ClientTimeout(total=3600, connect=15, sock_read=900),
            )
            self.watcher = asyncio.create_task(self.watch())

        async def stop(_):
            self.watcher.cancel()
            await asyncio.gather(self.watcher, return_exceptions=True)
            await self.session.close()

        app.on_startup.append(start)
        app.on_cleanup.append(stop)
        return app


def serve(path):
    gateway = Gateway(path)
    c = gateway.config["listen"]
    private_text(c["key_file"])
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(c["certificate_file"], c["key_file"])
    context.load_verify_locations(c["client_ca_file"])
    context.verify_mode = ssl.CERT_REQUIRED
    web.run_app(
        gateway.app(),
        host=c["host"],
        port=c["port"],
        ssl_context=context,
        access_log=None,
        handler_cancellation=True,
    )
