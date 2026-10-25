import asyncio
import hashlib
import os
import json
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
import unittest
from unittest.mock import patch

try:
    from aiohttp import web, ClientSession, TCPConnector, ClientError
    from sparkfleet.gateway import Gateway, load_config
except ImportError:
    Gateway = None


@unittest.skipUnless(
    Gateway and shutil.which("openssl"), "optional aiohttp and openssl required"
)
class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.token = "fixture-only-credential"
        self.cancelled = asyncio.Event()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.forwarded = []

        def openssl(*args):
            subprocess.run(
                ["openssl", *args],
                cwd=self.root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        (self.root / "ca.cnf").write_text(
            "[req]\ndistinguished_name=dn\n[dn]\n[v3_ca]\nbasicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid:always\n"
        )
        openssl(
            "req",
            "-config",
            "ca.cnf",
            "-extensions",
            "v3_ca",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=SparkFleet Test CA",
            "-keyout",
            "ca.key",
            "-out",
            "ca.pem",
        )
        for name in ["server", "client"]:
            openssl(
                "req",
                "-new",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-subj",
                "/CN=" + ("localhost" if name == "server" else name),
                "-keyout",
                name + ".key",
                "-out",
                name + ".csr",
            )
            (self.root / "ext.cnf").write_text(
                "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\nsubjectAltName=DNS:localhost\nextendedKeyUsage="
                + ("serverAuth" if name == "server" else "clientAuth")
                + "\n"
            )
            openssl(
                "x509",
                "-req",
                "-in",
                name + ".csr",
                "-CA",
                "ca.pem",
                "-CAkey",
                "ca.key",
                "-CAcreateserial",
                "-days",
                "1",
                "-extfile",
                "ext.cnf",
                "-out",
                name + ".pem",
            )

        async def backend(request):
            p = await request.json()
            self.forwarded.append(p)
            if p.get("hold"):
                self.entered.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
            if p.get("stream"):
                r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await r.prepare(request)
                await r.write(
                    b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'
                )
                return r
            return web.json_response({"model": p["model"], "ok": True})

        app = web.Application()
        app.router.add_post("/v1/messages", backend)
        app.router.add_post("/v1/chat/completions", backend)
        self.backend_runner = web.AppRunner(
            app, handler_cancellation=True, access_log=None
        )
        await self.backend_runner.setup()
        site = web.TCPSite(self.backend_runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        q = self.root / "quality.json"
        q.write_text(
            json.dumps(
                {
                    "model": "glm-5.3",
                    "complete": True,
                    "passed": True,
                    "endpoints": ["/v1/messages", "/v1/chat/completions"],
                }
            )
        )
        self.policy = {
            "listen": {"host": "127.0.0.1", "port": 8443},
            "max_active": 1,
            "models": {
                "glm-5.3": {
                    "enabled": True,
                    "base_url": f"http://127.0.0.1:{port}",
                    "endpoints": ["/v1/messages", "/v1/chat/completions"],
                    "qualification": {
                        "file": str(q),
                        "sha256": hashlib.sha256(q.read_bytes()).hexdigest(),
                    },
                }
            },
            "clients": {
                "fixture": {
                    "enabled": True,
                    "api_key_sha256": hashlib.sha256(self.token.encode()).hexdigest(),
                    "certificate_sha256": hashlib.sha256(
                        ssl.PEM_cert_to_DER_cert((self.root / "client.pem").read_text())
                    ).hexdigest(),
                    "models": ["glm-5.3"],
                }
            },
        }
        self.path = self.root / "gateway.json"
        self.write_policy()
        self.gateway = Gateway(self.path)
        self.runner = web.AppRunner(
            self.gateway.app(), handler_cancellation=True, access_log=None
        )
        await self.runner.setup()
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(self.root / "server.pem", self.root / "server.key")
        context.load_verify_locations(self.root / "ca.pem")
        context.verify_mode = ssl.CERT_REQUIRED
        site = web.TCPSite(self.runner, "127.0.0.1", 0, ssl_context=context)
        await site.start()
        self.url = f"https://localhost:{site._server.sockets[0].getsockname()[1]}"
        context = ssl.create_default_context(cafile=str(self.root / "ca.pem"))
        context.load_cert_chain(self.root / "client.pem", self.root / "client.key")
        self.session = ClientSession(connector=TCPConnector(ssl=context))
        self.headers = {"Authorization": "Bearer " + self.token}

    def write_policy(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.policy))
        tmp.replace(self.path)

    async def asyncTearDown(self):
        self.release.set()
        await self.session.close()
        await self.runner.cleanup()
        await self.backend_runner.cleanup()
        self.temp.cleanup()

    async def test_auth_catalog_acceptance_hot_reload_and_stream(self):
        async with self.session.get(self.url + "/v1/models") as r:
            self.assertEqual(r.status, 401)
        async with self.session.get(self.url + "/v1/models", headers=self.headers) as r:
            self.assertEqual((await r.json())["data"][0]["id"], "glm-5.3")
        from sparkfleet.api import Client

        with patch.dict(
            os.environ,
            {
                "SPARKFLEET_CA_FILE": str(self.root / "ca.pem"),
                "SPARKFLEET_CERT_FILE": str(self.root / "client.pem"),
                "SPARKFLEET_KEY_FILE": str(self.root / "client.key"),
            },
        ):
            client = Client(self.url + "/v1", self.token)
            status, data, _ = await asyncio.to_thread(client.request, "/v1/models")
            self.assertEqual(status, 200)
            self.assertEqual(data["data"][0]["id"], "glm-5.3")
        self.policy["models"]["glm-5.3"]["enabled"] = False
        self.write_policy()
        async with self.session.get(self.url + "/v1/models", headers=self.headers) as r:
            self.assertEqual((await r.json())["data"], [])
        async with self.session.post(
            self.url + "/v1/messages", headers=self.headers, json={"model": "glm-5.3"}
        ) as r:
            self.assertEqual(r.status, 503)
        self.policy["models"]["glm-5.3"]["enabled"] = True
        self.write_policy()
        async with self.session.post(
            self.url + "/v1/chat/completions",
            headers=self.headers,
            json={"model": "glm-5.3", "stream": True},
        ) as r:
            self.assertEqual(r.status, 200)
            self.assertIn("[DONE]", await r.text())
        self.policy["models"]["glm-5.3"]["qualification"]["sha256"] = "0" * 64
        self.write_policy()
        self.assertFalse(self.gateway.reload())
        with self.assertRaises(ValueError):
            load_config(self.path)

    async def test_backpressure_cancel_and_revoke(self):
        task = asyncio.create_task(
            self.session.post(
                self.url + "/v1/messages",
                headers=self.headers,
                json={"model": "glm-5.3", "hold": True},
            )
        )
        await asyncio.wait_for(self.entered.wait(), 3)
        async with self.session.post(
            self.url + "/v1/messages", headers=self.headers, json={"model": "glm-5.3"}
        ) as r:
            self.assertEqual(r.status, 429)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(self.cancelled.wait(), 3)
        for _ in range(30):
            if not self.gateway.active:
                break
            await asyncio.sleep(0.05)
        self.assertEqual(self.gateway.active, 0)
        self.policy["clients"]["fixture"]["enabled"] = False
        self.write_policy()
        async with self.session.get(self.url + "/v1/models", headers=self.headers) as r:
            self.assertEqual(r.status, 401)

    async def test_certificate_required(self):
        context = ssl.create_default_context(cafile=str(self.root / "ca.pem"))
        async with ClientSession(connector=TCPConnector(ssl=context)) as session:
            with self.assertRaises((ClientError, ConnectionError)):
                async with session.get(
                    self.url + "/v1/models", headers=self.headers
                ) as r:
                    await r.read()
