# Qualification-gated gateway

The optional gateway is newly packaged reference software. CPU tests exercise real TLS, catalog gating, hot policy changes, streaming, cancellation and revocation. It has not undergone independent security review or fresh multi-node GPU qualification. Keep native model ports loopback-only.

## Install and configure

```sh
python -m pip install -e '.[gateway]'
cp examples/gateway.json work/gateway.json
```

Every model starts disabled. Supply your fixed upstream URLs, approved endpoint lists, completed exact-model qualification file and its SHA256. The receipt must cover **every** endpoint you enable. Default qualification covers Chat Completions and, unless `--openai-only`, Messages/count. Responses needs separate acceptance coverage.

For two different head nodes, use an operator-managed private authenticated path. For example, a local forwarding tunnel on the gateway can make a GLM head's loopback API available as a separate loopback backend:

```sh
ssh -N -L 127.0.0.1:8001:127.0.0.1:8000 operator@YOUR-GLM-HEAD
```

Do not point both models at one backend and assume the model name performs routing. Preserve backend authentication where configured using `upstream_key_file`; that owner-private regular file is read without printing its value.

## Device credentials

For a new isolated test or small fleet, this local helper creates a CA, a DNS-valid server certificate, a 30-day client certificate and a random device API key:

```sh
python scripts/bootstrap_pki.py --out work/pki-new \
  --server-host YOUR-GATEWAY --device home-device
```

It refuses an existing output directory, uses private file permissions and prints only fingerprints. For established fleets, use the existing managed CA/enrollment process. Protect the CA private key offline; the running gateway needs the public CA, server certificate and server key, not the CA signing key. Deliver only that device's client certificate/key, public CA and API key over your own trusted channel. Do not put any of these files in Git or an issue.

Copy `api_key_sha256` and `certificate_sha256` from the enrollment metadata into the device policy; grant only desired model IDs. Set the device enabled only after securely delivering its credentials. The key is not the hash: clients use the value in `api-key`, while policy stores its SHA256.

Point `listen.certificate_file`, `listen.key_file` and `listen.client_ca_file` at the actual private deployment paths. The server hostname must match the certificate SAN. Set a private-network listener when ready; the example binds loopback intentionally. Test network reachability, certificate expiry and device revocation before remote use.

## Run and enable

```sh
sparkfleet gateway --config work/gateway.json
```

A device must present **both** a CA-validated client certificate with the policy fingerprint and the matching bearer/API key. No trusted forwarded-certificate header is accepted. If you place a network proxy in front, retain end-to-end TLS pass-through or explicitly review a different authentication design; ordinary TLS termination would remove the client certificate from this gateway.

The gateway reloads a valid atomic policy for new requests. Changing `enabled:false` hides a model from `/v1/models` and refuses new requests while existing streams drain. Enabling requires a completed passing receipt for the exact model and all configured routes. A listener change requires a new process; use a blue/green migration if connections must stay live.

Prepare a complete policy in a temporary file, validate it with `sparkfleet.gateway.load_config`, then replace atomically. An invalid policy fails closed. Do not edit a live JSON file in place while it is being read. Validation is not a substitute for operator review of the receipt's runtime identity and wider production gates.

## Runtime behavior

- Bounded JSON body, explicit model/route allowlists, fixed upstream URLs and no redirects.
- Device-level grants; no arbitrary destination forwarding and no environment proxy inheritance.
- Configured global active-request limit with real 429/Retry-After; transport failures are 502 rather than false “model overloaded” messages.
- Transparent SSE bytes and client cancellation propagation. No synthetic tokens or early fake success.
- No access logs, request bodies, prompts, environment values or auth headers in reports.
- Device disable/key rotation/certificate expiry invalidates existing registered TLS transports on the next one-second watch check. CA and server-certificate rotation need a planned process migration.

v0.1 has no user portal, automatic certificate renewal, distributed quota store, per-user scheduler or included desktop bridge. Rotate credentials through your operator process and preserve recovery access. See [client setup](clients.md) and [operations](operations.md).
