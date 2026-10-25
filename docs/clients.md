# APIs and client integration

Use the exact enabled model ID. Catalog order must never choose a child agent's model implicitly. The API key is a gateway/device credential, not an operating-system password.

| Model | OpenAI base URL | Model ID |
|---|---|---|
| DeepSeek | `https://YOUR-GATEWAY:8443/v1` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| GLM | Same gateway base | `glm-5.3` |

`YOUR-GATEWAY` is the reachable name you own and provisioned with a matching certificate. It is a placeholder, not a hosted SparkFleet service. Over a home/private mesh network, connect the client to that network first. No owner-specific hostname, certificate or API key is shipped here.

## OpenAI-compatible curl

The optional SparkFleet gateway requires both a device certificate and the device's API key. Read the key from a private file; do not paste it into an issue or shell history as a literal.

```sh
export SPARKFLEET_URL='https://YOUR-GATEWAY:8443'
export SPARKFLEET_API_KEY="$(cat /path/to/device-api-key)"
export SPARKFLEET_CA_FILE='/path/to/ca.pem'
export SPARKFLEET_CERT_FILE='/path/to/client.pem'
export SPARKFLEET_KEY_FILE='/path/to/client.key'

# Keep credentials off process argv as well as shell history.
curl --config - --cacert "$SPARKFLEET_CA_FILE" \
  --cert "$SPARKFLEET_CERT_FILE" --key "$SPARKFLEET_KEY_FILE" \
  "$SPARKFLEET_URL/v1/chat/completions" <<EOF_CURL
header = "Authorization: Bearer $SPARKFLEET_API_KEY"
header = "Content-Type: application/json"
data = "{\"model\":\"glm-5.3\",\"max_tokens\":128,\"messages\":[{\"role\":\"user\",\"content\":\"Reply briefly: hello\"}]}"
EOF_CURL
```

Change only the model value to use DeepSeek. Remove unneeded environment variables from shared shells afterward. Never use `curl -k` as the fix for a certificate error. For local native loopback testing, the model service may have a different authentication policy; do not expose that port directly.

The qualifier and benchmark CLI use the same `SPARKFLEET_*` TLS variables. An SDK must be configured with the CA and client certificate as well; a base URL and key alone cannot authenticate to this mTLS gateway.

## Anthropic Messages

The route is `POST /v1/messages` (not `/v1/chat/completions`) with `anthropic-version: 2023-06-01`, explicit model, max tokens and messages. The gateway accepts either bearer authorization or `x-api-key`, never both. Start with a small output budget and verify your client's request envelope.

For Claude-style reasoning selection:

| Client effort | DeepSeek native template level | GLM native effort |
|---|---:|---|
| low | 25 | low |
| medium | 38 | high |
| high | 50 | high |
| xhigh | 75 | max |
| max | 100 | max |

This is an adapter policy, not a claim of equivalent reasoning quality. DeepSeek Messages `thinking`/`output_config` is normalized into known template fields; the helper preserves opaque tool inputs. GLM's supported native values remain selectable. The acceptance suite defaults to low to bound test cost.

## Claude Code, Claude desktop, Cherry Studio and Chatbox

Protocol compatibility and a UI's provider support are separate. This repository does not ship or patch proprietary Claude desktop bundles, and does not promise that an unmodified official desktop app offers a custom provider field. Check the client version's supported integration mechanism before installation. The source deployment's local app integration was private/version-specific; it is not a portable native desktop feature.

For a client supporting custom OpenAI providers, supply the gateway `/v1` base, exact model and key. If it cannot present a client certificate, use an operator-reviewed **loopback-only bridge** that owns the remote certificate, or a private SSH tunnel to an authenticated local endpoint. The bridge is not included in v0.1. Do not weaken the remote gateway to accommodate a UI field limitation. A future portable bridge is a contribution opportunity.

For an Anthropic-compatible development client, configure its documented custom-provider base and model mapping through its supported extension mechanism. Keep the model explicit for all parent and child requests. Do not instruct a model that it has a tool unless that tool is actually registered in the running client.

## Child agents and session messages

- Default child model = the current parent model, resolved at invocation time. A GLM parent should not spawn DeepSeek because it sorts first in a menu.
- The client must issue the real child-agent tool, capture a returned task/session ID and surface progress. A sentence saying “dispatched task 4” is not a dispatch receipt.
- Expose queued, delivered and read/consumed states from actual transport events. Do not invent read receipts from successful enqueueing.
- Distinguish operator permission classification failures from generation failures. A classifier timeout must not be mislabeled as the main model being offline.
- Append inter-session messages in the native conversation/tool activity flow. A large fixed overlay can obscure the composer and break the whole app layout.

These are client integration requirements and regression cases, not functionality magically provided by an inference endpoint. SparkFleet provides protocol helpers and test guidance; desktop UI work remains separately versioned and tested.
