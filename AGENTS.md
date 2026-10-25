# Contributor and agent instructions

This repository contains portable tooling and reference overlays, not owner
authorization for any cluster. Use only the access and scope granted by the
operator of your target cluster. Never copy another deployment's one-use plans,
incident acknowledgements, boot identities, device keys or experiment data.

Keep doctor/render commands read-only on cluster machines. Cluster mutation is
confined to explicit `sparkfleet run ... --execute`. Preserve all four locks,
exact ownership checks, fail-stop coordination, persistent incident fences,
85 C test abort, 90 C upper trip envelope and 8 GiB memory floor. Do not turn
normal API traffic into a fault; drain it before an owner-managed switch.

Do not modify a live cluster to make documentation or tests pass. Test the
portable control plane with explicit CPU fixtures; mark fresh GPU qualification
separately. Do not present historical reference results as a reproduction of
the current checkout. New images, overlays, models and topologies need their
own API, correctness, thermal and performance evidence.

Reference overlays are version-locked third-party derivatives. Retain notices,
preimage checks and modified-file markers. Never force-copy an overlay over an
unknown runtime or transplant GLM kernels into DeepSeek. Keep raw conversations,
credentials, private addresses, corporate data, weights and proprietary desktop
client bundles outside this public tree. Run `sparkfleet audit-public .` and
review staged files before publishing.

Default subagents in an integrated client inherit the current parent model.
Catalog order is not a routing rule. Protocol transforms must not recurse into
opaque tool inputs. Do not claim success after an error or a mere catalog 200.

Validation: `python -m unittest discover -s tests -v`; optional gateway tests
require `pip install '.[gateway]'`. Never use real production sessions as test
fixtures. Do not add recurring automation or post promotion to external social
channels without an explicit user request.
