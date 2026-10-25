# Contributing to SparkFleet

Thank you for helping make multi-node inference easier to reproduce. A careful reproduction, a clear failure report or a small compatibility fix can be more useful than a large speculative rewrite.

## Start with one concrete result

- **Reproduce a recipe:** publish sanitized runtime/weight identities, topology, fabric settings, exact commands, cache state, correctness and paired latency/throughput.
- **Port a runtime:** supply source commits, build steps, license provenance, preimage changes and the full acceptance matrix. A newer image tag alone is not a port.
- **Improve acceptance:** contribute synthetic long tool loops, output-quality cases, cancellation checks and client-routing regression tests.
- **Measure the fabric:** report per-rail TCP/RDMA/NCCL results with units and actual transport, including failures.
- **Improve docs and first use:** run the planning workflow from a clean checkout and report where a prerequisite or instruction was missing.

Use a bug or benchmark issue, or discuss a large change before implementation. Good first issues are intentionally bounded; ask for clarification in the issue if the expected evidence is unclear.

## Development

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[gateway]'
python -m unittest discover -s tests -v
python -m compileall -q src scripts
sparkfleet validate examples/deepseek-tp4.json --template
sparkfleet validate examples/glm-tp8.json --template
sparkfleet audit-public .
```

The gateway tests use temporary localhost TLS certificates generated with OpenSSL and CPU backend fixtures. They never contact a production model. CI tests Python 3.11–3.13 on Linux. The core CLI has no runtime Python dependencies; aiohttp is optional for the gateway.

Keep changes narrow and readable. Add meaningful tests for guard, ownership, protocol or metric behavior changes; do not add a test merely to mirror a constant. Do not reformat or edit an exported overlay without updating its manifest and explaining the new behavioral scope. Documentation-only changes usually need link/command review rather than a new test.

## Evidence and review

A performance PR needs a theory of exposed work, expected net savings including overhead, paired baseline/candidate/recheck data and correctness evidence. Show regressions and negative results. Never call a gain proven when the baseline drift is larger than the benefit. Keep promotion separate from measurement.

Do not include secrets, certificates/private keys, internal addresses, raw projects/conversations, model weights, compiled proprietary artifacts or one-use cluster authority. Run the export scanner and inspect staged files. Its pattern matching is a backstop, not a proof that a file is public-safe.

Retain upstream license notices and identify modified files. Contributions are accepted under Apache-2.0 unless an existing file's notice requires otherwise. Follow [AGENTS.md](AGENTS.md) for operator and agent work. Report exploitable security issues privately through GitHub's security reporting mechanism, not a public issue containing credentials.
