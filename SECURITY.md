# Security policy

v0.1 is alpha reference tooling. It has CPU and local TLS tests, but no independent security audit or certification for hostile multi-tenant operation. Use only on fleets you are authorized to administer.

Report a suspected vulnerability through the repository's private GitHub **Security → Report a vulnerability** flow. Include the affected commit, a synthetic reproduction, impact and any mitigation. Do not disclose real keys, private addresses or production prompts. If private reporting is unavailable, open a minimal public issue asking for a private reporting channel without exploit details.

Trust boundaries include operator SSH/Docker access, runtime images and weights, local config/receipt ownership, client PKI, the optional gateway and third-party kernels. A content hash is integrity evidence, not proof of authorization or quality. A passing smoke suite is not a security review.

Supported security work currently targets the latest source release. Keep Python, OpenSSL, aiohttp, drivers and runtime dependencies within their supported upstream versions, and requalify ABI/model behavior after changes. Do not disable certificate verification, bypass host-key checks, clear incident fences blindly or expose native loopback model ports as a workaround.
