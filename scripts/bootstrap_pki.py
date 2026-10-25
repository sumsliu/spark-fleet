#!/usr/bin/env python3
"""Create a private, local test/small-fleet PKI. Never sends or prints keys.

For an established fleet, use its managed CA/enrollment process instead.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import ssl
import subprocess


def create(out, hostname, device):
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9.-]{0,250}", hostname
    ) or not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", device):
        raise ValueError("Use a DNS hostname and a lowercase device identifier")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    old = os.umask(0o077)

    def openssl(*args):
        subprocess.run(
            ["openssl", *args],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    try:
        (root / "ca.cnf").write_text(
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
            "rsa:3072",
            "-nodes",
            "-days",
            "730",
            "-subj",
            "/CN=SparkFleet Local CA",
            "-keyout",
            "ca.key",
            "-out",
            "ca.pem",
        )
        for name, cn, usage in [
            ("server", hostname, "serverAuth"),
            ("client", device, "clientAuth"),
        ]:
            openssl(
                "req",
                "-new",
                "-newkey",
                "rsa:3072",
                "-nodes",
                "-subj",
                "/CN=" + cn,
                "-keyout",
                name + ".key",
                "-out",
                name + ".csr",
            )
            ext = (
                "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\nextendedKeyUsage="
                + usage
                + "\n"
            )
            if name == "server":
                ext += "subjectAltName=DNS:" + hostname + "\n"
            (root / "leaf.cnf").write_text(ext)
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
                "30",
                "-extfile",
                "leaf.cnf",
                "-out",
                name + ".pem",
            )
        token = secrets.token_urlsafe(32)
        (root / "api-key").write_text(token + "\n")
        pin = hashlib.sha256(
            ssl.PEM_cert_to_DER_cert((root / "client.pem").read_text())
        ).hexdigest()
        metadata = {
            "device": device,
            "api_key_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "certificate_sha256": pin,
            "valid_days": 30,
            "note": "No credential values printed. Keep CA private key offline; never commit this directory.",
        }
        (root / "enrollment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for p in root.iterdir():
            p.chmod(0o600)
        return metadata
    finally:
        os.umask(old)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--server-host", required=True)
    parser.add_argument("--device", required=True)
    a = parser.parse_args()
    print(json.dumps(create(a.out, a.server_host, a.device), indent=2))
