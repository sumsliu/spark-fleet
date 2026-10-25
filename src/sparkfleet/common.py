"""Small primitives shared by planning, receipts and comparison."""

import hashlib
import json
import os
from pathlib import Path
import secrets


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value, *, private=True, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or (exclusive and path.exists()):
        raise ValueError(f"Refusing to overwrite {path}")
    temp = path.with_name("." + path.name + "." + secrets.token_hex(6))
    fd = os.open(
        temp,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600 if private else 0o644,
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(
                json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temp, path)  # Atomic no-clobber publication.
            temp.unlink()
        else:
            temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
