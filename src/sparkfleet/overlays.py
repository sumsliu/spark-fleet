"""Fingerprint-locked reference overlays. Never writes to an installed runtime."""

from pathlib import Path
import shutil
from .common import digest, read_json, save_json


def inspect_overlay(folder, runtime_root=None):
    folder = Path(folder).resolve()
    m = read_json(folder / "manifest.json")
    rows = []
    for name, row in m["files"].items():
        p = folder / name
        if (
            p.is_symlink()
            or not p.resolve().is_relative_to(folder)
            or not p.is_file()
            or digest(p.read_bytes()) != row["sha256"]
        ):
            raise ValueError("Overlay payload differs: " + name)
        if runtime_root is not None:
            root = Path(runtime_root).resolve()
            q = root / name
            if q.is_symlink() or not q.resolve().is_relative_to(root):
                raise ValueError("Unsafe runtime path")
            expected = row.get("preimage_sha256")
            if row.get("new_file"):
                if q.exists():
                    raise ValueError("New extension already exists: " + name)
            elif not expected or not q.is_file() or digest(q.read_bytes()) != expected:
                raise ValueError(
                    "Runtime preimage mismatch: "
                    + name
                    + "; port and requalify, do not force-copy"
                )
        rows.append(dict(path=name, sha256=row["sha256"], license=row["license"]))
    return dict(
        passed=True,
        name=m["name"],
        runtime_preimages_checked=runtime_root is not None,
        files=rows,
        status=m["status"],
    )


def materialize(folder, runtime_root, out):
    result = inspect_overlay(folder, runtime_root)
    folder = Path(folder).resolve()
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Output must be a new directory")
    if out.is_relative_to(Path(runtime_root).resolve()):
        raise ValueError("Do not write overlays into the runtime tree")
    out.mkdir(parents=True)
    for row in result["files"]:
        p = out / row["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / row["path"], p)
    save_json(
        out / "mounts.json",
        [
            dict(
                source=str(out / row["path"]),
                target="/usr/local/lib/python3.12/dist-packages/" + row["path"],
                sha256=row["sha256"],
            )
            for row in result["files"]
        ],
    )
    return result
