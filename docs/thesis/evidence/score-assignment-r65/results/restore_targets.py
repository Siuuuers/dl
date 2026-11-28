#!/usr/bin/env python3
"""Restore the exact recorded gzip from its verified repository parts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore(directory):
    manifest = json.loads((directory / "targets.parts.json").read_text())
    if manifest["schema_version"] != 1 or manifest["logical_file"] != "targets.json.gz":
        raise ValueError("Unexpected storage manifest")
    parts = manifest["parts"]
    if not parts or [part["index"] for part in parts] != list(range(len(parts))):
        raise ValueError("Part order is incomplete or invalid")
    if len({part["name"] for part in parts}) != len(parts):
        raise ValueError("Duplicate part name")
    target = directory / manifest["logical_file"]
    temporary = None
    try:
        digest = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".targets-", delete=False) as output:
            temporary = Path(output.name)
            for part in parts:
                if Path(part["name"]).name != part["name"]:
                    raise ValueError("Part name must be a filename")
                path = directory / part["name"]
                part_digest = hashlib.sha256()
                part_size = 0
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        part_digest.update(block)
                        size += len(block)
                        part_size += len(block)
                if part_size != part["bytes"] or part_digest.hexdigest() != part["sha256"]:
                    raise ValueError("Part size or SHA-256 mismatch: " + part["name"])
            output.flush()
            os.fsync(output.fileno())
        if size != manifest["bytes"] or digest.hexdigest() != manifest["sha256"]:
            raise ValueError("Reassembled size or SHA-256 differs from recorded gzip")
        if target.exists():
            if target.stat().st_size != size or file_hash(target) != manifest["sha256"]:
                raise FileExistsError("Refusing to overwrite different existing file: " + str(target))
            print("Verified parts and unchanged existing gzip:", target)
            return
        os.link(temporary, target)
        print("Restored and verified exact gzip:", target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    restore(parser.parse_args().directory.resolve())
