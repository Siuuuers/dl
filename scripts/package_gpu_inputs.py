#!/usr/bin/env python3
"""Package this clean Git revision and verified original GPU inputs as a ZIP."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

from evaluate_pose import preflight_dataset
from restore_gpu_inputs import (CANDIDATE, CHECKPOINT, KIND, SOURCE_METADATA,
                                contained_file, file_sha256, read_json, require,
                                run, verify_bundle)
from stage_pose_training import source_state

ROOT = Path(__file__).resolve().parents[1]
PACKAGER_BYTES = Path(__file__).read_bytes()


def git_identity(root):
    actual_root = Path(run(["git", "rev-parse", "--show-toplevel"], cwd=root)
                       .decode("utf-8").strip()).resolve()
    require(actual_root == root, "package must be built from the script's own Git root")
    require(run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root) == b"",
            "package requires a clean committed source checkout; commit changes first")
    return {"source_revision": run(["git", "rev-parse", "HEAD"], cwd=root).decode("ascii").strip(),
            "source_tree": run(["git", "rev-parse", "HEAD^{tree}"], cwd=root).decode("ascii").strip()}


def descriptor(path, relative):
    return {"path": relative, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def add_verified(archive, source, item):
    """Hash the very bytes written, rather than trusting an earlier filesystem read."""
    compression = zipfile.ZIP_STORED if item["path"].endswith(".png") else zipfile.ZIP_DEFLATED
    info = zipfile.ZipInfo(item["path"], datetime.now().timetuple()[:6])
    info.compress_type = compression
    info.external_attr = 0o100644 << 16
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as incoming, archive.open(info, "w", force_zip64=True) as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            outgoing.write(block)
    require(size == item["size_bytes"] and digest.hexdigest() == item["sha256"],
            f"source changed while packaging: {source}")


def package_inputs(output):
    require(not output.exists() and not output.is_symlink(),
            f"output already exists; choose a new archive: {output}")
    output = output.resolve()
    require(output != ROOT and ROOT not in output.parents,
            "write the GPU ZIP outside the source checkout")
    identity = git_identity(ROOT)
    revision = identity["source_revision"]
    # Pin the executing packager and exported bootstrap/guide to the same commit.
    for relative in ("scripts/package_gpu_inputs.py", "scripts/restore_gpu_inputs.py",
                     "docs/gpu-handoff.md"):
        committed = run(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
        current = PACKAGER_BYTES if relative.endswith("package_gpu_inputs.py") else contained_file(ROOT, relative).read_bytes()
        require(committed == current, f"executing/exported source differs from {revision}: {relative}")
    state = source_state(ROOT, ROOT / CANDIDATE)
    checked = preflight_dataset(ROOT, require_weights=True)
    card = read_json(ROOT / "metadata/model_card.json")
    require(card["checkpoint"] == CHECKPOINT and len(state["records"]) == 157,
            "source inventory differs from the baomai foundation")
    require(checked["weights_verified"], "original checkpoint bytes were not verified")
    assets = []
    for record in state["records"]:
        relative = "data/" + record["image"]
        path = contained_file(ROOT, relative)
        item = descriptor(path, "assets/" + relative)
        require(item["sha256"] == record["image_sha256"], f"image changed: {relative}")
        assets.append({**item, "restore_path": relative})
    weight = descriptor(contained_file(ROOT, CHECKPOINT), "assets/" + CHECKPOINT)
    require(weight["sha256"] == card["sha256"] and weight["size_bytes"] == card["size_bytes"],
            "checkpoint changed after validation")
    assets.append({**weight, "restore_path": CHECKPOINT})
    assets.sort(key=lambda item: item["restore_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gpu-inputs-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        bundle = temporary / "source.bundle"
        run(["git", "bundle", "create", str(bundle), "HEAD"], cwd=ROOT)
        verify_bundle(bundle, revision)
        manifest = {"schema_version": 1, "kind": KIND, "repository": "Siuuuers/dl",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), **identity,
                    "source_bundle": descriptor(bundle, "source.bundle"),
                    "bootstrap": descriptor(ROOT / "scripts/restore_gpu_inputs.py", "restore_gpu_inputs.py"),
                    "guide": descriptor(ROOT / "docs/gpu-handoff.md", "README.md"),
                    "source_metadata_sha256": {relative: file_sha256(ROOT / relative)
                                               for relative in SOURCE_METADATA},
                    "counts": state["counts"], "assets": assets}
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        archive_path = temporary / "inputs.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("manifest.json", manifest_bytes)
            archive.writestr("manifest.sha256", manifest_sha256 + "  manifest.json\n")
            add_verified(archive, bundle, manifest["source_bundle"])
            add_verified(archive, ROOT / "scripts/restore_gpu_inputs.py", manifest["bootstrap"])
            add_verified(archive, ROOT / "docs/gpu-handoff.md", manifest["guide"])
            for item in assets:
                add_verified(archive, ROOT / item["restore_path"], item)
        require(git_identity(ROOT) == identity, "source Git identity changed while packaging")
        # A hard link publishes the completed archive atomically without replacing
        # a file that appeared after our initial fresh-output check.
        os.link(archive_path, output)
    return {"status": "packaged", "archive": str(output), "size_bytes": output.stat().st_size,
            "sha256": file_sha256(output), "manifest_sha256": manifest_sha256,
            **identity, "original_images": 157, "checkpoints": 1,
            "counts": state["counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="fresh ZIP path outside this Git checkout")
    args = parser.parse_args()
    try:
        require(sys.version_info >= (3, 11), "Python 3.11 or newer is required")
        result = package_inputs(args.output.expanduser())
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"GPU input packaging failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
