#!/usr/bin/env python3
"""Package two completed paired-training bundles without loading either model."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

from evaluate_pose import file_sha256, preflight_dataset
from pose_checkpoint import _contained_path, _read_json, _same, select_checkpoint


INDEX_NAME = "gpu-results-manifest.json"
RUN_PATH_SETTINGS = {"data", "model", "project", "name", "save_dir"}


def portable_relative(name):
    """Reject ambiguous archive paths, including Windows separators and drives."""
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(ord(char) < 32 for char in name)):
        raise ValueError(f"artifact needs a portable relative path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name or name == ".":
        raise ValueError(f"artifact needs a canonical relative path: {name!r}")
    return name


def collect_arm(preflight, run, arm):
    manifest_path = _contained_path(run, "checkpoint-manifest.json")
    selected = select_checkpoint(preflight, manifest_path, require_weights=True)
    identity = selected["checkpoint_identity"]
    if identity["arm"] != arm:
        raise ValueError(f"--{arm} identifies a {identity['arm']} arm, not {arm}")
    manifest = _read_json(manifest_path)
    receipt = _read_json(_contained_path(run, manifest["training_receipt"]["path"]))
    staging = _read_json(_contained_path(run, manifest["staging_manifest"]["path"]))
    descriptors = [{"path": "checkpoint-manifest.json", "sha256": identity["manifest_sha256"]},
                   manifest["training_receipt"], manifest["staging_manifest"],
                   manifest["checkpoint"], receipt["results"], receipt["log"]]
    files = []
    for descriptor in descriptors:
        relative = portable_relative(descriptor["path"])
        path = _contained_path(run, relative)
        files.append({"path": f"{arm}/{relative}", "sha256": descriptor["sha256"],
                      "size_bytes": path.stat().st_size, "source_path": path})
    if len({row["path"] for row in files}) != 6:
        raise ValueError(f"{arm} bundle artifacts must use six distinct paths")
    return {"identity": identity, "receipt": receipt, "staging": staging, "files": files}


def shared_settings(receipt):
    return {key: value for key, value in receipt["training_settings"].items()
            if key not in RUN_PATH_SETTINGS}


def check_pair(reference, candidate):
    """Per-arm validators bind the recipe; this binds the two runs to one comparison."""
    pairs = [("training source", reference["receipt"]["source"], candidate["receipt"]["source"]),
             ("runtime", reference["receipt"]["runtime"], candidate["receipt"]["runtime"]),
             ("dataset proposal", reference["staging"]["source"], candidate["staging"]["source"]),
             ("shared training settings", shared_settings(reference["receipt"]),
              shared_settings(candidate["receipt"]))]
    for name, first, second in pairs:
        if not _same(first, second):
            raise ValueError(f"reference/candidate {name} differ; return a matched comparison")


def copy_member(archive, row):
    digest, size = hashlib.sha256(), 0
    info = zipfile.ZipInfo(row["path"])
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with row["source_path"].open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            target.write(chunk)
    if digest.hexdigest() != row["sha256"] or size != row["size_bytes"]:
        raise ValueError(f"artifact changed while packaging: {row['source_path']}")


def verify_archive(path, index_bytes, rows):
    """Read the ZIP back, including its raw logs, before publishing it."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        expected = {INDEX_NAME, *(row["path"] for row in rows)}
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("return ZIP inventory differs from the verified bundles")
        if archive.read(INDEX_NAME) != index_bytes:
            raise ValueError("return ZIP index changed")
        for row in rows:
            with archive.open(row["path"]) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != row["sha256"] or archive.getinfo(row["path"]).file_size != row["size_bytes"]:
                raise ValueError(f"return ZIP content differs: {row['path']}")


def package_results(root, reference, candidate, output):
    """Create a fresh ZIP; source images and initialization weights are never copied."""
    root = Path(root).resolve()
    runs = {}
    for arm, value in (("reference", reference), ("candidate", candidate)):
        path = Path(value)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"--{arm} must name an existing, nonsymlink run directory")
        runs[arm] = path.resolve()
    if any(runs[first].is_relative_to(runs[second])
           for first, second in (("reference", "candidate"), ("candidate", "reference"))):
        raise ValueError("reference and candidate run directories must not overlap")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists; choose a new ZIP: {output}")
    output = output.resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("output must have a .zip extension")
    if any(output.is_relative_to(path) for path in (root, *runs.values())):
        raise ValueError("output must be outside the source and both run directories")

    preflight = preflight_dataset(root, require_weights=False)
    arms = {arm: collect_arm(preflight, run, arm) for arm, run in runs.items()}
    check_pair(arms["reference"], arms["candidate"])
    rows = sorted((row for bundle in arms.values() for row in bundle["files"]), key=lambda row: row["path"])
    reference_receipt = arms["reference"]["receipt"]
    index = {
        "schema_version": 1, "kind": "baomai-paired-training-results",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": arms["reference"]["staging"]["source"],
        "training_source": reference_receipt["source"], "runtime": reference_receipt["runtime"],
        "shared_training_settings": shared_settings(reference_receipt),
        "arms": {arm: {"checkpoint_manifest": f"{arm}/checkpoint-manifest.json",
                       "checkpoint_identity": bundle["identity"]} for arm, bundle in arms.items()},
        "files": [{key: value for key, value in row.items() if key != "source_path"} for row in rows],
        "verification_scope": "Artifact integrity and paired-run eligibility; no model loading, inference, or accuracy approval.",
    }
    index_bytes = (json.dumps(index, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gpu-results-", dir=output.parent) as temporary:
        archive_path = Path(temporary) / "results.zip"
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr(INDEX_NAME, index_bytes)
            for row in rows:
                copy_member(archive, row)
        verify_archive(archive_path, index_bytes, rows)
        # A hard link on the same filesystem publishes atomically without overwriting a racing writer.
        os.link(archive_path, output)
    return {"archive": str(output), "sha256": file_sha256(output), "size_bytes": output.stat().st_size,
            "files": len(rows) + 1, "arms": index["arms"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="frozen source repository (defaults to this script's repository)")
    parser.add_argument("--reference", type=Path, required=True, help="completed reference run directory")
    parser.add_argument("--candidate", type=Path, required=True, help="completed candidate run directory")
    parser.add_argument("--output", type=Path, required=True, help="new ZIP outside the source and run directories")
    args = parser.parse_args()
    try:
        summary = package_results(args.root, args.reference, args.candidate, args.output)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"GPU return package refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
