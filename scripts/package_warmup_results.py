#!/usr/bin/env python3
"""Package a matched original-reference and reduced-bias-warmup comparison."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

from evaluate_pose import file_sha256, preflight_dataset
from package_gpu_results import RUN_PATH_SETTINGS, collect_arm, copy_member
from pose_checkpoint import ORIGINAL_RECIPE_ID, WARMUP_RECIPE_ID, _same, recipe_identity


INDEX_NAME = "warmup-results-manifest.json"
ROLE_RECIPES = {"control": ORIGINAL_RECIPE_ID, "warmup": WARMUP_RECIPE_ID}
CHANGED_SETTING = "warmup_bias_lr"


def collect_run(preflight, run, role):
    """Keep the public checkpoint gate; experiment roles are both reference labels."""
    bundle = collect_arm(preflight, run, "reference")
    if bundle["receipt"]["schema_version"] != 2:
        raise ValueError(f"--{role} requires a fresh schema-2 run; legacy results are not a matched control")
    expected = recipe_identity(ROLE_RECIPES[role])
    if any(bundle["identity"].get(key) != value for key, value in expected.items()):
        raise ValueError(f"--{role} requires recipe {expected['recipe_id']}")
    bundle["files"] = [{**row, "path": f"{role}/{row['path'].removeprefix('reference/')}"}
                       for row in bundle["files"]]
    return bundle


def shared_settings(receipt):
    return {key: value for key, value in receipt["training_settings"].items()
            if key not in RUN_PATH_SETTINGS and key != CHANGED_SETTING}


def check_pair(control, warmup):
    """Permit one declared recipe change while retaining source/data/runtime matching."""
    control_data = control["staging"]["variants"]["reference"]
    warmup_data = warmup["staging"]["variants"]["reference"]
    # Each staged YAML embeds its own absolute output path. Its hash is checked
    # within its bundle; the copied reference data must match across the runs.
    pairs = [("training source", control["receipt"]["source"], warmup["receipt"]["source"]),
             ("runtime", control["receipt"]["runtime"], warmup["receipt"]["runtime"]),
             ("staging source", control["staging"]["source"], warmup["staging"]["source"]),
             ("reference file inventory", control_data["files"], warmup_data["files"]),
             ("reference counts", control_data["counts"], warmup_data["counts"]),
             ("shared training settings", shared_settings(control["receipt"]),
              shared_settings(warmup["receipt"]))]
    for name, first, second in pairs:
        if not _same(first, second):
            raise ValueError(f"control/warmup {name} differ; return a matched comparison")


def verify_archive(path, index_bytes, rows):
    """Read the complete return ZIP back before publishing it."""
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


def package_results(root, control, warmup, output):
    """Create a new portable evidence ZIP without loading model code."""
    root = Path(root).resolve()
    runs = {}
    for role, value in (("control", control), ("warmup", warmup)):
        path = Path(value)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"--{role} must name an existing, nonsymlink run directory")
        runs[role] = path.resolve()
    if any(runs[first].is_relative_to(runs[second])
           for first, second in (("control", "warmup"), ("warmup", "control"))):
        raise ValueError("control and warmup run directories must not overlap")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists; choose a new ZIP: {output}")
    output = output.resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("output must have a .zip extension")
    if any(output.is_relative_to(path) for path in (root, *runs.values())):
        raise ValueError("output must be outside the source and both run directories")

    preflight = preflight_dataset(root, require_weights=False)
    bundles = {role: collect_run(preflight, run, role) for role, run in runs.items()}
    check_pair(bundles["control"], bundles["warmup"])
    rows = sorted((row for bundle in bundles.values() for row in bundle["files"]), key=lambda row: row["path"])
    control_receipt = bundles["control"]["receipt"]
    index = {
        "schema_version": 1, "kind": "baomai-bias-warmup-results",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": {"label_arm": "reference", "changed_setting": CHANGED_SETTING,
                       "values": {role: bundle["receipt"]["training_settings"][CHANGED_SETTING]
                                  for role, bundle in bundles.items()}},
        "source": bundles["control"]["staging"]["source"],
        "training_source": control_receipt["source"], "runtime": control_receipt["runtime"],
        "shared_training_settings": shared_settings(control_receipt),
        "reference_counts": bundles["control"]["staging"]["variants"]["reference"]["counts"],
        "runs": {role: {"checkpoint_manifest": f"{role}/checkpoint-manifest.json",
                        "checkpoint_identity": bundle["identity"]} for role, bundle in bundles.items()},
        "files": [{key: value for key, value in row.items() if key != "source_path"} for row in rows],
        "verification_scope": "Artifact integrity and paired-run eligibility; no model loading, inference, or accuracy approval.",
    }
    index_bytes = (json.dumps(index, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".warmup-results-", dir=output.parent) as temporary:
        archive_path = Path(temporary) / "results.zip"
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr(INDEX_NAME, index_bytes)
            for row in rows:
                copy_member(archive, row)
        verify_archive(archive_path, index_bytes, rows)
        # Publish on the same filesystem without overwriting a racing writer.
        os.link(archive_path, output)
    return {"archive": str(output), "sha256": file_sha256(output), "size_bytes": output.stat().st_size,
            "files": len(rows) + 1, "runs": index["runs"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="frozen source repository (defaults to this script's repository)")
    parser.add_argument("--control", type=Path, required=True,
                        help="completed original-recipe run on reference labels")
    parser.add_argument("--warmup", type=Path, required=True,
                        help="completed reduced-bias-warmup run on reference labels")
    parser.add_argument("--output", type=Path, required=True, help="new ZIP outside the source and run directories")
    args = parser.parse_args()
    try:
        summary = package_results(args.root, args.control, args.warmup, args.output)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"Warmup return package refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
