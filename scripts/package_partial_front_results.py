#!/usr/bin/env python3
"""Package the fixed old-reviewed versus extended-reviewed annotation experiment.

Both inner bundles truthfully remain ``arm=candidate``. This separate outer
contract never relabels either bundle as an original-label reference arm.
No training, inference, or checkpoint deserialization is performed here.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

from evaluate_pose import file_sha256, preflight_dataset
from package_gpu_results import (INDEX_NAME, collect_arm, copy_member,
                                 shared_settings, verify_archive)
from pose_checkpoint import _contained_path, _digest, _read_json, _require, _same, recipe_identity
from train_pose_comparison import source_identity
from verify_pose_annotation_candidate import verify_annotation_candidate


ROLES = ("old_reviewed", "extended_reviewed")
PROTOCOL_KIND = "baomai-partial-fronts-comparison-protocol"
RESULT_KIND = "baomai-partial-fronts-training-results"
DEFAULT_PROTOCOL = "experiments/partial-fronts-2026-11-11.json"
OLD_MANIFEST = "data/candidates/annotations-2026-11-10/manifest.json"
OLD_SHA256 = "28fc8f5da95605089d61fb062132200ed296c3a10b216f1a61dbda1798ea4ba7"
NEW_MANIFEST = "data/candidates/partial-fronts-2026-11-11/manifest.json"
RECIPE_ID = "reviewed-annotations-seed46-v1"
INITIAL_SHA256 = "e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e"
DELTA_IMAGES = ("new_000119", "new_000120", "new_000122", "new_000125", "new_000130")


def validate_protocol(root, protocol_path=DEFAULT_PROTOCOL):
    """Verify the fixed cumulative overlays and their exact effective-label delta."""
    root = Path(root).resolve()
    requested = Path(protocol_path)
    relative = requested.relative_to(root).as_posix() if requested.is_absolute() else requested.as_posix()
    path = _contained_path(root, relative)
    protocol = _read_json(path)
    _require(type(protocol.get("schema_version")) is int and protocol["schema_version"] == 1
             and protocol.get("kind") == PROTOCOL_KIND, "unsupported partial-front protocol")
    _require(isinstance(protocol.get("experiment_id"), str) and protocol["experiment_id"].strip(),
             "protocol needs an experiment_id")
    identity = recipe_identity(RECIPE_ID)
    _require(all(protocol.get(key) == value for key, value in identity.items()),
             "protocol must use the unchanged seed46 annotation recipe")
    _require(protocol.get("initial_checkpoint_sha256") == INITIAL_SHA256,
             "protocol must initialize both runs from frozen epoch173")
    _require(protocol.get("reference_manifest_sha256") == file_sha256(root / "data/manifest.json"),
             "protocol frozen reference identity differs")
    _require(protocol.get("delta_train_image_ids") == list(DELTA_IMAGES),
             "protocol must name exactly the five accepted training-image deltas")
    roles = protocol.get("roles")
    _require(isinstance(roles, dict) and set(roles) == set(ROLES), "protocol roles differ")
    records = _read_json(root / "data/manifest.json")["records"]
    reference = {row["id"]: row for row in records}
    effective, proposals, checks = {}, {}, {}
    for role, expected_path in zip(ROLES, (OLD_MANIFEST, NEW_MANIFEST)):
        spec = roles[role]
        _require(isinstance(spec, dict) and set(spec) == {
            "candidate_manifest", "candidate_manifest_sha256", "candidate_id"},
            f"invalid protocol role descriptor: {role}")
        _require(spec["candidate_manifest"] == expected_path
                 and _digest(spec["candidate_manifest_sha256"]), f"candidate path/hash differs: {role}")
        if role == "old_reviewed":
            _require(spec["candidate_manifest_sha256"] == OLD_SHA256
                     and spec["candidate_id"] == "annotations-2026-11-10",
                     "old-reviewed overlay must retain its fixed published identity")
        else:
            _require(spec["candidate_id"] == "partial-fronts-2026-11-11",
                     "extended-reviewed candidate ID differs")
        manifest = _contained_path(root, expected_path)
        _require(file_sha256(manifest) == spec["candidate_manifest_sha256"],
                 f"candidate manifest SHA-256 differs: {role}")
        proposal = _read_json(manifest)
        _require(proposal.get("candidate_id") == spec["candidate_id"], f"candidate ID differs: {role}")
        checks[role] = verify_annotation_candidate(root, manifest)
        proposals[role] = {change["image_id"]: change for change in proposal["changes"]}
        effective[role] = {row["id"]: proposals[role].get(row["id"], {}).get(
            "candidate_label_sha256", row["label_sha256"]) for row in records}
    old_changes, new_changes = (proposals[role] for role in ROLES)
    _require(len(old_changes) == 8 and set(old_changes) <= set(new_changes),
             "extended overlay must retain all eight old-reviewed files")
    for image_id, old in old_changes.items():
        new = new_changes[image_id]
        old_path = _contained_path((root / OLD_MANIFEST).parent, old["candidate_label"])
        new_path = _contained_path((root / NEW_MANIFEST).parent, new["candidate_label"])
        _require(old_path.read_bytes() == new_path.read_bytes(),
                 f"old-reviewed label bytes changed: {image_id}")
    changed = sorted(image_id for image_id in reference
                     if effective["old_reviewed"][image_id] != effective["extended_reviewed"][image_id])
    _require(changed == list(DELTA_IMAGES), "effective label delta differs from the fixed protocol")
    _require(all(reference[image_id]["split"] == "train" for image_id in changed),
             "validation label changes are forbidden")
    return {"path": relative, "sha256": file_sha256(path), "document": protocol,
            "effective_label_sha256": effective, "records": reference,
            "candidate_checks": checks, "delta_train_image_ids": changed}


def collect_reviewed_run(preflight, run, role, context):
    """Reuse the existing six-artifact validator, retaining its candidate identity."""
    _require(role in ROLES, "unknown reviewed-label role")
    bundle = collect_arm(preflight, Path(run), "candidate")
    spec = context["document"]["roles"][role]
    source = bundle["staging"]["source"]
    for key, value in spec.items():
        _require(source.get(key) == value, f"{role} staged {key} differs from the protocol")
    identity = bundle["identity"]
    _require(all(identity.get(key) == value for key, value in recipe_identity(RECIPE_ID).items()),
             f"{role} checkpoint recipe differs")
    _require(identity.get("initial_checkpoint_sha256") == INITIAL_SHA256,
             f"{role} initialization differs")
    _require(bundle["receipt"]["runtime"]["device"].startswith("cuda:"),
             f"{role} must be a CUDA training run; no CPU fallback")
    return bundle


def check_reviewed_pair(bundles, context):
    """Bind the two independently valid candidate bundles to this one comparison."""
    old, extended = (bundles[role] for role in ROLES)
    for name, first, second in (
        ("training source", old["receipt"]["source"], extended["receipt"]["source"]),
        ("runtime", old["receipt"]["runtime"], extended["receipt"]["runtime"]),
        ("shared training settings", shared_settings(old["receipt"]), shared_settings(extended["receipt"])),
    ):
        _require(_same(first, second), f"old/extended {name} differ")
    # The proposal identities intentionally differ. Everything else in the staged
    # source must match; comparing the whole source would reject this experiment.
    changing_source = {"candidate_id", "candidate_manifest", "candidate_manifest_sha256",
                       "declared_candidate_source_revision"}
    first, second = (bundle["staging"]["source"] for bundle in (old, extended))
    _require(_same({k: v for k, v in first.items() if k not in changing_source},
                   {k: v for k, v in second.items() if k not in changing_source}),
             "old/extended shared staging source differs")
    inventories = {}
    for role, bundle in bundles.items():
        inventories[role] = {row["path"]: row["output_sha256"]
                             for row in bundle["staging"]["variants"]["candidate"]["files"]}
        expected = {}
        for image_id, row in context["records"].items():
            expected[row["image"]] = row["image_sha256"]
            expected[row["label"]] = context["effective_label_sha256"][role][image_id]
        _require(inventories[role] == expected, f"{role} candidate inventory differs from reviewed labels")
    actual_delta = sorted(path for path in inventories["old_reviewed"]
                          if inventories["old_reviewed"][path] != inventories["extended_reviewed"][path])
    expected_delta = sorted(context["records"][image_id]["label"] for image_id in DELTA_IMAGES)
    _require(actual_delta == expected_delta, "training pair changes images, validation, or unapproved labels")


def bundle_rows(bundles):
    rows = []
    for role in ROLES:
        for source in bundles[role]["files"]:
            _require(source["path"].startswith("candidate/"), "unexpected inner bundle prefix")
            rows.append({**source, "path": role + "/" + source["path"].removeprefix("candidate/")})
    return sorted(rows, key=lambda row: row["path"])


def result_index(context, bundles, rows, created_at):
    old = bundles["old_reviewed"]
    return {"schema_version": 1, "kind": RESULT_KIND, "created_at_utc": created_at,
            "experiment_id": context["document"]["experiment_id"],
            "protocol": {"path": context["path"], "sha256": context["sha256"]},
            "packager_sha256": file_sha256(Path(__file__)),
            **recipe_identity(RECIPE_ID), "initial_checkpoint_sha256": INITIAL_SHA256,
            "training_source": old["receipt"]["source"], "runtime": old["receipt"]["runtime"],
            "shared_training_settings": shared_settings(old["receipt"]),
            "delta_train_image_ids": context["delta_train_image_ids"],
            "roles": {role: {**context["document"]["roles"][role],
                       "checkpoint_manifest": f"{role}/checkpoint-manifest.json",
                       "checkpoint_identity": bundles[role]["identity"],
                       "training_counts": bundles[role]["staging"]["variants"]["candidate"]["counts"]}
                      for role in ROLES},
            "files": [{k: v for k, v in row.items() if k != "source_path"} for row in rows],
            "verification_scope": "Two complete CUDA candidate-arm bundles; same source/runtime/recipe; "
                "fixed old-reviewed labels versus the declared extension; original validation unchanged. "
                "No model loading, inference, training execution, or accuracy approval by this packager."}


def package_results(root, old_reviewed, extended_reviewed, output, *,
                    protocol=DEFAULT_PROTOCOL, verify_existing=False):
    root = Path(root).resolve()
    runs = {}
    for role, value in zip(ROLES, (old_reviewed, extended_reviewed)):
        path = Path(value)
        _require(path.is_dir() and not path.is_symlink(), f"{role} needs a nonsymlink run directory")
        runs[role] = path.resolve()
    _require(not any(runs[a].is_relative_to(runs[b]) for a, b in (ROLES, tuple(reversed(ROLES)))),
             "old/extended run directories must not overlap")
    output = Path(output)
    _require(not output.is_symlink(), "output must not be a symlink")
    if output.exists() and not verify_existing:
        raise FileExistsError(f"output already exists; use --verify-existing to verify without overwriting: {output}")
    output = output.resolve()
    _require(output.suffix.lower() == ".zip", "output must have a .zip extension")
    _require(not any(output.is_relative_to(path) for path in (root, *runs.values())),
             "output must be outside the source and both run directories")
    context = validate_protocol(root, protocol)
    preflight = preflight_dataset(root, require_weights=False)
    bundles = {role: collect_reviewed_run(preflight, run, role, context) for role, run in runs.items()}
    current_source = source_identity(root)
    _require(current_source["working_tree_status"] == "", "packaging requires the clean pinned source checkout")
    _require(all(_same(bundle["receipt"]["source"], current_source) for bundle in bundles.values()),
             "training source differs from the packaging checkout")
    check_reviewed_pair(bundles, context)
    rows = bundle_rows(bundles)
    if verify_existing:
        with zipfile.ZipFile(output) as archive:
            index_bytes = archive.read(INDEX_NAME)
        index = json.loads(index_bytes)
        created = index.get("created_at_utc")
        _require(isinstance(created, str) and datetime.fromisoformat(created).utcoffset() is not None,
                 "existing result index lacks an aware timestamp")
        expected = result_index(context, bundles, rows, created)
        expected_bytes = (json.dumps(expected, indent=2, allow_nan=False) + "\n").encode("utf-8")
        _require(index_bytes == expected_bytes, "existing result index differs from this verified pair")
        verify_archive(output, index_bytes, rows)
    else:
        index = result_index(context, bundles, rows, datetime.now(timezone.utc).isoformat())
        index_bytes = (json.dumps(index, indent=2, allow_nan=False) + "\n").encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".partial-front-results-", dir=output.parent) as temporary:
            archive_path = Path(temporary) / "results.zip"
            with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                archive.writestr(INDEX_NAME, index_bytes)
                for row in rows:
                    copy_member(archive, row)
            verify_archive(archive_path, index_bytes, rows)
            os.link(archive_path, output)
    return {"status": "verified_existing" if verify_existing else "packaged", "archive": str(output),
            "sha256": file_sha256(output), "size_bytes": output.stat().st_size,
            "files": len(rows) + 1, "roles": index["roles"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--old-reviewed", type=Path, required=True)
    parser.add_argument("--extended-reviewed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    try:
        result = package_results(args.root, args.old_reviewed, args.extended_reviewed, args.output,
                                 protocol=args.protocol, verify_existing=args.verify_existing)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"Partial-front result package refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
