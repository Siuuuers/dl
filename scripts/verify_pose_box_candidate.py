#!/usr/bin/env python3
"""Verify an incremental training-box overlay without changing source labels."""

import argparse
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from evaluate_pose import _relative_file, file_sha256, parse_pose_labels
from verify_pose_candidate import (COUNT_KEYS, GEOMETRY_EPSILON_PX, check_geometry,
                                   read_reference, require, totals)


BOX_CANDIDATE = "data/candidates/box-refinement-2026-11-07/manifest.json"
BOX_KIND = "training_box_overlay"
EDGES = ("left", "top", "right", "bottom")


def verify_box_candidate(root, candidate):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    proposal = json.loads(candidate.read_text(encoding="utf-8"))
    require(isinstance(proposal, dict), "candidate manifest must be an object")
    require(type(proposal.get("schema_version")) is int and proposal["schema_version"] == 1,
            "unsupported box candidate schema_version")
    require(proposal.get("status") == "proposed" and proposal.get("kind") == BOX_KIND,
            "expected a proposed training_box_overlay")
    require(isinstance(proposal.get("candidate_id"), str) and proposal["candidate_id"].strip(),
            "candidate_id must be a nonempty string")
    require(isinstance(proposal.get("source_revision"), str)
            and re.fullmatch(r"[0-9a-f]{40}", proposal["source_revision"]), "expected full source revision SHA")
    policy = proposal["annotation_policy"]
    require(policy["id"] == "baomai-visible-front-v1"
            and policy["path"] == "docs/annotation-policy.md"
            and file_sha256(_relative_file(root, policy["path"])) == policy["sha256"],
            "annotation policy identity differs")
    review = proposal["review"]
    require(file_sha256(_relative_file(candidate.parent, review["path"])) == review["sha256"],
            "review evidence SHA-256 mismatch")
    reference_hash, records, by_id, source_bytes, gold, source_counts = read_reference(
        root, proposal["source_manifest_sha256"])
    changes = proposal["changes"]
    require(isinstance(changes, list) and changes, "changes must be a nonempty list")
    changed, listed_paths, changed_rows = set(), set(), 0
    effective = dict(gold)
    for change in changes:
        image_id = change["image_id"]
        require(image_id in by_id and image_id not in changed, "unknown or repeated changed image")
        changed.add(image_id)
        record = by_id[image_id]
        require(record["split"] == "train", f"validation changes are forbidden: {image_id}")
        require(change["source_label"] == change["candidate_label"] == record["label"],
                f"candidate/source label mapping mismatch: {image_id}")
        require(change["source_label_sha256"] == record["label_sha256"]
                and change["source_image_sha256"] == record["image_sha256"],
                f"source identity mismatch: {image_id}")
        lines = source_bytes[image_id].splitlines(keepends=True)
        require(len(lines) == len(gold[image_id]) and all(line.strip() for line in lines),
                f"box overlays require one nonblank source row per object: {image_id}")
        path = _relative_file(candidate.parent, change["candidate_label"])
        require(file_sha256(path) == change["candidate_label_sha256"],
                f"candidate label SHA-256 mismatch: {image_id}")
        listed_paths.add(change["candidate_label"])
        candidate_bytes = path.read_bytes()
        candidate_lines = candidate_bytes.splitlines(keepends=True)
        require(len(candidate_lines) == len(lines), f"box candidate cannot add or remove rows: {image_id}")
        objects = parse_pose_labels(candidate_bytes.decode("utf-8"), *record["original_size"], str(path))
        check_geometry(objects, *record["original_size"], str(path))
        require(len(objects) == len(gold[image_id]), f"box candidate object count changed: {image_id}")
        for key in ("source_objects", "candidate_objects"):
            require(type(change[key]) is int and change[key] == len(objects), f"invalid {key}: {image_id}")
        rows = change["rows"]
        require(isinstance(rows, list) and rows, f"nonempty box edit rows required: {image_id}")
        seen = set()
        expected_lines = list(lines)
        for row in rows:
            number = row["original_row"]
            require(type(number) is int and 1 <= number <= len(lines) and number not in seen,
                    f"invalid or repeated original row: {image_id}")
            seen.add(number)
            original = lines[number - 1].decode("utf-8").rstrip("\r\n")
            proposed = candidate_lines[number - 1].decode("utf-8").rstrip("\r\n")
            require(row["source_row_text"] == original and row["candidate_row_text"] == proposed,
                    f"box row audit text differs: {image_id}:{number}")
            before, after = original.split(), proposed.split()
            require(before[0] == after[0] and before[5:] == after[5:],
                    f"box candidate must preserve exact class and endpoint tokens: {image_id}:{number}")
            source_box, proposed_box = gold[image_id][number - 1]["bbox_xyxy"], objects[number - 1]["bbox_xyxy"]
            actual_edges = [name for name, a, b in zip(EDGES, source_box, proposed_box)
                            if abs(a - b) > GEOMETRY_EPSILON_PX]
            require(actual_edges and row["box_edges_changed"] == actual_edges,
                    f"changed box edge inventory differs: {image_id}:{number}")
            coordinates = row["candidate_bbox_xyxy_px"]
            require(isinstance(coordinates, list) and len(coordinates) == 4
                    and all(type(value) in (int, float) and math.isfinite(value) for value in coordinates)
                    and all(abs(a - b) <= GEOMETRY_EPSILON_PX for a, b in zip(coordinates, proposed_box)),
                    f"candidate box audit coordinates differ: {image_id}:{number}")
            require(row.get("review_status") == "accepted_incremental"
                    and isinstance(row.get("reason"), str) and row["reason"].strip(),
                    f"box edit needs an accepted incremental disposition and reason: {image_id}:{number}")
            expected_lines[number - 1] = candidate_lines[number - 1]
        require(candidate_bytes == b"".join(expected_lines),
                f"unlisted rows changed in box candidate: {image_id}")
        changed_rows += len(seen)
        effective[image_id] = objects
    label_dir = candidate.parent / "labels"
    require(not label_dir.is_symlink(), "overlay labels directory must not be a symlink")
    actual_paths = {path.relative_to(candidate.parent).as_posix() for path in label_dir.rglob("*")
                    if path.is_file() or path.is_symlink()}
    require(actual_paths == listed_paths, "unexpected or missing label files in box candidate overlay")
    expected = proposal["expected_counts"]
    require(isinstance(expected, dict) and set(expected) == {"train", "val"}, "invalid expected splits")
    for split in ("train", "val"):
        require(isinstance(expected[split], dict) and set(expected[split]) == set(COUNT_KEYS)
                and all(type(value) is int and value >= 0 for value in expected[split].values()),
                "invalid expected counts")
    actual_counts = totals(records, effective)
    require(actual_counts == expected == source_counts, "box candidate must preserve all source counts")
    return {"status": "verified", "candidate_id": proposal["candidate_id"], "kind": BOX_KIND,
            "candidate_manifest_sha256": file_sha256(candidate),
            "declared_source_revision": proposal["source_revision"], "source_revision_git_object_checked": False,
            "source_manifest_sha256": reference_hash, "reference_labels_verified": len(records),
            "changed_train_labels": len(changed), "changed_box_rows": changed_rows, "removed_objects": 0,
            "counts": actual_counts, "validation_labels_unchanged": True,
            "class_and_endpoint_tokens_unchanged": True, "image_bytes_verified": False,
            "images_staged_by_check": False, "dataset_built": False, "visual_correctness_verified": False,
            "mechanical_geometry_verified": True,
            "code_sha256": {f"scripts/{name}": file_sha256(Path(__file__).resolve().parent / name)
                            for name in ("verify_pose_box_candidate.py", "verify_pose_candidate.py",
                                         "evaluate_pose.py", "pose_scoring.py")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", type=Path, default=Path(BOX_CANDIDATE))
    args = parser.parse_args()
    try:
        candidate = args.candidate if args.candidate.is_absolute() else args.root / args.candidate
        result = verify_box_candidate(args.root, candidate)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, allow_nan=False))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
