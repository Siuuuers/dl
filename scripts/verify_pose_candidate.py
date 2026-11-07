#!/usr/bin/env python3
"""Read-only verification of a proposed deletion-only training-label overlay."""

import sys
sys.dont_write_bytecode = True

import argparse
import json
import re
from pathlib import Path

from evaluate_pose import _relative_file, file_sha256, parse_pose_labels


DEFAULT_CANDIDATE = "data/candidates/dedup-2026-11-06/manifest.json"
COUNT_KEYS = ("images", "objects", "visible_endpoints", "missing_endpoints", "empty_images")
GEOMETRY_EPSILON_PX = 1e-4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(objects):
    visible = sum(point[2] == 2 for obj in objects for point in obj["keypoints"])
    return dict(zip(COUNT_KEYS, (1, len(objects), visible, 2 * len(objects) - visible, int(not objects))))


def totals(records, objects):
    result = {split: dict.fromkeys(COUNT_KEYS, 0) for split in ("train", "val")}
    for record in records:
        for key, value in counts(objects[record["id"]]).items():
            result[record["split"]][key] += value
    return result


def check_geometry(objects, width, height, source):
    eps = GEOMETRY_EPSILON_PX
    for index, obj in enumerate(objects, 1):
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        require(x1 >= -eps and y1 >= -eps and x2 <= width + eps and y2 <= height + eps,
                f"box outside image: {source}:{index}")
        for x, y, visible in obj["keypoints"]:
            if visible == 0:
                continue
            require(0 <= x < width and 0 <= y < height,
                    f"visible endpoint outside strict image bounds: {source}:{index}")
            require(x1 - eps <= x <= x2 + eps and y1 - eps <= y <= y2 + eps,
                    f"visible endpoint outside its box: {source}:{index}")


def read_reference(root, expected_manifest_sha256):
    reference_path = _relative_file(root, "data/manifest.json")
    reference_hash = file_sha256(reference_path)
    require(reference_hash == expected_manifest_sha256, "source manifest SHA-256 mismatch")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    records = reference["records"]
    require(isinstance(records, list) and len(records) == 157, "expected all 157 reference records")
    by_id, source_bytes, gold = {}, {}, {}
    for record in records:
        image_id, split = record["id"], record["split"]
        require(isinstance(image_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", image_id), "invalid image ID")
        require(split in ("train", "val") and image_id not in by_id, "invalid split or duplicate image ID")
        require(record["label"] == f"labels/{split}/{image_id}.txt"
                and record["image"] == f"images/{split}/{image_id}.png", f"reference mapping mismatch: {image_id}")
        dimensions = record["original_size"]
        require(isinstance(dimensions, list) and len(dimensions) == 2
                and all(type(value) is int and value > 0 for value in dimensions), "invalid reference dimensions")
        path = _relative_file(root / "data", record["label"])
        require(file_sha256(path) == record["label_sha256"], f"source label SHA-256 mismatch: {image_id}")
        source_bytes[image_id] = path.read_bytes()
        gold[image_id] = parse_pose_labels(source_bytes[image_id].decode("utf-8"), *dimensions, str(path))
        actual = counts(gold[image_id])
        require(all(actual[key] == record["instances" if key == "objects" else key]
                    for key in ("objects", "visible_endpoints", "missing_endpoints")),
                f"reference label counts mismatch: {image_id}")
        by_id[image_id] = record
    source_counts = totals(records, gold)
    for split, size in (("train", 137), ("val", 20)):
        expected = reference["split_summary"][split]
        require(source_counts[split]["images"] == size
                and all(source_counts[split][key] == expected["instances" if key == "objects" else key]
                        for key in COUNT_KEYS), f"reference {split} totals mismatch")
    return reference_hash, records, by_id, source_bytes, gold, source_counts


def verify_candidate(root, candidate):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    proposal = json.loads(candidate.read_text(encoding="utf-8"))
    require(isinstance(proposal, dict), "candidate manifest must be an object")
    require(type(proposal.get("schema_version")) is int and proposal["schema_version"] == 1,
            "unsupported candidate schema_version")
    require(proposal.get("status") == "proposed" and proposal.get("kind") == "training_label_overlay",
            "expected a proposed training_label_overlay")
    require(isinstance(proposal.get("candidate_id"), str) and proposal["candidate_id"].strip(),
            "candidate_id must be a nonempty string")
    require(isinstance(proposal.get("source_revision"), str)
            and re.fullmatch(r"[0-9a-f]{40}", proposal["source_revision"]), "expected full source revision SHA")
    reference_hash, records, by_id, source_bytes, gold, source_counts = read_reference(
        root, proposal["source_manifest_sha256"])

    changes = proposal["changes"]
    require(isinstance(changes, list) and changes, "changes must be a nonempty list")
    changed, listed_paths, removed_total = set(), set(), 0
    effective = dict(gold)
    for change in changes:
        image_id = change["image_id"]
        require(image_id in by_id and image_id not in changed, "unknown or repeated changed image")
        changed.add(image_id)
        record = by_id[image_id]
        require(record["split"] == "train", f"validation changes are forbidden: {image_id}")
        require(change["source_label"] == record["label"] and change["candidate_label"] == record["label"],
                f"candidate/source label mapping mismatch: {image_id}")
        require(change["source_label_sha256"] == record["label_sha256"], f"change source hash mismatch: {image_id}")
        require(change["source_image_sha256"] == record["image_sha256"], f"image hash metadata mismatch: {image_id}")
        require(type(change["source_objects"]) is int and change["source_objects"] == len(gold[image_id]),
                f"source_objects mismatch: {image_id}")
        path = _relative_file(candidate.parent, change["candidate_label"])
        require(file_sha256(path) == change["candidate_label_sha256"], f"candidate label hash mismatch: {image_id}")
        listed_paths.add(change["candidate_label"])
        lines, remove, keep = source_bytes[image_id].splitlines(keepends=True), set(), set()
        pairs = change["pairs"]
        require(isinstance(pairs, list) and pairs, f"nonempty deletion pairs required: {image_id}")
        for pair in pairs:
            a, b = pair["remove_original_row"], pair["keep_original_row"]
            require(all(type(n) is int and 1 <= n <= len(lines) and lines[n - 1].strip() for n in (a, b)),
                    f"pair rows must be integer original nonblank line numbers: {image_id}")
            require(a not in remove and b not in keep and a != b, f"duplicate pair row: {image_id}")
            require(isinstance(pair.get("reason"), str) and pair["reason"].strip(), "pair reason is required")
            require(pair["removed_row_text"] == lines[a - 1].decode("utf-8").rstrip("\r\n")
                    and pair["retained_row_text"] == lines[b - 1].decode("utf-8").rstrip("\r\n"),
                    f"pair audit text differs from original rows: {image_id}:{a}/{b}")
            removed = parse_pose_labels(lines[a - 1].decode("utf-8"), *record["original_size"], f"{image_id}:{a}")
            require(len(removed) == 1 and all(point[2] == 0 for point in removed[0]["keypoints"]),
                    f"only both-missing rows may be removed: {image_id}:{a}")
            remove.add(a)
            keep.add(b)
        require(not remove & keep, f"a kept row is also removed: {image_id}")
        for pair in pairs:
            b = pair["keep_original_row"]
            require(type(pair["retained_candidate_row"]) is int
                    and pair["retained_candidate_row"] == b - sum(a < b for a in remove),
                    f"retained_candidate_row mismatch: {image_id}:{b}")
        expected_bytes = b"".join(line for number, line in enumerate(lines, 1) if number not in remove)
        actual_bytes = path.read_bytes()
        require(actual_bytes == expected_bytes, f"candidate is not the exact byte-preserving deletion: {image_id}")
        effective[image_id] = parse_pose_labels(actual_bytes.decode("utf-8"), *record["original_size"], str(path))
        require(type(change["candidate_objects"]) is int and change["candidate_objects"] == len(effective[image_id]),
                f"candidate_objects mismatch: {image_id}")
        removed_total += len(remove)
    label_dir = candidate.parent / "labels"
    require(not label_dir.is_symlink(), "overlay labels directory must not be a symlink")
    actual_paths = {path.relative_to(candidate.parent).as_posix() for path in label_dir.rglob("*")
                    if path.is_file() or path.is_symlink()}
    require(actual_paths == listed_paths, "unexpected or missing label files in candidate overlay")
    actual_counts, expected_counts = totals(records, effective), proposal["expected_counts"]
    require(isinstance(expected_counts, dict) and set(expected_counts) == {"train", "val"}, "invalid expected splits")
    for split in ("train", "val"):
        expected = expected_counts[split]
        require(isinstance(expected, dict) and set(expected) == set(COUNT_KEYS)
                and all(type(value) is int and value >= 0 for value in expected.values()), "invalid expected counts")
    require(actual_counts == expected_counts, "candidate totals differ from expected_counts")
    require(actual_counts["val"] == source_counts["val"], "validation totals changed")
    script_dir = Path(__file__).resolve().parent
    return {"status": "verified", "candidate_id": proposal["candidate_id"],
            "candidate_manifest_sha256": file_sha256(candidate), "declared_source_revision": proposal["source_revision"],
            "source_revision_git_object_checked": False, "source_manifest_sha256": reference_hash,
            "reference_labels_verified": len(records), "changed_train_labels": len(changed),
            "removed_objects": removed_total, "counts": actual_counts, "validation_labels_unchanged": True,
            "image_bytes_verified": False, "images_staged_by_check": False, "dataset_built": False,
            "visual_correctness_verified": False, "code_sha256": {
                f"scripts/{name}": file_sha256(script_dir / name) for name in
                ("verify_pose_candidate.py", "evaluate_pose.py", "pose_scoring.py")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", type=Path, help="manifest path; relative paths resolve from --root")
    args = parser.parse_args()
    candidate = args.candidate or Path(DEFAULT_CANDIDATE)
    try:
        result = verify_candidate(args.root, candidate if candidate.is_absolute() else args.root / candidate)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, allow_nan=False))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
