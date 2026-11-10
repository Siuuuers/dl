#!/usr/bin/env python3
"""Verify explicit, reviewed training annotation edits against frozen source bytes.

Rows are one-based original line numbers. Retained rows keep their order and
line endings; additions append with LF. A full row_map includes deleted rows
with candidate_row=null. Review decisions are hash-bound evidence, not a claim
that this mechanical verifier has adjudicated the images.
"""

import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import re

from evaluate_pose import _relative_file, file_sha256, parse_pose_labels
from verify_pose_candidate import COUNT_KEYS, check_geometry, read_reference, require, totals


ANNOTATION_CANDIDATE = "data/candidates/annotations-2026-11-10/manifest.json"
ANNOTATION_KIND = "training_annotation_overlay"
FIELD_SLICES = (("bbox", slice(1, 5)), ("left_endpoint", slice(5, 8)),
                ("right_endpoint", slice(8, 11)))


def read_json(path):
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonstandard JSON constant: {value}")

    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_fields,
                       parse_constant=reject_constant)
    require(isinstance(value, dict), f"JSON document must be an object: {path}")
    return value


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def row_text(value, dimensions, location):
    require(nonempty(value) and len(value.splitlines()) == 1 and "\n" not in value and "\r" not in value,
            f"row audit text must be one nonblank line without a newline: {location}")
    objects = parse_pose_labels(value, *dimensions, location)
    require(len(objects) == 1, f"row audit text must contain exactly one object: {location}")
    return value.split()


def check_review(operation, image_id, decisions):
    require(isinstance(operation, dict), f"annotation operation must be an object: {image_id}")
    identifier = operation.get("review_id")
    require(isinstance(identifier, str) and identifier in decisions,
            f"annotation operation needs a known review_id: {image_id}")
    decision = decisions[identifier]
    require(operation.get("review_status") == decision["review_status"] == "accepted_incremental"
            and operation.get("reason") == decision["reason"] and nonempty(operation.get("reason"))
            and decision["image_id"] == image_id,
            f"annotation operation differs from its accepted incremental review: {image_id}:{identifier}")


def verify_annotation_candidate(root, candidate):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    proposal = read_json(candidate)
    require(type(proposal.get("schema_version")) is int and proposal["schema_version"] == 1,
            "unsupported annotation candidate schema_version")
    require(proposal.get("status") == "proposed" and proposal.get("kind") == ANNOTATION_KIND,
            "expected a proposed training_annotation_overlay")
    require(nonempty(proposal.get("candidate_id")), "candidate_id must be a nonempty string")
    require(isinstance(proposal.get("source_revision"), str)
            and re.fullmatch(r"[0-9a-f]{40}", proposal["source_revision"]), "expected full source revision SHA")
    policy = proposal["annotation_policy"]
    require(isinstance(policy, dict) and policy["id"] == "baomai-visible-front-v1"
            and policy["path"] == "docs/annotation-policy.md"
            and file_sha256(_relative_file(root, policy["path"])) == policy["sha256"],
            "annotation policy identity differs")
    review = proposal["review"]
    require(isinstance(review, dict), "review descriptor must be an object")
    review_path = _relative_file(candidate.parent, review["path"])
    require(file_sha256(review_path) == review["sha256"], "review evidence SHA-256 mismatch")
    evidence = read_json(review_path)
    require(type(evidence.get("schema_version")) is int and evidence["schema_version"] == 1
            and evidence.get("candidate_id") == proposal["candidate_id"], "review candidate identity differs")
    reference_hash, records, by_id, source_bytes, gold, source_counts = read_reference(
        root, proposal["source_manifest_sha256"])
    require(isinstance(evidence.get("decisions"), list) and evidence["decisions"],
            "review decisions must be a nonempty list")
    decisions = {}
    for decision in evidence["decisions"]:
        require(isinstance(decision, dict) and nonempty(decision.get("id")), "invalid review decision")
        identifier, image_id = decision["id"], decision.get("image_id")
        require(identifier not in decisions and isinstance(image_id, str) and image_id in by_id
                and by_id[image_id]["split"] == "train", "repeated review ID or non-training review image")
        require(nonempty(decision.get("reason")) and nonempty(decision.get("review_status")),
                "review decision needs a status and reason")
        decisions[identifier] = decision

    changes = proposal["changes"]
    require(isinstance(changes, list) and changes, "changes must be a nonempty list")
    changed, listed_paths = set(), set()
    removed_total = replaced_total = added_total = merged_total = 0
    effective = dict(gold)
    for change in changes:
        require(isinstance(change, dict), "annotation change must be an object")
        image_id = change["image_id"]
        require(isinstance(image_id, str) and image_id in by_id and image_id not in changed,
                "unknown or repeated changed image")
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
                f"annotation overlays require one nonblank source row per object: {image_id}")
        texts = [line.decode("utf-8").rstrip("\r\n") for line in lines]
        require(type(change["source_objects"]) is int and change["source_objects"] == len(lines),
                f"source_objects mismatch: {image_id}")
        path = _relative_file(candidate.parent, change["candidate_label"])
        require(file_sha256(path) == change["candidate_label_sha256"],
                f"candidate label SHA-256 mismatch: {image_id}")
        listed_paths.add(change["candidate_label"])
        for key in ("deletions", "replacements", "additions"):
            require(isinstance(change[key], list), f"{key} must be a list: {image_id}")
        require(any(change[key] for key in ("deletions", "replacements", "additions")),
                f"changed image has no explicit annotation operations: {image_id}")

        def original_row(operation):
            check_review(operation, image_id, decisions)
            number = operation["original_row"]
            require(type(number) is int and 1 <= number <= len(lines),
                    f"invalid original row: {image_id}")
            require(operation["source_row_text"] == texts[number - 1],
                    f"source row audit text differs: {image_id}:{number}")
            return number

        removed = set()
        for operation in change["deletions"]:
            number = original_row(operation)
            require(number not in removed, f"repeated deletion row: {image_id}:{number}")
            removed.add(number)
        row_map, retained_count = [], 0
        for number in range(1, len(lines) + 1):
            if number not in removed:
                retained_count += 1
            row_map.append({"original_row": number,
                            "candidate_row": None if number in removed else retained_count})
        require(isinstance(change["row_map"], list)
                and json.dumps(change["row_map"], sort_keys=True, allow_nan=False)
                == json.dumps(row_map, sort_keys=True, allow_nan=False),
                f"full original-to-candidate row map differs: {image_id}")
        expected_lines, replaced, donated = list(lines), set(), set()
        for operation in change["replacements"]:
            number = original_row(operation)
            require(number not in removed and number not in replaced,
                    f"replaced row is deleted or repeated: {image_id}:{number}")
            replaced.add(number)
            require(type(operation["candidate_row"]) is int
                    and operation["candidate_row"] == row_map[number - 1]["candidate_row"],
                    f"replacement candidate_row differs: {image_id}:{number}")
            before = texts[number - 1].split()
            after = row_text(operation["candidate_row_text"], record["original_size"], f"{image_id}:{number}")
            require(before[0] == after[0], f"annotation edits must preserve the class token: {image_id}:{number}")
            for _, section in FIELD_SLICES[1:]:
                require(float(before[section][2]) != 2 or before[section] == after[section],
                        f"original visible endpoint tokens must be preserved: {image_id}:{number}")
            fields = [name for name, section in FIELD_SLICES
                      if [float(value) for value in before[section]] != [float(value) for value in after[section]]]
            require(fields and operation["changed_fields"] == fields,
                    f"changed field inventory differs: {image_id}:{number}")
            require(all(before[section] == after[section] for name, section in FIELD_SLICES if name not in fields),
                    f"unchanged field tokens differ: {image_id}:{number}")
            donors = operation["donor_endpoints"]
            require(isinstance(donors, list), f"donor_endpoints must be a list: {image_id}:{number}")
            recipients = set()
            for donor in donors:
                require(isinstance(donor, dict), f"invalid donor endpoint: {image_id}:{number}")
                original, endpoint, target = (donor["donor_original_row"], donor["donor_endpoint"],
                                               donor["candidate_endpoint"])
                require(type(original) is int and original in removed
                        and type(endpoint) is int and endpoint in (0, 1)
                        and type(target) is int and target in (0, 1),
                        f"donor must name a deleted original row and valid endpoint indices: {image_id}:{number}")
                require((original, endpoint) not in donated and target not in recipients,
                        f"donor or recipient endpoint repeated: {image_id}:{number}")
                donor_tokens = texts[original - 1].split()[5 + 3 * endpoint:8 + 3 * endpoint]
                require(donor["source_row_text"] == texts[original - 1] and nonempty(donor.get("reason"))
                        and float(donor_tokens[2]) == 2
                        and float(before[7 + 3 * target]) == 0
                        and donor_tokens == after[5 + 3 * target:8 + 3 * target]
                        and ("left_endpoint" if target == 0 else "right_endpoint") in fields,
                        f"donor endpoint provenance or exact tokens differ: {image_id}:{number}")
                donated.add((original, endpoint))
                recipients.add(target)
            ending = lines[number - 1][len(texts[number - 1].encode("utf-8")):]
            require(ending in (b"", b"\n", b"\r\n"), f"unsupported source line ending: {image_id}:{number}")
            expected_lines[number - 1] = operation["candidate_row_text"].encode("utf-8") + ending
        deleted_visible = {(number, side) for number in removed
                           for side, point in enumerate(gold[image_id][number - 1]["keypoints"])
                           if point[2] == 2}
        require(donated == deleted_visible,
                f"every deleted visible endpoint must have exactly one declared donor transfer: {image_id}")
        expected_lines = [line for number, line in enumerate(expected_lines, 1) if number not in removed]
        require(not change["additions"] or not expected_lines or expected_lines[-1].endswith(b"\n"),
                f"appended additions require newline-terminated retained rows: {image_id}")
        addition_ids = set()
        for index, operation in enumerate(change["additions"], retained_count + 1):
            check_review(operation, image_id, decisions)
            identifier = operation["addition_id"]
            require(nonempty(identifier) and identifier not in addition_ids, f"invalid/repeated addition_id: {image_id}")
            addition_ids.add(identifier)
            require(type(operation["candidate_row"]) is int and operation["candidate_row"] == index,
                    f"addition candidate_row must append in declared order: {image_id}")
            row_text(operation["candidate_row_text"], record["original_size"], f"{image_id}:addition:{identifier}")
            expected_lines.append(operation["candidate_row_text"].encode("utf-8") + b"\n")
        actual_bytes = path.read_bytes()
        require(actual_bytes == b"".join(expected_lines),
                f"candidate differs from exact declared edits; unlisted rows/bytes changed: {image_id}")
        require(actual_bytes != source_bytes[image_id], f"annotation operations have no net byte change: {image_id}")
        objects = parse_pose_labels(actual_bytes.decode("utf-8"), *record["original_size"], str(path))
        check_geometry(objects, *record["original_size"], str(path))
        require(type(change["candidate_objects"]) is int and change["candidate_objects"] == len(objects)
                == len(lines) - len(removed) + len(addition_ids), f"candidate_objects mismatch: {image_id}")
        effective[image_id] = objects
        removed_total += len(removed)
        replaced_total += len(replaced)
        added_total += len(addition_ids)
        merged_total += len(donated)

    label_dir = candidate.parent / "labels"
    require(not label_dir.is_symlink(), "overlay labels directory must not be a symlink")
    actual_paths = set()
    for path in label_dir.rglob("*"):
        require(not path.is_symlink(), "overlay labels must not contain symlinks")
        require(path.is_file() or path.is_dir(), "overlay labels must contain only regular files/directories")
        if path.is_file():
            actual_paths.add(path.relative_to(candidate.parent).as_posix())
    require(actual_paths == listed_paths, "unexpected or missing label files in annotation candidate overlay")
    expected = proposal["expected_counts"]
    require(isinstance(expected, dict) and set(expected) == {"train", "val"}, "invalid expected splits")
    for split in ("train", "val"):
        require(isinstance(expected[split], dict) and set(expected[split]) == set(COUNT_KEYS)
                and all(type(value) is int and value >= 0 for value in expected[split].values()),
                "invalid expected counts")
    actual_counts = totals(records, effective)
    require(actual_counts == expected, "candidate totals differ from expected_counts")
    require(actual_counts["val"] == source_counts["val"], "validation totals changed")
    scripts = Path(__file__).resolve().parent
    return {"status": "verified", "candidate_id": proposal["candidate_id"], "kind": ANNOTATION_KIND,
            "candidate_manifest_sha256": file_sha256(candidate), "declared_source_revision": proposal["source_revision"],
            "source_revision_git_object_checked": False, "source_manifest_sha256": reference_hash,
            "review_sha256": review["sha256"], "annotation_policy_sha256": policy["sha256"],
            "reference_labels_verified": len(records), "changed_train_labels": len(changed),
            "removed_objects": removed_total, "replaced_objects": replaced_total,
            "added_objects": added_total, "merged_endpoints": merged_total,
            "counts": actual_counts, "validation_labels_unchanged": True,
            "unchanged_rows_byte_preserved": True, "full_row_maps_verified": True,
            "image_bytes_verified": False, "images_staged_by_check": False, "dataset_built": False,
            "visual_correctness_verified": False, "mechanical_geometry_verified": True, "code_sha256": {
                f"scripts/{name}": file_sha256(scripts / name) for name in
                ("verify_pose_annotation_candidate.py", "verify_pose_candidate.py", "evaluate_pose.py", "pose_scoring.py")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", type=Path, default=Path(ANNOTATION_CANDIDATE),
                        help="manifest path; relative paths resolve from --root")
    args = parser.parse_args()
    candidate = args.candidate if args.candidate.is_absolute() else args.root / args.candidate
    try:
        result = verify_annotation_candidate(args.root, candidate)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, allow_nan=False))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
