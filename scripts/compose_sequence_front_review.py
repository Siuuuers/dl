#!/usr/bin/env python3
"""Compose the dated video-sequence review without changing training inputs."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from evaluate_pose import parse_pose_labels
from verify_pose_candidate import check_geometry, read_reference, require, totals


KIND = "review_sequence_front_refinement"
CANDIDATE_ID = "sequence-refined-fronts-2026-11-17"
OUTPUT = "data/candidates/" + CANDIDATE_ID
REVIEW = "results/review/sequence-refinement-20261117/accepted-review.json"
STATUS = "results/review/sequence-refinement-20261117/review-status.json"
PARENT = {
    "path": "data/candidates/complete-visible-fronts-2026-11-14/manifest.json",
    "sha256": "9d138930b523fc8d124e116a58aa584a23c30a81d874551fcb5625a1978639a8",
}
PARENT_STATUS = {
    "path": "results/review/complete-labels-20261114/review-status.json",
    "sha256": "257d18c2cbe69471fc5a0070748747a92534cda88d5065b14520b98ea37cc761",
}
ADJUDICATION = {
    "path": "results/review/physical-adjudication-20261114/decisions.json",
    "sha256": "07f545434877c4d9d44eb90b8e1c0ae4aa1c13882a500ccd81bfd2b54f9dbbc6",
}
DEVELOPMENT = {
    "path": "data/references/visible-front-completion-2026-11-14/manifest.json",
    "sha256": "bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1",
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path.read_bytes())}


def bound(root, item):
    path = (root / item["path"]).resolve()
    require(path.is_relative_to(root), "source path escapes repository")
    require(digest(path.read_bytes()) == item["sha256"], "source hash differs: " + item["path"])
    return path


def one_line(value, where):
    require(isinstance(value, str) and value.strip() == value
            and value.splitlines() == [value], "one nonblank row required: " + where)
    fields = value.split()
    require(len(fields) == 11, "eleven fields required: " + where)
    return fields


def check_corner_change(op, before, after):
    for side, offset in (("left", 5), ("right", 8)):
        old = before[offset:offset + 3] if before else ["0", "0", "0"]
        new = after[offset:offset + 3] if after else ["0", "0", "0"]
        if before and after and old == new:
            continue
        if all(float(value) == 0 for value in old + new):
            continue
        evidence = op.get("corner_evidence", {}).get(side, {})
        expected = ("actual_corner_visible_in_target_frame" if after and float(new[2]) == 2
                    else "non_target_feature" if after is None
                    else "unlocatable_in_target_frame")
        require(evidence.get("state") == expected
                and isinstance(evidence.get("reason"), str) and evidence["reason"].strip(),
                f"explicit {side} corner evidence required for {op['image_id']}:{op.get('base_row')}")


def compose_status(root, review, review_path, files):
    inherited = json.loads(bound(root, PARENT_STATUS).read_text())
    cases = deepcopy(inherited["cases"])
    by_id = {case["case_id"]: case for case in cases}
    adjudication = json.loads(bound(root, ADJUDICATION).read_text())
    for decision in adjudication["decisions"]:
        case = by_id[decision["case_id"]]
        case["status"] = "resolved"
        case["atlas_note"] = decision["explanation"]
        case["latest_review_path"] = ADJUDICATION["path"]
        for field in ("remaining_questions", "remaining_details"):
            if field in case:
                case["pre_sequence_" + field] = case[field]
                case[field] = []
    seen = set()
    for decision in review["case_decisions"]:
        cid = decision["case_id"]
        require(cid in by_id and cid not in seen, "unknown or repeated case decision: " + cid)
        require(decision["status"] in ("resolved", "pending"), "invalid case status")
        require(isinstance(decision.get("reason"), str) and decision["reason"].strip(),
                "case decision needs a reason")
        case = by_id[cid]
        case.update(status=decision["status"], previous_status=case["status"],
                    atlas_note=decision.get("atlas_note") or decision["reason"],
                    sequence_decision_reason=decision["reason"],
                    latest_review_path=review_path.relative_to(root).as_posix())
        for field in ("remaining_questions", "remaining_details"):
            if field in case:
                case["pre_sequence_" + field] = case[field]
        questions = ([] if decision["status"] == "resolved" else
                     decision.get("remaining_questions", [decision["reason"]]))
        case["remaining_questions"] = questions
        case["remaining_details"] = ([{"reason": decision["reason"], "questions": questions}]
                                     if questions else [])
        for field in ("held_original_rows", "held_addition_ids"):
            if field in decision:
                require(isinstance(decision[field], list), "held identities must be a list")
                require(set(decision[field]) <= set(case.get(field, [])),
                        "a case refinement may only narrow its inherited held identities")
                case["previous_" + field] = case.get(field, [])
                case[field] = decision[field]
        seen.add(cid)
    by_image = {item["image_id"]: item for item in files}
    for case in cases:
        item = by_image.get(case["image_id"])
        require(item is not None, "case is outside training inventory")
        row_map = item["row_map"]
        original = {m["original_row"]: m["candidate_row"] for m in row_map
                    if m["original_row"] is not None}
        additions = {m["addition_id"]: m["candidate_row"] for m in row_map
                     if m["original_row"] is None}
        held = [original.get(row) for row in case.get("held_original_rows", [])]
        held += [additions.get(name) for name in case.get("held_addition_ids", [])]
        if case["status"] != "resolved":
            require(all(row is not None for row in held),
                    "a pending case targets a deleted or absent row: " + case["case_id"])
        case["held_current_rows"] = sorted({row for row in held if row is not None})
    focuses = deepcopy(review["focus_pages"])
    require(isinstance(focuses, list), "focus_pages must be a list")
    for focus in focuses:
        item = by_image[focus["image_id"]]
        if "base_rows" in focus:
            mapping = {m["base_row"]: m["candidate_row"] for m in item["row_map"]
                       if m.get("base_row") is not None}
            focus["current_rows"] = [mapping[row] for row in focus.pop("base_rows")
                                     if mapping[row] is not None]
        selected = focus.get("current_rows")
        require(selected is None or set(selected) <= set(range(1, item["candidate_objects"] + 1)),
                "focus selects absent output row")
    pending = [case for case in cases if case["status"] != "resolved"]
    return {
        "schema_version": 1, "kind": "sequence_front_review_status",
        "review_revision": CANDIDATE_ID, "candidate_id": CANDIDATE_ID,
        "parent_status": PARENT_STATUS, "source_candidate": PARENT,
        "inherited_physical_adjudication": ADJUDICATION,
        "accepted_review_path": review_path.relative_to(root).as_posix(),
        "release_ready": False, "training_recipe_id": None,
        "full_dataset_certified": False, "case_counts_are_not_accuracy": True,
        "user_confirmations": inherited.get("user_confirmations", []) + review.get("user_confirmations", []),
        "cases": cases, "focus_pages": focuses,
        "remaining_all_case_groups": len(pending),
        "remaining_all_training_images": len({case["image_id"] for case in pending}),
        "all_case_group_counts": dict(Counter(case["status"] for case in cases)),
        "change_scope": "Supported sequence and visible-front extent corrections; PDF review before merge."
    }


def compose(root, review_path):
    parent_path = bound(root, PARENT)
    parent = json.loads(parent_path.read_text())
    bound(root, parent["annotation_policy"])
    bound(root, DEVELOPMENT)
    source_hash, records, by_id, original_bytes, original_objects, original_counts = read_reference(
        root, parent["source_manifest_sha256"])
    review = json.loads(review_path.read_text())
    for item in review.get("review_sources", []):
        bound(root, item)
    require(isinstance(review["operations"], list) and review["operations"], "no accepted operations")
    require(isinstance(review["case_decisions"], list), "case_decisions must be a list")
    for decision in review["case_decisions"]:
        require(isinstance(decision.get("evidence_image_ids"), list)
                and decision["evidence_image_ids"]
                and all(iid in by_id for iid in decision["evidence_image_ids"]),
                "case decision must identify available evidence frames")
    grouped, seen = defaultdict(list), set()
    for op in review["operations"]:
        iid, action = op["image_id"], op["operation"]
        require(iid in by_id and by_id[iid]["split"] == "train", "operation is not a training photo")
        require(action in ("replace", "delete", "add"), "unsupported operation")
        require(isinstance(op.get("reason"), str) and op["reason"].strip(), "operation needs physical evidence")
        require(isinstance(op["evidence_image_ids"], list) and op["evidence_image_ids"]
                and iid in op["evidence_image_ids"]
                and all(eid in by_id for eid in op["evidence_image_ids"]),
                "evidence must identify the target frame and available neighbors")
        row = op.get("base_row")
        if action == "add":
            require(row is None and op.get("before_line") is None
                    and isinstance(op.get("addition_id"), str) and op["addition_id"].strip(),
                    "addition needs a new identity and no base row")
            key = (iid, "addition", op["addition_id"])
        else:
            require(type(row) is int and row > 0, "base_row must index the full reviewed candidate")
            key = (iid, "row", row)
        require(key not in seen, "multiple operations on one base object; combine them")
        seen.add(key)
        grouped[iid].append(op)
    parent_files = {item["image_id"]: item for item in parent["changes"]}
    files, payloads, operation_log, effective, base_objects = [], {}, [], dict(original_objects), dict(original_objects)
    for record in sorted(records, key=lambda item: item["id"]):
        iid, size = record["id"], record["original_size"]
        image_path = root / "data" / record["image"]
        require(digest(image_path.read_bytes()) == record["image_sha256"], "original pixels differ: " + iid)
        if record["split"] != "train":
            continue
        prior = parent_files.get(iid)
        base_path = (parent_path.parent / prior["candidate_label"] if prior
                     else root / "data" / record["label"])
        before = base_path.read_bytes()
        require(digest(before) == (prior["candidate_label_sha256"] if prior else record["label_sha256"]),
                "full-reviewed baseline differs: " + iid)
        lines = before.decode().splitlines(keepends=True)
        require(all(line.strip() for line in lines), "blank physical rows are not supported")
        mapping = deepcopy(prior["row_map"] if prior else
                           [{"original_row": i, "candidate_row": i} for i in range(1, len(lines) + 1)])
        base_objects[iid] = parse_pose_labels(before.decode(), *size, iid + " baseline")
        identities = {m.get("addition_id") for m in mapping if m["original_row"] is None}
        removed, replacements, additions = set(), {}, []
        for op in grouped[iid]:
            row, action = op.get("base_row"), op["operation"]
            before_fields = None
            if action != "add":
                require(row <= len(lines) and lines[row - 1].rstrip("\r\n") == op["before_line"],
                        f"exact full-reviewed base row differs: {iid}:{row}")
                before_fields = one_line(op["before_line"], iid)
            else:
                require(op["addition_id"] not in identities, "addition reuses a stable identity")
                identities.add(op["addition_id"])
            after_fields = None
            if action != "delete":
                after_fields = one_line(op["after_line"], iid)
                parsed = parse_pose_labels(op["after_line"], *size, iid + " operation")
                check_geometry(parsed, *size, iid + " operation")
                require(len(parsed) == 1, "operation must produce one row")
                if before_fields:
                    require(op["before_line"] != op["after_line"], "ineffective replacement")
            else:
                require(op.get("after_line") is None, "deletion cannot include a replacement")
            check_corner_change(op, before_fields, after_fields)
            if action == "delete":
                removed.add(row)
            elif action == "replace":
                replacements[row] = op["after_line"] + "\n"
            else:
                additions.append(op)
        result = "".join(replacements.get(i, line) for i, line in enumerate(lines, 1) if i not in removed)
        if additions:
            require(not result or result.endswith("\n"), "base file needs a terminating newline before additions")
            result += "".join(op["after_line"] + "\n" for op in additions)
        result = result.encode()
        updated_map = []
        for entry in mapping:
            old = entry["candidate_row"]
            entry["base_row"] = old
            entry["candidate_row"] = (None if old is None or old in removed
                                      else old - sum(number < old for number in removed))
            updated_map.append(entry)
        for row, op in enumerate(additions, len(lines) - len(removed) + 1):
            updated_map.append({"original_row": None, "addition_id": op["addition_id"],
                                "parent_row": None, "base_row": None, "candidate_row": row})
        parsed = parse_pose_labels(result.decode(), *size, iid)
        check_geometry(parsed, *size, iid)
        effective[iid] = parsed
        require(sorted(m["candidate_row"] for m in updated_map if m["candidate_row"] is not None)
                == list(range(1, len(parsed) + 1)), "row map is not complete: " + iid)
        if not grouped[iid]:
            require(result == before, "unreviewed file changed")
        output_rows = {m["base_row"]: m["candidate_row"] for m in updated_map if m["base_row"] is not None}
        addition_rows = {m.get("addition_id"): m["candidate_row"] for m in updated_map
                         if m["original_row"] is None}
        for op in grouped[iid]:
            operation_log.append({**op, "output_row": (addition_rows[op["addition_id"]]
                                                       if op["operation"] == "add" else output_rows[op["base_row"]]),
                                  "source_image_sha256": record["image_sha256"],
                                  "base_label_sha256": digest(before)})
        payloads[record["label"]] = result
        files.append({"image_id": iid, "split": "train", "source_label": record["label"],
                      "candidate_label": record["label"], "source_label_sha256": record["label_sha256"],
                      "base_label": descriptor(root, base_path), "candidate_label_sha256": digest(result),
                      "source_image_sha256": record["image_sha256"], "source_objects": len(original_objects[iid]),
                      "base_objects": len(lines), "candidate_objects": len(parsed),
                      "changed_from_full_review": result != before, "row_map": updated_map})
    baseline_counts, result_counts = totals(records, base_objects), totals(records, effective)
    require(baseline_counts == parent["expected_counts"], "full-reviewed baseline counts differ")
    require(len(files) == 137 and result_counts["train"]["empty_images"] == 13,
            "the complete 137-photo / 13-empty training inventory must remain")
    require(result_counts["val"] == original_counts["val"], "original training validation changed")
    if "expected_counts" in review:
        require(result_counts == review["expected_counts"], "approved expected counts differ")
    status = compose_status(root, review, review_path, files)
    status_bytes = encoded(status)
    manifest = {
        "schema_version": 1, "kind": KIND, "candidate_id": CANDIDATE_ID,
        "status": "awaiting_pdf_review", "review_date": "2026-11-17",
        "source_manifest_sha256": source_hash, "parent": PARENT,
        "annotation_policy": parent["annotation_policy"], "review": descriptor(root, review_path),
        "review_sources": review.get("review_sources", []),
        "review_status": {"path": STATUS, "sha256": digest(status_bytes)},
        "release_ready": False, "training_recipe_id": None, "full_dataset_certified": False,
        "pdf_review_before_merge": True, "expected_counts": result_counts,
        "baseline_counts": baseline_counts,
        "development_reference": DEVELOPMENT,
        "development_reference_is_separate": True,
        "operation_counts": dict(Counter(op["operation"] for op in operation_log)),
        "changed_images_from_full_review": [item["image_id"] for item in files if item["changed_from_full_review"]],
        "remaining_case_groups": status["remaining_all_case_groups"],
        "remaining_training_images": status["remaining_all_training_images"],
        "files": files, "operations": operation_log,
        "evidence_images": [{"image_id": iid, "path": "data/" + by_id[iid]["image"],
                             "sha256": by_id[iid]["image_sha256"]}
                            for iid in sorted({eid for op in operation_log for eid in op["evidence_image_ids"]}
                                              | {eid for decision in review["case_decisions"]
                                                 for eid in decision["evidence_image_ids"]})],
        "limits": ["This is a complete training-label review, not a training-authorized candidate.",
                   "Validation/development references remain separate and unchanged.",
                   "Unchanged labels and unresolved cases are inherited, not newly certified.",
                   "Video continuity supports identity; it does not make hidden target-frame corners visible.",
                   "PR35 remains selected. No model training, inference or promotion is performed."]
    }
    payloads["manifest.json"] = encoded(manifest)
    return payloads, status_bytes, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--review", type=Path, default=Path(REVIEW))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    review_path = (args.review if args.review.is_absolute() else root / args.review).resolve()
    require(review_path.is_relative_to(root), "review must be in the repository")
    payloads, status_bytes, manifest = compose(root, review_path)
    output, status_path = root / OUTPUT, root / STATUS
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} if output.exists() else set()
    require(actual <= set(payloads) | {"README.md"}, "unexpected files in the dated preview directory")
    if not args.verify_only:
        output.mkdir(parents=True, exist_ok=True)
        for relative, payload in payloads.items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_bytes(status_bytes)
    require(status_path.read_bytes() == status_bytes, "saved status differs from composition")
    for relative, payload in payloads.items():
        require((output / relative).read_bytes() == payload, "saved preview differs: " + relative)
    print(json.dumps({"status": "verified_review_only", "candidate_id": CANDIDATE_ID,
                      "candidate_sha256": digest(payloads["manifest.json"]),
                      "complete_training_files": len(manifest["files"]),
                      "changed_images_from_full_review": manifest["changed_images_from_full_review"],
                      "operation_counts": manifest["operation_counts"], "counts": manifest["expected_counts"],
                      "remaining_case_groups": manifest["remaining_case_groups"],
                      "remaining_training_images": manifest["remaining_training_images"],
                      "release_ready": False, "training_recipe_id": None}, indent=2))


if __name__ == "__main__":
    main()
