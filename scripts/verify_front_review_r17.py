#!/usr/bin/env python3
"""Verify r17 saved rows, request provenance, inherited decisions and fixed inputs."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from build_front_review_r17 import (LEDGER, PARENT, REQUESTED, REVIEW_PATH, SOURCE_REVISION,
                                  experiment, read_audits, source_state)
from verify_pose_candidate import require
from verify_reconciled_pose_labels import bound


def verify_scope(root):
    parent, records, by_id, _, _, _, before = source_state(root)
    review = json.loads((root / REVIEW_PATH).read_text())
    requests = json.loads((root / LEDGER / "user-reference-index.json").read_text())
    require(requests["requested_image_ids"] == review["requested_image_ids"] == list(REQUESTED)
            and requests["source_review"] == "fronts-2026-11-22-r16"
            and requests["source_revision"] == SOURCE_REVISION, "Requested source/scope differs")
    bound(root, requests["text_feedback"])
    for item in requests["references"]:
        bound(root, item)
    operations, evidence, photos, dispositions = read_audits(root,
        [bound(root, d) for d in review["review_evidence"]], by_id, before)
    require(operations == json.loads(bound(root, review["operations"]).read_text()), "Operations differ from their bound audits")
    coverage = json.loads(bound(root, review["request_coverage"]).read_text())
    require(coverage["base_review"] == PARENT and coverage["source_revision"] == SOURCE_REVISION
            and coverage["audits"] == evidence and coverage["images"] == photos
            and coverage["request_dispositions"] == dispositions
            and coverage["inherited_user_approval"] == parent["user_approval"], "Request coverage differs from audits")
    maps = json.loads(bound(root, review["row_maps"]).read_text())
    expected_status = copy.deepcopy(json.loads(bound(root, parent["review_status"]).read_text()))
    expected_status.update(review_id=review["review_id"], inherited_status=parent["review_status"])
    for case in expected_status["cases"]:
        case["r16_base_rows"] = list(case["current_rows"])
        case["base_rows"] = list(case["current_rows"])
        lookup = {r["base_row"]: r["current_row"] for r in maps[case["image_id"]]["rows"]}
        case["current_rows"] = [lookup[r] for r in case["base_rows"] if r in lookup]
    status = json.loads(bound(root, review["review_status"]).read_text())
    require(status == expected_status and status["pending_train_photos"] == ["new_000015"]
            and status["unresolved_development_photos"] == [], "Inherited case decisions changed")
    approval = json.loads(bound(root, parent["user_approval"]).read_text())
    require(review["user_approval"] == parent["user_approval"] and len(approval["decisions"]) == 17,
            "Inherited displayed-case acceptance changed")
    changed = sorted({o["image_id"] for o in operations})
    require(changed == review["changed_image_ids"], "Changed inventory differs")
    for rec in records:
        iid = rec["id"]
        after = ((root / REVIEW_PATH).parent / rec["label"]).read_bytes()
        require((after != before[iid]) == (iid in changed), f"Changed bytes differ: {iid}")
        if iid not in REQUESTED:
            require(after == before[iid], f"Unrequested review file changed: {iid}")
    right_deletion = next(o for o in operations if o["image_id"] == "new_000074" and o["base_row"] == 10)
    require(right_deletion["after_line"].split()[8:11] == ["0", "0", "0"], "Requested right-point deletion is missing")
    pdf_ids = sorted(f["image_id"] for f in review["files"] if f["image_id"] in changed and f["objects"])
    require(review["pdf_image_ids"] == pdf_ids, "PDF scope differs from saved changed labels")
    excluded = sorted(r["id"] for r in records if r["id"].startswith("old_"))
    require(len(excluded) == 31 and review["excluded_training_image_ids"] == excluded, "Old treatment exclusions changed")
    return {"requested_training_photos": len(REQUESTED), "changed_photos": len(changed),
            "operation_counts": review["operation_counts"], "unchanged_review_files": len(records) - len(changed),
            "unrequested_review_files_identical_to_r16": len(records) - len(REQUESTED),
            "development_review_files_identical_to_r16": sum(r["split"] == "val" for r in records),
            "deleted_r16_rows": {"new_000074": [13]}, "inherited_displayed_case_approvals": 17,
            "held_training_photo_ids": status["pending_train_photos"], "old_review_files_preserved": 31,
            "pdf_image_ids": pdf_ids}


def verify_candidate(root, candidate=None):
    root = Path(root).resolve(); configured = experiment(root)
    receipt = configured.verify_candidate(root, candidate)
    receipt.update(verify_scope(root))
    receipt["excluded_training_image_ids"] = list(configured.excluded_training_image_ids)
    receipt["excluded_photo_labels_identical_between_arms"] = True
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify_candidate(args.root), indent=2))
