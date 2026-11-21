#!/usr/bin/env python3
"""Verify r16 replay, immutable source rows, inherited decisions and PDF scope."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from build_front_review_r16 import (CANDIDATE_ID, LEDGER, PARENT, REVIEW_PATH,
                                  SOURCE_REVISION, experiment, normalize_audits, source_state)
from verify_pose_candidate import require
from verify_reconciled_pose_labels import bound

CANDIDATE = f"data/candidates/{CANDIDATE_ID}/manifest.json"


def _verify_r16_scope(root, review=None, records=None):
    root = Path(root).resolve()
    review = review or json.loads((root / REVIEW_PATH).read_text())
    records = records or json.loads((root / "data/manifest.json").read_text())["records"]
    new_ids = sorted(r["id"] for r in records if r["id"].startswith("new_"))
    old_ids = sorted(r["id"] for r in records if r["id"].startswith("old_"))
    require(len(new_ids) == 126 and len(old_ids) == 31 and len(records) == 157, "Original inventory changed")
    requests = json.loads((root / LEDGER / "user-reference-index.json").read_text())
    for reference in requests["references"]:
        bound(root, {"path": reference["path"], "sha256": reference["sha256"]})
    feedback = bound(root, requests["text_feedback"]).read_text()
    requested = requests["requested_image_ids"]
    require(requests["source_review"] == "fronts-2026-11-21-r15"
            and requests["source_revision"] == SOURCE_REVISION
            and len(requested) == len(set(requested)) == 17 and set(requested) <= set(new_ids),
            "Requested photo inventory/source differs")
    require(review["requested_image_ids"] == requested, "Review scope differs from saved request")
    parent_path = bound(root, PARENT)
    parent = json.loads(parent_path.read_text())
    approval_source = parent["user_approval"]
    approval = json.loads(bound(root, approval_source).read_text())
    require(review["user_approval"] == approval_source, "Prior approval descriptor was changed")
    status = json.loads(bound(root, review["review_status"]).read_text())
    inherited = json.loads(bound(root, parent["review_status"]).read_text())
    cases = {c["case_id"]: c for c in status["cases"]}
    old_cases = {c["case_id"]: c for c in inherited["cases"]}
    require(set(cases) == set(old_cases), "This bounded amendment pass must not add/drop case identities")
    for decision in approval["decisions"]:
        case = cases[decision["case_id"]]
        require(case["status"] == "resolved" and case["user_confirmation"] == approval_source
                and case["reason"] == old_cases[case["case_id"]]["reason"],
                "Inherited displayed-case approval differs")
    require(len(approval["decisions"]) == 17
            and cases["r6-a-04-new_000015"]["status"] != "resolved"
            and status["pending_train_photos"] == ["new_000015"]
            and status["unresolved_development_photos"] == [],
            "The undrawn photo015 question must remain the sole hold")
    operations = json.loads(bound(root, review["operations"]).read_text())
    # The operation ledger must still describe the inspected proposals. Replaying
    # a self-consistent but independently edited ledger is insufficient evidence.
    _, source_records, by_id, _, _, _, before = source_state(root)
    audit_paths = [bound(root, descriptor) for descriptor in review["review_evidence"]]
    expected_ops, expected_cases, expected_updates, evidence, inspected, photo_reviews = normalize_audits(
        root, audit_paths, by_id, before, inherited)
    require(operations == expected_ops and evidence == review["review_evidence"]
            and inspected == requested and source_records == sorted(records, key=lambda r: (r["split"], r["id"])),
            "Saved edits or audit scope differ from normalized physical proposals")
    row_maps = json.loads(bound(root, review["row_maps"]).read_text())
    for case in expected_cases:
        lookup = {item["base_row"]: item["current_row"] for item in row_maps[case["image_id"]]["rows"]
                  if "base_row" in item}
        case["current_rows"] = [lookup[r] for r in case["base_rows"] if r in lookup]
    require(expected_cases == status["cases"]
            and expected_updates == json.loads((root / LEDGER / "status-updates.json").read_text()),
            "Saved case decisions differ from normalized physical audits")
    changed = sorted({o["image_id"] for o in operations})
    require(changed == review["changed_image_ids"], "Changed-photo inventory differs")
    deletions = [o for o in operations if o["action"] == "delete"]
    require(len(deletions) == 1 and deletions[0]["image_id"] == "new_000066"
            and deletions[0]["base_row"] == 22 and "row 22 be deleted" in feedback
            and all(o["action"] in ("replace", "delete") for o in operations),
            "The only instance-count change must be requested deletion066:22")
    pdf_ids = sorted(f["image_id"] for f in review["files"]
                     if f["image_id"] in changed and f["image_id"].startswith("new_") and f["objects"] > 0)
    require(review["pdf_image_ids"] == pdf_ids, "PDF scope must equal changed nonempty new photographs")
    coverage = json.loads(bound(root, review["request_coverage"]).read_text())
    require(coverage["base_review"] == PARENT and coverage["source_revision"] == SOURCE_REVISION
            and coverage["reviewed_image_ids"] == requested
            and coverage["visual_review_complete_for_requested_photos"] is True
            and coverage["inherited_user_approval"] == approval_source,
            "Visual audit provenance/inherited acceptance differs")
    require(len(coverage["images"]) == len(requested)
            and {r["image_id"] for r in coverage["images"]} == set(requested),
            "Each requested photo needs one review record")
    require(coverage["images"] == photo_reviews and coverage["audits"] == evidence,
            "Saved visual coverage differs from the bound physical audits")
    require(review["excluded_training_image_ids"] == coverage["excluded_old_image_ids"] == old_ids,
            "Old photos must remain separately excluded from treatment")
    inherited_count = 0
    for record in records:
        iid = record["id"]
        before = (parent_path.parent / record["label"]).read_bytes()
        after = ((root / REVIEW_PATH).parent / record["label"]).read_bytes()
        require((after != before) == (iid in changed), f"Changed inventory differs from saved bytes: {iid}")
        if iid not in requested:
            require(after == before, f"Unrequested label changed: {iid}")
            inherited_count += 1
    dispositions = []
    for descriptor in review["review_evidence"]:
        audit = json.loads(bound(root, descriptor).read_text())
        require(audit["source_review"] == "fronts-2026-11-21-r15", "Wrong immutable audit baseline")
        dispositions.extend({**d, "audit_source": descriptor} for d in audit["request_dispositions"])
    require(dispositions == coverage["request_dispositions"]
            and {d["image_id"] for d in dispositions} == set(requested)
            and all(d.get("reason") and d.get("request") and d.get("base_rows") for d in dispositions),
            "Request dispositions missing or detached from their audit")
    secondary_count = 0
    for descriptor in review["secondary_review_evidence"]:
        secondary = json.loads(bound(root, descriptor).read_text())
        require(secondary["kind"] == "bounded-secondary-physical-corroboration"
                and secondary["source_review"] == "fronts-2026-11-21-r15"
                and secondary["reviewed_audit"] in review["review_evidence"],
                "Secondary review source/kind differs")
        audit = json.loads(bound(root, secondary["reviewed_audit"]).read_text())
        proposed = {(o["image_id"], o["base_row"]): o for o in audit["proposed_operations"]}
        checked = 0
        for decision in secondary["decisions"]:
            require(decision["reason"] and decision["image_id"] in audit["reviewed_image_ids"],
                    "Secondary review outside assigned photos")
            for item in decision["evaluated_operations"]:
                key = (decision["image_id"], item["base_row"])
                require(item["base_row"] in decision["base_rows"] and key in proposed
                        and all(item[field] == proposed[key][field] for field in
                                ("before_line", "after_line", "source_label_sha256", "source_image_sha256")),
                        "Proposal changed after secondary review")
                checked += 1
        require(checked == secondary["summary"]["operations_checked"], "Secondary count differs")
        secondary_count += checked
    return {"requested_photos_visually_reviewed": len(requested),
            "changed_photos_in_pdf": len(pdf_ids), "nonempty_new_photos_in_pdf": len(pdf_ids),
            "unchanged_review_photos": len(records) - len(changed),
            "unrequested_review_files_byte_identical_to_r15": inherited_count,
            "confirmed_old_review_photos_byte_identical_to_r15": 31,
            "request_disposition_entries": len(dispositions),
            "inherited_approved_displayed_amber_cases": 17,
            "training_photos_released_in_this_round": 0,
            "deleted_r15_rows": {"new_000066": [22]},
            "secondary_physical_proposals_verified": secondary_count,
            "remaining_unboxed_question": "new_000015"}


def read_review(root):
    result = experiment(root).read_review(root)
    _verify_r16_scope(root, result[0], result[2])
    return result


def compose(root):
    return experiment(root).compose(root)


def protocol_document(root, manifest, source_receipt):
    return experiment(root).protocol_document(root, manifest, source_receipt)


def verify_candidate(root, candidate=None):
    configured = experiment(root)
    receipt = configured.verify_candidate(root, candidate)
    receipt.update(_verify_r16_scope(root))
    receipt["excluded_training_image_ids"] = list(configured.excluded_training_image_ids)
    receipt["excluded_photo_labels_identical_between_arms"] = True
    return receipt


def build(root):
    experiment(root).build(root)
    return verify_candidate(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.root) if args.build else verify_candidate(args.root), indent=2))
