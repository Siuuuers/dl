#!/usr/bin/env python3
"""Verify the requested r14 amendments, immutable originals and changed-only nonempty new-photo PDF scope."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from build_front_review_r14 import CANDIDATE_ID, LEDGER, PARENT, PROTOCOL_PATH, REVIEW_PATH, SOURCE_REVISION, experiment, read_user_approval
from verify_pose_candidate import require
from verify_reconciled_pose_labels import bound

CANDIDATE = f"data/candidates/{CANDIDATE_ID}/manifest.json"


def _verify_r14_scope(root, review=None, records=None):
    root = Path(root).resolve()
    review = review or json.loads((root / REVIEW_PATH).read_text())
    records = records or json.loads((root / "data/manifest.json").read_text())["records"]
    new_ids = sorted(r["id"] for r in records if r["id"].startswith("new_"))
    old_ids = sorted(r["id"] for r in records if r["id"].startswith("old_"))
    original_image_hashes = {r["id"]: r["image_sha256"] for r in records}
    require(len(new_ids) == 126 and len(old_ids) == 31 and len(records) == 157, "Original inventory changed")
    requests = json.loads((root / LEDGER / "user-reference-index.json").read_text())
    for reference in requests["references"]:
        bound(root, {"path": reference["path"], "sha256": reference["sha256"]})
    bound(root, requests["text_feedback"])
    requested = requests["requested_image_ids"]
    require(len(requested) == len(set(requested)) == 33 and set(requested) <= set(new_ids), "Requested photo inventory differs")
    require(review["requested_image_ids"] == requested, "Review scope differs from the user request ledger")
    approval, approval_source = read_user_approval(root)
    require(review["user_approval"] == requests["approval"] == approval_source,
            "The user approval ledger is not bound by the review and request index")
    status = json.loads(bound(root, review["review_status"]).read_text())
    cases = {case["case_id"]: case for case in status["cases"]}
    for decision in approval["decisions"]:
        case = cases[decision["case_id"]]
        require(case["status"] == "resolved" and case["user_confirmation"] == approval_source
                and case["reason"] == decision["reason"], "Displayed amber approval was not applied exactly")
    require(cases["r6-a-04-new_000015"]["status"] != "resolved"
            and status["pending_train_photos"] == ["new_000015"]
            and status["unresolved_development_photos"] == [],
            "The unboxed photo015 question must remain the sole unresolved case in this pass")
    operations = json.loads(bound(root, review["operations"]).read_text())
    changed = sorted({o["image_id"] for o in operations})
    require(changed == review["changed_image_ids"], "Changed-photo inventory differs")
    pdf_ids = sorted(f["image_id"] for f in review["files"] if f["image_id"] in changed and f["image_id"].startswith("new_") and f["objects"] > 0)
    require(review["pdf_image_ids"] == pdf_ids, "PDF must include exactly changed nonempty new photographs and no unchanged, old or empty photographs")
    coverage = json.loads(bound(root, review["request_coverage"]).read_text())
    require(coverage["base_review"] == PARENT and coverage["source_revision"] == SOURCE_REVISION
            and coverage["reviewed_image_ids"] == requested and coverage["visual_review_complete_for_requested_photos"] is True,
            "Visual audit provenance differs")
    require(coverage["user_approval"] == approval_source,
            "User approval must remain separate from fresh visual review coverage")
    require(len(coverage["images"]) == len(requested)
            and {r["image_id"] for r in coverage["images"]} == set(requested), "Each requested photo needs one review record")
    require(review["excluded_training_image_ids"] == coverage["excluded_old_image_ids"] == old_ids,
            "Accepted old photos must remain separately excluded from treatment")
    parent = bound(root, PARENT).parent
    inherited = []
    for record in records:
        iid = record["id"]
        before = (parent / record["label"]).read_bytes()
        after = ((root / REVIEW_PATH).parent / record["label"]).read_bytes()
        require((after != before) == (iid in changed), f"Changed-photo inventory is not an exact byte comparison: {iid}")
        if iid not in requested:
            require(after == before, f"Unrequested label changed: {iid}")
            inherited.append(iid)
    dispositions, secondary_count = [], 0
    for descriptor in review["review_evidence"]:
        audit = json.loads(bound(root, descriptor).read_text())
        require(audit["source_review"] == "fronts-2026-11-21-r13", "Audit uses the wrong row baseline")
        dispositions.extend(audit.get("request_dispositions", audit.get("request_coverage", [])))
    require(len(dispositions) == len(coverage["request_dispositions"]), "Request dispositions were omitted")
    for descriptor in review["secondary_review_evidence"]:
        secondary = json.loads(bound(root, descriptor).read_text())
        require(secondary["kind"] in ("bounded-secondary-physical-adjudication", "bounded-secondary-physical-corroboration")
                and secondary["source_review"] == "fronts-2026-11-21-r13", "Secondary source differs")
        reviewed = json.loads(bound(root, secondary["reviewed_audit"]).read_text())
        proposed = {(o["image_id"], o.get("base_row"), o.get("addition_id")): o
                    for o in reviewed["proposed_operations"]}
        if secondary["kind"] == "bounded-secondary-physical-corroboration":
            decisions = []
            for group in secondary["decisions"]:
                require(group["image_id"] in reviewed["reviewed_image_ids"] and group["reason"],
                        "Secondary corroboration outside audited scope")
                for item in group["evaluated_operations"]:
                    require(item["base_row"] in group["base_rows"], "Corroborated row outside declared scope")
                    decisions.append({"image_id": group["image_id"], "base_row": item["base_row"],
                        "evaluated_before_line": item["before_line"], "evaluated_after_line": item["after_line"],
                        "source_label_sha256": item["source_label_sha256"], "source_image_sha256": item["source_image_sha256"]})
        else:
            decisions = secondary["decisions"]
        for decision in decisions:
            key = (decision["image_id"], decision.get("base_row"), decision.get("addition_id"))
            require(key in proposed, "Secondary decision has no matching proposal")
            operation = proposed[key]
            require(decision.get("evaluated_before_line") == operation.get("before_line")
                    and decision["evaluated_after_line"] == operation["after_line"]
                    and decision["source_label_sha256"] == operation["source_label_sha256"]
                    and decision["source_image_sha256"] == original_image_hashes[operation["image_id"]]
                    and operation.get("source_image_sha256", decision["source_image_sha256"]) == decision["source_image_sha256"],
                    "Proposal changed after secondary adjudication")
        if secondary["kind"] == "bounded-secondary-physical-adjudication":
            require(secondary["summary"]["operations_checked"] == len(decisions), "Secondary count differs")
        secondary_count += len(decisions)
    return {"requested_photos_visually_reviewed": len(requested), "changed_photos_in_pdf": len(changed), "nonempty_new_photos_in_pdf": len(pdf_ids),
            "unchanged_review_photos": len(records) - len(changed),
            "unrequested_review_files_byte_identical_to_r13": len(inherited),
            "confirmed_old_review_photos_byte_identical_to_r13": 31,
            "request_disposition_entries": len(dispositions),
            "approved_displayed_amber_cases": len(approval["decisions"]),
            "training_photos_released_by_user_approval": 14,
            "remaining_unboxed_question": "new_000015",
            "secondary_physical_proposals_verified": secondary_count}


def read_review(root):
    result = experiment(root).read_review(root)
    _verify_r14_scope(root, result[0], result[2])
    return result


def compose(root):
    return experiment(root).compose(root)


def protocol_document(root, manifest, source_receipt):
    return experiment(root).protocol_document(root, manifest, source_receipt)


def verify_candidate(root, candidate=None):
    configured = experiment(root)
    receipt = configured.verify_candidate(root, candidate)
    receipt.update(_verify_r14_scope(root))
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
