#!/usr/bin/env python3
"""Replay the round 10 review and its fixed PR35 comparison composition."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from build_front_review_r10 import (
    CANDIDATE_ID, PARENT, PROTOCOL_PATH, REVIEW_PATH, SOURCE_REVISION, experiment,
)
from verify_pose_candidate import require
from verify_reconciled_pose_labels import bound

CANDIDATE = f"data/candidates/{CANDIDATE_ID}/manifest.json"


def _verify_r10_scope(root, review=None, records=None):
    """Check r10 coverage and confirmed exclusions without changing older gates."""
    root = Path(root).resolve()
    if review is None:
        review = json.loads((root / REVIEW_PATH).read_text())
    if records is None:
        records = json.loads((root / "data/manifest.json").read_text())["records"]
    new_ids = sorted(row["id"] for row in records if row["id"].startswith("new_"))
    old_ids = sorted(row["id"] for row in records if row["id"].startswith("old_"))
    require(len(new_ids) == 126 and len(old_ids) == 31
            and len(set(new_ids + old_ids)) == len(records) == 157,
            "r10 source must contain exactly 126 new and 31 confirmed old photos.")
    require(review["requested_image_ids"] == review["complete_atlas_image_ids"] == new_ids,
            "r10 review and atlas must cover all 126 new photos exactly.")
    coverage = json.loads(bound(root, review["request_coverage"]).read_text())
    require(coverage["review_id"] == review["review_id"]
            and coverage["base_review"] == PARENT
            and coverage["source_revision"] == SOURCE_REVISION
            and coverage["reviewed_image_ids"] == new_ids
            and coverage["visual_review_complete_for_requested_photos"] is True,
            "r10 visual coverage identity or complete new-photo inventory differs.")
    entries = coverage["images"]
    require(isinstance(entries, list) and len(entries) == 126
            and {entry["image_id"] for entry in entries} == set(new_ids),
            "r10 coverage must contain exactly one review record per new photo.")
    require(review["excluded_training_image_ids"] == coverage["excluded_old_image_ids"] == old_ids,
            "r10 must separately exclude all 31 confirmed old photos from treatment.")
    parent_path = bound(root, PARENT)
    for row in records:
        if row["id"] in old_ids:
            current = (root / REVIEW_PATH).parent / row["label"]
            previous = parent_path.parent / row["label"]
            require(current.read_bytes() == previous.read_bytes(),
                    f"r10 changed a confirmed old review label: {row['id']}")

    secondary_count = 0
    for descriptor in review["review_evidence"]:
        audit = json.loads(bound(root, descriptor).read_text())
        secondary_descriptor = audit.get("secondary_review")
        if secondary_descriptor is None:
            continue
        secondary = json.loads(bound(root, secondary_descriptor).read_text())
        require(secondary["kind"] == "bounded-secondary-physical-adjudication"
                and secondary["source_review"] == "fronts-2026-11-19-r9-recovery",
                "r10 secondary adjudication has a different scope or source.")
        operations = {(op["image_id"], op.get("base_row")): op
                      for op in audit["proposed_operations"]}
        for decision in secondary["decisions"]:
            key = (decision["image_id"], decision["base_row"])
            require(key in operations, "Secondary adjudication refers to an absent root proposal.")
            operation = operations[key]
            require(decision["evaluated_before_line"] == operation["before_line"]
                    and decision["evaluated_after_line"] == operation["after_line"]
                    and decision["source_label_sha256"] == operation["source_label_sha256"]
                    and decision["source_image_sha256"] == operation["source_image_sha256"],
                    "A root proposal changed after its secondary physical adjudication.")
        require(secondary["summary"]["operations_checked"] == len(secondary["decisions"]),
                "Secondary adjudication count differs from its decisions.")
        secondary_count += len(secondary["decisions"])
    return {"new_photos_visually_reviewed": 126,
            "confirmed_old_review_photos_byte_identical_to_r9": 31,
            "secondary_physical_proposals_verified": secondary_count}


def read_review(root):
    result = experiment(root).read_review(root)
    _verify_r10_scope(root, result[0], result[2])
    return result


def compose(root):
    return experiment(root).compose(root)


def protocol_document(root, manifest, source_receipt):
    return experiment(root).protocol_document(root, manifest, source_receipt)


def verify_candidate(root, candidate=None):
    configured = experiment(root)
    receipt = configured.verify_candidate(root, candidate)
    receipt.update(_verify_r10_scope(root))
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
