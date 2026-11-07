#!/usr/bin/env python3
"""Check that box proposals cannot silently change points, rows or validation."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_pose import file_sha256
from verify_pose_box_candidate import verify_box_candidate
from verify_pose_candidate import verify_candidate


SOURCE_ROW = "0 0.45 0.45 0.70 0.70 0.20 0.60 2 0.70 0.60 2"
BOX_ROW = "0 0.45 0.40 0.70 0.60 0.20 0.60 2 0.70 0.60 2"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class BoxCandidateChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        records = []
        source_hash = hashlib.sha256((SOURCE_ROW + "\n" + SOURCE_ROW + "\n").encode()).hexdigest()
        image_hash = "1" * 64
        for split, size in (("train", 137), ("val", 20)):
            for index in range(size):
                name = f"{split}_{index:03d}"
                label = f"labels/{split}/{name}.txt"
                path = self.root / "data" / label
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(SOURCE_ROW + "\n" + SOURCE_ROW + "\n")
                records.append({"id": name, "split": split, "label": label,
                    "image": f"images/{split}/{name}.png", "original_size": [100, 100],
                    "instances": 2, "visible_endpoints": 4, "missing_endpoints": 0,
                    "label_sha256": source_hash, "image_sha256": image_hash})
        source = {"records": records, "split_summary": {
            split: {"images": size, "instances": 2 * size, "visible_endpoints": 4 * size,
                    "missing_endpoints": 0, "empty_images": 0} for split, size in (("train", 137), ("val", 20))}}
        write_json(self.root / "data/manifest.json", source)
        policy = self.root / "docs/annotation-policy.md"
        policy.parent.mkdir()
        policy.write_text("Synthetic box-policy identity for mechanical checks.\n")
        self.folder = self.root / "data/candidates/fixture"
        self.label = self.folder / "labels/train/train_000.txt"
        self.label.parent.mkdir(parents=True)
        self.label.write_text(BOX_ROW + "\n" + SOURCE_ROW + "\n")
        review = self.folder / "review.json"
        write_json(review, {"synthetic": True, "not_a_physical_review": True})
        self.manifest = self.folder / "manifest.json"
        self.proposal = {"schema_version": 1, "kind": "training_box_overlay", "status": "proposed",
            "candidate_id": "synthetic-box-fixture", "source_revision": "2" * 40,
            "source_manifest_sha256": file_sha256(self.root / "data/manifest.json"),
            "annotation_policy": {"id": "baomai-visible-front-v1", "path": "docs/annotation-policy.md",
                                  "sha256": file_sha256(policy)},
            "review": {"path": "review.json", "sha256": file_sha256(review)},
            "changes": [{"image_id": "train_000", "source_label": "labels/train/train_000.txt",
                "candidate_label": "labels/train/train_000.txt", "source_label_sha256": source_hash,
                "source_image_sha256": image_hash, "candidate_label_sha256": file_sha256(self.label),
                "source_objects": 2, "candidate_objects": 2,
                "rows": [{"original_row": 1, "source_row_text": SOURCE_ROW, "candidate_row_text": BOX_ROW,
                          "box_edges_changed": ["bottom"], "candidate_bbox_xyxy_px": [10, 10, 80, 70],
                          "review_status": "accepted_incremental", "reason": "Synthetic guard fixture."}]}],
            "expected_counts": {split: {"images": size, "objects": 2 * size, "visible_endpoints": 4 * size,
                "missing_endpoints": 0, "empty_images": 0} for split, size in (("train", 137), ("val", 20))}}
        self.save()

    def save(self):
        write_json(self.manifest, self.proposal)

    def refresh_candidate(self, text, *, first_row=True):
        self.label.write_text(text)
        self.proposal["changes"][0]["candidate_label_sha256"] = file_sha256(self.label)
        if first_row:
            self.proposal["changes"][0]["rows"][0]["candidate_row_text"] = text.splitlines()[0]
        self.save()

    def test_accepts_an_explicit_box_edge_only(self):
        result = verify_box_candidate(self.root, self.manifest)
        self.assertEqual(result["changed_box_rows"], 1)
        self.assertEqual(result["removed_objects"], 0)
        self.assertTrue(result["class_and_endpoint_tokens_unchanged"])
        self.assertFalse(result["visual_correctness_verified"])

    def test_original_deletion_verifier_rejects_box_contract(self):
        with self.assertRaisesRegex(ValueError, "training_label_overlay"):
            verify_candidate(self.root, self.manifest)

    def test_refreshed_hashes_cannot_hide_endpoint_changes(self):
        changed = BOX_ROW.replace("0.20 0.60", "0.21 0.60")
        self.refresh_candidate(changed + "\n" + SOURCE_ROW + "\n")
        with self.assertRaisesRegex(ValueError, "endpoint tokens"):
            verify_box_candidate(self.root, self.manifest)

    def test_unlisted_row_changes_are_rejected(self):
        self.refresh_candidate(BOX_ROW + "\n" + BOX_ROW + "\n")
        with self.assertRaisesRegex(ValueError, "unlisted rows"):
            verify_box_candidate(self.root, self.manifest)

    def test_row_removal_is_rejected(self):
        self.refresh_candidate(BOX_ROW + "\n")
        with self.assertRaisesRegex(ValueError, "add or remove"):
            verify_box_candidate(self.root, self.manifest)

    def test_validation_changes_are_rejected(self):
        change = self.proposal["changes"][0]
        change.update(image_id="val_000", source_label="labels/val/val_000.txt")
        self.save()
        with self.assertRaisesRegex(ValueError, "validation changes"):
            verify_box_candidate(self.root, self.manifest)

    def test_wrong_edge_inventory_or_unaccepted_edit_is_rejected(self):
        row = self.proposal["changes"][0]["rows"][0]
        for field, value, message in (("box_edges_changed", ["top"], "edge inventory"),
                                      ("review_status", "unresolved", "accepted incremental")):
            old = copy.deepcopy(row[field])
            row[field] = value
            self.save()
            with self.assertRaisesRegex(ValueError, message):
                verify_box_candidate(self.root, self.manifest)
            row[field] = old

    def test_box_cannot_exclude_an_existing_visible_corner(self):
        invalid = BOX_ROW.replace("0.40 0.70 0.60", "0.25 0.70 0.30")
        self.refresh_candidate(invalid + "\n" + SOURCE_ROW + "\n")
        with self.assertRaisesRegex(ValueError, "endpoint outside its box"):
            verify_box_candidate(self.root, self.manifest)

    def test_review_evidence_and_extra_labels_are_bound(self):
        (self.folder / "review.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "review evidence"):
            verify_box_candidate(self.root, self.manifest)
        self.proposal["review"]["sha256"] = file_sha256(self.folder / "review.json")
        self.save()
        (self.label.parent / "unexpected.txt").write_text(SOURCE_ROW + "\n")
        with self.assertRaisesRegex(ValueError, "unexpected or missing"):
            verify_box_candidate(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
