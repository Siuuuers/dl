#!/usr/bin/env python3
"""Focused endpoint-overlay tamper checks using copies of the reviewed labels.

No images, inference, training or source files are changed. Negative fixtures
rehash their edited payloads so rejection tests the review/preservation contract.
"""

import sys
sys.dont_write_bytecode = True

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from evaluate_pose import file_sha256
from pose_checkpoint import ORIGINAL_RECIPE_ID, RECIPE_IDS, check_recipe_candidate_kind
from verify_pose_annotation_candidate import ANNOTATION_CANDIDATE, ANNOTATION_KIND, verify_annotation_candidate
from verify_pose_endpoint_candidate import ENDPOINT_CANDIDATE, ENDPOINT_KIND, verify_endpoint_candidate


REPOSITORY = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class EndpointCandidateChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="endpoint-candidate-check-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative in ("data/labels", str(Path(ANNOTATION_CANDIDATE).parent),
                         str(Path(ENDPOINT_CANDIDATE).parent)):
            shutil.copytree(REPOSITORY / relative, self.root / relative)
        for relative in ("data/manifest.json", "docs/annotation-policy.md"):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY / relative, target)
        self.manifest = self.root / ENDPOINT_CANDIDATE
        self.folder = self.manifest.parent
        self.proposal = json.loads(self.manifest.read_text())
        self.review_path = self.folder / self.proposal["review"]["path"]
        self.review = json.loads(self.review_path.read_text())
        self.evidence_path = self.root / self.review["evidence"]["path"]
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / self.review["evidence"]["path"], self.evidence_path)
        self.change = self.proposal["changes"][8]
        self.label = self.folder / self.change["candidate_label"]

    def save(self):
        write_json(self.review_path, self.review)
        self.proposal["review"]["sha256"] = file_sha256(self.review_path)
        write_json(self.manifest, self.proposal)

    def rehash_label(self, content):
        self.label.write_bytes(content)
        self.change["candidate_label_sha256"] = file_sha256(self.label)
        self.save()

    def sync_operation_review(self, operation):
        decision = next(row for row in self.review["decisions"] if row["id"] == operation["review_id"])
        for key, value in operation.items():
            if key not in ("review_id", "donor_endpoints"):
                decision[key] = copy.deepcopy(value)
        self.save()

    def reject(self, message):
        with self.assertRaisesRegex(ValueError, message):
            verify_endpoint_candidate(self.root, self.manifest)

    def test_real_candidate_has_exact_parent_and_nine_slot_delta(self):
        result = verify_endpoint_candidate(self.root, self.manifest)
        self.assertEqual(result["counts"]["train"], {
            "images": 137, "objects": 3771, "visible_endpoints": 6841,
            "missing_endpoints": 701, "empty_images": 13})
        self.assertEqual(result["new_endpoint_edit_counts"], self.proposal["new_endpoint_edit_counts"])
        self.assertEqual(result["new_endpoint_edit_counts"]["changed_endpoint_slots"], 9)
        self.assertTrue(result["parent_labels_and_reviews_preserved"])
        self.assertTrue(result["validation_labels_unchanged"])
        self.assertFalse(result["visual_correctness_verified"])
        self.assertFalse(result["training_recipe_supported"])

    def test_rehashed_unlisted_edit_is_rejected(self):
        lines = self.label.read_bytes().splitlines(keepends=True)
        tokens = lines[0].decode().split()
        tokens[5] = str(float(tokens[5]) + 0.001)
        lines[0] = (" ".join(tokens) + "\n").encode()
        self.rehash_label(b"".join(lines))
        self.reject("unlisted rows/bytes")

    def test_rehashed_row_deletion_and_reordering_are_rejected(self):
        original = self.label.read_bytes().splitlines(keepends=True)
        for rows in (original[1:], [original[1], original[0], *original[2:]]):
            with self.subTest(operation="delete" if len(rows) != len(original) else "reorder"):
                self.rehash_label(b"".join(rows))
                self.reject("unlisted rows/bytes")

    def test_declared_class_and_box_edits_are_rejected_even_with_matching_review(self):
        operation = self.change["replacements"][0]
        original = operation["candidate_row_text"]
        for index, value in ((0, "1"), (1, "0.45")):
            with self.subTest(field=index):
                tokens = original.split()
                tokens[index] = value
                operation["candidate_row_text"] = " ".join(tokens)
                self.sync_operation_review(operation)
                self.reject("class must be 0|preserve class and bbox")

    def test_deleted_row_and_nonidentity_map_cannot_be_declared(self):
        self.change["deletions"] = [{"original_row": 22}]
        self.save()
        self.reject("cannot add or delete")
        self.change["deletions"] = []
        self.change["row_map"][0]["candidate_row"] = True
        self.save()
        self.reject("row map must preserve")

    def test_parent_cannot_be_replaced_by_rehashing(self):
        parent = self.root / ANNOTATION_CANDIDATE
        parent.write_bytes(parent.read_bytes() + b"\n")
        self.reject("parent manifest SHA-256")
        self.proposal["parent"]["sha256"] = file_sha256(parent)
        self.save()
        self.reject("exact PR35")

    def test_inherited_records_labels_and_decisions_are_immutable(self):
        inherited = self.proposal["changes"][0]
        path = self.folder / inherited["candidate_label"]
        original_bytes = path.read_bytes()
        path.write_bytes(original_bytes + b"\n")
        self.reject("inherited label bytes")
        inherited["candidate_label_sha256"] = file_sha256(path)
        self.save()
        self.reject("inherited change records")
        inherited["candidate_label_sha256"] = file_sha256(
            self.root / Path(ANNOTATION_CANDIDATE).parent / inherited["candidate_label"])
        path.write_bytes(original_bytes)
        self.review["decisions"][0]["reason"] += " Changed."
        self.save()
        self.reject("inherited review decisions")

    def test_validation_and_source_image_identity_are_protected(self):
        image_id = self.change["image_id"]
        self.change["image_id"] = "new_000007"
        self.save()
        self.reject("validation changes")
        self.change["image_id"] = image_id
        self.change["source_image_sha256"] = "0" * 64
        self.save()
        self.reject("source identity mismatch")

    def test_slot_inventory_before_tokens_and_changed_fields_are_bound(self):
        operation = self.change["replacements"][0]
        original = copy.deepcopy(operation)
        for mutation, error in (
            (lambda op: op["endpoints"].reverse(), "slot inventory"),
            (lambda op: op["endpoints"][0].update(before=["0", "0", "0"]), "slot audit differs"),
            (lambda op: op.update(changed_fields=["left_endpoint", "right_endpoint"]), "field inventory"),
            (lambda op: op["endpoints"][0].update(accepted_original_pixels=[1, 2]), "coordinate provenance"),
        ):
            with self.subTest(error=error):
                operation.clear()
                operation.update(copy.deepcopy(original))
                mutation(operation)
                self.sync_operation_review(operation)
                self.reject(error)

    def test_missing_and_outside_box_points_fail_even_when_review_rehashed(self):
        operation = self.change["replacements"][0]
        original = copy.deepcopy(operation)
        tokens = operation["candidate_row_text"].split()
        tokens[8] = "0.1"
        operation["candidate_row_text"] = " ".join(tokens)
        operation["endpoints"][1]["after"] = tokens[8:11]
        self.sync_operation_review(operation)
        self.reject("missing endpoint must be 0 0 0")

        # Even mutually consistent rehashed evidence cannot admit bad geometry.
        operation.clear()
        operation.update(original)
        tokens = operation["candidate_row_text"].split()
        tokens[5] = "0.99"
        operation["candidate_row_text"] = " ".join(tokens)
        operation["endpoints"][0]["after"] = tokens[5:8]
        point, ranges = [1267.2, 550], [[1265, 1270], [548, 553]]
        operation["endpoints"][0]["accepted_original_pixels"] = point
        operation["endpoints"][0]["coordinate_uncertainty_pixels"] = {
            "x_range": ranges[0], "y_range": ranges[1]}
        recovered = json.loads(self.evidence_path.read_text())
        row = next(item for item in recovered["endpoint_rows"]
                   if item["image_id"] == self.change["image_id"]
                   and item["original_row"] == operation["original_row"])
        row.update(after_tokens=tokens[5:], left_original_pixels=point,
                   visible_point_descriptive_ranges_xy=ranges)
        write_json(self.evidence_path, recovered)
        self.review["evidence"]["sha256"] = file_sha256(self.evidence_path)
        for decision in self.review["decisions"][12:]:
            decision["evidence"] = copy.deepcopy(self.review["evidence"])
        self.sync_operation_review(operation)
        lines = self.label.read_bytes().splitlines(keepends=True)
        lines[operation["original_row"] - 1] = (operation["candidate_row_text"] + "\n").encode()
        self.rehash_label(b"".join(lines))
        self.reject("outside its box")

    def test_omitted_review_and_unexpected_validation_label_are_rejected(self):
        decision = self.review["decisions"].pop()
        self.save()
        self.reject("one new accepted review")
        self.review["decisions"].append(decision)
        self.save()
        target = self.folder / "labels/val/new_000007.txt"
        target.parent.mkdir()
        shutil.copyfile(self.root / "data/labels/val/new_000007.txt", target)
        self.reject("unexpected or missing")

    def test_old_annotation_contract_does_not_accept_endpoint_amendments(self):
        with self.assertRaisesRegex(ValueError, "training_annotation_overlay"):
            verify_annotation_candidate(self.root, self.manifest)
        self.proposal["kind"] = ANNOTATION_KIND
        self.save()
        with self.assertRaisesRegex(ValueError, "original visible endpoint tokens must be preserved"):
            verify_annotation_candidate(self.root, self.manifest)

    def test_no_existing_recipe_admits_endpoint_or_unknown_overlay(self):
        for recipe in RECIPE_IDS:
            with self.subTest(recipe=recipe), self.assertRaisesRegex(ValueError, "staging-only"):
                check_recipe_candidate_kind(recipe, ENDPOINT_KIND)
        with self.assertRaisesRegex(ValueError, "unsupported training candidate kind"):
            check_recipe_candidate_kind(ORIGINAL_RECIPE_ID, "undeclared_overlay")
        with self.assertRaisesRegex(ValueError, "unsupported training recipe"):
            check_recipe_candidate_kind("undeclared_recipe", "training_label_overlay")


if __name__ == "__main__":
    unittest.main()
