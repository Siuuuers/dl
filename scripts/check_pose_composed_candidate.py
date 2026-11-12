#!/usr/bin/env python3
"""Focused composed-contract tests, including rehashed hostile label fixtures.

Only temporary label/provenance snapshots are changed. No PNG, checkpoint,
inference, training, or GPU is required.
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
from verify_pose_composed_candidate import (
    COMPOSED_CANDIDATE, CONTROL, CURRENT_POLICY, FROZEN_SPEC, HISTORICAL_POLICY,
    PARENT, RECOVERY, STRICT_CODE_SHA256, verify_composed_candidate,
)
from verify_pose_endpoint_candidate import verify_endpoint_candidate


REPOSITORY = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def copy_fixture(destination, repository=REPOSITORY):
    """Copy just the real contract inputs; reusable by narrow integration tests."""
    destination, repository = Path(destination), Path(repository)
    for relative in ("data/labels", str(Path(PARENT["manifest"]).parent),
                     str(Path(CONTROL["manifest"]).parent), str(Path(COMPOSED_CANDIDATE).parent)):
        shutil.copytree(repository / relative, destination / relative)
    control = json.loads((repository / CONTROL["manifest"]).read_text())
    control_review = json.loads((repository / Path(CONTROL["manifest"]).parent / control["review"]["path"]).read_text())
    spec = json.loads((repository / FROZEN_SPEC["path"]).read_text())
    files = {"data/manifest.json", CURRENT_POLICY["path"], HISTORICAL_POLICY["path"],
             FROZEN_SPEC["path"], RECOVERY["path"], control_review["evidence"]["path"],
             "scripts/pose_checkpoint.py", *STRICT_CODE_SHA256}
    files.update(item["box_proposal"]["path"] for item in spec["files"])
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, target)
    return destination / COMPOSED_CANDIDATE


class ComposedCandidateChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="composed-candidate-check-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = copy_fixture(self.root)
        self.folder = self.manifest.parent
        self.proposal = json.loads(self.manifest.read_text())
        self.review_path = self.folder / self.proposal["review"]["path"]
        self.review = json.loads(self.review_path.read_text())
        self.change = next(item for item in self.proposal["changes"] if item["image_id"] == "new_000023")
        self.label = self.folder / self.change["candidate_label"]

    def save(self):
        write_json(self.review_path, self.review)
        self.proposal["review"]["sha256"] = file_sha256(self.review_path)
        write_json(self.manifest, self.proposal)

    def rehash_label(self, payload):
        self.label.write_bytes(payload)
        self.change["candidate_label_sha256"] = file_sha256(self.label)
        self.save()

    def reject(self, pattern):
        with self.assertRaisesRegex((ValueError, OSError), pattern):
            verify_composed_candidate(self.root, self.manifest)

    def test_real_contract_preserves_exact_control_and_original_validation(self):
        policy_before = (self.root / CURRENT_POLICY["path"]).read_bytes()
        receipt = verify_composed_candidate(self.root, self.manifest)
        self.assertEqual(receipt["counts"]["train"], {
            "images": 137, "objects": 3770, "visible_endpoints": 6842,
            "missing_endpoints": 698, "empty_images": 13})
        self.assertEqual(receipt["control"]["manifest_sha256"], CONTROL["sha256"])
        self.assertEqual(receipt["control"]["counts"]["train"]["objects"], 3771)
        self.assertEqual(receipt["control"]["changed_endpoint_slots"], 10)
        self.assertEqual(receipt["changed_train_labels"], 11)
        self.assertEqual(receipt["removed_objects"], 7)
        self.assertEqual(receipt["replaced_objects"], 26)
        self.assertEqual(receipt["delta_from_control"]["changed_endpoint_slots"], 0)
        self.assertTrue(receipt["validation_labels_unchanged"])
        self.assertTrue(receipt["historical_validation"]["isolated_snapshot"])
        self.assertFalse(receipt["image_bytes_verified"])
        self.assertFalse(receipt["visual_correctness_verified"])
        self.assertEqual((self.root / CURRENT_POLICY["path"]).read_bytes(), policy_before)
        rows = self.change["row_map"]
        self.assertIsNone(rows[21]["candidate_row"])
        self.assertEqual(rows[32]["candidate_row"], 32)
        self.assertEqual(rows[35]["candidate_row"], 35)

    def test_old_strict_validator_still_rejects_live_policy_mismatch(self):
        with self.assertRaisesRegex(ValueError, "annotation policy identity differs"):
            verify_endpoint_candidate(self.root, self.root / CONTROL["manifest"])

    def test_control_identity_cannot_be_rehashed_or_swapped(self):
        control = self.root / CONTROL["manifest"]
        control.write_bytes(control.read_bytes() + b"\n")
        self.proposal["control"]["sha256"] = file_sha256(control)
        self.save()
        self.reject("control identity differs")

    def test_rehashed_unrecorded_bbox_cannot_enter_treatment(self):
        lines = self.label.read_bytes().splitlines(keepends=True)
        tokens = lines[0].decode().split()
        tokens[1] = f"{float(tokens[1]) + 0.0001:.8f}"
        lines[0] = (" ".join(tokens) + "\n").encode()
        self.rehash_label(b"".join(lines))
        self.reject("change records/row maps differ")

    def test_older_nine_slot_endpoint_payload_is_rejected_even_when_rehashed(self):
        lines = self.label.read_bytes().splitlines(keepends=True)
        # Original source row33 is treatment row32 after the duplicate deletion.
        tokens = lines[31].decode().split()
        tokens[8:11] = ["0.00000000", "0.00000000", "0"]
        lines[31] = (" ".join(tokens) + "\n").encode()
        replacement = next(item for item in self.change["replacements"] if item["original_row"] == 33)
        replacement["candidate_row_text"] = " ".join(tokens)
        self.rehash_label(b"".join(lines))
        self.reject("change records/row maps differ")

    def test_wrong_duplicate_identity_cannot_be_declared(self):
        deletion = self.change["deletions"][0]
        deletion["original_row"] = 36
        deletion["retained_original_row"] = 22
        for row in self.change["row_map"]:
            n = row["original_row"]
            row["candidate_row"] = None if n == 36 else n
        self.save()
        self.reject("change records/row maps differ")

    def test_full_row_map_rejects_shift_and_boolean_substitution(self):
        original = copy.deepcopy(self.change["row_map"])
        for number, value in ((33, 33), (1, True)):
            with self.subTest(source_row=number):
                self.change["row_map"] = copy.deepcopy(original)
                self.change["row_map"][number - 1]["candidate_row"] = value
                self.save()
                self.reject("change records/row maps differ")

    def test_rehashed_inherited_parent_file_is_rejected(self):
        inherited = self.proposal["changes"][0]
        path = self.folder / inherited["candidate_label"]
        path.write_bytes(path.read_bytes() + b"\n")
        inherited["candidate_label_sha256"] = file_sha256(path)
        self.save()
        self.reject("change records/row maps differ")

    def test_missing_extra_and_symlink_overlay_files_are_rejected(self):
        original = self.label.read_bytes()
        self.label.unlink()
        self.reject("unexpected or missing composed overlay")
        self.label.write_bytes(original)
        extra = self.folder / "labels/train/not_an_approved_image.txt"
        extra.write_bytes(b"")
        self.reject("unexpected or missing composed overlay")
        extra.unlink()
        extra.symlink_to(self.label)
        self.reject("regular files/directories")

    def test_original_validation_byte_edit_is_rejected(self):
        path = self.root / "data/labels/val/new_000010.txt"
        path.write_bytes(path.read_bytes() + b"\n")
        self.reject("source label SHA-256 mismatch")

    def test_both_policy_pins_are_enforced_without_rewriting_history(self):
        current = self.root / CURRENT_POLICY["path"]
        before = current.read_bytes()
        current.write_bytes(before + b"\nChanged policy.\n")
        self.reject("current policy SHA-256 mismatch")
        current.write_bytes(before)
        historical = self.root / HISTORICAL_POLICY["path"]
        historical.write_bytes(historical.read_bytes() + b"\n")
        self.reject("historical policy SHA-256 mismatch")

    def test_rehashing_specification_does_not_authorize_new_operations(self):
        path = self.root / FROZEN_SPEC["path"]
        spec = json.loads(path.read_text())
        spec["operations"][0]["after_tokens"][0] = "0.50000000"
        write_json(path, spec)
        self.proposal["frozen_specification"]["sha256"] = file_sha256(path)
        self.review["frozen_specification"] = copy.deepcopy(self.proposal["frozen_specification"])
        self.save()
        self.reject("frozen_specification identity differs")

    def test_strict_historical_validator_cannot_be_modified(self):
        path = self.root / "scripts/verify_pose_endpoint_candidate.py"
        path.write_bytes(path.read_bytes() + b"\n# unapproved validator modification\n")
        self.reject("historical strict validator/helper changed")

    def test_manifest_counts_cannot_mask_wrong_arm_or_object_inventory(self):
        self.proposal["expected_counts"]["train"]["objects"] = 3771
        self.save()
        self.reject("composed counts or unchanged validation totals differ")


if __name__ == "__main__":
    unittest.main()
