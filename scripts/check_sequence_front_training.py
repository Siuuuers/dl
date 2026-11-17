#!/usr/bin/env python3
"""Check the actual v2 staging and immutable hold/reference rules without a GPU.

The original images must be present. Weights, network access and a training
runtime are not required; this does not claim that a GPU run has succeeded.
"""

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

from evaluate_pose import preflight_dataset
from pose_checkpoint import _check_staging, training_label_identity
from replay_corner_reference import reference_counts
from rescore_reference_amendment import load_api, verify_amendment
from stage_pose_training import stage_datasets, verify_staging
from verify_incremental_pose_candidate import (
    SEQUENCE_CANDIDATE, SEQUENCE_PRIMARY_REFERENCE, SEQUENCE_PROTOCOL_PATH,
    SEQUENCE_REVIEW, verify_incremental_candidate,
)
from verify_reconciled_pose_labels import encoded, sha


ROOT = Path(__file__).resolve().parents[1]


def replace_bytes(path, payload):
    # Copy on write: mutations must never reach the repository through a link.
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class SequenceTrainingChecks(unittest.TestCase):
    def snapshot(self):
        temporary = tempfile.TemporaryDirectory(prefix="sequence-release-check-", dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "snapshot"
        for relative in ("data", "docs", "results/review", "metadata", "weights", "scripts", "experiments"):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(ROOT / relative, target, copy_function=os.link)
        return root

    def test_staged_bytes_and_separate_references(self):
        receipt = verify_incremental_candidate(ROOT, SEQUENCE_CANDIDATE)
        protocol = json.loads((ROOT / SEQUENCE_PROTOCOL_PATH).read_text())
        reviewed_path = ROOT / SEQUENCE_REVIEW["path"]
        reviewed = json.loads(reviewed_path.read_text())
        reviewed_files = {item["image_id"]: item for item in reviewed["files"]}
        held = set(receipt["retained_legacy_image_ids"])
        self.assertIn("new_000034", held)
        self.assertNotIn("new_000034", receipt["changed_images_vs_control"])
        self.assertEqual(receipt["counts"]["train"]["images"], 137)
        self.assertEqual(receipt["counts"]["train"]["empty_images"], 13)
        preflight = preflight_dataset(ROOT, require_weights=False)
        with tempfile.TemporaryDirectory(prefix="sequence-staging-check-", dir=ROOT / "tmp") as temporary:
            output = Path(temporary) / "staged-data"
            stage_datasets(ROOT, ROOT / SEQUENCE_CANDIDATE, output)
            self.assertEqual(verify_staging(ROOT, ROOT / SEQUENCE_CANDIDATE, output)["status"], "verified")
            staging = json.loads((output / "staging-manifest.json").read_text())
            _check_staging(preflight, staging)
            self.assertEqual(staging["code_sha256"]["scripts/verify_incremental_pose_candidate.py"],
                             sha(ROOT / "scripts/verify_incremental_pose_candidate.py"))
            records = json.loads((ROOT / "data/manifest.json").read_text())["records"]
            changed = []
            for record in records:
                reference = (output / "reference" / record["label"]).read_bytes()
                candidate = (output / "candidate" / record["label"]).read_bytes()
                if record["split"] == "val":
                    self.assertEqual(reference, candidate)
                    self.assertEqual(candidate, (ROOT / "data" / record["label"]).read_bytes())
                elif record["id"] in held:
                    self.assertEqual(reference, candidate)
                else:
                    entry = reviewed_files[record["id"]]
                    self.assertEqual(candidate, (reviewed_path.parent / entry["candidate_label"]).read_bytes())
                if reference != candidate:
                    changed.append(record["id"])
            self.assertEqual(sorted(changed), sorted(receipt["changed_images_vs_control"]))
            for arm in ("reference", "candidate"):
                self.assertEqual(staging["variants"][arm]["counts"], protocol["roles"][arm]["counts"])
                self.assertEqual(training_label_identity(staging, arm)["role"],
                                 "pr35_annotation_control" if arm == "reference" else "accepted_fronts_treatment")
            forged = copy.deepcopy(staging)
            forged["code_sha256"]["scripts/verify_incremental_pose_candidate.py"] = "0" * 64
            (output / "staging-manifest.json").write_bytes(encoded(forged))
            with self.assertRaisesRegex(ValueError, "configuration or code"):
                verify_staging(ROOT, ROOT / SEQUENCE_CANDIDATE, output)
        _, api = load_api(ROOT)
        for role in ("primary", "diagnostic"):
            descriptor = protocol["evaluation"][role + "_reference"]
            reference, gold, *_ = verify_amendment(ROOT, (ROOT / descriptor).parent, preflight, api)
            counts, groups = reference_counts(gold)
            self.assertEqual(counts, protocol["evaluation"][role + "_counts"])
            self.assertEqual(groups, reference["objects_by_locatable_corners"])
        self.assertEqual(protocol["evaluation"]["primary_reference_sha256"], SEQUENCE_PRIMARY_REFERENCE["sha256"])
        self.assertTrue(protocol["evaluation"]["diagnostic_reference_required"])

    def test_self_consistent_manifest_cannot_release_held_photo(self):
        root = self.snapshot()
        path = root / SEQUENCE_CANDIDATE
        manifest = json.loads(path.read_text())
        manifest["retained_legacy_image_ids"].remove("new_000034")
        manifest["unresolved_case_ids"] = [case for case in manifest["unresolved_case_ids"]
                                            if not case.startswith("new_000034")]
        replace_bytes(path, encoded(manifest))
        protocol_path = root / SEQUENCE_PROTOCOL_PATH
        protocol = json.loads(protocol_path.read_text())
        protocol["candidate"]["sha256"] = sha(path)
        for role in protocol["roles"].values():
            role["candidate_manifest_sha256"] = sha(path)
        replace_bytes(protocol_path, encoded(protocol))
        with self.assertRaisesRegex(ValueError, "exact unresolved-photo composition"):
            verify_incremental_candidate(root, SEQUENCE_CANDIDATE)

    def test_protocol_cannot_drop_historical_comparison_or_promote(self):
        root = self.snapshot()
        path = root / SEQUENCE_PROTOCOL_PATH
        protocol = json.loads(path.read_text())
        protocol["evaluation"]["diagnostic_reference_required"] = False
        protocol["execution"]["automatic_promotion"] = True
        replace_bytes(path, encoded(protocol))
        with self.assertRaisesRegex(ValueError, "fixed experiment and evaluation scope"):
            verify_incremental_candidate(root, SEQUENCE_CANDIDATE)


if __name__ == "__main__":
    (ROOT / "tmp").mkdir(exist_ok=True)
    unittest.main(verbosity=2)
