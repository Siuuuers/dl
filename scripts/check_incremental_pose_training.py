#!/usr/bin/env python3
"""Exercise the actual incremental artifacts and reject unsafe release mutations.

All mutations occur in disposable hard-linked snapshots with copy-on-write;
the repository labels, source evidence and release gates remain untouched.
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
from pose_checkpoint import (INCREMENTAL_RECIPE_ID, OBSERVED_RECIPE_ID, _check_staging,
                             check_recipe_candidate_kind, get_recipe, recipe_identity,
                             training_label_identity)
from stage_pose_training import stage_datasets
from train_pose_comparison import worker_training_labels
from verify_incremental_pose_candidate import (ADJUDICATION, DEFAULT_CANDIDATE, PROTOCOL_PATH,
                                              SOURCE_CANDIDATE, SOURCE_PROTOCOL, SOURCE_STATUS,
                                              verify_incremental_candidate)
from verify_observed_training_candidate import verify_observed_training_candidate
from verify_reconciled_pose_labels import encoded, sha


ROOT = Path(__file__).resolve().parents[1]


def replace_bytes(path, payload):
    """Unlink before writing so snapshot mutations never follow hard links."""
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class IncrementalTrainingChecks(unittest.TestCase):
    def snapshot(self):
        temporary = tempfile.TemporaryDirectory(prefix="incremental-release-check-", dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "snapshot"
        for relative in ("data", "docs", "results/review", "metadata", "weights", "scripts", "experiments"):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(ROOT / relative, target, copy_function=os.link)
        return root

    def change_manifest(self, root, mutate):
        path = root / DEFAULT_CANDIDATE
        document = json.loads(path.read_text())
        mutate(document)
        replace_bytes(path, encoded(document))
        # Refresh the public protocol digest too: a self-consistent forged hash
        # must not bypass reconstruction from the independently bound evidence.
        protocol_path = root / PROTOCOL_PATH
        protocol = json.loads(protocol_path.read_text())
        protocol["candidate"]["sha256"] = sha(path)
        for role in protocol["roles"].values():
            role["candidate_manifest_sha256"] = sha(path)
        replace_bytes(protocol_path, encoded(protocol))

    def test_exact_staging_preserves_cohort_legacy_bytes_and_arm_identity(self):
        with tempfile.TemporaryDirectory(prefix="incremental-staging-check-") as directory:
            output = Path(directory) / "staged-data"
            stage_datasets(ROOT, ROOT / DEFAULT_CANDIDATE, output)
            staging = json.loads((output / "staging-manifest.json").read_text())
            receipt = verify_incremental_candidate(ROOT)
            self.assertEqual(receipt["counts"]["train"], {
                "images": 137, "objects": 3788, "visible_endpoints": 6858,
                "missing_endpoints": 718, "empty_images": 13})
            self.assertEqual(len(receipt["changed_images_vs_control"]), 71)
            self.assertEqual(len(receipt["retained_legacy_image_ids"]), 21)
            self.assertEqual(len(receipt["unresolved_case_ids"]), 24)
            self.assertEqual(len(receipt["unresolved_reference_rows"]), 13)
            records = json.loads((ROOT / "data/manifest.json").read_text())["records"]
            for record in records:
                if record["split"] == "val" or record["id"] in receipt["retained_legacy_image_ids"]:
                    self.assertEqual((output / "reference" / record["label"]).read_bytes(),
                                     (output / "candidate" / record["label"]).read_bytes())
            preflight = preflight_dataset(ROOT, require_weights=False)
            _check_staging(preflight, staging)
            for arm in ("reference", "candidate"):
                labels = training_label_identity(staging, arm)
                self.assertEqual(labels["role"], "pr35_annotation_control" if arm == "reference"
                                 else "accepted_fronts_treatment")
                config = {"recipe_id": INCREMENTAL_RECIPE_ID, "arm": arm,
                          "candidate_manifest": DEFAULT_CANDIDATE, "training_labels": labels}
                self.assertEqual(worker_training_labels(config, staging), labels)
                config["training_labels"] = training_label_identity(staging, "candidate" if arm == "reference" else "reference")
                with self.assertRaisesRegex(ValueError, "worker control/treatment label identity"):
                    worker_training_labels(config, staging)
            forged = copy.deepcopy(staging)
            forged["source"]["retained_legacy_image_ids"].pop()
            with self.assertRaisesRegex(ValueError, "legacy-photo inventory differs"):
                _check_staging(preflight, forged)

    def test_omitted_held_image_and_case_fail_after_protocol_hash_refresh(self):
        root = self.snapshot()
        def omit(document):
            document["retained_legacy_image_ids"].remove("new_000094")
            document["unresolved_case_ids"] = [case for case in document["unresolved_case_ids"]
                                               if not case.startswith("new_000094")]
            row = next(item for item in document["image_sources"] if item["image_id"] == "new_000094")
            row["unresolved_case_ids"] = []
            row["selection"] = "accepted_observed_source"
        self.change_manifest(root, omit)
        with self.assertRaisesRegex(ValueError, "exact unresolved-photo composition"):
            verify_incremental_candidate(root)

    def test_retained_photo_cannot_receive_reviewed_labels_after_hash_refresh(self):
        root = self.snapshot()
        source_path = root / SOURCE_CANDIDATE["path"]
        source = json.loads(source_path.read_text())
        change = next(item for item in source["changes"] if item["image_id"] == "new_000030")
        new_label = root / DEFAULT_CANDIDATE
        new_label = new_label.parent / change["candidate_label"]
        replace_bytes(new_label, (source_path.parent / change["candidate_label"]).read_bytes())
        def substitute(document):
            document["changes"].append({key: change[key] for key in (
                "image_id", "source_label", "candidate_label", "source_label_sha256",
                "candidate_label_sha256", "source_image_sha256", "source_objects", "candidate_objects")})
            document["changed_images_vs_control"].append("new_000030")
        self.change_manifest(root, substitute)
        with self.assertRaisesRegex(ValueError, "exact unresolved-photo composition"):
            verify_incremental_candidate(root)

    def test_source_and_adjudication_hash_drift_fail_closed(self):
        root = self.snapshot()
        for descriptor in (SOURCE_CANDIDATE, SOURCE_PROTOCOL, SOURCE_STATUS, ADJUDICATION):
            with self.subTest(path=descriptor["path"]):
                path = root / descriptor["path"]
                original = path.read_bytes()
                replace_bytes(path, original + b"\n")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    verify_incremental_candidate(root)
                replace_bytes(path, original)

    def test_protocol_cannot_silently_drop_reference_diagnostic_or_promote(self):
        root = self.snapshot()
        path = root / PROTOCOL_PATH
        document = json.loads(path.read_text())
        document["evaluation"]["diagnostic_reference_required"] = False
        document["execution"]["automatic_promotion"] = True
        replace_bytes(path, encoded(document))
        with self.assertRaisesRegex(ValueError, "fixed experiment and evaluation scope"):
            verify_incremental_candidate(root)

    def test_full_review_gate_stays_blocked_and_recipes_cannot_be_swapped(self):
        with self.assertRaisesRegex(ValueError, "26 unresolved training case groups.*13 unresolved development-reference rows"):
            verify_observed_training_candidate(ROOT, ROOT / SOURCE_CANDIDATE["path"])
        self.assertEqual(get_recipe(INCREMENTAL_RECIPE_ID), get_recipe(OBSERVED_RECIPE_ID))
        self.assertEqual(recipe_identity(INCREMENTAL_RECIPE_ID)["recipe_sha256"],
                         "f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8")
        for recipe, kind in ((INCREMENTAL_RECIPE_ID, "training_observed_review_composition"),
                             (OBSERVED_RECIPE_ID, "training_incremental_observed_overlay")):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                check_recipe_candidate_kind(recipe, kind)


if __name__ == "__main__":
    unittest.main(verbosity=2)
