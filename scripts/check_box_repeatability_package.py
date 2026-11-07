#!/usr/bin/env python3
"""Synthetic four-run packaging checks; no training, inference or model loading."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from check_pose_checkpoint import bundle_fixture, write_json
from evaluate_pose import file_sha256
from package_box_repeatability import (ARMS, CANDIDATE_ID, CANDIDATE_MANIFEST,
                                       INDEX_NAME, SEEDS, STUDY_ID, package_repeatability)
from pose_checkpoint import (BOX_RECIPE_ID, BOX_REPEATABILITY_RECIPES, PINNED_TRAINER_DEFAULTS,
                             get_recipe, recipe_identity, select_checkpoint, write_checkpoint_manifest)
from train_pose_comparison import descriptor


class RepeatabilityPackageChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="check-box-repeatability-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, template, self.preflight, receipt, staging, _ = bundle_fixture(
            self.base, arm="reference", recipe_id=BOX_REPEATABILITY_RECIPES[43])
        proposal = json.loads((self.root / "data/candidates/fixture/manifest.json").read_text())
        proposal["candidate_id"] = CANDIDATE_ID
        proposal_path = self.root / CANDIDATE_MANIFEST
        proposal_path.parent.mkdir(parents=True)
        write_json(proposal_path, proposal)
        self.fixture_candidate_hash = file_sha256(proposal_path)
        # Only the declared proposal byte identity is substituted for synthetic data.
        # Production retains its literal reviewed-manifest hash; recipes/seeds are real.
        candidate_pin = patch("package_box_repeatability.CANDIDATE_SHA256", self.fixture_candidate_hash)
        candidate_pin.start()
        self.addCleanup(candidate_pin.stop)
        self.runs = {}
        for seed in SEEDS:
            self.runs[seed] = {}
            for arm in ARMS:
                run = self.base / f"seed-{seed}-{arm}"
                shutil.copytree(template, run)
                staged = copy.deepcopy(staging)
                staged["output_root"] = str(run / "staged-data")
                staged["source"].update(candidate_id=CANDIDATE_ID, candidate_manifest=CANDIDATE_MANIFEST,
                                         candidate_manifest_sha256=self.fixture_candidate_hash)
                write_json(run / "staging-manifest.json", staged)
                value = copy.deepcopy(receipt)
                value.update(arm=arm, **recipe_identity(BOX_REPEATABILITY_RECIPES[seed]))
                value["staging_manifest_sha256"] = file_sha256(run / "staging-manifest.json")
                value["data_yaml_sha256"] = staged["variants"][arm]["data_yaml_sha256"]
                value["training_settings"] = {**PINNED_TRAINER_DEFAULTS, **get_recipe(value["recipe_id"]),
                    "model": self.preflight["weights"], "data": str(run / "staged-data" / arm / "data.yaml"),
                    "device": "cpu", "project": str(run), "save_dir": str(run / "fit")}
                weights = run / "fit/weights/last.pt"
                weights.write_bytes(f"Synthetic seed {seed} {arm}; never deserialize.\n".encode())
                value["checkpoint"] = descriptor(weights, run, checkpoint=True)
                value.update(batches_completed=1050, images_processed=4110)
                write_json(run / "training-receipt.json", value)
                (run / "checkpoint-manifest.json").unlink()
                write_checkpoint_manifest(run, self.preflight)
                self.runs[seed][arm] = run
        self.output = self.base / "returned.zip"

    def package(self, **changes):
        return package_repeatability(**{"root": self.root, "runs": self.runs,
                                        "output": self.output, **changes})

    def rewrite(self, seed, arm, update):
        run = self.runs[seed][arm]
        receipt = json.loads((run / "training-receipt.json").read_text())
        update(receipt)
        write_json(run / "training-receipt.json", receipt)
        manifest = json.loads((run / "checkpoint-manifest.json").read_text())
        manifest.update({key: receipt[key] for key in ("recipe_id", "recipe_sha256")})
        manifest["training_receipt"] = descriptor(run / "training-receipt.json", run)
        write_json(run / "checkpoint-manifest.json", manifest)

    def test_four_complete_runs_roundtrip_as_two_flat_pairs(self):
        result = self.package()
        self.assertEqual(result["study_id"], STUDY_ID)
        self.assertEqual(result["seeds"], [43, 44])
        self.assertEqual(result["files"], 25)
        with zipfile.ZipFile(self.output) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), 25)
            index = json.loads(archive.read(INDEX_NAME))
            self.assertEqual(index["kind"], "baomai-box-repeatability-results")
            self.assertEqual(index["source"]["candidate_manifest_sha256"], self.fixture_candidate_hash)
            self.assertNotIn("seed", index["shared_training_settings"])
            self.assertIsNone(index["resource_trial"])
            archive.extractall(self.base / "restored")
        self.assertEqual(result["sha256"], file_sha256(self.output))
        for seed in SEEDS:
            for arm in ARMS:
                relative = f"seed-{seed}/{arm}/checkpoint-manifest.json"
                self.assertEqual(index["pairs"][f"seed-{seed}"]["arms"][arm]["checkpoint_manifest"], relative)
                selected = select_checkpoint(self.preflight, self.base / "restored" / relative)
                self.assertEqual(selected["checkpoint_identity"]["recipe_id"], BOX_REPEATABILITY_RECIPES[seed])
                self.assertEqual(selected["checkpoint_identity"]["arm"], arm)
                self.assertEqual(selected["checkpoint_identity"]["completed_epochs"], 30)

    def test_wrong_seed_recipe_and_legacy_seed42_are_refused(self):
        original = {name: (self.runs[43]["candidate"] / name).read_bytes()
                    for name in ("training-receipt.json", "checkpoint-manifest.json")}
        for wrong_recipe in (BOX_REPEATABILITY_RECIPES[44], BOX_RECIPE_ID):
            with self.subTest(recipe=wrong_recipe):
                def change(receipt):
                    receipt.update(recipe_identity(wrong_recipe))
                    receipt["training_settings"].update(get_recipe(wrong_recipe))
                self.rewrite(43, "candidate", change)
                # The wrong-seed bundle is valid in isolation; the study must reject its role.
                select_checkpoint(self.preflight, self.runs[43]["candidate"] / "checkpoint-manifest.json")
                with self.assertRaisesRegex(ValueError, "seed 43.*exact repeatability recipe"):
                    self.package()
                self.assertFalse(self.output.exists())
                for name, data in original.items():
                    (self.runs[43]["candidate"] / name).write_bytes(data)

    def test_missing_or_partial_arm_never_creates_a_return_zip(self):
        run = self.runs[44]["candidate"]
        path = run / "checkpoint-manifest.json"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing artifact"):
            self.package()
        self.assertFalse(self.output.exists())
        path.write_bytes(original)
        self.rewrite(44, "candidate", lambda r: r.update(status="failed", completed_epochs=list(range(1, 20))))
        with self.assertRaisesRegex(ValueError, "completed paired-finetune"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_changed_candidate_bytes_are_rejected_even_when_all_receipts_agree(self):
        with (self.root / CANDIDATE_MANIFEST).open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "exact reviewed box-candidate manifest"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_cross_seed_runtime_mismatch_is_rejected_after_each_pair_validates(self):
        for arm in ARMS:
            self.rewrite(44, arm, lambda r: r["runtime"].update(platform="different synthetic environment"))
        with self.assertRaisesRegex(ValueError, "cross-seed runtime"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_existing_return_is_preserved(self):
        self.output.write_bytes(b"preserve the existing result")
        with self.assertRaises(FileExistsError):
            self.package()
        self.assertEqual(self.output.read_bytes(), b"preserve the existing result")

    def test_trial_is_only_supplemental_and_must_complete_four_batches(self):
        trial = self.base / "resource-trial"
        shutil.copytree(self.runs[43]["candidate"], trial)
        staging = json.loads((trial / "staging-manifest.json").read_text())
        staging["output_root"] = str(trial / "staged-data")
        write_json(trial / "staging-manifest.json", staging)
        receipt = json.loads((trial / "training-receipt.json").read_text())
        receipt.update(purpose="resource-trial", status="trial-complete", completed_epochs=[],
                       batches_completed=4, optimizer_steps=4, images_processed=16,
                       checkpoint_selection="none", training_returned=False,
                       finalization_completed=False, validation_executed=False)
        for name in ("checkpoint", "results"):
            receipt.pop(name)
        receipt["staging_manifest_sha256"] = file_sha256(trial / "staging-manifest.json")
        receipt["training_settings"].update(data=str(trial / "staged-data/candidate/data.yaml"),
                                            project=str(trial), save_dir=str(trial / "fit"))
        write_json(trial / "training-receipt.json", receipt)
        (trial / "checkpoint-manifest.json").unlink()
        result = self.package(trial=trial)
        self.assertEqual(result["files"], 28)
        with zipfile.ZipFile(self.output) as archive:
            index = json.loads(archive.read(INDEX_NAME))
            self.assertFalse(index["resource_trial"]["eligible_checkpoint"])
            self.assertEqual({name for name in archive.namelist() if name.startswith("resource-trial/")},
                             {"resource-trial/training-receipt.json", "resource-trial/staging-manifest.json", "resource-trial/train.log"})
        self.output.unlink()
        receipt["batches_completed"] = 1
        write_json(trial / "training-receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "four batches"):
            self.package(trial=trial)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
