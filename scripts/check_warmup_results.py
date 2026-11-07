#!/usr/bin/env python3
"""Synthetic warmup-package checks; never train, load a model, or claim accuracy."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from check_pose_checkpoint import bundle_fixture, write_json
from evaluate_pose import file_sha256
from package_gpu_results import package_results as package_label_comparison
from package_warmup_results import INDEX_NAME, ROLE_RECIPES, package_results
from pose_checkpoint import ORIGINAL_RECIPE_ID, WARMUP_RECIPE_ID, get_recipe, recipe_identity
from pose_checkpoint import select_checkpoint, write_checkpoint_manifest
from train_pose_comparison import descriptor


class WarmupPackageChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        (self.root, self.control, self.preflight, control_receipt,
         control_staging, _) = bundle_fixture(self.base, arm="reference", recipe_id=ORIGINAL_RECIPE_ID)
        self.warmup = self.base / "warmup-run"
        shutil.copytree(self.control, self.warmup)
        staging = copy.deepcopy(control_staging)
        staging["output_root"] = str(self.warmup / "staged-data")
        # Real staged YAMLs embed different absolute paths in the two run directories.
        staging["variants"]["reference"]["data_yaml_sha256"] = hashlib.sha256(
            f"synthetic YAML reference at {staging['output_root']}".encode()).hexdigest()
        write_json(self.warmup / "staging-manifest.json", staging)
        receipt = copy.deepcopy(control_receipt)
        receipt.update(recipe_identity(WARMUP_RECIPE_ID))
        receipt["staging_manifest_sha256"] = file_sha256(self.warmup / "staging-manifest.json")
        receipt["data_yaml_sha256"] = staging["variants"]["reference"]["data_yaml_sha256"]
        receipt["training_settings"].update(get_recipe(WARMUP_RECIPE_ID))
        receipt["training_settings"]["data"] = str(Path(staging["output_root"]) / "reference/data.yaml")
        receipt["training_settings"]["project"] = str(self.warmup)
        receipt["training_settings"]["save_dir"] = str(self.warmup / "fit")
        checkpoint = self.warmup / "fit/weights/last.pt"
        checkpoint.write_bytes(b"synthetic warmup checkpoint; never deserialize\n")
        receipt["checkpoint"] = descriptor(checkpoint, self.warmup, checkpoint=True)
        self.log_bytes = b"Synthetic fixture. No training was performed.\rprogress\r\nfinished\n"
        (self.warmup / "train.log").write_bytes(self.log_bytes)
        receipt["log"] = descriptor(self.warmup / "train.log", self.warmup)
        write_json(self.warmup / "training-receipt.json", receipt)
        (self.warmup / "checkpoint-manifest.json").unlink()
        write_checkpoint_manifest(self.warmup, self.preflight)
        self.runs = {"control": self.control, "warmup": self.warmup}
        self.receipts = {"control": control_receipt, "warmup": receipt}
        self.stagings = {"control": control_staging, "warmup": staging}
        self.manifests = {role: json.loads((run / "checkpoint-manifest.json").read_text())
                          for role, run in self.runs.items()}
        self.output = self.base / "returned.zip"

    def save_receipt(self, receipt, role="warmup", staging=None):
        run = self.runs[role]
        manifest = copy.deepcopy(self.manifests[role])
        if staging is not None:
            write_json(run / "staging-manifest.json", staging)
            receipt["staging_manifest_sha256"] = file_sha256(run / "staging-manifest.json")
            manifest["staging_manifest"] = descriptor(run / "staging-manifest.json", run)
        write_json(run / "training-receipt.json", receipt)
        manifest["training_receipt"] = descriptor(run / "training-receipt.json", run)
        write_json(run / "checkpoint-manifest.json", manifest)

    def package(self, **changes):
        return package_results(**{"root": self.root, "control": self.control,
                                  "warmup": self.warmup, "output": self.output, **changes})

    def copy_control(self, name, *, legacy=False, arm="reference"):
        run = self.base / name
        shutil.copytree(self.control, run)
        receipt = copy.deepcopy(self.receipts["control"])
        if legacy:
            receipt["schema_version"] = 1
            for key in recipe_identity(ORIGINAL_RECIPE_ID):
                receipt.pop(key)
        receipt["arm"] = arm
        variant = self.stagings["control"]["variants"][arm]
        receipt["data_yaml_sha256"] = variant["data_yaml_sha256"]
        receipt["training_settings"]["data"] = str(Path(self.stagings["control"]["output_root"]) / variant["data_yaml"])
        write_json(run / "training-receipt.json", receipt)
        (run / "checkpoint-manifest.json").unlink()
        write_checkpoint_manifest(run, self.preflight)
        return run

    def test_completed_reference_pair_roundtrips_with_distinct_recipe_identities(self):
        Path(self.preflight["weights"]).unlink()
        for relative in ("staged-data/reference/images/train/extra.png", "fit/weights/best.pt",
                         "fit/weights/epoch10.pt", "cache/labels.cache"):
            path = self.warmup / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"excluded synthetic artifact")
        self.assertNotEqual(self.receipts["control"]["data_yaml_sha256"],
                            self.receipts["warmup"]["data_yaml_sha256"])
        summary = self.package()
        expected_files = {"checkpoint-manifest.json", "training-receipt.json", "staging-manifest.json",
                          "fit/weights/last.pt", "fit/results.csv", "train.log"}
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(set(archive.namelist()), {INDEX_NAME, *(f"{role}/{path}"
                for role in ROLE_RECIPES for path in expected_files)})
            self.assertEqual(archive.read("warmup/train.log"), self.log_bytes)
            index = json.loads(archive.read(INDEX_NAME))
            self.assertEqual(index["kind"], "baomai-bias-warmup-results")
            self.assertEqual(index["comparison"], {"label_arm": "reference", "changed_setting": "warmup_bias_lr",
                                                  "values": {"control": 0.1, "warmup": 0.0001}})
            self.assertEqual(len(index["files"]), 12)
            self.assertEqual(index["training_source"], self.receipts["control"]["source"])
            self.assertEqual(index["runtime"], self.receipts["control"]["runtime"])
            self.assertNotIn("warmup_bias_lr", index["shared_training_settings"])
            archive.extractall(self.base / "relocated")
        self.assertEqual(summary["files"], 13)
        self.assertEqual(summary["sha256"], file_sha256(self.output))
        self.assertEqual(summary["size_bytes"], self.output.stat().st_size)
        for role, recipe_id in ROLE_RECIPES.items():
            selected = select_checkpoint(self.preflight, self.base / "relocated" / role / "checkpoint-manifest.json")
            identity = selected["checkpoint_identity"]
            self.assertEqual(identity["arm"], "reference")
            self.assertEqual(identity["completed_epochs"], 30)
            for key, value in recipe_identity(recipe_id).items():
                self.assertEqual(identity[key], value)

    def test_swapped_roles_and_same_recipe_are_refused(self):
        with self.assertRaisesRegex(ValueError, "--control requires recipe"):
            self.package(control=self.warmup, warmup=self.control)
        original = self.copy_control("second-original")
        with self.assertRaisesRegex(ValueError, "--warmup requires recipe"):
            self.package(warmup=original)
        self.assertFalse(self.output.exists())

    def test_valid_legacy_results_cannot_silently_supply_the_control(self):
        legacy = self.copy_control("legacy-control", legacy=True)
        selected = select_checkpoint(self.preflight, legacy / "checkpoint-manifest.json")
        self.assertNotIn("recipe_id", selected["checkpoint_identity"])
        with self.assertRaisesRegex(ValueError, "fresh schema-2 run"):
            self.package(control=legacy)
        self.assertFalse(self.output.exists())

    def test_candidate_labels_and_existing_label_packer_reject_mixed_comparisons(self):
        candidate = self.copy_control("candidate-original", arm="candidate")
        with self.assertRaisesRegex(ValueError, "identifies a candidate"):
            self.package(control=candidate)
        with self.assertRaisesRegex(ValueError, "shared training settings"):
            package_label_comparison(self.root, self.warmup, candidate, self.output)
        self.assertFalse(self.output.exists())

    def test_mismatched_source_hardware_runtime_and_other_settings_are_refused(self):
        cases = [("source", "revision", "4" * 40, "training source"),
                 ("runtime", "python", "3.12.99", "runtime"),
                 ("runtime", "device_name", "different synthetic device", "device identity"),
                 ("training_settings", "workers", 12, "recipe"),
                 ("training_settings", "seed", 43, "recipe"),
                 ("training_settings", "warmup_bias_lr", 0.0, "recipe")]
        for scope, key, value, message in cases:
            with self.subTest(scope=scope, key=key):
                receipt = copy.deepcopy(self.receipts["warmup"])
                receipt[scope][key] = value
                self.save_receipt(receipt)
                with self.assertRaisesRegex(ValueError, message):
                    self.package()
                self.assertFalse(self.output.exists())

    def test_individually_valid_different_cuda_hardware_is_refused(self):
        for role, name, capability in (("control", "synthetic A100", [8, 0]),
                                       ("warmup", "synthetic H100", [9, 0])):
            receipt = copy.deepcopy(self.receipts[role])
            receipt["training_settings"]["device"] = "0"
            receipt["runtime"].update({"device": "cuda:0", "device_name": name,
                                       "device_capability": capability, "torch_cuda": "13.0"})
            self.save_receipt(receipt, role)
            select_checkpoint(self.preflight, self.runs[role] / "checkpoint-manifest.json")
        with self.assertRaisesRegex(ValueError, "control/warmup runtime differ"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_missing_hardware_and_dirty_source_are_refused(self):
        for scope, key in (("runtime", "device_name"), ("runtime", "device_capability")):
            with self.subTest(key=key):
                receipt = copy.deepcopy(self.receipts["warmup"])
                receipt[scope].pop(key)
                self.save_receipt(receipt)
                with self.assertRaises(ValueError):
                    self.package()
                self.assertFalse(self.output.exists())
        receipt = copy.deepcopy(self.receipts["warmup"])
        receipt["source"]["working_tree_status"] = " M scripts/train_pose_comparison.py"
        self.save_receipt(receipt)
        with self.assertRaisesRegex(ValueError, "clean recorded revision"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_changed_staged_reference_evidence_is_refused_after_rehashing(self):
        for change in ("source", "inventory", "counts"):
            with self.subTest(change=change):
                staging = copy.deepcopy(self.stagings["warmup"])
                if change == "source":
                    staging["source"]["different_provenance"] = "synthetic extra field"
                elif change == "inventory":
                    first = staging["variants"]["reference"]["files"][0]
                    first["source_sha256"] = first["output_sha256"] = "4" * 64
                else:
                    staging["variants"]["reference"]["counts"]["train"]["objects"] += 1
                self.save_receipt(copy.deepcopy(self.receipts["warmup"]), staging=staging)
                with self.assertRaises(ValueError):
                    self.package()
                self.assertFalse(self.output.exists())

    def test_partial_and_resource_trial_outputs_are_refused(self):
        changes = [{"completed_epochs": list(range(1, 30))},
                   {"purpose": "resource-trial", "status": "trial-complete"},
                   {"status": "failed"}, {"execution": {"exit_code": 1, "stop_reason": "RSS limit"}}]
        for change in changes:
            with self.subTest(change=change):
                self.save_receipt({**copy.deepcopy(self.receipts["warmup"]), **change})
                with self.assertRaises(ValueError):
                    self.package()
                self.assertFalse(self.output.exists())
        (self.warmup / "checkpoint-manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing artifact"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_changed_checkpoint_or_log_is_refused(self):
        for name in ("fit/weights/last.pt", "train.log"):
            with self.subTest(artifact=name):
                path = self.warmup / name
                original = path.read_bytes()
                path.write_bytes(original + b"tampered")
                try:
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        self.package()
                    self.assertFalse(self.output.exists())
                finally:
                    path.write_bytes(original)

    def test_existing_output_and_overlapping_locations_are_refused(self):
        self.output.write_bytes(b"preserve existing output")
        with self.assertRaises(FileExistsError):
            self.package()
        self.assertEqual(self.output.read_bytes(), b"preserve existing output")
        self.output.unlink()
        for path in (self.root / "returned.zip", self.control / "returned.zip", self.warmup / "returned.zip"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "outside"):
                self.package(output=path)
            self.assertFalse(path.exists())
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            self.package(warmup=self.control)
        with self.assertRaisesRegex(ValueError, "zip extension"):
            self.package(output=self.output.with_suffix(".tar"))

    def test_ambiguous_archive_path_and_symlink_are_refused(self):
        log = self.warmup / "windows\\train.log"
        log.write_bytes(self.log_bytes)
        receipt = copy.deepcopy(self.receipts["warmup"])
        receipt["log"] = descriptor(log, self.warmup)
        self.save_receipt(receipt)
        with self.assertRaisesRegex(ValueError, "portable relative path"):
            self.package()
        self.save_receipt(copy.deepcopy(self.receipts["warmup"]))
        path = self.warmup / "train.log"
        path.unlink()
        path.symlink_to(log)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.package()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
