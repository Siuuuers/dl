#!/usr/bin/env python3
"""Synthetic return-package checks; no real training, model loading, or accuracy evidence."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from check_pose_checkpoint import bundle_fixture, write_json
from evaluate_pose import file_sha256
from package_gpu_results import INDEX_NAME, package_results
from pose_checkpoint import select_checkpoint, write_checkpoint_manifest
from train_pose_comparison import descriptor


class ReturnPackageChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, self.reference, self.preflight, receipt, staging, _ = bundle_fixture(self.base, arm="reference")
        self.candidate = self.base / "candidate-run"
        shutil.copytree(self.reference, self.candidate)
        staging = copy.deepcopy(staging)
        staging["output_root"] = str(self.candidate / "staged-data")
        write_json(self.candidate / "staging-manifest.json", staging)
        receipt = copy.deepcopy(receipt)
        receipt["arm"] = "candidate"
        receipt["staging_manifest_sha256"] = file_sha256(self.candidate / "staging-manifest.json")
        receipt["data_yaml_sha256"] = staging["variants"]["candidate"]["data_yaml_sha256"]
        receipt["training_settings"]["data"] = str(Path(staging["output_root"]) / "candidate/data.yaml")
        checkpoint = self.candidate / "fit/weights/last.pt"
        checkpoint.write_bytes(b"synthetic candidate checkpoint; never deserialize\n")
        receipt["checkpoint"] = descriptor(checkpoint, self.candidate, checkpoint=True)
        self.log_bytes = b"Synthetic fixture. No training was performed.\rprogress\r\nfinished\n"
        (self.candidate / "train.log").write_bytes(self.log_bytes)
        receipt["log"] = descriptor(self.candidate / "train.log", self.candidate)
        write_json(self.candidate / "training-receipt.json", receipt)
        (self.candidate / "checkpoint-manifest.json").unlink()
        write_checkpoint_manifest(self.candidate, self.preflight)
        self.receipt = receipt
        self.manifest = json.loads((self.candidate / "checkpoint-manifest.json").read_text())
        self.output = self.base / "returned.zip"

    def save_receipt(self, receipt):
        write_json(self.candidate / "training-receipt.json", receipt)
        manifest = copy.deepcopy(self.manifest)
        manifest["training_receipt"] = descriptor(self.candidate / "training-receipt.json", self.candidate)
        write_json(self.candidate / "checkpoint-manifest.json", manifest)

    def package(self, **changes):
        return package_results(**{"root": self.root, "reference": self.reference,
                                  "candidate": self.candidate, "output": self.output, **changes})

    def test_completed_synthetic_pair_roundtrips_with_only_six_files_per_arm(self):
        # The return validator needs frozen validation data and identities, not initialization bytes.
        Path(self.preflight["weights"]).unlink()
        for relative in ("staged-data/candidate/images/train/extra.png", "fit/weights/best.pt",
                         "fit/weights/epoch10.pt", "cache/labels.cache"):
            path = self.candidate / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"excluded synthetic artifact")
        summary = self.package()
        expected_files = {"checkpoint-manifest.json", "training-receipt.json", "staging-manifest.json",
                          "fit/weights/last.pt", "fit/results.csv", "train.log"}
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(set(archive.namelist()), {INDEX_NAME, *(f"{arm}/{path}"
                for arm in ("reference", "candidate") for path in expected_files)})
            self.assertEqual(archive.read("candidate/train.log"), self.log_bytes)
            index = json.loads(archive.read(INDEX_NAME))
            self.assertEqual(len(index["files"]), 12)
            self.assertEqual(index["source"]["candidate_id"], "synthetic-provenance-fixture")
            archive.extractall(self.base / "relocated")
        self.assertEqual(summary["files"], 13)
        self.assertEqual(summary["sha256"], file_sha256(self.output))
        self.assertEqual(summary["size_bytes"], self.output.stat().st_size)
        for arm in ("reference", "candidate"):
            selected = select_checkpoint(self.preflight, self.base / "relocated" / arm / "checkpoint-manifest.json")
            self.assertEqual(selected["checkpoint_identity"]["arm"], arm)
            self.assertEqual(selected["checkpoint_identity"]["completed_epochs"], 30)

    def test_mismatched_source_runtime_and_settings_are_refused(self):
        cases = [("source", "revision", "4" * 40, "training source"),
                 ("runtime", "device", "cuda:0", "runtime"),
                 ("training_settings", "workers", 12, "shared training settings"),
                 ("training_settings", "batch", 8, "paired recipe")]
        for scope, key, value, message in cases:
            with self.subTest(scope=scope, key=key):
                receipt = copy.deepcopy(self.receipt)
                receipt[scope][key] = value
                self.save_receipt(receipt)
                with self.assertRaisesRegex(ValueError, message):
                    self.package()
                self.assertFalse(self.output.exists())

    def test_swapped_arms_are_refused(self):
        with self.assertRaisesRegex(ValueError, "--reference identifies a candidate"):
            self.package(reference=self.candidate, candidate=self.reference)
        self.assertFalse(self.output.exists())

    def test_partial_and_resource_trial_outputs_are_refused(self):
        variants = [{"completed_epochs": list(range(1, 30))},
                    {"purpose": "resource-trial", "status": "trial-complete"},
                    {"status": "failed"}, {"execution": {"exit_code": 1, "stop_reason": "RSS limit"}}]
        for changes in variants:
            with self.subTest(changes=changes):
                self.save_receipt({**copy.deepcopy(self.receipt), **changes})
                with self.assertRaises(ValueError):
                    self.package()
                self.assertFalse(self.output.exists())
        (self.candidate / "checkpoint-manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing artifact"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_changed_final_checkpoint_is_refused(self):
        with (self.candidate / "fit/weights/last.pt").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_existing_output_and_overlapping_locations_are_refused(self):
        self.output.write_bytes(b"preserve existing output")
        with self.assertRaises(FileExistsError):
            self.package()
        self.assertEqual(self.output.read_bytes(), b"preserve existing output")
        self.output.unlink()
        for path in (self.root / "returned.zip", self.reference / "returned.zip", self.candidate / "returned.zip"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "outside"):
                self.package(output=path)
            self.assertFalse(path.exists())
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            self.package(candidate=self.reference)

    def test_ambiguous_archive_path_and_symlink_are_refused(self):
        log = self.candidate / "windows\\train.log"
        log.write_bytes(self.log_bytes)
        receipt = copy.deepcopy(self.receipt)
        receipt["log"] = descriptor(log, self.candidate)
        self.save_receipt(receipt)
        with self.assertRaisesRegex(ValueError, "portable relative path"):
            self.package()
        self.assertFalse(self.output.exists())
        self.save_receipt(self.receipt)
        path = self.candidate / "train.log"
        path.unlink()
        path.symlink_to(log)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.package()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
