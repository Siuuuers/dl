#!/usr/bin/env python3
"""Check process budgets and input preservation without training a model."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from train_pose_comparison import (file_sha256, monitor_worker, run_parent,
                                   verify_input_files)


class ExecutionChecks(unittest.TestCase):
    def monitor(self, source, *, max_seconds=5, max_rss_mib=128):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            result = monitor_worker([sys.executable, "-c", source], run,
                {"threads": 1, "max_seconds": max_seconds, "max_rss_mib": max_rss_mib})
            return result, (run / "train.log").read_text()

    def test_completed_worker_is_distinct_from_a_budget_stop(self):
        result, log = self.monitor("print('worker completed', flush=True)")
        self.assertEqual(result["exit_code"], 0)
        self.assertIsNone(result["stop_reason"])
        self.assertEqual(log.strip(), "worker completed")

    def test_wall_budget_terminates_real_worker(self):
        result, _ = self.monitor("import time; time.sleep(5)", max_seconds=0.2)
        self.assertEqual(result["stop_reason"], "wall-time limit exceeded")
        self.assertNotEqual(result["exit_code"], 0)
        self.assertLess(result["wall_seconds"], 4)

    def test_memory_budget_terminates_real_worker(self):
        result, _ = self.monitor("import time; data=bytearray(64*1024*1024); time.sleep(5)", max_rss_mib=32)
        self.assertEqual(result["stop_reason"], "process RSS limit exceeded")
        self.assertGreater(result["sampled_peak_process_tree_rss_mib"], 32)
        self.assertNotEqual(result["exit_code"], 0)

    def test_existing_run_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            marker = run / "keep.txt"
            marker.write_text("preserve me")
            with self.assertRaisesRegex(ValueError, "fresh directory"):
                run_parent(SimpleNamespace(root=Path(__file__).resolve().parents[1], output=run))
            self.assertEqual(marker.read_text(), "preserve me")

    def test_input_mutation_is_detected_after_cache_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            staging = {"variants": {}}
            for arm in ("reference", "candidate"):
                arm_root = run / "staged-data" / arm
                label = arm_root / "labels/train/example.txt"
                label.parent.mkdir(parents=True)
                label.write_text("controlled input")
                yaml = arm_root / "data.yaml"
                yaml.write_text("recorded dataset configuration")
                staging["variants"][arm] = {
                    "data_yaml": f"{arm}/data.yaml", "data_yaml_sha256": file_sha256(yaml),
                    "files": [{"path": "labels/train/example.txt", "output_sha256": file_sha256(label)}]}
            text = json.dumps(staging)
            (run / "staging-manifest.json").write_text(text)
            (run / "staged-data/staging-manifest.json").write_text(text)
            (run / "staged-data/reference/labels/train.cache").write_bytes(b"disposable loader cache")
            verify_input_files(run, staging)
            label.write_text("unexpected replacement")
            with self.assertRaisesRegex(ValueError, "training input changed"):
                verify_input_files(run, staging)


if __name__ == "__main__":
    unittest.main()
