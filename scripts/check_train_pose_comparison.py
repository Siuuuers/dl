#!/usr/bin/env python3
"""Check process budgets and input preservation without training a model."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pose_checkpoint import (ANNOTATION_RECIPE_IDS, BOX_RECIPE_ID, BOX_RECIPE_IDS,
                             ENDPOINT_RECIPE_ID, ORIGINAL_RECIPE_ID, WARMUP_RECIPE_ID, get_recipe)
from verify_pose_annotation_candidate import ANNOTATION_CANDIDATE
from verify_pose_box_candidate import BOX_CANDIDATE
from verify_pose_candidate import DEFAULT_CANDIDATE
from verify_pose_endpoint_candidate import ENDPOINT_CANDIDATE
from train_pose_comparison import (file_sha256, main, monitor_worker, resolved_runtime,
                                   run_parent, verify_input_files)


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

    def test_cli_preserves_original_default_and_thirty_epoch_trial_schedule(self):
        arguments = ["--output", "/unused/fixture", "--arm", "reference", "--device", "cpu",
                     "--trial", "--max-seconds", "1"]
        selections = [([], ORIGINAL_RECIPE_ID)] + [(["--recipe", recipe_id], recipe_id)
            for recipe_id in (WARMUP_RECIPE_ID, *BOX_RECIPE_IDS, *ANNOTATION_RECIPE_IDS, ENDPOINT_RECIPE_ID)]
        for extra, expected in selections:
            with self.subTest(recipe=expected), patch("train_pose_comparison.run_parent", return_value=0) as parent:
                self.assertEqual(main(arguments + extra), 0)
                args = parent.call_args.args[0]
                self.assertEqual(args.recipe, expected)
                self.assertTrue(args.trial)
                self.assertEqual(get_recipe(args.recipe)["epochs"], 30)

    def test_cli_rejects_seed_override_and_undeclared_recipe_before_execution(self):
        arguments = ["--output", "/unused/fixture", "--arm", "reference", "--device", "cpu",
                     "--trial", "--max-seconds", "1"]
        for extra in (["--seed", "43"], ["--seed", "46"], ["--recipe", "box-supervision-warmup-seed45-v1"],
                      ["--recipe", "reviewed-annotations-seed47-v1"]):
            with self.subTest(arguments=extra), patch("train_pose_comparison.run_parent") as parent:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    main(arguments + extra)
                self.assertEqual(raised.exception.code, 2)
                parent.assert_not_called()

    def test_warmup_candidate_is_rejected_before_staging_or_importing_training_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            args = SimpleNamespace(root=Path(__file__).resolve().parents[1], output=output,
                threads=1, max_seconds=1, max_rss_mib=128, trial_batches=1,
                device="cpu", recipe=WARMUP_RECIPE_ID, arm="candidate")
            with self.assertRaisesRegex(ValueError, "requires the reference label arm"):
                run_parent(args)
            self.assertFalse(output.exists())

    def test_box_recipe_and_candidate_are_selected_together_before_training(self):
        selections = [(ORIGINAL_RECIPE_ID, BOX_CANDIDATE), (WARMUP_RECIPE_ID, BOX_CANDIDATE)]
        selections.extend((recipe_id, DEFAULT_CANDIDATE) for recipe_id in BOX_RECIPE_IDS)
        for recipe_id, candidate in selections:
            with self.subTest(recipe=recipe_id), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "fresh"
                args = SimpleNamespace(root=Path(__file__).resolve().parents[1], output=output,
                    threads=1, max_seconds=1, max_rss_mib=128, trial_batches=1,
                    device="cpu", recipe=recipe_id, arm="reference", candidate=candidate)
                with self.assertRaisesRegex(ValueError, "must be selected together"):
                    run_parent(args)
                self.assertFalse(output.exists())

    def test_candidate_path_must_stay_in_repository_candidate_directory(self):
        for candidate in ("/tmp/candidate.json", "data/candidates/../../candidate.json", "docs/candidate.json"):
            with self.subTest(candidate=candidate), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "fresh"
                args = SimpleNamespace(root=Path(__file__).resolve().parents[1], output=output,
                    threads=1, max_seconds=1, max_rss_mib=128, trial_batches=1,
                    device="cpu", recipe=BOX_RECIPE_ID, arm="reference", candidate=candidate)
                with self.assertRaisesRegex(ValueError, "repository-relative"):
                    run_parent(args)
                self.assertFalse(output.exists())

    def test_endpoint_recipe_kind_and_arm_are_checked_before_runtime_or_output_creation(self):
        selections = [(recipe, ENDPOINT_CANDIDATE, "candidate", "must be selected together")
                      for recipe in (ORIGINAL_RECIPE_ID, *ANNOTATION_RECIPE_IDS)]
        selections.extend((ENDPOINT_RECIPE_ID, candidate, "candidate", "must be selected together")
                          for candidate in (DEFAULT_CANDIDATE, BOX_CANDIDATE, ANNOTATION_CANDIDATE))
        selections.append((ENDPOINT_RECIPE_ID, ENDPOINT_CANDIDATE, "reference", "requires the candidate label arm"))
        for recipe_id, candidate, arm, message in selections:
            with self.subTest(recipe=recipe_id, candidate=candidate, arm=arm), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "fresh"
                args = SimpleNamespace(root=Path(__file__).resolve().parents[1], output=output,
                    threads=1, max_seconds=1, max_rss_mib=128, trial_batches=1,
                    device="cpu", recipe=recipe_id, arm=arm, candidate=candidate)
                with patch("train_pose_comparison.importlib.metadata.version") as runtime:
                    with self.assertRaisesRegex(ValueError, message):
                        run_parent(args)
                    runtime.assert_not_called()
                self.assertFalse(output.exists())

    def test_endpoint_candidate_admission_reaches_input_preflight_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            args = SimpleNamespace(root=Path(__file__).resolve().parents[1], output=output,
                threads=1, max_seconds=1, max_rss_mib=128, trial_batches=1,
                device="cpu", recipe=ENDPOINT_RECIPE_ID, arm="candidate", candidate=ENDPOINT_CANDIDATE)
            with patch("train_pose_comparison.importlib.metadata.version", return_value="8.4.182"), \
                    patch("train_pose_comparison.preflight_dataset", side_effect=RuntimeError("preflight sentinel")) as preflight:
                with self.assertRaisesRegex(RuntimeError, "preflight sentinel"):
                    run_parent(args)
                preflight.assert_called_once_with(args.root, require_weights=True)
            self.assertFalse(output.exists())

    def test_runtime_records_the_resolved_cuda_device_and_observes_backend_flags(self):
        class Device:
            type = "cuda"
            def __str__(self):
                return "cuda:1"
        device = Device()
        parameter = SimpleNamespace(ndim=4, is_contiguous=Mock(return_value=True))
        torch = SimpleNamespace(channels_last=object(), get_num_threads=lambda: 4, get_num_interop_threads=lambda: 1,
            get_float32_matmul_precision=lambda: "high", are_deterministic_algorithms_enabled=lambda: True,
            is_deterministic_algorithms_warn_only_enabled=lambda: True,
            cuda=SimpleNamespace(get_device_name=Mock(return_value="Synthetic resolved GPU"),
                                 get_device_capability=Mock(return_value=(8, 0))),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(allow_tf32=False, version=lambda: 90100, deterministic=True, benchmark=False)))
        model = SimpleNamespace(parameters=lambda: iter([parameter, SimpleNamespace(ndim=1)]))
        result = resolved_runtime(torch, model, device)
        torch.cuda.get_device_name.assert_called_once_with(device)
        torch.cuda.get_device_capability.assert_called_once_with(device)
        parameter.is_contiguous.assert_called_once_with(memory_format=torch.channels_last)
        self.assertEqual(result["device_name"], "Synthetic resolved GPU")
        self.assertEqual(result["device_capability"], [8, 0])
        self.assertEqual(result["backend"], {"model_channels_last": True, "float32_matmul_precision": "high",
            "cuda_matmul_allow_tf32": True, "cudnn_allow_tf32": False, "cudnn_version": 90100,
            "deterministic_algorithms_enabled": True, "deterministic_warn_only": True,
            "cudnn_deterministic": True, "cudnn_benchmark": False})
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)


if __name__ == "__main__":
    unittest.main()
