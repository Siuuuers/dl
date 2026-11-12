#!/usr/bin/env python3
"""Exercise the composed-fronts notebook with GPU/network/install substitutes.

These tests verify orchestration and recovery; they do not run training.
"""

import sys
sys.dont_write_bytecode = True

import ast
import copy
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pose_checkpoint import get_recipe, recipe_identity


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/baomai_composed_fronts.ipynb"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


class ComposedNotebookChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="check-composed-notebook-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        notebook = json.loads(NOTEBOOK.read_text())
        self.cells = {i: "".join(cell["source"]) for i, cell in enumerate(notebook["cells"])
                      if cell["cell_type"] == "code"}
        self.commands = []

    def execute_workspace(self, revision="a" * 40, protocol="b" * 64):
        tree = ast.parse(self.cells[2])
        replacements = {"SOURCE_REF": revision, "PROTOCOL_SHA256": protocol,
                        "RUN_BASE": str(self.base / "run")}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in replacements:
                    value = ast.Constant(replacements[name])
                    node.value = (ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[value], keywords=[])
                                  if name == "RUN_BASE" else value)
        ast.fix_missing_locations(tree)

        def fake_run(command, **kwargs):
            self.commands.append(command)
            if command[:2] == [sys.executable, "-c"]:
                compile(command[2], "GPU substitute probe", "exec")
            else:
                self.assertEqual(command, ["nvidia-smi"])
            return SimpleNamespace(returncode=0)

        scope = {}
        with patch("subprocess.run", side_effect=fake_run), \
                patch.dict(sys.modules, {"google.colab": None}), redirect_stdout(StringIO()):
            exec(compile(tree, "composed workspace cell", "exec"), scope)
        self.assertEqual(self.commands, [], "Workspace setup must defer GPU actions until a missing arm is known.")
        return scope

    def recovery_scope(self):
        scope = self.execute_workspace()
        functions = ast.Module(body=[node for node in ast.parse(self.cells[8]).body
                                     if isinstance(node, ast.FunctionDef)], type_ignores=[])
        specs = {role: {"candidate_manifest": "data/candidates/composed-fronts-2026-11-12/manifest.json",
                       "candidate_manifest_sha256": "1" * 64,
                       "candidate_id": "composed-fronts-2026-11-12", "arm": role,
                       **recipe_identity(recipe)}
                 for role, recipe in scope["ROLE_RECIPES"].items()}
        context = {"sha256": scope["PROTOCOL_SHA256"],
                   "delta_train_image_ids": ["new_000023", "new_000031", "new_000045"],
                   "document": {"roles": specs, "experiment_id": scope["EXPERIMENT_ID"],
                                "initial_checkpoint_sha256": "d" * 64,
                                "evaluation": {"primary_reference": "data/references/visible-front-adjudication-2026-11-12/manifest.json",
                                    "primary_reference_sha256": "e" * 64,
                                    "all_object_denominator": 663,
                                    "locatable_endpoint_denominator": 1227,
                                    "both_corner_denominator": 584}}}
        scope.update(CONTEXT=context, PREFLIGHT={}, get_recipe=get_recipe, recipe_identity=recipe_identity,
                     source_identity=lambda root: {"revision": "a" * 40, "dirty": False},
                     COMPLETED={}, HOST_RSS_LIMIT_MIB=12000,
                     run=lambda command: self.commands.append(command),
                     psutil=SimpleNamespace(NoSuchProcess=RuntimeError), time=time)
        exec(compile(functions, "composed recovery functions", "exec"), scope)
        return scope

    def trial_fixture(self, scope):
        folder = scope["RUN_BASE"] / "resource-trial"
        folder.mkdir()
        spec = scope["CONTEXT"]["document"]["roles"]["candidate"]
        write_json(folder / "staging-manifest.json", {"source": {key: spec[key] for key in
                   ("candidate_manifest", "candidate_manifest_sha256", "candidate_id")}})
        (folder / "train.log").write_text("Synthetic trial log; no GPU execution.\n")
        receipt = {"status": "trial-complete", "purpose": "resource-trial", "arm": "candidate",
            "source": scope["source_identity"](scope["REPO"]),
            "recipe_id": spec["recipe_id"], "recipe_sha256": spec["recipe_sha256"],
            "initial_checkpoint_sha256": "d" * 64, "inputs_reverified_after_worker": True,
            "execution": {"exit_code": 0, "stop_reason": None}, "batches_completed": 4,
            "images_processed": 16, "batch_samples": [{"shape": [4, 3, 736, 1280]} for _ in range(4)],
            "training_settings": get_recipe(spec["recipe_id"]), "optimizer_steps": 1,
            "optimizer_parameter_change_observed": True,
            "log": {"path": "train.log", "sha256": scope["sha256"](folder / "train.log")},
            "runtime": {"device": "cuda:0", "device_name": "synthetic GPU", "backend": {"cudnn_version": 1}},
            "staging_manifest_sha256": scope["sha256"](folder / "staging-manifest.json")}
        write_json(folder / "training-receipt.json", receipt)
        return folder, receipt

    def test_code_cells_and_inline_preflight_compile_and_pin_state_is_explicit(self):
        self.assertEqual(len(self.cells), 7)
        for index, source in self.cells.items():
            compile(source, f"cell {index}", "exec")
        for index, name in ((4, "input_preflight"), (8, "probe")):
            tree = ast.parse(self.cells[index])
            value = next(ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
            compile(value, name, "exec")
        constants = {node.targets[0].id: ast.literal_eval(node.value)
                     for node in ast.parse(self.cells[2]).body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in ("SOURCE_REF", "PROTOCOL_SHA256", "PROTOCOL_PATH")}
        if constants["SOURCE_REF"] == "PENDING_IMPLEMENTATION_COMMIT":
            self.assertEqual(constants["PROTOCOL_SHA256"], "PENDING_PROTOCOL_SHA256")
        else:
            self.assertRegex(constants["SOURCE_REF"], r"^[0-9a-f]{40}$")
            self.assertEqual(constants["PROTOCOL_SHA256"],
                             hashlib.sha256((ROOT / constants["PROTOCOL_PATH"]).read_bytes()).hexdigest())

    def test_unfilled_pins_stop_before_gpu_and_workspace_actions(self):
        for revision, protocol in (("PENDING_IMPLEMENTATION_COMMIT", "b" * 64),
                                   ("a" * 40, "PENDING_PROTOCOL_SHA256")):
            with self.subTest(revision=revision), self.assertRaisesRegex(AssertionError, "pinned"):
                self.execute_workspace(revision, protocol)
        self.assertEqual(self.commands, [])
        self.assertFalse((self.base / "run").exists())

    def test_workspace_can_resume_only_matching_source_and_protocol(self):
        scope = self.execute_workspace()
        marker = scope["RUN_BASE"] / "preserve.txt"
        marker.write_text("Keep interrupted work.")
        self.execute_workspace()
        with self.assertRaisesRegex(AssertionError, "different pins"):
            self.execute_workspace(protocol="c" * 64)
        self.assertEqual(marker.read_text(), "Keep interrupted work.")

    def test_install_cell_constrains_existing_cuda_packages(self):
        scope = {"RUN_BASE": self.base, "sys": sys, "run": lambda cmd: self.commands.append(list(map(str, cmd)))}
        versions = {"torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130"}
        with patch("importlib.metadata.version", side_effect=versions.__getitem__), \
                patch.dict(sys.modules, {"psutil": SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=20 * 2**30))}), \
                redirect_stdout(StringIO()):
            exec(self.cells[6], scope)
        self.assertEqual((self.base / "torch-constraints.txt").read_text(),
                         "torch==2.14.1+cu130\ntorchvision==0.29.1+cu130\n")
        self.assertEqual(self.commands[0][-2:], ["ultralytics==8.4.182", "psutil"])
        self.assertEqual(scope["HOST_RSS_LIMIT_MIB"], 12000)
        self.assertFalse(any("cuda.is_available" in str(arg) or "get_device_name" in str(arg)
                             for command in self.commands for arg in command))

    def test_inline_preflight_reads_distinct_arm_counts_and_separate_evaluation_reference(self):
        scope = self.recovery_scope()
        inline = next(ast.literal_eval(node.value) for node in ast.parse(self.cells[4]).body
                      if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == "input_preflight")
        observed, evaluated = [], []
        spec = scope["CONTEXT"]["document"]["roles"]["candidate"]
        val = {"images": 20, "objects": 664, "visible_endpoints": 1227, "missing_endpoints": 101, "empty_images": 3}
        counts = {role: {"val": dict(val), "train": {
                    "images": 137, "objects": 3771 if role == "reference" else 3770,
                    "visible_endpoints": 6842, "missing_endpoints": 700 if role == "reference" else 698,
                    "empty_images": 13}} for role in scope["ROLES"]}
        evaluation_counts = {"images": 20, "objects": 663, "visible_endpoints": 1227,
                             "missing_endpoints": 99, "empty_images": 3}
        groups = {"0": 20, "1": 59, "2": 584}

        def state(root, candidate):
            observed.append(candidate)
            return {"records": [{}] * 157,
                    "source": {"initial_checkpoint_sha256": "d" * 64,
                               **{key: spec[key] for key in ("candidate_manifest", "candidate_manifest_sha256", "candidate_id")}},
                    "counts": counts}

        def verify(root, reference_dir, initial, api):
            evaluated.append(reference_dir)
            return {"counts": evaluation_counts, "objects_by_locatable_corners": groups}, {}, {}, {}, []

        modules = {
            "evaluate_pose": SimpleNamespace(preflight_dataset=lambda *a, **k: {"model_sha256": "d" * 64, "weights_verified": True},
                                             file_sha256=lambda path: "e" * 64),
            "stage_pose_training": SimpleNamespace(source_state=state),
            "package_composed_results": SimpleNamespace(validate_protocol=lambda *a: scope["CONTEXT"]),
            "rescore_reference_amendment": SimpleNamespace(load_api=lambda *a: (None, {}), verify_amendment=verify),
            "replay_corner_reference": SimpleNamespace(reference_counts=lambda gold: (evaluation_counts, groups)),
        }
        executed = {}
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", ["preflight", scope["PROTOCOL_PATH"], scope["PROTOCOL_SHA256"]]), \
                patch.object(sys, "path", sys.path[:]), patch("pathlib.Path.cwd", return_value=scope["REPO"]), redirect_stdout(StringIO()):
            exec(inline, executed)
        self.assertEqual(observed, [scope["REPO"] / spec["candidate_manifest"]])
        self.assertEqual(executed["counts"]["reference"]["train"]["objects"], 3771)
        self.assertEqual(executed["counts"]["candidate"]["train"]["objects"], 3770)
        self.assertEqual(executed["counts"]["reference"]["val"]["objects"], 664)
        self.assertEqual(executed["evaluation_counts"]["objects"], 663)
        self.assertEqual(evaluated, [scope["REPO"] / "data/references/visible-front-adjudication-2026-11-12"])
        # A stale denominator must fail before any training command is assembled.
        scope["CONTEXT"]["document"]["evaluation"]["locatable_endpoint_denominator"] = 1228
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", ["preflight", scope["PROTOCOL_PATH"], scope["PROTOCOL_SHA256"]]), \
                patch.object(sys, "path", sys.path[:]), patch("pathlib.Path.cwd", return_value=scope["REPO"]), \
                redirect_stdout(StringIO()), self.assertRaises(AssertionError):
            exec(inline, {})

    def test_drive_backup_restore_verifies_protocol_identity_and_bytes(self):
        scope = self.recovery_scope()
        folder, _ = self.trial_fixture(scope)
        scope["DRIVE_ROOT"] = self.base / "drive"
        scope["DRIVE_ROOT"].mkdir()
        with redirect_stdout(StringIO()):
            scope["save_to_drive"](folder)
            marker_path = next((scope["DRIVE_ROOT"] / "resource-trials").glob("*.complete"))
            relative = json.loads(marker_path.read_text())["relative_path"]
            restored = scope["restore_from_drive"](relative)
        self.assertEqual((restored / "train.log").read_bytes(), (folder / "train.log").read_bytes())
        shutil.rmtree(restored)
        backup = scope["DRIVE_ROOT"] / relative
        original_log = (backup / "train.log").read_bytes()
        (backup / "train.log").write_bytes(b"Corrupt copy")
        with self.assertRaisesRegex(AssertionError, "hash mismatch"):
            scope["restore_from_drive"](relative)
        self.assertFalse(restored.exists())
        (backup / "train.log").write_bytes(original_log)
        marker = json.loads(marker_path.read_text())
        marker["protocol_sha256"] = "0" * 64
        write_json(marker_path, marker)
        with self.assertRaisesRegex(AssertionError, "different source/protocol"):
            scope["restore_from_drive"](relative)

    def test_trial_receipt_binds_treatment_recipe_candidate_source_and_schedule(self):
        scope = self.recovery_scope()
        folder, original = self.trial_fixture(scope)
        self.assertEqual(scope["assert_finished"](folder, "candidate", trial=True)["batches_completed"], 4)
        for name, mutate in (
            ("recipe", lambda r: r.update(recipe_id="reviewed-endpoints-seed46-v1")),
            ("source", lambda r: r["source"].update(revision="f" * 40)),
            ("batch count", lambda r: r.update(batches_completed=3)),
            ("schedule", lambda r: r["training_settings"].update(epochs=1)),
        ):
            with self.subTest(name=name):
                receipt = copy.deepcopy(original)
                mutate(receipt)
                write_json(folder / "training-receipt.json", receipt)
                with self.assertRaises(AssertionError):
                    scope["assert_finished"](folder, "candidate", trial=True)
        write_json(folder / "training-receipt.json", original)
        stage = json.loads((folder / "staging-manifest.json").read_text())
        stage["source"]["candidate_id"] = "reference"
        write_json(folder / "staging-manifest.json", stage)
        with self.assertRaises(AssertionError):
            scope["assert_finished"](folder, "candidate", trial=True)

    def test_completed_receipt_cannot_be_reused_under_the_other_runner_arm(self):
        scope = self.recovery_scope()
        scope["collect_composed_run"] = lambda *args: {"receipt": {
            "arm": "candidate", "source": scope["source_identity"](scope["REPO"])}}
        with self.assertRaisesRegex(AssertionError, "wrong runner arm"):
            scope["assert_finished"](scope["RUN_BASE"] / "reference", "reference")

    def test_partial_local_trial_is_preserved_and_control_trial_is_forbidden(self):
        scope = self.recovery_scope()
        folder = scope["RUN_BASE"] / "resource-trial"
        folder.mkdir()
        (folder / "partial.log").write_text("Interrupted trial")
        with self.assertRaisesRegex(FileExistsError, "Preserve incomplete"):
            scope["train_one"]("candidate", trial=True)
        self.assertTrue((folder / "partial.log").exists())
        with self.assertRaisesRegex(AssertionError, "candidate labels only"):
            scope["train_one"]("reference", trial=True)

    def test_train_commands_select_the_correct_recipe_and_candidate_without_initializing_from_a_trial(self):
        scope = self.recovery_scope()
        scope["HOST_RSS_LIMIT_MIB"] = 10000
        scope["assert_finished"] = lambda output, role, **kw: {"role": role}
        scope["save_to_drive"] = lambda output: None
        commands = []

        def popen(command, **kwargs):
            commands.append(command)
            self.assertEqual(kwargs["cwd"], scope["REPO"])
            return SimpleNamespace(pid=123, poll=lambda: 0, communicate=lambda: ("", None), returncode=0)

        with patch("subprocess.Popen", side_effect=popen), redirect_stdout(StringIO()):
            scope["train_one"]("reference")
            scope["train_one"]("candidate")
            scope["train_one"]("candidate", trial=True)
        for command, role, trial in zip(commands, ("reference", "candidate", "candidate"), (False, False, True)):
            self.assertEqual(command[command.index("--arm") + 1], role)
            self.assertEqual(command[command.index("--recipe") + 1], scope["ROLE_RECIPES"][role])
            self.assertEqual(command[command.index("--candidate") + 1], scope["CONTEXT"]["document"]["roles"][role]["candidate_manifest"])
            self.assertEqual(command[command.index("--trial-batches") + 1], "4")
            self.assertEqual(command[command.index("--max-seconds") + 1], "600" if trial else "21600")
            self.assertIn("--trial" if trial else "--train", command)
            self.assertNotIn("--resume", command)
            self.assertNotIn("--weights", command)

    def execute_recovery_tail(self, scope):
        tree = ast.parse(self.cells[8])
        start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "COMPLETED")
        with redirect_stdout(StringIO()):
            exec(compile(ast.Module(body=tree.body[start:], type_ignores=[]), "actual recovery flow", "exec"), scope)

    def test_both_completed_arms_skip_probe_trial_and_training(self):
        scope = self.recovery_scope()
        for role in scope["ROLES"]:
            (scope["RUN_BASE"] / role).mkdir()
        scope.update(restore_from_drive=lambda role: None, assert_finished=lambda *a: {"runtime": {}},
                     collect_composed_run=lambda *a: {"receipt": {"runtime": {}}},
                     check_composed_pair=Mock(), recover_trial=Mock(side_effect=AssertionError("must skip trial")))
        with patch("subprocess.check_output", side_effect=AssertionError("must skip GPU probe")):
            self.execute_recovery_tail(scope)
        scope["check_composed_pair"].assert_called_once()
        scope["recover_trial"].assert_not_called()
        self.assertIsNone(scope["TRIAL_RUNTIME"])

    def test_both_completed_training_cell_skips_process_creation(self):
        scope = self.recovery_scope()
        for role in scope["ROLES"]:
            (scope["RUN_BASE"] / role).mkdir()
        scope["TRIAL_RUNTIME"] = None
        scope["assert_finished"] = lambda output, role, **kwargs: {
            "arm": role, "runtime": {"device": "cuda:0"}, "checkpoint": {"sha256": "f" * 64}}
        scope["save_to_drive"] = Mock()
        with patch("subprocess.Popen", side_effect=AssertionError("must not restart completed training")), redirect_stdout(StringIO()):
            exec(self.cells[10], scope)
        self.assertEqual(scope["save_to_drive"].call_count, 2)

    def test_runtime_mismatch_stops_before_trial_or_missing_full_arm(self):
        scope = self.recovery_scope()
        (scope["RUN_BASE"] / "reference").mkdir()
        scope.update(restore_from_drive=lambda role: None,
                     assert_finished=lambda *a: {"runtime": {"device_name": "old GPU", "backend": {"cudnn_version": 1}}},
                     recover_trial=Mock(side_effect=AssertionError("must not start trial")))
        with patch("subprocess.check_output", return_value=json.dumps({"device_name": "different GPU", "cudnn_version": 1})), \
                self.assertRaisesRegex(AssertionError, "Current runtime differs"):
            self.execute_recovery_tail(scope)
        scope["recover_trial"].assert_not_called()

    def test_saved_trial_search_skips_bad_copy_and_reuses_compatible_copy(self):
        scope = self.recovery_scope()
        scope["DRIVE_ROOT"] = self.base / "drive"
        folder = scope["DRIVE_ROOT"] / "resource-trials"
        folder.mkdir(parents=True)
        (folder / "old-good.complete").write_text("fixture")
        (folder / "new-bad.complete").write_text("fixture")
        runtime = {"device_name": "same GPU", "backend": {"cudnn_version": 1}}
        scope["COMPLETED"] = {"reference": {"runtime": runtime}}
        scope["restore_from_drive"] = lambda relative: Path(relative)

        def finished(path, role, **kwargs):
            self.assertEqual(role, "candidate")
            self.assertTrue(kwargs["trial"])
            if "new-bad" in str(path):
                raise ValueError("incompatible trial")
            return {"runtime": runtime}

        scope["assert_finished"] = finished
        scope["train_one"] = Mock(side_effect=AssertionError("must reuse trial"))
        with redirect_stdout(StringIO()):
            restored, _ = scope["recover_trial"]({"device_name": "same GPU", "cudnn_version": 1})
        self.assertEqual(restored, Path("resource-trials/old-good"))
        scope["train_one"].assert_not_called()

    def test_control_runtime_mismatch_prevents_starting_treatment(self):
        scope = self.recovery_scope()
        calls = []
        scope["TRIAL_RUNTIME"] = {"device_name": "trial GPU"}

        def train(role):
            calls.append(role)
            return Path(role), {"runtime": {"device_name": "changed GPU"}}

        scope["train_one"] = train
        with self.assertRaisesRegex(AssertionError, "Reference runtime differs"):
            exec(self.cells[10], scope)
        self.assertEqual(calls, ["reference"])

    def test_package_command_preserves_both_roles_and_reverifies_existing_zip(self):
        scope = self.recovery_scope()
        scope.update(REFERENCE_RUN=scope["RUN_BASE"] / "reference",
                     CANDIDATE_RUN=scope["RUN_BASE"] / "candidate",
                     restore_from_drive=lambda name: None, save_to_drive=lambda path: None)
        commands = []

        def run(command, **kwargs):
            commands.append(list(map(str, command)))
            result = Path(command[command.index("--output") + 1])
            if not result.exists():
                result.write_bytes(b"Synthetic archive fixture; no model")

        scope["run"] = run
        with redirect_stdout(StringIO()):
            exec(self.cells[12], scope)
            exec(self.cells[12], scope)
        self.assertNotIn("--verify-existing", commands[0])
        self.assertIn("--verify-existing", commands[1])
        self.assertIn("--reference", commands[0])
        self.assertIn("--candidate", commands[0])
        self.assertEqual(scope["RESULTS_ZIP"].name, "baomai-composed-fronts-seed46-results.zip")


if __name__ == "__main__":
    unittest.main(verbosity=2)
