#!/usr/bin/env python3
"""Check fixed-grid replay and the conservative selection gate without model inference."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import experiment_pose_postprocess as experiment


class PostprocessChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = [{"id": f"image_{index:02}", "image": f"images/val/image_{index:02}.png",
                         "label": f"labels/val/image_{index:02}.txt", "image_sha256": "a" * 64,
                         "label_sha256": "b" * 64, "width": 100, "height": 100,
                         "gold": [{"class_id": 0, "bbox_xyxy": [10, 10, 30, 30],
                                   "keypoints": [[10, 30, 2], [30, 30, 2]]}]} for index in range(20)]
        self.preflight = {**experiment.IDENTITY, "records": self.records}
        self.header = {"record_type": "run", "schema": experiment.SCHEMA, "schema_version": 1,
                       "plan": copy.deepcopy(experiment.PLAN), "inference_base": copy.deepcopy(experiment.INFERENCE),
                       **experiment.IDENTITY, "images": experiment.reference_images(self.preflight),
                       "source": experiment.source_identity(), "runtime": {"backend_end2end": False,
                       "backend_fp16": False, "loaded_kpt_shape": [2, 3], "ultralytics": "8.4.182"},
                       "warmup": "one explicit zero-image prediction excluded from all timings"}
        self.items = []
        for nms in experiment.PLAN["nms_ious"]:
            for threshold in experiment.PLAN["box_confidences"]:
                for record in self.records:
                    predicted = {**copy.deepcopy(record["gold"][0]), "confidence": 0.9}
                    predicted["keypoints"] = [[10, 30, 0.9], [30, 30, 0.9]]
                    self.items.append({"record_type": "prediction", "nms_iou": nms, "threshold": threshold,
                                       "image_id": record["id"], "inference": {**experiment.INFERENCE, "iou": nms, "conf": threshold},
                                       **{key: record[key] for key in ("image_sha256", "label_sha256", "width", "height")},
                                       "predictions": [predicted], "stage_ms": {"preprocess": 1, "inference": 1, "postprocess": 1},
                                       "predict_call_wall_ms": 3})
        self.footer = {"record_type": "completion", "status": "complete", "prediction_records": 240}

    def write(self, header=None, items=None, footer=None):
        path = self.root / "predictions.jsonl"
        rows = [self.header if header is None else header,
                *(self.items if items is None else items), self.footer if footer is None else footer]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_full_grid_replay_reconstructs_counts_quality_and_match_ids(self):
        header, items, footer = experiment.read_replay(self.write(), self.preflight)
        summary = experiment.reports(self.preflight, items, header, self.root)
        self.assertEqual(len(items), 240)
        self.assertEqual(len(summary["metrics"]), 12)
        for cell in summary["metrics"]:
            self.assertEqual((cell["tp"], cell["fn"], cell["fp"], cell["fn_plus_fp"]), (20, 0, 0, 0))
            self.assertEqual((cell["precision"], cell["recall"], cell["f1"]), (1, 1, 1))
            self.assertEqual(cell["quality"]["both_visible_objects"], 20)
            self.assertEqual([r["both_visible_objects_within_radius"] for r in cell["quality"]["radii"]], [20, 20, 20])
        details = json.loads((self.root / "per_image.json").read_text())
        self.assertEqual(details[0]["matches"], [{"gold_index": 0, "prediction_index": 0, "iou": 1.0}])
        self.assertFalse(any(row["passes_declared_gate"] for row in summary["candidate_comparisons"]))

    def test_missing_duplicate_outside_and_unfinished_pairs_fail_closed(self):
        duplicate = copy.deepcopy(self.items)
        duplicate[-1] = duplicate[0]
        outside = copy.deepcopy(self.items)
        outside[-1]["nms_iou"] = 0.55
        for items in (self.items[:-1], duplicate, outside):
            with self.subTest(kind=len(items)), self.assertRaises(ValueError):
                experiment.read_replay(self.write(items=items), self.preflight)
        with self.assertRaises(ValueError):
            experiment.read_replay(self.write(footer={"record_type": "completion", "status": "failed", "prediction_records": 240}), self.preflight)

    def test_plan_identity_source_and_runtime_tampering_are_rejected(self):
        changes = [("plan", {**experiment.PLAN, "matching_iou": 0.4}),
                   ("inference_base", {**experiment.INFERENCE, "nms": False}),
                   ("model_sha256", "c" * 64), ("images", self.header["images"][:-1]),
                   ("source", {**self.header["source"], "files_sha256": {}}),
                   ("runtime", {**self.header["runtime"], "backend_fp16": True})]
        for key, value in changes:
            header = {**self.header, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment.read_replay(self.write(header=header), self.preflight)

    def test_prediction_reference_threshold_class_and_geometry_are_checked(self):
        mutations = [lambda item: item.update(width=99), lambda item: item.update(image_sha256="d" * 64),
                     lambda item: item["predictions"][0].update(confidence=item["threshold"]),
                     lambda item: item["predictions"][0].update(class_id=1),
                     lambda item: item["predictions"][0].update(bbox_xyxy=[10, 10, 10, 20]),
                     lambda item: item["stage_ms"].update(inference=-1)]
        for change in mutations:
            items = copy.deepcopy(self.items)
            change(items[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                experiment.read_replay(self.write(items=items), self.preflight)

    def test_equal_cost_improvement_cannot_hide_a_missed_object_regression(self):
        baseline = dict(zip(experiment.GATE_FIELDS, (39, 63, 43, 1.6, 5.1)))
        candidate = {**baseline, "fn": 40, "fp": 59}
        self.assertLess(candidate["fn"] + candidate["fp"], baseline["fn"] + baseline["fp"])
        self.assertFalse(experiment.gate_result(candidate, baseline)["passes_declared_gate"])
        self.assertTrue(experiment.gate_result({**baseline, "fp": 62}, baseline)["passes_declared_gate"])
        self.assertFalse(experiment.gate_result(baseline, baseline)["passes_declared_gate"])
        self.assertFalse(experiment.gate_result({**baseline, "fp": 62, "conditional_p95_error_px": None}, baseline)["passes_declared_gate"])

    def test_import_and_replay_do_not_require_machine_learning_packages(self):
        program = '''
import importlib.abc, sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'numpy', 'ultralytics', 'PIL', 'cv2'}:
            raise ImportError('machine-learning dependency forbidden: ' + fullname)
sys.meta_path.insert(0, NoML())
import experiment_pose_postprocess as experiment
import json
from pathlib import Path
fixture = json.loads(Path(sys.argv[2]).read_text())
assert len(experiment.read_replay(Path(sys.argv[1]), fixture)[1]) == 240
'''
        fixture = self.root / "fixture.json"
        fixture.write_text(json.dumps(self.preflight))
        result = subprocess.run([sys.executable, "-c", program, str(self.write()), str(fixture)],
                                cwd=Path(experiment.__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_inference_failure_leaves_explicit_failure_status(self):
        output = self.root / "failed-run"
        with patch.object(experiment, "fixed_preflight", return_value=self.preflight), \
                patch.object(experiment, "infer", side_effect=RuntimeError("backend failed")), \
                contextlib.redirect_stderr(io.StringIO()):
            code = experiment.main(["--output", str(output)])
        self.assertEqual(code, 1)
        status = json.loads((output / "run_status.json").read_text())
        self.assertEqual((status["status"], status["exit_code"], status["error_type"]), ("failed", 1, "RuntimeError"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
