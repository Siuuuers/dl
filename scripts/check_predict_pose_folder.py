#!/usr/bin/env python3
"""Focused folder-runner checks with temporary images and a substituted model adapter."""

import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageOps

import predict_pose_folder as runner
from check_pose_checkpoint import bundle_fixture
from pose_checkpoint import ANNOTATION_CONFIRMATION_RECIPE_ID
from train_pose_comparison import descriptor


def detection(points=None):
    return {"class_id": 0, "bbox_xyxy": [1.0, 1.0, 11.0, 9.0], "confidence": 0.9,
            "keypoints": points or [[2.0, 8.0, 0.8], [10.0, 8.0, 0.9]]}


def fake_prediction(rgb):
    predictions = [] if rgb.getpixel((0, 0)) == (0, 0, 0) else [detection()]
    return {"predictions": predictions, "stage_ms": {"inference": 1.0}, "predict_call_wall_ms": 2.0}


class FolderChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "results"
        self.identity = {"weights": str(self.root / "model.pt"), "epoch": 173,
                         "model_sha256": "e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e",
                         "model_card_sha256": "a" * 64}

    def image(self, relative, color="white"):
        path = self.inputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), color).save(path)
        return path

    def invoke(self, predictor=fake_prediction, extra=(), output=None, setup_error=None):
        with patch.object(runner, "verify_baseline", return_value=self.identity), \
                patch.object(runner, "prepare_predictor", return_value=(predictor, {"test_adapter": True}),
                             side_effect=setup_error) as prepare, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = runner.main(["--input", str(self.inputs), "--output", str(output or self.output), *extra])
        return result, prepare

    def records(self):
        return [json.loads(line) for line in (self.output / "predictions.jsonl").read_text().splitlines()]

    def summary(self):
        return json.loads((self.output / "run_status.json").read_text())

    def completed_bundle(self):
        root, run, _, receipt, staging, manifest_path = bundle_fixture(
            self.root / "bundle-fixture", recipe_id=ANNOTATION_CONFIRMATION_RECIPE_ID)
        card_path = root / "metadata/model_card.json"
        card = json.loads(card_path.read_text())
        card["epoch"] = 173
        card_path.write_text(json.dumps(card))
        card_sha256 = runner.file_sha256(card_path)
        staging["source"]["model_card_sha256"] = card_sha256
        (run / "staging-manifest.json").write_text(json.dumps(staging))
        receipt.update(model_card_sha256=card_sha256,
                       staging_manifest_sha256=runner.file_sha256(run / "staging-manifest.json"))
        (run / "training-receipt.json").write_text(json.dumps(receipt))
        manifest = json.loads(manifest_path.read_text())
        manifest.update(training_receipt=descriptor(run / "training-receipt.json", run),
                        staging_manifest=descriptor(run / "staging-manifest.json", run))
        manifest_path.write_text(json.dumps(manifest))
        for directory in ("data/images", "data/labels", "weights"):
            shutil.rmtree(root / directory)
        return root, run, manifest_path, card

    def invoke_bundle(self, root, manifest, card):
        with patch.object(runner, "ROOT", root), \
                patch.object(runner, "BASELINE_SHA256", card["sha256"]), \
                patch.object(runner, "BASELINE_BYTES", card["size_bytes"]), \
                patch.object(runner, "verify_baseline", side_effect=AssertionError("unexpected baseline fallback")) as baseline, \
                patch.object(runner, "_source_identity", return_value={"files_sha256": {}}), \
                patch.object(runner, "prepare_predictor", return_value=(fake_prediction, {"test_adapter": True})) as prepare, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = runner.main(["--input", str(self.inputs), "--output", str(self.output),
                                "--checkpoint-manifest", str(manifest)])
        return code, prepare, baseline

    def test_completed_bundle_uses_explicit_weights_without_reference_payloads(self):
        self.image("dense.png")
        self.image("empty.png", "black")
        root, run, manifest_path, card = self.completed_bundle()
        code, prepare, baseline = self.invoke_bundle(root, manifest_path, card)
        self.assertEqual(code, 0)
        baseline.assert_not_called()
        self.assertEqual(prepare.call_args.args[0], str(run / "fit/weights/last.pt"))
        self.assertEqual(prepare.call_args.kwargs, {"conf": 0.25, "nms_iou": 0.70})
        summary = self.summary()
        self.assertEqual(summary["model_sha256"], runner.file_sha256(run / "fit/weights/last.pt"))
        self.assertEqual(summary["reference_model_sha256"], card["sha256"])
        self.assertEqual(summary["reference_model_card_sha256"], runner.file_sha256(root / "metadata/model_card.json"))
        self.assertEqual(summary["reference_manifest_sha256"], runner.file_sha256(root / "data/manifest.json"))
        self.assertNotIn("epoch", summary)
        identity = summary["checkpoint_identity"]
        self.assertEqual((identity["arm"], identity["completed_epochs"], identity["selection"], identity["recipe_id"]),
                         ("candidate", 30, "final_epoch", ANNOTATION_CONFIRMATION_RECIPE_ID))
        self.assertEqual(identity["manifest_sha256"], runner.file_sha256(manifest_path))
        self.assertEqual(identity["training_receipt_sha256"], runner.file_sha256(run / "training-receipt.json"))
        self.assertEqual(identity["staging_manifest_sha256"], runner.file_sha256(run / "staging-manifest.json"))
        self.assertEqual(summary["checkpoint_verification"], {
            "bundle_bytes_and_recorded_completion": True, "repository_metadata": True,
            "reference_image_and_label_bytes": False, "initial_checkpoint_bytes": False})
        self.assertIs(summary["weights_verified"], True)
        self.assertIs(summary["ground_truth_available"], False)
        self.assertEqual([r["prediction_count"] for r in self.records()], [1, 0])
        for directory in ("data/images", "data/labels", "weights"):
            self.assertFalse((root / directory).exists())

    def test_invalid_completed_bundle_fails_before_inference_without_fallback(self):
        self.image("frame.png")
        root, run, manifest_path, card = self.completed_bundle()
        weights, receipt_path = run / "fit/weights/last.pt", run / "training-receipt.json"
        originals = {path: path.read_bytes() for path in (weights, receipt_path, manifest_path)}
        for failure in ("changed_weights", "incomplete_receipt", "wrong_pose_contract"):
            with self.subTest(failure=failure):
                if failure == "changed_weights":
                    weights.write_bytes(originals[weights] + b"changed")
                elif failure == "incomplete_receipt":
                    receipt = json.loads(originals[receipt_path])
                    receipt["status"] = "running"
                    receipt_path.write_text(json.dumps(receipt))
                    manifest = json.loads(originals[manifest_path])
                    manifest["training_receipt"] = descriptor(receipt_path, run)
                    manifest_path.write_text(json.dumps(manifest))
                else:
                    manifest = json.loads(originals[manifest_path])
                    manifest["kpt_shape"] = [5, 3]
                    manifest_path.write_text(json.dumps(manifest))
                try:
                    code, prepare, baseline = self.invoke_bundle(root, manifest_path, card)
                    self.assertNotEqual(code, 0)
                    prepare.assert_not_called()
                    baseline.assert_not_called()
                    self.assertFalse(self.output.exists())
                finally:
                    for path, contents in originals.items():
                        path.write_bytes(contents)

    def test_label_free_nested_files_zero_predictions_and_no_source_changes(self):
        self.image("a/frame.png", "black")
        self.image("a/frame.jpg")
        self.image("b/frame.png")
        (self.inputs / "a/frame.txt").write_text("This is deliberately not a valid annotation.\n")
        hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.inputs.rglob("*") if path.is_file()}
        code, prepare = self.invoke()
        self.assertEqual(code, 0)
        prepare.assert_called_once()
        self.assertEqual(prepare.call_args.kwargs["nms_iou"], 0.70)
        records = self.records()
        self.assertEqual([r["image"] for r in records], ["a/frame.jpg", "a/frame.png", "b/frame.png"])
        self.assertEqual([r["prediction_count"] for r in records], [1, 0, 1])
        self.assertEqual({r["overlay"] for r in records},
                         {"overlays/a/frame.jpg.prediction.png", "overlays/a/frame.png.prediction.png",
                          "overlays/b/frame.png.prediction.png"})
        for record in records:
            with Image.open(self.output / record["overlay"]) as overlay:
                self.assertEqual(overlay.size, (12, 10))
            self.assertFalse({"fn", "fp", "precision", "recall", "accuracy"} & record.keys())
        summary = self.summary()
        self.assertEqual((summary["status"], summary["selected_images"], summary["succeeded_images"],
                          summary["failed_images"], summary["prediction_count"]), ("complete", 3, 3, 0, 2))
        self.assertIs(summary["ground_truth_available"], False)
        self.assertEqual(summary["inference"]["iou"], 0.70)
        with (self.output / "summary.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)
        self.assertEqual(hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in hashes})

    def test_corrupt_image_is_accounted_and_remaining_images_continue(self):
        self.image("01-good.png")
        (self.inputs / "02-bad.PNG").write_bytes(b"not an image")
        self.image("03-good.jpeg", "black")
        code, _ = self.invoke()
        self.assertNotEqual(code, 0)
        records = self.records()
        self.assertEqual([r["status"] for r in records], ["complete", "failed", "complete"])
        self.assertEqual(records[1]["image"], "02-bad.PNG")
        self.assertIsNone(records[1]["prediction_count"])
        summary = self.summary()
        self.assertNotEqual(summary["status"], "complete")
        self.assertEqual((summary["selected_images"], summary["succeeded_images"], summary["failed_images"]), (3, 2, 1))

    def test_all_unreadable_never_loads_model(self):
        (self.inputs / "bad.png").write_bytes(b"bad")
        code, prepare = self.invoke()
        self.assertNotEqual(code, 0)
        prepare.assert_not_called()
        self.assertEqual(self.summary()["failed_images"], 1)
        self.assertEqual(len(self.records()), 1)

    def test_backend_failure_accounts_for_unprocessed_images(self):
        for name in ("1.png", "2.png", "3.png"):
            self.image(name)
        calls = []

        def fail_second(rgb):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("simulated backend failure")
            return fake_prediction(rgb)

        code, _ = self.invoke(predictor=fail_second)
        self.assertNotEqual(code, 0)
        self.assertEqual(len(calls), 2)
        records = self.records()
        self.assertEqual([r["status"] for r in records], ["complete", "failed", "failed"])
        self.assertIs(records[-1]["attempted"], False)
        summary = self.summary()
        self.assertEqual((summary["selected_images"], summary["succeeded_images"], summary["failed_images"],
                          summary["unprocessed_images"]), (3, 1, 2, 1))

    def test_model_setup_failure_is_not_an_empty_success(self):
        self.image("1.png")
        self.image("2.png")
        code, _ = self.invoke(setup_error=RuntimeError("model backend unavailable"))
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.records()), 2)
        self.assertEqual(self.summary()["succeeded_images"], 0)
        self.assertNotEqual(self.summary()["status"], "complete")

    def test_existing_output_and_overlapping_trees_are_refused(self):
        self.image("frame.png")
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep me")
        dangling = self.root / "dangling-output"
        dangling.symlink_to(self.root / "missing-output")
        for target in (self.output, self.inputs / "new-output", self.root, self.inputs, dangling):
            with self.subTest(target=target):
                code, prepare = self.invoke(output=target)
                self.assertNotEqual(code, 0)
                prepare.assert_not_called()
        self.assertEqual(sentinel.read_text(), "keep me")
        self.assertFalse((self.inputs / "new-output").exists())
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.assertTrue(dangling.is_symlink())

    def test_empty_input_and_invalid_confidence_are_rejected(self):
        code, prepare = self.invoke()
        self.assertNotEqual(code, 0)
        prepare.assert_not_called()
        self.assertFalse(self.output.exists())
        self.image("frame.png")
        for extra in (("--conf", "nan"), ("--endpoint-conf", "1.1"), ("--threads", "0")):
            with self.subTest(extra=extra):
                try:
                    code, prepare = self.invoke(extra=extra)
                except SystemExit as error:
                    code = error.code
                self.assertNotEqual(code, 0)
                self.assertFalse(self.output.exists())

    def test_nondefault_nms_is_forwarded_and_recorded_without_changing_baseline(self):
        self.image("frame.png")
        original = copy.deepcopy(runner.INFERENCE)
        code, prepare = self.invoke(extra=("--nms-iou", "0.68"))
        self.assertEqual(code, 0)
        self.assertEqual(prepare.call_args.kwargs["nms_iou"], 0.68)
        self.assertEqual(self.summary()["inference"], {**original, "iou": 0.68, "conf": 0.25})
        self.assertEqual(runner.INFERENCE, original)

    def test_predictor_uses_selected_nms_for_warmup_and_each_image(self):
        import numpy as np
        rgb = Image.fromarray(np.full((10, 12, 3), 255, dtype=np.uint8))
        original = copy.deepcopy(runner.INFERENCE)
        for kwargs, expected in (({}, 0.70), ({"nms_iou": 0.68}, 0.68)):
            with self.subTest(nms_iou=expected):
                result = SimpleNamespace(orig_shape=(10, 12), speed={"inference": 1.0})
                model = SimpleNamespace(predict=Mock(return_value=[result]), predictor=SimpleNamespace(
                    model=SimpleNamespace(end2end=False, fp16=False, device="cpu")))
                torch = SimpleNamespace(set_num_threads=Mock(), get_num_threads=Mock(return_value=4))
                with patch.dict(sys.modules, {"torch": torch}), \
                        patch.object(runner, "_runtime_and_model", return_value=(model, {})), \
                        patch.object(runner, "_prediction_records", return_value=[]):
                    predict, _ = runner.prepare_predictor("model.pt", "cpu", 4, rgb, **kwargs)
                    predict(rgb)
                    predict(rgb)
                self.assertEqual(model.predict.call_count, 3)
                for call in model.predict.call_args_list:
                    self.assertEqual({key: call.kwargs[key] for key in original}, {**original, "iou": expected})
                self.assertEqual(runner.INFERENCE, original)

    def test_invalid_nms_is_rejected_before_creating_output(self):
        self.image("frame.png")
        for value in ("nan", "inf", "-0.1", "1.1"):
            with self.subTest(value=value):
                code, prepare = self.invoke(extra=("--nms-iou", value))
                self.assertNotEqual(code, 0)
                prepare.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_display_flags_preserve_raw_coordinates_confidence_and_slot_order(self):
        raw = [detection([[0.0, 0.0, 0.5], [12.0, 4.0, 0.9]]),
               detection([[11.0, 9.0, 0.49], [None, 3.0, 0.95]]),
               detection([[10.0, 8.0, 0.8], [2.0, 8.0, 0.9]])]
        original = copy.deepcopy(raw)
        described = runner.describe_predictions(raw, 12, 10, 0.5)
        self.assertEqual(raw, original)
        self.assertEqual([r["keypoints"] for r in described], [r["keypoints"] for r in original])
        self.assertEqual([[s["displayed"] for s in r["endpoint_status"]] for r in described],
                         [[True, False], [False, False], [True, True]])
        self.assertTrue(described[0]["endpoint_status"][1]["passes_display_threshold"])
        self.assertFalse(described[0]["endpoint_status"][1]["in_image"])
        self.assertFalse(described[1]["endpoint_status"][0]["passes_display_threshold"])
        self.assertFalse(any("visibility" in item or "v" in item for r in described for item in r["endpoint_status"]))

    def test_exif_rotation_is_applied_once_and_original_bytes_are_identified(self):
        image = Image.new("RGB", (12, 8), "white")
        image.putpixel((0, 0), (255, 0, 0))
        exif = Image.Exif()
        exif[274] = 6
        source = self.inputs / "rotated.jpg"
        image.save(source, exif=exif)
        decoded, metadata = runner.decode_image(source)
        with Image.open(source) as original:
            expected = ImageOps.exif_transpose(original).convert("RGB")
        self.assertEqual(decoded.size, (8, 12))
        self.assertEqual(decoded.tobytes(), expected.tobytes())
        self.assertEqual((metadata["source_width"], metadata["source_height"], metadata["exif_orientation"]), (12, 8, 6))
        self.assertEqual((metadata["width"], metadata["height"]), (8, 12))
        self.assertEqual(metadata["image_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_high_bit_depth_and_multiple_frames_are_explicit_errors(self):
        high = self.inputs / "depth.png"
        Image.new("I;16", (12, 10), 1000).save(high)
        animated = self.inputs / "animated.png"
        Image.new("RGB", (12, 10), "white").save(animated, save_all=True,
                                                    append_images=[Image.new("RGB", (12, 10), "black")], duration=100)
        for path in (high, animated):
            with self.subTest(path=path), self.assertRaises((ValueError, RuntimeError)):
                runner.decode_image(path)

    def test_rehashed_other_checkpoint_cannot_replace_original_epoch173(self):
        model_root = self.root / "model-root"
        (model_root / "metadata").mkdir(parents=True)
        (model_root / "weights").mkdir()
        card = json.loads((Path(runner.__file__).resolve().parents[1] / "metadata/model_card.json").read_text())
        payload = b"different checkpoint with a matching edited card"
        (model_root / card["checkpoint"]).write_bytes(payload)
        card.update(size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        (model_root / "metadata/model_card.json").write_text(json.dumps(card))
        with self.assertRaises((ValueError, RuntimeError)):
            runner.verify_baseline(model_root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
