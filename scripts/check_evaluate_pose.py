#!/usr/bin/env python3
"""Focused evaluator checks using only the standard library; no inference or weights.

Run from the repository root with ``python3 scripts/check_evaluate_pose.py``.
Expected values below are small, hand-computable examples of the documented
matching and endpoint rules, independent of the scoring implementation.
"""

import copy
import contextlib
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from pose_scoring import aggregate_scores, score_image


def gold(box=(10, 10, 50, 50), points=((15, 45, 2), (45, 45, 2)), class_id=0):
    return {"class_id": class_id, "bbox_xyxy": list(box),
            "keypoints": [list(point) for point in points]}


def prediction(box=(10, 10, 50, 50), points=((15, 45, 0.9), (45, 45, 0.9)),
               confidence=0.9, class_id=0):
    return {"class_id": class_id, "bbox_xyxy": list(box),
            "keypoints": [list(point) for point in points], "confidence": confidence}


class PoseScoringChecks(unittest.TestCase):
    def score(self, reference, predictions, **kwargs):
        return score_image(reference, predictions, width=100, height=100, **kwargs)

    def assert_counts(self, result, **expected):
        for name, value in expected.items():
            with self.subTest(metric=name):
                self.assertEqual(result[name], value)

    def test_perfect_match_has_full_coverage_and_zero_error(self):
        result = self.score([gold()], [prediction()])
        self.assert_counts(result, gold_count=1, prediction_count=1, tp=1, fn=0, fp=0,
                           duplicate_fp=0, other_fp=0, visible_endpoints=2,
                           returned_endpoints=2, missing_endpoints=0)
        self.assertEqual(result["endpoint_coverage"], 1)
        self.assertEqual(result["endpoint_errors_px"], [0, 0])
        self.assertEqual(result["conditional_median_error_px"], 0)
        self.assertEqual(result["conditional_p95_error_px"], 0)

    def test_missed_objects_remain_in_endpoint_denominator(self):
        reference = [gold(), gold((60, 10, 90, 50), ((65, 45, 2), (85, 45, 2)))]
        result = self.score(reference, [prediction()])
        self.assert_counts(result, tp=1, fn=1, fp=0, visible_endpoints=4,
                           returned_endpoints=2, missing_endpoints=2)
        self.assertEqual(result["endpoint_coverage"], 0.5)
        self.assertEqual(result["conditional_p95_error_px"], 0)
        self.assertEqual(result["unmatched_gold_indices"], [1])

    def test_confidence_order_separates_duplicates_from_other_false_positives(self):
        predictions = [
            prediction(points=((25, 45, 0.9), (55, 45, 0.9)), confidence=0.6),
            prediction(confidence=0.9),
            prediction((70, 70, 90, 90), confidence=0.95),
        ]
        result = self.score([gold()], predictions)
        self.assert_counts(result, tp=1, fn=0, fp=2, duplicate_fp=1, other_fp=1)
        self.assertEqual(result["matches"][0]["prediction_index"], 1)
        self.assertEqual(result["duplicate_prediction_indices"], [0])
        self.assertEqual(result["other_fp_prediction_indices"], [2])
        self.assertEqual(result["endpoint_errors_px"], [0, 0])

    def test_equal_confidence_uses_original_prediction_index(self):
        predictions = [prediction(points=((25, 45, 0.9), (55, 45, 0.9))), prediction()]
        result = self.score([gold()], predictions)
        self.assertEqual(result["matches"][0]["prediction_index"], 0)
        self.assertEqual(result["endpoint_errors_px"], [10, 10])
        self.assert_counts(result, tp=1, duplicate_fp=1)

    def test_highest_iou_beats_reference_row_order(self):
        reference = [gold((10, 10, 60, 50), ((15, 45, 2), (45, 45, 2))),
                     gold((20, 10, 60, 50), ((25, 45, 2), (55, 45, 2)))]
        result = self.score(reference, [prediction((20, 10, 60, 50),
                                                  ((25, 45, 0.9), (55, 45, 0.9)))])
        self.assertEqual(result["matches"][0]["gold_index"], 1)
        self.assertEqual(result["endpoint_errors_px"], [0, 0])
        self.assertEqual(result["unmatched_gold_indices"], [0])

    def test_greedy_matching_and_iou_tie_use_reference_row_order(self):
        # The first prediction has IoU 0.6 with both gold boxes. Greedy matching
        # chooses row 0, even though a global assignment could produce two TPs.
        reference = [gold((0, 0, 20, 20), ((5, 15, 2), (15, 15, 2))),
                     gold((10, 0, 30, 20), ((15, 15, 2), (25, 15, 2)))]
        predictions = [prediction((5, 0, 25, 20), ((5, 15, 0.9), (15, 15, 0.9)), 0.9),
                       prediction((0, 0, 20, 20), ((5, 15, 0.9), (15, 15, 0.9)), 0.8)]
        result = self.score(reference, predictions)
        self.assert_counts(result, tp=1, fn=1, fp=1, duplicate_fp=1, other_fp=0)
        self.assertEqual([(m["prediction_index"], m["gold_index"])
                          for m in result["matches"]], [(0, 0)])
        self.assertAlmostEqual(result["matches"][0]["iou"], 0.6)

    def test_disjoint_objects_match_independently(self):
        reference = [gold(), gold((60, 10, 90, 50), ((65, 45, 2), (85, 45, 2)))]
        predictions = [prediction((60, 10, 90, 50), ((65, 45, 0.9), (85, 45, 0.9))),
                       prediction(confidence=0.8)]
        result = self.score(reference, predictions)
        self.assert_counts(result, tp=2, fn=0, fp=0, returned_endpoints=4)
        self.assertEqual([(m["prediction_index"], m["gold_index"])
                          for m in result["matches"]], [(0, 1), (1, 0)])

    def test_matching_requires_same_class_and_includes_iou_threshold(self):
        different_class = self.score([gold()], [prediction(class_id=1)])
        self.assert_counts(different_class, tp=0, fn=1, fp=1, duplicate_fp=0, other_fp=1)
        reference = [gold((0, 0, 20, 20), ((5, 15, 2), (15, 15, 2)))]
        predictions = [prediction((0, 0, 10, 20), ((5, 15, 0.9), (15, 15, 0.9)))]
        self.assertEqual(self.score(reference, predictions)["tp"], 1)
        self.assertEqual(self.score(reference, predictions, matching_iou=0.5001)["tp"], 0)

    def test_missing_gold_endpoint_is_excluded_not_imputed(self):
        reference = [gold(points=((15, 45, 2), (0, 0, 0)))]
        predictions = [prediction(points=((18, 49, 0.9), (90, 90, 0.9)))]
        result = self.score(reference, predictions)
        self.assert_counts(result, tp=1, visible_endpoints=1, returned_endpoints=1,
                           missing_endpoints=0)
        self.assertEqual(result["endpoint_coverage"], 1)
        self.assertEqual(result["endpoint_errors_px"], [5])
        self.assertEqual(result["conditional_median_error_px"], 5)
        self.assertEqual(result["conditional_p95_error_px"], 5)

    def test_bad_predicted_endpoint_counts_missing_without_dropping_object(self):
        invalid_points = [
            (15, 45, 0.49), (math.nan, 45, 0.9), (math.inf, 45, 0.9),
            (15, -math.inf, 0.9), (None, 45, 0.9), (15, None, 0.9),
            (-0.001, 45, 0.9), (100, 45, 0.9), (15, -0.001, 0.9), (15, 100, 0.9),
        ]
        for point in invalid_points:
            with self.subTest(point=point):
                result = self.score([gold()], [prediction(points=(point, (45, 45, 0.9)))])
                self.assert_counts(result, tp=1, fn=0, visible_endpoints=2,
                                   returned_endpoints=1, missing_endpoints=1)
                self.assertEqual(result["endpoint_coverage"], 0.5)
                self.assertEqual(result["endpoint_errors_px"], [0])

    def test_endpoint_threshold_and_in_frame_boundaries(self):
        reference = [gold((0, 0, 100, 100), ((0, 0, 2), (99.999, 99.999, 2)))]
        predictions = [prediction((0, 0, 100, 100), ((0, 0, 0.5), (99.999, 99.999, 0.5)))]
        result = self.score(reference, predictions)
        self.assert_counts(result, returned_endpoints=2, missing_endpoints=0)
        self.assertEqual(result["endpoint_errors_px"], [0, 0])

    def test_left_and_right_are_never_swapped_to_reduce_error(self):
        result = self.score([gold()], [prediction(points=((45, 45, 0.9), (15, 45, 0.9)))])
        self.assertEqual(result["endpoint_errors_px"], [30, 30])
        self.assertEqual(result["conditional_median_error_px"], 30)

    def test_empty_scenes_and_no_returned_points_have_null_errors(self):
        empty = self.score([], [])
        self.assert_counts(empty, tp=0, fn=0, fp=0, visible_endpoints=0, returned_endpoints=0)
        negative = self.score([], [prediction()])
        self.assert_counts(negative, tp=0, fn=0, fp=1, other_fp=1, duplicate_fp=0)
        no_visible = self.score([gold(points=((0, 0, 0), (0, 0, 0)))], [prediction()])
        self.assert_counts(no_visible, tp=1, visible_endpoints=0, returned_endpoints=0)
        for result in (empty, negative, no_visible):
            self.assertIsNone(result["endpoint_coverage"])
            self.assertIsNone(result["conditional_median_error_px"])
            self.assertIsNone(result["conditional_p95_error_px"])
        missed = self.score([gold()], [])
        self.assert_counts(missed, fn=1, visible_endpoints=2, returned_endpoints=0,
                           missing_endpoints=2)
        self.assertEqual(missed["endpoint_coverage"], 0)
        self.assertIsNone(missed["conditional_median_error_px"])
        self.assertIsNone(missed["conditional_p95_error_px"])

    def test_aggregate_pools_errors_and_weights_coverage_by_endpoint_count(self):
        first = self.score([gold()], [prediction(points=((15, 45, 0.9), (55, 45, 0.9)))])
        reference = [gold(), gold((60, 10, 90, 50), ((65, 45, 2), (85, 45, 2))),
                     gold((60, 60, 90, 90), ((65, 85, 2), (85, 85, 2)))]
        second = self.score(reference, [prediction(points=((35, 45, 0.9), (75, 45, 0.9)))])
        result = aggregate_scores([first, second])
        self.assert_counts(result, image_count=2, gold_count=4, prediction_count=2,
                           tp=2, fn=2, fp=0, visible_endpoints=8,
                           returned_endpoints=4, missing_endpoints=4)
        # Pool [0,10] and [20,30]; per-image averages would give different answers.
        self.assertEqual(result["endpoint_coverage"], 0.5)
        self.assertEqual(result["conditional_median_error_px"], 15)
        self.assertAlmostEqual(result["conditional_p95_error_px"], 28.5)
        empty = aggregate_scores([])
        self.assert_counts(empty, image_count=0, tp=0, fn=0, fp=0, visible_endpoints=0)
        self.assertIsNone(empty["endpoint_coverage"])
        self.assertIsNone(empty["conditional_p95_error_px"])

    def test_scoring_preserves_original_input_order_and_values(self):
        reference = [gold()]
        predictions = [prediction(confidence=0.6), prediction(confidence=0.9)]
        before = copy.deepcopy((reference, predictions))
        self.score(reference, predictions)
        self.assertEqual((reference, predictions), before)

    def test_invalid_structure_and_parameters_fail_explicitly(self):
        invalid = [
            ([gold(box=(10, 10, 10, 50))], [prediction()]),
            ([gold(points=((15, 45, 1), (45, 45, 2)))], [prediction()]),
            ([gold(class_id=True)], [prediction()]),
            ([gold()], [prediction(confidence=math.nan)]),
            ([gold()], [prediction(points=((15, 45, math.nan), (45, 45, 0.9)))]),
        ]
        for reference, predictions in invalid:
            with self.subTest(reference=reference, predictions=predictions):
                with self.assertRaises(ValueError):
                    self.score(reference, predictions)
        for kwargs in ({"matching_iou": 0}, {"matching_iou": 1.1},
                       {"endpoint_conf": -0.1}, {"endpoint_conf": 1.1}):
            with self.subTest(parameters=kwargs), self.assertRaises(ValueError):
                self.score([], [], **kwargs)
        with self.assertRaises(ValueError):
            score_image([], [], width=0, height=100)


class LabelAndOutputChecks(unittest.TestCase):
    ROW = "0 0.5 0.5 0.4 0.2 0.3 0.6 2 0.7 0.6 2"

    @classmethod
    def setUpClass(cls):
        cls.evaluator = importlib.import_module("evaluate_pose")

    def parse(self, text):
        return self.evaluator.parse_pose_labels(text, 1280, 720, source="case.txt")

    def test_normalized_labels_convert_to_original_image_pixels(self):
        result = self.parse(self.ROW + "\n")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class_id"], 0)
        for actual, expected in zip(result[0]["bbox_xyxy"], (384, 288, 896, 432)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(result[0]["keypoints"], ((384, 432, 2), (896, 432, 2))):
            for value, wanted in zip(actual, expected):
                self.assertAlmostEqual(value, wanted)

    def test_empty_labels_and_one_missing_partner_remain_valid(self):
        self.assertEqual(self.parse(""), [])
        self.assertEqual(self.parse("\n  \t\n"), [])
        for row, missing_index in (
            ("0 .5 .5 .4 .2 0 0 0 .7 .6 2", 0),
            ("0 .5 .5 .4 .2 .3 .6 2 0 0 0", 1),
        ):
            with self.subTest(row=row):
                result = self.parse(row)
                self.assertEqual(result[0]["keypoints"][missing_index], [0, 0, 0])
                self.assertEqual(result[0]["keypoints"][1 - missing_index][2], 2)

    def test_malformed_or_semantically_invalid_rows_are_rejected(self):
        rows = [
            "0 .5", self.ROW + " 0", self.ROW.replace("0 0.5", "1 0.5", 1),
            "0 .5 .5 0 .2 .3 .6 2 .7 .6 2",
            "0 .5 .5 .4 -.2 .3 .6 2 .7 .6 2",
            "0 1.1 .5 .4 .2 .3 .6 2 .7 .6 2",
            "0 .5 .5 .4 .2 nan .6 2 .7 .6 2",
            "0 .5 .5 .4 .2 .3 .6 2 inf .6 2",
            "0 .5 .5 .4 .2 .3 .6 1 .7 .6 2",
            "0 .5 .5 .4 .2 .3 .6 0 .7 .6 2",
            "0 .5 .5 .4 .2 .3 .6 2 .7 .6 0",
            "0 .5 .5 .4 .2 .7 .6 2 .3 .6 2",
        ]
        for row in rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.parse(row)

    def test_parse_failure_names_source(self):
        with self.assertRaisesRegex(ValueError, "case[.]txt"):
            self.parse("0 .5")

    def test_output_creation_never_overwrites_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result"
            created = self.evaluator.create_output_directory(target)
            self.assertEqual(Path(created), target)
            self.assertTrue(target.is_dir())
            sentinel = target / "evidence.txt"
            sentinel.write_text("keep original evidence", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.evaluator.create_output_directory(target)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep original evidence")
            empty = Path(directory) / "empty"
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                self.evaluator.create_output_directory(empty)


def png_bytes(width=100, height=100):
    """A real RGB PNG, generated without Pillow or borrowed dataset images."""
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = (b"\x00" + b"\x00\x00\x00" * width) * height
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def preflight_fixture(root):
    """A 20-image integrity fixture; checkpoint bytes are never deserialized."""
    for directory in ("data/images/val", "data/labels/val", "metadata", "weights"):
        (root / directory).mkdir(parents=True)
    checkpoint = root / "weights/cassette_lower_epoch173.pt"
    checkpoint.write_bytes(b"test checkpoint bytes, never used for inference\n")
    model_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    card = {"checkpoint": "weights/cassette_lower_epoch173.pt", "sha256": model_hash,
            "size_bytes": checkpoint.stat().st_size, "task": "pose",
            "class_names": ["cassette_front"], "kpt_shape": [2, 3],
            "flip_idx": [1, 0], "validation_images": 20}
    (root / "metadata/model_card.json").write_text(json.dumps(card), encoding="utf-8")
    records = []
    image_bytes = png_bytes()
    for index in range(20):
        image_id = f"case_{index:02d}"
        image_path = f"images/val/{image_id}.png"
        label_path = f"labels/val/{image_id}.txt"
        label_text = LabelAndOutputChecks.ROW + "\n" if index == 0 else ""
        (root / "data" / image_path).write_bytes(image_bytes)
        (root / "data" / label_path).write_text(label_text, encoding="utf-8")
        records.append({
            "id": image_id, "split": "val", "image": image_path, "label": label_path,
            "original_size": [100, 100], "instances": 1 if index == 0 else 0,
            "visible_endpoints": 2 if index == 0 else 0, "missing_endpoints": 0,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "label_sha256": hashlib.sha256(label_text.encode()).hexdigest(),
        })
    manifest = {
        "schema_version": 1, "model_sha256": model_hash, "kpt_shape": [2, 3],
        "split_summary": {"val": {"images": 20, "instances": 1,
                                  "visible_endpoints": 2, "missing_endpoints": 0,
                                  "empty_images": 19}},
        "records": records,
    }
    (root / "data/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class PreflightChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = importlib.import_module("evaluate_pose")

    def test_valid_fixture_accounts_for_all_images_including_empty_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = preflight_fixture(root)
            result = self.evaluator.preflight_dataset(root)
            records = result["records"]
            self.assertEqual([r["id"] for r in records], [r["id"] for r in manifest["records"]])
            self.assertEqual(sum(len(record["gold"]) for record in records), 1)
            self.assertEqual(sum(not record["gold"] for record in records), 19)
            self.assertTrue(all((r["width"], r["height"]) == (100, 100) for r in records))
            self.assertEqual(result["model_sha256"], manifest["model_sha256"])
            self.assertEqual(result["manifest_sha256"],
                             hashlib.sha256((root / "data/manifest.json").read_bytes()).hexdigest())

    def test_one_image_preflight_does_not_require_other_image_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = preflight_fixture(root)
            for record in manifest["records"][1:]:
                (root / "data" / record["image"]).unlink()
            result = self.evaluator.preflight_dataset(root, selected_ids=["case_00"])
            self.assertEqual([r["id"] for r in result["records"]], ["case_00"])
            with self.assertRaises(ValueError):
                self.evaluator.preflight_dataset(root, selected_ids=["unknown_image"])

    def test_changed_image_label_and_checkpoint_bytes_are_rejected(self):
        for relative in ("data/images/val/case_00.png", "data/labels/val/case_00.txt",
                         "weights/cassette_lower_epoch173.pt"):
            with self.subTest(asset=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                preflight_fixture(root)
                path = root / relative
                path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaises(ValueError):
                    self.evaluator.preflight_dataset(root)

    def test_lfs_pointer_is_identified_before_inference(self):
        pointer = ("version https://git-lfs.github.com/spec/v1\n"
                   "oid sha256:" + "0" * 64 + "\nsize 12345\n")
        for relative in ("data/images/val/case_00.png", "weights/cassette_lower_epoch173.pt"):
            with self.subTest(asset=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                preflight_fixture(root)
                (root / relative).write_text(pointer, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "(?i)lfs|pointer"):
                    self.evaluator.preflight_dataset(root)

    def test_missing_label_is_not_treated_as_an_empty_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight_fixture(root)
            (root / "data/labels/val/case_19.txt").unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.evaluator.preflight_dataset(root)

    def test_wrong_png_dimensions_are_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = preflight_fixture(root)
            changed = png_bytes(100, 101)
            (root / "data/images/val/case_00.png").write_bytes(changed)
            manifest["records"][0]["image_sha256"] = hashlib.sha256(changed).hexdigest()
            (root / "data/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.evaluator.preflight_dataset(root)

    def test_label_counts_must_match_manifest_even_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = preflight_fixture(root)
            changed = (LabelAndOutputChecks.ROW + "\n") * 2
            (root / "data/labels/val/case_00.txt").write_text(changed, encoding="utf-8")
            manifest["records"][0]["label_sha256"] = hashlib.sha256(changed.encode()).hexdigest()
            (root / "data/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.evaluator.preflight_dataset(root)


class ReplayChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = importlib.import_module("evaluate_pose")

    def reference(self):
        return {
            "model_sha256": "a" * 64, "manifest_sha256": "b" * 64,
            "model_card_sha256": "c" * 64,
            "records": [
                {"id": f"case_{index:02d}", "width": 100, "height": 100,
                 "image": f"images/val/case_{index:02d}.png",
                 "label": f"labels/val/case_{index:02d}.txt",
                 "image_sha256": hashlib.sha256(f"image {index}".encode()).hexdigest(),
                 "label_sha256": hashlib.sha256(f"label {index}".encode()).hexdigest()}
                for index in range(20)
            ],
        }

    def replay_data(self, preflight, thresholds=(0.15, 0.25, 0.40)):
        header = {
            "record_type": "run", "schema_version": 1,
            "model_sha256": preflight["model_sha256"],
            "manifest_sha256": preflight["manifest_sha256"],
            "model_card_sha256": preflight["model_card_sha256"],
            "inference": copy.deepcopy(self.evaluator.INFERENCE),
            "box_confidences": list(thresholds),
            "source": {"commit": "original-inference-revision", "dirty": False},
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "runtime": {"device_actual": "cpu", "ultralytics": "8.4.182", "fixture": True},
            "warmup": "one explicit zero-image prediction excluded from all timings",
            "images": [{key: record[key] for key in
                        ("id", "image", "label", "image_sha256", "label_sha256", "width", "height")}
                       for record in preflight["records"]],
        }
        items = []
        # Header duplicates must be caught as such, rather than indirectly by
        # generating duplicate records that would fail a different check.
        for threshold in dict.fromkeys(thresholds):
            for index, record in enumerate(preflight["records"]):
                items.append({
                    "record_type": "prediction", "image_id": record["id"], "threshold": threshold,
                    "image_sha256": record["image_sha256"], "label_sha256": record["label_sha256"],
                    "width": record["width"], "height": record["height"],
                    "predictions": [prediction((30, 40, 70, 60),
                                               ((30, 60, 0.9), (70, 60, 0.9)))] if index == 0 else [],
                    "stage_ms": {"preprocess": 0.1, "inference": 1.0, "postprocess": 0.1},
                    "predict_call_wall_ms": 1.2,
                })
        return header, items

    def write_replay(self, path, header, items):
        path.write_text("".join(json.dumps(item, allow_nan=False) + "\n"
                                for item in (header, *items)), encoding="utf-8")

    def read_replay(self, preflight, header, items, requested=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            self.write_replay(path, header, items)
            return self.evaluator._read_replay(path, preflight, requested)

    def test_complete_twenty_image_three_threshold_replay_includes_empty_predictions(self):
        preflight = self.reference()
        header, items = self.replay_data(preflight)
        actual_header, chosen, thresholds = self.read_replay(preflight, header, items)
        self.assertEqual(actual_header, header)
        self.assertEqual(thresholds, [0.15, 0.25, 0.40])
        self.assertEqual(chosen, items)
        self.assertEqual(len(chosen), 60)
        self.assertEqual(sum(not item["predictions"] for item in chosen), 57)

    def test_missing_or_duplicate_selected_pair_is_rejected_including_empty_scene(self):
        preflight = self.reference()
        header, items = self.replay_data(preflight)
        # Index 21 is the empty case_01 scene at confidence 0.25.
        self.assertEqual((items[21]["image_id"], items[21]["threshold"]), ("case_01", 0.25))
        self.assertEqual(items[21]["predictions"], [])
        for broken in (items[:21] + items[22:], items + [copy.deepcopy(items[21])]):
            with self.subTest(length=len(broken)), self.assertRaises(ValueError):
                self.read_replay(preflight, header, broken, requested=[0.25])

    def test_model_manifest_image_and_label_identity_must_match(self):
        preflight = self.reference()
        for scope, field in (("header", "model_sha256"), ("header", "manifest_sha256"),
                             ("item", "image_sha256"), ("item", "label_sha256")):
            with self.subTest(scope=scope, field=field):
                header, items = self.replay_data(preflight)
                (header if scope == "header" else items[0])[field] = "d" * 64
                with self.assertRaises(ValueError):
                    self.read_replay(preflight, header, items)

    def test_changed_inference_settings_are_rejected(self):
        preflight = self.reference()
        for setting, changed in (("iou", 0.5), ("half", True), ("nms", False)):
            with self.subTest(setting=setting):
                header, items = self.replay_data(preflight)
                header["inference"][setting] = changed
                with self.assertRaises(ValueError):
                    self.read_replay(preflight, header, items)

    def test_requested_threshold_subset_must_have_been_inferred(self):
        preflight = self.reference()
        header, items = self.replay_data(preflight)
        _, chosen, thresholds = self.read_replay(preflight, header, items, requested=[0.25])
        self.assertEqual(thresholds, [0.25])
        self.assertEqual(len(chosen), 20)
        self.assertEqual({item["threshold"] for item in chosen}, {0.25})
        with self.assertRaises(ValueError):
            self.read_replay(preflight, header, items, requested=[0.30])

    def test_retained_detection_must_exceed_its_claimed_box_threshold(self):
        preflight = self.reference()
        for confidence in (0.24, 0.25):
            with self.subTest(confidence=confidence):
                header, items = self.replay_data(preflight, thresholds=[0.25])
                items[0]["predictions"][0]["confidence"] = confidence
                with self.assertRaises(ValueError):
                    self.read_replay(preflight, header, items)

    def test_header_thresholds_must_be_nonempty_unique_and_in_range(self):
        preflight = self.reference()
        for thresholds in ([], [0.25, 0.25], [-0.1], [1.1]):
            with self.subTest(thresholds=thresholds):
                header, items = self.replay_data(preflight, thresholds)
                with self.assertRaises(ValueError):
                    self.read_replay(preflight, header, items)

    def test_replaying_a_replay_keeps_original_inference_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reference"
            preflight_fixture(root)
            preflight = self.evaluator.preflight_dataset(root)
            header, items = self.replay_data(preflight)
            original = Path(directory) / "original.jsonl"
            self.write_replay(original, header, items)
            first, second = Path(directory) / "first", Path(directory) / "second"
            common = ["--root", str(root), "--overlays", "0"]
            errors = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
                first_exit = self.evaluator.main(common + ["--output", str(first),
                                                  "--replay", str(original), "--conf", "0.25"])
                self.assertEqual(first_exit, 0, errors.getvalue())
                second_exit = self.evaluator.main(common + ["--output", str(second),
                                                   "--replay", str(first / "predictions.jsonl")])
                self.assertEqual(second_exit, 0, errors.getvalue())
            report = json.loads((second / "summary.json").read_text())
            self.assertEqual(report["box_confidences"], [0.25])
            self.assertEqual(report["replay"]["inference_source"], header["source"])
            self.assertEqual(report["replay"]["inference_created_at_utc"], header["created_at_utc"])
            self.assertEqual(report["runtime"], header["runtime"])
            self.assertEqual(report["warmup"], header["warmup"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
