import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_detection_errors_r62 import inspect_image, miss_stage, overlap_stratum


def target(box=(0, 0, 30, 30), available=2):
    points = [[2, 28, 2], [28, 28, 2]]
    for index in range(available, 2):
        points[index] = [0, 0, 0]
    return {"class_id": 0, "bbox_xyxy": list(box), "keypoints": points}


def prediction(box=(0, 0, 30, 30), confidence=0.9):
    value = target(box)
    value["confidence"] = confidence
    for point in value["keypoints"]:
        point[2] = 0.9
    return value


class DetectionErrorsTest(unittest.TestCase):
    def test_overlap_boundaries_and_duplicate_precedence(self):
        self.assertEqual(overlap_stratum(0.1, False), "near_reference_overlap")
        self.assertEqual(overlap_stratum(0.1 - 1e-9, False), "low_reference_overlap")
        self.assertEqual(overlap_stratum(0.5 - 1e-9, False), "near_reference_overlap")
        self.assertEqual(overlap_stratum(0.5, True), "duplicate")
        with self.assertRaises(RuntimeError):
            overlap_stratum(0.5, False)

    def test_duplicate_is_subset_and_half_overlap_matches(self):
        gold = [target()]
        predictions = [prediction(), prediction((0, 0, 15, 30), 0.8)]
        before = copy.deepcopy((gold, predictions))
        counts, cases, score = inspect_image("ref", "model", "photo", gold, predictions, predictions)
        self.assertEqual((score["tp"], score["fp"], score["duplicate_fp"]), (1, 1, 1))
        self.assertEqual(counts["extra_strata"], {"duplicate": 1, "near_reference_overlap": 0, "low_reference_overlap": 0})
        self.assertEqual(cases[0]["nearest_reference_row"], 1)
        self.assertTrue(cases[0]["nearest_reference_matched"])
        self.assertEqual((gold, predictions), before)

    def test_empty_image_has_no_invented_nearest_reference(self):
        counts, cases, score = inspect_image("ref", "model", "empty", [], [prediction()], [prediction()])
        self.assertTrue(counts["reference_empty"])
        self.assertEqual(score["fp"], 1)
        self.assertEqual(cases[0]["stratum"], "low_reference_overlap")
        self.assertIsNone(cases[0]["nearest_reference_index"])
        self.assertIsNone(cases[0]["nearest_reference_matched"])

    def test_incomplete_fronts_remain_missed_detection_targets(self):
        gold = [target(available=2), target((40, 0, 70, 30), 1), target((80, 0, 110, 30), 0)]
        counts, cases, score = inspect_image("ref", "model", "partial", gold, [], [])
        self.assertEqual(score["fn"], 3)
        self.assertEqual(counts["by_available_corners"], {str(i): {"fronts": 1, "missed": 1} for i in range(3)})
        self.assertEqual({case["available_corners"] for case in cases}, {0, 1, 2})
        self.assertTrue(all(case["cache_stage"] == "no_eligible_cached_candidate" for case in cases))

    def test_trace_distinguishes_known_cache_stages(self):
        gold = target()
        self.assertEqual(miss_stage(gold, [], [])["cache_stage"], "no_eligible_cached_candidate")
        self.assertEqual(miss_stage(gold, [prediction(confidence=0.25)], [])["cache_stage"], "explicit_confidence_filter")
        self.assertEqual(miss_stage(gold, [prediction()], [])["cache_stage"], "extra_postclip_suppression")
        self.assertEqual(miss_stage(gold, [prediction()], [prediction()])["cache_stage"], "retained_assignment_contention")

    def test_competing_references_do_not_turn_one_detection_into_two(self):
        gold = [target(), target((1, 0, 31, 30))]
        predictions = [prediction()]
        counts, cases, score = inspect_image("ref", "model", "crowded", gold, predictions, predictions)
        self.assertEqual((score["tp"], score["fn"]), (1, 1))
        self.assertEqual(cases[0]["reference_row"], 2)
        self.assertEqual(cases[0]["cache_stage"], "retained_assignment_contention")


if __name__ == "__main__":
    unittest.main()
