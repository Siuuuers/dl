#!/usr/bin/env python3
"""Hand-computable checks of the prespecified detection ranking metric."""

import unittest

from detection_ranking_r61 import score_detection_ranking


def target(box=(0, 0, 20, 20)):
    return {"class_id": 0, "bbox_xyxy": list(box), "keypoints": [[0, 0, 0], [0, 0, 0]]}


def prediction(box=(0, 0, 20, 20), confidence=0.9):
    return {**target(box), "confidence": confidence}


class DetectionRankingChecks(unittest.TestCase):
    def test_empty_predictions_with_targets_score_zero(self):
        result = score_detection_ranking({"scene": [target()]}, {"scene": []})
        self.assertEqual((result["ap50"], result["positive_count"], result["fn"]), (0.0, 1, 1))
        self.assertEqual(result["curve"], [])
        self.assertEqual(len(result["interpolated_precision"]), 101)

    def test_no_targets_is_undefined_with_false_positives_retained(self):
        for predictions in ([], [prediction()]):
            with self.subTest(predictions=len(predictions)):
                result = score_detection_ranking({"empty": []}, {"empty": predictions})
                self.assertIsNone(result["ap50"])
                self.assertEqual(result["status"], "undefined_no_reference_targets")
                self.assertEqual(result["fp"], len(predictions))
                self.assertIsNone(result["interpolated_precision"])
                if predictions:
                    self.assertIsNone(result["curve"][0]["recall"])

    def test_perfect_single_match_scores_one(self):
        result = score_detection_ranking({"scene": [target()]}, {"scene": [prediction()]})
        self.assertEqual(result["ap50"], 1.0)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))

    def test_duplicate_between_true_positives_has_hand_computable_ap(self):
        gold = [target(), target((40, 0, 60, 20))]
        predictions = [prediction((0, 0, 10, 20), 0.9), prediction((10, 0, 20, 20), 0.8),
                       prediction((40, 0, 60, 20), 0.7)]
        result = score_detection_ranking({"scene": gold}, {"scene": predictions})
        self.assertEqual([row["tp"] for row in result["curve"]], [1, 0, 1])
        self.assertTrue(result["curve"][1]["duplicate_fp"])
        self.assertEqual(result["curve"][0]["matched_iou"], 0.5)
        self.assertAlmostEqual(result["ap50"], (51 + 50 * 2 / 3) / 101)

    def test_score_ties_use_image_then_original_prediction_index(self):
        result = score_detection_ranking(
            {"b_front": [target()], "a_empty": []},
            {"b_front": [prediction((0, 0, 10, 20)), prediction((10, 0, 20, 20))],
             "a_empty": [prediction()]})
        self.assertEqual([(row["image_id"], row["prediction_index"]) for row in result["curve"]],
                         [("a_empty", 0), ("b_front", 0), ("b_front", 1)])
        self.assertEqual([row["tp"] for row in result["curve"]], [0, 1, 0])
        self.assertEqual(result["ap50"], 0.5)

    def test_missed_target_stays_in_recall_denominator(self):
        result = score_detection_ranking(
            {"found": [target()], "missed": [target()]}, {"found": [prediction()], "missed": []})
        self.assertEqual(result["positive_count"], 2)
        self.assertEqual(result["maximum_recall"], 0.5)
        self.assertAlmostEqual(result["ap50"], 51 / 101)

    def test_floor_cap_and_complete_image_set_are_enforced(self):
        cases = [
            ({"scene": [target()]}, {"scene": [prediction(confidence=0.001)]}),
            ({"scene": [target()]}, {"scene": [prediction()] * 301}),
            ({"scene": [target()], "empty": []}, {"scene": [prediction()]}),
        ]
        for gold, predictions in cases:
            with self.subTest(predictions=predictions.keys()):
                with self.assertRaises(ValueError):
                    score_detection_ranking(gold, predictions)


if __name__ == "__main__":
    unittest.main()
