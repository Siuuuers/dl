#!/usr/bin/env python3
"""Hand-computable pose-quality checks; standard library only, no model or data.

Run from the repository root with ``python3 scripts/check_pose_quality.py``.
"""

import math
import unittest

from diagnose_pose_quality import aggregate_quality, quality_for_image


COUNT_KEYS = (
    "gold_count", "prediction_count", "tp", "fn", "fp", "visible_endpoints",
    "returned_visible_endpoints", "both_visible_objects",
    "both_visible_objects_with_both_returned", "unlocatable_reference_endpoints",
    "unlocatable_on_matched_objects", "returned_on_unlocatable_reference",
)
RATE_KEYS = (
    "object_precision", "object_recall", "visible_endpoint_coverage",
    "both_visible_objects_returned_fraction", "returned_on_unlocatable_reference_fraction",
)


def reference(box=(10, 10, 50, 50), points=((15, 45, 2), (45, 45, 2))):
    return {"class_id": 0, "bbox_xyxy": list(box), "keypoints": [list(p) for p in points]}


def prediction(box=(10, 10, 50, 50), points=((15, 45, 0.9), (45, 45, 0.9)), confidence=0.9):
    return {"class_id": 0, "bbox_xyxy": list(box), "keypoints": [list(p) for p in points],
            "confidence": confidence}


class PoseQualityChecks(unittest.TestCase):
    def quality(self, gold, predictions, radii=(0, 5)):
        return quality_for_image(gold, predictions, width=100, height=100, radii=list(radii))

    def assert_counts(self, result, **expected):
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertEqual(result[key], value)

    def assert_radius_counts(self, result, expected):
        actual = [(r["radius_px"], r["visible_endpoints_within_radius"],
                   r["both_visible_objects_within_radius"]) for r in result["radii"]]
        self.assertEqual(actual, expected)

    def test_radius_boundary_and_zero_radius_include_exact_distances(self):
        # The left point is 3 px across and 4 px down: exactly 5 px away.
        result = self.quality([reference()],
                              [prediction(points=((18, 49, 0.5), (45, 45, 0.5)))],
                              radii=(5, 0, 4))
        self.assert_counts(result, tp=1, visible_endpoints=2, returned_visible_endpoints=2,
                           both_visible_objects=1, both_visible_objects_with_both_returned=1)
        self.assert_radius_counts(result, [(5, 2, 1), (0, 1, 0), (4, 1, 0)])

    def test_two_half_good_objects_do_not_make_one_both_corner_success(self):
        gold = [reference(), reference((60, 10, 90, 50), ((65, 45, 2), (85, 45, 2)))]
        predictions = [prediction(points=((15, 45, 0.9), (35, 45, 0.9))),
                       prediction((60, 10, 90, 50), ((75, 45, 0.9), (85, 45, 0.9)))]
        result = self.quality(gold, predictions, radii=(5, 10))
        self.assert_counts(result, tp=2, visible_endpoints=4, returned_visible_endpoints=4,
                           both_visible_objects=2, both_visible_objects_with_both_returned=2)
        self.assert_radius_counts(result, [(5, 2, 0), (10, 4, 2)])

    def test_missed_object_remains_in_visible_and_both_visible_denominators(self):
        gold = [reference(), reference((60, 10, 90, 50), ((65, 45, 2), (85, 45, 2)))]
        result = self.quality(gold, [prediction()])
        self.assert_counts(result, gold_count=2, prediction_count=1, tp=1, fn=1, fp=0,
                           visible_endpoints=4, returned_visible_endpoints=2,
                           both_visible_objects=2, both_visible_objects_with_both_returned=1)
        self.assert_radius_counts(result, [(0, 2, 1), (5, 2, 1)])

    def test_partial_and_unlocatable_objects_have_separate_denominators(self):
        gold = [reference(points=((15, 45, 2), (0, 0, 0))),
                reference((60, 10, 90, 50), ((0, 0, 0), (0, 0, 0))),
                reference((10, 60, 50, 90), ((0, 0, 0), (0, 0, 0)))]
        predictions = [prediction(), prediction((60, 10, 90, 50),
                                                  ((65, 45, 0.9), (85, 45, 0.9)))]
        result = self.quality(gold, predictions)
        self.assert_counts(result, tp=2, fn=1, fp=0, visible_endpoints=1,
                           returned_visible_endpoints=1, both_visible_objects=0,
                           both_visible_objects_with_both_returned=0,
                           unlocatable_reference_endpoints=5, unlocatable_on_matched_objects=3,
                           returned_on_unlocatable_reference=3)
        self.assert_radius_counts(result, [(0, 1, 0), (5, 1, 0)])
        summary = aggregate_quality([result])
        self.assertEqual(summary["returned_on_unlocatable_reference_fraction"], 1)
        self.assertIsNone(summary["both_visible_objects_returned_fraction"])

    def test_false_objects_do_not_inflate_endpoint_or_object_success(self):
        predictions = [prediction(), prediction(confidence=0.8),
                       prediction((60, 60, 90, 90), ((65, 85, 0.9), (85, 85, 0.9)), 0.95)]
        result = self.quality([reference()], predictions)
        self.assert_counts(result, gold_count=1, prediction_count=3, tp=1, fn=0, fp=2,
                           visible_endpoints=2, returned_visible_endpoints=2,
                           both_visible_objects=1, both_visible_objects_with_both_returned=1,
                           unlocatable_reference_endpoints=0, returned_on_unlocatable_reference=0)
        self.assert_radius_counts(result, [(0, 2, 1), (5, 2, 1)])
        summary = aggregate_quality([result])
        self.assertAlmostEqual(summary["object_precision"], 1 / 3)
        self.assertEqual(summary["object_recall"], 1)

    def test_unavailable_visible_point_fails_even_at_large_radius(self):
        bad_points = [(15, 45, 0.499), (math.nan, 45, 0.9), (15, math.inf, 0.9),
                      (None, 45, 0.9), (-0.001, 45, 0.9), (100, 45, 0.9),
                      (15, -0.001, 0.9), (15, 100, 0.9)]
        for point in bad_points:
            with self.subTest(point=point):
                result = self.quality([reference()],
                                      [prediction(points=(point, (45, 45, 0.9)))],
                                      radii=(0, 5, 1000))
                self.assert_counts(result, tp=1, visible_endpoints=2, returned_visible_endpoints=1,
                                   both_visible_objects=1, both_visible_objects_with_both_returned=0)
                self.assert_radius_counts(result, [(0, 1, 0), (5, 1, 0), (1000, 1, 0)])

    def test_returns_on_unlocatable_reference_use_the_same_validity_gate(self):
        cases = [((45, 45, 0.5), 1), ((0, 0, 0.5), 1), ((45, 45, 0.499), 0),
                 ((math.nan, 45, 0.9), 0), ((None, None, 0.9), 0),
                 ((-0.001, 45, 0.9), 0), ((100, 45, 0.9), 0), ((45, 100, 0.9), 0)]
        for point, returned in cases:
            with self.subTest(point=point):
                result = self.quality([reference(points=((15, 45, 2), (0, 0, 0)))],
                                      [prediction(points=((15, 45, 0.9), point))])
                self.assert_counts(result, visible_endpoints=1, returned_visible_endpoints=1,
                                   both_visible_objects=0, both_visible_objects_with_both_returned=0,
                                   unlocatable_reference_endpoints=1, unlocatable_on_matched_objects=1,
                                   returned_on_unlocatable_reference=returned)
                self.assert_radius_counts(result, [(0, 1, 0), (5, 1, 0)])

    def test_fixed_iou_gate_cannot_be_bypassed_with_accurate_endpoints(self):
        gold = [reference((0, 0, 20, 20), ((5, 15, 2), (15, 15, 2)))]
        points = ((5, 15, 0.9), (15, 15, 0.9))
        boundary = self.quality(gold, [prediction((0, 0, 40, 20), points)], radii=(0,))
        self.assert_counts(boundary, tp=1, fn=0, fp=0, returned_visible_endpoints=2)
        self.assert_radius_counts(boundary, [(0, 2, 1)])
        below = self.quality(gold, [prediction((0, 0, 40.001, 20), points)], radii=(1000,))
        self.assert_counts(below, tp=0, fn=1, fp=1, visible_endpoints=2,
                           returned_visible_endpoints=0, both_visible_objects=1,
                           both_visible_objects_with_both_returned=0)
        self.assert_radius_counts(below, [(1000, 0, 0)])

    def test_aggregate_uses_summed_denominators_instead_of_mean_image_rates(self):
        first = self.quality([reference()], [prediction()])
        two_missed = self.quality([reference(), reference((60, 10, 90, 50),
                                                           ((65, 45, 2), (85, 45, 2)))], [])
        empty_with_extra = self.quality([], [prediction()])
        result = aggregate_quality([first, two_missed, empty_with_extra])
        self.assert_counts(result, image_count=3, gold_count=3, prediction_count=2,
                           tp=1, fn=2, fp=1, visible_endpoints=6, returned_visible_endpoints=2,
                           both_visible_objects=3, both_visible_objects_with_both_returned=1)
        self.assertEqual(result["object_precision"], 0.5)
        for key in ("object_recall", "visible_endpoint_coverage", "both_visible_objects_returned_fraction"):
            self.assertAlmostEqual(result[key], 1 / 3)
        self.assertIsNone(result["returned_on_unlocatable_reference_fraction"])
        self.assert_radius_counts(result, [(0, 2, 1), (5, 2, 1)])
        for row in result["radii"]:
            self.assertAlmostEqual(row["visible_endpoint_success_rate"], 1 / 3)
            self.assertAlmostEqual(row["both_visible_object_success_rate"], 1 / 3)

    def test_zero_denominators_produce_null_rates(self):
        empty = aggregate_quality([])
        self.assertEqual(empty["image_count"], 0)
        self.assertEqual(empty["radii"], [])
        for key in COUNT_KEYS:
            self.assertEqual(empty[key], 0)
        for key in RATE_KEYS:
            self.assertIsNone(empty[key])
        for predictions in ([], [prediction()]):
            with self.subTest(predictions=len(predictions)):
                result = aggregate_quality([self.quality([], predictions)])
                self.assertEqual(result["fp"], len(predictions))
                self.assertEqual(result["object_precision"], 0 if predictions else None)
                for key in RATE_KEYS[1:]:
                    self.assertIsNone(result[key])
                for row in result["radii"]:
                    self.assertEqual(row["visible_endpoints_within_radius"], 0)
                    self.assertEqual(row["both_visible_objects_within_radius"], 0)
                    self.assertIsNone(row["visible_endpoint_success_rate"])
                    self.assertIsNone(row["both_visible_object_success_rate"])

    def test_invalid_radii_are_rejected(self):
        for radii in ([], [-1], [math.nan], [math.inf], [1, 1], [1, 1.0], [True], [False]):
            with self.subTest(radii=radii):
                with self.assertRaises(ValueError):
                    self.quality([reference()], [prediction()], radii=radii)

    def test_aggregate_requires_identical_radius_values_and_order(self):
        first = self.quality([], [], radii=(0, 5))
        for radii in ((5, 0), (0, 10), (0,)):
            with self.subTest(radii=radii):
                second = self.quality([], [], radii=radii)
                with self.assertRaises(ValueError):
                    aggregate_quality([first, second])


if __name__ == "__main__":
    unittest.main(verbosity=2)
