#!/usr/bin/env python3
"""Focused fresh-evidence and paired-diagnostic checks; no weights or training."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from check_evaluate_pose import gold, prediction
from diagnose_pose_quality import aggregate_quality, quality_for_image
from pose_scoring import score_image
from pose_checkpoint import ANNOTATION_RECIPE_IDS, ORIGINAL_RECIPE_ID
from rescore_annotation_comparison import (common_diagnostics, returned_endpoints, summarize,
                                           verify_fresh_inference)


class FreshComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.revision = "a" * 40
        source = {"revision": self.revision, "working_tree_status": "", "files_sha256": {"fixture": "b" * 64}}
        self.headers = {name: {"source": copy.deepcopy(source), "created_at_utc": "2026-11-10T10:05:00+00:00"}
                        for name in ("m0", "m1", "m2")}
        self.receipts = {"reference": {"created_at_utc": "2026-11-10T10:01:00+00:00"},
                         "candidate": {"created_at_utc": "2026-11-10T10:02:00+00:00"}}

    def test_complete_post_training_sweeps_accept_one_clean_revision(self):
        result = verify_fresh_inference(self.headers, self.receipts, self.revision)
        self.assertEqual(result["both_training_receipts_completed_by_utc"], "2026-11-10T10:02:00+00:00")

    def test_old_m0_cannot_be_reused_between_new_arms(self):
        self.headers["m0"]["created_at_utc"] = "2026-11-10T10:01:59+00:00"
        with self.assertRaisesRegex(ValueError, "m0: inference predates"):
            verify_fresh_inference(self.headers, self.receipts, self.revision)

    def test_mixed_revision_dirty_source_replay_and_missing_arm_are_refused(self):
        variants = []
        changed = copy.deepcopy(self.headers)
        changed["m1"]["source"]["revision"] = "c" * 40
        variants.append(changed)
        changed = copy.deepcopy(self.headers)
        changed["m2"]["source"]["working_tree_status"] = " M scripts/evaluate_pose.py"
        variants.append(changed)
        changed = copy.deepcopy(self.headers)
        changed["m0"]["replay"] = {}
        variants.append(changed)
        changed = copy.deepcopy(self.headers)
        del changed["m1"]
        variants.append(changed)
        for value in variants:
            with self.subTest(headers=value), self.assertRaises(ValueError):
                verify_fresh_inference(value, self.receipts, self.revision)

    def test_naive_timestamp_cannot_claim_freshness(self):
        self.receipts["candidate"]["created_at_utc"] = "2026-11-10T10:02:00"
        with self.assertRaisesRegex(ValueError, "explicitly identify UTC"):
            verify_fresh_inference(self.headers, self.receipts, self.revision)

    def test_only_registered_annotation_recipes_reach_candidate_verification(self):
        class ReachedCandidateVerification(Exception):
            pass
        root = Path(__file__).resolve().parents[1]
        arguments = (root, Path('/unused/evidence'), 'data/candidates/annotations-2026-11-10/manifest.json')
        for recipe_id in ANNOTATION_RECIPE_IDS:
            with self.subTest(recipe=recipe_id), patch(
                    'verify_pose_annotation_candidate.verify_annotation_candidate',
                    side_effect=ReachedCandidateVerification) as verifier:
                with self.assertRaises(ReachedCandidateVerification):
                    summarize(*arguments, recipe_id, Path('/unused/reference'), self.revision)
                verifier.assert_called_once()
        for recipe_id in (ORIGINAL_RECIPE_ID, 'reviewed-annotations-seed47-v1'):
            with self.subTest(recipe=recipe_id), patch(
                    'verify_pose_annotation_candidate.verify_annotation_candidate') as verifier:
                with self.assertRaisesRegex(ValueError, 'registered annotation-overlay recipe'):
                    summarize(*arguments, recipe_id, Path('/unused/reference'), self.revision)
                verifier.assert_not_called()


class PairedDiagnosticChecks(unittest.TestCase):
    def test_common_points_and_both_corner_identity_keep_misses_in_denominator(self):
        # Two two-corner objects plus one object whose right corner is hidden.
        # M2 recovers object 1 but loses a corner on object 0 and adds an FP.
        targets = [gold(), gold((60, 10, 90, 50), ((65, 45, 2), (85, 45, 2))),
                   gold((10, 60, 50, 90), ((15, 85, 2), (0, 0, 0)))]
        third = prediction((10, 60, 50, 90), ((17, 85, .9), (45, 85, .9)))
        batches = {
            "m0": [prediction(), third],
            "m1": [prediction(points=((19, 45, .9), (53, 45, .9))),
                   prediction((10, 60, 50, 90), ((17, 85, .4), (45, 85, .9)))],
            "m2": [prediction(points=((19, 45, .9), (45, 45, .4))),
                   prediction((60, 10, 90, 50), ((65, 45, .9), (85, 45, .9))), third,
                   prediction((150, 150, 170, 170), ((155, 165, .9), (165, 165, .9)))],
        }
        record = {"id": "fixture", "width": 200, "height": 200}
        points, scores, quality = {}, {}, {}
        for name, predictions in batches.items():
            scores[name] = score_image(targets, predictions, 200, 200, .50, .50)
            points[name] = returned_endpoints(targets, predictions, scores[name], record)
            quality[name] = aggregate_quality([quality_for_image(targets, predictions, 200, 200, [2, 5, 10])])
        result = common_diagnostics(points, {"fixture": targets})
        self.assertEqual((scores["m0"]["fn"], scores["m2"]["fn"], scores["m2"]["fp"]), (1, 0, 1))
        self.assertEqual(scores["m2"]["missing_endpoints"], 1)
        self.assertEqual(result["count"], 1)
        self.assertEqual([result["metrics"][name]["error_px"]["median"] for name in batches], [0, 4, 4])
        self.assertTrue(all(("fixture", 2, 1) not in rows for rows in points.values()))
        self.assertEqual(quality["m2"]["returned_on_unlocatable_reference"], 1)
        paired = result["paired"]["m2_vs_m0"]
        self.assertEqual((paired["common_returned_endpoints"], paired["returned_only_old"], paired["returned_only_new"]), (2, 1, 2))
        for radius in paired["both_corners"]:
            self.assertEqual(radius["eligible_objects"], 2)
            self.assertEqual((radius["old_success"], radius["new_success"], radius["success_both"],
                              radius["became_failure"], radius["became_success"]), (1, 1, 0, 1, 1))
        control = result["paired"]["m2_vs_m1"]["both_corners"]
        self.assertEqual([row["old_success"] for row in control], [0, 1])
        self.assertEqual([row["both_visible_object_success_rate"] for row in quality["m2"]["radii"]], [.5, .5, .5])

    def test_empty_common_subset_serializes_as_null_without_dropping_eligible_objects(self):
        result = common_diagnostics({name: {} for name in ("m0", "m1", "m2")}, {"fixture": [gold()]})
        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["metrics"]["m0"]["error_px"]["median"])
        self.assertEqual(result["paired"]["m2_vs_m1"]["both_corners"][0]["eligible_objects"], 1)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
