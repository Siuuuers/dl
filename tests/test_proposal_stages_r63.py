import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_proposal_stages_r63 import (
    eligibility_graph, iou_matrix, record_difference, stage_label)
from pose_scoring import _box_iou
from replay_front_confidence_r51 import native_select


class ProposalStagesTest(unittest.TestCase):
    def test_iou_boundary_and_exact_scorer_formula(self):
        boxes = np.array([[0., 0., 10., 10.], [1.2, 3.4, 18.6, 23.7], [0., 0., 0., 10.]])
        gold = [{'bbox_xyxy': [0., 0., 20., 10.]}, {'bbox_xyxy': [0., 0., 20.00000001, 10.]}]
        overlaps = iou_matrix(boxes, gold)
        expected = np.array([[_box_iou(box.tolist(), item['bbox_xyxy']) for item in gold] for box in boxes])
        np.testing.assert_array_equal(overlaps, expected)
        self.assertEqual(overlaps[0, 0], .5)
        self.assertLess(overlaps[0, 1], .5)
        self.assertEqual(eligibility_graph(overlaps, [0])['references_with_any_candidate'], 1)

    def test_strict_confidence_gate_and_distinct_suppression_stages(self):
        above = np.nextafter(np.float32(.25), np.float32(1))
        scores = np.array([.25, above], dtype=np.float32)
        self.assertEqual(stage_label([], scores, [], []), 'no_iou_eligible_decoded_box')
        self.assertEqual(stage_label([0], scores, [], []), 'confidence_gate')
        self.assertEqual(stage_label([1], scores, [], []), 'ordinary_nms')
        self.assertEqual(stage_label([1], scores, [1], []), 'postclip_suppression')
        self.assertEqual(stage_label([1], scores, [1], [1]), 'retained_assignment_contention')

    def test_native_nms_boundary_is_strict_and_preserves_suppressor(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 7, 10], [0, 0, 8, 10]], dtype=np.float32)
        scores = np.array([.9, .8, .7], dtype=np.float32)
        kept, suppressed = native_select(boxes, scores, .25)
        self.assertEqual(kept, [0, 1])
        self.assertEqual(set(suppressed), {2})
        self.assertEqual(suppressed[2]['suppressor'], 0)
        self.assertGreater(suppressed[2]['pair_iou_fp32'], .7)

    def test_shared_candidate_is_not_two_recoveries(self):
        result = eligibility_graph(np.array([[.7, .8]]), [0])
        self.assertEqual(result['references_with_any_candidate'], 2)
        self.assertEqual(result['maximum_one_to_one_matches'], 1)
        self.assertEqual(result['candidates_eligible_for_multiple_references'], 1)

    def test_many_proposals_are_not_many_matched_fronts(self):
        overlaps = np.array([[.9], [.8], [.7]])
        original = overlaps.copy()
        result = eligibility_graph(overlaps, [0, 1, 2])
        self.assertEqual(result['eligible_candidate_count'], 3)
        self.assertEqual(result['references_with_multiple_candidates'], 1)
        self.assertEqual(result['maximum_one_to_one_matches'], 1)
        np.testing.assert_array_equal(overlaps, original)

    def test_maximum_matching_can_reassign_an_ambiguous_candidate(self):
        result = eligibility_graph(np.array([[.9, .6], [.8, .1]]), [0, 1])
        self.assertEqual(result['maximum_one_to_one_matches'], 2)
        self.assertEqual(result['references_with_any_candidate'], 2)

    def test_empty_image_and_no_candidates(self):
        empty_gold = eligibility_graph(np.empty((3, 0)), [0, 1, 2])
        self.assertEqual(empty_gold['candidate_count'], 3)
        self.assertEqual(empty_gold['maximum_one_to_one_matches'], 0)
        self.assertEqual(empty_gold['uncovered_reference_indices'], [])
        no_candidates = eligibility_graph(np.empty((0, 2)), [])
        self.assertEqual(no_candidates['uncovered_reference_indices'], [0, 1])
        self.assertEqual(no_candidates['maximum_one_to_one_matches'], 0)

    def test_cross_runtime_comparison_is_bounded_not_claimed_exact(self):
        prediction = {'class_id': 0, 'bbox_xyxy': [0., 0., 20., 10.], 'confidence': .8,
                      'keypoints': [[1., 9., .9], [19., 9., .9]]}
        changed = copy.deepcopy(prediction)
        changed['bbox_xyxy'][0] += .0001
        changed['keypoints'][1][2] += .000001
        result = record_difference([prediction], [changed])
        self.assertGreater(result['maximum_coordinate_difference_px'], 0)
        self.assertGreater(result['maximum_score_difference'], 0)
        changed['confidence'] = .4
        with self.assertRaisesRegex(ValueError, 'tolerances'):
            record_difference([prediction], [changed])


if __name__ == '__main__':
    unittest.main()
