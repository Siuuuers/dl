"""Focused data-boundary and availability invariants for model anchoring."""

import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("context_corner_model_anchors", ROOT / "scripts/context_corner_model_anchors.py")
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


class ModelAnchorTests(unittest.TestCase):
    def setUp(self):
        self.earlier = [[20., 20., .9], [80., 20., .8]]
        self.later = [[20., 40., .7], [80., 40., .6]]
        self.target = [[21., 32., .91], [82.123456789, 31.987654321, .87654321]]
        self.dimensions = (100, 100)

    def outputs(self, target=None, earlier=None, later=None, t=.5, **kwargs):
        return PILOT.evaluate_outputs(self.target if target is None else target,
            self.earlier if earlier is None else earlier, self.later if later is None else later,
            t, self.dimensions, **kwargs)

    def test_right_anchor_preserves_exact_direct_right_and_null_left_confidence(self):
        outputs = self.outputs()
        anchored = outputs["right_anchored_homography"]
        self.assertEqual(anchored["endpoints"]["right"], outputs["direct_pr35"]["endpoints"]["right"])
        self.assertEqual(anchored["endpoints"]["right"]["xy"], self.target[1][:2])
        self.assertEqual(anchored["endpoints"]["right"]["confidence"], self.target[1][2])
        self.assertIsNone(anchored["endpoints"]["left"]["confidence"])
        np.testing.assert_allclose(anchored["endpoints"]["left"]["xy"],
                                   np.array(self.target[1][:2]) + [-60, 0], atol=1e-10, rtol=0)

    def test_target_left_poison_changes_only_the_direct_baseline(self):
        original = self.outputs()
        changed = copy.deepcopy(self.target)
        changed[0] = [float("nan"), float("nan"), .0001]
        poisoned = self.outputs(target=changed)
        for strategy in ("model_donor_homography", "right_anchored_homography"):
            self.assertEqual(original[strategy], poisoned[strategy])
        self.assertFalse(poisoned["direct_pr35"]["endpoints"]["left"]["available"])
        self.assertNotIn("target", inspect.signature(PILOT.model_donor_homography).parameters)
        self.assertEqual(list(inspect.signature(PILOT.anchor_to_right).parameters),
                         ["geometric_result", "target_right_keypoint", "dimensions", "confidence_threshold"])

    def test_designated_donor_confidence_gate_preserves_available_target_right(self):
        for confidence, expected in ((.499999, False), (.5, True)):
            donor = copy.deepcopy(self.earlier)
            donor[0][2] = confidence
            result = self.outputs(earlier=donor)
            raw = result["model_donor_homography"]
            anchored = result["right_anchored_homography"]
            self.assertEqual(raw["structurally_valid_geometry"], expected)
            self.assertEqual(anchored["endpoints"]["left"]["available"], expected)
            self.assertTrue(anchored["endpoints"]["right"]["available"])
            self.assertEqual(anchored["endpoints"]["right"]["xy"], self.target[1][:2])

    def test_target_left_missing_does_not_disable_anchor_or_create_hidden_accuracy(self):
        target = [None, self.target[1]]
        result = self.outputs(target=target)
        self.assertTrue(result["right_anchored_homography"]["endpoints"]["left"]["available"])
        PILOT.score_outputs(result, {"visible": np.array([False, True]),
                                     "xy": np.array([[0., 0.], [82., 32.]])})
        for output in result.values():
            left = output["endpoints"]["left"]
            self.assertIsNone(left["error_px"])
            self.assertIsNone(left["within_radius"]["5"])
            self.assertIsNone(output["both_within_radius"]["5"])

    def test_structural_geometry_failure_retains_right(self):
        outputs = self.outputs(later=self.earlier)
        self.assertFalse(outputs["model_donor_homography"]["structurally_valid_geometry"])
        anchored = outputs["right_anchored_homography"]
        self.assertFalse(anchored["endpoints"]["left"]["available"])
        self.assertTrue(anchored["endpoints"]["right"]["available"])

    def test_out_of_frame_raw_geometry_can_supply_an_in_frame_anchored_vector(self):
        earlier = [[20., 70., .9], [80., 70., .9]]
        later = [[20., 90., .9], [80., 90., .9]]
        outputs = self.outputs(earlier=earlier, later=later, t=2., allow_extrapolation=True)
        raw = outputs["model_donor_homography"]
        self.assertTrue(raw["structurally_valid_geometry"])
        self.assertFalse(raw["endpoints"]["left"]["available"])
        self.assertFalse(raw["endpoints"]["right"]["available"])
        self.assertTrue(outputs["right_anchored_homography"]["endpoints"]["left"]["available"])
        np.testing.assert_allclose(raw["raw_pair_xy"], [[20., 110.], [80., 110.]], atol=1e-10, rtol=0)

    def test_final_points_use_half_open_frame_without_clipping(self):
        for point in ([100., 50., .9], [50., 100., .9], [-.001, 50., .9]):
            result = PILOT.direct_endpoint(point, self.dimensions)
            self.assertFalse(result["available"])
            self.assertEqual(result["xy"], point[:2])
        target = [[0., 0., .9], [20., 30., .9]]
        result = self.outputs(target=target)["right_anchored_homography"]
        self.assertFalse(result["endpoints"]["left"]["available"])
        self.assertTrue(result["endpoints"]["right"]["available"])
        self.assertAlmostEqual(result["endpoints"]["left"]["xy"][0], -40., places=8)

    def test_frozen_input_hash_mismatch_stops_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "inputs.json"
            path.write_text(json.dumps({"key": "frozen"}))
            digest = PILOT.geometry.sha256(path.read_bytes())
            value, identity = PILOT.read_pinned_json(root, "inputs.json", digest)
            self.assertEqual(value, {"key": "frozen"})
            self.assertEqual(identity["sha256"], digest)
            path.write_bytes(b"not even valid JSON")
            with self.assertRaisesRegex(PILOT.PilotError, "hash mismatch"):
                PILOT.read_pinned_json(root, "inputs.json", digest)

    def test_failures_stay_in_success_denominators_and_paired_losses(self):
        good = self.outputs()
        bad_donor = copy.deepcopy(self.earlier)
        bad_donor[0][2] = .1
        bad = self.outputs(earlier=bad_donor)
        target = {"visible": np.array([True, True]), "xy": np.array([self.target[0][:2], self.target[1][:2]])}
        for output in (good, bad):
            PILOT.score_outputs(output, target)
        cases = [{"case_id": str(index), "strategies": output} for index, output in enumerate((good, bad))]
        summary = PILOT.summarize(cases)
        self.assertEqual(summary["target_pair_denominator"], 2)
        anchored = summary["strategies"]["right_anchored_homography"]
        self.assertEqual(anchored["slots"]["left"]["available"], 1)
        self.assertEqual(anchored["slots"]["right"]["available"], 2)
        comparison = summary["paired_comparisons"][1]["scopes"]["left"]
        self.assertEqual(comparison["at_radius"]["5"]["loss_case_ids"], ["1"])


if __name__ == "__main__":
    unittest.main()
