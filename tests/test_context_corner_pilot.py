"""Focused invariants for the geometric diagnostic, not model-quality tests."""

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/context_corner_pilot.py"
SPEC = importlib.util.spec_from_file_location("context_corner_pilot", SCRIPT)
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


class ContextCornerTests(unittest.TestCase):
    def setUp(self):
        self.h = np.array([[80., 8., 100.], [2., 35., 60.], [.03, .07, 1.]])

    def projected_pair(self, rank):
        points = (self.h @ np.array([[0, rank, 1], [1, rank, 1]]).T).T
        return points[:, :2] / points[:, 2, None]

    def test_known_projective_interpolation(self):
        result = PILOT.estimate_pair("homography", self.projected_pair(0), self.projected_pair(1), .4)
        np.testing.assert_allclose(result, self.projected_pair(.4), atol=1e-9, rtol=0)

    def test_known_projective_extrapolation(self):
        for t in (2., 3.):
            result = PILOT.estimate_pair("homography", self.projected_pair(0), self.projected_pair(1), t,
                                         allow_extrapolation=True)
            np.testing.assert_allclose(result, self.projected_pair(t), atol=1e-8, rtol=0)

    def test_linear_known_affine_case(self):
        above = np.array([[10., 20.], [90., 24.]])
        below = above + np.array([6., 40.])
        np.testing.assert_allclose(PILOT.estimate_pair("linear", above, below, .25), above + [1.5, 10.])

    def test_degenerate_and_crossed_geometry_abstains(self):
        above = [[10., 20.], [90., 20.]]
        for below in (above, [[0., 20.], [100., 20.]], [[10., 30.], [90., 10.]]):
            for method in PILOT.METHODS:
                with self.assertRaises(PILOT.PilotError):
                    PILOT.estimate_pair(method, above, below, .5)

    def test_no_implicit_or_unbounded_extrapolation(self):
        for method in PILOT.METHODS:
            for t in (-1, 0, 1, 2, 3):
                with self.assertRaises(PILOT.PilotError):
                    PILOT.estimate_pair(method, self.projected_pair(0), self.projected_pair(1), t)
            for t in (-1, 3.01):
                with self.assertRaises(PILOT.PilotError):
                    PILOT.estimate_pair(method, self.projected_pair(0), self.projected_pair(1), t,
                                         allow_extrapolation=True)

    def test_target_coordinates_and_bbox_cannot_enter_estimator(self):
        self.assertEqual(list(inspect.signature(PILOT.estimate_pair).parameters),
                         ["method", "earlier_xy", "later_xy", "t", "allow_extrapolation"])
        stack = {"members": [{"row": 1, "rank": 0}, {"row": 2, "rank": 1}, {"row": 3, "rank": 2}]}
        data = {"rows": {1: {"visible": np.array([True, True]), "xy": self.projected_pair(0)},
                         2: {"visible": None, "xy": object(), "bbox": object()},
                         3: {"visible": np.array([True, True]), "xy": self.projected_pair(1)}}}
        # Poisoned target fields would fail if donor selection read the held-out object.
        donors, t = PILOT.select_donors(stack, data, 1, {2})
        self.assertEqual([donor["row"] for donor in donors], [1, 3])
        self.assertEqual(t, .5)
        result = PILOT.estimate_pair("homography", donors[0]["xy"], donors[1]["xy"], t)
        np.testing.assert_allclose(result, self.projected_pair(.5), atol=1e-9, rtol=0)
        with self.assertRaises(TypeError):
            PILOT.estimate_pair("linear", donors[0]["xy"], donors[1]["xy"], t, target_bbox=[0, 0, 1, 1])

    def test_extrapolation_holds_out_both_target_rows(self):
        stack = {"members": [{"row": i + 1, "rank": i} for i in range(4)]}
        data = {"rows": {1: {"visible": np.array([True, True]), "xy": self.projected_pair(0)},
                         2: {"visible": np.array([True, True]), "xy": self.projected_pair(1)},
                         3: {"visible": None, "xy": object()}, 4: {"visible": None, "xy": object()}}}
        donors, t = PILOT.select_donors(stack, data, 3, {3, 4}, extrapolation=True, first_rank=2)
        self.assertEqual([donor["row"] for donor in donors], [1, 2])
        self.assertEqual(t, 3)

    def test_hidden_case_requires_exact_declared_row_rank(self):
        stack = {"id": "stack", "image_id": "example", "members": [
            {"row": 1, "rank": 0}, {"row": 2, "rank": 1}, {"row": 3, "rank": 2}]}
        data = {"example": {"width": 500, "height": 500, "rows": {
            1: {"visible": np.array([True, True]), "xy": self.projected_pair(0)},
            2: {"visible": np.array([True, True]), "xy": self.projected_pair(1)},
            3: {"visible": np.array([False, True]), "xy": self.projected_pair(2)}}}}
        case = {"image_id": "example", "row": 3, "rank": 3,
                "stack_id": "stack", "disposition": "estimate_extrapolation"}
        with self.assertRaisesRegex(PILOT.PilotError, "row/rank"):
            PILOT.evaluate_hidden(case, {"stack": stack}, data)

    def test_reference_image_and_label_identities(self):
        def chunk(name, payload):
            return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload))
        image = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        image += chunk(b"IDAT", zlib.compress(b"\0" * 14)) + chunk(b"IEND", b"")
        labels = b"0 0.5 0.5 0.8 0.8 0.2 0.6 2 0.8 0.6 2\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ip = root / "data/images/val/example.png"; ip.parent.mkdir(parents=True); ip.write_bytes(image)
            lp = root / "reference/labels/example.txt"; lp.parent.mkdir(parents=True); lp.write_bytes(labels)
            manifest = {"files": [{"image_id": "example", "image_sha256": hashlib.sha256(image).hexdigest(),
                                  "width": 2, "height": 2, "label_path": "labels/example.txt",
                                  "label_sha256": hashlib.sha256(labels).hexdigest()}]}
            mp = root / "reference/manifest.json"; mp.write_text(json.dumps(manifest))
            cohort = {"reference_manifest": "reference/manifest.json", "reference_manifest_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
                      "stacks": [{"image_id": "example"}]}
            loaded, identities = PILOT.load_reference(root, cohort)
            self.assertEqual(loaded["example"]["width"], 2)
            self.assertEqual(len(identities), 1)
            lp.write_bytes(labels + b"\n")
            with self.assertRaisesRegex(PILOT.PilotError, "label hash mismatch"):
                PILOT.load_reference(root, cohort)
            lp.write_bytes(labels); ip.write_bytes(b"not an image")
            with self.assertRaisesRegex(PILOT.PilotError, "image hash mismatch"):
                PILOT.load_reference(root, cohort)


if __name__ == "__main__":
    unittest.main()
