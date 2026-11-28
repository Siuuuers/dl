"""Protect correction-origin identity, target eligibility, masks and return gates."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from PIL import Image
import residual_corner_r59 as correction
import run_residual_corner_r59 as run
import build_residual_corner_r59_notebook as notebook
from predict_pose_folder import postclip_nms


def record(box=(20., 30., 80., 50.), points=None, confidence=.9):
    return {'class_id': 0, 'confidence': confidence, 'bbox_xyxy': list(box),
            'keypoints': points if points is not None else [[20., 50., .8], [80., 50., .9]]}


def output(raw):
    kept, audit = postclip_nms(raw)
    return {'raw': raw, 'postclip': kept, 'filter': audit}


class CorrectionTest(unittest.TestCase):
    def test_zero_offsets_preserve_original_serialized_coordinates_and_scores(self):
        for points in ([[20.123456789, 50.333333333, .8], [80., 50., .9]],
                       [[-0.0, 0., .8], [None, None, .4]],
                       [[20., 50., .499999], [1279.999, 719.999, .5]]):
            original = record(points=points)
            geom = correction.crop_geometry(original['bbox_xyxy'])
            actual = correction.apply_corrections(original, [[0., -0.0], [-0.0, 0.]], geom)
            self.assertEqual(run.encoded(actual), run.encoded(original))
            self.assertIsNot(actual, original)

    def test_target_delta_and_application_use_actual_independent_xy_scales(self):
        original = record(box=(240.2, 180.7, 423.4, 291.1),
                          points=[[250.17, 280.23, .9], [411.33, 282.75, .9]])
        gold = [[252.75, 279.81, 2], [409.21, 286.46, 2]]
        geom = correction.crop_geometry(original['bbox_xyxy'])
        self.assertNotEqual(*geom['scale_xy'])
        target, available, states = correction.correction_targets(gold, original, geom)
        self.assertEqual(available.tolist(), [1., 1.])
        self.assertEqual(states, ['supervised', 'supervised'])
        restored = correction.apply_corrections(original, target.tolist(), geom)
        np.testing.assert_allclose(np.array(restored['keypoints'])[:, :2], np.array(gold)[:, :2], atol=1e-6, rtol=0)
        # Applying an offset must never rescale the original coordinates or padding.
        moved = correction.apply_corrections(original, [[geom['scale_xy'][0], -geom['scale_xy'][1]], [0., 0.]], geom)
        self.assertEqual(moved['keypoints'][0], [original['keypoints'][0][0]+1., original['keypoints'][0][1]-1., .9])

    def test_unknown_reference_and_unavailable_original_have_distinct_masks(self):
        original = record(points=[[20., 50., .49], [80., 50., .9]])
        target, available, states = correction.correction_targets([[22., 49., 2], [0., 0., 0]], original,
                                                                  correction.crop_geometry(original['bbox_xyxy']))
        self.assertEqual(states, ['initial_point_unavailable', 'reference_unavailable'])
        self.assertFalse(available.any())
        self.assertFalse(target.any())
        original['keypoints'][0][2] = .5
        target, available, _ = correction.correction_targets([[22., 49., 2], [0., 0., 0]], original,
                                                             correction.crop_geometry(original['bbox_xyxy']))
        self.assertEqual(available.tolist(), [1., 0.])

    def test_known_outside_crop_is_a_valid_unconstrained_regression_target(self):
        original = record()
        geom = correction.crop_geometry(original['bbox_xyxy'])
        target, available, states = correction.correction_targets([[200., 100., 2], [0., 0., 0]], original, geom)
        self.assertEqual(available.tolist(), [1., 0.])
        self.assertEqual(states, ['supervised', 'reference_unavailable'])
        corrected = correction.apply_corrections(original, target.tolist(), geom)
        np.testing.assert_allclose(corrected['keypoints'][0][:2], [200., 100.], atol=1e-5, rtol=0)

    def test_unavailable_inputs_stay_unchanged_and_invalid_new_points_are_accounted(self):
        original = record(points=[[None, None, .9], [80., 50., .9]])
        geom = correction.crop_geometry(original['bbox_xyxy'])
        result = correction.apply_corrections(original, [[200., 300.], [1e6, 0.]], geom)
        self.assertEqual(result['keypoints'], [[None, None, .9], [None, None, .9]])
        self.assertEqual(run.correction_states(original, result), ['initial_unavailable_preserved', 'invalid_corrected_point'])
        original = record(points=[[20., 50., .49], [80., 50., .9]])
        result = correction.apply_corrections(original, [[1e6, 1e6], [None, 1.]], geom)
        self.assertEqual(result['keypoints'][0], original['keypoints'][0])
        self.assertEqual(result['keypoints'][1], [None, None, .9])
        self.assertEqual(original['keypoints'][1], [80., 50., .9])

    def test_hint_input_has_no_training_noise_and_keeps_original_rgb(self):
        original = record()
        geom = correction.crop_geometry(original['bbox_xyxy'])
        image = Image.new('RGB', (1280, 720), (60, 80, 90))
        actual = correction.input_tensor(image, original, geom, 'one')
        other = correction.input_tensor(image, original, geom, 'different identity')
        expected = correction.crop.crop_tensor(image, original, geom, 'initial', epoch=0)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, other)

    def test_loss_masks_both_components_and_normalizes_only_available_endpoints(self):
        import torch
        predicted = torch.tensor([[[2., -.5], [10., -10.]]], requires_grad=True)
        loss = correction.masked_correction_loss(predicted, torch.zeros_like(predicted), torch.tensor([[1., 0.]]))
        self.assertAlmostEqual(loss.item(), (1.5+.125)/2)
        loss.backward()
        self.assertEqual(predicted.grad[0, 1].tolist(), [0., 0.])
        self.assertEqual(predicted.grad[0, 0].tolist(), [.5, -.25])
        with self.assertRaises(ValueError):
            correction.masked_correction_loss(predicted, torch.zeros_like(predicted), torch.zeros(1, 2))


class AuditTest(unittest.TestCase):
    def test_correction_replay_requires_all_proposals_and_exact_origins(self):
        control = {'scene': output([record(), record(confidence=.8)])}
        raw, details = [], []
        for index, original in enumerate(control['scene']['raw']):
            geom = correction.crop_geometry(original['bbox_xyxy'])
            delta = [[.25, -.5], [.5, 1.]]
            refined = correction.apply_corrections(original, delta, geom)
            raw.append(refined)
            details.append({'image_id': 'scene', 'raw_index': index, 'geometry': geom,
                            'original_points': original['keypoints'], 'canvas_offsets': delta,
                            'output_points': refined['keypoints'], 'endpoint_states': run.correction_states(original, refined)})
        treatment = {'scene': output(raw)}
        run.replay_transforms(control, treatment, details)
        run.shared.frozen_detection_check(control, treatment)
        self.assertEqual(run.zero_output_gate(control)['raw_records_checked'], 2)
        with self.assertRaises(RuntimeError):
            run.replay_transforms(control, treatment, details[:1])
        altered = deepcopy(details)
        altered[0]['original_points'][0][0] += 1
        with self.assertRaises(RuntimeError):
            run.replay_transforms(control, treatment, altered)

    def test_common_threshold_is_strict_and_replays_suppression(self):
        controls = {'scene': output([record(confidence=.25), record(confidence=.250001)])}
        result = run.at_confidence(controls)
        self.assertEqual(len(result['scene']['raw']), 1)
        self.assertEqual(result['scene']['raw'][0]['confidence'], .250001)
        run.shared.replay_postclip(result)

    def test_failed_control_never_creates_model_or_training_claim_and_archive_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            gold = [{'class_id': 0, 'bbox_xyxy': [20, 30, 80, 50], 'keypoints': [[20, 50, 2], [80, 50, 2]]}]
            refs, prior = {'reference': {'scene': gold}}, {'scene': output([record()])}
            control = {'scene': output([])}
            gate = run.r53.baseline_equivalence(refs, prior, control)
            self.assertFalse(gate['passed'])
            fixture = {'synthetic_test': True}
            context = {'work': work, 'contract': fixture, 'preflight': fixture, 'receipt': fixture,
                       'refs': refs, 'prior': prior}
            payloads = {'baseline-equivalence.json': gate, 'detector-evaluation.json': control}
            payloads.update({name: fixture for name in ('preflight.json', 'input-audit.json', 'staging-r58.json',
                                                       'detector-runtime.json', 'model-smoke.json')})
            for name, value in payloads.items():
                run.frozen(work/name, value)
            summary = run.blocked_package(context, control)
            self.assertFalse(summary['training'])
            self.assertFalse(summary['interpretation_allowed'])
            manifest = json.loads((work/'return-manifest.json').read_text())
            self.assertEqual(manifest['parts'], [])
            bundle = work/run.UPLOAD_NAME
            bundle.unlink()
            run.verify(context)
            with zipfile.ZipFile(bundle) as stream:
                self.assertEqual(len(stream.namelist()), 2)
            evidence = work/manifest['evidence_file']
            evidence.write_bytes(evidence.read_bytes()+b'modified')
            with self.assertRaises(RuntimeError):
                run.verify(context)

    def test_notebook_compiles_and_requires_immutable_pins(self):
        content = notebook.build('a'*40, 'b'*64)
        codes = [cell for cell in content['cells'] if cell['cell_type'] == 'code']
        self.assertEqual(len(codes), 4)
        for cell in codes:
            compile(''.join(cell['source']), '<residual-notebook>', 'exec')
        self.assertTrue(content['metadata']['training'])
        self.assertFalse(content['metadata']['automatic_promotion'])
        with self.assertRaises(ValueError):
            notebook.build('main', 'b'*64)


if __name__ == '__main__':
    unittest.main()
