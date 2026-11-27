"""Check flip geometry, point identity, baseline protections and exported replay."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_flip_inference_r56 as run
import build_flip_inference_r56_notebook as notebook
from predict_pose_folder import postclip_nms


def fixture():
    gold = [{'class_id': 0, 'bbox_xyxy': [10, 10, 30, 30],
             'keypoints': [[10, 30, 2], [30, 30, 2]]},
            {'class_id': 0, 'bbox_xyxy': [50, 10, 70, 30],
             'keypoints': [[50, 30, 2], [0, 0, 0]]},
            {'class_id': 0, 'bbox_xyxy': [90, 10, 110, 30],
             'keypoints': [[0, 0, 0], [0, 0, 0]]}]
    raw = [{'class_id': 0, 'bbox_xyxy': [10, 10, 30, 30],
            'keypoints': [[10, 30, .9], [30, 30, .8]], 'confidence': .9},
           {'class_id': 0, 'bbox_xyxy': [50, 10, 70, 30],
            'keypoints': [[50, 30, .9], [70, 30, .2]], 'confidence': .8}]
    native = {'control': {'scene': raw, 'empty': []},
              'treatment': {'scene': run.inverse_horizontal_predictions(raw), 'empty': []}}
    arms = {arm: {image: run.original_frame_output(values, arm) for image, values in output.items()}
            for arm, output in native.items()}
    refs = {name: {'scene': deepcopy(gold), 'empty': []}
            for name in ('frozen666', 'historical663', 'r38', 'r50')}
    maps = {'scene': {'rows': [{'stable_id': f'stable:{j+1}'} for j in range(3)]},
            'empty': {'rows': []}}
    return refs, maps, arms, native


def payload_and_context(path, blocked=False):
    refs, maps, arms, native = fixture()
    prior = deepcopy(arms['control'])
    if blocked:
        native = {'control': {'scene': [], 'empty': []}}
        arms = {'control': {im: run.original_frame_output(values, 'control')
                            for im, values in native['control'].items()}}
    baseline = run.baseline_equivalence(refs, prior, arms['control'])
    scored = {arm: run.reports(refs, output) for arm, output in arms.items()}
    cases = run.case_inventory(refs, maps, arms, scored)
    repeated, repeated_native, checks, measurements = [], [], [], []
    if not blocked:
        for repeat in range(1, 4):
            repeated.append(deepcopy(arms)); repeated_native.append(deepcopy(native))
            for arm in ('control', 'treatment'):
                checks.append({'arm': arm, 'repeat': repeat,
                               **run.baseline_equivalence(refs, arms[arm], arms[arm])})
                for image in ('scene', 'empty'):
                    measurements.append({'arm': arm, 'image_id': image, 'repeat': repeat,
                        'input_tensor_shape': [1, 3, 736, 1280], 'input_transform_ms': .1,
                        'predict_call_ms': 1., 'record_and_postclip_ms': .2, 'array_to_output_ms': 1.3,
                        'stage_ms': {'preprocess': .1, 'inference': .8, 'postprocess': .1},
                        'cuda_allocated_before_bytes': None, 'cuda_peak_allocated_bytes': None,
                        'cuda_peak_reserved_bytes': None})
    contract = {'settings': run.settings(), 'synthetic_test': True}
    summary = {'contract': contract, 'status': 'blocked_baseline' if blocked else 'completed',
        'baseline_passed': baseline['passed'], 'repeated_outputs_stable': True,
        'interpretation_allowed': baseline['passed'], 'treatment_executed': not blocked,
        'fresh_inference': True, 'training': False, 'automatic_promotion': False,
        'fusion_performed': False, 'reports': scored}
    context = {'work': path, 'contract': contract, 'preflight': {'synthetic_test': True},
               'receipt': {'synthetic_test': True}, 'records': {'scene': {}, 'empty': {}},
               'prior': prior, 'refs': refs, 'maps': maps}
    payload = {'summary.json': summary, 'predictions.json': arms, 'native-predictions.json': native,
        'baseline-equivalence.json': baseline, 'cases.json': cases,
        'complementarity.json': run.complementarity(cases, scored),
        'timing.json': {'measurements': measurements, 'summary': run.summarize_timing(measurements)},
        'runtime.json': {'synthetic_test': True}, 'repeatability.json': checks,
        'repeated-predictions.json': repeated, 'repeated-native-predictions.json': repeated_native}
    return context, payload


class GeometryTest(unittest.TestCase):
    def test_pixels_reverse_without_mutation_or_shared_storage(self):
        import numpy as np
        image = np.arange(7*13*3, dtype=np.uint8).reshape(7, 13, 3)
        before = image.copy()
        flipped = run.transform_bgr(image, 'horizontal_flip')
        self.assertTrue(np.array_equal(flipped[:, 0, :], before[:, -1, :]))
        self.assertTrue(flipped.flags.c_contiguous)
        self.assertFalse(np.shares_memory(flipped, image))
        self.assertTrue(np.array_equal(run.transform_bgr(flipped, 'horizontal_flip'), before))
        self.assertTrue(np.array_equal(image, before))
        for bad in (image.astype('float32'), image[:, :, 0], image[:0]):
            with self.assertRaises(RuntimeError):
                run.transform_bgr(bad, 'horizontal_flip')

    def test_continuous_inverse_swaps_slots_with_scores_and_is_involutive(self):
        source = [{'class_id': 0, 'confidence': .73, 'bbox_xyxy': [0., 3., 20.5, 18.],
                   'keypoints': [[0., 17., .91], [18.5, 16., .27]], 'identity': 'keep'}]
        before = deepcopy(source)
        output = run.inverse_horizontal_predictions(source, 100)
        self.assertEqual(output[0]['bbox_xyxy'], [79.5, 3., 100., 18.])
        self.assertEqual(output[0]['keypoints'], [[81.5, 16., .27], [100., 17., .91]])
        self.assertEqual(output[0]['confidence'], .73)
        self.assertEqual(output[0]['identity'], 'keep')
        self.assertEqual(run.inverse_horizontal_predictions(output, 100), source)
        self.assertEqual(source, before)
        source[0]['keypoints'][1][0] = None
        restored = run.inverse_horizontal_predictions(source, 100)
        self.assertIsNone(restored[0]['keypoints'][0][0])
        self.assertEqual(restored[0]['keypoints'][0][2], .27)

    def test_native_clipped_bounds_are_mapped_before_postclip_nms(self):
        native = [{'class_id': 0, 'confidence': confidence, 'bbox_xyxy': [0, 0, edge, 10],
                   'keypoints': [[0, 9, .9], [edge, 9, .4]]}
                  for edge, confidence in ((20, .9), (18, .8))]
        output = run.original_frame_output(native, 'treatment')
        self.assertEqual(output['raw'][0]['bbox_xyxy'], [1260, 0, 1280, 10])
        self.assertEqual(output['raw'][0]['keypoints'][1], [1280, 9, .9])
        self.assertEqual(output['filter']['retained_primary_prediction_ids'], [1])
        suppressed = output['filter']['suppressed_predictions'][0]
        self.assertEqual(suppressed['prediction']['bbox_xyxy'], [1262, 0, 1280, 10])
        self.assertEqual(suppressed['primary_prediction_id'], 2)

    def test_small_numeric_drift_cannot_cross_a_corner_success_boundary(self):
        refs, _, arms, _ = fixture()
        before = deepcopy(arms['control'])
        before['scene']['raw'][0]['keypoints'][0][0] = 14.99
        before['scene']['postclip'], before['scene']['filter'] = postclip_nms(before['scene']['raw'])
        after = deepcopy(before)
        after['scene']['raw'][0]['keypoints'][0][0] = 15.01
        after['scene']['postclip'], after['scene']['filter'] = postclip_nms(after['scene']['raw'])
        gate = run.baseline_equivalence(refs, before, after)
        self.assertFalse(gate['passed'])
        self.assertLess(gate['maximum_drift']['coordinate_component_px'], .05)
        self.assertTrue(any('endpoint_threshold_state' in reason for reason in gate['failures']))


class EvidenceTest(unittest.TestCase):
    def test_completed_synthetic_archive_replays_and_rejects_wrong_inverse_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            context, payload = payload_and_context(Path(tmp))
            result = run.package(context, payload)
            self.assertEqual(result['status'], 'completed')
            archive = Path(tmp)/run.archive_name()
            before = hashlib.sha256(archive.read_bytes()).hexdigest()
            run.verify_return(context)
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), before)
            with zipfile.ZipFile(archive) as z:
                members = {name: z.read(name) for name in z.namelist()}
            native = json.loads(members['native-predictions.json'])
            native['treatment']['scene'][0]['keypoints'][0][0] += 1
            members['native-predictions.json'] = run.encoded(native)
            inventory = json.loads(members['inventory.json'])
            data = members['native-predictions.json']
            inventory['native-predictions.json'] = {'bytes': len(data), 'sha256': run.sha(data)}
            members['inventory.json'] = run.encoded(inventory)
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
                for name, data in members.items():
                    z.writestr(name, data)
            with self.assertRaisesRegex(RuntimeError, 'Inverse mapping'):
                run.verify_return(context)

    def test_failed_baseline_has_no_treatment_and_remains_replayable(self):
        with tempfile.TemporaryDirectory() as tmp:
            context, payload = payload_and_context(Path(tmp), blocked=True)
            result = run.package(context, payload)
            self.assertEqual(result['status'], 'blocked_baseline')
            self.assertFalse(result['interpretation_allowed'])
            self.assertFalse(result['treatment_executed'])
            self.assertEqual(set(payload['native-predictions.json']), {'control'})

    def test_complementarity_counts_reference_identities_and_empty_scenes(self):
        refs, maps, arms, native = fixture()
        extra = {'class_id': 0, 'bbox_xyxy': [90, 10, 110, 30],
                 'keypoints': [[90, 30, .1], [110, 30, .2]], 'confidence': .6}
        native['treatment']['scene'] += run.inverse_horizontal_predictions([extra])
        native['treatment']['scene'] = native['treatment']['scene'][1:]
        arms['treatment']['scene'] = run.original_frame_output(native['treatment']['scene'], 'treatment')
        scored = {arm: run.reports(refs, output) for arm, output in arms.items()}
        result = run.complementarity(run.case_inventory(refs, maps, arms, scored), scored)
        r50 = result['references']['r50']
        group = r50['by_locatable_corner_count']['all']
        self.assertEqual(group['control_only_case_ids'], ['r50/scene/R001'])
        self.assertEqual(group['flip_only_case_ids'], ['r50/scene/R003'])
        self.assertEqual(group['matched_by_either_diagnostic_bound'], 3)
        self.assertEqual(group['missed_by_both'], 0)
        self.assertEqual(r50['reference_empty_scene_outputs']['empty'], {'control': 0, 'treatment': 0})
        self.assertFalse(result['fusion_performed'])

    def test_notebook_has_four_compilable_cells_and_explicit_protocol(self):
        result = notebook.build('a'*40, 'b'*64)
        codes = [''.join(c['source']) for c in result['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(codes), 4)
        for code in codes:
            compile(code, '<r56-notebook>', 'exec')
        self.assertIn('run_flip_inference_r56.py', codes[0])
        self.assertIn('\\n', codes[2])
        self.assertEqual(result['metadata']['experiment'], 'flip')
        self.assertFalse(result['metadata']['training'])
        self.assertFalse(result['metadata']['fusion_performed'])
        with self.assertRaises(ValueError):
            notebook.build('main', 'b'*64)


if __name__ == '__main__':
    unittest.main()
