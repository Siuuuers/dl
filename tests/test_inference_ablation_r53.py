"""Verify the actual risks: unchanged geometry, score gates, IDs and export replay."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_inference_ablation_r53 as run
import build_inference_ablation_r53_notebooks as notebook
from predict_pose_folder import postclip_nms


def target(box, points):
    return {'class_id': 0, 'bbox_xyxy': box, 'keypoints': points}


def prediction(box, points, confidence=.9):
    return {'class_id': 0, 'bbox_xyxy': box, 'keypoints': points, 'confidence': confidence}


def fixture():
    gold = [target([10,10,30,30], [[10,30,2],[30,30,2]]),
            target([50,10,70,30], [[50,30,2],[0,0,0]]),
            target([90,10,110,30], [[0,0,0],[0,0,0]])]
    raw = [prediction([10,10,30,30], [[10,30,.9],[30,30,.9]], .9),
           prediction([50,10,70,30], [[50,30,.9],[70,30,.2]], .8),
           prediction([16,10,36,30], [[16,30,.9],[36,30,.9]], .7)]
    kept, audit = postclip_nms(raw)
    output = {'test': {'raw': raw, 'postclip': kept, 'filter': audit}}
    refs = {name: {'test': deepcopy(gold)} for name in ('frozen666','historical663','r38','r50')}
    maps = {'test': {'rows': [{'stable_id': f'stable:{j+1}', 'current_row': j+1} for j in range(3)]}}
    return refs, maps, output


class PreprocessingTest(unittest.TestCase):
    def test_actual_lab_transform_keeps_geometry_and_source_unchanged(self):
        import cv2
        import numpy as np
        image = np.zeros((73,127,3), dtype=np.uint8)
        image[:,:] = [18,31,47]
        image[11:61,17:101] = np.arange(84, dtype=np.uint8)[None,:,None] + 20
        before = image.copy()
        identity = run.preprocess_bgr(image, 'identity')
        self.assertTrue(np.array_equal(identity, image))
        self.assertFalse(np.shares_memory(identity, image))
        changed = run.preprocess_bgr(image, 'lab_l_clahe')
        self.assertEqual(changed.shape, image.shape)
        self.assertEqual(changed.dtype, np.uint8)
        self.assertTrue(changed.flags.c_contiguous)
        self.assertTrue(np.array_equal(image, before))
        self.assertFalse(np.array_equal(changed, image))
        self.assertTrue(np.array_equal(changed, run.preprocess_bgr(image, 'lab_l_clahe')))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        original_ab = lab[:,:,1:].copy()
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(lab[:,:,0])
        self.assertTrue(np.array_equal(lab[:,:,1:], original_ab))
        self.assertTrue(np.array_equal(changed, cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)))

    def test_dtype_and_channel_errors_are_rejected(self):
        import numpy as np
        for bad in (np.zeros((8,8),dtype=np.uint8), np.zeros((8,8,3),dtype=np.float32),
                    np.zeros((0,8,3),dtype=np.uint8)):
            with self.assertRaises(RuntimeError):
                run.preprocess_bgr(bad,'lab_l_clahe')


class ScoringContractTest(unittest.TestCase):
    def test_same_cache_and_small_float_noise_pass_without_rounding(self):
        refs, _, before = fixture()
        self.assertTrue(run.baseline_equivalence(refs,before,before)['exact_json_equal'])
        after = deepcopy(before)
        for p in after['test']['raw']:
            p['confidence'] += .000001
            p['bbox_xyxy'] = [x+.001 for x in p['bbox_xyxy']]
            p['keypoints'] = [[x+.001,y+.001,c] for x,y,c in p['keypoints']]
        after['test']['postclip'], after['test']['filter'] = postclip_nms(after['test']['raw'])
        check = run.baseline_equivalence(refs,before,after)
        self.assertTrue(check['passed'])
        self.assertFalse(check['exact_json_equal'])
        self.assertGreater(check['maximum_drift']['coordinate_component_px'],0)

    def test_subpixel_threshold_crossing_fails_even_within_numeric_tolerance(self):
        refs, _, before = fixture()
        before['test']['raw'][0]['keypoints'][0][0] = 14.99
        before['test']['postclip'],before['test']['filter'] = postclip_nms(before['test']['raw'])
        after = deepcopy(before)
        after['test']['raw'][0]['keypoints'][0][0] = 15.01
        after['test']['postclip'],after['test']['filter'] = postclip_nms(after['test']['raw'])
        check = run.baseline_equivalence(refs,before,after)
        self.assertFalse(check['passed'])
        self.assertLess(check['maximum_drift']['coordinate_component_px'],.05)
        self.assertTrue(any('endpoint_threshold_state' in x for x in check['failures']))

    def test_misses_and_formal_duplicates_remain_distinct(self):
        refs,maps,output = fixture()
        scored = run.reports(refs,output)
        d = scored['r50']['postclip']['detection']
        self.assertEqual((d['fn'],d['fp'],d['duplicate_fp'],d['other_fp']),(1,1,1,0))
        inventory = run.case_inventory(refs,maps,{'control':output},{'control':scored})
        one = [x for x in inventory['rows'] if x['reference']=='r50']
        self.assertEqual([x['locatable_count'] for x in one],[2,1,0])
        self.assertEqual([x['stable_row_id'] for x in one],['stable:1','stable:2','stable:3'])
        unmatched = [x for x in inventory['unmatched_outputs'] if x['reference']=='r50']
        self.assertEqual(unmatched[0]['case_id'],'r50/test/control/out003(P003)')
        self.assertEqual(unmatched[0]['formal_type'],'duplicate')

    def test_paired_recovery_is_tied_to_reference_row_not_output_number(self):
        refs,maps,control = fixture()
        treatment = deepcopy(control)
        treatment['test']['raw'].append(prediction([90,10,110,30], [[90,30,.2],[110,30,.2]], .6))
        treatment['test']['postclip'],treatment['test']['filter'] = postclip_nms(treatment['test']['raw'])
        arms = {'control':control,'treatment':treatment}
        scores = {arm:run.reports(refs,o) for arm,o in arms.items()}
        cases = run.case_inventory(refs,maps,arms,scores)
        recovered = [x for x in cases['paired_rows'] if x['reference']=='r50' and x['recovered']]
        self.assertEqual([x['stable_row_id'] for x in recovered],['stable:3'])
        self.assertFalse(recovered[0]['within5_gain'])

    def test_checkpoint_coordinates_are_scored_in_original_pixels(self):
        refs,_,control = fixture()
        target_row=refs['r50']['test'][0]
        p=deepcopy(control['test']['raw'][0])
        p['keypoints'][0][:2]=[13,34]
        self.assertEqual(run.endpoint_state(target_row,p)['errors_px'][0],5)
        self.assertTrue(run.endpoint_state(target_row,p)['within5'])
        p['keypoints'][0][:2]=[1280,720]
        self.assertEqual(run.endpoint_state(target_row,p)['returned_visible_slots'],[1])
        self.assertFalse(run.endpoint_state(target_row,p)['within10'])


class ExportAndNotebookTest(unittest.TestCase):
    def test_blocked_baseline_archive_is_replayable_without_model_runtime(self):
        refs,maps,prior = fixture()
        actual = deepcopy(prior)
        actual['test']['raw'] = []
        actual['test']['postclip'],actual['test']['filter'] = postclip_nms([])
        baseline = run.baseline_equivalence(refs,prior,actual)
        self.assertFalse(baseline['passed'])
        arms={'control':actual}; scored={'control':run.reports(refs,actual)}
        contract={'settings':run.settings('clahe')}
        summary={'contract':contract,'status':'blocked_baseline','baseline_passed':False,
                 'repeated_outputs_stable':True,'interpretation_allowed':False,
                 'treatment_executed':False,'training':False,'automatic_promotion':False,
                 'fresh_inference':True,'reports':scored}
        with tempfile.TemporaryDirectory() as tmp:
            context={'work':Path(tmp),'contract':contract,'preflight':{'test_only':True},
                     'receipt':{'synthetic':True},'records':{'test':{}},'prior':prior,
                     'refs':refs,'maps':maps}
            payload={'summary.json':summary,'predictions.json':arms,'baseline-equivalence.json':baseline,
                     'cases.json':run.case_inventory(refs,maps,arms,scored),
                     'timing.json':{'measurements':[],'summary':{}},'runtime.json':{'synthetic':True},
                     'repeatability.json':[],'repeated-predictions.json':[]}
            result=run.package(context,payload)
            self.assertEqual(result['status'],'blocked_baseline')
            archive=Path(tmp)/run.archive_name('clahe')
            original=hashlib.sha256(archive.read_bytes()).hexdigest()
            run.verify_return(context)
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),original)

    def test_builder_compiles_four_cells_and_pins_treatments_separately(self):
        for experiment in ('clahe','scale'):
            result=notebook.build('a'*40,'b'*64,experiment)
            codes=[''.join(c['source']) for c in result['cells'] if c['cell_type']=='code']
            self.assertEqual(len(codes),4)
            for code in codes:
                compile(code,'<test-notebook>','exec')
            self.assertIn('\\n',codes[2])
            self.assertEqual(result['metadata']['experiment'],experiment)
            self.assertFalse(result['metadata']['training'])


if __name__=='__main__':
    unittest.main()
