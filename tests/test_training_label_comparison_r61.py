import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_training_label_comparison_r61 as experiment
from pose_scoring import score_image
from predict_pose_folder import postclip_nms
from run_corner_average_r58 import report, score_image_counting_clipped
from run_flip_inference_r56 import inverse_horizontal_predictions


class TrainingLabelComparisonTests(unittest.TestCase):
    def test_only_training_labels_change(self):
        plan = experiment.label_plan(ROOT)
        old, new = (plan['roles'][arm] for arm in experiment.ARMS)
        self.assertEqual(92, len(plan['changed_training_files']))
        self.assertTrue(all(row['numeric_content_changed'] for row in plan['changed_training_files']))
        self.assertEqual(4110, plan['budget_per_arm']['images'])
        self.assertEqual(1050, plan['budget_per_arm']['batches'])
        self.assertEqual(78, plan['budget_per_arm']['steps'])
        for a, b in zip(old, new):
            self.assertEqual((a['id'], a['image_sha256'], a['split']), (b['id'], b['image_sha256'], b['split']))
            if a['split'] == 'val':
                self.assertEqual(a, b)
        self.assertFalse(any(row['id'].startswith('frame_') for row in old + new))

    def test_real_saved_predictions_reproduce_three_references(self):
        archive = ROOT / 'docs/thesis/evidence/original-baseline/fresh/baomai-original-comparison-r59-results.zip'
        with ZipFile(archive) as stream:
            native = json.loads(gzip.decompress(stream.read('native-predictions.json.gz')))
        records = json.loads((ROOT / 'data/manifest.json').read_text())['records']
        refs, _ = experiment.read_references(ROOT, records)
        views = {'original': native['trained'], 'reflected': {
            identity: inverse_horizontal_predictions(values) for identity, values in native['reflected'].items()}}
        measured, details = experiment.score_native(refs, {'a': views, 'b': copy.deepcopy(views)}, 'a')
        expected = {'supplied664': (46, 69, 8, 514, 565, 584), 'frozen666': (34, 55, 9, 525, 574, 584),
                    'r50': (18, 51, 5, 569, 583, 583)}
        for reference, values in expected.items():
            scored = measured['references'][reference]['a']['two_view']
            d, p = scored['detection'], scored['pairs']
            self.assertEqual(values, (d['fn'], d['fp'], d['duplicate_fp'], p['within5'], p['within10'], p['eligible']))
            self.assertEqual(scored, measured['references'][reference]['b']['two_view'])
            self.assertEqual(0, measured['paired_changes'][reference]['b']['two_view']['5']['gained'])
            self.assertEqual(0, measured['paired_changes'][reference]['b']['two_view']['5']['lost'])
        self.assertEqual(566, measured['references']['r50']['a']['original']['pairs']['within5'])
        for identity in native['trained']:
            for a, b in zip(details['predictions']['a']['original'][identity], details['predictions']['a']['two_view'][identity]):
                self.assertEqual({k: v for k, v in a.items() if k != 'keypoints'},
                                 {k: v for k, v in b.items() if k != 'keypoints'})
        incomplete = copy.deepcopy(views)
        incomplete['reflected'].pop(next(iter(incomplete['reflected'])))
        with self.assertRaisesRegex(RuntimeError, 'membership differs'):
            experiment.score_native(refs, {'a': views, 'b': incomplete})

    def test_clipped_zero_height_box_is_counted_and_cannot_match(self):
        zero = {'class_id': 0, 'confidence': 0.25628936290740967,
                'bbox_xyxy': [782.99072265625, 0.0, 880.15234375, 0.0],
                'keypoints': [[800.0, 0.0, 0.9], [860.0, 0.0, 0.8]]}
        reflected_zero = {'class_id': 0, 'confidence': 0.3080700933933258,
                          'bbox_xyxy': [662.5360107421875, 0.0, 759.7286376953125, 0.0],
                          'keypoints': [[700.0, 0.0, 0.4], [740.0, 0.0, 0.4]]}
        with self.assertRaisesRegex(ValueError, r'predictions\[0\]\.bbox_xyxy must have finite positive area'):
            score_image([], [zero], 1280, 720, .5, .5)
        with self.assertRaisesRegex(ValueError, 'must be finite'):
            score_image_counting_clipped([], [{**zero, 'bbox_xyxy': [1.0, 0.0, 2.0, float('nan')]}], 1280, 720, .5, .5)
        overlap = {'class_id': 0, 'confidence': 0.99, 'bbox_xyxy': [520.0, 0.0, 620.0, 40.0],
                   'keypoints': [[530.0, 20.0, 0.9], [600.0, 20.0, 0.9]]}
        kept, audit = postclip_nms([overlap, zero])
        self.assertEqual([1, 2], audit['retained_primary_prediction_ids'])
        self.assertEqual([], audit['suppressed_predictions'])
        original = []
        for index in range(21):
            x = (index % 7) * 40.0
            y = 80.0 + (index // 7) * 40.0
            original.append({'class_id': 0, 'confidence': 0.95 - index * 0.01,
                             'bbox_xyxy': [x, y, x + 20.0, y + 20.0],
                             'keypoints': [[x + 1.0, y + 19.0, 0.9], [x + 19.0, y + 19.0, 0.9]]})
        duplicate = {**original[0], 'confidence': 0.5}
        suppressed, suppression = postclip_nms([original[0], duplicate, zero])
        self.assertEqual([row['bbox_xyxy'] for row in suppressed], [original[0]['bbox_xyxy'], zero['bbox_xyxy']])
        self.assertEqual([2], [row['primary_prediction_id'] for row in suppression['suppressed_predictions']])
        original.extend((zero, duplicate))
        gold = {'new_000057': [{'class_id': 0, 'bbox_xyxy': [0.0, 80.0, 20.0, 100.0],
                                'keypoints': [[1.0, 99.0, 2], [19.0, 99.0, 2]]}]}
        native = {'original': {'new_000057': original}, 'reflected': {'new_000057': [reflected_zero]}}
        measured, details = experiment.score_native({'r50': gold}, {'candidate': native, 'reference': copy.deepcopy(native)}, 'reference')
        self.assertEqual(measured['protocol']['clipped_nonpositive_predictions'],
                         'retained as other unmatched false positives; excluded from matching')
        kept_original = details['predictions']['candidate']['original']['new_000057']
        kept_average = details['predictions']['candidate']['two_view']['new_000057']
        self.assertEqual(21, next(index for index, row in enumerate(kept_original) if row['bbox_xyxy'] == zero['bbox_xyxy']))
        self.assertEqual(22, kept_original[21]['primary_prediction_id'])
        self.assertEqual([row['bbox_xyxy'] for row in kept_original], [row['bbox_xyxy'] for row in kept_average])
        self.assertEqual([zero['bbox_xyxy']], [row['bbox_xyxy'] for row in kept_original if row['bbox_xyxy'][3] <= row['bbox_xyxy'][1]])
        self.assertTrue(all(row['primary_prediction_id'] != 23 for row in kept_original))
        reflected_audit = details['suppression']['candidate']['new_000057']['reflected']
        self.assertEqual([1], reflected_audit['retained_primary_prediction_ids'])
        self.assertEqual([], reflected_audit['suppressed_predictions'])
        for output in ('original', 'two_view'):
            score = details['per-image']['r50']['candidate'][output]['new_000057']
            self.assertIn(21, score['other_fp_prediction_indices'])
            self.assertNotIn(21, score['duplicate_prediction_indices'])
            self.assertTrue(all(pair['prediction_index'] != 21 for pair in score['matches']))
            self.assertEqual(1, score['tp'])
            self.assertEqual(score['prediction_count'] - score['tp'], score['fp'])
            self.assertEqual(score['fp'], score['duplicate_fp'] + score['other_fp'])
            summary = measured['references']['r50']['candidate'][output]
            self.assertEqual(score['fp'], summary['detection']['fp'])
            self.assertEqual(1, summary['pairs']['eligible'])
            self.assertEqual(0, summary['pairs']['missed_box'])
        alone = report({'scene': []}, {'scene': [zero]})
        self.assertEqual((1, 1, 1, 0), (alone[0]['detection']['prediction_count'], alone[0]['detection']['fp'],
                                         alone[0]['detection']['other_fp'], alone[0]['detection']['tp']))

    def test_multipart_boundary_replay_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            payload = b'checkpoint-test-' * 1100000
            checkpoint = work / 'fixture.pt'
            checkpoint.write_bytes(payload)
            manifest = experiment.package_evidence(work, {'summary.json': b'{"fixture":true}\n'},
                                                   {'control': checkpoint}, 'fixture')
            parts = manifest['models']['control']['parts']
            self.assertEqual(2, len(parts))
            reconstructed = b''
            for part in parts:
                with ZipFile(work / part['file']) as stream:
                    reconstructed += stream.read(part['member'])
            self.assertEqual(payload, reconstructed)
            wrapper = work / parts[-1]['file']
            wrapper.write_bytes(wrapper.read_bytes() + b'corruption')
            with self.assertRaisesRegex(RuntimeError, 'Input differs'):
                experiment.verify_package(work, 'fixture')


if __name__ == '__main__':
    unittest.main()
