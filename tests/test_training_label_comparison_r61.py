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
