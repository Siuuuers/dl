#!/usr/bin/env python3
"""CPU-only R28 input, loader, initializer and control-identity smoke check."""
import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--staged', type=Path, required=True)
parser.add_argument('--receipt', type=Path, required=True)
args = parser.parse_args()
root, staged = args.root.resolve(), args.staged.resolve()
sys.path.insert(0, str(root / 'scripts'))
from evaluate_pose import file_sha256, preflight_dataset
from front_review_r28_training import CANDIDATE_PATH, PROTOCOL_PATH, DELTA_PATH
from pose_checkpoint import _check_staging, get_recipe, training_label_identity
from stage_pose_training import MANIFEST, stage_datasets, verify_staging
from train_pose_comparison import worker_training_labels

candidate = root / CANDIDATE_PATH
protocol = json.loads((root / PROTOCOL_PATH).read_text())
delta = json.loads((root / DELTA_PATH).read_text())
initial = preflight_dataset(root, require_weights=True)
if not staged.exists():
    stage_datasets(root, candidate, staged)
stage_result = verify_staging(root, candidate, staged)
stage = json.loads((staged / MANIFEST).read_text())
_check_staging(initial, stage)
identities = {}
for arm, spec in protocol['roles'].items():
    identity = training_label_identity(stage, arm)
    assert identity['role'] == spec['supervision']
    config = {'recipe_id': spec['recipe_id'], 'arm': arm,
              'candidate_manifest': spec['candidate_manifest'], 'training_labels': identity}
    assert worker_training_labels(config, stage) == identity
    identities[arm] = identity
rejected = []
for name in ('wrong_control_role', 'altered_input_delta', 'wrong_worker_label_role'):
    bad_stage = copy.deepcopy(stage)
    bad_config = copy.deepcopy(config)
    if name == 'wrong_control_role':
        bad_stage['source']['control']['supervision'] = 'pr35_annotation_control'
    elif name == 'altered_input_delta':
        bad_stage['source']['training_input_delta']['report']['summary']['changed_rows'] += 1
    else:
        bad_config['training_labels']['role'] = 'pr35_annotation_control'
    try:
        if name == 'wrong_worker_label_role':
            worker_training_labels(bad_config, stage)
        else:
            _check_staging(initial, bad_stage)
    except ValueError:
        rejected.append(name)
    else:
        raise AssertionError('An incompatible provenance fixture was accepted: ' + name)

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.utils.torch_utils import init_seeds

assert importlib.metadata.version('ultralytics') == '8.4.182'
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
recipe = get_recipe(protocol['recipe_id'])
loader_rows = []
samples = {}
selected = delta['changed_image_ids'][:4]
assert len(selected) == 4, 'This bounded smoke expects at least four changed training photos.'
for arm in ('reference', 'candidate'):
    arm_root = staged / arm
    data = check_det_dataset(str(arm_root / 'data.yaml'), autodownload=False)
    assert data['kpt_shape'] == [2, 3] and data['flip_idx'] == [1, 0]
    assert data['names'] == {0: 'cassette_front'}
    for split in ('train', 'val'):
        init_seeds(46, deterministic=True)
        cfg = get_cfg(overrides={**recipe, 'task': 'pose', 'device': 'cpu', 'workers': 0})
        dataset = build_yolo_dataset(cfg, data[split], batch=4, data=data, mode=split, stride=32)
        expected = stage['variants'][arm]['counts'][split]
        actual = {'images': len(dataset), 'objects': sum(len(x['cls']) for x in dataset.labels),
                  'visible_endpoints': sum(int((x['keypoints'][:, :, 2] == 2).sum()) for x in dataset.labels),
                  'missing_endpoints': sum(int((x['keypoints'][:, :, 2] == 0).sum()) for x in dataset.labels),
                  'empty_images': sum(len(x['cls']) == 0 for x in dataset.labels)}
        assert actual == expected
        files = {entry['path']: entry['output_sha256'] for entry in stage['variants'][arm]['files']}
        for image, label in zip(dataset.im_files, img2label_paths(dataset.im_files)):
            for value in (image, label):
                path = Path(value).resolve()
                relative = path.relative_to(arm_root).as_posix()
                assert file_sha256(path) == files[relative]
        by_id = {Path(path).stem: i for i, path in enumerate(dataset.im_files)}
        chosen = selected if split == 'train' else list(by_id)[:4]
        init_seeds(46, deterministic=True)
        batch = dataset.collate_fn([dataset[by_id[image_id]] for image_id in chosen])
        assert list(batch['img'].shape)[:2] == [4, 3]
        assert max(batch['img'].shape[-2:]) == (1280 if split == 'train' else 1312)
        assert all(size % 32 == 0 for size in batch['img'].shape[-2:])
        assert tuple(batch['keypoints'].shape[1:]) == (2, 3)
        assert batch['bboxes'].shape[-1] == 4
        assert all(torch.isfinite(batch[key]).all() for key in ('img', 'keypoints', 'bboxes', 'cls'))
        if split == 'train':
            samples[arm] = batch['img'][:1].float() / 255
        loader_rows.append({'arm': arm, 'split': split, 'counts': actual,
                            'sample_image_ids': chosen, 'batch_shape': list(batch['img'].shape),
                            'batch_objects_after_augmentation': len(batch['cls']),
                            'keypoint_tensor_shape': list(batch['keypoints'].shape)})
        del dataset, batch
assert torch.equal(samples['reference'], samples['candidate']), 'Input pixels must match under the same seeded transform.'
weights = root / 'weights/cassette_lower_epoch173.pt'
model = YOLO(str(weights), task='pose').model.to('cpu').float().eval()
head = model.model[-1]
assert list(head.kpt_shape) == [2, 3] and head.nc == 1
with torch.inference_mode():
    output = model(samples['candidate'])
tensors = []
def visit(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
        tensors.append(list(value.shape))
    elif isinstance(value, dict):
        for item in value.values(): visit(item)
    elif isinstance(value, (tuple, list)):
        for item in value: visit(item)
visit(output)
assert tensors and file_sha256(weights) == protocol['initial_checkpoint_sha256']
assert not model.training
cache_paths = sorted(staged.glob('*/labels/*.cache'))
assert len(cache_paths) == 4
for path in cache_paths:
    path.unlink()
assert verify_staging(root, candidate, staged) == stage_result
receipt = {'schema_version': 1, 'kind': 'r28_cpu_training_input_smoke', 'status': 'passed',
           'verified_at_utc': datetime.now(timezone.utc).isoformat(),
           'candidate_manifest_sha256': file_sha256(candidate),
           'protocol_sha256': file_sha256(root / PROTOCOL_PATH),
           'training_input_delta_sha256': file_sha256(root / DELTA_PATH),
           'staging': stage_result, 'training_label_identities': identities,
           'rejected_incompatible_provenance': rejected,
           'ultralytics': importlib.metadata.version('ultralytics'), 'torch': torch.__version__,
           'device': 'cpu', 'loader_checks': loader_rows,
           'seeded_sample_pixels_identical_between_arms': True,
           'initializer_forward': {'passed': True, 'batch_size': 1, 'classes': head.nc,
                                   'kpt_shape': list(head.kpt_shape), 'tensor_shapes': tensors},
           'staging_unchanged_after_loader_cache_cleanup': True,
           'optimizer_steps': 0, 'training_executed': False, 'gpu_executed': False,
           'limitations': ['CPU input/loader/initializer smoke only; no CUDA trial, training, metric evaluation or model promotion.']}
args.receipt.parent.mkdir(parents=True, exist_ok=True)
args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
