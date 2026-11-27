#!/usr/bin/env python3
"""Check native equivalence and selective BCE gradients on two real training photos.

This bounded CPU guard performs forward/backward differentiation, not an
optimizer update or an accuracy experiment. A smaller image size saves CPU work
and is explicitly unrelated to the fixed 1280-pixel paired training recipe.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
from types import MethodType
import zipfile

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from partial_classification_r51 import (
    BRANCHES, CRITERION_CONTRACT, SPEC, digest_json, make_criterion, require, verify_native_sources,
)

INITIALIZER_SHA = 'e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e'
PR137_SHA = 'fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023'
IMAGE_IDS = ('new_000001', 'new_000094')
PREDICTION_KEYS = ('boxes', 'scores', 'kpts', 'kpts_sigma')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_batch(root, inputs, archive):
    require(sha(archive) == PR137_SHA, 'PR137 dataset archive differs')
    records = {row['id']: row for row in json.loads((root / 'data/manifest.json').read_text())['records']}
    images, rows, sources = [], [], []
    with zipfile.ZipFile(archive) as source:
        for idx, image_id in enumerate(IMAGE_IDS):
            image_path = inputs / f'data/images/train/{image_id}.png'
            record = records[image_id]
            require(record['split'] == 'train' and sha(image_path) == record['image_sha256'], 'Real training image differs')
            image = Image.open(image_path).convert('RGB').resize((320, 192))
            images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255)
            member = f'datasets/candidate/labels/train/{image_id}.txt'
            data = source.read(member)
            for line in data.decode().splitlines():
                values = list(map(float, line.split()))
                require(len(values) == 11, 'Unexpected candidate pose label')
                rows.append((idx, values))
            sources.append({'image_id': image_id, 'image_sha256': record['image_sha256'],
                            'label_member': member, 'label_sha256': hashlib.sha256(data).hexdigest()})
    # The third image is a synthetic blank, testing padding/background in a mixed batch.
    images.append(torch.zeros_like(images[0]))
    batch = {'img': torch.stack(images),
             'batch_idx': torch.tensor([idx for idx, _ in rows], dtype=torch.float32),
             'cls': torch.tensor([[v[0]] for _, v in rows]),
             'bboxes': torch.tensor([v[1:5] for _, v in rows]),
             'keypoints': torch.tensor([v[5:] for _, v in rows]).reshape(-1, 2, 3)}
    return batch, sources


def clone_predictions(value, key=None):
    if isinstance(value, dict):
        return {k: clone_predictions(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_predictions(v, key) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_predictions(v, key) for v in value)
    if isinstance(value, torch.Tensor):
        return value.detach().clone().requires_grad_(key in PREDICTION_KEYS)
    return deepcopy(value)


def run_criterion(model, criterion, predictions, batch):
    captured = {}
    for name in BRANCHES:
        branch = getattr(criterion, name)
        original = branch.get_assigned_targets_and_loss
        def wrapped(self, preds, data, _original=original, _name=name, **kwargs):
            result = _original(preds, data, **kwargs)
            captured[_name] = tuple(x.detach().clone() for x in result[0])
            return result
        branch.get_assigned_targets_and_loss = MethodType(wrapped, branch)
    model.zero_grad(set_to_none=True)
    preds = clone_predictions(predictions)
    loss, components = criterion(preds, batch)
    require(bool(torch.isfinite(loss).all()), 'Non-finite guard loss')
    loss.sum().backward()
    gradients = {name: {key: value.grad.detach().clone() for key, value in branch.items()
                        if key in PREDICTION_KEYS and value.grad is not None}
                 for name, branch in preds.items()}
    flow_gradients = {name: p.grad.detach().clone() for name, p in model.model[-1].flow_model.named_parameters()
                      if p.grad is not None}
    require(all(bool(torch.isfinite(value).all()) for branch in gradients.values() for value in branch.values())
            and all(bool(torch.isfinite(value).all()) for value in flow_gradients.values()), 'Non-finite guard gradients')
    return {'loss': loss.detach(), 'components': components, 'gradients': gradients,
            'assigned': captured, 'flow_gradients': flow_gradients}


def same_tensor(left, right):
    require(torch.equal(left, right), 'Tensor differs from native at bitwise precision')


def assert_same(left, right):
    same_tensor(left['loss'], right['loss'])
    require(set(left['components']) == set(right['components']), 'Loss component inventory differs')
    for key in left['components']:
        same_tensor(left['components'][key], right['components'][key])
    for branch in left['gradients']:
        require(set(left['gradients'][branch]) == set(right['gradients'][branch]), 'Gradient inventory differs')
        for key in left['gradients'][branch]:
            same_tensor(left['gradients'][branch][key], right['gradients'][branch][key])
        for a, b in zip(left['assigned'][branch], right['assigned'][branch]):
            same_tensor(a, b)
    require(set(left['flow_gradients']) == set(right['flow_gradients']), 'RLE flow parameter gradient inventory differs')
    for key in left['flow_gradients']:
        same_tensor(left['flow_gradients'][key], right['flow_gradients'][key])


def partial_anchors(batch, fg, assigned):
    batch_size = len(batch['img'])
    max_targets = max(int((batch['batch_idx'] == idx).sum()) for idx in range(batch_size))
    partial = torch.zeros((batch_size, max_targets), dtype=torch.bool)
    for idx in range(batch_size):
        visible = batch['keypoints'][batch['batch_idx'] == idx, :, 2]
        partial[idx, :len(visible)] = (visible > 0).sum(-1) < 2
    return fg & partial.gather(1, assigned) if max_targets else torch.zeros_like(fg)


def guard_case(model, predictions, batch, *, require_partial=True):
    native = run_criterion(model, model.init_criterion(), predictions, batch)
    control_stats, candidate_stats = [], []
    control = run_criterion(model, make_criterion(model, 1, control_stats.append), predictions, batch)
    candidate = run_criterion(model, make_criterion(model, 2, candidate_stats.append), predictions, batch)
    assert_same(native, control)
    checks = {}
    for name in BRANCHES:
        fg, assigned = control['assigned'][name][:2]
        for a, b in zip(control['assigned'][name], candidate['assigned'][name]):
            same_tensor(a, b)
        partial = partial_anchors(batch, fg, assigned)
        if require_partial:
            require(bool(partial.any()) and bool((fg & ~partial).any()), 'Guard lacks assigned partial or full targets')
        grad = control['gradients'][name]['scores']
        expected = grad * (1 + partial.unsqueeze(1).to(grad.dtype))
        # Multiplication by two is exact in this float32 experiment. A looser
        # tolerance would conceal unintended changes to untreated anchors.
        same_tensor(candidate['gradients'][name]['scores'], expected)
        if require_partial:
            require(float(grad[partial.unsqueeze(1)].abs().sum()) > 0, 'Partial classification gradient is absent')
        for key in ('boxes', 'kpts', 'kpts_sigma'):
            if key in control['gradients'][name]:
                same_tensor(control['gradients'][name][key], candidate['gradients'][name][key])
        checks[name] = {'partial_foreground_anchors': int(partial.sum()),
                        'full_foreground_anchors': int((fg & ~partial).sum()),
                        'background_anchors': int((~fg).sum()),
                        'partial_score_gradient_factor_exact': 2,
                        'full_and_background_score_gradients_bitwise_unchanged': True,
                        'box_keypoint_and_sigma_gradients_bitwise_unchanged': True}
    # Native PoseLoss26 order: box, pose, visibility, classification, L1, RLE.
    for index in (0, 1, 2, 4, 5):
        same_tensor(control['loss'][index], candidate['loss'][index])
    for key in control['flow_gradients']:
        same_tensor(control['flow_gradients'][key], candidate['flow_gradients'][key])
    if not require_partial and not any(row['partial_foreground_anchors'] for row in checks.values()):
        assert_same(native, candidate)
    for left, right in zip(control_stats, candidate_stats):
        for field in ('original_target_score_sum', 'normalizer', 'partial_target_score_sum'):
            require(left[field] == right[field], 'Target scores or unweighted denominator changed')
        require(left['weighted_classification_terms'] == 0
                and right['weighted_classification_terms'] == right['partial_classification_terms'],
                'Applied BCE term inventory differs')
    return {'factor1_native_loss_and_prediction_gradients_bitwise_equal': True,
            'factor2_assignment_and_target_geometry_bitwise_unchanged': True,
            'factor2_nonclassification_loss_components_bitwise_equal': True,
            'factor2_rle_flow_parameter_gradients_bitwise_equal': True,
            'selective_classification_gradient_check': checks,
            'native_loss_components': native['loss'].tolist(),
            'factor2_loss_components': candidate['loss'].tolist(),
            'control_observations': control_stats, 'candidate_observations': candidate_stats}


def full_model_equivalence(model, batch):
    """Check the control through the actual complete network, with no optimizer."""
    initial = deepcopy(model.state_dict())
    outcomes = []
    for factor in (None, 1):
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(46)
        criterion = model.init_criterion() if factor is None else make_criterion(model, factor)
        loss, _ = criterion(model(batch['img']), batch)
        loss.sum().backward()
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        require(bool(torch.isfinite(loss).all()) and all(bool(torch.isfinite(g).all()) for g in gradients.values()),
                'Non-finite full-model differentiation')
        outcomes.append((loss.detach(), gradients))
    same_tensor(outcomes[0][0], outcomes[1][0])
    require(set(outcomes[0][1]) == set(outcomes[1][1]), 'Native/control parameter gradient inventory differs')
    for key in outcomes[0][1]:
        same_tensor(outcomes[0][1][key], outcomes[1][1][key])
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    return {'factor1_native_full_network_parameter_gradients_bitwise_equal': True,
            'parameter_tensors_with_gradient': len(outcomes[0][1]), 'optimizer_updates': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--inputs', type=Path, required=True, help='Verified initializer and source image root')
    parser.add_argument('--dataset-archive', type=Path, required=True, help='Verified PR137 result ZIP with candidate labels')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, inputs = args.root.resolve(), args.inputs.resolve()
    require(not args.output.exists(), 'Refusing to replace an existing gradient guard receipt')
    native_hashes = verify_native_sources()
    weights = inputs / 'weights/cassette_lower_epoch173.pt'
    require(sha(weights) == INITIALIZER_SHA, 'Epoch173 initializer differs')
    torch.set_num_threads(4)
    torch.manual_seed(46)
    # Exported checkpoints may mark parameters frozen for inference. The native
    # trainer re-enables them; do the same before testing real differentiation.
    model = YOLO(weights).model.float().train().requires_grad_(True)
    model.args = get_cfg(overrides={'epochs': 30, 'batch': 4, 'imgsz': 1280})
    batch, sources = make_batch(root, inputs, args.dataset_archive)
    with torch.no_grad():
        predictions = model(batch['img'])
    require(set(predictions) == set(BRANCHES), 'Expected both native prediction branches')
    mixed = guard_case(model, predictions, batch)

    # Verify optional native class weights are applied before the partial weight.
    previous_weights = getattr(model, 'class_weights', None)
    model.class_weights = torch.tensor([1.7])
    class_weighted = guard_case(model, predictions, batch)
    model.class_weights = previous_weights
    empty = {**batch, **{key: batch[key][:0] for key in ('batch_idx', 'cls', 'bboxes', 'keypoints')}}
    empty_result = guard_case(model, predictions, empty, require_partial=False)

    # Controlled synthetic visibility changes test use of the consumed batch;
    # these are never saved as annotations or used in the paired training run.
    full = {**batch, 'keypoints': batch['keypoints'].clone()}
    full['keypoints'][..., 2] = 2
    full_result = guard_case(model, predictions, full, require_partial=False)
    changed = {**batch, 'keypoints': batch['keypoints'].clone()}
    full_row = int(torch.nonzero((changed['keypoints'][..., 2] > 0).sum(-1) == 2)[0])
    changed['keypoints'][full_row, 0, 2] = 0
    changed_result = guard_case(model, predictions, changed)
    for branch in BRANCHES:
        before = next(row for row in mixed['control_observations'] if row['branch'] == branch)
        after = next(row for row in changed_result['control_observations'] if row['branch'] == branch)
        require(after['partial_targets'] == before['partial_targets'] + 1, 'Consumed visibility change was ignored')
    full_network = full_model_equivalence(model, batch)

    # The wrapper must leave the complete native branch-weight schedule intact.
    schedules = []
    for factor in (None, 1, 2):
        criterion = model.init_criterion() if factor is None else make_criterion(model, factor)
        values = []
        for _ in range(30):
            values.append([criterion.updates, criterion.o2m, criterion.o2o])
            criterion.update()
        schedules.append(values)
    require(schedules[0] == schedules[1] == schedules[2], 'Native branch schedule differs')
    result = {'schema_version': 1, 'status': 'passed',
              'purpose': 'loss correctness and differentiation; no training or accuracy claim',
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                          'device': 'cpu', 'threads': torch.get_num_threads()},
              'native_sources_sha256': native_hashes, 'initializer_sha256': INITIALIZER_SHA,
              'dataset_archive_sha256': PR137_SHA, 'real_training_sources': sources,
              'guard_input_shape': list(batch['img'].shape),
              'guard_image_transform': 'RGB resize to 320x192; normalized coordinates retained; one synthetic blank appended',
              'production_training_imgsz': 1280, 'optimizer_updates': 0,
              'source_sha256': {name: sha(root / 'scripts' / name) for name in
                                 ('partial_classification_r51.py', 'check_partial_classification_r51.py')},
              'loss_weighting_spec_sha256': digest_json(SPEC), 'criterion_contract': CRITERION_CONTRACT,
              'mixed_real_batch': mixed, 'native_class_weights_1_7': class_weighted,
              'empty_batch': empty_result, 'all_fully_visible_synthetic_batch': full_result,
              'consumed_visibility_mutation': changed_result, 'full_network_control_equivalence': full_network,
              'native_branch_schedule_bitwise_equal': True, 'native_branch_schedule': schedules[0],
              'soft_target_interpretation': 'The weighted derivative is factor*(sigmoid(logit)-soft_target); weighting can raise or lower logits.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': 'passed', 'output': str(args.output), 'branches': mixed['selective_classification_gradient_check'],
                      'full_network': full_network}, indent=2))


if __name__ == '__main__':
    main()
