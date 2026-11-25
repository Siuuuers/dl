#!/usr/bin/env python3
"""Check R31 native equivalence and selective regression gradients on real image inputs."""
import sys
sys.dont_write_bytecode = True

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from partial_box_loss_r31 import make_criterion, verify_native_sources
from predict_pose_folder import BASELINE_SHA256


def make_batch(root):
    """Two source images plus one empty image; downsize solely for this gradient check."""
    images, rows = [], []
    for idx, image_id in enumerate(('new_000001', 'new_000094')):
        image = Image.open(root / f'data/images/train/{image_id}.png').convert('RGB').resize((320, 192))
        images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255)
        path = root / f'data/reviews/fronts-2026-11-24-r31/labels/train/{image_id}.txt'
        for line in path.read_text().splitlines():
            fields = list(map(float, line.split()))
            rows.append((idx, fields))
    images.append(torch.zeros_like(images[0]))
    batch = {'img': torch.stack(images),
             'batch_idx': torch.tensor([idx for idx, _ in rows], dtype=torch.float32),
             'cls': torch.tensor([[v[0]] for _, v in rows]),
             'bboxes': torch.tensor([v[1:5] for _, v in rows]),
             'keypoints': torch.tensor([v[5:] for _, v in rows]).reshape(-1, 2, 3)}
    return batch


def clone_predictions(value, key=None):
    if isinstance(value, dict):
        return {k: clone_predictions(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_predictions(v, key) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_predictions(v, key) for v in value)
    if isinstance(value, torch.Tensor):
        return value.detach().clone().requires_grad_(key in ('boxes', 'scores', 'kpts', 'kpts_sigma'))
    return deepcopy(value)


def run_criterion(criterion, predictions, batch):
    captured = {}
    for name in ('one2many', 'one2one'):
        branch = getattr(criterion, name)
        original = branch.get_assigned_targets_and_loss
        def wrapped(self, preds, data, _original=original, _name=name, **kwargs):
            result = _original(preds, data, **kwargs)
            captured[_name] = tuple(x.detach().clone() for x in result[0][:2])
            return result
        branch.get_assigned_targets_and_loss = MethodType(wrapped, branch)
    preds = clone_predictions(predictions)
    loss, components = criterion(preds, batch)
    loss.sum().backward()
    gradients = {name: {key: value.grad.detach().clone() for key, value in branch.items()
                        if key in ('boxes', 'scores', 'kpts', 'kpts_sigma') and value.grad is not None}
                 for name, branch in preds.items()}
    return {'loss': loss.detach(), 'components': components, 'gradients': gradients, 'assigned': captured}


def assert_same(left, right, *, exact=True):
    torch.testing.assert_close(left['loss'], right['loss'], rtol=0 if exact else 1e-6, atol=0 if exact else 1e-6)
    assert set(left['components']) == set(right['components'])
    for key in left['components']:
        torch.testing.assert_close(left['components'][key], right['components'][key], rtol=0, atol=0)
    for branch in left['gradients']:
        assert set(left['gradients'][branch]) == set(right['gradients'][branch])
        for key in left['gradients'][branch]:
            torch.testing.assert_close(left['gradients'][branch][key], right['gradients'][branch][key], rtol=0, atol=0)
        for a, b in zip(left['assigned'][branch], right['assigned'][branch]):
            assert torch.equal(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    native_hashes = verify_native_sources()
    weights = root / 'weights/cassette_lower_epoch173.pt'
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == BASELINE_SHA256
    torch.set_num_threads(4)
    torch.manual_seed(46)
    model = YOLO(weights).model.float().train()
    model.args = get_cfg(overrides={'epochs': 30, 'batch': 4, 'imgsz': 1280})
    batch = make_batch(root)
    with torch.no_grad():
        predictions = model(batch['img'])
    assert set(predictions) == {'one2many', 'one2one'}
    native = run_criterion(model.init_criterion(), predictions, batch)
    control_stats, candidate_stats = [], []
    control = run_criterion(make_criterion(model, 1, control_stats.append), predictions, batch)
    candidate = run_criterion(make_criterion(model, 2, candidate_stats.append), predictions, batch)
    assert_same(native, control)
    checks = {}
    for name in ('one2many', 'one2one'):
        fg, assigned = control['assigned'][name]
        for a, b in zip(control['assigned'][name], candidate['assigned'][name]):
            assert torch.equal(a, b), 'factor changed native assignment'
        max_targets = max(int((batch['batch_idx'] == idx).sum()) for idx in range(len(batch['img'])))
        partial = torch.zeros((len(batch['img']), max_targets), dtype=torch.bool)
        for idx in range(len(batch['img'])):
            visible = batch['keypoints'][batch['batch_idx'] == idx, :, 2]
            partial[idx, :len(visible)] = (visible > 0).sum(-1) < 2
        partial_anchor = fg & partial.gather(1, assigned)
        assert partial_anchor.any() and (fg & ~partial_anchor).any(), 'check lacks assigned partial or full targets'
        expected = control['gradients'][name]['boxes'] * (1 + partial_anchor.unsqueeze(1).float())
        torch.testing.assert_close(candidate['gradients'][name]['boxes'], expected, rtol=1e-6, atol=1e-7)
        assert control['gradients'][name]['boxes'][partial_anchor.unsqueeze(1).expand_as(expected)].abs().sum() > 0
        for key in ('scores', 'kpts', 'kpts_sigma'):
            if key in control['gradients'][name]:
                torch.testing.assert_close(control['gradients'][name][key], candidate['gradients'][name][key], rtol=0, atol=0)
        checks[name] = {'partial_foreground_anchors': int(partial_anchor.sum()),
                        'full_foreground_anchors': int((fg & ~partial_anchor).sum()),
                        'background_anchors': int((~fg).sum()),
                        'partial_box_gradient_factor': 2, 'full_box_gradient_factor': 1,
                        'classification_gradient_exactly_unchanged': True,
                        'keypoint_and_sigma_gradients_exactly_unchanged': True}
    # Component order follows pinned PoseLoss26: box, pose, kobj, cls, nativeL1, RLE.
    for index in (1, 2, 3, 5):
        torch.testing.assert_close(control['loss'][index], candidate['loss'][index], rtol=0, atol=0)
    empty = {**batch, 'batch_idx': batch['batch_idx'][:0], 'cls': batch['cls'][:0],
             'bboxes': batch['bboxes'][:0], 'keypoints': batch['keypoints'][:0]}
    native_empty = run_criterion(model.init_criterion(), predictions, empty)
    control_empty = run_criterion(make_criterion(model, 1), predictions, empty)
    candidate_empty = run_criterion(make_criterion(model, 2), predictions, empty)
    assert_same(native_empty, control_empty)
    assert_same(native_empty, candidate_empty)
    assert torch.isfinite(candidate_empty['loss']).all()
    result = {'schema_version': 1, 'status': 'passed', 'purpose': 'loss correctness; no accuracy claim',
              'native_sources_sha256': native_hashes, 'initializer_sha256': BASELINE_SHA256,
              'factor1_native_loss_and_prediction_gradients_bitwise_equal': True,
              'factor2_assignment_exactly_unchanged': True,
              'factor2_native_nonbox_components_bitwise_equal': True,
              'factor2_selective_box_gradient_check': checks,
              'empty_batch_native_equivalence_for_both_factors': True,
              'native_loss_components': native['loss'].tolist(),
              'factor2_loss_components': candidate['loss'].tolist(),
              'control_assignment_observations': control_stats,
              'candidate_assignment_observations': candidate_stats}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'output': str(args.output), 'branches': checks}, indent=2))


if __name__ == '__main__':
    main()
