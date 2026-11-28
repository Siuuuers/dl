#!/usr/bin/env python3
"""Verify QFL derivatives and native invariants on actual pretrained Pose26 outputs.

This is a CPU differentiation guard. It performs no optimizer update and does
not measure detector accuracy. The small guard images are not training inputs.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from check_partial_classification_r51 import (
    INITIALIZER_SHA, PR137_SHA, make_batch, run_criterion, same_tensor, assert_same,
)
from quality_focal_r61 import (
    BRANCHES, CRITERION_CONTRACT, SPEC, digest_json, make_criterion,
    quality_focal_terms, require, verify_native_sources,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analytic_derivative(logits, targets):
    probability = logits.sigmoid()
    error = probability - targets
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return error.pow(3) + 2 * error * probability * (1 - probability) * bce


def scalar_gradient_guard():
    logits = torch.tensor([-40., -8., -2., -.1, 0., .7, 4., 9., 40.], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0., 1., .2, .7, .5, .4, .9, 1., 0.], dtype=torch.float64)
    native = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    require(quality_focal_terms(logits, targets, 0, native) is native, 'Beta zero must return native terms directly')
    actual = torch.autograd.grad(quality_focal_terms(logits, targets, 2).sum(), logits)[0]
    expected = analytic_derivative(logits.detach(), targets)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.autograd.gradcheck(lambda z: quality_focal_terms(z, targets, 2), (logits,), eps=1e-6, atol=1e-6, rtol=1e-4)
    detached_modulation_gradient = (logits.detach().sigmoid() - targets).pow(3)
    require(float((actual - detached_modulation_gradient).abs().max()) > .01,
            'Gradient guard does not detect accidentally detached modulation')
    require(actual[4] == 0, 'A perfectly matched soft target must have zero QFL gradient')
    return {'status': 'passed', 'double_precision_gradcheck': True,
            'analytic_derivative': '(p-y)^3 + 2*(p-y)*p*(1-p)*BCEWithLogits(z,y)',
            'maximum_analytic_gradient_difference': float((actual - expected).abs().max()),
            'maximum_absolute_expected_gradient': float(expected.abs().max()),
            'analytic_gradient_rtol': 1e-12, 'analytic_gradient_atol': 1e-12,
            'detached_modulation_is_detectably_wrong': True,
            'zero_modulation_control_returns_original_tensor': True,
            'extreme_logits_checked': [-40, 40]}


def run_with_targets(model, criterion, predictions, batch):
    targets, handles = {}, []
    for name in BRANCHES:
        def capture(module, args, _name=name):
            require(_name not in targets, 'Unexpected repeated classification call')
            targets[_name] = args[1].detach().clone()
        handles.append(getattr(criterion, name).bce.register_forward_pre_hook(capture))
    try:
        result = run_criterion(model, criterion, predictions, batch)
    finally:
        for handle in handles:
            handle.remove()
    require(set(targets) == set(BRANCHES), 'Both native classification branches must run')
    result['classification_targets'] = targets
    return result


def guard_case(model, predictions, batch):
    native = run_with_targets(model, model.init_criterion(), predictions, batch)
    control_stats, candidate_stats = [], []
    control_criterion = make_criterion(model, 0, control_stats.append)
    candidate_criterion = make_criterion(model, 2, candidate_stats.append)
    control = run_with_targets(model, control_criterion, predictions, batch)
    candidate = run_with_targets(model, candidate_criterion, predictions, batch)
    assert_same(native, control)
    checks = {}
    for name in BRANCHES:
        for a, b in zip(control['assigned'][name], candidate['assigned'][name]):
            same_tensor(a, b)
        targets = control['classification_targets'][name]
        same_tensor(targets, candidate['classification_targets'][name])
        same_tensor(targets, native['classification_targets'][name])
        logits = predictions[name]['scores'].permute(0, 2, 1).detach()
        branch = getattr(candidate_criterion, name)
        expected = analytic_derivative(logits, targets)
        if branch.class_weights is not None:
            expected = expected * branch.class_weights
        expected = expected / targets.sum().clamp(min=1) * branch.hyp.cls * len(batch['img'])
        expected = expected * (candidate_criterion.o2m if name == 'one2many' else candidate_criterion.o2o)
        actual = candidate['gradients'][name]['scores'].permute(0, 2, 1)
        torch.testing.assert_close(actual, expected, rtol=5e-6, atol=3e-8)
        for key in ('boxes', 'kpts', 'kpts_sigma'):
            if key in control['gradients'][name]:
                same_tensor(control['gradients'][name][key], candidate['gradients'][name][key])
        fg = control['assigned'][name][0]
        checks[name] = {
            'foreground_anchors': int(fg.sum()), 'background_anchors': int((~fg).sum()),
            'classification_terms': targets.numel(),
            'maximum_analytic_gradient_difference': float((actual - expected).abs().max()),
            'maximum_absolute_expected_gradient': float(expected.abs().max()),
            'analytic_gradient_rtol': 5e-6, 'analytic_gradient_atol': 3e-8,
            'box_keypoint_and_sigma_gradients_bitwise_unchanged': True,
            'assignments_geometry_and_soft_targets_bitwise_unchanged': True,
        }
    for index in (0, 1, 2, 4, 5):
        same_tensor(control['loss'][index], candidate['loss'][index])
    for key in control['flow_gradients']:
        same_tensor(control['flow_gradients'][key], candidate['flow_gradients'][key])
    for left, right in zip(control_stats, candidate_stats):
        for field in ('original_target_score_sum', 'normalizer', 'partial_target_score_sum'):
            require(left[field] == right[field], 'Native targets or normalizer changed')
        require(left['modulated_classification_terms'] == 0
                and right['modulated_classification_terms'] == right['classification_terms'],
                'QFL did not cover the declared term inventory')
    return {
        'beta0_native_loss_and_prediction_gradients_bitwise_equal': True,
        'beta2_nonclassification_loss_components_bitwise_equal': True,
        'beta2_rle_flow_parameter_gradients_bitwise_equal': True,
        'analytic_classification_gradient_check': checks,
        'native_loss_components': native['loss'].tolist(),
        'beta2_loss_components': candidate['loss'].tolist(),
        'control_observations': control_stats, 'candidate_observations': candidate_stats,
    }, candidate


def full_model_equivalence(model, batch):
    initial = deepcopy(model.state_dict())
    outcomes = []
    for beta in (None, 0):
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(46)
        criterion = model.init_criterion() if beta is None else make_criterion(model, beta)
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
    return {'beta0_native_full_network_parameter_gradients_bitwise_equal': True,
            'parameter_tensors_with_gradient': len(outcomes[0][1]), 'optimizer_updates': 0}


def verify_guard(receipt):
    """Validate the numerical gate receipt, including its executed source identities."""
    require(receipt['schema_version'] == 1 and receipt['status'] == 'passed'
            and receipt['optimizer_updates'] == 0
            and receipt['initializer_sha256'] == INITIALIZER_SHA
            and receipt['dataset_archive_sha256'] == PR137_SHA
            and receipt['native_sources_sha256'] == verify_native_sources()
            and receipt['loss_weighting_spec_sha256'] == digest_json(SPEC)
            and receipt['criterion_contract'] == CRITERION_CONTRACT,
            'QFL guard receipt identity differs')
    script_root = Path(__file__).resolve().parent
    for name in ('quality_focal_r61.py', 'check_quality_focal_r61.py',
                 'check_partial_classification_r51.py', 'partial_classification_r51.py'):
        require(receipt['source_sha256'][name] == sha(script_root / name), 'QFL guard source differs: ' + name)
    scalar = receipt['scalar_gradient_guard']
    require(scalar['status'] == 'passed' and scalar['double_precision_gradcheck'] is True
            and scalar['detached_modulation_is_detectably_wrong'] is True
            and scalar['zero_modulation_control_returns_original_tensor'] is True
            and math.isfinite(scalar['maximum_analytic_gradient_difference'])
            and 0 <= scalar['maximum_analytic_gradient_difference'] <= 1e-12,
            'QFL scalar derivative verification is incomplete')
    cases = ('mixed_real_batch', 'native_class_weights_1_7', 'empty_batch',
             'unavailable_coordinate_mutation', 'all_corners_unavailable_synthetic_batch')
    for case_name in cases:
        case = receipt[case_name]
        require(all(case[field] is True for field in (
            'beta0_native_loss_and_prediction_gradients_bitwise_equal',
            'beta2_nonclassification_loss_components_bitwise_equal',
            'beta2_rle_flow_parameter_gradients_bitwise_equal')), 'QFL native invariants failed: ' + case_name)
        require(set(case['analytic_classification_gradient_check']) == set(BRANCHES), 'QFL guard omitted a branch')
        for name, branch in case['analytic_classification_gradient_check'].items():
            difference, magnitude = branch['maximum_analytic_gradient_difference'], branch['maximum_absolute_expected_gradient']
            require(branch['box_keypoint_and_sigma_gradients_bitwise_unchanged'] is True
                    and branch['assignments_geometry_and_soft_targets_bitwise_unchanged'] is True
                    and branch['analytic_gradient_rtol'] == 5e-6 and branch['analytic_gradient_atol'] == 3e-8
                    and math.isfinite(difference) and math.isfinite(magnitude)
                    and 0 <= difference <= 3e-8 + 5e-6 * magnitude,
                    'QFL analytic branch derivative check failed: ' + case_name + '/' + name)
        for field in ('native_loss_components', 'beta2_loss_components'):
            require(len(case[field]) == 6 and all(math.isfinite(x) for x in case[field]), 'QFL loss component inventory differs')
        require({x['branch'] for x in case['control_observations']} == set(BRANCHES)
                and {x['branch'] for x in case['candidate_observations']} == set(BRANCHES),
                'QFL observations omit a native branch')
        for control, candidate in zip(case['control_observations'], case['candidate_observations']):
            require(control['branch'] == candidate['branch']
                    and control['focal_beta'] == 0 and candidate['focal_beta'] == 2
                    and control['normalizer'] == candidate['normalizer']
                    and control['original_target_score_sum'] == candidate['original_target_score_sum']
                    and control['modulated_classification_terms'] == 0
                    and candidate['modulated_classification_terms'] == candidate['classification_terms'],
                    'QFL observation contract differs')
    full = receipt['full_network_control_equivalence']
    require(full['beta0_native_full_network_parameter_gradients_bitwise_equal'] is True
            and full['parameter_tensors_with_gradient'] > 0 and full['optimizer_updates'] == 0
            and receipt['unavailable_coordinates_have_no_supervision_effect'] is True
            and receipt['native_branch_schedule_bitwise_equal'] is True,
            'QFL complete-network or missing-coordinate guard is incomplete')
    expected_schedule = []
    for epoch in range(30):
        o2m = .8 if epoch == 0 else max(1 - epoch / 29, 0) * (.8 - .1) + .1
        expected_schedule.append([epoch, o2m, max(1 - o2m, 0)])
    require(receipt['native_branch_schedule'] == expected_schedule, 'QFL native branch schedule differs')
    return {'status': 'passed', 'cases': len(cases), 'native_branches': list(BRANCHES),
            'native_parameter_gradient_tensors': full['parameter_tensors_with_gradient'],
            'optimizer_updates': 0, 'loss_weighting_spec_sha256': digest_json(SPEC)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--inputs-dir', '--inputs', dest='inputs', type=Path, required=True)
    parser.add_argument('--archive', '--dataset-archive', dest='archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, inputs = args.root.resolve(), args.inputs.resolve()
    require(not args.output.exists(), 'Refusing to replace an existing gradient guard receipt')
    native_hashes = verify_native_sources()
    weights = inputs / 'weights/cassette_lower_epoch173.pt'
    require(sha(weights) == INITIALIZER_SHA, 'Epoch173 initializer differs')
    torch.set_num_threads(4)
    torch.manual_seed(46)
    scalar = scalar_gradient_guard()
    model = YOLO(weights).model.float().train().requires_grad_(True)
    model.args = get_cfg(overrides={'epochs': 30, 'batch': 4, 'imgsz': 1280})
    batch, sources = make_batch(root, inputs, args.archive)
    with torch.no_grad():
        predictions = model(batch['img'])
    require(set(predictions) == set(BRANCHES), 'Expected both native prediction branches')
    mixed, mixed_outputs = guard_case(model, predictions, batch)
    previous_weights = getattr(model, 'class_weights', None)
    model.class_weights = torch.tensor([1.7])
    class_weighted, _ = guard_case(model, predictions, batch)
    model.class_weights = previous_weights
    empty = {**batch, **{key: batch[key][:0] for key in ('batch_idx', 'cls', 'bboxes', 'keypoints')}}
    empty_result, _ = guard_case(model, predictions, empty)
    missing = {**batch, 'keypoints': batch['keypoints'].clone()}
    unavailable = missing['keypoints'][..., 2] == 0
    require(bool(unavailable.any()), 'Guard lacks unavailable corners')
    missing['keypoints'][..., :2][unavailable] = .37
    missing_result, missing_outputs = guard_case(model, predictions, missing)
    assert_same(mixed_outputs, missing_outputs)
    for name in BRANCHES:
        same_tensor(mixed_outputs['classification_targets'][name], missing_outputs['classification_targets'][name])
    invisible = {**batch, 'keypoints': torch.zeros_like(batch['keypoints'])}
    invisible_result, _ = guard_case(model, predictions, invisible)
    full_network = full_model_equivalence(model, batch)
    schedules = []
    for beta in (None, 0, 2):
        criterion = model.init_criterion() if beta is None else make_criterion(model, beta)
        values = []
        for _ in range(30):
            values.append([criterion.updates, criterion.o2m, criterion.o2o])
            criterion.update()
        schedules.append(values)
    require(schedules[0] == schedules[1] == schedules[2], 'Native branch schedule differs')
    source_names = ('quality_focal_r61.py', 'check_quality_focal_r61.py',
                    'check_partial_classification_r51.py', 'partial_classification_r51.py')
    result = {
        'schema_version': 1, 'status': 'passed',
        'purpose': 'QFL correctness and differentiation; no training or accuracy claim',
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'device': 'cpu', 'threads': torch.get_num_threads()},
        'native_sources_sha256': native_hashes, 'initializer_sha256': INITIALIZER_SHA,
        'dataset_archive_sha256': PR137_SHA, 'real_training_sources': sources,
        'guard_input_shape': list(batch['img'].shape),
        'guard_image_transform': 'RGB resize to 320x192; normalized coordinates retained; one synthetic blank appended',
        'production_training_imgsz': 1280, 'optimizer_updates': 0,
        'source_sha256': {name: sha(root / 'scripts' / name) for name in source_names},
        'loss_weighting_spec_sha256': digest_json(SPEC), 'criterion_contract': CRITERION_CONTRACT,
        'scalar_gradient_guard': scalar, 'mixed_real_batch': mixed,
        'native_class_weights_1_7': class_weighted, 'empty_batch': empty_result,
        'unavailable_coordinate_mutation': missing_result,
        'unavailable_coordinates_have_no_supervision_effect': True,
        'all_corners_unavailable_synthetic_batch': invisible_result,
        'full_network_control_equivalence': full_network,
        'native_branch_schedule_bitwise_equal': True, 'native_branch_schedule': schedules[0],
        'adaptation': 'Differentiable QFL modulation on existing soft targets, not full GFL or a new calibration guarantee',
    }
    verify_guard(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': 'passed', 'output': str(args.output),
                      'branches': mixed['analytic_classification_gradient_check'],
                      'full_network': full_network}, indent=2))


if __name__ == '__main__':
    main()
