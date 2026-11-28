#!/usr/bin/env python3
"""Recompute two diagnostic images using the returned residual checkpoint on CPU."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

IMAGES = ('new_000008', 'new_000010')
MODEL_SHA = '19c7e95aca0019c3b1eb60de4c36c6ecd30d35fdcaaa103606cdd7d553f7006a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--return-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root/'scripts'))
    import torch
    import torchvision
    from PIL import Image
    import residual_corner_r59 as correction
    import run_residual_corner_r59 as runner
    from audit_reflection_return import load_references
    from pose_scoring import score_image
    from predict_pose_folder import postclip_nms

    read = lambda name: json.loads((args.return_dir/name).read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(args.model) == MODEL_SHA
    summary, predictions, traces = read('summary.json'), read('predictions.json'), read('correction-traces.json')
    refs = load_references(args.root, summary['contract'])
    by_trace = {(row['image_id'], row['raw_index']): row for row in traces}
    runner.shared.runtime_setup('cpu')
    saved = torch.load(args.model, map_location='cpu', weights_only=True)
    assert saved['recipe'] == runner.RECIPE and saved['backbone_sha256'] == runner.BACKBONE_SHA
    network = correction.build_model()
    network.load_state_dict(saved['model'], strict=True)
    network.eval()
    rows, coordinate_drift, delta_drift = [], [], []
    scopes = []
    for identity in IMAGES:
        path = args.images_dir/(identity+'.png')
        assert digest(path) == summary['contract']['images'][identity]
        with Image.open(path) as source:
            image = source.convert('RGB')
        original = predictions['control'][identity]
        raw = []
        with torch.inference_mode():
            for index, proposal in enumerate(original['raw']):
                geometry = correction.crop_geometry(proposal['bbox_xyxy'])
                array = correction.input_tensor(image, proposal, geometry, f'{identity}:p{index}')
                offsets = network(torch.from_numpy(array[None]))[0].tolist()
                replayed = correction.apply_corrections(proposal, offsets, geometry)
                trace = by_trace[(identity, index)]
                delta_drift.extend(abs(a-b) for p, q in zip(offsets, trace['canvas_offsets']) for a, b in zip(p, q))
                for p, q in zip(replayed['keypoints'], trace['output_points']):
                    assert (p[0] is None) == (q[0] is None)
                    if p[0] is not None:
                        coordinate_drift.extend(abs(a-b) for a, b in zip(p[:2], q[:2]))
                    assert p[2] == q[2]
                assert {k: v for k, v in replayed.items() if k != 'keypoints'} == {
                    k: v for k, v in proposal.items() if k != 'keypoints'}
                raw.append(replayed)
                rows.append({'image_id': identity, 'raw_index': index, 'canvas_offsets': offsets,
                             'output_points': replayed['keypoints']})
        for threshold in (.13, .25):
            actual, _ = postclip_nms([p for p in raw if p['confidence'] > threshold])
            expected, _ = postclip_nms([p for p in predictions['treatment'][identity]['raw'] if p['confidence'] > threshold])
            for reference, gold in refs.items():
                a = score_image(gold[identity], actual, 1280, 720, .5, .5)
                b = score_image(gold[identity], expected, 1280, 720, .5, .5)
                assert a['unmatched_gold_indices'] == b['unmatched_gold_indices']
                assert a['unmatched_prediction_indices'] == b['unmatched_prediction_indices']
                assert a['duplicate_prediction_indices'] == b['duplicate_prediction_indices']
                assert [(m['gold_index'], m['prediction_index']) for m in a['matches']] == [(m['gold_index'], m['prediction_index']) for m in b['matches']]
                signatures = []
                for outputs, score in [(actual, a), (expected, b)]:
                    matches = {m['gold_index']: m['prediction_index'] for m in score['matches']}
                    states = []
                    for index, target in enumerate(gold[identity]):
                        if not all(p[2] == 2 for p in target['keypoints']):
                            continue
                        pi = matches.get(index)
                        pred = outputs[pi] if pi is not None else None
                        errors = [(p[0]-g[0])**2+(p[1]-g[1])**2 for p, g in zip(pred['keypoints'], target['keypoints'])] if pred and all(correction.point_available(p) for p in pred['keypoints']) else None
                        states.append((index, errors is not None and max(errors) <= 25, errors is not None and max(errors) <= 100))
                    signatures.append(states)
                assert signatures[0] == signatures[1]
                scopes.append({'image': identity, 'reference': reference, 'box_threshold': threshold,
                               'matching_and_per_pair_states_unchanged': True})
    assert max(coordinate_drift) < .05
    receipt = {'schema': 'residual-selected-cpu-forward-r60-v1', 'status': 'passed',
        'selection': 'Two images chosen after audit for the largest reviewed regression and a threshold-crossing improvement; diagnostic, not a new accuracy experiment.',
        'images': list(IMAGES), 'raw_predictions_recomputed': len(rows), 'score_scopes_checked': len(scopes),
        'model_sha256': MODEL_SHA, 'source_sha256': digest(Path(__file__)),
        'maximum_canvas_offset_component_difference': max(delta_drift),
        'maximum_image_coordinate_component_difference_px': max(coordinate_drift),
        'all_matching_and_pair_success_states_unchanged': True, 'training_executed': False,
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'torchvision': str(torchvision.__version__), 'device': 'cpu',
                    'numpy': importlib.metadata.version('numpy'), 'Pillow': importlib.metadata.version('Pillow')},
        'original_runtime': read('detector-runtime.json'), 'scope_checks': scopes,
        'predictions': rows,
        'limitation': 'CPU Torch2.9 replay of selected images is compared with L4 Torch2.14 outputs. It does not rerun or independently verify training execution.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'raw_predictions_recomputed', 'maximum_canvas_offset_component_difference', 'maximum_image_coordinate_component_difference_px', 'score_scopes_checked')}, indent=2))


if __name__ == '__main__':
    main()
