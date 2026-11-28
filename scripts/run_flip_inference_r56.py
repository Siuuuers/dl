#!/usr/bin/env python3
"""R56 explicit horizontal-flip inference with original-coordinate pose scoring."""
import argparse
from copy import deepcopy
import importlib.metadata as md
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import zipfile

sys.dont_write_bytecode = True
import run_inference_ablation_r53 as r53
from diagnose_fronts_r50 import (CONF, MATCH, R50_SHA, checked, encoded, git, need,
                                 prepare as prepare_inputs, put, sha,
                                 source_check as inherited_source_check)
from run_inference_ablation_r53 import (PINNED_PACKAGES, ULTRALYTICS_WHEEL, TOLERANCES,
    TIMING_REPEATS, baseline_equivalence, reports, case_inventory)

PARENT_SOURCE = '5001d83439eaf31c7ecae378c27a0de847e65686'
R53_DRIVER_SHA = '8c8a37c3dcaccde5c568e9df6f44651cdb05965ca5f5ed5b0f2978852387dcc1'
INFERENCE_SOURCE_HASHES = {**r53.INFERENCE_SOURCE_HASHES,
    'utils/instance.py': 'ebc8b6337f4636c1457473c079d007693b3d8766c4891e2b5dd521c22c2e5527',
    'data/augment.py': '802333ea359def55e88f61d6726e35c29d28a6491110840a5e912ec1fafd1de3'}
FLIP_RULE = {'pixel_transform': 'contiguous BGR array[:, ::-1, :]',
    'coordinate_convention': 'continuous image coordinates; x_original = width - x_flipped',
    'box_inverse': '[width-xmax, ymin, width-xmin, ymax]', 'keypoint_slot_order': [1, 0],
    'keypoint_scores': 'travel with their point and are swapped with slots',
    'clipping_order': 'native inverse-letterbox and clipping, inverse horizontal mapping, then original-frame postclip NMS',
    'extra_coordinate_clipping': False, 'generic_augmented_prediction': False, 'fusion': False}


def settings():
    from evaluate_pose import INFERENCE
    return {'experiment': 'flip',
        'control': {'geometry': 'identity', 'imgsz': 1280},
        'treatment': {'geometry': 'horizontal_flip', 'imgsz': 1280},
        'common': {**INFERENCE, 'augment': False, 'conf': CONF,
                   'matching_iou': MATCH, 'endpoint_conf': .5},
        'flip': FLIP_RULE, 'baseline_tolerances': TOLERANCES,
        'timing_repeats': TIMING_REPEATS, 'training': False, 'automatic_promotion': False}


def transform_bgr(image, method):
    import numpy as np
    need(isinstance(image, np.ndarray) and image.dtype == np.uint8
         and image.ndim == 3 and image.shape[2] == 3 and min(image.shape[:2]) > 0,
         'Expected nonempty uint8 H x W x 3 BGR image')
    need(method in ('identity', 'horizontal_flip'), 'Unknown fixed geometric arm')
    return np.ascontiguousarray((image if method == 'identity' else image[:, ::-1, :]).copy())


def inverse_horizontal_predictions(predictions, width=1280):
    need(isinstance(width, (int, float)) and not isinstance(width, bool)
         and math.isfinite(width) and width > 0, 'Expected a positive finite image width')
    restored = []
    for prediction in predictions:
        box, points = prediction['bbox_xyxy'], prediction['keypoints']
        need(len(box) == 4 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in box)
             and len(points) == 2 and all(len(point) == 3 for point in points),
             'Expected one continuous xyxy box and two three-value corner slots')
        need(all(all(x is None or isinstance(x, (int, float)) and math.isfinite(x)
                     for x in point[:2]) and isinstance(point[2], (int, float))
                 and math.isfinite(point[2]) for point in points), 'Nonfinite landmark record')
        result = deepcopy(prediction)
        result['bbox_xyxy'] = [width-box[2], box[1], width-box[0], box[3]]
        result['keypoints'] = [[None if points[j][0] is None else width-points[j][0],
                               points[j][1], points[j][2]] for j in (1, 0)]
        restored.append(result)
    return restored


def original_frame_output(native_predictions, arm):
    from predict_pose_folder import postclip_nms
    need(arm in ('control', 'treatment'), 'Unknown comparison arm')
    raw = (deepcopy(native_predictions) if arm == 'control'
           else inverse_horizontal_predictions(native_predictions))
    kept, audit = postclip_nms(raw)
    return {'raw': raw, 'postclip': kept, 'filter': audit}


def source_check(args):
    root = args.root.resolve()
    need(re.fullmatch('[0-9a-f]{64}', args.script_sha256) is not None, 'Full driver SHA256 required')
    checked(root / 'scripts/run_flip_inference_r56.py', args.script_sha256)
    checked(root / 'scripts/run_inference_ablation_r53.py', R53_DRIVER_SHA)
    need(Path(__file__).resolve() == root / 'scripts/run_flip_inference_r56.py',
         'Run the R56 driver from its pinned source checkout')
    context = inherited_source_check(root, args.source_commit, r53.R50_DRIVER_SHA, R50_SHA)
    inherited = context['contract']
    need(inherited['source_files_sha256'].get('scripts/run_flip_inference_r56.py') == args.script_sha256,
         'R56 driver is not tracked by the pinned Git tree')
    metadata = {}
    for relative in ('metadata/model_card.json', 'models/cassette_lower_yolo26s_pose.yaml'):
        data = (root / relative).read_bytes()
        blob = git(root, 'rev-parse', 'HEAD:' + relative)
        import hashlib
        need(hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == blob,
             'Model metadata differs from pinned Git source')
        inherited['source_files_sha256'][relative] = sha(data)
        metadata[relative] = sha(data)
    card = json.loads((root / 'metadata/model_card.json').read_text())
    need(card['model'] == 'YOLO26s-Pose' and card['kpt_shape'] == [2, 3]
         and card['flip_idx'] == [1, 0] and card['checkpoint_dependency'] == 'ultralytics 8.4.182',
         'Model card or keypoint-slot contract differs')
    context['contract'] = {'schema': 'baomai-r56-flip-inputs-v1',
        'source': args.source_commit, 'parent_source': PARENT_SOURCE,
        'script_sha256': args.script_sha256, 'r53_driver_sha256': R53_DRIVER_SHA,
        'source_files_sha256': inherited['source_files_sha256'],
        'references': inherited['references'], 'row_maps': inherited['row_maps'],
        'images': inherited['images'], 'settings': settings(),
        'primary_reference': 'frozen666', 'reviewed_reference': 'r50',
        'model': {'architecture': card['model'], 'task': 'pose', 'kpt_shape': [2, 3],
                  'flip_idx': [1, 0], 'classes': ['cassette_front'], 'native_head': 'Pose26',
                  'checkpoint_role': 'PR137 candidate; research checkpoint, not deployment selection',
                  'epoch173_initializer_sha256': card['sha256'], 'metadata_sha256': metadata},
        'required_packages': PINNED_PACKAGES, 'ultralytics_wheel_provenance': ULTRALYTICS_WHEEL,
        'required_inference_source_sha256': INFERENCE_SOURCE_HASHES,
        'development_only': True, 'training': False, 'automatic_promotion': False}
    return prepare_inputs(args, context)


def complementarity(cases, scored):
    result = {}
    for reference in scored['control']:
        selected = [row for row in cases['paired_rows'] if row['reference'] == reference]
        if 'treatment' not in scored:
            result[reference] = {'treatment_executed': False}
            continue
        groups = {}
        for count in (None, 0, 1, 2):
            rows = [row for row in selected if count is None or row['locatable_count'] == count]
            both = sum(row['control']['output_number'] is not None and
                       row['treatment']['output_number'] is not None for row in rows)
            only_control = [row['case_id'] for row in rows if row['newly_missed']]
            only_treatment = [row['case_id'] for row in rows if row['recovered']]
            groups['all' if count is None else str(count)] = {
                'reference_rows': len(rows), 'matched_by_both': both,
                'control_only_case_ids': only_control, 'flip_only_case_ids': only_treatment,
                'matched_by_either_diagnostic_bound': both+len(only_control)+len(only_treatment),
                'missed_by_both': len(rows)-both-len(only_control)-len(only_treatment),
                'within5_gain_case_ids': [row['case_id'] for row in rows if row['within5_gain']],
                'within5_loss_case_ids': [row['case_id'] for row in rows if row['within5_loss']],
                'within10_gain_case_ids': [row['case_id'] for row in rows if row['within10_gain']],
                'within10_loss_case_ids': [row['case_id'] for row in rows if row['within10_loss']]}
        empties = {}
        for image, control in scored['control'][reference]['postclip']['per_image'].items():
            if control['gold_count'] == 0:
                empties[image] = {arm: score[reference]['postclip']['per_image'][image]['prediction_count']
                                 for arm, score in scored.items()}
        result[reference] = {'treatment_executed': True, 'by_locatable_corner_count': groups,
                             'reference_empty_scene_outputs': empties}
    return {'schema': 'r56-paired-complementarity-v1', 'fusion_performed': False,
        'interpretation': 'Reference-associated either-view recovery is a diagnostic bound; it is not an implemented selector or a fused detector.',
        'references': result}


def summarize_timing(measurements):
    result = {}
    for arm in ('control', 'treatment'):
        rows = [row for row in measurements if row['arm'] == arm]
        if not rows:
            continue
        entry = {'measured_calls': len(rows), 'images': len({row['image_id'] for row in rows}),
                 'input_tensor_shapes': sorted({tuple(row['input_tensor_shape']) for row in rows})}
        for key in ('input_transform_ms', 'predict_call_ms', 'record_and_postclip_ms', 'array_to_output_ms'):
            values = sorted(row[key] for row in rows)
            entry[key] = {'median': statistics.median(values), 'mean': statistics.mean(values),
                          'p95_nearest_rank': values[math.ceil(.95*len(values))-1]}
        entry['stage_ms_mean'] = {key: statistics.mean(row['stage_ms'][key] for row in rows)
                                 for key in rows[0]['stage_ms']}
        for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes', 'cuda_allocated_before_bytes'):
            values = [row[key] for row in rows if row[key] is not None]
            entry[key] = max(values) if values else None
        result[arm] = entry
    return json.loads(encoded(result))


def archive_name():
    return 'baomai-flip-inference-r56-results.zip'


def package(context, payload):
    work = context['work']
    entries = {name: encoded(value) for name, value in payload.items()}
    entries['preflight.json'] = encoded(context['preflight'])
    entries['input-audit.json'] = encoded(context['receipt'])
    entries['run_flip_inference_r56.py'] = Path(__file__).read_bytes()
    entries['run_inference_ablation_r53.py'] = Path(r53.__file__).read_bytes()
    inventory = {name: {'bytes': len(data), 'sha256': sha(data)} for name, data in entries.items()}
    for name, data in entries.items():
        put(work / name, data)
    entries['inventory.json'] = encoded(inventory)
    target = work / archive_name()
    need(not target.exists(), 'Completed archive exists; verify and reuse it')
    temp = target.with_suffix('.part')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(entries.items()):
            z.writestr(name, data)
    temp.replace(target)
    return verify_return(context)


def verify_return(context):
    target = context['work'] / archive_name()
    with zipfile.ZipFile(target) as z:
        expected = {'summary.json', 'predictions.json', 'native-predictions.json',
            'baseline-equivalence.json', 'cases.json', 'complementarity.json',
            'timing.json', 'runtime.json', 'repeatability.json', 'repeated-predictions.json',
            'repeated-native-predictions.json', 'preflight.json', 'input-audit.json',
            'run_flip_inference_r56.py', 'run_inference_ablation_r53.py', 'inventory.json'}
        names = z.namelist()
        need(len(names) == len(set(names)) and set(names) == expected and z.testzip() is None,
             'Return membership or CRC differs; preserved')
        inventory = json.loads(z.read('inventory.json'))
        need(set(inventory) == expected - {'inventory.json'}, 'Return inventory differs')
        for name, entry in inventory.items():
            data = z.read(name)
            need(entry == {'bytes': len(data), 'sha256': sha(data)}, 'Return hash differs: ' + name)
        need(z.read('run_flip_inference_r56.py') == Path(__file__).read_bytes(), 'Returned R56 driver differs')
        need(z.read('run_inference_ablation_r53.py') == Path(r53.__file__).read_bytes(), 'Returned R53 helper differs')
        need(json.loads(z.read('preflight.json')) == context['preflight'], 'Return preflight differs')
        need(json.loads(z.read('input-audit.json')) == context['receipt'], 'Return input audit differs')
        summary, arms, native = (json.loads(z.read(name)) for name in
                                 ('summary.json', 'predictions.json', 'native-predictions.json'))
        need(summary['contract'] == context['contract'], 'Return source or settings differ')
        need(set(arms) in ({'control'}, {'control', 'treatment'}) and set(native) == set(arms),
             'Return arm membership differs')
        def check_outputs(outputs, native_outputs):
            need(set(outputs) == set(native_outputs), 'Native arm membership differs')
            for arm, values in outputs.items():
                need(set(values) == set(context['records']) and set(native_outputs[arm]) == set(values),
                     'Return image membership differs')
                for image, row in values.items():
                    need(original_frame_output(native_outputs[arm][image], arm) == row,
                         'Inverse mapping or original-frame postclip replay differs: ' + arm + '/' + image)
        check_outputs(arms, native)
        baseline = baseline_equivalence(context['refs'], context['prior'], arms['control'])
        need(json.loads(z.read('baseline-equivalence.json')) == baseline, 'Baseline gate replay differs')
        need(not ('treatment' in arms and not baseline['passed']), 'Treatment ran after a failed baseline')
        scored = {arm: reports(context['refs'], output) for arm, output in arms.items()}
        need(scored == summary['reports'], 'Returned score replay differs')
        cases = case_inventory(context['refs'], context['maps'], arms, scored)
        need(json.loads(z.read('cases.json')) == cases, 'Paired case replay differs')
        need(json.loads(z.read('complementarity.json')) == complementarity(cases, scored),
             'Paired complementarity differs')
        repeated = json.loads(z.read('repeated-predictions.json'))
        repeated_native = json.loads(z.read('repeated-native-predictions.json'))
        need(len(repeated) == len(repeated_native) == (TIMING_REPEATS if baseline['passed'] else 0),
             'Repeated output membership differs')
        checks = []
        for index, (observation, native_observation) in enumerate(zip(repeated, repeated_native), 1):
            need(set(observation) == {'control', 'treatment'}, 'Repeated arm membership differs')
            check_outputs(observation, native_observation)
            for arm, output in observation.items():
                checks.append({'arm': arm, 'repeat': index,
                               **baseline_equivalence(context['refs'], arms[arm], output)})
        need(checks == json.loads(z.read('repeatability.json')), 'Repeatability replay differs')
        repeatable = all(row['passed'] for row in checks)
        valid = baseline['passed'] and repeatable
        status = 'completed' if valid else 'blocked_baseline' if not baseline['passed'] else 'blocked_repeatability'
        need(summary['status'] == status and summary['baseline_passed'] == baseline['passed']
             and summary['repeated_outputs_stable'] == repeatable
             and summary['interpretation_allowed'] == valid
             and summary['treatment_executed'] == ('treatment' in arms), 'Return status or gates differ')
        timing = json.loads(z.read('timing.json'))
        need(timing['summary'] == summarize_timing(timing['measurements']), 'Timing summary replay differs')
        expected_calls = {(arm, image, repeat) for arm in ('control', 'treatment')
                          for image in context['records'] for repeat in range(1, TIMING_REPEATS+1)} if baseline['passed'] else set()
        calls = [(row['arm'], row['image_id'], row['repeat']) for row in timing['measurements']]
        need(len(calls) == len(set(calls)) and set(calls) == expected_calls, 'Timing call membership differs')
        need(summary['fresh_inference'] is True and summary['training'] is False
             and summary['automatic_promotion'] is False and summary['fusion_performed'] is False,
             'Return experiment declarations differ')
    print('Verified R56 evidence:', target, 'SHA256:', sha(target.read_bytes()), flush=True)
    return summary

def run(context, device):
    if (context['work'] / archive_name()).exists():
        return verify_return(context)
    for package_name, required in PINNED_PACKAGES.items():
        need(md.version(package_name) == required, f'Expected {package_name}=={required}')
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose.predict import PosePredictor
    from evaluate_pose import INFERENCE, _prediction_records
    from partial_classification_r51 import verify_native_sources
    from pose_checkpoint import check_loaded_pose_model
    need(device == 'cpu' or torch.cuda.is_available(), 'Requested GPU is unavailable; select a GPU runtime')
    native = verify_native_sources()
    package_root = Path(importlib.util.find_spec('ultralytics').origin).parent
    for relative, expected in INFERENCE_SOURCE_HASHES.items():
        checked(package_root / relative, expected)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    class ShapePredictor(PosePredictor):
        def preprocess(self, im):
            need(len(im) == 1 and not self.scale_fill and self.model.format == 'pt',
                 'Expected one-image, aspect-preserving native PT letterbox')
            original_h, original_w = im[0].shape[:2]
            tensor = super().preprocess(im)
            self.r56_input_shape = list(tensor.shape)
            gain = min(self.imgsz[0]/original_h, self.imgsz[1]/original_w)
            resized_h, resized_w = round(original_h*gain), round(original_w*gain)
            pad_h, pad_w = tensor.shape[2]-resized_h, tensor.shape[3]-resized_w
            need(pad_h >= 0 and pad_w >= 0, 'Letterbox padding/gain contract differs')
            self.r56_letterbox = {'scale_gain': gain, 'resized_hw': [resized_h,resized_w],
                'padded_hw': list(tensor.shape[2:]),
                'padding_ltrb': [round(pad_w/2-.1),round(pad_h/2-.1),
                                 round(pad_w/2+.1),round(pad_h/2+.1)],
                'derivation': 'Verified native LetterBox default scaleup=True, scale_fill=False; original shape and self.imgsz.'}
            return tensor

    model = YOLO(str(context['weights'])); check_loaded_pose_model(model)
    head = model.model.model[-1]
    need(type(head).__name__ == 'Pose26' and head.nc == 1 and head.reg_max == 1,
         'Expected original native Pose26 checkpoint architecture')
    config = context['contract']['settings']
    def sync():
        if device != 'cpu':
            torch.cuda.synchronize(backend.device)
    for _ in range(2):
        model.predict(source=np.zeros((720, 1280, 3), dtype=np.uint8),
            predictor=ShapePredictor, device=device, conf=CONF, augment=False, **INFERENCE)
    backend = model.predictor.model
    need(not backend.end2end and not backend.fp16, 'Expected ordinary one-to-many NMS and FP32')
    torch.set_num_threads(4)
    runtime = {'python': platform.python_version(), 'platform': platform.platform(),
        'device': str(backend.device), 'gpu': torch.cuda.get_device_name(backend.device) if device != 'cpu' else None,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'packages': {n: md.version(n) for n in (*PINNED_PACKAGES, 'torchvision')},
        'native_sources_verified': native, 'inference_sources_verified': INFERENCE_SOURCE_HASHES,
        'ultralytics_wheel_provenance': ULTRALYTICS_WHEEL,
        'backend_end2end': bool(backend.end2end), 'backend_fp16': bool(backend.fp16),
        'torch_threads': torch.get_num_threads(), 'tf32': False, 'cudnn_benchmark': False,
        'prior_runtime': context['old_summary']['runtime'],
        'warmup': 'Two zero-image1280 predictions before baseline; two predictions per arm before timing, all excluded.',
        'timing_scope': 'Original decoded BGR array through identity/flip, prediction, inverse pose mapping and original-frame postclip. Excludes file I/O, model load, archive audit and camera acquisition. This is not camera FPS.',
        'memory_scope': 'CUDA allocator peak during each call; allocated and reserved differ and do not measure total device use.'}
    images = {}
    for identity in sorted(context['records']):
        array = cv2.imread(str(context['originals'] / (identity + '.png')), cv2.IMREAD_COLOR)
        need(array is not None and array.shape == (720, 1280, 3) and array.dtype == np.uint8,
             'Decoded image dimensions/type differ: ' + identity)
        images[identity] = array
    pixel_hashes = {i: sha(a.tobytes()) for i, a in images.items()}
    runtime['decoded_bgr_sha256'] = pixel_hashes
    runtime['flip_convention'] = FLIP_RULE
    measurements = []

    def infer(identity, arm, repeat, measured):
        source = images[identity]
        choice = config[arm]
        sync()
        if measured and device != 'cpu':
            torch.cuda.reset_peak_memory_stats(backend.device)
        before = torch.cuda.memory_allocated(backend.device) if device != 'cpu' else None
        start = time.perf_counter()
        transformed = transform_bgr(source, choice['geometry'])
        after_transform = time.perf_counter()
        result = model.predict(source=transformed, device=device, conf=CONF, augment=False,
            **{**INFERENCE, 'imgsz': choice['imgsz']})[0]
        sync(); after_predict = time.perf_counter()
        need(tuple(result.orig_shape) == (720, 1280), 'Original-coordinate contract changed')
        native_raw = _prediction_records(result)
        row = original_frame_output(native_raw, arm)
        sync(); finished = time.perf_counter()
        if measured:
            measurements.append({'image_id': identity, 'arm': arm, 'repeat': repeat,
                'input_transform_ms': (after_transform-start)*1000,
                'predict_call_ms': (after_predict-after_transform)*1000,
                'record_and_postclip_ms': (finished-after_predict)*1000,
                'array_to_output_ms': (finished-start)*1000,
                'stage_ms': {k: float(v) for k, v in result.speed.items()},
                'input_tensor_shape': model.predictor.r56_input_shape,
                'letterbox': model.predictor.r56_letterbox,
                'cuda_allocated_before_bytes': before,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device) if device != 'cpu' else None,
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device) if device != 'cpu' else None,
                'transformed_bgr_sha256': sha(transformed.tobytes())})
        return row, native_raw

    control, native_control = {}, {}
    for identity in sorted(images):
        control[identity], native_control[identity] = infer(identity, 'control', 0, False)
    baseline = baseline_equivalence(context['refs'], context['prior'], control)
    arms, repeatability, repeated = {'control': control}, [], []
    native_arms, native_repeated = {'control': native_control}, []
    if baseline['passed']:
        print('Baseline reproduction passed; running the one prespecified treatment.', flush=True)
        first = sorted(images)[0]
        for arm in ('control', 'treatment'):
            for _ in range(2):
                infer(first, arm, -1, False)
        for repeat in range(TIMING_REPEATS):
            observed = {'control': {}, 'treatment': {}}
            native_observed = {'control': {}, 'treatment': {}}
            for number, identity in enumerate(sorted(images)):
                order = ('control', 'treatment') if (number+repeat) % 2 == 0 else ('treatment', 'control')
                for arm in order:
                    observed[arm][identity], native_observed[arm][identity] = infer(identity, arm, repeat+1, True)
            if repeat == 0:
                arms['treatment'] = observed['treatment']
                native_arms['treatment'] = native_observed['treatment']
            repeated.append(observed)
            native_repeated.append(native_observed)
            for arm in ('control', 'treatment'):
                repeatability.append({'arm': arm, 'repeat': repeat+1,
                    **baseline_equivalence(context['refs'], arms[arm], observed[arm])})
            print(f'Finished paired timing repeat {repeat+1}/{TIMING_REPEATS}', flush=True)
    else:
        print('BASELINE GATE FAILED: treatment was not run. Download the diagnostic archive.', flush=True)
    need(pixel_hashes == {i: sha(a.tobytes()) for i, a in images.items()}, 'Original pixel array mutated')
    scored = {arm: reports(context['refs'], output) for arm, output in arms.items()}
    repeatable = all(r['passed'] for r in repeatability)
    valid = baseline['passed'] and repeatable
    summary = {'schema': 'baomai-r56-flip-results-v1', 'contract': context['contract'],
        'status': 'completed' if valid else 'blocked_baseline' if not baseline['passed'] else 'blocked_repeatability',
        'baseline_passed': baseline['passed'], 'repeated_outputs_stable': repeatable,
        'treatment_executed': 'treatment' in arms, 'interpretation_allowed': valid,
        'fresh_inference': True, 'training': False, 'automatic_promotion': False, 'fusion_performed': False,
        'reports': scored, 'interpretation': 'Development comparison only. No benefit is established until this returned evidence is reviewed; no model, label, filter or threshold is promoted.'}
    cases = case_inventory(context['refs'], context['maps'], arms, scored)
    payload = {'summary.json': summary, 'predictions.json': arms, 'native-predictions.json': native_arms,
        'baseline-equivalence.json': baseline, 'cases.json': cases,
        'complementarity.json': complementarity(cases, scored),
        'timing.json': {'measurements': measurements, 'summary': summarize_timing(measurements),
                        'order': 'Alternating arm order by image index and repeat; fixed20 images,3 repeats.'},
        'runtime.json': runtime, 'repeatability.json': repeatability,
        'repeated-predictions.json': repeated, 'repeated-native-predictions.json': native_repeated}
    return package(context, payload)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'verify'))
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--script-sha256', required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--originals-dir', type=Path)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    args.root = args.root.resolve()
    context = source_check(args)
    if args.action == 'prepare':
        print('R56 CPU preflight passed. No model was loaded and no fresh inference was performed.')
    else:
        result = run(context, args.device) if args.action == 'run' else verify_return(context)
        print(json.dumps({'status': result['status'], 'interpretation_allowed': result['interpretation_allowed'],
                          'treatment_executed': result['treatment_executed']}, indent=2))


if __name__ == '__main__':
    main()
