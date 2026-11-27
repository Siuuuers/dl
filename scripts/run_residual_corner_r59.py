#!/usr/bin/env python3
"""Train one fixed residual corner correction pilot from an exact identity output."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata as md
import json
import math
from pathlib import Path
import re
import sys
import time
import zipfile

sys.dont_write_bytecode = True
import residual_corner_r59 as correction
import run_corner_refinement_r58 as shared
from evaluate_pose import INFERENCE, parse_pose_labels
from pose_scoring import score_image
from predict_pose_folder import postclip_nms

r50, r53 = shared.r50, shared.r53
need, encoded, put = r50.need, r50.encoded, r50.put
sha, read, frozen, checked = shared.sha, shared.read, shared.frozen, shared.checked
SEED, EPOCHS, BATCH = 46, 30, 32
BACKBONE_SHA = shared.BACKBONE_SHA
RESULT_NAME = 'baomai-residual-corner-r59-results.zip'
UPLOAD_NAME = 'baomai-residual-corner-r59-upload.zip'
MODEL_NAME = 'correction-final.pt'
SCHEMA = 'baomai-residual-corner-r59-results-v1'
SOURCE_MEMBERS = ('run_residual_corner_r59.py', 'residual_corner_r59.py',
                  'run_corner_refinement_r58.py', 'corner_refinement_r58.py')
RECIPE = {
    'seed': SEED, 'epochs': EPOCHS, 'batch_size': BATCH, 'precision': 'float32',
    'optimizer': 'AdamW', 'learning_rate': .0001, 'weight_decay': .0001,
    'scheduler': 'CosineAnnealingLR', 'minimum_learning_rate': .000001,
    'gradient_clip_norm': 10., 'workers': 0, 'drop_last': False,
    'checkpoint_selection': 'fixed final epoch; no development-based selection',
    'model': 'ImageNet-pretrained ResNet50 V1.5, global average pool, zero-initialized 2048-to-4 linear head',
    'input_channels': 5, 'crop_factor': 1.5, 'canvas': [256, 256],
    'hint_sigma_canvas_pixels': 8., 'training_hint_noise_std_canvas_pixels': 0.,
    'rgb_augmentation': 'none', 'hint_augmentation': 'none; original point is both hint centre and correction origin',
    'output': 'two unconstrained XY offsets in the resized canvas coordinate system',
    'loss': 'SmoothL1, beta=1 canvas pixel, mean over eligible endpoint coordinate components',
    'target': '(reference_image_point - original_image_point) * actual_xy_resize_scale',
    'application': 'original_image_point + predicted_canvas_offset / actual_xy_resize_scale',
    'zero_output': 'preserves original point records exactly, without a forward/inverse coordinate round trip',
    'training_eligibility': 'reference visibility=2 and original point finite, within image, score>=0.5',
    'known_target_outside_crop': 'retained and counted; residual regression has no heatmap-domain restriction',
    'unavailable_initial_point': 'excluded from coordinate loss and preserved unchanged at inference',
    'invalid_corrected_point': 'nonfinite or outside 1280x720 image becomes unavailable; no clamping or reference-based fallback',
    'corner_confidence': 'original detector scores retained for controlled comparison; not recalibrated',
    'training_proposals': 'frozen research detector on the same 142 training photos; one-to-one box IoU>=0.5 matches',
    'negative_proposal_supervision': 'none; every raw proposal still passes through correction at evaluation',
    'relationship_to_IEF': 'one-step adaptation of image-plus-pose correction; not the full iterative training recipe',
}


def source_and_inputs(args):
    root, work = args.root.resolve(), args.work.resolve()
    need(not work.is_relative_to(root), 'Keep generated work outside the pinned source checkout')
    need(re.fullmatch('[0-9a-f]{64}', args.script_sha256) is not None, 'Full driver SHA256 required')
    checked(root/'scripts/run_residual_corner_r59.py', args.script_sha256)
    need(Path(__file__).resolve() == root/'scripts/run_residual_corner_r59.py', 'Run the driver from the pinned checkout')
    context = r50.source_check(root, args.source_commit, r53.R50_DRIVER_SHA, r50.R50_SHA)
    need(context['contract']['source_files_sha256'].get('scripts/run_residual_corner_r59.py') == args.script_sha256,
         'The driver must be tracked in the published source')
    work.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.resolve() if args.cache_dir else work.parent/'cache'
    cache.mkdir(parents=True, exist_ok=True)
    archive = args.archive.resolve() if args.archive else shared.prior.obtain_archive(cache)
    stage = shared.stage_training(root, work, cache, archive)
    backbone = shared.obtain_backbone(cache)
    context['contract'].update(schema='baomai-residual-corner-r59-inputs-v1',
        script_sha256=args.script_sha256, settings={'experiment': 'residual_corner_correction',
        'recipe': RECIPE, 'detector': {**INFERENCE, 'conf': .13, 'augment': False}},
        training=True, training_staging_sha256=sha(work/'staging-r58.json'),
        pretrained_backbone={'url': correction.crop.WEIGHTS_URL, 'sha256': BACKBONE_SHA,
                             'bytes': shared.BACKBONE_BYTES}, automatic_promotion=False)
    args.cache_dir, args.archive = cache, archive
    context = r50.prepare(args, context)
    frozen(work/'input-audit.json', context['receipt'])
    context.update(stage=stage, backbone=backbone)
    return context


def model_smoke(context):
    """Real pretrained CPU forward/backward; disposable fixtures, never an experiment result."""
    import numpy as np
    from PIL import Image
    import torch
    shared.runtime_setup('cpu')
    samples = []
    for row in context['stage']['records']:
        if row['split'] != 'train':
            continue
        gold = parse_pose_labels((context['work']/row['label']).read_text(), 1280, 720, row['id'])
        reference = next((g for g in gold if any(p[2] == 2 for p in g['keypoints'])), None)
        if reference is None:
            continue
        # Small fixed perturbations create a nonzero learning target. These annotation-box
        # fixtures test mechanics only and are discarded before frozen-proposal training.
        points = [[min(1279., max(0., p[0]+2.)), min(719., max(0., p[1]-1.)), .9]
                  if p[2] == 2 else [p[0], p[1], 0.] for p in reference['keypoints']]
        proposal = {'bbox_xyxy': reference['bbox_xyxy'], 'keypoints': points}
        geom = correction.crop_geometry(proposal['bbox_xyxy'])
        target, availability, _ = correction.correction_targets(reference['keypoints'], proposal, geom)
        if not availability.any():
            continue
        with Image.open(context['work']/row['image']) as original:
            array = correction.input_tensor(original.convert('RGB'), proposal, geom, row['id'])
        samples.append((array, target, availability, row['id'], proposal, geom))
        if len(samples) == 2:
            break
    need(len(samples) == 2, 'Two real training crops are required for the mechanics check')
    network = correction.build_model(context['backbone'])
    initial_identity = correction.model_identity(network)
    x, target, available = [torch.from_numpy(np.stack([item[i] for item in samples])).float() for i in range(3)]
    network.eval()
    with torch.inference_mode():
        initial = network(x)
    need(list(initial.shape) == [2, 2, 2] and torch.count_nonzero(initial).item() == 0,
         'The actual pretrained network must initially produce exactly zero offsets')
    need(all(correction.apply_corrections(item[4], delta.tolist(), item[5]) == item[4]
             for item, delta in zip(samples, initial)), 'Zero offsets changed an original point record')
    before = network.body.fc.weight.detach().clone()
    optimizer = torch.optim.AdamW(network.parameters(), lr=RECIPE['learning_rate'],
                                 weight_decay=RECIPE['weight_decay'], foreach=False)
    network.train()
    prediction = network(x)
    need(torch.count_nonzero(prediction).item() == 0, 'Training mode changed the exact zero initial output')
    loss = correction.masked_correction_loss(prediction, target, available)
    need(torch.isfinite(loss).item() and loss.item() > 0, 'Mechanics loss must be finite and nonzero')
    loss.backward()
    need(network.body.fc.weight.grad is not None and torch.isfinite(network.body.fc.weight.grad).all().item()
         and torch.count_nonzero(network.body.fc.weight.grad).item() > 0, 'Zero head did not receive a finite gradient')
    torch.nn.utils.clip_grad_norm_(network.parameters(), RECIPE['gradient_clip_norm'], error_if_nonfinite=True)
    optimizer.step()
    need(not torch.equal(before, network.body.fc.weight.detach()), 'Mechanics step did not update the correction head')
    probe = torch.ones(1, 2, 2, requires_grad=True)
    correction.masked_correction_loss(probe, torch.zeros_like(probe), torch.tensor([[1., 0.]])).backward()
    need(probe.grad[:, 1].abs().sum().item() == 0 and probe.grad[:, 0].abs().sum().item() > 0,
         'Unavailable endpoint gradient mask differs')
    receipt = {'status': 'passed', 'device': 'cpu', 'training_result': False,
        'description': 'Two real training photographs; disposable perturbed annotation-box mechanics fixtures. Training reinitializes the network.',
        'image_ids': [s[3] for s in samples], 'shape': list(initial.shape),
        'initial_offsets_exactly_zero': True, 'original_records_exactly_preserved': True,
        'finite_nonzero_loss': float(loss.item()), 'head_gradient_finite_nonzero': True,
        'parameters_changed': True, 'unavailable_endpoint_gradient_zero': True,
        'backbone_sha256': BACKBONE_SHA, 'initial_model': initial_identity,
        'torch': str(torch.__version__), 'torchvision': md.version('torchvision')}
    path = context['work']/'model-smoke.json'
    if path.exists():
        old = read(path)
        need(old['status'] == 'passed' and old['backbone_sha256'] == BACKBONE_SHA
             and old['torch'] == receipt['torch'] and old['torchvision'] == receipt['torchvision']
             and old['initial_model'] == initial_identity, 'Existing mechanics check has a different runtime or initialization')
    else:
        frozen(path, receipt)
    print('Pretrained CPU mechanics passed: exact zero offsets, exact coordinate identity, finite learning step and masked gradients.', flush=True)
    return receipt


def distribution(values):
    import numpy as np
    if not values:
        return {'count': 0}
    data = np.asarray(values, dtype='float64')
    return {'count': len(values), 'mean': float(data.mean()), 'maximum': float(data.max()),
            'quantiles': {str(q): float(np.quantile(data, q)) for q in (0., .25, .5, .75, .9, .95, .99, 1.)}}


def training_crops(context, predictions, save=True):
    records, image_counts, accounting = [], {}, Counter()
    canvas_norms, image_norms, signed = [], [], []
    for entry in sorted(context['stage']['records'], key=lambda row: row['id']):
        if entry['split'] != 'train':
            continue
        identity = entry['id']
        gold = parse_pose_labels((context['work']/entry['label']).read_text(), 1280, 720, identity)
        output = predictions[identity]['postclip']
        score = score_image(gold, output, 1280, 720, .5, .5)
        accounting.update(images=1, reference_fronts=len(gold), predictions=len(output),
                          matched=score['tp'], missed=score['fn'], unmatched=score['fp'])
        kept = 0
        for match in score['matches']:
            gi, pi = match['gold_index'], match['prediction_index']
            proposal = output[pi]
            geom = correction.crop_geometry(proposal['bbox_xyxy'])
            target, availability, states = correction.correction_targets(gold[gi]['keypoints'], proposal, geom)
            accounting.update(states)
            l, t, r, b = geom['xyxy']
            for slot, state in enumerate(states):
                if state != 'supervised':
                    continue
                initial, goal = proposal['keypoints'][slot], gold[gi]['keypoints'][slot]
                image_norms.append(math.hypot(goal[0]-initial[0], goal[1]-initial[1]))
                canvas_norms.append(math.hypot(*map(float, target[slot])))
                signed.append(target[slot].tolist())
                if not (l <= goal[0] < r and t <= goal[1] < b):
                    accounting['supervised_reference_outside_predicted_crop'] += 1
                if not (l <= initial[0] < r and t <= initial[1] < b):
                    accounting['supervised_initial_point_outside_predicted_crop'] += 1
            if not availability.any():
                accounting['matched_crops_without_usable_supervision'] += 1
                continue
            records.append({'identity': f'{identity}:p{pi}', 'image_id': identity, 'image_path': entry['image'],
                'reference_index': gi, 'prediction_index': pi, 'prediction': proposal, 'geometry': geom,
                'reference_keypoints': gold[gi]['keypoints'], 'availability': availability.tolist(),
                'target_canvas_delta': target.tolist(), 'endpoint_states': states})
            kept += 1
        image_counts[identity] = kept
    need(accounting['images'] == 142 and accounting['reference_fronts'] == 3999 and records,
         'The training photograph/front inventory differs')
    residuals = {'definition': 'Known reference minus available original point; Euclidean norm per eligible endpoint.',
        'image_pixels': distribution(image_norms), 'canvas_pixels': distribution(canvas_norms),
        'signed_canvas_x': distribution([d[0] for d in signed]),
        'signed_canvas_y': distribution([d[1] for d in signed])}
    result = {'records': records, 'counts': dict(accounting), 'usable_crops': len(records),
        'usable_crops_by_photo': image_counts, 'target_residual_distribution': residuals,
        'input_predictions_sha256': r50.sha(encoded(predictions)),
        'hint_jitter_canvas_pixels': 0., 'loss_eligibility': RECIPE['training_eligibility']}
    if save:
        frozen(context['work']/'training-crops.json', result)
    return result


def zero_output_gate(control):
    count = 0
    for row in control.values():
        for original in row['raw']:
            geom = correction.crop_geometry(original['bbox_xyxy'])
            need(correction.apply_corrections(original, [[0., 0.], [0., 0.]], geom) == original,
                 'Zero corrections changed a fixed detector record')
            count += 1
    return {'passed': True, 'raw_records_checked': count,
            'scope': 'Exact coordinate application for every raw proposal; actual zero network verified on two training photographs.'}


def trained_checkpoint(context, crops, device):
    import numpy as np
    from PIL import Image
    import torch
    shared.runtime_setup(device)
    target_device = torch.device('cpu' if device == 'cpu' else 'cuda:'+device)
    network = correction.build_model(context['backbone']).to(target_device)
    need(correction.model_identity(network) == read(context['work']/'model-smoke.json')['initial_model'],
         'Training initialization differs from the actual pretrained mechanics check')
    optimizer = torch.optim.AdamW(network.parameters(), lr=RECIPE['learning_rate'],
                                 weight_decay=RECIPE['weight_decay'], foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=RECIPE['minimum_learning_rate'])
    work = context['work']
    checkpoint, metadata = work/'last-training.pt', work/'last-training.json'
    preflight_sha, crops_sha = sha(work/'preflight.json'), sha(work/'training-crops.json')
    start, history = 0, []
    if checkpoint.exists() or metadata.exists():
        need(checkpoint.exists() and metadata.exists(), 'Incomplete checkpoint preserved; use a fresh work directory')
        receipt = read(metadata)
        checked(checkpoint, receipt['sha256'])
        need(receipt['preflight_sha256'] == preflight_sha and receipt['crops_sha256'] == crops_sha,
             'Resume input identity differs')
        need(receipt['runtime'] == read(work/'detector-runtime.json'), 'Resume runtime differs')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        need(saved['recipe'] == RECIPE, 'Resume recipe differs')
        network.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(target_device)
        start, history = saved['epoch'], saved['history']
        torch.set_rng_state(saved['torch_rng'])
        if device != 'cpu':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    data, images = crops['records'], {}
    for item in data:
        if item['image_id'] not in images:
            with Image.open(work/item['image_path']) as image:
                images[item['image_id']] = image.convert('RGB')
    updates = sum(h['updates'] for h in history)
    for epoch in range(start+1, EPOCHS+1):
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
        network.train()
        total, count, steps, started = 0., 0, 0, time.perf_counter()
        for begin in range(0, len(order), BATCH):
            batch = [data[i] for i in order[begin:begin+BATCH]]
            arrays = [correction.input_tensor(images[item['image_id']], item['prediction'], item['geometry'], item['identity']) for item in batch]
            targets = [item['target_canvas_delta'] for item in batch]
            availability = [item['availability'] for item in batch]
            x, target, available = [torch.from_numpy(np.asarray(values, dtype='float32')).to(target_device)
                                    for values in (arrays, targets, availability)]
            optimizer.zero_grad(set_to_none=True)
            loss = correction.masked_correction_loss(network(x), target, available)
            need(torch.isfinite(loss).item(), 'Nonfinite training loss; last completed epoch preserved')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), RECIPE['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()*len(batch)
            count += len(batch)
            steps += 1
        scheduler.step()
        updates += steps
        history.append({'epoch': epoch, 'samples': count, 'updates': steps, 'mean_loss': total/count,
            'next_learning_rate': scheduler.get_last_lr()[0], 'seconds': time.perf_counter()-started,
            'sample_order_sha256': r50.sha(encoded([data[i]['identity'] for i in order]))})
        saved = {'model': {k: v.detach().cpu() for k, v in network.state_dict().items()},
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'epoch': epoch, 'history': history, 'recipe': RECIPE, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device != 'cpu' else []}
        temporary = checkpoint.with_suffix('.part')
        torch.save(saved, temporary)
        temporary.replace(checkpoint)
        receipt = {'sha256': sha(checkpoint), 'preflight_sha256': preflight_sha, 'crops_sha256': crops_sha,
                   'completed_epoch': epoch, 'runtime': read(work/'detector-runtime.json')}
        temporary = metadata.with_suffix('.part')
        temporary.write_bytes(encoded(receipt))
        temporary.replace(metadata)
        print(f'Residual correction: epoch {epoch}/{EPOCHS}, loss {total/count:.6g}, updates {updates}', flush=True)
    need([h['epoch'] for h in history] == list(range(1, EPOCHS+1)), 'Fixed training history incomplete')
    final = work/MODEL_NAME
    if not final.exists():
        torch.save({'model': {k: v.detach().cpu() for k, v in network.state_dict().items()},
                    'recipe': RECIPE, 'backbone_sha256': BACKBONE_SHA}, final)
    saved = torch.load(final, map_location='cpu', weights_only=True)
    need(saved['recipe'] == RECIPE and saved['backbone_sha256'] == BACKBONE_SHA, 'Final checkpoint contract differs')
    need(set(saved['model']) == set(network.state_dict()) and
         all(torch.equal(value.detach().cpu(), saved['model'][key]) for key, value in network.state_dict().items()),
         'Final checkpoint differs from completed training')
    frozen(work/'training-log.json', history)
    frozen(work/'training-receipt.json', {'epochs': EPOCHS, 'samples': len(data)*EPOCHS, 'updates': updates,
        'expected_updates': math.ceil(len(data)/BATCH)*EPOCHS, 'model_sha256': sha(final),
        'model_bytes': final.stat().st_size, 'training_crops_sha256': crops_sha,
        'preflight_sha256': preflight_sha, 'selection': 'fixed final epoch',
        'training_log_sha256': sha(work/'training-log.json'), 'recipe': RECIPE})
    network.load_state_dict(saved['model'], strict=True)
    network.eval()
    return network


def correction_states(original, refined):
    return ['initial_unavailable_preserved' if not correction.point_available(a) else
            ('corrected_available' if correction.point_available(b) else 'invalid_corrected_point')
            for a, b in zip(original['keypoints'], refined['keypoints'])]


def refine_outputs(context, network, control):
    from PIL import Image
    import torch
    result, details = {}, []
    device = next(network.parameters()).device
    by_id = {r['id']: r for r in context['stage']['records']}
    with torch.inference_mode():
        for identity, record in sorted(control.items()):
            with Image.open(context['work']/by_id[identity]['image']) as source:
                image = source.convert('RGB')
            raw = []
            for index, original in enumerate(record['raw']):
                geom = correction.crop_geometry(original['bbox_xyxy'])
                array = correction.input_tensor(image, original, geom, f'{identity}:p{index}')
                predicted = network(torch.from_numpy(array[None]).to(device))[0].cpu().tolist()
                delta = [[float(v) if math.isfinite(v) else None for v in point] for point in predicted]
                refined = correction.apply_corrections(original, delta, geom)
                raw.append(refined)
                details.append({'image_id': identity, 'raw_index': index, 'geometry': geom,
                    'original_points': original['keypoints'], 'canvas_offsets': delta,
                    'output_points': refined['keypoints'], 'endpoint_states': correction_states(original, refined)})
            kept, audit = postclip_nms(raw)
            result[identity] = {'raw': raw, 'postclip': kept, 'filter': audit}
    return result, details


def replay_transforms(control, treatment, details):
    expected = {(image, index) for image, row in control.items() for index in range(len(row['raw']))}
    observed = set()
    for item in details:
        key = item['image_id'], item['raw_index']
        need(key in expected and key not in observed, 'Correction trace membership differs')
        observed.add(key)
        original = control[key[0]]['raw'][key[1]]
        need(item['geometry'] == correction.crop_geometry(original['bbox_xyxy']), 'Correction crop geometry differs')
        need(item['original_points'] == original['keypoints'], 'Correction origin differs')
        refined = correction.apply_corrections(original, item['canvas_offsets'], item['geometry'])
        need(refined == treatment[key[0]]['raw'][key[1]] and refined['keypoints'] == item['output_points'],
             'Saved offset application differs')
        need(item['endpoint_states'] == correction_states(original, refined), 'Correction availability states differ')
    need(observed == expected, 'Not every raw detector record received a correction trace')


def at_confidence(outputs, threshold=.25):
    """Filter raw scores and rerun the same postclip rule; no rematching shortcut."""
    result = {}
    for image, row in outputs.items():
        raw = [p for p in row['raw'] if p['confidence'] > threshold]
        kept, audit = postclip_nms(raw)
        result[image] = {'raw': raw, 'postclip': kept, 'filter': audit}
    return result


def source_payloads():
    return {name: Path(__file__).with_name(name).read_bytes() for name in SOURCE_MEMBERS}


def evidence_zip(work, names):
    entries = {name: (work/name).read_bytes() for name in names}
    entries.update(source_payloads())
    entries['inventory.json'] = encoded({name: {'bytes': len(data), 'sha256': r50.sha(data)} for name, data in entries.items()})
    archive = work/RESULT_NAME
    with zipfile.ZipFile(archive.with_suffix('.part'), 'w', zipfile.ZIP_DEFLATED) as stream:
        for name, data in sorted(entries.items()):
            stream.writestr(name, data)
    archive.with_suffix('.part').replace(archive)
    return archive


def ensure_upload_bundle(work, manifest):
    paths = [work/manifest['evidence_file'], work/'return-manifest.json', *[work/p['file'] for p in manifest['parts']]]
    expected = {path.name: sha(path) for path in paths}
    bundle = work/UPLOAD_NAME
    valid = False
    if bundle.exists():
        try:
            with zipfile.ZipFile(bundle) as stream:
                valid = (len(stream.namelist()) == len(expected) and set(stream.namelist()) == set(expected)
                         and stream.testzip() is None and all(r50.sha(stream.read(name)) == digest for name, digest in expected.items()))
        except (OSError, zipfile.BadZipFile):
            pass
    if not valid:
        with zipfile.ZipFile(bundle.with_suffix('.part'), 'w', zipfile.ZIP_STORED) as stream:
            for path in paths:
                stream.write(path, path.name)
        bundle.with_suffix('.part').replace(bundle)
    return bundle


def blocked_package(context, control):
    work = context['work']
    need(not read(work/'baseline-equivalence.json')['passed'], 'A blocked return requires a failed detector gate')
    summary = {'schema': SCHEMA, 'status': 'blocked_baseline', 'contract': context['contract'],
        'reports': {'control': r53.reports(context['refs'], control)}, 'baseline_passed': False,
        'training': False, 'correction_executed': False, 'interpretation_allowed': False,
        'automatic_promotion': False, 'development_only': True}
    frozen(work/'summary.json', summary)
    evidence = evidence_zip(work, ['summary.json', 'preflight.json', 'input-audit.json', 'staging-r58.json',
        'baseline-equivalence.json', 'detector-evaluation.json', 'detector-runtime.json', 'model-smoke.json'])
    frozen(work/'return-manifest.json', {'status': 'blocked_baseline', 'model_file': None, 'model_sha256': None,
        'model_bytes': 0, 'parts': [], 'evidence_file': evidence.name, 'evidence_sha256': sha(evidence)})
    return verify(context)


def package(context, arms, details):
    work = context['work']
    shared.frozen_detection_check(arms['control'], arms['treatment'])
    scored = {arm: r53.reports(context['refs'], output) for arm, output in arms.items()}
    common = {arm: at_confidence(output) for arm, output in arms.items()}
    summary = {'schema': SCHEMA, 'status': 'completed', 'contract': context['contract'],
        'reports': scored, 'reports_at_common_confidence_025': {arm: r53.reports(context['refs'], output) for arm, output in common.items()},
        'frozen_detection_records': True, 'training': True, 'fresh_inference': True,
        'automatic_promotion': False, 'development_only': True, 'correction_executed': True,
        'interpretation_allowed': True, 'training_receipt': read(work/'training-receipt.json'),
        'baseline_passed': read(work/'baseline-equivalence.json')['passed'],
        'original_corner_scores_reused': True, 'scores_recalibrated': False,
        'control_description': 'Fixed project research detector, not the untouched original supplied baomai model.'}
    content = {'summary.json': summary, 'predictions.json': arms, 'correction-traces.json': details,
        'cases.json': r53.case_inventory(context['refs'], context['maps'], arms, scored),
        'pair-accounting.json': shared.pair_accounting(context['refs'], arms),
        'pair-accounting-common025.json': shared.pair_accounting(context['refs'], common)}
    for name, value in content.items():
        frozen(work/name, value)
    names = list(content)+['preflight.json', 'input-audit.json', 'staging-r58.json', 'detector-evaluation.json',
        'detector-training.json', 'detector-runtime.json', 'detector-receipt.json', 'baseline-equivalence.json',
        'training-crops.json', 'training-log.json', 'training-receipt.json', 'model-smoke.json', 'zero-output-gate.json']
    evidence = evidence_zip(work, names)
    model = work/MODEL_NAME
    payload, parts = model.read_bytes(), []
    for index, begin in enumerate(range(0, len(payload), 16*1024*1024)):
        chunk = payload[begin:begin+16*1024*1024]
        member = f'{MODEL_NAME}.part{index:02d}'
        path = work/(member+'.zip')
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as stream:
            stream.writestr(member, chunk)
        parts.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path),
                      'member': member, 'member_bytes': len(chunk), 'member_sha256': r50.sha(chunk)})
    frozen(work/'return-manifest.json', {'status': 'completed', 'model_file': MODEL_NAME,
        'model_sha256': sha(model), 'model_bytes': len(payload), 'parts': parts,
        'evidence_file': evidence.name, 'evidence_sha256': sha(evidence),
        'instructions': 'Verify each listed wrapper and member in order; concatenate member bytes, then verify the full model SHA256.'})
    return verify(context)


def verify(context):
    work = context['work']
    manifest = read(work/'return-manifest.json')
    need(manifest['status'] in ('completed', 'blocked_baseline'), 'Unknown return status')
    checked(work/manifest['evidence_file'], manifest['evidence_sha256'])
    model_hash, model_bytes = hashlib.sha256(), 0
    for part in manifest['parts']:
        path = checked(work/part['file'], part['sha256'])
        need(path.stat().st_size == part['bytes'], 'Checkpoint wrapper length differs')
        with zipfile.ZipFile(path) as stream:
            need(stream.namelist() == [part['member']] and stream.testzip() is None, 'Checkpoint wrapper membership/CRC differs')
            data = stream.read(part['member'])
            need(len(data) == part['member_bytes'] and r50.sha(data) == part['member_sha256'], 'Checkpoint member differs')
            model_hash.update(data)
            model_bytes += len(data)
    if manifest['status'] == 'completed':
        need(model_bytes == manifest['model_bytes'] and model_hash.hexdigest() == manifest['model_sha256'], 'Reassembled model differs')
    else:
        need(not manifest['parts'] and manifest['model_sha256'] is None and manifest['model_bytes'] == 0, 'Blocked run contains a trained model')
    with zipfile.ZipFile(work/manifest['evidence_file']) as stream:
        inventory = json.loads(stream.read('inventory.json'))
        need(set(stream.namelist()) == set(inventory)|{'inventory.json'} and len(stream.namelist()) == len(set(stream.namelist()))
             and stream.testzip() is None, 'Evidence membership/CRC differs')
        for name, expected in inventory.items():
            data = stream.read(name)
            need(expected == {'bytes': len(data), 'sha256': r50.sha(data)}, 'Evidence member differs: '+name)
        for name, data in source_payloads().items():
            need(stream.read(name) == data, 'Saved implementation differs: '+name)
        load = lambda name: json.loads(stream.read(name))
        summary = load('summary.json')
        need(summary['contract'] == context['contract'] and summary['status'] == manifest['status'], 'Result input contract/status differs')
        need(load('preflight.json') == context['preflight'] and load('input-audit.json') == context['receipt'],
             'Saved preflight or source-archive audit differs')
        control = load('detector-evaluation.json')
        shared.replay_postclip(control)
        gate = r53.baseline_equivalence(context['refs'], context['prior'], control)
        need(gate == load('baseline-equivalence.json'), 'Saved baseline replay differs')
        if manifest['status'] == 'blocked_baseline':
            need(not gate['passed'] and not summary['training'] and not summary['correction_executed']
                 and not summary['interpretation_allowed'], 'A failed control must block experiment interpretation')
            need(summary['reports'] == {'control': r53.reports(context['refs'], control)}, 'Blocked score replay differs')
            ensure_upload_bundle(work, manifest)
            print('Verified failed-control diagnostic; no experiment training or correction.', flush=True)
            return summary
        need(gate['passed'] and summary['baseline_passed'], 'Experiment control did not pass')
        arms = load('predictions.json')
        need(arms['control'] == control, 'Control cache identity differs')
        shared.frozen_detection_check(control, arms['treatment'])
        for output in arms.values():
            shared.replay_postclip(output)
        replay_transforms(control, arms['treatment'], load('correction-traces.json'))
        need(load('zero-output-gate.json') == zero_output_gate(control), 'Zero-correction identity replay differs')
        scored = {arm: r53.reports(context['refs'], output) for arm, output in arms.items()}
        common = {arm: at_confidence(output) for arm, output in arms.items()}
        need(scored == summary['reports'] and summary['reports_at_common_confidence_025'] ==
             {arm: r53.reports(context['refs'], output) for arm, output in common.items()}, 'Score replay differs')
        need(load('cases.json') == r53.case_inventory(context['refs'], context['maps'], arms, scored), 'Case replay differs')
        need(load('pair-accounting.json') == shared.pair_accounting(context['refs'], arms)
             and load('pair-accounting-common025.json') == shared.pair_accounting(context['refs'], common), 'Offered-pair accounting differs')
        crops, receipt, history = load('training-crops.json'), load('training-receipt.json'), load('training-log.json')
        need(crops == training_crops(context, load('detector-training.json'), save=False), 'Training targets/masks/residual distribution differ')
        for field, filename in (('training_log_sha256', 'training-log.json'), ('training_crops_sha256', 'training-crops.json'),
                                ('preflight_sha256', 'preflight.json')):
            need(receipt[field] == r50.sha(stream.read(filename)), 'Training receipt input differs: '+field)
        need(receipt['recipe'] == RECIPE and receipt['selection'] == 'fixed final epoch', 'Training recipe/selection differs')
        count, steps = len(crops['records']), math.ceil(len(crops['records'])/BATCH)
        need(receipt['epochs'] == EPOCHS and receipt['samples'] == count*EPOCHS and receipt['updates'] == steps*EPOCHS
             and receipt['expected_updates'] == steps*EPOCHS, 'Training budget differs')
        need([h['epoch'] for h in history] == list(range(1, EPOCHS+1)), 'Epoch history differs')
        import torch
        for epoch in history:
            order = torch.randperm(count, generator=torch.Generator().manual_seed(SEED+epoch['epoch'])).tolist()
            expected = r50.sha(encoded([crops['records'][i]['identity'] for i in order]))
            need(epoch['samples'] == count and epoch['updates'] == steps and expected == epoch['sample_order_sha256'],
                 'Per-epoch exposure/order differs')
            rate = RECIPE['minimum_learning_rate']+(RECIPE['learning_rate']-RECIPE['minimum_learning_rate'])*(1+math.cos(math.pi*epoch['epoch']/EPOCHS))/2
            need(math.isfinite(epoch['mean_loss']) and math.isclose(epoch['next_learning_rate'], rate, rel_tol=1e-12, abs_tol=1e-15),
                 'Finite loss/cosine learning-rate receipt differs')
        need(receipt['model_sha256'] == manifest['model_sha256'] and receipt['model_bytes'] == manifest['model_bytes']
             and summary['training_receipt'] == receipt, 'Final model receipt differs')
        smoke = load('model-smoke.json')
        need(smoke['status'] == 'passed' and smoke['initial_offsets_exactly_zero'] and smoke['original_records_exactly_preserved']
             and smoke['unavailable_endpoint_gradient_zero'] and smoke['backbone_sha256'] == BACKBONE_SHA,
             'Pretrained model mechanics receipt differs')
    ensure_upload_bundle(work, manifest)
    print('Verified every saved correction, score, training mask/order, budget and exact model part.', flush=True)
    return summary


def run(context, device):
    if (context['work']/'return-manifest.json').exists():
        return verify(context)
    shared.runtime_setup(device)
    model_smoke(context)
    control, training, _ = shared.restored_detector(context, device)
    if training is None:
        return blocked_package(context, control)
    frozen(context['work']/'zero-output-gate.json', zero_output_gate(control))
    crops = training_crops(context, training)
    network = trained_checkpoint(context, crops, device)
    refined, details = refine_outputs(context, network, control)
    return package(context, {'control': control, 'treatment': refined}, details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'smoke', 'run', 'verify'))
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--script-sha256', required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--originals-dir', type=Path)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    context = source_and_inputs(args)
    if args.action == 'prepare':
        print('Verified 142 training photos, 20 development photos, research detector and pretrained backbone. No model inference.', flush=True)
    elif args.action == 'smoke':
        print(json.dumps(model_smoke(context), indent=2))
    elif args.action == 'run':
        run(context, args.device)
    else:
        verify(context)


if __name__ == '__main__':
    main()
