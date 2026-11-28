#!/usr/bin/env python3
"""Classification-only Quality Focal Loss experiment on the native Pose26 model.

The existing assigner's soft targets remain unchanged. This applies the QFL
modulating factor to the native BCE terms; it is not a full GFL reproduction.
"""
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path

NATIVE_SOURCE_HASHES = {
    'utils/loss.py': 'f8db64a3093ef310b23cb83d587dd5c8aa3db20a27da94d95a62a734a6c055d2',
    'utils/tal.py': '7fc3127e4b094247ccbbcf8af9829972295f96fc53027313ef51b076d27274cd',
    'engine/trainer.py': '7016131a47b61a8b9b9ea51ef5a9e59b0fb896a328eed4901c89ac7a83169655',
    'data/augment.py': '802333ea359def55e88f61d6726e35c29d28a6491110840a5e912ec1fafd1de3',
}
SPEC = {
    'control_beta': 0, 'candidate_beta': 2,
    'classification': 'native_BCEWithLogits_times_abs_sigmoid_minus_native_soft_target_power_beta',
    'modulation_has_gradient': True,
    'soft_targets': 'unchanged_native_detached_pose_task_aligned_targets_not_replaced_by_raw_IoU',
    'normalizer': 'original_unweighted_target_scores_sum_clamped_at_one',
    'native_class_weights_preserved': True,
    'native_assignment_and_soft_targets_preserved': True,
    'box': 'native_ciou_and_normalized_l1_unchanged',
    'keypoint_visibility_and_rle_losses_unchanged': True,
    'classification_scope': 'all_foreground_and_background_terms_in_both_native_branches',
    'partial_definition_for_reporting_only': 'post_augmentation_fewer_than_two_nonzero_keypoint_visibilities',
    'sampling': 'ordinary_once_per_image_per_epoch',
    'native_branch_schedule_unchanged': True,
    'paper_adaptation': 'QFL_classification_modulation_only_not_full_GFL',
}
CRITERION_CONTRACT = {
    'model_head': 'Pose26', 'criterion': 'E2ELoss',
    'branches': ['PoseLoss26', 'PoseLoss26'], 'reg_max': 1,
    'keypoint_shape': [2, 3], 'classes': 1,
    'box_auxiliary': 'normalized_l1', 'rle_unchanged': True,
    'box_and_keypoint_losses_unchanged': True,
}
BRANCHES = ('one2many', 'one2one')
COUNT_FIELDS = (
    'targets', 'partial_targets', 'full_targets', 'assigned_partial_targets',
    'unassigned_partial_targets', 'assigned_full_targets', 'unassigned_full_targets',
    'partial_foreground_anchors', 'foreground_anchors', 'background_anchors',
    'classification_terms', 'positive_classification_terms',
    'partial_classification_terms', 'partial_positive_classification_terms',
    'modulated_classification_terms',
)
NUMERATOR_FIELDS = (
    'native_bce_numerator', 'focal_bce_numerator',
    'native_foreground_bce_numerator', 'native_background_bce_numerator',
    'focal_foreground_bce_numerator', 'focal_background_bce_numerator',
    'partial_native_bce_numerator', 'partial_focal_bce_numerator',
)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def verify_native_sources():
    require(importlib.metadata.version('ultralytics') == '8.4.182', 'Pinned Ultralytics 8.4.182 required')
    package = Path(importlib.util.find_spec('ultralytics').origin).parent
    for name, expected in NATIVE_SOURCE_HASHES.items():
        require(hashlib.sha256((package / name).read_bytes()).hexdigest() == expected,
                'Native training source differs: ' + name)
    return dict(NATIVE_SOURCE_HASHES)


def quality_focal_terms(logits, targets, beta, native_terms=None):
    """Return elementwise loss; beta zero bypasses modulation exactly."""
    import torch.nn.functional as functional
    require(type(beta) is int and beta in (0, 2), 'Focal beta must be integer zero or two')
    terms = functional.binary_cross_entropy_with_logits(logits, targets, reduction='none') \
        if native_terms is None else native_terms
    if beta == 0:
        return terms
    return terms * (logits.sigmoid() - targets).abs().pow(beta)


def _observation(branch, name, batch, assigned, loss, capture, beta):
    import torch
    fg_mask, target_gt_idx = assigned[:2]
    targets, native_bce, focal_bce, modulation = capture
    batch_size = len(batch['img'])
    batch_idx = batch['batch_idx'].view(-1).long().to(fg_mask.device)
    keypoints = batch['keypoints'].to(fg_mask.device)
    require(keypoints.shape == (len(batch_idx), 2, 3), 'Unexpected consumed keypoint shape')
    require(not len(batch_idx) or bool(torch.all(batch_idx[1:] >= batch_idx[:-1])),
            'Native packing requires grouped batch indices')
    require(not len(batch_idx) or bool((batch_idx >= 0).all() and (batch_idx < batch_size).all()),
            'Consumed target batch index is outside the batch')
    counts = torch.bincount(batch_idx, minlength=batch_size)
    width = int(counts.max()) if len(counts) else 0
    partial_gt = torch.zeros((batch_size, width), dtype=torch.bool, device=fg_mask.device)
    valid_gt = torch.arange(width, device=fg_mask.device)[None] < counts[:, None]
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    within = torch.arange(len(batch_idx), device=fg_mask.device) - offsets[batch_idx]
    partial_gt[batch_idx, within] = (keypoints[..., 2] > 0).sum(-1) < 2
    partial_anchor = fg_mask & partial_gt.gather(1, target_gt_idx) if width else torch.zeros_like(fg_mask)
    assigned_counts = torch.zeros_like(partial_gt, dtype=torch.long)
    if width:
        assigned_counts.scatter_add_(1, target_gt_idx, fg_mask.long())
    full_gt = valid_gt & ~partial_gt
    if branch.class_weights is not None:
        native_bce = native_bce * branch.class_weights
        focal_bce = focal_bce * branch.class_weights
    score_sum = targets.sum()
    statistics = {
        'branch': name, 'targets': int(valid_gt.sum()),
        'partial_targets': int(partial_gt.sum()), 'full_targets': int(full_gt.sum()),
        'assigned_partial_targets': int((partial_gt & assigned_counts.gt(0)).sum()),
        'unassigned_partial_targets': int((partial_gt & assigned_counts.eq(0)).sum()),
        'assigned_full_targets': int((full_gt & assigned_counts.gt(0)).sum()),
        'unassigned_full_targets': int((full_gt & assigned_counts.eq(0)).sum()),
        'partial_foreground_anchors': int(partial_anchor.sum()),
        'foreground_anchors': int(fg_mask.sum()), 'background_anchors': int((~fg_mask).sum()),
        'classification_terms': int(native_bce.numel()),
        'positive_classification_terms': int((targets > 0).sum()),
        'partial_classification_terms': int(native_bce[partial_anchor].numel()),
        'partial_positive_classification_terms': int((targets[partial_anchor] > 0).sum()),
        'modulated_classification_terms': int(native_bce.numel()) if beta else 0,
        'original_target_score_sum': float(score_sum),
        'normalizer': float(score_sum.clamp(min=1)),
        'partial_target_score_sum': float(targets[partial_anchor].sum()),
        'native_bce_numerator': float(native_bce.sum()),
        'focal_bce_numerator': float(focal_bce.sum()),
        'native_foreground_bce_numerator': float(native_bce[fg_mask].sum()),
        'native_background_bce_numerator': float(native_bce[~fg_mask].sum()),
        'focal_foreground_bce_numerator': float(focal_bce[fg_mask].sum()),
        'focal_background_bce_numerator': float(focal_bce[~fg_mask].sum()),
        'partial_native_bce_numerator': float(native_bce[partial_anchor].sum()),
        'partial_focal_bce_numerator': float(focal_bce[partial_anchor].sum()),
        'modulation_min': float(modulation.min()), 'modulation_max': float(modulation.max()),
        'modulation_mean': float(modulation.mean()),
        'classification_loss_gain_scaled': float(loss[1].detach()),
        'classification_gain': float(branch.hyp.cls), 'focal_beta': beta,
        'native_class_weights_applied': branch.class_weights is not None,
    }
    return statistics


def make_criterion(model, beta, observer=None):
    """Wrap the native classification BCE only; do not copy its assignment body."""
    verify_native_sources()
    import torch
    from types import MethodType
    from ultralytics.utils.loss import E2ELoss, PoseLoss26
    require(type(beta) is int and beta in (0, 2), 'Focal beta must be integer zero or two')
    native = model.init_criterion()
    head = model.model[-1]
    require(type(native) is E2ELoss and type(native.one2many) is type(native.one2one) is PoseLoss26
            and type(head).__name__ == 'Pose26' and head.reg_max == 1 and head.nc == 1
            and list(head.kpt_shape) == [2, 3]
            and native.one2many.rle_loss is not None and native.one2one.rle_loss is not None,
            'Fixed Pose26 / E2ELoss / normalized-L1 / RLE architecture differs')

    class ClassificationBCE(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
            self.capture = None

        def forward(self, logits, targets):
            original = self.original(logits, targets)
            focal = quality_focal_terms(logits, targets, beta, original)
            if observer is not None:
                require(self.capture is None, 'Unconsumed classification observation')
                modulation = (logits.detach().sigmoid() - targets.detach()).abs().pow(beta) \
                    if beta else torch.ones_like(targets)
                self.capture = (targets.detach().clone(), original.detach().clone(),
                                focal.detach().clone(), modulation)
            return focal

    for name in BRANCHES:
        branch = getattr(native, name)
        require(type(branch.bce) is torch.nn.BCEWithLogitsLoss
                and branch.bce.reduction == 'none' and branch.bce.weight is None
                and branch.bce.pos_weight is None, 'Native classification BCE contract differs')
        branch.bce = ClassificationBCE(branch.bce)
        if observer is None:
            continue
        original_method = branch.get_assigned_targets_and_loss

        def observed(self, preds, batch, _original=original_method, _name=name, **kwargs):
            result = _original(preds, batch, **kwargs)
            require(self.bce.capture is not None, 'Native loss did not consume classification BCE')
            capture, self.bce.capture = self.bce.capture, None
            observer(_observation(self, _name, batch, result[0], result[1], capture, beta))
            return result

        branch.get_assigned_targets_and_loss = MethodType(observed, branch)
    return native


class QualityFocalLoss:
    """Install on the training model after EMA creation and audit each consumed batch."""

    def __init__(self, image_ids, beta, role=None, protocol_sha256=None):
        require(type(beta) is int and beta in (0, 2), 'Loss beta must be zero or two')
        require(len(image_ids) == len(set(image_ids)) == 142, 'R61 requires the fixed 142-photo inventory')
        self.image_ids, self.beta = sorted(image_ids), beta
        self.role, self.protocol_sha256 = role, protocol_sha256
        self.native_sources = verify_native_sources()
        self.epochs, self.pending, self.criterion = [], None, None
        self.exposures = Counter({image_id: 0 for image_id in self.image_ids})

    def start_epoch(self, model, ema_model, epoch):
        require(epoch == len(self.epochs) + 1 and self.pending is None, 'Loss observer epoch sequence differs')
        require(model is not ema_model, 'Loss treatment must not be installed on EMA')
        if self.criterion is None:
            self.criterion = make_criterion(model, self.beta, self.record_loss)
            model.criterion = self.criterion
        require(model.criterion is self.criterion and getattr(ema_model, 'criterion', None) is not self.criterion,
                'Training-only criterion routing differs')
        self.epochs.append({'epoch': epoch, 'observed_image_ids': [], 'batches': []})

    def prepare_batch(self, batch, epoch):
        require(self.pending is None and self.epochs[-1]['epoch'] == epoch, 'Loss observer batch order differs')
        self.pending = {'image_ids': [Path(name).stem for name in batch['im_file']],
                        'objects': len(batch['cls']), 'statistics': {}}

    def record_loss(self, statistics):
        branch = statistics['branch']
        require(self.pending is not None and branch in BRANCHES and branch not in self.pending['statistics'],
                'Unexpected repeated loss branch call')
        require(statistics['targets'] == self.pending['objects'], 'Loss target count differs from consumed batch')
        self.pending['statistics'][branch] = statistics

    def finish_batch(self, image_ids, epoch):
        require(self.pending is not None and set(self.pending['statistics']) == set(BRANCHES)
                and self.pending['image_ids'] == image_ids and self.epochs[-1]['epoch'] == epoch,
                'Loss instrumentation did not observe the completed batch')
        require(all(image_id in self.exposures for image_id in image_ids), 'Non-training image entered loss treatment')
        record, branches = self.epochs[-1], self.pending['statistics']
        require(all(branches['one2many'][key] == branches['one2one'][key]
                    for key in ('targets', 'partial_targets', 'full_targets')), 'Native branches saw different targets')
        record['observed_image_ids'].extend(image_ids)
        record['batches'].append({'batch': len(record['batches']) + 1, 'image_ids': list(image_ids),
                                  'branches': branches,
                                  'native_branch_schedule': {'updates': self.criterion.updates,
                                                           'one2many': self.criterion.o2m, 'one2one': self.criterion.o2o}})
        self.exposures.update(image_ids)
        self.pending = None

    def finish_epoch(self, epoch):
        require(self.pending is None and self.epochs[-1]['epoch'] == epoch, 'Incomplete loss epoch')
        observed = self.epochs[-1]['observed_image_ids']
        require(len(observed) == len(set(observed)) == 142 and set(observed) == set(self.exposures),
                'Ordinary per-epoch image exposure changed')

    def state(self):
        require(self.pending is None, 'Cannot record an unconsumed loss batch')
        batches = [batch for epoch in self.epochs for batch in epoch['batches']]
        return {'schema_version': 1, 'role': self.role, 'protocol_sha256': self.protocol_sha256,
                'image_ids_sha256': digest_json(self.image_ids), 'native_source_hashes': self.native_sources,
                'loss_weighting_spec_sha256': digest_json(SPEC), 'focal_beta': self.beta,
                'epochs': self.epochs, 'images_seen': sum(self.exposures.values()), 'batches_seen': len(batches),
                'per_image': dict(sorted(self.exposures.items())),
                'assignment_totals': {branch: {name: sum(batch['branches'][branch][name] for batch in batches)
                                               for name in COUNT_FIELDS} for branch in BRANCHES},
                'criterion_contract': dict(CRITERION_CONTRACT),
                'criterion_scope': 'training_model_only_after_ema_creation'}


def validate_state(state, image_ids, beta, role=None, protocol_sha256=None, *, complete=False, trial=False):
    """Verify the observed native schedule, modulation and 142-photo exposure."""
    require(type(beta) is int and beta in (0, 2), 'Focal beta must be integer zero or two')
    require(len(image_ids) == len(set(image_ids)) == 142, 'R61 image inventory differs')
    require(not (complete and trial), 'A trial cannot be a complete paired arm')
    require(state['schema_version'] == 1 and state['role'] == role and state['protocol_sha256'] == protocol_sha256
            and state['image_ids_sha256'] == digest_json(sorted(image_ids))
            and state['native_source_hashes'] == NATIVE_SOURCE_HASHES
            and state['loss_weighting_spec_sha256'] == digest_json(SPEC)
            and state['focal_beta'] == beta and state['criterion_contract'] == CRITERION_CONTRACT
            and state['criterion_scope'] == 'training_model_only_after_ema_creation', 'Loss receipt identity differs')
    totals = {branch: Counter({name: 0 for name in COUNT_FIELDS}) for branch in BRANCHES}
    exposures = Counter({image_id: 0 for image_id in sorted(image_ids)})
    require(1 <= len(state['epochs']) <= 30, 'Missing or excessive loss exposure')
    batch_count = 0
    for epoch, record in enumerate(state['epochs'], 1):
        require(record['epoch'] == epoch and 1 <= len(record['batches']) <= 36, 'Loss epoch sequence differs')
        observed = []
        for index, batch in enumerate(record['batches'], 1):
            ids = batch['image_ids']
            require(batch['batch'] == index and len(ids) == (2 if index == 36 else 4)
                    and set(ids) <= set(exposures), 'Loss batch identity or shape differs')
            require(set(batch['branches']) == set(BRANCHES), 'Native branch inventory differs')
            for branch, stats in batch['branches'].items():
                require(stats['branch'] == branch and all(type(stats[name]) is int and stats[name] >= 0 for name in COUNT_FIELDS),
                        'Invalid loss observation count')
                require(stats['targets'] == stats['partial_targets'] + stats['full_targets']
                        and stats['partial_targets'] == stats['assigned_partial_targets'] + stats['unassigned_partial_targets']
                        and stats['full_targets'] == stats['assigned_full_targets'] + stats['unassigned_full_targets']
                        and stats['assigned_partial_targets'] <= stats['partial_foreground_anchors'] <= stats['foreground_anchors']
                        and stats['assigned_full_targets'] <= stats['foreground_anchors'] - stats['partial_foreground_anchors']
                        and stats['classification_terms'] == stats['foreground_anchors'] + stats['background_anchors']
                        and stats['positive_classification_terms'] <= stats['foreground_anchors']
                        and stats['partial_classification_terms'] == stats['partial_foreground_anchors']
                        and stats['partial_positive_classification_terms'] <= stats['partial_classification_terms']
                        and stats['modulated_classification_terms'] == (stats['classification_terms'] if beta else 0),
                        'Inconsistent assignment or classification counts')
                numeric = (*NUMERATOR_FIELDS, 'original_target_score_sum', 'normalizer', 'partial_target_score_sum',
                           'modulation_min', 'modulation_max', 'modulation_mean',
                           'classification_loss_gain_scaled', 'classification_gain')
                require(all(isinstance(stats[name], (int, float)) and math.isfinite(stats[name]) and stats[name] >= 0
                            for name in numeric), 'Invalid QFL observation')
                score, partial = stats['original_target_score_sum'], stats['partial_target_score_sum']
                require(partial <= score + 1e-6 * max(score, 1) and stats['normalizer'] == max(score, 1)
                        and stats['focal_beta'] == beta
                        and 0 <= stats['modulation_min'] <= stats['modulation_mean'] <= stats['modulation_max'] <= 1 + 1e-6,
                        'Original normalizer, exponent or modulation bounds changed')
                for prefix in ('native', 'focal'):
                    require(math.isclose(stats[prefix + '_bce_numerator'],
                                         stats[prefix + '_foreground_bce_numerator'] + stats[prefix + '_background_bce_numerator'],
                                         rel_tol=3e-6, abs_tol=1e-5), 'Foreground/background loss partition differs')
                require(math.isclose(stats['classification_loss_gain_scaled'],
                                     stats['focal_bce_numerator'] / stats['normalizer'] * stats['classification_gain'],
                                     rel_tol=3e-6, abs_tol=1e-6), 'Classification normalization changed')
                require(stats['focal_bce_numerator'] <= stats['native_bce_numerator'] * (1 + 3e-6) + 1e-6,
                        'QFL modulation increased an element beyond native BCE')
                if beta == 0:
                    require(stats['modulation_min'] == stats['modulation_mean'] == stats['modulation_max'] == 1
                            and stats['focal_bce_numerator'] == stats['native_bce_numerator'],
                            'Control differs from unmodulated native BCE')
                totals[branch].update({name: stats[name] for name in COUNT_FIELDS})
            require(all(batch['branches']['one2many'][key] == batch['branches']['one2one'][key]
                        for key in ('targets', 'partial_targets', 'full_targets')), 'Branches saw different target inventories')
            o2m = .8 if epoch == 1 else max(1 - (epoch - 1) / 29, 0) * (.8 - .1) + .1
            require(batch['native_branch_schedule'] == {'updates': epoch - 1, 'one2many': o2m, 'one2one': max(1 - o2m, 0)},
                    'Native E2E branch schedule changed')
            observed.extend(ids)
            batch_count += 1
        require(observed == record['observed_image_ids'] and len(observed) == len(set(observed)),
                'Ordinary image exposure repeats or differs')
        if complete or epoch < len(state['epochs']):
            require(len(observed) == 142 and set(observed) == set(exposures), 'Incomplete ordinary image epoch')
        exposures.update(observed)
    require(state['images_seen'] == sum(exposures.values()) and state['batches_seen'] == batch_count
            and state['per_image'] == dict(sorted(exposures.items()))
            and state['assignment_totals'] == {branch: dict(value) for branch, value in totals.items()}, 'Loss receipt totals differ')
    if complete:
        require(len(state['epochs']) == 30 and batch_count == 1080 and state['images_seen'] == 4260
                and set(exposures.values()) == {30}, 'Completed loss exposure differs from fixed budget')
    if trial:
        require(len(state['epochs']) == 1 and batch_count == 4 and state['images_seen'] == 16,
                'Resource trial must observe exactly four ordinary batches')
    return {'images_seen': state['images_seen'], 'batches_seen': batch_count, 'epochs_seen': len(state['epochs']),
            'assignment_totals': {branch: dict(value) for branch, value in totals.items()}}
