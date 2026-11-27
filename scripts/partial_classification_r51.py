#!/usr/bin/env python3
"""R51 classification-only partial-target ablation on the pinned native pose loss.

The experiment weights the *whole* soft-target BCE term on an anchor already
assigned to a target with fewer than two visible corners. It does not replace
soft targets by ones, change assignment, or promise to raise confidence. R30
already combined this classification treatment with box weighting; R51 isolates
classification while leaving native box, pose, visibility and RLE losses alone.
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
    'control_factor': 1, 'candidate_factor': 2,
    'partial_definition': 'post_augmentation_fewer_than_two_nonzero_keypoint_visibilities',
    'classification': 'entire_native_soft_target_bce_on_assigned_partial_anchors',
    'normalizer': 'original_unweighted_target_scores_sum',
    'native_class_weights_preserved': True,
    'native_assignment_and_soft_targets_preserved': True,
    'box': 'native_ciou_and_normalized_l1_unchanged',
    'keypoint_visibility_and_rle_losses_unchanged': True,
    'background_and_full_target_bce_unchanged': True,
    'sampling': 'ordinary_once_per_image_per_epoch',
    'native_branch_schedule_unchanged': True,
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
    'weighted_classification_terms',
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


def make_criterion(model, factor, observer=None):
    """Keep native E2ELoss/PoseLoss26; change only assigned partial-anchor BCE.

    The detection-loss body follows the hash-pinned native implementation.
    The native assigner sees detached, unweighted predictions, and bbox_loss
    receives its original target scores and original denominator. Factor 1
    avoids even an identity multiplication, enabling bitwise native checks.
    """
    verify_native_sources()
    import torch
    from types import MethodType
    from ultralytics.utils.loss import E2ELoss, PoseLoss26
    from ultralytics.utils.tal import make_anchors

    require(type(factor) is int and factor in (1, 2), 'Loss factor must be integer one or two')
    native = model.init_criterion()
    head = model.model[-1]
    require(type(native) is E2ELoss and type(native.one2many) is type(native.one2one) is PoseLoss26
            and type(head).__name__ == 'Pose26' and head.reg_max == 1 and head.nc == 1
            and list(head.kpt_shape) == [2, 3]
            and native.one2many.rle_loss is not None and native.one2one.rle_loss is not None,
            'Fixed Pose26 / E2ELoss / normalized-L1 / RLE architecture differs')

    def get_assigned_targets_and_loss(self, preds, batch, **assign_kwargs):
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (preds['boxes'].permute(0, 2, 1).contiguous(),
                                    preds['scores'].permute(0, 2, 1).contiguous())
        anchor_points, stride_tensor = make_anchors(preds['feats'], self.stride, 0.5)
        dtype, batch_size = pred_scores.dtype, pred_scores.shape[0]
        imgsz = torch.tensor(preds['feats'][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt, **assign_kwargs)
        target_scores_sum = target_scores.sum().clamp_(min=1)

        # Read actual post-augmentation labels in native packed-target order.
        # Padding, unassigned targets, and background anchors receive no weight.
        partial_gt = torch.zeros(gt_bboxes.shape[:2], dtype=torch.bool, device=self.device)
        batch_idx = batch['batch_idx'].view(-1).long().to(self.device)
        keypoints = batch['keypoints'].to(self.device)
        require(keypoints.shape == (len(batch_idx), 2, 3), 'Unexpected post-augmentation keypoint shape')
        require(not len(batch_idx) or bool(torch.all(batch_idx[1:] >= batch_idx[:-1])),
                'Native target packing requires grouped batch indices')
        require(not len(batch_idx) or bool((batch_idx >= 0).all() and (batch_idx < batch_size).all()),
                'Post-augmentation batch indices are outside the batch')
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within = torch.arange(len(batch_idx), device=self.device) - offsets[batch_idx]
        partial_gt[batch_idx, within] = (keypoints[..., 2] > 0).sum(-1) < 2
        partial_gt &= mask_gt.squeeze(-1).bool()
        partial_anchor = (fg_mask & partial_gt.gather(1, target_gt_idx)) if partial_gt.shape[1] else torch.zeros_like(fg_mask)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        native_bce = bce_loss
        if factor != 1:
            anchor_weight = 1 + (factor - 1) * partial_anchor.unsqueeze(-1).to(dtype)
            bce_loss = bce_loss * anchor_weight
        loss[1] = bce_loss.sum() / target_scores_sum
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
            target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        if observer is not None:
            assigned = torch.zeros_like(partial_gt, dtype=torch.long)
            if partial_gt.shape[1]:
                assigned.scatter_add_(1, target_gt_idx, fg_mask.long())
            full_gt = mask_gt.squeeze(-1).bool() & ~partial_gt
            statistics = {
                'branch': self._partial_classification_branch,
                'targets': int(mask_gt.sum()), 'partial_targets': int(partial_gt.sum()),
                'full_targets': int(full_gt.sum()),
                'assigned_partial_targets': int((partial_gt & assigned.gt(0)).sum()),
                'unassigned_partial_targets': int((partial_gt & assigned.eq(0)).sum()),
                'assigned_full_targets': int((full_gt & assigned.gt(0)).sum()),
                'unassigned_full_targets': int((full_gt & assigned.eq(0)).sum()),
                'partial_foreground_anchors': int(partial_anchor.sum()),
                'foreground_anchors': int(fg_mask.sum()), 'background_anchors': int((~fg_mask).sum()),
                'classification_terms': int(bce_loss.numel()),
                'positive_classification_terms': int((target_scores > 0).sum()),
                'partial_classification_terms': int(native_bce[partial_anchor].numel()),
                'partial_positive_classification_terms': int((target_scores[partial_anchor] > 0).sum()),
                'weighted_classification_terms': int(native_bce[partial_anchor].numel()) if factor != 1 else 0,
                'original_target_score_sum': float(target_scores.sum().detach()),
                'normalizer': float(target_scores_sum.detach()),
                'partial_target_score_sum': float(target_scores[partial_anchor].sum().detach()),
                'native_bce_numerator': float(native_bce.sum().detach()),
                'partial_native_bce_numerator': float(native_bce[partial_anchor].sum().detach()),
                'weighted_bce_numerator': float(bce_loss.sum().detach()),
                'loss_factor': factor, 'partial_classification_factor': factor,
                'background_classification_factor': 1, 'full_classification_factor': 1,
                'partial_box_factor': 1, 'native_class_weights_applied': self.class_weights is not None,
            }
            observer(statistics)
        return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss,
                dict(zip(self.loss_names, loss.detach())))

    for name in BRANCHES:
        branch = getattr(native, name)
        branch._partial_classification_branch = name
        branch.get_assigned_targets_and_loss = MethodType(get_assigned_targets_and_loss, branch)
    return native


class PartialClassificationLoss:
    """Install on the training model after EMA creation and audit each consumed batch."""

    def __init__(self, image_ids, factor, role=None, protocol_sha256=None):
        require(type(factor) is int and factor in (1, 2), 'Loss factor must be one or two')
        require(len(image_ids) == len(set(image_ids)) == 142, 'R51 requires the fixed 142-photo inventory')
        self.image_ids, self.factor = sorted(image_ids), factor
        self.role, self.protocol_sha256 = role, protocol_sha256
        self.native_sources = verify_native_sources()
        self.epochs, self.pending, self.criterion = [], None, None
        self.exposures = Counter({image_id: 0 for image_id in self.image_ids})

    def start_epoch(self, model, ema_model, epoch):
        require(epoch == len(self.epochs) + 1 and self.pending is None, 'Loss observer epoch sequence differs')
        require(model is not ema_model, 'Loss treatment must not be installed on EMA')
        if self.criterion is None:
            self.criterion = make_criterion(model, self.factor, self.record_loss)
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
                'loss_weighting_spec_sha256': digest_json(SPEC), 'loss_factor': self.factor,
                'epochs': self.epochs, 'images_seen': sum(self.exposures.values()), 'batches_seen': len(batches),
                'per_image': dict(sorted(self.exposures.items())),
                'assignment_totals': {branch: {name: sum(batch['branches'][branch][name] for batch in batches)
                                               for name in COUNT_FIELDS} for branch in BRANCHES},
                'criterion_contract': dict(CRITERION_CONTRACT),
                'criterion_scope': 'training_model_only_after_ema_creation'}


def validate_state(state, image_ids, factor, role=None, protocol_sha256=None, *, complete=False, trial=False):
    """Verify source/weight identity, native schedule and ordinary 142-photo exposure."""
    require(len(image_ids) == len(set(image_ids)) == 142, 'R51 image inventory differs')
    require(not (complete and trial), 'A trial cannot be a complete paired arm')
    require(state['schema_version'] == 1 and state['role'] == role and state['protocol_sha256'] == protocol_sha256
            and state['image_ids_sha256'] == digest_json(sorted(image_ids))
            and state['native_source_hashes'] == NATIVE_SOURCE_HASHES
            and state['loss_weighting_spec_sha256'] == digest_json(SPEC)
            and state['loss_factor'] == factor and state['criterion_contract'] == CRITERION_CONTRACT
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
                        and stats['weighted_classification_terms'] == (stats['partial_classification_terms'] if factor == 2 else 0),
                        'Inconsistent assignment or classification counts')
                numeric = ('original_target_score_sum', 'normalizer', 'partial_target_score_sum',
                           'native_bce_numerator', 'partial_native_bce_numerator', 'weighted_bce_numerator')
                require(all(isinstance(stats[name], (int, float)) and math.isfinite(stats[name]) and stats[name] >= 0
                            for name in numeric), 'Invalid target scores or BCE observation')
                score, partial = stats['original_target_score_sum'], stats['partial_target_score_sum']
                require(partial <= score + 1e-6 * max(score, 1) and stats['normalizer'] == max(score, 1)
                        and stats['loss_factor'] == stats['partial_classification_factor'] == factor
                        and stats['partial_box_factor'] == stats['background_classification_factor'] == stats['full_classification_factor'] == 1,
                        'Loss weights or original normalizer changed')
                expected_bce = stats['native_bce_numerator'] + (factor - 1) * stats['partial_native_bce_numerator']
                require(math.isclose(stats['weighted_bce_numerator'], expected_bce, rel_tol=2e-6, abs_tol=1e-6),
                        'Observed BCE weighting differs from assigned partial terms')
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
