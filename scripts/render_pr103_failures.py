#!/usr/bin/env python3
"""Render returned PR103 R26 candidate failures against the unchanged R26 review.

Reuse the established review layout and exact scorer. No inference, annotation
amendment, threshold search, or model selection is performed by this renderer.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import render_front_review_r26_errors as layout
from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou, score_image


MODEL_SHA = "a2e71037a9995ed864ec21ab5c03a719e200301766788b88765c745638e7ad9a"
PROTOCOL_SHA = "232f81cdf6d367fc9ece902e01b4ab5a03da5ae0898f5f75abae25ef5001372f"
SCORER_SHA = "04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018"
EVIDENCE = Path("results/evaluation/pr103-r26-20261123")
ANNOTATION_REVIEW = Path("results/review/front-review-20261123-r27/row-review.json")
LATER_AMENDED_ROWS = {
    'new_000007': [24, 33, 37, 41, 42], 'new_000008': [33, 34, 35],
    'new_000009': [48, 49, 50, 51, 52], 'new_000056': [12],
}
SIDES = layout.SIDES
WIDTH, HEIGHT, MARGIN = layout.WIDTH, layout.HEIGHT, layout.MARGIN
INK, MUTED, RULE, PALE = layout.INK, layout.MUTED, layout.RULE, layout.PALE
GREEN, ORANGE = layout.GREEN, layout.ORANGE
read, sha, require = layout.read, layout.sha, layout.require
text, paragraph = layout.text, layout.paragraph


def corner_failure(item):
    return any(point['reference_visibility'] == 2 and
               (not point['returned_at_fixed_threshold'] or point['error_px'] > 5)
               for point in item['corners'].values())


def unassigned_corners(reference, prediction, width, height):
    result = {}
    for index, side in enumerate(SIDES):
        ref = reference['keypoints'][index] if reference else None
        pred = prediction['keypoints'][index] if prediction else None
        locatable = ref is not None and ref[2] == 2
        result[side] = {
            'reference_visibility': ref[2] if ref is not None else None,
            'reference_xy_px': ref[:2] if locatable else None,
            'prediction_xy_px': pred[:2] if pred is not None else None,
            'prediction_confidence': pred[2] if pred is not None else None,
            'returned_at_fixed_threshold': bool(pred is not None and layout.available(pred, width, height)),
            'error_px': None,
            'status': 'no_assigned_pair',
        }
    return result


def gather(root, predictions_path, comparison_path):
    protocol_path = root / 'experiments/front-review-2026-11-23-r26.json'
    require(sha(protocol_path) == PROTOCOL_SHA, 'Frozen R26 protocol differs')
    require(sha(root / 'scripts/pose_scoring.py') == SCORER_SHA, 'Frozen scorer differs')
    protocol = read(protocol_path)
    descriptor = protocol['evaluation']['current_review_reference']
    manifest_path = root / descriptor['path']
    require(sha(manifest_path) == descriptor['sha256'], 'R26 reference manifest differs')
    manifest = read(manifest_path)
    require(manifest['review_id'] == 'fronts-2026-11-23-r26', 'This report must use the unamended R26 review')
    comparison = read(comparison_path)
    measured = comparison['references']['current_r26_exploratory']
    require(measured['manifest_sha256'] == sha(manifest_path), 'Comparison used a different reference')
    require(comparison['models']['candidate']['predictions_sha256'] == sha(predictions_path),
            'Comparison used different candidate predictions')
    lines = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
    headers = [line for line in lines if line.get('record_type') == 'run']
    require(len(headers) == 1 and headers[0]['model_sha256'] == MODEL_SHA, 'Wrong returned model')
    header = headers[0]
    rows = [line for line in lines if line.get('record_type') == 'prediction']
    require(all(line['threshold'] == .13 for line in rows), 'Prediction threshold differs')
    predictions = {line['image_id']: line for line in rows}
    files = sorted((item for item in manifest['files'] if item['split'] == 'val'), key=lambda item: item['image_id'])
    require(len(predictions) == len(rows) == 20, 'Incomplete candidate prediction records')
    require(set(predictions) == {item['image_id'] for item in files}, 'Development image sets differ')
    row_maps_path = root / manifest['row_maps']['path']
    require(sha(row_maps_path) == manifest['row_maps']['sha256'], 'R26 row-map identity differs')
    row_maps = read(row_maps_path)
    images, sources, matched, missed, unmatched = {}, [], [], [], []
    for spec in files:
        image_id = spec['image_id']
        record = predictions[image_id]
        image_path = root / 'data/images/val' / (image_id + '.png')
        label_path = manifest_path.parent / spec['label_path']
        require(sha(image_path) == spec['source_image_sha256'] == record['image_sha256'], 'Original image differs')
        require(sha(label_path) == spec['label_sha256'], 'R26 label bytes differ')
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        require((width, height) == (record['width'], record['height']), 'Image dimensions differ')
        raw_text = label_path.read_text()
        require(all(line.strip() for line in raw_text.splitlines()), 'Blank lines make row IDs ambiguous')
        gold = parse_pose_labels(raw_text, width, height, str(label_path))
        require(len(gold) == spec['objects'], 'Reference object count differs')
        outputs = record['predictions']
        require(all(item['confidence'] >= .13 for item in outputs), 'Below-threshold output in retained predictions')
        lineage = {item['current_row']: item for item in row_maps.get(image_id, {}).get('rows', [])}
        items, score = layout.matched_records(image_id, gold, outputs, width, height, lineage)
        expected = measured['matching_traces']['candidate'][image_id]
        require(all(score[key] == value for key, value in expected.items()), 'Inventory differs from frozen comparison')
        matched.extend(items)
        assigned = {pair['prediction_index']: pair['gold_index'] + 1 for pair in score['matches']}
        for index in score['unmatched_gold_indices']:
            target = gold[index]
            overlaps = [(p, _box_iou(target['bbox_xyxy'], output['bbox_xyxy']))
                        for p, output in enumerate(outputs) if output['class_id'] == target['class_id']]
            best = max(overlaps, key=lambda pair: (pair[1], -pair[0])) if overlaps else None
            best = best if best and best[1] > 0 else None
            best_index, best_iou = best if best else (None, None)
            output = outputs[best_index] if best else None
            assigned_row = assigned.get(best_index)
            reason = ('no_retained_overlap' if best is None else
                      'below_matching_iou' if best_iou < .5 else 'assigned_elsewhere')
            require(reason != 'assigned_elsewhere' or assigned_row is not None,
                    'Unexplained above-threshold unassigned overlap')
            missed.append({
                'identity': f'{image_id}:{index + 1}', 'image_id': image_id, 'current_row': index + 1,
                'matched': False, 'selection_category': 'miss', 'prediction_index_0based': best_index,
                'reference_box_xyxy_px': target['bbox_xyxy'],
                'prediction_box_xyxy_px': output['bbox_xyxy'] if output else None,
                'box_iou': None, 'box_confidence': output['confidence'] if output else None,
                'best_retained_overlap_iou': best_iou, 'best_overlap_assigned_to_row': assigned_row,
                'matching_explanation': reason, 'corners': unassigned_corners(target, None, width, height),
                'row_lineage': lineage.get(index + 1), 'selected_for_pdf': True,
            })
        for index in score['unmatched_prediction_indices']:
            output = outputs[index]
            overlaps = [(g, _box_iou(target['bbox_xyxy'], output['bbox_xyxy']))
                        for g, target in enumerate(gold) if target['class_id'] == output['class_id']]
            best = max(overlaps, key=lambda pair: (pair[1], -pair[0])) if overlaps else None
            best = best if best and best[1] > 0 else None
            target = gold[best[0]] if best else None
            unmatched.append({
                'identity': f'{image_id}:output:{index + 1}', 'image_id': image_id, 'current_row': None,
                'matched': False, 'selection_category': 'unmatched_output', 'prediction_index_0based': index,
                'reference_box_xyxy_px': target['bbox_xyxy'] if target else None,
                'prediction_box_xyxy_px': output['bbox_xyxy'], 'box_iou': None,
                'box_confidence': output['confidence'], 'best_reference_overlap_iou': best[1] if best else None,
                'best_reference_row': best[0] + 1 if best else None,
                'scorer_classification': 'duplicate' if index in score['duplicate_prediction_indices'] else 'other_unmatched',
                'corners': unassigned_corners(None, output, width, height), 'selected_for_pdf': False,
            })
        images[image_id] = image
        sources.append({'image_id': image_id, 'image_path': str(image_path.relative_to(root)),
                        'image_sha256': sha(image_path), 'label_path': str(label_path.relative_to(root)),
                        'label_sha256': sha(label_path), 'width': width, 'height': height,
                        'reference_objects': len(gold), 'predictions': len(outputs),
                        'matched': score['tp'], 'misses': score['fn'], 'unmatched_outputs': score['fp']})
    corners = [item for item in matched if corner_failure(item)]
    corners.sort(key=lambda item: (-sum(p['reference_visibility'] == 2 and not p['returned_at_fixed_threshold']
                                      for p in item['corners'].values()),
                                  -(item['maximum_measured_corner_error_px'] or 0), item['image_id'], item['current_row']))
    used = {item['identity'] for item in corners}
    box_pool = sorted((item for item in matched if item['box_iou'] < .70 and item['identity'] not in used),
                      key=lambda item: (item['box_iou'], item['image_id'], item['current_row']))
    boxes = box_pool[:8]
    output_pool = sorted(unmatched, key=lambda item: (-item['box_confidence'], item['image_id'], item['prediction_index_0based']))
    output_cases = output_pool[:6]
    for category, items in [('miss', missed), ('matched_corner_failure', corners), ('matched_box_overlap', boxes),
                            ('unmatched_output', output_cases)]:
        for rank, item in enumerate(items, 1):
            item.update(selected_for_pdf=True, selection_category=category, selection_rank=rank)
    selected = missed + corners + boxes + output_cases
    require(len({item['identity'] for item in selected}) == len(selected), 'Repeated selected case')
    counts = measured['models']['candidate']['detection']
    require((len(matched), len(missed), len(unmatched)) == (counts['tp'], counts['fn'], counts['fp']),
            'Inventory totals differ from measured comparison')
    summary = {
        'reference_objects': 661, 'images': 20, 'matched_targets': len(matched), 'misses': len(missed),
        'unmatched_outputs': len(unmatched), 'duplicate_outputs': counts['duplicate_fp'],
        'selected_misses': len(missed), 'matched_corner_failure_targets': len(corners),
        'matched_returned_corner_errors_over_5px': sum(p['error_px'] is not None and p['error_px'] > 5
                                                    for item in matched for p in item['corners'].values()),
        'matched_locatable_corners_not_returned': sum(p['reference_visibility'] == 2 and not p['returned_at_fixed_threshold']
                                                    for item in matched for p in item['corners'].values()),
        'additional_box_candidates_below_0_70': len(box_pool), 'selected_box_cases': len(boxes),
        'selected_unmatched_outputs': len(output_cases), 'selected_total': len(selected),
    }
    inventory = {
        'schema_version': 1, 'kind': 'pr103_returned_r26_candidate_full_failures_and_examples',
        'review_id': manifest['review_id'], 'review_manifest': {'path': str(manifest_path.relative_to(root)), 'sha256': sha(manifest_path)},
        'comparison': {'path': layout.describe_path(comparison_path, root), 'sha256': sha(comparison_path)},
        'prediction_source': {'path': layout.describe_path(predictions_path, root), 'sha256': sha(predictions_path),
                              'model_sha256': MODEL_SHA, 'source_revision': header.get('source', {}).get('revision'),
                              'inference': header['inference'], 'runtime': header['runtime']},
        'thresholds': {'box_confidence': .13, 'endpoint_confidence': .5, 'matching_iou': .5,
                       'corner_error_px': 5, 'box_example_iou_below': .7},
        'selection': {'misses': 'All scored unmatched reference targets, by image and row.',
                      'corners': 'All matched targets with any locatable corner not returned or returned with error >5 original pixels.',
                      'boxes': 'Eight worst remaining matched box overlaps below 0.70; no repeated target.',
                      'unmatched_outputs': 'Up to six highest-confidence unmatched outputs; duplicate status is the scorer classification.',
                      'unknown_reference_corners_inferred': False, 'unassigned_corner_distances_computed': False,
                      'raw_suppression_cause_inferred': False},
        'summary': summary, 'reference_counts': measured['counts'],
        'current_review_metrics': measured['models']['candidate'],
        'frozen_primary_metrics': {role: comparison['references']['primary']['models'][role]
                                   for role in ('candidate', 'pr35', 'reference', 'r25', 'r25control')},
        'verification': {'all_reference_and_image_hashes_verified': True, 'exact_comparison_matching_traces_verified': True,
                         'all_misses_selected': True, 'all_matched_corner_failures_selected': True,
                         'scorer_sha256': SCORER_SHA, 'renderer_sha256': sha(__file__),
                         'layout_helper_sha256': sha(Path(layout.__file__))},
        'source_images': sources, 'selected_identities': [item['identity'] for item in selected],
        'missed_targets': missed, 'matched_targets': matched, 'unmatched_outputs': unmatched,
    }
    return inventory, images, selected


def add_annotation_crossreferences(root, inventory, selected, ledger_path):
    ledger = read(ledger_path)
    entries = [item for item in ledger['rows'] if item['split'] == 'val' and item['action'] == 'replace']
    expected = {f'{image_id}:{row}' for image_id, rows in LATER_AMENDED_ROWS.items() for row in rows}
    identities = {f"{item['image_id']}:{item.get('base_row', item.get('row'))}" for item in entries}
    require(identities == expected, 'Final R27 development amendment identities differ from the accepted set')
    sources = {item['image_id']: item for item in inventory['source_images']}
    for item in entries:
        row = item.get('base_row', item.get('row'))
        original = root / sources[item['image_id']]['label_path']
        require(original.read_text().splitlines()[row - 1] == item['before_line'], 'R27 ledger does not identify the scored R26 row')
    flagged, context_flagged = [], []
    for item in selected:
        identity = item['identity']
        context_only = False
        if item['current_row'] is None and item.get('best_reference_row') is not None:
            identity = f"{item['image_id']}:{item['best_reference_row']}"
            context_only = True
        if identity not in identities:
            continue
        item['later_annotation_amendment'] = {
            'review_id': 'fronts-2026-11-23-r27', 'r26_identity': identity, 'context_only': context_only,
            'notice': ('Unassigned reference context was subsequently amended in R27 - see annotation PDF.'
                       if context_only else 'Scored against unchanged R26 labels; this annotation was subsequently amended in R27 - see annotation PDF.'),
        }
        (context_flagged if context_only else flagged).append(item['identity'])
    inventory['later_annotation_review'] = {
        'path': layout.describe_path(ledger_path, root), 'sha256': sha(ledger_path),
        'changed_r26_development_identities': sorted(identities),
        'selected_target_cases_flagged': flagged, 'selected_context_cases_flagged': context_flagged,
        'original_r26_scores_selection_and_row_identities_unchanged': True,
    }


def footer(pdf, page, pages):
    pdf.setStrokeColor(RULE); pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH - MARGIN, 31)
    text(pdf, 'PR103 R26 CANDIDATE  /  UNAMENDED R26 661-FRONT REVIEW', MARGIN, 18, 7, MUTED)
    pdf.setFont('Review', 7); pdf.setFillColor(MUTED)
    pdf.drawRightString(WIDTH - MARGIN, 18, f'{page} / {pages}')


def legend(pdf, x, y):
    layout.symbol(pdf, x + 4, y + 3)
    text(pdf, 'R26 reference: ring + solid box', x + 15, y, 8, GREEN)
    layout.symbol(pdf, x + 257, y + 3, True)
    text(pdf, 'Returned candidate: cross + dashed box', x + 268, y, 8, ORANGE)


def intro(pdf, inventory, pages):
    summary = inventory['summary']
    text(pdf, 'RETURNED MODEL DIAGNOSTICS  /  PR103', MARGIN, HEIGHT - 49, 9, MUTED, True)
    text(pdf, 'R26: misses and errors', MARGIN, HEIGHT - 87, 27, INK, True)
    y = paragraph(pdf, 'The returned R26 <b>candidate</b>, scored against the unchanged R26 review of 20 development images and 661 fronts. Row numbers refer to that review; later R27 amendments are not used here.',
                  MARGIN, HEIGHT - 109, WIDTH - 2 * MARGIN, 10.5, 16)
    y -= 78
    card_width = (WIDTH - 2 * MARGIN - 20) / 3
    for index, (number, label) in enumerate([(summary['misses'], 'all scored misses'),
                                            (summary['matched_corner_failure_targets'], 'all corner-failure targets'),
                                            (summary['selected_box_cases'] + summary['selected_unmatched_outputs'], 'box / output examples')]):
        x = MARGIN + index * (card_width + 10)
        pdf.setFillColor(PALE); pdf.roundRect(x, y, card_width, 60, 4, fill=1, stroke=0)
        text(pdf, str(number), x + 12, y + 29, 22, INK, True)
        text(pdf, label, x + 12, y + 13, 8, MUTED)
    y -= 27
    text(pdf, 'Keep the reference versions separate', MARGIN, y, 11, INK, True)
    columns = [MARGIN, MARGIN + 160, MARGIN + 231, MARGIN + 330, MARGIN + 421]
    y -= 20
    for value, x in zip(['Reference', 'Misses', 'Unmatched', 'Both <=5 px', 'Both <=10 px'], columns):
        text(pdf, value, x, y, 8, MUTED, True)
    y -= 17
    for value, x in zip(['R26 exploratory (661)', '15', '92', '558 / 584', '582 / 584'], columns):
        text(pdf, value, x, y, 8.5)
    y -= 17
    for value, x in zip(['Frozen primary (666)', '36', '108', '519 / 584', '573 / 584'], columns):
        text(pdf, value, x, y, 8.5)
    y = paragraph(pdf, 'On frozen 666, retained PR35 has 18 misses and 80 unmatched outputs, with 533 / 584 and 582 / 584 both-corner successes at 5 and 10 px. R26 does not establish a replacement. The selected visual cases below use <b>R26 661 only</b>.',
                  MARGIN, y - 12, WIDTH - 2 * MARGIN, 9, 13)
    y -= 25
    text(pdf, 'Complete coverage and bounded examples', MARGIN, y, 11, INK, True)
    y = paragraph(pdf, f"1. All {summary['misses']} scored misses, with the best retained overlapping box when one exists.<br/>"
                  f"2. All {summary['matched_corner_failure_targets']} matched targets containing a returned locatable-corner error >5 px or a locatable corner not returned: {summary['matched_returned_corner_errors_over_5px']} measured corner errors and {summary['matched_locatable_corners_not_returned']} not-returned corner.<br/>"
                  f"3. {summary['selected_box_cases']} additional matched targets with the lowest box IoU below 0.70.<br/>"
                  f"4. {summary['selected_unmatched_outputs']} highest-confidence unmatched outputs, explicitly classified as duplicate or other unmatched.",
                  MARGIN, y - 13, WIDTH - 2 * MARGIN, 9, 14)
    y -= 25
    text(pdf, 'Read the evidence literally', MARGIN, y, 11, INK, True)
    legend(pdf, MARGIN, y - 20)
    y = paragraph(pdf, 'Both panels use identical original pixels and crop bounds. Box confidence is 0.13, corner confidence 0.50, and matching IoU 0.50. Corner distances are in original-image pixels and are scored only for assigned matches with locatable reference points. Missing reference corners receive no invented location. Unmatched outputs are matching outcomes, not adjudicated background false positives. Retained predictions do not establish a raw suppression cause.',
                  MARGIN, y - 33, WIDTH - 2 * MARGIN, 9, 13)
    y -= 24
    text(pdf, 'Limits and provenance', MARGIN, y, 11, INK, True)
    y = paragraph(pdf, 'These reused development images do not establish new-session reliability. R26 trained on A100 and R25 on L4; their unchanged controls have different weights, so a cross-round difference cannot be assigned solely to three corrected training rows. A scored disagreement is not automatically a physical model error: cases whose annotations were subsequently amended in R27 are flagged and cross-referenced to the annotation PDF. The companion JSON preserves all 646 matches, all 15 misses, all 92 unmatched outputs, precise measurements and source hashes.',
                  MARGIN, y - 13, WIDTH - 2 * MARGIN, 8.5, 12.5)
    y = paragraph(pdf, 'PR: github.com/Siuuuers/dl/pull/103<br/>Candidate SHA-256: ' + MODEL_SHA[:24] + '...<br/>'
                  'Reference: fronts-2026-11-23-r26 | Report date: 23 November 2026',
                  MARGIN, y - 14, WIDTH - 2 * MARGIN, 7.5, 11, MUTED)
    require(y > 43, 'Introduction overflows footer')
    footer(pdf, 1, pages); pdf.showPage()


def crop_bounds(item, size):
    width, height = size
    boxes = [item[key] for key in ('reference_box_xyxy_px', 'prediction_box_xyxy_px') if item[key] is not None]
    xs = [box[index] for box in boxes for index in (0, 2)]
    ys = [box[index] for box in boxes for index in (1, 3)]
    for point in item['corners'].values():
        for key in ('reference_xy_px', 'prediction_xy_px'):
            value = point[key]
            if value is not None and (key == 'reference_xy_px' or point['returned_at_fixed_threshold']):
                xs.append(value[0]); ys.append(value[1])
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_width = max(70, (max(xs) - min(xs)) / 2 + 22)
    half_height = max(45, (max(ys) - min(ys)) / 2 + 28)
    return [max(0, math.floor(cx - half_width)), max(0, math.floor(cy - half_height)),
            min(width, math.ceil(cx + half_width)), min(height, math.ceil(cy + half_height))]


def draw_crop(pdf, crop, bounds, item, x, top, panel_width, panel_height, overlay=False):
    scale = min(panel_width / crop.width, panel_height / crop.height)
    dw, dh = crop.width * scale, crop.height * scale
    ox, oy = x + (panel_width - dw) / 2, top - dh
    pdf.drawImage(ImageReader(crop), ox, oy, dw, dh)
    pdf.setStrokeColor(RULE); pdf.setLineWidth(.5); pdf.rect(ox, oy, dw, dh, stroke=1, fill=0)
    if not overlay:
        return
    def transform(point):
        return ox + (point[0] - bounds[0]) * scale, top - (point[1] - bounds[1]) * scale
    for key, color, dashed in [('reference_box_xyxy_px', GREEN, False), ('prediction_box_xyxy_px', ORANGE, True)]:
        box = item[key]
        if box is None:
            continue
        a, b = transform(box[:2]), transform(box[2:])
        pdf.saveState(); pdf.setStrokeColor(color); pdf.setLineWidth(.95)
        if dashed:
            pdf.setDash(3, 2)
        pdf.rect(a[0], b[1], b[0] - a[0], a[1] - b[1], stroke=1, fill=0); pdf.restoreState()
    for point in item['corners'].values():
        ref, pred = point['reference_xy_px'], point['prediction_xy_px']
        if item['matched'] and point['error_px'] is not None:
            a, b = transform(ref), transform(pred)
            pdf.setStrokeColor(layout.colors.HexColor('#E8A563')); pdf.setLineWidth(.65)
            pdf.line(a[0], a[1], b[0], b[1])
        if ref is not None:
            layout.symbol(pdf, *transform(ref))
        if pred is not None and point['returned_at_fixed_threshold']:
            layout.symbol(pdf, *transform(pred), prediction=True)


def describe_case(item):
    category = item['selection_category']
    if category == 'miss':
        best = item['best_retained_overlap_iou']
        if best is None:
            return 'No retained same-class box overlaps this target at confidence 0.13. No assigned prediction; point distances are not scored.'
        index = item['prediction_index_0based'] + 1
        assigned = item['best_overlap_assigned_to_row']
        relation = 'below the 0.50 matching threshold' if best < .5 else f'assigned to row {assigned} instead'
        extra = f' It is assigned to row {assigned}.' if assigned is not None and best < .5 else ''
        return f'Best retained overlap: output p{index}, IoU {best:.3f}, {relation}.{extra} Its box is shown for context; no corner distances are scored.'
    if category == 'unmatched_output':
        row = item['best_reference_row']
        overlap = item['best_reference_overlap_iou']
        context = f'Best overlap: reference row {row}, IoU {overlap:.3f}; its box is context only.' if row else 'No retained same-class reference overlap.'
        return f"Scorer classification: {item['scorer_classification'].replace('_', ' ')}. {context} No target is assigned, so point distances are not scored."
    missing = [side for side, point in item['corners'].items() if point['reference_visibility'] == 2 and not point['returned_at_fixed_threshold']]
    if missing:
        return 'Locatable reference corner not returned at the fixed threshold: ' + ', '.join(missing) + '. Its reference ring remains visible; no displacement is fabricated.'
    if category == 'matched_box_overlap':
        return 'Additional low-overlap scorer match, selected after all corner-failure targets. The target is not repeated in the corner section.'
    return 'The prediction is assigned to this target. Distances use locatable reference corners only; unlocatable slots remain unscored.'


def case_card(pdf, item, image, index, top, crop_dir):
    category = {'miss': 'SCORED MISS', 'matched_corner_failure': 'CORNER FAILURE',
                'matched_box_overlap': 'BOX OVERLAP', 'unmatched_output': 'UNMATCHED OUTPUT'}[item['selection_category']]
    identity = f"row {item['current_row']}" if item['current_row'] is not None else f"output p{item['prediction_index_0based'] + 1}"
    text(pdf, f"{index:02d}   {item['image_id']}  /  {identity}", MARGIN, top, 11.5, INK, True)
    pdf.setFont('ReviewBold', 8); pdf.setFillColor(MUTED)
    pdf.drawRightString(WIDTH - MARGIN, top + 1, f"{category}  #{item['selection_rank']:02d}")
    if item['matched']:
        metric = f"Matched IoU {item['box_iou']:.3f}  |  Box confidence {item['box_confidence']:.3f}"
        largest = item['maximum_measured_corner_error_px']
        if largest is not None:
            metric += f'  |  Largest measured corner error {largest:.2f} px'
    elif item['selection_category'] == 'miss':
        metric = 'Unmatched reference target'
        if item['box_confidence'] is not None:
            metric += f"  |  Best-overlap output confidence {item['box_confidence']:.3f}"
    else:
        metric = f"Unmatched output  |  Box confidence {item['box_confidence']:.3f}  |  No assigned target"
    text(pdf, metric, MARGIN, top - 18, 8.3, MUTED)
    gap = 15
    panel_width = (WIDTH - 2 * MARGIN - gap) / 2
    text(pdf, 'ORIGINAL / UNMARKED', MARGIN, top - 37, 8, MUTED, True)
    text(pdf, 'REFERENCE + RETURNED R26', MARGIN + panel_width + gap, top - 37, 8, MUTED, True)
    bounds = crop_bounds(item, image.size)
    crop = image.crop(bounds)
    filename = f"{index:02d}-{item['image_id']}-{identity.replace(' ', '-')}-original.png"
    crop_path = crop_dir / filename
    crop.save(crop_path)
    require(Image.open(crop_path).convert('RGB').tobytes() == image.crop(bounds).tobytes(), 'Crop pixels changed')
    item.update(crop_xyxy_px=bounds, pdf_case_number=index, crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest())
    for overlay, x in [(False, MARGIN), (True, MARGIN + panel_width + gap)]:
        draw_crop(pdf, crop, bounds, item, x, top - 45, panel_width, 160, overlay)
    bottom = paragraph(pdf, describe_case(item), MARGIN, top - 214, WIDTH - 2 * MARGIN, 8, 10.5)
    require(bottom >= top - 247, 'Case explanation exceeds reserved area')
    y = top - 259
    columns = [MARGIN, MARGIN + 54, MARGIN + 188, MARGIN + 331, MARGIN + 430]
    for label, x in zip(('Corner', 'Reference (x, y)', 'Prediction (x, y)', 'Point conf.', 'Error (px)'), columns):
        text(pdf, label, x, y, 7.8, MUTED, True)
    for side in SIDES:
        y -= 16
        point = item['corners'][side]
        if point['reference_visibility'] is None:
            ref = 'unassigned'
        else:
            ref = layout.pair_text(point['reference_xy_px']) if point['reference_xy_px'] is not None else 'unlocatable'
        pred = (layout.pair_text(point['prediction_xy_px']) if point['returned_at_fixed_threshold'] else
                'unassigned' if point['prediction_confidence'] is None else 'not returned')
        conf = f"{point['prediction_confidence']:.3f}" if point['prediction_confidence'] is not None else '-'
        error = f"{point['error_px']:.2f}" if point['error_px'] is not None else 'not scored'
        for value, x in zip((side.capitalize(), ref, pred, conf, error), columns):
            text(pdf, value, x, y, 8)
    text(pdf, f"Crop x={bounds[0]}:{bounds[2]}, y={bounds[1]}:{bounds[3]}  |  Original image {image.width} x {image.height} px",
         MARGIN, top - 309, 7, MUTED)
    if item.get('later_annotation_amendment'):
        text(pdf, item['later_annotation_amendment']['notice'], MARGIN, top - 321, 6.8, ORANGE, True)
    return top - 321


def render(inventory, images, selected, output, crop_dir):
    layout.register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    pages = 1 + math.ceil(len(selected) / 2)
    pdf = canvas.Canvas(str(output), pagesize=layout.A4, pageCompression=1, invariant=1)
    pdf.setTitle('PR103 returned R26 candidate: misses and errors')
    pdf.setAuthor('Embedding vision annotation review')
    pdf.setSubject('Unamended R26 661-front development reference; frozen 666 reported separately')
    intro(pdf, inventory, pages)
    for start in range(0, len(selected), 2):
        text(pdf, 'PR103 / R26 candidate failures', MARGIN, HEIGHT - 35, 11, INK, True)
        legend(pdf, MARGIN, HEIGHT - 55)
        for slot, item in enumerate(selected[start:start + 2]):
            top = HEIGHT - 89 - slot * 350
            bottom = case_card(pdf, item, images[item['image_id']], start + slot + 1, top, crop_dir)
            require(bottom > 36, 'Case card overflows footer')
            if slot == 0:
                pdf.setStrokeColor(RULE); pdf.setLineWidth(.6)
                pdf.line(MARGIN, top - 326, WIDTH - MARGIN, top - 326)
        footer(pdf, 2 + start // 2, pages); pdf.showPage()
    pdf.save()
    inventory['pdf'] = {'filename': output.name, 'sha256': sha(output), 'pages': pages,
                        'cases_per_page': 2, 'intro_pages': 1, 'unmarked_crops_pixel_identical': True,
                        'numeric_display': 'Coordinates and distances: two decimals; confidence and IoU: three decimals.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--predictions', type=Path, default=EVIDENCE / 'inference/eval-candidate/predictions.jsonl')
    parser.add_argument('--comparison', type=Path, default=EVIDENCE / 'fixed-comparison.json')
    parser.add_argument('--inventory', type=Path, default=EVIDENCE / 'candidate-error-inventory.json')
    parser.add_argument('--annotation-review', type=Path, default=ANNOTATION_REVIEW)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--crop-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda path: path.resolve() if path.is_absolute() else root / path
    inventory, images, selected = gather(root, resolve(args.predictions), resolve(args.comparison))
    add_annotation_crossreferences(root, inventory, selected, resolve(args.annotation_review))
    render(inventory, images, selected, args.output.resolve(), args.crop_dir.resolve())
    inventory_path = resolve(args.inventory)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'pdf': str(args.output.resolve()), 'inventory': str(inventory_path),
                      'pages': inventory['pdf']['pages'], **inventory['summary']}, indent=2))


if __name__ == '__main__':
    main()
