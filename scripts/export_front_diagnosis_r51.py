#!/usr/bin/env python3
"""Render a focused R51 report from the verified PR143 R49 diagnosis return.

The report consumes saved decoded proposals and outputs; it never loads a model,
trains, writes labels or silently changes a deployment threshold.
"""
import argparse
from collections import Counter
import hashlib
import html
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from audit_additional_front_pilot_r45_results import corner_groups
from evaluate_pose import parse_pose_labels
from pose_scoring import aggregate_scores, score_image, _box_iou

ARCHIVE_SHA = '892c8ee79ff470990bb47dc59f947ad6bd2577a86bc9827e9dd57f81c56213f3'
ARCHIVE_BYTES = 15631804
SCRIPT_SHA = '188df92036d5dc8fd8e13fc8d8815728a56c33ced654de2cf4716788eac47811'
R50_PATH = 'data/reviews/fronts-2026-11-27-r50/manifest.json'
R50_SHA = 'f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f'
OBS_PATH = 'results/evaluation/front-errors-r50-20261127/case-observations.json'
GUARD_PATH = 'results/evaluation/pr143-r49-20261127/partial-classification-gradient-guard.json'
RESULTS = 'results/review/front-diagnosis-r51-20261127'
STAGE_NAMES = {
    'confidence_gate': 'Confidence gate',
    'no_iou_eligible_decoded_box': 'No IoU-eligible decoded box',
    'ordinary_nms': 'Ordinary NMS',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def compact_report(report):
    detection = {k: v for k, v in report['detection'].items() if k != 'endpoint_errors_px'}
    return {'detection': detection, 'corners_by_locatable_count': report['corners_by_locatable_count']}


def original_boxes(raw, trace):
    raw = raw[0]
    boxes = np.column_stack((raw[0] - raw[2] / 2, raw[1] - raw[3] / 2,
                             raw[0] + raw[2] / 2, raw[1] + raw[3] / 2))
    height, width = trace['input_shape'][2:]
    gain = min(height / 720, width / 1280)
    px, py = round((width - 1280 * gain) / 2 - .1), round((height - 720 * gain) / 2 - .1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / gain
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, 1280)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, 720)
    return boxes


def collect(root, archive, images_dir, audit_dir, probe_path):
    from PIL import Image
    audit = read_json(audit_dir / 'verification.json')
    need(audit.get('status') == 'passed' or audit.get('passed') is True,
         'The independent return audit must pass before report generation')
    data = archive.read_bytes()
    need(len(data) == ARCHIVE_BYTES and sha(data) == ARCHIVE_SHA, 'PR143 archive differs')
    with zipfile.ZipFile(archive) as z:
        inventory = json.loads(z.read('inventory.json'))
        need(set(z.namelist()) == set(inventory) | {'inventory.json'} and z.testzip() is None,
             'Return membership or CRC differs')
        for name, entry in inventory.items():
            content = z.read(name)
            need(entry == {'bytes': len(content), 'sha256': sha(content)}, 'Member differs: ' + name)
        need(sha(z.read('diagnose_r49.py')) == SCRIPT_SHA, 'Returned script differs')
        summary = json.loads(z.read('summary.json'))
        predictions = json.loads(z.read('predictions.json'))
        traces = json.loads(z.read('trace.json'))
        misses = json.loads(z.read('missed-cases.json'))
        decoded = {identity: np.load(io.BytesIO(z.read('decoded/' + identity + '.npy')), allow_pickle=False)
                   for identity in traces}
    need(summary['saved_baseline_exact_images'] == 20 and len(misses) == 15,
         'Unexpected baseline/miss population')
    need(summary['fresh_inference'] and not summary['training'] and not summary['automatic_promotion'],
         'Returned experiment scope differs')
    need(Counter(c['mechanism'] for c in misses) ==
         {'confidence_gate': 10, 'no_iou_eligible_decoded_box': 3, 'ordinary_nms': 2}, 'Stages differ')
    need(sha((root / R50_PATH).read_bytes()) == R50_SHA, 'R50 reference differs')
    manifest = read_json(root / R50_PATH)
    gold, originals, image_hashes, label_hashes = {}, {}, {}, {}
    for entry in manifest['files']:
        identity = entry['image_id']
        lp = (root / R50_PATH).parent / entry['label_path']
        need(sha(lp.read_bytes()) == entry['label_sha256'], 'R50 label differs: ' + identity)
        gold[identity] = parse_pose_labels(lp.read_text(), 1280, 720)
        label_hashes[str(lp.relative_to(root))] = entry['label_sha256']
        image = images_dir / (identity + '.png')
        need(sha(image.read_bytes()) == entry['source_image_sha256'], 'Original differs: ' + identity)
        originals[identity] = Image.open(image).convert('RGB')
        need(originals[identity].size == (1280, 720), 'Original dimensions differ')
        image_hashes[identity] = entry['source_image_sha256']
    reports = dict(summary['reports'])
    reports['r50'] = {}
    for view in ('hard_raw', 'hard', 'soft_raw', 'soft'):
        scores, groups = {}, {str(n): Counter() for n in (0, 1, 2)}
        for identity, targets in gold.items():
            outputs = predictions[identity][view]
            scores[identity] = score_image(targets, outputs, 1280, 720, .5, .5)
            for key, values in corner_groups(targets, outputs, scores[identity]).items():
                groups[key].update(values)
        reports['r50'][view] = {'detection': aggregate_scores(list(scores.values())),
                                'corners_by_locatable_count': {k: dict(v) for k, v in groups.items()},
                                'per_image': scores}
    observations = {x['case_id']: x for x in read_json(root / OBS_PATH)['cases']}
    cases = []
    competitions = []
    for case in misses:
        identity, row = case['image'], case['r38_row']
        need(case['gold'] == gold[identity][row - 1], 'A miss reference changed: ' + identity)
        outputs = predictions[identity]['hard']
        near = max(range(len(outputs)), key=lambda j: _box_iou(case['gold']['bbox_xyxy'], outputs[j]['bbox_xyxy']))
        strongest = max(case['eligible'], key=lambda x: (x['confidence'], -x['anchor_index'])) if case['eligible'] else None
        record = dict(case)
        record.update({'case_id': f'{identity}:R{row:03d}', 'strongest_eligible': strongest,
                       'best_retained': {'prediction_number': near + 1, 'prediction': outputs[near],
                                         'iou': _box_iou(case['gold']['bbox_xyxy'], outputs[near]['bbox_xyxy'])},
                       'r50_geometry_unchanged': True,
                       'visual_observation': observations[f'M:{identity}:R{row:03d}']['observation']})
        if case['mechanism'] == 'ordinary_nms':
            raw = decoded[identity]
            boxes = original_boxes(raw, traces[identity])
            eligible_anchor = strongest['anchor_index']
            winners = [k for k in predictions[identity]['ordinary_indices']
                       if raw[0, 4, k] > strongest['confidence'] and _box_iou(boxes[k], boxes[eligible_anchor]) > .70]
            need(bool(winners), 'No higher-score suppressor found')
            winner = winners[0]
            raw_position = predictions[identity]['ordinary_indices'].index(winner)
            prediction = predictions[identity]['hard_raw'][raw_position]
            overlap = float(_box_iou(boxes[winner], boxes[eligible_anchor]))
            competition = {'image': identity, 'row': row, 'gold': case['gold'],
                'eligible': strongest, 'winner_anchor': winner, 'winner': prediction,
                'winner_iou': _box_iou(case['gold']['bbox_xyxy'], prediction['bbox_xyxy']),
                'pair_iou': overlap, 'score_after_first_soft_decay': strongest['confidence'] * (1 - overlap)}
            need(competition['score_after_first_soft_decay'] < .13, 'Soft decay expectation differs')
            competitions.append(competition)
            record['competition'] = competition
        cases.append(record)
    removed, added = [], []
    for identity, outputs in predictions.items():
        before, after = set(outputs['hard_indices']), set(outputs['soft_indices'])
        removed.extend((identity, k) for k in sorted(before - after))
        added.extend((identity, k) for k in sorted(after - before))
    need(removed == [('new_000090', 10313)] and not added, 'Final proposal difference changed')
    identity, anchor = removed[0]
    pos = predictions[identity]['hard_indices'].index(anchor)
    removed_output = predictions[identity]['hard'][pos]
    row = max(range(len(gold[identity])), key=lambda j: _box_iou(removed_output['bbox_xyxy'], gold[identity][j]['bbox_xyxy']))
    match = next(m for m in reports['r50']['hard']['per_image'][identity]['matches'] if m['gold_index'] == row)
    removal = {'image': identity, 'anchor_index': anchor, 'prediction_number': pos + 1,
               'prediction': removed_output, 'row': row + 1, 'gold': gold[identity][row],
               'owner': predictions[identity]['hard'][match['prediction_index']],
               'owner_prediction_number': match['prediction_index'] + 1}
    probe = read_json(probe_path)
    need(probe['archive_sha256'] == ARCHIVE_SHA and probe['settings']['confidences'] == [.13, .05]
         and probe['post_hoc_diagnostic'] and not probe['fresh_inference'] and not probe['fresh_training']
         and probe['same_decoded_tensors'] and not probe['automatic_promotion'], 'Confidence probe differs')
    receipts = probe['baseline_reconstruction']
    need(len(receipts) == 20 and {r['image'] for r in receipts} == set(traces)
         and all(r['native_anchor_indices_exact'] and r['native_full_predictions_exact']
                 and r['native_postclip_audit_exact'] for r in receipts), 'Confidence replay baseline differs')
    for name in ('frozen666', 'historical663', 'r38', 'r50'):
        need(compact_report(probe['reports'][name]['hard013']) == compact_report(reports[name]['hard']),
             'Confidence probe baseline differs: ' + name)
    guard = read_json(root / GUARD_PATH)
    full = guard['full_network_control_equivalence']
    need(guard['status'] == 'passed' and guard['optimizer_updates'] == 0
         and guard['native_branch_schedule_bitwise_equal']
         and full['factor1_native_full_network_parameter_gradients_bitwise_equal']
         and full['parameter_tensors_with_gradient'] == 498 and full['optimizer_updates'] == 0,
         'Native classification guard differs')
    for name, digest in guard['source_sha256'].items():
        need(sha((root / 'scripts' / name).read_bytes()) == digest, 'Guarded source changed: ' + name)
    for name in ('mixed_real_batch', 'native_class_weights_1_7', 'empty_batch',
                 'all_fully_visible_synthetic_batch', 'consumed_visibility_mutation'):
        case = guard[name]
        need(case['factor1_native_loss_and_prediction_gradients_bitwise_equal']
             and case['factor2_assignment_and_target_geometry_bitwise_unchanged']
             and case['factor2_nonclassification_loss_components_bitwise_equal']
             and case['factor2_rle_flow_parameter_gradients_bitwise_equal'], 'Guard case differs: ' + name)
        for checks in case['selective_classification_gradient_check'].values():
            need(checks['box_keypoint_and_sigma_gradients_bitwise_unchanged']
                 and checks['full_and_background_score_gradients_bitwise_unchanged']
                 and checks['partial_score_gradient_factor_exact'] == 2, 'Selective gradient differs')
    return dict(summary=summary, reports=reports, cases=cases, competitions=competitions,
                removal=removal, originals=originals, image_hashes=image_hashes, label_hashes=label_hashes,
                probe=probe, probe_sha256=sha(probe_path.read_bytes()),
                observations_sha256=sha((root / OBS_PATH).read_bytes()),
                native_guard_sha256=sha((root / GUARD_PATH).read_bytes()),
                audit_sha256=sha((audit_dir / 'verification.json').read_bytes()))


def render(data, output):
    from PIL import Image
    import reportlab
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    fonts = Path(reportlab.__file__).parent / 'fonts'
    pdfmetrics.registerFont(TTFont('R51', str(fonts / 'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('R51Bold', str(fonts / 'VeraBd.ttf')))
    pdfmetrics.registerFontFamily('R51', normal='R51', bold='R51Bold', italic='R51', boldItalic='R51Bold')
    W, H = landscape(A4)
    M, BW = 30, W - 60
    ink, muted, teal, rule, cyan, amber, magenta = map(HexColor,
        ('#183344', '#506875', '#087f85', '#d7e2e6', '#00d7e9', '#ffae32', '#e734a9'))
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=(W, H), invariant=1, pageCompression=1)
    pdf.setTitle('R51 - completed cassette diagnosis, confidence bottleneck and next steps')
    pdf.setAuthor('Siuuuers/dl project review')
    page, index, bounds_log, text_bounds_log = 0, [], [], []
    esc = lambda x: html.escape(str(x))

    def text(x, y, value, size=9, bold=False, color=ink):
        font = 'R51Bold' if bold else 'R51'
        width = pdfmetrics.stringWidth(str(value), font, size)
        need(x >= M - .01 and x + width <= W - M + .01,
             f'Text width overflow page {page}: {value}')
        text_bounds_log.append({'page': page, 'left': x, 'right': x + width, 'baseline': y})
        pdf.setFont(font, size); pdf.setFillColor(color); pdf.drawString(x, y, str(value))

    def para(value, x, y, width=BW, size=9, leading=13, bottom=40, color=ink):
        style = ParagraphStyle('r51', fontName='R51', fontSize=size, leading=leading, textColor=color)
        p = Paragraph(value, style); _, h = p.wrap(width, 1200)
        need(y - h >= bottom - .01, f'Paragraph overflow page {page}: {y-h:.1f} < {bottom}')
        p.drawOn(pdf, x, y - h)
        bounds_log.append({'page': page, 'bottom': round(y-h, 3), 'minimum': bottom})
        return y - h

    def start(section, title, subtitle):
        nonlocal page
        page += 1
        pdf.setFillColor(teal); pdf.rect(0, H - 7, W, 7, stroke=0, fill=1)
        text(M, H - 30, section.upper(), 8, True, teal)
        text(M, H - 57, title, 18, True)
        para(esc(subtitle), M, H - 73, size=8.8, leading=11.5)

    def end():
        pdf.setStrokeColor(rule); pdf.line(M, 31, W - M, 31)
        text(M, 17, 'R51 | 27 Nov 2026 HKT | PR143 completed diagnosis | frozen666 primary; R38/R50 development review', 7, color=muted)
        text(W - M - 21, 17, f'{page:02d}', 9, True, teal); pdf.showPage()

    def table(headers, rows, y, widths, height=25, size=8.5):
        for n, row in enumerate([headers] + rows):
            pdf.setFillColor(HexColor('#e8f2f3') if n == 0 else HexColor('#f4f7f9') if n % 2 else white)
            pdf.rect(M, y-height, sum(widths), height, stroke=0, fill=1)
            x = M
            for value, width in zip(row, widths):
                value = esc(value)
                para('<b>' + value + '</b>' if n == 0 else value, x+6, y-5, width-12,
                     size, size+2, bottom=y-height+2)
                x += width
            y -= height
        return y - 13

    def crop_bounds(boxes):
        x0, y0 = min(b[0] for b in boxes)-22, min(b[1] for b in boxes)-20
        x1, y1 = max(b[2] for b in boxes)+22, max(b[3] for b in boxes)+20
        w, h = min(1280, max(92, x1-x0)), min(720, max(68, y1-y0))
        cx, cy = (x0+x1)/2, (y0+y1)/2
        x0, y0 = min(max(0, cx-w/2), 1280-w), min(max(0, cy-h/2), 720-h)
        return math.floor(x0), math.floor(y0), math.ceil(x0+w), math.ceil(y0+h)

    def panel(identity, region, x, y, w, h, target=None, proposal=None, second=None):
        crop = data['originals'][identity].crop(region)
        zoom = crop.resize((crop.width*3, crop.height*3), Image.Resampling.NEAREST)
        scale = min(w/crop.width, h/crop.height)
        iw, ih = crop.width*scale, crop.height*scale
        ox, oy = x+(w-iw)/2, y+(h-ih)/2
        pdf.drawImage(ImageReader(zoom), ox, oy, iw, ih)
        def xy(px, py): return ox+(px-region[0])*scale, oy+ih-(py-region[1])*scale
        def box(obj, color, dashed=False, points=False):
            left, bottom = xy(obj['bbox_xyxy'][0], obj['bbox_xyxy'][3])
            right, top = xy(obj['bbox_xyxy'][2], obj['bbox_xyxy'][1])
            pdf.setStrokeColor(color); pdf.setLineWidth(.85)
            pdf.setDash(2, 2) if dashed else pdf.setDash()
            pdf.rect(left, bottom, right-left, top-bottom, stroke=1, fill=0); pdf.setDash()
            if points:
                for px, py, visible in obj.get('keypoints', []):
                    if visible == 2:
                        xx, yy = xy(px, py); pdf.circle(xx, yy, 1.8, stroke=1, fill=0)
        pdf.saveState(); clip = pdf.beginPath(); clip.rect(ox, oy, iw, ih); pdf.clipPath(clip, stroke=0)
        if target: box(target, cyan, points=True)
        if second: box(second, magenta, True)
        if proposal: box(proposal, amber)
        pdf.restoreState()

    start('Completed result', 'The diagnosis completed. Soft-NMS recovered no missed fronts.',
          'Same frozen PR137 candidate | 20 original development photographs | freshly inferred on Tesla T4')
    y = para('PR143 returns the completed R49 experiment. Its fresh hard-NMS baseline exactly reproduces the saved PR137 candidate on <b>all 20 images</b>. The fixed linear Soft-NMS pipeline removes one final output while preserving every missed target and the reported corner-success counts.', M, 483, size=10.4, leading=14.5)
    rows = []
    for name, label in [('frozen666', 'Frozen666 / primary'), ('historical663', 'Historical663'), ('r38', 'R38 / development'), ('r50', 'R50 / proposal')]:
        for arm in ('hard', 'soft'):
            r = data['reports'][name][arm]; d = r['detection']; c = r['corners_by_locatable_count']['2']
            rows.append([label, arm.title(), d['fn'], d['fp'], d['duplicate_fp'], f"{c['within5']} / {c['eligible']}"])
    y = table(['Reference', 'Pipeline', 'Misses', 'Unmatched', 'Dup. subset', 'Both corners <=5px'], rows, y-16,
              [185, 85, 75, 100, 100, BW-545], height=26, size=8.8)
    y = para('<b>Interpretation:</b> the measured benefit is one fewer unmatched output, with no recall gain. This does not justify automatically changing the selected model or inference defaults. PR35 remains selected under the existing frozen-reference gate.', M, y, size=10, leading=14)
    para('These reference versions describe the same images, not independent test sets. R50 is a CPU rescore of the returned tensors/outputs using the existing R50 labels; the submitted R49 notebook originally scored the first three references.', M, y-12, size=9, leading=13, color=muted)
    index.append({'page': page, 'section': 'result_summary'}); end()

    start('Measured loss stages', 'Where the 15 missed targets are lost',
          'All 15 remain missed by the submitted Soft-NMS pipeline; their reference geometry is identical in R38 and R50.')
    rows = []
    for c in data['cases']:
        best, strongest = c['best_decoded'], c['strongest_eligible']
        rows.append([c['image'].replace('new_000', '') + ' / R' + f"{c['r38_row']:03d}",
                     STAGE_NAMES[c['mechanism']], c['locatable_corners'], f"{best['iou']:.4f}",
                     f"{strongest['confidence']:.5f}" if strongest else 'none'])
    y = table(['Image / row', 'First loss stage', 'Corners', 'Best decoded IoU', 'Max score with IoU>=.50'],
              rows, 481, [120, 220, 70, 145, BW-555], height=20.5, size=8.2)
    y = para('<b>10 confidence-gate cases:</b> at least one decoded box reaches IoU 0.50, but every such box has score at or below 0.13. <b>3 geometry cases:</b> no decoded box reaches that matching IoU. <b>2 NMS cases:</b> suitable above-threshold boxes exist, then lose to a higher-score nearby proposal.', M, y, size=9.4, leading=13)
    para('The best-IoU box and highest-score eligible box may be different proposals. A pipeline stage is now measured; the reason the network learned that score or extent is still a separate hypothesis. Scores are model outputs, not calibrated probabilities of physical truth.', M, y-12, size=9, leading=12.5, color=muted)
    index.append({'page': page, 'section': 'all_miss_stage_table'}); end()

    start('Filter effect', 'Only one final proposal is removed',
          'R38 numbers below separate ordinary suppression from the unchanged postclip filter.')
    rows = []
    for view, label in [('hard_raw', 'Hard NMS / before postclip'), ('soft_raw', 'Soft-NMS / before postclip'),
                        ('hard', 'Hard NMS / after postclip'), ('soft', 'Soft-NMS / after postclip')]:
        d = data['reports']['r38'][view]['detection']
        rows.append([label, d['prediction_count'], d['fn'], d['fp'], d['duplicate_fp']])
    y = table(['Pipeline stage', 'Outputs', 'Misses', 'Unmatched', 'Dup. subset'], rows, 482,
              [310, 100, 100, 130, BW-640], height=25, size=8.9)
    y = para('Before postclip, Soft-NMS adds 14 proposal identities and removes one. Its 14 additions are subsequently suppressed by the unchanged postclip filter. <b>Raw Soft-NMS already recovers zero targets</b>, so postclip did not erase a measured recall recovery.', M, y, size=9.3, leading=13)
    removal = data['removal']; region = crop_bounds([removal['gold']['bbox_xyxy'], removal['prediction']['bbox_xyxy'], removal['owner']['bbox_xyxy']])
    top = y-13; pw=(BW-24)/3; labels=['Original pixels', 'Hard: extra P060 and assigned P043', 'Soft: assigned front remains']
    for j in range(3):
        x=M+j*(pw+12)
        panel(removal['image'], region, x, top-125, pw, 125,
              removal['gold'] if j else None,
              removal['prediction'] if j==1 else removal['owner'] if j==2 else None,
              removal['owner'] if j==1 else None)
        text(x, top-138, labels[j], 8, True)
    y = top-154
    y = para('<b>Removed:</b> new_000090 / hard-output P060, decoded anchor 10313, score 0.13649. It is a repeated hypothesis for R041, with its right edge extending toward the gripper. The separately assigned P043 remains. Cyan is the reference; amber is the highlighted output; magenta dotted is the assigned owner.', M, y, size=9.2, leading=12.7)
    para('The final selected proposal set is otherwise identical on all 20 images. This is a bounded development-set result of this complete filter pipeline, not a claim that every possible Soft-NMS configuration has been tested.', M, y-10, size=8.8, leading=12, color=muted)
    index.append({'page': page, 'section': 'soft_pipeline_and_removed_output', 'image': removal['image'], 'anchor_index': 10313}); end()

    def miss_card(case, x, y, w, h):
        pdf.setStrokeColor(rule); pdf.roundRect(x, y, w, h, 5, stroke=1, fill=0)
        text(x+10, y+h-18, case['image'] + ' / R' + f"{case['r38_row']:03d}" + ' | ' + STAGE_NAMES[case['mechanism']], 10, True)
        best, strong, retained = case['best_decoded'], case['strongest_eligible'], case['best_retained']
        text(x+10, y+h-32, f"{case['locatable_corners']} locatable corner(s) | hard and submitted Soft-NMS both miss", 8.2, color=muted)
        left_w, right_x, right_w = 455, x+477, w-489
        has_retained_overlap = retained['iou'] > 0
        region_boxes = [case['gold']['bbox_xyxy'], best['bbox_xyxy']]
        if has_retained_overlap:
            region_boxes.append(retained['prediction']['bbox_xyxy'])
        region=crop_bounds(region_boxes)
        pw=(left_w-18)/3
        for j, label in enumerate(['Original pixels', 'Best decoded geometry',
                                   'Nearest retained hard output' if has_retained_overlap else 'No retained box overlaps']):
            xx=x+10+j*(pw+9)
            panel(case['image'],region,xx,y+50,pw,120,case['gold'] if j else None,
                  best if j==1 else retained['prediction'] if j==2 and has_retained_overlap else None)
            text(xx,y+40,label,6.9,True)
        retained_note = (f"Retained P{retained['prediction_number']:03d}: IoU {retained['iou']:.4f}."
                         if has_retained_overlap else 'No retained output has positive overlap.')
        bottom_note=f"Decoded anchor {best['anchor_index']}: IoU {best['iou']:.4f}, score {best['confidence']:.5f}. {retained_note}"
        para(esc(bottom_note),x+10,y+29,left_w,7.6,9.6,bottom=y+7)
        if case['mechanism']=='confidence_gate':
            conclusion=(f"Geometry reaches the match gate, but the highest score among eligible boxes is <b>{strong['confidence']:.5f}</b>, below the strict 0.13 threshold. No eligible proposal reaches NMS at this operating point.")
            action='Use the measured confidence probe to quantify recovery cost. Review eligible training-side partial-front supervision before selecting a new controlled treatment.'
        elif case['mechanism']=='no_iou_eligible_decoded_box':
            conclusion=f"The largest decoded IoU is <b>{best['iou']:.4f}</b>, below 0.50. Lowering confidence alone cannot make a geometrically ineligible box pass that fixed gate."
            action='Inspect visible extent and ownership, including the near-gate case. Keep matching fixed; make any annotation correction only from original-image evidence.'
        else:
            comp=case['competition']
            conclusion=(f"An eligible box scores <b>{strong['confidence']:.5f}</b>. A higher-score nearby box survives with reference IoU <b>{comp['winner_iou']:.4f}</b>. Linear decay pushes the eligible score below 0.13.")
            action='Inspect the competing box extents and their ranking. See the NMS competition page; ordinary overlap suppression and confidence are interacting here.'
        content='<b>Visible context:</b> '+esc(case['visual_observation'])+'<br/><b>Measured:</b> '+conclusion+'<br/><b>Next:</b> '+esc(action)
        para(content,right_x,y+h-44,right_w,8.35,10.55,bottom=y+8)
        index.append({'page':page,'section':'miss','image':case['image'],'row':case['r38_row'],
                      'stage':case['mechanism'],'best_decoded_anchor':best['anchor_index'],
                      'highest_score_eligible_anchor':strong['anchor_index'] if strong else None})

    for offset in range(0,len(data['cases']),2):
        start('Complete miss review', 'All 15 misses: pixels, proposals and retained outputs',
              f"Cases {offset+1}-{min(offset+2,15)} | Cyan: reference and visible reference points; amber: displayed proposal. Crops are original pixels.")
        for j,case in enumerate(data['cases'][offset:offset+2]):miss_card(case,M,270 if j==0 else 42,BW,217)
        end()

    start('NMS competition', 'Strong eligible boxes lose to slightly worse extents',
          'The comparisons below use the highest-score eligible proposal, not the highest-IoU proposal.')
    for j,c in enumerate(data['competitions']):
        y=280 if j==0 else 57
        text(M,y+184,c['image']+' / R'+f"{c['row']:03d}",11,True)
        region=crop_bounds([c['gold']['bbox_xyxy'],c['eligible']['bbox_xyxy'],c['winner']['bbox_xyxy']])
        panel(c['image'],region,M,y+25,220,145,c['gold'],c['eligible'],c['winner'])
        x=M+240
        content=(f"<b>Eligible anchor {c['eligible']['anchor_index']}:</b> score {c['eligible']['confidence']:.5f}; reference IoU {c['eligible']['iou']:.5f}.<br/>"
                 f"<b>Winning anchor {c['winner_anchor']}:</b> score {c['winner']['confidence']:.5f}; reference IoU {c['winner_iou']:.5f}.<br/>"
                 f"<b>Proposal-to-proposal IoU:</b> {c['pair_iou']:.5f}. The above-threshold eligible box is suppressed by ordinary NMS.<br/>"
                 f"<b>Linear Soft-NMS:</b> its score after this overlap decay is {c['score_after_first_soft_decay']:.5f}, still below 0.13.<br/><br/>"
                 'The surviving box is very close to the 0.50 reference-match boundary. This demonstrates score/extent ranking competition, not proof that the reference is physically exact or that suppression alone is the learning cause.')
        para(content,x,y+167,BW-240,9.1,12.8,bottom=y+18)
        text(M,y+12,'Cyan: reference | Amber: eligible box | Magenta dotted: winner',7.3,color=muted)
    index.append({'page':page,'section':'nms_competitions'});end()

    start('One post-hoc CPU probe', 'Lower confidence recovers six fronts, with 74 extra unmatched outputs',
          'Same returned tensors; native hard NMS and postclip; confidence 0.13 versus 0.05. This is not a new model inference or threshold promotion.')
    rows=[]
    for name,label in [('frozen666','Frozen666 / primary'),('historical663','Historical663'),('r38','R38 / development'),('r50','R50 / proposal')]:
        r=data['probe']['reports'][name]
        a,b=(r[v]['detection'] for v in ('hard013','hard005'))
        rows.append([label,f"{a['fn']} -> {b['fn']}",f"{a['fp']} -> {b['fp']}",f"{a['duplicate_fp']} -> {b['duplicate_fp']}",
                     f"{r['hard005']['corners_by_locatable_count']['2']['within5']} / {r['hard005']['corners_by_locatable_count']['2']['eligible']}"])
    y=table(['Reference','Misses','Unmatched','Dup. subset','Both corners <=5px'],rows,481,[185,125,130,140,BW-580],height=30,size=9)
    y=para('The R38/R50 recovered rows are <b>008:R001, 008:R034, 010:R030, 010:R033, 011:R059 and 090:R039</b>. There are no newly missed R38/R50 targets. Both-corner 5px success remains 566/583; recovery of partial fronts is not a new two-corner gain.',M,y,size=10,leading=14)
    y=para('Five recoveries come from the confidence-gate group. The sixth, new_000090/R039, was a baseline NMS case: a different lower-score, narrower proposal becomes available and survives. The original high-score eligible proposal is not magically unsuppressed by lowering confidence.',M,y-13,size=10,leading=14)
    y=para('<b>Interpretation:</b> the model already contains useful partial-front geometry for some misses. A global lower threshold admits many additional outputs as well. The measured exchange is six recovered R38/R50 targets for 74 more unmatched outputs; the 0.05 probe is diagnostic evidence, not a new default.',M,y-13,size=10,leading=14)
    y=para('The CPU replay first reproduced all 20 native baseline proposal-index sets, boxes, keypoints and postclip outcomes exactly at 0.13. It then tested this single additional cutoff. It is a post-hoc probe of reused development data, not an independent performance estimate or an optimization sweep.',M,y-13,size=9.2,leading=13,color=muted)
    para('The known exclusions and held regions remain unchanged. Unmatched boxes are not automatically physical background, and none are copied into the annotation set by this replay.',M,y-13,size=9.2,leading=13,color=muted)
    index.append({'page':page,'section':'confidence_probe'});end()

    start('Next step and scope', 'Use the diagnosis to choose one new training question',
          'The professor assignment remains annotation and training; the thesis topic is open. Existing data are sufficient for a bounded follow-up.')
    y=para('<b>Established:</b> ten misses are blocked by confidence at the recorded threshold; three lack matching decoded geometry; two arise from competing proposal extents. The fixed Soft-NMS pilot recovers none. One lower-confidence replay recovers six but adds 74 unmatched outputs.',M,482,size=10.5,leading=15)
    y=para('<b>Next isolated comparison:</b> native classification BCE on already-assigned partial targets, weighted by <b>factor 1 versus factor 2</b>. Keep native assignment, its soft target scores and the unweighted normalizer. Box, pose and visibility loss formulas remain native. R30/PR109 jointly weighted box and classification; R31 weighted boxes only, with a failed seed-47 screen. R51 isolates classification; it is not the first partial-target weighting study.',M,y-15,size=10,leading=14)
    y=para('<b>Both arms:</b> the same 142 photographs / 3,999 fronts (3,397 two-corner, 465 one-corner and 137 zero-corner), original epoch 173 initializer, seed 46, 30 epochs, batch 4, input 1280, FP32 and AdamW. Each arm has 1,080 batches, 80 expected optimizer updates and 4,260 image presentations. All held photographs, labels and reference versions stay fixed.',M,y-14,size=9.7,leading=13.5)
    y=para('<b>CPU validation passed:</b> factor 1 reproduces the native loss and all 498 network-parameter gradients bit-for-bit. At fixed outputs, factor 2 doubles only assigned-partial classification gradients; other output/RLE gradients, assignment and normalization remain exact. Empty/full, visibility and class-weight checks passed. No optimizer updates were run. Shared-feature learning can still change future boxes and corners; native soft targets do not guarantee a score increase.',M,y-14,size=9.7,leading=13.5)
    y=para('No new annotation is manufactured from a low-confidence prediction. The R50 bottom correction and all explicit exclusions/holds remain unchanged. The R49 diagnosis has already returned, so rerunning its equivalent R50 inference notebook merely to obtain the same tensors is unnecessary. This report contains no GPU result for the new training comparison.',M,y-14,size=9.5,leading=13)
    y=para('<b>Interpretation limit:</b> these photographs have repeatedly guided development. The result is a controlled within-dataset diagnosis. New capture sessions can later test broader generalization, but their absence does not prevent the present reproducible work.',M,y-15,size=9.4,leading=13,color=muted)
    para('Sources: PR143 return and its pinned R49 script; scripts/pose_scoring.py; R50 versioned labels; R51 return audit and single-confidence replay. Exact source hashes, image hashes and page/case indexes are stored beside the report-generation verification in results/review/front-diagnosis-r51-20261127/.',M,y-14,size=8.5,leading=12,color=muted)
    index.append({'page':page,'section':'next_step_and_limits'});end();pdf.save()
    return page,index,bounds_log,text_bounds_log


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--images-dir',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path)
    parser.add_argument('--confidence-probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--results-dir',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    audit_dir=args.audit_dir or root/'results/evaluation/pr143-r49-20261127'
    results=args.results_dir or root/RESULTS;results.mkdir(parents=True,exist_ok=True)
    data=collect(root,args.archive,args.images_dir,audit_dir,args.confidence_probe)
    pages,index,bounds,text_bounds=render(data,args.output)
    need(len([p for p in index if p['section']=='miss'])==15,'Incomplete miss coverage')
    for relative,digest in data['label_hashes'].items():need(sha((root/relative).read_bytes())==digest,'Label changed during rendering')
    pdfhash=sha(args.output.read_bytes())
    compact_cases=[{k:v for k,v in c.items() if k!='eligible'} for c in data['cases']]
    records={
        'pdf-index.json':{'schema':'r51-diagnostic-pdf-index-v1','pages':pages,'pdf_sha256':pdfhash,'entries':index},
        'case-diagnostics.json':{'schema':'r51-diagnostic-card-data-v1','archive_sha256':ARCHIVE_SHA,
            'r50_manifest_sha256':R50_SHA,'cases':compact_cases,'competitions':data['competitions'],'removed_output':data['removal']},
        'verification.json':{'schema':'r51-diagnostic-pdf-verification-v1','status':'passed','pages':pages,
            'pdf_bytes':args.output.stat().st_size,'pdf_sha256':pdfhash,'archive_sha256':ARCHIVE_SHA,
            'archive_members_verified':27,'audit_verification_sha256':data['audit_sha256'],
            'confidence_probe_sha256':data['probe_sha256'],'r50_manifest_sha256':R50_SHA,
            'native_gradient_guard_sha256':data['native_guard_sha256'],
            'r50_visual_observations_sha256':data['observations_sha256'],
            'original_image_sha256':data['image_hashes'],'misses_covered':15,'unique_miss_ids':15,
            'all_miss_reference_geometry_unchanged_r38_to_r50':True,'paragraph_bounds_pass':True,
            'paragraph_count':len(bounds),'minimum_body_bottom':min(p['bottom'] for p in bounds),
            'drawn_text_width_bounds_pass':True,'drawn_text_count':len(text_bounds),
            'new_inference_by_report':False,'new_training':False,'labels_modified':False,
            'visual_inspection':'See visual-verification.json for rendered-page review.'}
    }
    for name,value in records.items():(results/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'pages':pages,'pdf_sha256':pdfhash,'misses':15},indent=2))


if __name__=='__main__':main()
