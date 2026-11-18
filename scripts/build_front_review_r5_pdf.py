#!/usr/bin/env python3
"""Render the r5 changes with every current label on each changed photograph."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from PIL import Image
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from build_front_review_r4_pdf import (
    INK, MUTED, CYAN, MAGENTA, RED, AMBER, PAPER, WIDTH, HEIGHT,
    register_fonts, text, wrap, ranges, panel,
)
from evaluate_pose import parse_pose_labels
from verify_front_review_r5 import REVIEW, PARENT, read_review
from verify_reconciled_pose_labels import sha


def header(c, title, subtitle, number):
    c.setFillColor(white)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    text(c, 'CASSETTE FRONT REVIEW / ROUND 5', 30, HEIGHT - 30, 10, MUTED, True)
    text(c, title, 30, HEIGHT - 62, 24, INK, True)
    text(c, subtitle, 30, HEIGHT - 83, 11, MUTED)
    text(c, f'{number:02d}', WIDTH - 56, 22, 10, MUTED)
    text(c, '19 November 2026 / Geometry is measured in the original camera image.', 30, 22, 9, MUTED)


def raw_panel(c, image, crop, x, y, w, h):
    a, b, d, e = crop
    scale = min(w / (d - a), h / (e - b))
    iw, ih = (d - a) * scale, (e - b) * scale
    c.drawImage(ImageReader(image.crop(crop)), x + (w - iw) / 2,
                y + (h - ih) / 2, iw, ih)


def definition_page(c, number):
    header(c, 'Two physical corners and one visible envelope',
           'The existing front-face task / Illustration is schematic, not an additional annotation.', number)
    x, y = 64, HEIGHT - 363
    c.setFillColor(PAPER); c.rect(30, y - 22, 490, 247, fill=1, stroke=0)
    vertices = [(x + 35, y + 180), (x + 410, y + 145),
                (x + 365, y + 25), (x + 10, y + 72)]
    outline = c.beginPath(); outline.moveTo(*vertices[0])
    for vertex in vertices[1:]: outline.lineTo(*vertex)
    outline.close()
    c.setFillColor(HexColor('#c5d6dd')); c.setStrokeColor(INK); c.setLineWidth(1.2)
    c.drawPath(outline, fill=1, stroke=1)
    c.setStrokeColor(CYAN); c.setLineWidth(2); c.setDash(5, 3)
    c.rect(x + 10, y + 25, 400, 155, fill=0, stroke=1); c.setDash()
    c.setFillColor(MAGENTA); c.setStrokeColor(INK)
    c.circle(*vertices[3], 5, fill=1, stroke=1)
    c.rect(vertices[2][0] - 5, vertices[2][1] - 5, 10, 10, fill=1, stroke=1)
    text(c, 'Left lower-front junction', x + 8, y + 3, 11, INK, True)
    text(c, 'Right junction', x + 310, y - 14, 11, INK, True)
    wrap(c, 'The rectangle contains the visible front. Its corners need not coincide with the two physical lower-front junctions.',
         555, HEIGHT - 154, WIDTH - 590, 17, 26)
    wrap(c, 'Perspective explains this geometric difference. It does not justify an inset point, a point on a groove, or a box cutting off the visible rim. Those are separately reviewable errors.',
         555, HEIGHT - 255, WIDTH - 590, 14, 22)
    y = HEIGHT - 422
    items = [
        ('FRONT MATERIAL', 'Include this front\'s own visible rim. Exclude another cassette, upper surface, projecting tray, rack or gripper from the target\'s outer extent.'),
        ('OCCLUSION', 'A rectangle may span an occluder between two visible patches of one front. Keep one instance when their ownership is supported; do not split solely to avoid interior rack pixels.'),
        ('UNAVAILABLE POINTS', 'An identifiable front stays labeled even if one or both corners cannot be seen. Each unavailable slot is 0 0 0. Do not infer a coordinate from stack spacing.'),
        ('FILTERS', 'Original pixels are the evidence. A grayscale, zero-cutoff autocontrast view can clarify an existing transition. It cannot recover an occluded corner. Training images remain unchanged.'),
        ('CHECKS', 'Exact replay, geometry and PDF checks establish consistency of the saved decisions. They do not certify physical edge ownership or eliminate blur-related uncertainty.'),
    ]
    for label, explanation in items:
        text(c, label, 30, y, 11, MUTED, True)
        y = wrap(c, explanation, 205, y, WIDTH - 240, 13, 20) - 17
    text(c, 'Project rule: docs/annotation-policy.md / Explanation: docs/front-labeling-guide.md', 30, 58, 10, MUTED)


def build(root, output, audit_output):
    register_fonts()
    root = Path(root).resolve()
    review, status, records, _, _, _, _ = read_review(root)
    by_id = {item['id']: item for item in records}
    operations = json.loads((root / review['operations']['path']).read_text())
    maps = json.loads((root / review['row_maps']['path']).read_text())
    coverage = json.loads((root / review['request_coverage']['path']).read_text())
    parent = json.loads((root / PARENT['path']).read_text())
    parent_files = {item['image_id']: item for item in parent['files']}
    current_files = {item['image_id']: item for item in review['files']}
    op_map = defaultdict(list)
    for op in operations: op_map[op['image_id']].append(op)
    changed_ids = sorted(op_map)
    counts = Counter(op['action'] for op in operations)
    held = [case for case in status['cases'] if case['split'] == 'train' and case['status'] != 'resolved']

    def data(image_id):
        record = by_id[image_id]
        image_path = root / 'data' / record['image']
        if sha(image_path) != record['image_sha256']:
            raise ValueError(f'original image hash differs: {image_id}')
        image = Image.open(image_path).convert('RGB')
        before_path = (root / PARENT['path']).parent / parent_files[image_id]['label_path']
        after_path = (root / REVIEW['path']).parent / current_files[image_id]['label_path']
        before = parse_pose_labels(before_path.read_text(), *image.size, str(before_path))
        after = parse_pose_labels(after_path.read_text(), *image.size, str(after_path))
        touched = {item['current_row'] for item in maps[image_id]['rows'] if item['changed']}
        return image, before, after, touched

    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle('Cassette front changes / Round 5 / 19 November 2026')
    c.setAuthor('Siuuuers/dl / assisted original-image review')
    c.setSubject('All labels on changed photos; magenta marks this pass. Original pixels and exact row identities retained.')
    pages, number = [], 1
    header(c, 'The changes in this pass', '17 requested photographs / Complete current labels and focused evidence.', number)
    text(c, f'{len(changed_ids)} changed photos', 30, HEIGHT - 133, 28, INK, True)
    text(c, f'{counts["replace"]} edited rows / {counts["add"]} added front / {counts["delete"]} deleted rows',
         430, HEIGHT - 129, 20, INK, True)
    y = wrap(c, 'Cyan = inherited row. Magenta and * = row edited or added in this pass. Amber = unresolved row. '
             'Circle = image-left lower corner; square = image-right lower corner. No marker means that corner is unavailable. '
             'Magenta marks the changed row; the operation ledger identifies the exact fields changed.',
             30, HEIGHT - 172, WIDTH - 60, 13, 20)
    y = wrap(c, 'Full-photo pages use CURRENT row numbers. Requests and operations use the immutable BASE numbers from the PR69 / round 4 labels. '
             'Base 11 and 12 in new_000014 are removed by the user\'s explicit decision. The visible front below new_000095 base 19 is added. '
             'Their detail pages explain the changed numbering.', 30, y - 7, WIDTH - 60, 12, 18)
    y = wrap(c, 'For new_000096, the descending request "28 to 20" is interpreted as rows 20-28. Nearby rows 18, 19, 29 and 30 were inspected as context; any supported extra edits are explicitly recorded. '
             'This is a focused review, not a certificate that every inherited label is correct.', 30, y - 7, WIDTH - 60, 12, 18)
    if held:
        y = wrap(c, 'Remaining training holds: ' + '; '.join(case['case_id'] for case in held)
                 + '. Exact PR35 supervision is retained on each whole held photo in both experiment arms.',
                 30, y - 7, WIDTH - 60, 12, 18)
    else:
        y = wrap(c, 'The prior new_000014 base 11 hold is closed by the explicit deletion request. No training-photo hold remains in this pass; inherited development uncertainty is recorded separately.',
                 30, y - 7, WIDTH - 60, 12, 18)
    y -= 20
    text(c, 'PHOTO INDEX', 30, y, 11, MUTED, True); y -= 27
    for index, image_id in enumerate(changed_ids):
        column, line = divmod(index, (len(changed_ids) + 1) // 2)
        xx, yy = 30 + column * 565, y - line * 24
        per = Counter(op['action'] for op in op_map[image_id])
        value = f'{image_id}     {per["replace"]} edit' + ('s' if per['replace'] != 1 else '')
        if per['add']: value += f' / +{per["add"]}'
        if per['delete']: value += f' / -{per["delete"]}'
        text(c, value, xx, yy, 12)
        text(c, str(index + 2), xx + 485, yy, 12, MUTED)
        c.linkRect('', f'photo-{image_id}', (xx, yy - 4, xx + 520, yy + 15), relative=0, thickness=0)
    pages.append({'page': number, 'type': 'index'}); c.showPage()

    for image_id in changed_ids:
        number += 1
        image, before, after, touched = data(image_id)
        per = Counter(op['action'] for op in op_map[image_id])
        summary = f'All {len(after)} current labels'
        for action, noun in [('replace', 'edit'), ('add', 'addition'), ('delete', 'deletion')]:
            summary += f' / {per[action]} {noun}' + ('s' if per[action] != 1 else '')
        header(c, image_id, summary, number)
        c.bookmarkPage(f'photo-{image_id}'); c.addOutlineEntry(image_id, f'photo-{image_id}', level=0, closed=False)
        held_base = {row for case in held if case['image_id'] == image_id for row in case['base_rows']}
        held_current = {item['current_row'] for item in maps[image_id]['rows'] if item.get('base_row') in held_base}
        shown = panel(c, image, after, 30, 116, WIDTH - 60, HEIGHT - 213, changed=touched, held=held_current)
        edited = [op['base_row'] for op in op_map[image_id] if op['action'] == 'replace']
        caption = 'Edited base rows: ' + ranges(edited) + '.'
        if per['delete']: caption += ' Deleted base rows: ' + ranges(maps[image_id]['deleted_base_rows']) + '.'
        if per['add']: caption += ' Added current rows: ' + ranges(item['current_row'] for item in maps[image_id]['rows'] if 'addition_id' in item) + '.'
        yy = wrap(c, caption, 30, 93, WIDTH - 60, 11, 17)
        notes = coverage.get('photo_notes', {}).get(image_id, [])
        if notes: wrap(c, ' '.join(notes), 30, yy - 3, WIDTH - 60, 10, 15, MUTED)
        pages.append({'page': number, 'type': 'complete_changed_photo', 'image_id': image_id, **shown,
                      'source_image_sha256': by_id[image_id]['image_sha256'], 'label_sha256': current_files[image_id]['label_sha256']})
        c.showPage()

    number += 1
    definition_page(c, number)
    pages.append({'page': number, 'type': 'physical_rule_illustration'}); c.showPage()

    details = coverage.get('pdf_details', [])
    for detail in details:
        number += 1
        image_id, base_rows = detail['image_id'], detail['base_rows']
        image, before, after, touched = data(image_id)
        relevant = [item['current_row'] for item in maps[image_id]['rows']
                    if item.get('base_row') in base_rows or item.get('after_base_row') in base_rows]
        boxes = [before[row - 1]['bbox_xyxy'] for row in base_rows] + [after[row - 1]['bbox_xyxy'] for row in relevant]
        crop = detail.get('crop') or (
            max(0, int(min(box[0] for box in boxes)) - 15), max(0, int(min(box[1] for box in boxes)) - 15),
            min(image.width, int(max(box[2] for box in boxes)) + 16), min(image.height, int(max(box[3] for box in boxes)) + 16))
        crop = tuple(map(int, crop))
        header(c, detail['title'], image_id + ' / Original pixels, immutable base rows and current labels.', number)
        pw = (WIDTH - 90) / 3
        for xx, label in [(30, 'ORIGINAL'), (45 + pw, 'BEFORE / BASE ROWS'), (60 + 2 * pw, 'CURRENT ROWS')]:
            text(c, label, xx, HEIGHT - 119, 12, MUTED, True)
        yy, ph = 205, HEIGHT - 345
        raw_panel(c, image, crop, 30, yy, pw, ph)
        deleted = set(maps[image_id]['deleted_base_rows'])
        panel(c, image, before, 45 + pw, yy, pw, ph, deleted=deleted, crop=crop)
        held_base = {row for case in held if case['image_id'] == image_id for row in case['base_rows']}
        held_current = {item['current_row'] for item in maps[image_id]['rows'] if item.get('base_row') in held_base}
        panel(c, image, after, 60 + 2 * pw, yy, pw, ph, changed=touched, held=held_current, crop=crop)
        yy = wrap(c, detail['note'], 30, 174, WIDTH - 60, 14, 21)
        identities = []
        for row in base_rows:
            if row in deleted: identities.append(f'base {row} removed')
        for item in maps[image_id]['rows']:
            if item.get('base_row') in base_rows or item.get('after_base_row') in base_rows:
                identities.append(f'base {item["base_row"]} -> current {item["current_row"]}' if 'base_row' in item
                                  else f'added -> current {item["current_row"]}')
        wrap(c, 'Row mapping: ' + '; '.join(identities) + '.', 30, yy - 8, WIDTH - 60, 12, 18, MUTED)
        pages.append({'page': number, 'type': 'original_before_current_detail', 'image_id': image_id,
                      'base_rows': base_rows, 'current_rows': relevant, 'crop': list(crop)})
        c.showPage()

    number += 1
    header(c, 'One physical-feature question remains',
           'new_000023 / Base rows 24-26 pending / These alternatives are review proposals, not accepted new targets.', number)
    question_record = root / 'results/review/front-review-20261119-r5/independent-corner-review.json'
    alternatives = json.loads(question_record.read_text())
    question, *_ = data('new_000023')
    crop = (790, 330, 1015, 500)
    panel_w, panel_h, panel_y = WIDTH / 2 - 45, HEIGHT - 345, 205
    text(c, 'ORIGINAL', 30, HEIGHT - 119, 12, MUTED, True)
    text(c, 'INHERITED POINT / UNACCEPTED OUTER PROPOSAL', WIDTH / 2 + 15, HEIGHT - 119, 11, MUTED, True)
    raw_panel(c, question, crop, 30, panel_y, panel_w, panel_h)
    xx = WIDTH / 2 + 15
    raw_panel(c, question, crop, xx, panel_y, panel_w, panel_h)
    scale = min(panel_w / (crop[2] - crop[0]), panel_h / (crop[3] - crop[1]))
    ox = xx + (panel_w - (crop[2] - crop[0]) * scale) / 2
    oy = panel_y + (panel_h - (crop[3] - crop[1]) * scale) / 2
    for case in alternatives['cases']:
        row = case['r4_base_row']
        if row not in (24, 25, 26):
            continue
        for key, label, color, dy in [
            ('existing_point_pixels', 'inset', CYAN, 10),
            ('proposed_point_pixels', 'outer?', HexColor('#ff9d00'), -15),
        ]:
            px, py = case[key]
            qx, qy = ox + (px - crop[0]) * scale, oy + (crop[3] - py) * scale
            c.setFillColor(color); c.setStrokeColor(INK); c.setLineWidth(.65)
            c.rect(qx - 3, qy - 3, 6, 6, fill=1, stroke=1)
            value = f'{row} {label}'
            c.setFillColor(INK); c.rect(qx - 79, qy + dy - 3, 72, 16, fill=1, stroke=0)
            text(c, value, qx - 76, qy + dy, 10, color, True)
    y = wrap(c, 'Cyan shows the inherited inset points. Orange shows proposed farther gray tips. The gray material belongs to the cassette, but this image does not settle whether its tip is the intended lower-front corner or lateral/projecting material. Rows 24-26 remain amber; neither alternative is certified.',
             30, 160, WIDTH - 60, 13, 20)
    wrap(c, 'The lowest row 27 has a more clearly observed rounded lower-front termination and is corrected. Until rows 24-26 are physically adjudicated, the next notebook retains the entire photograph\'s exact PR35 labels in both arms. Supported review edits on this photo remain visible in this PDF.',
         30, y - 7, WIDTH - 60, 12, 18)
    pages.append({'page': number, 'type': 'pending_landmark_question', 'image_id': 'new_000023',
                  'pending_base_rows': [24, 25, 26], 'original_crop_xyxy': list(crop),
                  'source_image_sha256': by_id['new_000023']['image_sha256'],
                  'alternative_coordinates_record_sha256': sha(question_record),
                  'accepted_row27_excluded_from_proposal_markers': True, 'right_marker_shape': 'square'})
    c.showPage(); c.save()
    audit = {'schema_version': 1, 'kind': 'focused_front_review_r5_pdf', 'review': REVIEW,
             'pdf_filename': output.name, 'pdf_sha256': sha(output), 'pages': pages,
             'changed_images': len(changed_ids), 'operation_counts': dict(counts),
             'all_current_labels_shown_on_each_changed_photo': True,
             'original_image_bytes_hash_verified': True, 'geometry_is_rendered_from_verified_labels': True,
             'generator_sha256': sha(Path(__file__)),
             'shared_renderer_sha256': sha(root / 'scripts/build_front_review_r4_pdf.py'), 'training_run': False}
    Path(audit_output).write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'pdf': str(output), 'pages': len(pages), 'bytes': output.stat().st_size, 'audit': str(audit_output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-output', type=Path, required=True)
    args = parser.parse_args(); build(args.root, args.output, args.audit_output)


if __name__ == '__main__':
    main()
