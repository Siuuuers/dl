#!/usr/bin/env python3
"""Render the complete 157-photo atlas for the complete r8 visual pass.

Only exact field changes from the preserved r7 review are highlighted. The
renderer reads accepted coordinates; it does not choose annotation geometry.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from PIL import Image
from reportlab.lib.colors import white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from build_front_review_r6_pdf import (
    INK, MUTED, CYAN, MAGENTA, RED, AMBER, PAPER, WIDTH, HEIGHT,
    register_fonts, text, wrap as base_wrap, ranges as base_ranges, panel, number_word,
)
from evaluate_pose import parse_pose_labels
from verify_front_review_r8 import REVIEW, PARENT, read_review
from verify_reconciled_pose_labels import sha


def ranges(numbers):
    return base_ranges(numbers).replace("\u2013", "-")


def readable_prose(value):
    """Expand compact manifest prose without changing identifiers or geometry."""
    return re.sub(
        r'\b(CURRENT|current|Right|right|bottom|moves|to|about|proposed|complete|fresh)(?=\d)',
        r'\1 ', str(value),
    )


def wrap(c, value, *args, **kwargs):
    return base_wrap(c, readable_prose(value), *args, **kwargs)


def header(c, title, subtitle, page):
    c.setFillColor(white)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    text(c, 'CASSETTE FRONT REVIEW / ROUND 8 FOLLOW-UP', 30, HEIGHT - 29, 10, MUTED, True)
    size = 24
    while pdfmetrics.stringWidth(title, 'Review-Bold', size) > WIDTH - 60 and size > 18:
        size -= 1
    text(c, title, 30, HEIGHT - 60, size, INK, True)
    text(c, subtitle, 30, HEIGHT - 81, 11, MUTED)
    text(c, f'{page:03d}', WIDTH - 57, 21, 10, MUTED)
    text(c, 'Round 8 / Original camera pixels / CURRENT r8 rows unless marked BASE r7', 30, 21, 9, MUTED)


def line_count(value, width, size):
    if not value:
        return 0
    count, line = 1, ''
    for word in readable_prose(value).split():
        trial = (line + ' ' + word).strip()
        if line and pdfmetrics.stringWidth(trial, 'Review', size) > width:
            count += 1
            line = word
        else:
            line = trial
    return count


def definition_page(c, page):
    header(c, 'A box and a corner answer different questions',
           'The visible front defines the target. Marked screenshots guide review; original pixels support coordinates.', page)
    rules = [
        ('FOUR EXTREMES', 'Find the leftmost, highest, rightmost and lowest visible points of the front and its own frontal rim. Their extrema define the axis-aligned rectangle. A tilted contour can leave background wedges inside it. Rack, neighboring fronts and the rear perforated plane do not define its outer bounds.'),
        ('LEFT / RIGHT', 'Use the actual lower-left and lower-right corners of the front, ordered by image X. A corner is a physical junction, not automatically a box corner. Blur and rounded edges limit precision; no constant inset is applied to every cassette.'),
        ('UNAVAILABLE', 'A recognizable front may remain labeled with one or both points unavailable. Each unavailable point is exactly 0 0 0. Preserve its left/right slot, and infer neither hidden corners nor additional instances from regular stack spacing.'),
        ('RACK OR RIM', 'Trace connections into the surrounding mesh and through neighboring gaps. A continuous rail belongs to the rack; an own rim ends and turns with its cassette. Dark color alone is insufficient. A strip can be shaded cassette material. Do not use one shared inset for every front.'),
        ('COLOR AND SCOPE', 'Magenta marks only r8 changes from r7. Cyan is inherited, not newly certified. Amber marks unresolved fields. Changed fields take magenta priority; photo and detail notes identify any remaining held fields.'),
        ('OWNERSHIP HOLDS', 'The old-series pale flange belongs to the cassette. Its outer turn versus front-wall junction remains a landmark-definition question, preserved in the ledger. The confirmed old_000013 original bytes remain intact. Other clipping or occlusion holds remain explicit; a review pass does not silently clear them.'),
        ('TRAINING AND SCORING', 'The review atlas and the composed training labels are separate artifacts. Unresolved training photos retain exact PR35 supervision in both comparison arms where the protocol specifies a whole-photo hold. Development edits do not replace frozen validation or scoring references.'),
    ]
    y = HEIGHT - 132
    for label, body in rules:
        text(c, label, 30, y, 11, MUTED, True)
        y = wrap(c, body, 202, y, WIDTH - 237, 13.3, 19.8) - 17
    url = 'https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md'
    text(c, 'Corner definition: original baomai data README. Box-boundary details are the project review convention.', 30, 62, 10, MUTED)
    c.linkURL(url, (30, 57, WIDTH - 30, 76), relative=0, thickness=0)


def result_summary_page(c, page, summary):
    """Optional verified summary supplied by request-coverage, not inferred here."""
    header(c, 'What PR #74 measured',
           summary.get('subtitle', 'Returned round 7 experiment / Fixed 666-target primary reference; no r8 training result yet.'), page)
    columns = [
        ('Model / role', 'role', 30),
        ('Matched', 'matched', 330),
        ('Missed', 'missed', 495),
        ('Extra', 'extra', 625),
        ('Both corners <=5 px', 'pair_5', 760),
        ('Both corners <=10 px', 'pair_10', 960),
    ]
    y = HEIGHT - 151
    for label, _, x in columns:
        text(c, label, x, y, 12, MUTED, True)
    y -= 20
    c.setStrokeColor(MUTED); c.setLineWidth(.5); c.line(30, y, WIDTH - 30, y)
    for row in summary['rows']:
        y -= 47
        for _, key, x in columns:
            value = row[key]
            text(c, value, x, y, 17 if key == 'role' else 20, INK, key == 'role')
    y -= 55
    notes = summary.get('notes', [])
    if isinstance(notes, str):
        notes = [notes]
    for note in notes:
        y = wrap(c, note, 30, y, WIDTH - 60, 15, 23) - 17
    y = wrap(c, 'A training run completing successfully does not establish better detection. '
             'The frozen target, matching rule, endpoint tolerance and both missed and extra detections '
             'must be considered together. R8 prepares a separate annotation comparison and has no fresh training result.',
             30, y, WIDTH - 60, 14, 22)
    source = summary.get('source_url')
    if source:
        text(c, 'Source: PR #74 archive, fresh fixed-reference inference and the recorded evaluation.', 30, 62, 11, MUTED)
        c.linkURL(source, (30, 57, WIDTH - 30, 77), relative=0, thickness=0)


def extrema_example_page(c, page, image, before, after, fields, held, operation):
    """Show recorded witness coordinates; never derive new annotation positions."""
    header(c, 'Four visible extremes, two physical corners',
           'new_000001 / BASE row 22 / The screenshot is a guide; these positions use original camera pixels.', page)
    crop = (905, 16, 1135, 238)
    panel_width = (WIDTH - 90) / 2
    x1, x2, y, height = 30, 60 + panel_width, 260, HEIGHT - 390
    text(c, 'ORIGINAL CONTEXT', x1, HEIGHT - 117, 12, MUTED, True)
    text(c, 'CURRENT BOX + FOUR EXTREME WITNESSES', x2, HEIGHT - 117, 12, MUTED, True)
    panel(c, image, before, x1, y, panel_width, height, crop=crop, raw=True)
    panel(c, image, after, x2, y, panel_width, height, crop=crop, fields=fields, held=held)
    scale = min(panel_width / (crop[2] - crop[0]), height / (crop[3] - crop[1]))
    iw, ih = (crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale
    ox, oy = x2 + (panel_width - iw) / 2, y + (height - ih) / 2
    witnesses = operation['extreme_witnesses']
    for name in ('left', 'top', 'right', 'bottom'):
        px, py = witnesses[name]
        qx, qy = ox + (px - crop[0]) * scale, oy + ih - (py - crop[1]) * scale
        c.setStrokeColor(AMBER); c.setLineWidth(1.4)
        c.line(qx - 4, qy, qx + 4, qy); c.line(qx, qy - 4, qx, qy + 4)
        label = {'left': 'X min', 'top': 'Y min', 'right': 'X max', 'bottom': 'Y max'}[name]
        tx = qx + 6 if name != 'right' else qx - 40
        text(c, label, tx, qy + 5, 10, AMBER, True)
    columns = [('LEFT', 'left'), ('TOP', 'top'), ('RIGHT', 'right'), ('BOTTOM', 'bottom')]
    for index, (label, key) in enumerate(columns):
        x = 30 + index * (WIDTH - 60) / 4
        text(c, label, x, 223, 11, MUTED, True)
        px, py = witnesses[key]
        text(c, f'({px:g}, {py:g}) px', x, 200, 17, INK, True)
    body = ('Amber crosses are review witnesses, not extra training keypoints. Their X/Y extrema define the box. '
            'The circle and square retain their separate physical lower-corner meaning. '
            'In this example the top and right bounds expand to recover cassette material while the bottom removes rack margin; '
            'the right landmark moves to the actual front turn. A blanket inward shrink would be wrong.')
    wrap(c, body, 30, 157, WIDTH - 60, 14, 22)
    text(c, 'Nominal coordinates describe visible contours; decimal precision is not a physical accuracy claim.', 30, 62, 11, MUTED)
    return {'page': page, 'type': 'four_extrema_example', 'image_id': operation['image_id'],
            'base_row': operation['base_row'], 'crop': list(crop), 'witnesses': witnesses}


def build(root, output, audit_output):
    register_fonts()
    root = Path(root).resolve()
    review, status, records, *_ = read_review(root)
    by_id = {r['id']: r for r in records}
    ids = sorted(by_id)
    operations = json.loads((root / review['operations']['path']).read_text())
    maps = json.loads((root / review['row_maps']['path']).read_text())
    coverage = json.loads((root / review['request_coverage']['path']).read_text())
    parent = json.loads((root / PARENT['path']).read_text())
    before_files = {r['image_id']: r for r in parent['files']}
    after_files = {r['image_id']: r for r in review['files']}
    op_map, case_map = defaultdict(list), defaultdict(list)
    for op in operations:
        op_map[op['image_id']].append(op)
    for case in status['cases']:
        if case['status'] != 'resolved':
            case_map[case['image_id']].append(case)
    counts = Counter(o['action'] for o in operations)
    targeted = set(coverage.get('targeted_image_ids', list(op_map)))
    summary = coverage.get('pr74_summary')
    index_page = 5 if summary else 4
    first_photo_page = index_page + 1
    photo_page_numbers = {iid: i + first_photo_page for i, iid in enumerate(ids)}

    def data(iid):
        record = by_id[iid]
        image_path = root / 'data' / record['image']
        if sha(image_path) != record['image_sha256']:
            raise ValueError(f'Original image hash changed: {iid}')
        image = Image.open(image_path).convert('RGB')
        before_path = (root / PARENT['path']).parent / before_files[iid]['label_path']
        after_path = (root / REVIEW['path']).parent / after_files[iid]['label_path']
        before = parse_pose_labels(before_path.read_text(), *image.size, str(before_path))
        after = parse_pose_labels(after_path.read_text(), *image.size, str(after_path))
        edits = {o['base_row']: o for o in op_map[iid] if o['action'] != 'add'}
        fields, held, held_before = {}, {}, {}
        for case in case_map[iid]:
            for base in case['base_rows']:
                held_before.setdefault(base, set()).update(case.get('fields', ['box', 'left', 'right']))
        for identity in maps[iid]['rows']:
            row, base = identity['current_row'], identity.get('base_row')
            if identity.get('addition_id'):
                fields[row] = ['box', 'left', 'right']
            elif base in edits:
                fields[row] = edits[base]['changed_groups']
            flags = set(held_before.get(base, ()))
            for case in case_map[iid]:
                if (row in case.get('current_rows', []) or
                    identity.get('addition_id') in case.get('addition_ids', [])):
                    flags.update(case.get('fields', ['box', 'left', 'right']))
            if flags:
                held[row] = sorted(flags)
        return image, before, after, fields, held, {row: sorted(groups) for row, groups in held_before.items()}

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle('All 157 cassette photographs / Round 8 follow-up')
    c.setAuthor('Siuuuers/dl / original-image annotation review')
    c.setSubject('Complete visual pass from preserved r7. Only new r8 fields are highlighted; all 157 current label sets are shown.')
    pages = []
    header(c, 'The front ends where its own material ends',
           'Complete 157-photo atlas / Only round 8 changes from the preserved round 7 baseline are highlighted.', 1)
    text(c, f'{len(op_map)} changed photos', 30, HEIGHT - 137, 29, INK, True)
    text(c, f'{counts["replace"]} edited rows / {number_word(counts["add"], "addition")} / {number_word(counts["delete"], "deletion")}',
         445, HEIGHT - 133, 20, INK, True)
    y = HEIGHT - 181
    legends = [
        ('MAGENTA + *', 'The box, left point or right point changed in r8. Other fields on the same row keep their own colors.', MAGENTA),
        ('CYAN', 'Unchanged from r7. All photographs were inspected, but an unchanged field is not a claim of exact physical truth.', CYAN),
        ('AMBER', 'Unresolved physical ownership or landmark availability. A changed field is magenta; its remaining uncertainty is stated in the notes.', AMBER),
        ('CIRCLE / SQUARE', 'Image-left / image-right lower-front corner. An unavailable point has no marker.', MUTED),
    ]
    for label, body, color in legends:
        text(c, label, 30, y, 12, color, True)
        y = wrap(c, body, 230, y, WIDTH - 265, 14, 21) - 17
    y = wrap(c, f'This pass inspects all {len(targeted)} original photographs and their inherited annotations. '
             'Changed boundaries and uncertain regions are checked in enlarged original-pixel views. '
             'Four observed extrema support each changed box. This coverage does not certify every physical boundary.',
             30, y - 5, WIDTH - 60, 14, 22)
    empty_scenes = sum(v['empty_images'] for v in review['expected_counts'].values())
    y = wrap(c, f'The source cohort remains 137 training and 20 development photographs, including {empty_scenes} empty scenes. '
             f'The displayed review has {review["expected_counts"]["train"]["objects"]:,} training fronts and '
             f'{review["expected_counts"]["val"]["objects"]:,} development fronts. '
             'Use CURRENT r8 row numbers here and BASE r7 row numbers for before panels and operations.',
             30, y - 12, WIDTH - 60, 14, 22)
    wrap(c, 'No new training has been run by this follow-up. Corrected review labels, controlled training inputs '
         'and frozen scoring targets remain separately identified; no model improvement or promotion is implied.',
         30, y - 12, WIDTH - 60, 14, 22)
    text(c, 'Use the linked photo index or PDF bookmarks to jump to a photograph.', 30, 62, 12, MUTED)
    pages.append({'page': 1, 'type': 'cover'}); c.showPage()
    definition_page(c, 2)
    pages.append({'page': 2, 'type': 'physical_definition'}); c.showPage()
    example = next(o for o in operations if o['image_id'] == 'new_000001' and o.get('base_row') == 22)
    image, before, after, fields, held, _ = data(example['image_id'])
    pages.append(extrema_example_page(c, 3, image, before, after, fields, held, example)); c.showPage()
    if summary:
        result_summary_page(c, 4, summary)
        pages.append({'page': 4, 'type': 'prior_pr74_results', 'source': summary}); c.showPage()

    header(c, 'Photo index', '* = r8 change / ? = unresolved case / Numbers at right are PDF pages.', index_page)
    c.bookmarkPage('photo-index'); c.addOutlineEntry('Photo index', 'photo-index', level=0, closed=False)
    per_column = (len(ids) + 2) // 3
    column_width = (WIDTH - 60) / 3
    for i, iid in enumerate(ids):
        column, line = divmod(i, per_column)
        x, y = 30 + column * column_width, HEIGHT - 123 - line * 12.2
        suffix = (' *' if op_map.get(iid) else '') + (' ?' if case_map.get(iid) else '')
        text(c, iid + suffix, x, y, 9.3, INK, True)
        text(c, f'{after_files[iid]["objects"]} fronts', x + 159, y, 9.3, MUTED)
        text(c, photo_page_numbers[iid], x + column_width - 43, y, 9.3, MUTED)
        c.linkRect('', 'photo-' + iid, (x, y - 3, x + column_width - 17, y + 10), relative=0, thickness=0)
    pages.append({'page': index_page, 'type': 'photo_index'}); c.showPage()

    for iid in ids:
        number = photo_page_numbers[iid]
        image, before, after, fields, held, _ = data(iid)
        actions = Counter(o['action'] for o in op_map[iid])
        split_name = 'TRAIN REVIEW' if by_id[iid]['split'] == 'train' else 'DEVELOPMENT REVIEW / EXPLORATORY'
        scope = 'R8 ORIGINAL-PIXEL REVIEW' if iid in targeted else 'R7 REVIEW CARRIED FORWARD'
        unresolved = len(case_map[iid])
        header(c, iid, f'{split_name} / {scope} / All {len(after)} current fronts / '
               f'{number_word(actions["replace"], "edit")}, {number_word(actions["add"], "addition")} / '
               f'{number_word(unresolved, "unresolved case")}', number)
        c.bookmarkPage('photo-' + iid); c.addOutlineEntry(iid, 'photo-' + iid, level=0, closed=False)
        edited = [o['base_row'] for o in op_map[iid] if o['action'] == 'replace']
        caption = 'Changed BASE r7 rows: ' + ranges(edited) + '.'
        if actions['delete']:
            caption += ' Deleted BASE rows: ' + ranges(maps[iid]['deleted_base_rows']) + '.'
        additions = [row['current_row'] for row in maps[iid]['rows'] if row.get('addition_id')]
        if additions:
            caption += ' Added CURRENT r8 rows: ' + ranges(additions) + '.'
        if held:
            caption += ' Unresolved CURRENT rows: ' + ranges(held) + '.'
        if any(not item['base_rows'] for item in case_map[iid]):
            caption += ' An additional possible-front region remains unresolved.'
        if not after:
            caption = 'Reviewed empty-rack scene. Its empty label file is preserved; no identifiable cassette front was found.'
        note = coverage.get('photo_notes', {}).get(iid, '')
        if isinstance(note, list):
            note = ' '.join(note)
        if iid not in targeted and not note:
            note = 'Complete current labeling is reproduced from the preserved r7 review. No fresh geometry decision was made on this photograph in r8.'
        caption_height = line_count(caption, WIDTH - 60, 10.5) * 15
        note_height = line_count(note, WIDTH - 60, 10) * 14.5
        image_bottom = max(118, 48 + caption_height + note_height + 10)
        image_top = HEIGHT - 99
        shown = panel(c, image, after, 30, image_bottom, WIDTH - 60,
                      image_top - image_bottom, fields=fields, held=held)
        y = wrap(c, caption, 30, image_bottom - 19, WIDTH - 60, 10.5, 15)
        if note:
            end_y = wrap(c, note, 30, y - 3, WIDTH - 60, 10, 14.5, MUTED)
            if end_y < 32:
                raise ValueError(f'Photo notes collide with footer: {iid}')
        text(c, 'Photo index', WIDTH - 162, 21, 9, MUTED)
        c.linkRect('', 'photo-index', (WIDTH - 165, 16, WIDTH - 92, 33), relative=0, thickness=0)
        pages.append({'page': number, 'type': 'complete_photo', 'image_id': iid,
                      'followup_reviewed': iid in targeted, **shown,
                      'deleted_base_rows': maps[iid]['deleted_base_rows'],
                      'added_current_rows': additions,
                      'changed_current_fields': {str(k): list(v) for k, v in sorted(fields.items())},
                      'source_image_sha256': by_id[iid]['image_sha256'],
                      'base_label_sha256': before_files[iid]['label_sha256'],
                      'label_sha256': after_files[iid]['label_sha256']})
        c.showPage()

    for detail in coverage.get('pdf_details', []):
        number = len(pages) + 1
        iid, crop = detail['image_id'], detail['crop']
        image, before, after, fields, held, held_before = data(iid)
        header(c, detail['title'], iid + ' / Unmodified original pixels, BASE r7 labels and CURRENT r8 labels.', number)
        panel_width = (WIDTH - 90) / 3
        note = detail['note']
        if isinstance(note, list):
            note = ' '.join(note)
        note_height = line_count(note, WIDTH - 60, 14) * 21
        panel_bottom = max(180, 70 + note_height)
        panel_top = HEIGHT - 139
        for x, label in [(30, 'ORIGINAL'), (45 + panel_width, 'BEFORE / BASE r7'),
                         (60 + 2 * panel_width, 'CURRENT r8')]:
            text(c, label, x, HEIGHT - 119, 12, MUTED, True)
        raw_panel = panel(c, image, before, 30, panel_bottom, panel_width,
                          panel_top - panel_bottom, crop=crop, raw=True)
        before_panel = panel(c, image, before, 45 + panel_width, panel_bottom, panel_width,
                             panel_top - panel_bottom, crop=crop, held=held_before,
                             deleted=maps[iid]['deleted_base_rows'])
        after_panel = panel(c, image, after, 60 + 2 * panel_width, panel_bottom, panel_width,
                            panel_top - panel_bottom, crop=crop, fields=fields, held=held)
        focus = set(detail.get('base_rows', []))
        if not focus <= set(before_panel['displayed_rows']) or not focus <= set(after_panel['displayed_rows']):
            raise ValueError(f'Advertised detail rows are absent from crop: {iid}')
        end_y = wrap(c, note, 30, panel_bottom - 30, WIDTH - 60, 14, 21)
        if end_y < 38:
            raise ValueError(f'Detail note collides with footer: {detail["title"]}')
        c.bookmarkPage(f'detail-{number}')
        c.addOutlineEntry(iid + ': ' + detail['title'], f'detail-{number}', level=0, closed=False)
        pages.append({'page': number, 'type': 'original_before_current_detail',
                      'image_id': iid, 'crop': crop, 'base_rows': detail.get('base_rows', []),
                      'raw': raw_panel, 'before': before_panel, 'current': after_panel,
                      'source_image_sha256': by_id[iid]['image_sha256'],
                      'base_label_sha256': before_files[iid]['label_sha256'],
                      'label_sha256': after_files[iid]['label_sha256']})
        c.showPage()
    c.save()
    photos = [p for p in pages if p['type'] == 'complete_photo']
    assert len(photos) == 157 and {p['image_id'] for p in photos} == set(ids)
    assert all(p['displayed_rows'] == list(range(1, p['objects'] + 1)) for p in photos)
    assert [p['page'] for p in pages] == list(range(1, len(pages) + 1))
    audit = {'schema_version': 1, 'kind': 'all157_front_review_r8_followup_pdf',
             'review': REVIEW, 'parent': PARENT, 'pages': pages, 'page_count': len(pages),
             'complete_photo_pages': 157, 'targeted_followup_image_ids': sorted(targeted),
             'all_current_rows_displayed': True, 'field_level_change_highlighting': True,
             'highlight_comparison': 'r8 versus preserved r7 only',
             'fresh_all_row_review_claimed': True, 'original_images_unchanged': True,
             'pdf_sha256': sha(output), 'pdf_bytes': output.stat().st_size,
             'visual_layout_qa': 'pending separate rendered-page inspection'}
    audit_output = Path(audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2) + '\n')
    return {k: v for k, v in audit.items() if k != 'pages'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output, args.audit_output), indent=2))


if __name__ == '__main__':
    main()
