#!/usr/bin/env python3
"""Render changed photographs with exact fields and immutable parent row identities.

Run after the r16 review manifest and ledger have been built:
    python scripts/build_front_review_r16_pdf.py --root .

Optional --output and --qa-output paths select the PDF and its audit receipt.
All current labels are displayed on each included photo. Deleted parent identities
are dashed Dn ghosts, never current rows or training annotations. Original PNG
pixels, manifest hashes and exact parent-to-current operation replay are checked.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from PIL import Image
from reportlab.lib.colors import white
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from front_review_experiment import apply_operations
from build_front_review_r9_pdf import (
    WIDTH, HEIGHT, INK, MUTED, CYAN, MAGENTA, AMBER, GROUPS,
    TAG_SIZE, TAG_HEIGHT, digest, read_json, register_fonts, text,
    lines_for, wrap, ranges, parse_line, parse_labels, place_tags, intersects,
)

DEFAULT_REVIEW = 'data/reviews/fronts-2026-11-22-r16'
DEFAULT_LEDGER = 'results/review/front-review-20261122-r16'
PROVENANCE = ('Only new-series photographs with nonempty saved labels changed in round {round} are displayed, with every current front. '
              'Unchanged photographs, old-series photographs and empty scenes are excluded from this PDF only. '
              'Magenta identifies fields changed since round {parent}; cyan identifies retained fields. '
              'Original PNG pixels and exact r{parent} row provenance are preserved.')
INDEX_ROWS_PER_COLUMN = 17
INDEX_COLUMNS = 2
INDEX_PAGE_CAPACITY = INDEX_ROWS_PER_COLUMN * INDEX_COLUMNS


def load_inputs(root, review_root, ledger_root):
    def bound(descriptor):
        path = (root / descriptor['path']).resolve()
        if not path.is_relative_to(root) or digest(path) != descriptor['sha256']:
            raise ValueError(f'Input descriptor differs: {descriptor}')
        return path

    manifest_path = review_root / 'manifest.json'
    manifest = read_json(manifest_path)
    parent_path = bound(manifest['parent'])
    parent = read_json(parent_path)
    def review_round(value):
        match = re.fullmatch(r'fronts-\d{4}-\d{2}-\d{2}-r(\d+)', value)
        if not match:
            raise ValueError('Expected a dated front review identity')
        return int(match[1])
    round_number = review_round(manifest.get('review_id', ''))
    parent_round = review_round(parent.get('review_id', ''))
    if round_number != parent_round + 1:
        raise ValueError('Expected the immediately preceding immutable review')
    source_path = root / 'data/manifest.json'
    if digest(source_path) != manifest['source_manifest_sha256']:
        raise ValueError('Original source manifest differs')
    source = read_json(source_path)
    records = {r['id']: r for r in source['records']}
    ids = manifest['pdf_image_ids']
    changed_ids = manifest['changed_image_ids']
    if ids != sorted(set(ids)) or not ids or any(not i.startswith('new_') or i not in records for i in ids):
        raise ValueError('The PDF inventory must contain sorted unique new-series photographs')
    if changed_ids != sorted(set(changed_ids)) or any(not i.startswith('new_') or i not in records for i in changed_ids):
        raise ValueError('Changed photograph identities must be sorted unique new-series photographs')
    listed = {f['image_id']: f for f in manifest['files']}
    parent_files = {f['image_id']: f for f in parent['files']}
    if set(listed) != set(parent_files) or set(listed) != set(records):
        raise ValueError('Review/parent/source photograph inventories differ')
    operations_path = bound(manifest['operations'])
    status_path = bound(manifest['review_status'])
    maps_path = bound(manifest['row_maps'])
    if any(p.parent != ledger_root for p in (operations_path, status_path, maps_path)):
        raise ValueError('Review manifest does not name the supplied ledger')
    operations = read_json(operations_path)
    if not isinstance(operations, list):
        operations = operations['operations']
    by_image = defaultdict(list)
    for op in operations:
        if op['image_id'] not in changed_ids:
            raise ValueError('An operation changes a photograph outside the declared changes')
        by_image[op['image_id']].append(op)
    if changed_ids != sorted(by_image):
        raise ValueError('Changed photographs differ from the exact operation inventory')
    maps = read_json(maps_path)
    files, edits, deleted, identity_maps = {}, {}, {}, {}
    numeric_only_differences = []
    for iid, record in records.items():
        f, old = listed[iid], parent_files[iid]
        path = review_root / f['label_path']
        old_path = parent_path.parent / old['label_path']
        if f['label_path'] != record['label'] or f['split'] != record['split']:
            raise ValueError(f'Current source mapping differs: {iid}')
        if f['source_image_sha256'] != record['image_sha256']:
            raise ValueError(f'Current source image identity differs: {iid}')
        if digest(path) != f['label_sha256'] or digest(old_path) != old['label_sha256']:
            raise ValueError(f'Label bytes differ from a manifest: {iid}')
        expected, row_map = apply_operations(old_path.read_bytes(), by_image[iid], iid)
        if (path.read_bytes() != old_path.read_bytes()) != (iid in changed_ids):
            raise ValueError(f'Declared changed-photo byte comparison differs: {iid}')
        if path.read_bytes() != expected or maps[iid] != row_map:
            raise ValueError(f'Exact parent-to-current replay or row identity differs: {iid}')
        raw, rows = parse_labels(path)
        if len(rows) != f['objects']:
            raise ValueError(f'Instance count differs: {iid}')
        files[iid] = {'path': path, 'split': record['split'], 'raw': raw,
                      'rows': rows, 'sha256': digest(path)}
        identities = {r['current_row']: r for r in row_map['rows']}
        base_to_current = {r['base_row']: r['current_row'] for r in row_map['rows'] if 'base_row' in r}
        additions = {r['addition_id']: r['current_row'] for r in row_map['rows'] if 'addition_id' in r}
        identity_maps[iid] = identities
        image_edits, image_deleted = defaultdict(set), []
        for op in by_image[iid]:
            action = op.get('action', 'replace')
            if action == 'delete':
                if op['base_row'] not in row_map['deleted_base_rows']:
                    raise ValueError(f'Deleted parent identity remains current: {iid}')
                image_deleted.append({'base_row': op['base_row'],
                                      'values': parse_line(op['before_line'], f'{iid}:deleted'),
                                      'reason': op['reason']})
                continue
            if action == 'add':
                row = additions[op['addition_id']]
                changed = set(GROUPS)
            elif action == 'replace':
                row = base_to_current[op['base_row']]
                before = parse_line(op['before_line'], f'{iid}:before')
                after = parse_line(op['after_line'], f'{iid}:after')
                changed = {g for g, sl in GROUPS.items() if before[sl] != after[sl]}
                declared = set(op.get('changed_groups', changed)) & set(GROUPS)
                if declared != changed:
                    numeric_only_differences.append({'image_id': iid, 'base_row': op['base_row'],
                                                     'declared': sorted(declared), 'actual': sorted(changed)})
            else:
                raise ValueError(f'Unsupported operation: {action}')
            after = parse_line(op['after_line'], f'{iid}:after')
            if rows[row - 1] != after or raw[row - 1] != op['after_line']:
                raise ValueError(f'Operation after-line differs from current row: {iid}:{row}')
            image_edits[row].update(changed)
        if image_edits:
            edits[iid] = dict(image_edits)
        if image_deleted:
            deleted[iid] = image_deleted
    expected_ids = sorted(iid for iid, item in files.items() if iid in changed_ids and iid.startswith('new_') and item['rows'])
    if ids != expected_ids:
        raise ValueError('PDF inventory must contain exactly changed nonempty new-series photographs once')
    status = read_json(status_path)
    cases = status if isinstance(status, list) else status['cases']
    notes_path = ledger_root / 'pdf-case-notes.json'
    display_notes, notes_descriptor = {}, None
    if notes_path.exists():
        notes = read_json(notes_path)
        if notes.get('source_review_status_sha256') != digest(status_path):
            raise ValueError('Optional PDF case notes are stale')
        display_notes = notes['case_notes']
        unresolved_ids = {c['case_id'] for c in cases if c['status'] != 'resolved'}
        if set(display_notes) != unresolved_ids:
            raise ValueError('Optional PDF case notes must cover every unresolved case exactly')
        notes_descriptor = {'path': notes_path.relative_to(root).as_posix(), 'sha256': digest(notes_path)}
    held, case_map = {}, defaultdict(list)
    for case in cases:
        if case['status'] == 'resolved':
            continue
        iid = case['image_id']
        if iid not in records or iid.startswith('old_'):
            raise ValueError('Unresolved case is outside the new-series review')
        lookup = {v['base_row']: k for k, v in identity_maps[iid].items() if 'base_row' in v}
        expected_rows = [lookup[r] for r in case['base_rows'] if r in lookup]
        rows = case.get('current_rows', expected_rows)
        if rows != expected_rows or len(rows) != len(case['base_rows']):
            raise ValueError(f'Unresolved case lost or changed its parent identity: {case["case_id"]}')
        for row in rows:
            held.setdefault(iid, {}).setdefault(row, set()).update(case['fields'])
        c = dict(case, current_rows=rows)
        if c['case_id'] in display_notes:
            c['short_reason'] = display_notes[c['case_id']]
        case_map[iid].append(c)
    amendment_notes_path = ledger_root / 'pdf-amendment-notes.json'
    amendment_notes = {}
    amendment_questions = {}
    amendment_notes_descriptor = None
    if amendment_notes_path.exists():
        display = read_json(amendment_notes_path)
        if display.get('source_review_manifest_sha256') != digest(manifest_path):
            raise ValueError('Optional PDF amendment notes are stale')
        amendment_notes = display['photo_notes']
        amendment_questions = display.get('pending_questions', {})
        if not set(amendment_questions).issubset(ids):
            raise ValueError('Pending amendment questions must name included photographs')
        if not set(amendment_notes).issubset(ids) or any(not isinstance(v, list) for v in amendment_notes.values()):
            raise ValueError('Amendment notes must name included photographs and list strings')
        amendment_notes_descriptor = {'path': amendment_notes_path.relative_to(root).as_posix(), 'sha256': digest(amendment_notes_path)}
    op_counts = Counter(op.get('action', 'replace') for op in operations)
    replacement_fields = Counter(g for op in operations if op.get('action', 'replace') == 'replace'
                                 for g, sl in GROUPS.items()
                                 if parse_line(op['before_line'], 'before')[sl] != parse_line(op['after_line'], 'after')[sl])
    summary = {'source_photos': len(records), 'included_photos': len(ids),
               'excluded_old_photos': sum(i.startswith('old_') for i in records),
               'omitted_new_photos': sum(i.startswith('new_') and i not in ids for i in records),
               'excluded_empty_new_photos': sum(i.startswith('new_') and not files[i]['rows'] for i in records),
               'empty_photos': sum(not files[i]['rows'] for i in ids),
               'positive_photos': sum(bool(files[i]['rows']) for i in ids),
               'changed_photos': sum(bool(by_image[i]) for i in ids),
               'excluded_unchanged_nonempty_new_photos': sum(i.startswith('new_') and bool(files[i]['rows']) and i not in changed_ids for i in records),
               'operation_counts': {a: op_counts[a] for a in ('replace', 'add', 'delete')},
               'changed_fields_in_replacements': {g: replacement_fields[g] for g in GROUPS},
               'highlighted_current_rows': sum(len(e) for e in edits.values()),
               'held_photos': sum(i in ids for i in case_map), 'held_cases': sum(len(cs) for i,cs in case_map.items() if i in ids),
               'held_training_photos': sum(files[i]['split'] == 'train' for i in case_map if i in ids),
               'total_held_photos_in_ledger': len(case_map),
               'total_held_cases_in_ledger': sum(len(cs) for cs in case_map.values()),
               'held_photos_omitted': sorted(i for i in case_map if i not in ids),
               'included_objects': sum(len(files[i]['rows']) for i in ids),
               'row_number_changes': sum(r != m.get('base_row', r) for i in ids for r, m in identity_maps[i].items())}
    return {'manifest': manifest, 'manifest_path': manifest_path, 'parent': manifest['parent'],
            'round_number': round_number, 'parent_round': parent_round,
            'records': records, 'files': files, 'ids': ids, 'edits': edits, 'deleted': deleted,
            'identities': identity_maps, 'held': held, 'cases': dict(case_map), 'summary': summary,
            'operations': operations, 'display_case_notes': notes_descriptor,
            'amendment_notes': amendment_notes, 'amendment_questions': amendment_questions,
            'display_amendment_notes': amendment_notes_descriptor,
            'numeric_only_differences': numeric_only_differences,
            'input_descriptors': {key: manifest[key] for key in ('operations', 'review_status', 'row_maps')}}


def draw_photo(c, image, rows, edits, held, identities, deleted, parent_round=15):
    area = (30, 145, WIDTH - 60, HEIGHT - 241)
    scale = min(area[2] / image.width, area[3] / image.height)
    iw, ih = image.width * scale, image.height * scale
    x, y = area[0] + (area[2] - iw) / 2, area[1] + (area[3] - ih) / 2
    bounds = (x, y, x + iw, y + ih)
    c.drawImage(ImageReader(image), x, y, iw, ih)

    def point(px, py):
        return x + px * scale, y + ih - py * scale

    def color(row, group):
        return MAGENTA if group in edits.get(row, ()) else AMBER if group in held.get(row, ()) else CYAN

    specs, keypoints, protected_fronts = [], [], []
    c.saveState()
    clip = c.beginPath(); clip.rect(x, y, iw, ih); c.clipPath(clip, stroke=0)
    for row, v in enumerate(rows, 1):
        left, top = (v[1] - v[3] / 2) * image.width, (v[2] - v[4] / 2) * image.height
        right, bottom = (v[1] + v[3] / 2) * image.width, (v[2] + v[4] / 2) * image.height
        qx, qy = point(left, bottom)
        if 'box' in edits.get(row, ()) and 'box' in held.get(row, ()):
            c.setStrokeColor(AMBER); c.setLineWidth(2)
            c.rect(qx, qy, (right - left) * scale, (bottom - top) * scale, stroke=1, fill=0)
        c.setStrokeColor(color(row, 'box'))
        c.setLineWidth(0.95 if 'box' in edits.get(row, ()) else 0.55)
        c.rect(qx, qy, (right - left) * scale, (bottom - top) * scale, stroke=1, fill=0)
        for side, start in (('left', 5), ('right', 8)):
            if not v[start + 2]:
                continue
            px, py = point(v[start] * image.width, v[start + 1] * image.height)
            keypoints.append((px, py)); radius = 2.0 if side in edits.get(row, ()) else 1.6
            c.setFillColor(color(row, side)); c.setStrokeColor(AMBER if side in held.get(row, ()) else INK)
            c.setLineWidth(0.7 if side in held.get(row, ()) else 0.35)
            if side == 'left':
                c.circle(px, py, radius, fill=1, stroke=1)
            else:
                c.rect(px - radius, py - radius, 2 * radius, 2 * radius, fill=1, stroke=1)
        identity = identities[row]
        added = 'addition_id' in identity
        base = identity.get('base_row')
        label = str(row)
        if base is not None and base != row:
            label += f'[{base}]'
        label += '+' if added else '*' if edits.get(row) else ''
        tag_width = pdfmetrics.stringWidth(label, 'Review-Bold', TAG_SIZE) + 5
        box_width, box_height = (right - left) * scale, (bottom - top) * scale
        if (box_width <= tag_width * 2 and box_height <= TAG_HEIGHT * 2) or (added and (box_width <= tag_width * 2 or box_height <= TAG_HEIGHT * 2)):
            protected_fronts.append({'current_row': row,
                                     'rect': [qx, qy, qx + box_width, qy + box_height]})
            keypoints.extend((qx + box_width * fx, qy + box_height * fy)
                             for fx in (0, .5, 1) for fy in (0, .5, 1))
        specs.append({'row': row, 'current_row': row, f'r{parent_round}_base_row': base,
                      'addition_id': identity.get('addition_id'), 'kind': 'current',
                      'label': label, 'anchor': list(point(left, top)),
                      'width': tag_width,
                      'box_height': max(0, box_height),
                      'tag_color': 'magenta' if edits.get(row) else 'amber' if held.get(row) else 'cyan',
                      'held': bool(held.get(row))})
    # Deleted annotations are ghost rectangles, never assigned a current row.
    for item in deleted:
        v, base = item['values'], item['base_row']
        left, top = (v[1] - v[3] / 2) * image.width, (v[2] - v[4] / 2) * image.height
        right, bottom = (v[1] + v[3] / 2) * image.width, (v[2] + v[4] / 2) * image.height
        qx, qy = point(left, bottom); rw, rh = (right - left) * scale, (bottom - top) * scale
        c.setStrokeColor(MAGENTA); c.setLineWidth(0.7); c.setDash(3, 2)
        c.rect(qx, qy, rw, rh, stroke=1, fill=0); c.setDash()
        # A short cross avoids drawing a large diagonal across nearby real fronts.
        cx, cy = qx + rw / 2, qy + rh / 2
        c.line(cx - 3, cy - 3, cx + 3, cy + 3); c.line(cx - 3, cy + 3, cx + 3, cy - 3)
        label = f'D{base}'
        specs.append({'row': -base, 'current_row': None, f'r{parent_round}_base_row': base, 'kind': 'deleted',
                      'label': label, 'anchor': list(point(left, top)),
                      'width': pdfmetrics.stringWidth(label, 'Review-Bold', TAG_SIZE) + 5,
                      'box_height': max(0, rh), 'tag_color': 'magenta', 'held': False})
    c.restoreState()
    tags, collisions, clipped = place_tags(specs, bounds, keypoints)
    covered_fronts = [region['current_row'] for region in protected_fronts
                      if any(intersects(region['rect'], tag['rect']) for tag in tags)]
    if covered_fronts:
        raise ValueError(f'Tag hides a protected small front: {covered_fronts}')
    palette = {'magenta': MAGENTA, 'amber': AMBER, 'cyan': CYAN}
    for tag in tags:
        if not tag['moved']:
            continue
        ax, ay = tag['anchor']; rect = tag['rect']
        ax = max(x + .6, min(ax, x + iw - .6)); ay = max(y + .6, min(ay, y + ih - .6))
        tx, ty = max(rect[0], min(ax, rect[2])), max(rect[1], min(ay, rect[3]))
        c.setStrokeColor(INK); c.setLineWidth(1.35); c.line(ax, ay, tx, ty)
        c.setStrokeColor(palette[tag['tag_color']]); c.setLineWidth(.65); c.line(ax, ay, tx, ty)
        c.setFillColor(palette[tag['tag_color']]); c.circle(ax, ay, .85, stroke=0, fill=1)
    for tag in tags:
        x0, y0, x1, y1 = tag['rect']
        c.setFillColor(INK); c.setStrokeColor(AMBER if tag['held'] else palette[tag['tag_color']])
        c.setLineWidth(.45); c.rect(x0, y0, x1 - x0, y1 - y0, fill=1, stroke=1)
        text(c, tag['label'], x0 + 2.5, y0 + 2.7, TAG_SIZE, palette[tag['tag_color']], True)
    return {'image_bounds_pt': list(bounds), 'tags': tags, 'tag_count': len(tags),
            'current_tag_count': len(rows), 'deletion_tag_count': len(deleted),
            'protected_small_front_regions': protected_fronts,
            'moved_tags': sum(t['moved'] for t in tags), 'tag_overlap_pairs': collisions,
            'clipped_tag_rows': clipped, 'displayed_rows': list(range(1, len(rows) + 1))}


def header(c, title, subtitle, page, data):
    c.setFillColor(white); c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    text(c, f'CASSETTE FRONT REVIEW / ROUND {data["round_number"]}', 30, HEIGHT - 29, 10, MUTED, True)
    text(c, title, 30, HEIGHT - 59, 24, INK, True)
    text(c, subtitle, 30, HEIGHT - 80, 10.5, MUTED)
    text(c, f'Original camera pixels / Current r{data["round_number"]} rows [r{data["parent_round"]} base] / Rendering does not move coordinates', 30, 20, 9, MUTED)
    text(c, f'{page:03d}', WIDTH - 56, 20, 10, MUTED)


def index_groups(data):
    return [data['ids'][start:start + INDEX_PAGE_CAPACITY]
            for start in range(0, len(data['ids']), INDEX_PAGE_CAPACITY)]


def intro(c, data, page_numbers, ids, page, index_count):
    s = data['summary']; counts = s['operation_counts']
    header(c, f'Changed photographs / Round {data["round_number"]}',
           'Only photos with saved label changes in this pass. Every current front on each included photo is shown.', page, data)
    text(c, f'{s["included_photos"]} photographs', 30, HEIGHT - 125, 27, INK, True)
    text(c, f'{s["included_objects"]} current fronts', 390, HEIGHT - 125, 27, INK, True)
    text(c, f'{s["highlighted_current_rows"]} edited rows', 775, HEIGHT - 125, 27, INK, True)
    case_word = "case" if s["held_cases"] == 1 else "cases"
    photo_word = "photograph" if s["held_photos"] == 1 else "photographs"
    text(c, f'{s["changed_photos"]} photos changed / {counts["replace"]} row replacements / {counts["add"]} additions / {counts["delete"]} removal{"" if counts["delete"] == 1 else "s"} / '
         f'{s["held_cases"]} displayed held {case_word} in {s["held_photos"]} {photo_word}', 30, HEIGHT - 148, 11)
    c.setFillColor(INK); c.roundRect(30, HEIGHT - 195, WIDTH - 60, 34, 4, fill=1, stroke=0)
    text(c, f'CYAN  unchanged vs r{data["parent_round"]}', 42, HEIGHT - 182, 10.5, CYAN, True)
    text(c, 'MAGENTA  changed fields', 352, HEIGHT - 182, 10.5, MAGENTA, True)
    text(c, '*  edited row    -  removal    ?  open question', 715, HEIGHT - 182, 10.5, white, True)
    text(c, 'Circle = left front corner. Square = right. Missing points have no marker; their row numbers are listed below each photograph.',
         30, HEIGHT - 214, 10)
    text(c, f'A dashed Dn marker shows a deleted r{data["parent_round"]} row. Bracketed numbers show r{data["parent_round"]} identities when current row numbers shift.',
         30, HEIGHT - 231, 10)
    text(c, 'PHOTO INDEX  /  click a photograph name', 30, HEIGHT - 254, 10, MUTED, True)
    width = (WIDTH - 60) / INDEX_COLUMNS
    index_rows = (len(ids) + INDEX_COLUMNS - 1) // INDEX_COLUMNS
    for i, iid in enumerate(ids):
        column, row = divmod(i, index_rows)
        x, y = 30 + column * width, HEIGHT - 274 - row * 23.0
        flags = (' *' if iid in data['edits'] else '') + (' -' if iid in data['deleted'] else '') + (' ?' if iid in data['cases'] or iid in data.get('amendment_questions', {}) else '')
        text(c, iid + flags, x, y, 12, INK, True)
        label = 'empty' if not data['files'][iid]['rows'] else f'{len(data["files"][iid]["rows"])} fronts'
        text(c, label, x + 170, y, 11, MUTED)
        text(c, str(page_numbers[iid]), x + width - 43, y, 11, MUTED)
        c.linkRect('', 'photo-' + iid, (x, y - 3, x + width - 17, y + 10), relative=0, thickness=0)
    if data.get('amendment_questions'):
        text(c, 'Pending amendment: ' + '; '.join(data['amendment_questions'].values()),
             30, 95, 10, INK)
    if s['held_photos_omitted']:
        text(c, 'Unchanged unresolved photographs remain in the saved ledger: ' + ', '.join(s['held_photos_omitted']) + '.',
             30, 79, 10, MUTED)
    text(c, 'Unchanged photos, old-series photos and empty scenes are omitted from this PDF only. The full saved dataset is retained.',
         30, 63, 10, MUTED)
    text(c, f'Cross-row anchors refer to immutable r{data["parent_round"]} rows. Index numbers link to photo pages; click the page header to return.',
         30, 47, 10, MUTED)


def row_trace(identities, parent_round):
    shifted = [(current, v['base_row']) for current, v in sorted(identities.items())
               if 'base_row' in v and current != v['base_row']]
    groups = []
    for current, base in shifted:
        if groups and current == groups[-1][-1][0] + 1 and base == groups[-1][-1][1] + 1:
            groups[-1].append((current, base))
        else:
            groups.append([(current, base)])
    return '; '.join(f'r{parent_round} {ranges(b for _, b in g)} -> {ranges(r for r, _ in g)}' for g in groups)


def footer(c, iid, data):
    rows = data['files'][iid]['rows']; edits = data['edits'].get(iid, {})
    identities = data['identities'][iid]; deleted = data['deleted'].get(iid, [])
    additions = [r for r, v in identities.items() if 'addition_id' in v]
    columns = [(30, 330), (385, 370), (780, WIDTH - 810)]
    text(c, 'CHANGED FIELDS / CURRENT ROWS', columns[0][0], 126, 9.5, MUTED, True)
    changed_lines = [f'{g.capitalize()}: {ranges(r for r, gs in edits.items() if g in gs)}' for g in GROUPS]
    if additions: changed_lines.append('Added current rows: ' + ranges(additions))
    if deleted: changed_lines.append(f'Deleted r{data["parent_round"]} rows: ' + ranges(d['base_row'] for d in deleted))
    text(c, 'POINT AVAILABILITY / ROW PROVENANCE', columns[1][0], 126, 9.5, MUTED, True)
    missing = {side: [i for i, v in enumerate(rows, 1) if not v[start + 2]]
               for side, start in (('left', 5), ('right', 8))}
    missing_lines = [f'0 0 0 left: {ranges(missing["left"])}', f'0 0 0 right: {ranges(missing["right"])}']
    trace = row_trace(identities, data['parent_round'])
    if trace: missing_lines.append(trace)
    text(c, 'REVIEW NOTES', columns[2][0], 126, 9.5, MUTED, True)
    notes = []
    for case in data['cases'].get(iid, []):
        prefix = 'Rows ' + ranges(case['current_rows']) + ' (' + '/'.join(case['fields']) + ')' if case['current_rows'] else 'Possible extra front'
        reason = case.get('short_reason') or case.get('reason') or 'Physical ownership remains unresolved.'
        notes.append(prefix + ': ' + reason)
    notes.extend(data.get('amendment_notes', {}).get(iid, []))
    if not notes:
        notes = ['No remaining held case on this photograph.']
    blocks = [changed_lines, missing_lines, notes]
    rendered = []
    for (x, width), values in zip(columns, blocks):
        size, leading = 9.2, 12.0
        value = '\n'.join(values)
        while len(lines_for(value, width, size)) * leading > 86 and size > 8.3:
            size -= .2; leading = size + 2.5
        lines = lines_for(value, width, size)
        overflow = len(lines) * leading > 86
        if overflow:
            allowed = int(86 / leading) - 1
            lines = lines[:allowed] + ['Full case details: review-status.json.']
        for n, line in enumerate(lines):
            text(c, line, x, 110 - n * leading, size,
                 MAGENTA if line.startswith('Deleted') or line.startswith('Added') else INK)
        rendered.append({'lines': lines, 'font_size': size, 'truncated': overflow})
    return {'changed_fields_block': rendered[0], 'availability_trace_block': rendered[1],
            'case_notes_block': rendered[2], 'missing_point_rows': missing,
            'added_current_rows': additions, f'deleted_r{data["parent_round"]}_rows': [d['base_row'] for d in deleted],
            'row_trace': trace}


def verify_saved_pdf(root, output, data):
    """Read back page navigation and compare decoded images with original PNG pixels."""
    import fitz
    from pypdf import PdfReader

    receipts = []
    reader = PdfReader(output)
    groups = index_groups(data)
    index_count = len(groups)
    page_refs = {(page.indirect_reference.idnum, page.indirect_reference.generation): number
                 for number, page in enumerate(reader.pages)}

    def actual_link_targets(page):
        result = []
        for annotation_ref in page.get('/Annots', []):
            annotation = annotation_ref.get_object()
            if annotation.get('/Subtype') != '/Link':
                raise ValueError('Unexpected non-link annotation in the review atlas')
            destination = annotation['/Dest'][0]
            result.append(page_refs[(destination.idnum, destination.generation)])
        return result

    with fitz.open(output) as document:
        if len(document) != len(data['ids']) + index_count:
            raise ValueError('Saved PDF page count differs from the changed-photo atlas')
        targets, offset = [], index_count
        index_for_photo = {}
        for index_page, ids in enumerate(groups):
            index_links = document[index_page].get_links()
            actual = actual_link_targets(reader.pages[index_page])
            if len(index_links) != len(ids) or actual != list(range(offset, offset + len(ids))):
                raise ValueError(f'Saved index links differ on index page {index_page + 1}')
            targets.extend(actual)
            offset += len(ids)
            for iid, link in zip(ids, index_links):
                hitbox = fitz.Rect(link['from'])
                hitbox.x0 -= 1; hitbox.y0 -= 2; hitbox.x1 += 1; hitbox.y1 += 2
                if iid not in document[index_page].get_text(clip=hitbox):
                    raise ValueError(f'Saved index link label differs: {iid}')
                index_for_photo[iid] = index_page
        if targets != list(range(index_count, len(document))):
            raise ValueError('Saved index links do not reach every photograph exactly in order')
        if any(abs(page.rect.width - WIDTH) > .02 or abs(page.rect.height - HEIGHT) > .02
               for page in document):
            raise ValueError('Saved PDF page geometry differs from A3 landscape')
        expected_bookmarks = [[1, f'Photo index {i + 1} of {index_count}', i + 1] for i in range(index_count)] + [
            [1, iid, number + index_count + 1] for number, iid in enumerate(data['ids'])]
        if document.get_toc() != expected_bookmarks:
            raise ValueError('Saved PDF bookmark identities or destinations differ')
        for page_index, iid in enumerate(data['ids'], index_count):
            page = document[page_index]
            title = page.get_text(clip=fitz.Rect(20, 30, 500, 70))
            if iid not in title:
                raise ValueError(f'Saved PDF photograph title differs: {iid}')
            if actual_link_targets(reader.pages[page_index]) != [index_for_photo[iid]]:
                raise ValueError(f'Saved PDF index return link differs: {iid}')
            images = page.get_images(full=True)
            if len(images) != 1:
                raise ValueError(f'Expected one original photograph image on page: {iid}')
            xref = images[0][0]
            pixmap = fitz.Pixmap(document, xref)
            path = root / 'data' / data['records'][iid]['image']
            with Image.open(path) as original:
                rgb = original.convert('RGB')
                expected_pixels = rgb.tobytes()
                expected_size = rgb.size
            if (pixmap.alpha or pixmap.n != 3 or
                    (pixmap.width, pixmap.height) != expected_size or
                    pixmap.samples != expected_pixels):
                raise ValueError(f'Saved embedded RGB pixels differ from original PNG: {iid}')
            receipts.append({'page': page_index + 1, 'image_id': iid, 'image_xref': xref,
                             'original_image_sha256': digest(path),
                             'decoded_rgb_sha256': hashlib.sha256(expected_pixels).hexdigest(),
                             'width': pixmap.width, 'height': pixmap.height,
                             'decoded_rgb_matches_original_png': True})
    return {'all_decoded_rgb_pixels_match_original_pngs': True,
            'all_page_ids_and_index_links_verified': True,
            'index_page_count': index_count,
            'index_destination_pages': [target + 1 for target in targets],
            'all_bookmark_destinations_verified': True, 'embedded_image_receipts': receipts}


def build(root, review_root, ledger_root, output, qa_output):
    register_fonts()
    data = load_inputs(root, review_root, ledger_root)
    provenance = PROVENANCE.format(round=data['round_number'], parent=data['parent_round'])
    groups = index_groups(data)
    index_count = len(groups)
    page_numbers = {iid: n + index_count + 1 for n, iid in enumerate(data['ids'])}
    index_for_photo = {iid: n + 1 for n, ids in enumerate(groups) for iid in ids}
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle(f'Cassette front review r{data["round_number"]} - changed photographs')
    c.setAuthor('Siuuuers/dl'); c.setSubject(provenance)
    for n, ids in enumerate(groups, 1):
        c.bookmarkPage(f'index-{n}'); c.addOutlineEntry(f'Photo index {n} of {index_count}', f'index-{n}', level=0)
        intro(c, data, page_numbers, ids, n, index_count); c.showPage()
    pages = []
    for iid in data['ids']:
        record, item = data['records'][iid], data['files'][iid]
        image_path = root / 'data' / record['image']
        if digest(image_path) != record['image_sha256']:
            raise ValueError(f'Original image changed: {iid}')
        with Image.open(image_path) as original:
            image = original.convert('RGB')
        if list(image.size) != record['original_size']:
            raise ValueError(f'Original image dimensions differ: {iid}')
        edits, held = data['edits'].get(iid, {}), data['held'].get(iid, {})
        deleted = data['deleted'].get(iid, [])
        split = 'TRAIN REVIEW' if item['split'] == 'train' else 'DEVELOPMENT REVIEW'
        title_note = f'{split} / {len(item["rows"])} current fronts / {len(edits)} marked current row{"" if len(edits) == 1 else "s"}'
        if deleted: title_note += f' / {len(deleted)} deleted r{data["parent_round"]} row{"" if len(deleted) == 1 else "s"}'
        open_cases = len(data['cases'].get(iid, []))
        title_note += f' / {open_cases} held case{"" if open_cases == 1 else "s"}'
        if iid in data.get('amendment_questions', {}):
            title_note += ' / 1 pending amendment'
        header(c, iid, title_note, page_numbers[iid], data)
        c.bookmarkPage('photo-' + iid); c.addOutlineEntry(iid, 'photo-' + iid, level=0)
        c.linkRect('', f'index-{index_for_photo[iid]}', (30, HEIGHT - 33, 310, HEIGHT - 17), relative=0, thickness=0)
        visual = draw_photo(c, image, item['rows'], edits, held, data['identities'][iid], deleted, data['parent_round'])
        foot = footer(c, iid, data)
        pages.append({**visual, **foot, 'page': page_numbers[iid], 'image_id': iid,
                      'split': item['split'], 'label_sha256': item['sha256'],
                      'original_image_sha256': record['image_sha256'], 'embedded_original_size': list(image.size),
                      'changed_fields': {str(k): sorted(v) for k, v in edits.items()},
                      'held_fields': {str(k): sorted(v) for k, v in held.items()},
                      'case_ids': [case['case_id'] for case in data['cases'].get(iid, [])]})
        c.showPage()
    c.save()
    if any(digest(data['files'][iid]['path']) != data['files'][iid]['sha256'] for iid in data['ids']):
        raise ValueError('A label changed while the PDF was rendering')
    saved_pdf_checks = verify_saved_pdf(root, output, data)
    truncated_blocks = [{'image_id': page['image_id'], 'page': page['page'], 'block': block}
                        for page in pages
                        for block in ('changed_fields_block', 'availability_trace_block', 'case_notes_block')
                        if page[block]['truncated']]
    qa = {'schema_version': 1, 'summary': data['summary'], 'provenance': provenance,
          'pdf': str(output), 'pdf_sha256': digest(output), 'page_count': len(data['ids']) + index_count,
          'review_manifest': {'path': data['manifest_path'].relative_to(root).as_posix(), 'sha256': digest(data['manifest_path'])},
          'parent_review': data['parent'], 'input_descriptors': data['input_descriptors'],
          'included_image_ids': data['ids'], 'excluded_old_image_ids': sorted(i for i in data['files'] if i.startswith('old_')),
          'omitted_new_image_ids': sorted(i for i in data['files'] if i.startswith('new_') and i not in data['ids']), 'index_link_count': len(data['ids']),
          'unresolved_tag_collisions': sum(len(p['tag_overlap_pairs']) for p in pages),
          'clipped_tags': sum(len(p['clipped_tag_rows']) for p in pages),
          'tag_count': sum(p['tag_count'] for p in pages), 'current_tag_count': sum(p['current_tag_count'] for p in pages),
          'deletion_tag_count': sum(p['deletion_tag_count'] for p in pages),
          'moved_tags': sum(p['moved_tags'] for p in pages), 'geometry_modified_by_renderer': False,
          'all_included_original_hashes_verified': True, 'all_current_labels_exact_operation_replay': True,
          'row_identity_maps_verified': True, 'full_original_resolution_embedded': True,
          'display_case_notes': data['display_case_notes'], 'display_amendment_notes': data['display_amendment_notes'], 'numeric_only_operation_differences': data['numeric_only_differences'],
          'truncated_footer_blocks': truncated_blocks, **saved_pdf_checks,
          'builder_sha256': digest(Path(__file__)), 'pages': pages}
    qa_output.parent.mkdir(parents=True, exist_ok=True)
    qa_output.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + '\n')
    if qa['unresolved_tag_collisions'] or qa['clipped_tags'] or truncated_blocks:
        raise ValueError(f'PDF layout requires correction; inspect {qa_output}')
    return qa


def main(default_review=DEFAULT_REVIEW, default_ledger=DEFAULT_LEDGER,
         default_output='output/pdf/front-review-r16-changed-photos.pdf'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--review-root', type=Path, default=Path(default_review))
    parser.add_argument('--ledger-root', type=Path, default=Path(default_ledger))
    parser.add_argument('--output', type=Path, default=Path(default_output))
    parser.add_argument('--qa-output', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    def resolve(path):
        return path.resolve() if path.is_absolute() else (root / path).resolve()
    output = resolve(args.output)
    qa_output = resolve(args.qa_output) if args.qa_output else output.with_suffix('.qa.json')
    result = build(root, resolve(args.review_root), resolve(args.ledger_root), output, qa_output)
    print(json.dumps({'pdf': str(output), 'qa': str(qa_output), **result['summary'],
                      'unresolved_tag_collisions': result['unresolved_tag_collisions'],
                      'clipped_tags': result['clipped_tags'], 'deletion_tag_count': result['deletion_tag_count']}, indent=2))


if __name__ == '__main__':
    main()
