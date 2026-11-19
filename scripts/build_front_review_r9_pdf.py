#!/usr/bin/env python3
"""Render changed or unresolved photographs without moving any label geometry."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from PIL import Image
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


WIDTH, HEIGHT = landscape(A3)
INK = HexColor('#132b38')
MUTED = HexColor('#506573')
CYAN = HexColor('#41e1ed')
MAGENTA = HexColor('#ff36d1')
AMBER = HexColor('#ffb547')
GROUPS = {'box': slice(1, 5), 'left': slice(5, 8), 'right': slice(8, 11)}
TAG_SIZE, TAG_HEIGHT, TAG_GAP = 7.2, 11.0, 1.4
DEFAULT_REVIEW = 'data/reviews/fronts-2026-11-19-r9-recovery'
DEFAULT_LEDGER = 'results/review/front-review-20261119-r9-recovery'
PROVENANCE = ('Reconstructed from preserved r8; interrupted unsaved r9 coordinates unavailable. '
              'Counts and highlights below describe only this saved reconstruction.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def register_fonts():
    fonts = Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('Review', str(fonts / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('Review-Bold', str(fonts / 'DejaVuSans-Bold.ttf')))


def text(c, value, x, y, size=11, color=INK, bold=False):
    c.setFont('Review-Bold' if bold else 'Review', size)
    c.setFillColor(color)
    c.drawString(x, y, str(value))


def lines_for(value, width, size=11, bold=False):
    font = 'Review-Bold' if bold else 'Review'
    result = []
    for paragraph in str(value).split('\n'):
        line = ''
        for word in paragraph.split():
            trial = (line + ' ' + word).strip()
            if line and pdfmetrics.stringWidth(trial, font, size) > width:
                result.append(line)
                line = word
            else:
                line = trial
        result.append(line)
    return result


def wrap(c, value, x, y, width, size=11, leading=15, color=INK):
    for line in lines_for(value, width, size):
        text(c, line, x, y, size, color)
        y -= leading
    return y


def ranges(values):
    groups = []
    for n in sorted(set(values)):
        if groups and n == groups[-1][-1] + 1:
            groups[-1].append(n)
        else:
            groups.append([n])
    return ', '.join(str(g[0]) if len(g) == 1 else f'{g[0]}-{g[-1]}'
                     for g in groups) or 'none'


def parse_line(line, source):
    values = [float(v) for v in line.split()]
    if len(values) != 11 or not all(math.isfinite(v) for v in values):
        raise ValueError(f'{source}: expected eleven finite values')
    if values[0] != 0 or values[3] <= 0 or values[4] <= 0:
        raise ValueError(f'{source}: invalid class or box')
    for start in (5, 8):
        if values[start + 2] not in (0, 2):
            raise ValueError(f'{source}: unsupported keypoint availability')
        if values[start + 2] == 0 and values[start:start + 2] != [0, 0]:
            raise ValueError(f'{source}: missing point must be 0 0 0')
    return values


def parse_labels(path):
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    return rows, [parse_line(line, f'{path}:{i}') for i, line in enumerate(rows, 1)]


def current_row(operation, mapping):
    if 'current_row' in operation:
        return int(operation['current_row'])
    base = operation.get('base_row', operation.get('base_row1based', operation.get('row')))
    if base is None:
        raise ValueError(f'Operation has no row: {operation}')
    base = int(base)
    return mapping.get(base, base)


def load_display_notes(ledger_root, status_path, cases):
    path = ledger_root / 'pdf-case-notes.json'
    if not path.exists():
        return {}, None
    document = read_json(path)
    status_hash = digest(status_path)
    if document.get('source_review_status_sha256') != status_hash:
        raise ValueError('PDF case notes do not match the saved review-status.json')
    notes = document.get('case_notes')
    if not isinstance(notes, dict):
        raise ValueError('PDF case notes require a case_notes object keyed by case_id')
    held_ids = [case.get('case_id') for case in cases if case.get('status') == 'held']
    if any(not isinstance(i, str) or not i for i in held_ids) or len(set(held_ids)) != len(held_ids):
        raise ValueError('Held cases must have unique nonempty case IDs for PDF notes')
    if set(notes) != set(held_ids):
        raise ValueError(f'PDF case-note coverage mismatch: missing={sorted(set(held_ids) - set(notes))}, '
                         f'extra={sorted(set(notes) - set(held_ids))}')
    if any(not isinstance(note, str) or not note.strip() or len(note) > 190
           or '\n' in note for note in notes.values()):
        raise ValueError('Every PDF case note must be one nonempty line of at most 190 characters')
    return notes, {'path': str(path), 'sha256': digest(path),
                   'source_review_status_sha256': status_hash,
                   'held_case_count': len(held_ids), 'exact_held_case_coverage': True,
                   'display_only': True}


def load_inputs(root, review_root, ledger_root):
    source = read_json(root / 'data/manifest.json')
    records = {r['id']: r for r in source['records']}
    files = {}
    for split in ('train', 'val'):
        for path in sorted((review_root / 'labels' / split).glob('*.txt')):
            if path.stem in files:
                raise ValueError(f'Duplicate review image: {path.stem}')
            raw, rows = parse_labels(path)
            files[path.stem] = {'path': path, 'split': split, 'raw': raw,
                                'rows': rows, 'sha256': digest(path)}
    if set(files) != set(records):
        raise ValueError('Review labels must cover the complete original cohort')
    for iid, item in files.items():
        if item['split'] != records[iid]['split']:
            raise ValueError(f'Split changed: {iid}')
    maps_path = ledger_root / 'row-maps.json'
    maps = read_json(maps_path) if maps_path.exists() else {}
    base_maps = {iid: {int(r['base_row']): int(r['current_row'])
                      for r in value['rows'] if r.get('base_row') is not None}
                 for iid, value in maps.items()}
    operations_path = ledger_root / 'operations.json'
    raw_operations = read_json(operations_path)
    operations = (raw_operations if isinstance(raw_operations, list)
                  else raw_operations['operations'])
    edits = defaultdict(lambda: defaultdict(set))
    actual_operations = []
    for op in operations:
        iid = op['image_id']
        if iid not in files:
            raise ValueError(f'Unknown operation image: {iid}')
        if op.get('action', 'replace') != 'replace':
            raise ValueError('This recovery renderer requires stable rows and replace operations')
        before = parse_line(op['before_line'], f'{iid} before')
        after = parse_line(op['after_line'], f'{iid} after')
        if before[0] != after[0]:
            raise ValueError(f'Class changed: {iid}')
        changed = {name for name, part in GROUPS.items() if before[part] != after[part]}
        row = current_row(op, base_maps.get(iid, {}))
        if row < 1 or row > len(files[iid]['rows']):
            raise ValueError(f'Operation row out of range: {iid}:{row}')
        if files[iid]['rows'][row - 1] != after:
            raise ValueError(f'Operation after-line differs from saved row: {iid}:{row}')
        if changed:
            edits[iid][row].update(changed)
            actual_operations.append(dict(op, current_row=row, actual_changed_groups=sorted(changed)))
    status_path = ledger_root / 'review-status.json'
    status = read_json(status_path)
    cases = status if isinstance(status, list) else status['cases']
    display_notes, display_notes_audit = load_display_notes(ledger_root, status_path, cases)
    held, case_map = defaultdict(lambda: defaultdict(set)), defaultdict(list)
    for case in cases:
        if case.get('status') != 'held':
            continue
        case = dict(case)
        if display_notes:
            case['short_reason'] = display_notes[case['case_id']]
        iid = case['image_id']
        if iid not in files:
            raise ValueError(f'Unknown held image: {iid}')
        case_map[iid].append(case)
        rows = case.get('current_rows')
        if rows is None:
            rows = [base_maps.get(iid, {}).get(int(r), int(r))
                    for r in case.get('base_rows', [])]
        for row in rows:
            if row < 1 or row > len(files[iid]['rows']):
                raise ValueError(f'Held row out of range: {iid}:{row}')
            held[iid][row].update(case.get('fields', GROUPS))
    included = sorted(set(edits) | set(case_map))
    counts = Counter(g for rows in edits.values() for groups in rows.values() for g in groups)
    summary = {'source_photos': len(records), 'included_photos': len(included),
               'omitted_unchanged_without_holds': len(records) - len(included),
               'coordinate_changed_photos': len(edits),
               'edited_rows': sum(len(rows) for rows in edits.values()),
               'changed_fields': {g: counts[g] for g in GROUPS},
               'held_photos': len(case_map), 'held_cases': sum(map(len, case_map.values())),
               'held_training_photos': sum(files[i]['split'] == 'train' for i in case_map),
               'review_objects': sum(len(v['rows']) for v in files.values()),
               'included_objects': sum(len(files[i]['rows']) for i in included)}
    return records, files, edits, held, case_map, included, summary, actual_operations, display_notes_audit


def intersects(a, b, gap=0):
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0]
                or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def inside(rect, bounds, tolerance=0.00001):
    return (rect[0] >= bounds[0] - tolerance and rect[1] >= bounds[1] - tolerance
            and rect[2] <= bounds[2] + tolerance and rect[3] <= bounds[3] + tolerance)


def place_tags(specs, bounds, keypoints):
    """Keep thin boxes first; choose nearby free rectangles without hiding points."""
    placed = []
    bx0, by0, bx1, by1 = bounds
    for spec in sorted(specs, key=lambda t: (t['box_height'], -t['anchor'][1], t['row'])):
        width, height = spec['width'], TAG_HEIGHT
        ax, ay = spec['anchor']
        natural = (ax + 1, ay - height - 1)
        desired = (max(bx0 + 1, min(natural[0], bx1 - width - 1)),
                   max(by0 + 1, min(natural[1], by1 - height - 1)))
        occupied = [p['rect'] for p in placed]

        def candidate(x, y):
            x = max(bx0 + 1, min(x, bx1 - width - 1))
            y = max(by0 + 1, min(y, by1 - height - 1))
            rect = (x, y, x + width, y + height)
            if any(intersects(rect, other, TAG_GAP) for other in occupied):
                return None
            covers = sum(rect[0] - 2 <= px <= rect[2] + 2
                         and rect[1] - 2 <= py <= rect[3] + 2 for px, py in keypoints)
            distance = (x - desired[0]) ** 2 + (y - desired[1]) ** 2
            return (covers * 10000 + distance, rect)

        choices = []
        initial = candidate(*desired)
        if initial and initial[0] == 0:
            best = initial
        else:
            for radius in range(0, 11):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if max(abs(dx), abs(dy)) != radius:
                            continue
                        found = candidate(desired[0] + dx * 4, desired[1] + dy * 4)
                        if found:
                            choices.append(found)
                if choices and min(v[0] for v in choices) < (radius * 4) ** 2:
                    break
            if not choices:
                for y in range(math.ceil(by0 + 1), math.floor(by1 - height), 4):
                    for x in range(math.ceil(bx0 + 1), math.floor(bx1 - width), 4):
                        found = candidate(x, y)
                        if found:
                            choices.append(found)
            if not choices:
                raise ValueError(f'No collision-free tag position for row {spec["row"]}')
            best = min(choices, key=lambda v: (v[0], v[1]))
        rect = best[1]
        # Moving even one tag does not change its physical anchor or any labels.
        moved = math.hypot(rect[0] - natural[0], rect[1] - natural[1]) > 0.5
        placed.append(dict(spec, rect=rect, desired=list(natural), clamped_start=list(desired), moved=moved,
                           displacement_pt=math.hypot(rect[0] - natural[0], rect[1] - natural[1])))
    collisions = [[a['row'], b['row']] for i, a in enumerate(placed)
                  for b in placed[i + 1:] if intersects(a['rect'], b['rect'])]
    clipped = [t['row'] for t in placed if not inside(t['rect'], bounds)
               or not inside(t['rect'], (0, 0, WIDTH, HEIGHT))]
    if collisions or clipped:
        raise ValueError(f'Tag-layout audit failed: {collisions=}, {clipped=}')
    return sorted(placed, key=lambda t: t['row']), collisions, clipped


def draw_photo(c, image, rows, edits, held):
    area = (30, 128, WIDTH - 60, HEIGHT - 224)
    scale = min(area[2] / image.width, area[3] / image.height)
    iw, ih = image.width * scale, image.height * scale
    x, y = area[0] + (area[2] - iw) / 2, area[1] + (area[3] - ih) / 2
    bounds = (x, y, x + iw, y + ih)
    c.drawImage(ImageReader(image), x, y, iw, ih)

    def point(px, py):
        return x + px * scale, y + ih - py * scale

    def color(row, group):
        return MAGENTA if group in edits.get(row, ()) else AMBER if group in held.get(row, ()) else CYAN

    specs, keypoints = [], []
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
            keypoints.append((px, py))
            radius = 2.0 if side in edits.get(row, ()) else 1.6
            c.setFillColor(color(row, side))
            c.setStrokeColor(AMBER if side in held.get(row, ()) else INK)
            c.setLineWidth(0.7 if side in held.get(row, ()) else 0.35)
            if side == 'left':
                c.circle(px, py, radius, fill=1, stroke=1)
            else:
                c.rect(px - radius, py - radius, 2 * radius, 2 * radius, fill=1, stroke=1)
        label = str(row) + ('*' if edits.get(row) else '')
        anchor = point(left, top)
        specs.append({'row': row, 'label': label, 'anchor': list(anchor),
                      'width': pdfmetrics.stringWidth(label, 'Review-Bold', TAG_SIZE) + 5,
                      'box_height': max(0, (bottom - top) * scale),
                      'tag_color': 'magenta' if edits.get(row) else 'amber' if held.get(row) else 'cyan',
                      'held': bool(held.get(row))})
    c.restoreState()
    tags, collisions, clipped = place_tags(specs, bounds, keypoints)
    palette = {'magenta': MAGENTA, 'amber': AMBER, 'cyan': CYAN}
    # All leaders precede all filled number tags, so later rows cannot obscure tags.
    for tag in tags:
        if not tag['moved']:
            continue
        ax, ay = tag['anchor']; rect = tag['rect']
        ax = max(x + 0.6, min(ax, x + iw - 0.6)); ay = max(y + 0.6, min(ay, y + ih - 0.6))
        tx, ty = max(rect[0], min(ax, rect[2])), max(rect[1], min(ay, rect[3]))
        c.setStrokeColor(INK); c.setLineWidth(1.35); c.line(ax, ay, tx, ty)
        c.setStrokeColor(palette[tag['tag_color']]); c.setLineWidth(0.65); c.line(ax, ay, tx, ty)
        c.setFillColor(palette[tag['tag_color']]); c.circle(ax, ay, 0.85, stroke=0, fill=1)
    for tag in tags:
        x0, y0, x1, y1 = tag['rect']
        c.setFillColor(INK); c.setStrokeColor(AMBER if tag['held'] else palette[tag['tag_color']])
        c.setLineWidth(0.45); c.rect(x0, y0, x1 - x0, y1 - y0, fill=1, stroke=1)
        text(c, tag['label'], x0 + 2.5, y0 + 2.7, TAG_SIZE, palette[tag['tag_color']], True)
    return {'image_bounds_pt': list(bounds), 'tags': tags, 'tag_count': len(tags),
            'moved_tags': sum(t['moved'] for t in tags), 'tag_overlap_pairs': collisions,
            'clipped_tag_rows': clipped, 'displayed_rows': list(range(1, len(rows) + 1))}


def header(c, title, subtitle, page):
    c.setFillColor(white); c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    text(c, 'CASSETTE FRONT REVIEW / RECOVERED FOLLOW-UP', 30, HEIGHT - 29, 10, MUTED, True)
    text(c, title, 30, HEIGHT - 59, 24, INK, True)
    text(c, subtitle, 30, HEIGHT - 80, 10.5, MUTED)
    text(c, 'Original camera pixels / Current saved row numbers / Coordinates unchanged by rendering', 30, 20, 9, MUTED)
    text(c, f'{page:03d}', WIDTH - 56, 20, 10, MUTED)


def intro(c, summary, ids, files, edits, cases, page_numbers):
    header(c, 'Only the photographs that need attention',
           'Every current object is shown on an included photograph. Unchanged photographs without an open case are omitted.', 1)
    text(c, f'{summary["included_photos"]} included', 30, HEIGHT - 132, 30, INK, True)
    text(c, f'{summary["omitted_unchanged_without_holds"]} omitted', 310, HEIGHT - 132, 30, INK, True)
    text(c, f'{summary["source_photos"]} originals', 620, HEIGHT - 132, 30, INK, True)
    counts = summary['changed_fields']
    text(c, f'{summary["coordinate_changed_photos"]} changed photos / {summary["edited_rows"]} edited rows / '
         f'{counts["box"]} boxes, {counts["left"]} left points, {counts["right"]} right points / '
         f'{summary["held_cases"]} open cases in {summary["held_photos"]} photos', 30, HEIGHT - 161, 12)
    wrap(c, PROVENANCE, 30, HEIGHT - 187, WIDTH - 60, 12, 17, MUTED)
    legends = [('CYAN', 'Inherited / unchanged; not physical certification.', CYAN),
               ('MAGENTA + *', 'Only changed fields; * identifies an edited row.', MAGENTA),
               ('AMBER', 'Unresolved field; amber outlines also retain uncertainty.', AMBER)]
    col_width = (WIDTH - 60) / 3
    for i, (label, body, color) in enumerate(legends):
        bx = 30 + i * col_width
        c.setFillColor(INK); c.roundRect(bx, HEIGHT - 271, col_width - 14, 53, 5, fill=1, stroke=0)
        text(c, label, bx + 12, HEIGHT - 238, 12, color, True)
        wrap(c, body, bx + 12, HEIGHT - 255, col_width - 38, 9.5, 12, white)
    wrap(c, 'Circle = image-left lower-front corner. Square = image-right corner. Missing points have no marker. '
         'Moved number tags use leaders; no box or endpoint moves. The front and its own rim define the box; '
         'the two physical lower-front corners are independent of the rectangle.', 30, HEIGHT - 296, WIDTH - 60, 11.5, 16)
    wrap(c, 'Model weights and frozen scoring references are unchanged. The separately staged input comparison '
         f'uses exact PR35 labels in both arms for all {summary["held_training_photos"]} held training photographs.',
         30, HEIGHT - 338, WIDTH - 60, 10.5, 15, MUTED)
    text(c, 'PHOTO INDEX', 30, HEIGHT - 378, 11, MUTED, True)
    draw_index(c, ids, files, edits, cases, page_numbers, HEIGHT - 401, 29)


def draw_index(c, ids, files, edits, cases, page_numbers, top, per_column):
    width = (WIDTH - 60) / 3
    for i, iid in enumerate(ids):
        column, row = divmod(i, per_column)
        x, y = 30 + column * width, top - row * 12.3
        flags = (' *' if iid in edits else '') + (' ?' if iid in cases else '')
        text(c, iid + flags, x, y, 10.1, INK, True)
        text(c, f'{len(files[iid]["rows"])} fronts', x + 157, y, 9.5, MUTED)
        text(c, str(page_numbers[iid]), x + width - 45, y, 10.1, MUTED)
        c.linkRect('', 'photo-' + iid, (x, y - 3, x + width - 18, y + 10), relative=0, thickness=0)


def concise_case(case, mapping):
    rows = case.get('current_rows')
    if rows is None:
        rows = [mapping.get(int(r), int(r)) for r in case.get('base_rows', [])]
    fields = '/'.join(case.get('fields', ['box']))
    prefix = f'Rows {ranges(rows)} ({fields})' if rows else 'Possible extra-front region'
    reason = str(case.get('short_reason') or case.get('reason') or 'Physical ownership remains unresolved.')
    reason = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', reason)
    first = re.split(r'(?<=[.!?])\s+', reason)[0]
    if len(first) > 190:
        first = first[:187].rsplit(' ', 1)[0] + '...'
    return prefix + ': ' + first


def build(root, review_root, ledger_root, output, qa_output):
    register_fonts()
    records, files, edits, held, cases, ids, summary, operations, display_notes_audit = load_inputs(root, review_root, ledger_root)
    maps_path = ledger_root / 'row-maps.json'
    maps = read_json(maps_path) if maps_path.exists() else {}
    extra_index_pages = math.ceil(max(0, len(ids) - 87) / 150)
    first_photo = 2 + extra_index_pages
    page_numbers = {iid: first_photo + i for i, iid in enumerate(ids)}
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle('Cassette front review - recovered focused follow-up')
    c.setAuthor('Siuuuers/dl')
    c.setSubject(PROVENANCE)
    intro(c, summary, ids[:87], files, edits, cases, page_numbers); c.showPage()
    for n in range(extra_index_pages):
        header(c, 'Photo index, continued', '* = edited row(s) / ? = unresolved case(s)', n + 2)
        draw_index(c, ids[87 + 150 * n:87 + 150 * (n + 1)], files, edits, cases,
                   page_numbers, HEIGHT - 120, 50)
        c.showPage()
    pages = []
    for iid in ids:
        item, record, page = files[iid], records[iid], page_numbers[iid]
        image_path = root / 'data' / record['image']
        image_hash = digest(image_path)
        if image_hash != record['image_sha256']:
            raise ValueError(f'Original image changed: {iid}')
        with Image.open(image_path) as original:
            image = original.convert('RGB')
        if list(image.size) != list(record['original_size']):
            raise ValueError(f'Original image dimensions changed: {iid}')
        split = 'TRAIN REVIEW' if item['split'] == 'train' else 'DEVELOPMENT REVIEW'
        header(c, iid, f'{split} / All {len(item["rows"])} current fronts / '
               f'{len(edits[iid])} edited rows / {len(cases[iid])} unresolved cases', page)
        c.bookmarkPage('photo-' + iid); c.addOutlineEntry(iid, 'photo-' + iid, level=0)
        visual = draw_photo(c, image, item['rows'], edits[iid], held[iid])
        text(c, 'CHANGED FIELDS', 30, 109, 10, MUTED, True)
        changes = [f'{g.capitalize()}: {ranges(r for r, gs in edits[iid].items() if g in gs)}'
                   for g in GROUPS]
        wrap(c, '\n'.join(changes), 30, 93, 330, 10, 13)
        text(c, 'OPEN QUESTIONS', 390, 109, 10, MUTED, True)
        mapping = {r['base_row']: r['current_row'] for r in maps.get(iid, {}).get('rows', [])
                   if r.get('base_row') is not None}
        notes = [concise_case(case, mapping) for case in cases[iid]] or ['No remaining held case on this photograph.']
        note_text = ' '.join(notes)
        size, leading = 10.0, 12.8
        while len(lines_for(note_text, WIDTH - 420, size)) * leading > 65 and size > 8.7:
            size -= 0.2; leading = size + 2.5
        note_lines = lines_for(note_text, WIDTH - 420, size)
        if len(note_lines) * leading > 65:
            allowance = max(2, int(65 / leading) - 1)
            note_lines = note_lines[:allowance] + ['Full per-case evidence is preserved in review-status.json.']
        for n, line in enumerate(note_lines):
            text(c, line, 390, 93 - n * leading, size)
        pages.append(dict(visual, page=page, image_id=iid, split=item['split'],
                          label_sha256=item['sha256'], original_image_sha256=image_hash,
                          changed_fields={str(k): sorted(v) for k, v in edits[iid].items()},
                          held_fields={str(k): sorted(v) for k, v in held[iid].items()},
                          case_ids=[case.get('case_id') for case in cases[iid]],
                          notes_lines=note_lines))
        c.showPage()
    c.save()
    for iid in ids:
        if digest(files[iid]['path']) != files[iid]['sha256']:
            raise ValueError(f'Label file changed during rendering: {iid}')
    qa = {'schema_version': 1, 'summary': summary, 'provenance': PROVENANCE,
          'pdf': str(output), 'pdf_sha256': digest(output),
          'page_count': 1 + extra_index_pages + len(ids),
          'included_image_ids': ids, 'omitted_image_ids': sorted(set(files) - set(ids)),
          'unresolved_tag_collisions': sum(len(p['tag_overlap_pairs']) for p in pages),
          'clipped_tags': sum(len(p['clipped_tag_rows']) for p in pages),
          'tag_count': sum(p['tag_count'] for p in pages),
          'moved_tags': sum(p['moved_tags'] for p in pages),
          'geometry_modified_by_renderer': False, 'all_included_original_hashes_verified': True,
          'display_case_notes': display_notes_audit,
          'operations_sha256': digest(ledger_root / 'operations.json'),
          'review_status_sha256': digest(ledger_root / 'review-status.json'), 'pages': pages}
    qa_output.parent.mkdir(parents=True, exist_ok=True)
    qa_output.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + '\n')
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--review-root', type=Path, default=Path(DEFAULT_REVIEW))
    parser.add_argument('--ledger-root', type=Path, default=Path(DEFAULT_LEDGER))
    parser.add_argument('--output', type=Path, default=Path('output/pdf/cassette-front-review-r9-recovery.pdf'))
    parser.add_argument('--qa-output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    def resolve(path):
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    output = resolve(args.output)
    qa_output = resolve(args.qa_output) if args.qa_output else output.with_suffix('.qa.json')
    result = build(root, resolve(args.review_root), resolve(args.ledger_root), output, qa_output)
    print(json.dumps({'pdf': str(output), 'qa': str(qa_output), **result['summary'],
                      'unresolved_tag_collisions': result['unresolved_tag_collisions'],
                      'clipped_tags': result['clipped_tags'], 'moved_tags': result['moved_tags']}, indent=2))


if __name__ == '__main__':
    main()
