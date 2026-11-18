#!/usr/bin/env python3
"""Render this pass only: every current label on each changed original photo.

Magenta marks an edited or added row; cyan marks an inherited row. Exact row
changes remain in the operation ledger. This renderer does not choose geometry.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from PIL import Image
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from evaluate_pose import parse_pose_labels
from verify_front_review_r4 import REVIEW, PARENT, read_review
from verify_reconciled_pose_labels import sha


INK = HexColor('#132b38')
MUTED = HexColor('#506573')
CYAN = HexColor('#41e1ed')
MAGENTA = HexColor('#ff36d1')
RED = HexColor('#ff665e')
AMBER = HexColor('#ffb547')
PAPER = HexColor('#f4f8fa')
WIDTH, HEIGHT = landscape(A3)


def register_fonts():
    root = Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('Review', str(root / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('Review-Bold', str(root / 'DejaVuSans-Bold.ttf')))


def text(c, value, x, y, size=12, color=INK, bold=False):
    c.setFont('Review-Bold' if bold else 'Review', size)
    c.setFillColor(color)
    c.drawString(x, y, str(value))


def wrap(c, value, x, y, width, size=12, leading=18, color=INK):
    line = ''
    for word in value.split():
        trial = (line + ' ' + word).strip()
        if line and pdfmetrics.stringWidth(trial, 'Review', size) > width:
            text(c, line, x, y, size, color)
            y -= leading
            line = word
        else:
            line = trial
    if line:
        text(c, line, x, y, size, color)
        y -= leading
    return y


def ranges(numbers):
    numbers = sorted(set(numbers))
    groups = []
    for value in numbers:
        if groups and value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return ', '.join(str(group[0]) if len(group) == 1 else f'{group[0]}–{group[-1]}' for group in groups) or 'none'


def header(c, title, subtitle, number):
    c.setFillColor(white)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    text(c, 'CASSETTE FRONT REVIEW  /  ROUND 4', 30, HEIGHT - 30, 10, MUTED, True)
    text(c, title, 30, HEIGHT - 62, 24, INK, True)
    text(c, subtitle, 30, HEIGHT - 83, 11, MUTED)
    text(c, f'{number:02d}', WIDTH - 56, 22, 10, MUTED)
    text(c, '18 November 2026 · All positions use the original camera image.', 30, 22, 9, MUTED)


def panel(c, image, objects, x, y, w, h, changed=(), deleted=(), held=(), crop=None, label=True, omit_label_rows=()):
    """Draw a vector overlay; x,y are lower left in PDF coordinates."""
    crop = (0, 0, *image.size) if crop is None else tuple(map(int, crop))
    x0, y0, x1, y1 = crop
    scale = min(w / (x1 - x0), h / (y1 - y0))
    iw, ih = (x1 - x0) * scale, (y1 - y0) * scale
    x += (w - iw) / 2
    y += (h - ih) / 2
    c.drawImage(ImageReader(image.crop(crop)), x, y, iw, ih)
    c.saveState()
    clip = c.beginPath(); clip.rect(x, y, iw, ih); c.clipPath(clip, stroke=0)
    def point(px, py): return x + (px - x0) * scale, y + ih - (py - y0) * scale
    for row, obj in enumerate(objects, 1):
        bx0, by0, bx1, by1 = obj['bbox_xyxy']
        if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
            continue
        color = RED if row in deleted else AMBER if row in held else MAGENTA if row in changed else CYAN
        ax, ay = point(bx0, by1)
        bw, bh = (bx1 - bx0) * scale, (by1 - by0) * scale
        c.setStrokeColor(color); c.setLineWidth(1.2 if row in changed or row in deleted else .65)
        c.setDash(3, 2) if row in deleted else c.setDash()
        c.rect(ax, ay, bw, bh, fill=0, stroke=1)
        c.setDash()
        for side, (px, py, visible) in enumerate(obj['keypoints']):
            if not visible:
                continue
            qx, qy = point(px, py)
            radius = 2.2 if row in changed else 1.65
            c.setFillColor(color); c.setStrokeColor(INK); c.setLineWidth(.4)
            if side == 0:
                c.circle(qx, qy, radius, fill=1, stroke=1)
            else:
                c.rect(qx - radius, qy - radius, 2 * radius, 2 * radius, fill=1, stroke=1)
        if label and row not in omit_label_rows:
            value = str(row) + ('×' if row in deleted else '*' if row in changed else '')
            size = 7.2 if crop == (0, 0, *image.size) else 10
            tw = pdfmetrics.stringWidth(value, 'Review-Bold', size) + 4
            tx, ty = point(bx0, by0)
            tx = max(x + 1, min(tx, x + iw - tw - 1))
            ty = max(y + size + 3, min(ty, y + ih - 1))
            c.setFillColor(INK); c.rect(tx, ty - size - 3, tw, size + 3, fill=1, stroke=0)
            text(c, value, tx + 2, ty - size, size, color, True)
    c.restoreState()
    return {'crop': list(crop), 'objects': len(objects), 'changed_rows': sorted(changed),
            'deleted_rows_shown_only_as_before_evidence': sorted(deleted)}


def build(root, output, audit_output):
    register_fonts()
    root = Path(root).resolve()
    review, status, records, _, _, _, _ = read_review(root)
    by_id = {item['id']: item for item in records}
    operations = json.loads((root / review['operations']['path']).read_text())
    maps = json.loads((root / review['row_maps']['path']).read_text())
    coverage = json.loads((root / review['request_coverage']['path']).read_text())
    op_map = defaultdict(list)
    for op in operations: op_map[op['image_id']].append(op)
    changed_ids = sorted(op_map)
    counts = Counter(op['action'] for op in operations)
    parent = json.loads((root / PARENT['path']).read_text())
    parent_files = {item['image_id']: item for item in parent['files']}
    current_files = {item['image_id']: item for item in review['files']}
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle('Cassette front changes · Round 4 · 18 November 2026')
    c.setAuthor('Siuuuers/dl · assisted original-image review')
    c.setSubject('All labels on changed photos; magenta marks this pass. Review and training are separately versioned.')
    pages, number = [], 1
    header(c, 'The changes in this pass', '31 requested photographs + 2 inherited serialization repairs, with complete current labeling.', number)
    text(c, f'{len(changed_ids)} changed photos', 30, HEIGHT - 133, 28, INK, True)
    text(c, f'{counts["replace"]} edited rows  ·  {counts["add"]} added fronts  ·  {counts["delete"]} deleted rows',
         430, HEIGHT - 129, 20, INK, True)
    y = wrap(c, 'Cyan = inherited row. Magenta and * = row edited or added in this pass. Amber = unresolved row. '
             'Circle = image-left lower corner; square = image-right lower corner. No marker means that corner is unavailable. '
             'Magenta identifies the changed row, not a claim that every field in that row changed.',
             30, HEIGHT - 172, WIDTH - 60, 13, 20)
    y = wrap(c, 'Photo pages use CURRENT row numbers. All requests and operation records use the BASE row numbers from the PR68 atlas. '
             'The count-change detail pages show the mapping explicitly. Red dashed boxes appear only in deletion evidence, '
             'and are absent from the current labels.', 30, y - 7, WIDTH - 60, 12, 18)
    y = wrap(c, 'The three requested tiny-row removals are specific user decisions, not a general size rule. '
             'New boxes follow visible front material and its own lip. Occluded corners are not reconstructed from spacing. '
             'Soft boundaries remain approximate; this pass does not certify every inherited label or establish model improvement.',
             30, y - 7, WIDTH - 60, 12, 18)
    held = [case for case in status['cases'] if case['split'] == 'train' and case['status'] != 'resolved']
    if held:
        y = wrap(c, 'Still held for review: ' + '; '.join(case['case_id'] for case in held)
                 + '. The controlled notebook retains exact PR35 labels on each whole held photograph in both arms.',
                 30, y - 7, WIDTH - 60, 12, 18)
    y -= 17
    text(c, 'PHOTO INDEX', 30, y, 11, MUTED, True); y -= 27
    for index, image_id in enumerate(changed_ids):
        column, line = divmod(index, (len(changed_ids)+1)//2)
        xx, yy = 30 + column * 565, y - line * 20
        per = Counter(op['action'] for op in op_map[image_id])
        value = f'{image_id}     {per["replace"]} edit' + ('s' if per['replace'] != 1 else '')
        if per['add']: value += f' / +{per["add"]}'
        if per['delete']: value += f' / −{per["delete"]}'
        text(c, value, xx, yy, 11)
        text(c, str(index + 2), xx + 485, yy, 11, MUTED)
        c.linkRect('', f'photo-{image_id}', (xx, yy - 4, xx + 520, yy + 14), relative=0, thickness=0)
    pages.append({'page': number, 'type': 'index'})
    c.showPage()

    def data(image_id):
        record = by_id[image_id]
        image_path = root / 'data' / record['image']
        if sha(image_path) != record['image_sha256']: raise ValueError(f'original image hash differs: {image_id}')
        image = Image.open(image_path).convert('RGB')
        before_path = (root / PARENT['path']).parent / parent_files[image_id]['label_path']
        after_path = (root / REVIEW['path']).parent / current_files[image_id]['label_path']
        before = parse_pose_labels(before_path.read_text(), *image.size, str(before_path))
        after = parse_pose_labels(after_path.read_text(), *image.size, str(after_path))
        touched = {item['current_row'] for item in maps[image_id]['rows'] if item['changed']}
        return image, before, after, touched

    for image_id in changed_ids:
        number += 1
        image, before, after, touched = data(image_id)
        per = Counter(op['action'] for op in op_map[image_id])
        summary = f'All {len(after)} current labels · {per["replace"]} edit' + ('s' if per['replace'] != 1 else '')
        summary += f' · {per["add"]} addition' + ('s' if per['add'] != 1 else '')
        summary += f' · {per["delete"]} deletion' + ('s' if per['delete'] != 1 else '')
        header(c, image_id, summary, number)
        c.bookmarkPage(f'photo-{image_id}')
        c.addOutlineEntry(image_id, f'photo-{image_id}', level=0, closed=False)
        held_base = {row for case in held if case['image_id'] == image_id for row in case['base_rows']}
        held_current = {item['current_row'] for item in maps[image_id]['rows'] if item.get('base_row') in held_base}
        shown = panel(c, image, after, 30, 116, WIDTH - 60, HEIGHT - 213, changed=touched, held=held_current)
        base_rows = [op['base_row'] for op in op_map[image_id] if op['action'] == 'replace']
        caption = 'Edited base rows: ' + ranges(base_rows) + '.'
        if per['delete']: caption += ' Deleted base rows: ' + ranges(maps[image_id]['deleted_base_rows']) + '.'
        if per['add']: caption += ' Added current rows: ' + ranges(item['current_row'] for item in maps[image_id]['rows'] if 'addition_id' in item) + '.'
        yy = wrap(c, caption, 30, 93, WIDTH - 60, 11, 17)
        notes = coverage.get('photo_notes', {}).get(image_id, [])
        if notes: yy = wrap(c, ' '.join(notes), 30, yy - 3, WIDTH - 60, 10, 15, MUTED)
        pages.append({'page': number, 'type': 'complete_changed_photo', 'image_id': image_id, **shown,
                      'source_image_sha256': by_id[image_id]['image_sha256'], 'label_sha256': current_files[image_id]['label_sha256']})
        c.showPage()

    details = [
        ('new_000014', [11], 'One clipped-edge question remains open',
         'Base row 11 is a very narrow candidate at the right image boundary. The available pixels do not reliably distinguish its material from neighboring clipped fronts or rack. Its label is retained for review; the training candidate uses exact PR35 labels for this whole photograph in both arms.'),
        ('new_000015', [56,57,58], 'Four fronts below base row 56',
         'The four separate visible tiers are labeled. The two newly added partial fronts have no locatable actual lower corners. The nearby pre-existing bottom-left cassette is retained with its own identity.'),
        ('new_000023', [22], 'Two fronts within the former row 22',
         'The soft horizontal seam separates two visible fragments. The retained upper front also has a narrow visible strip beside the gripper; its box spans that occlusion. Both true corners remain unavailable for both fragments.'),
        ('new_000005', [47,48,49], 'Visible left corners restored',
         'The exposed lower-left front corners beside the gripper are locatable in this image. Their supervision changes from missing or misplaced points to the observed contour turns.'),
        ('new_000002', [38,39], 'Unavailable left corners removed',
         'The left corner slots for base rows 38 and 39 are missing in the current labels. The remaining visible front material stays labeled; a gripper edge is not substituted for a hidden corner.'),
        ('new_000019', [58,59,60,61,62], 'The gripper edge is not a cassette corner',
         'The right lower-front corners of these five fronts are unobservable. Their right slots are 0 0 0, while the identifiable fronts and their available left corners remain.'),
        ('new_000030', [21], 'A visible corner is retained as supervision',
         'The actual left lower-front corner of base row 21 is visible and now has a locatable point. Availability is decided from this frame, independently for each side.'),
    ]
    for image_id, base_rows, title, note in details:
        number += 1
        image, before, after, touched = data(image_id)
        relevant = [item['current_row'] for item in maps[image_id]['rows']
                    if item.get('base_row') in base_rows or item.get('after_base_row') in base_rows]
        boxes = [before[row - 1]['bbox_xyxy'] for row in base_rows] + [after[row - 1]['bbox_xyxy'] for row in relevant]
        crop = (max(0, int(min(box[0] for box in boxes)) - 24), max(0, int(min(box[1] for box in boxes)) - 24),
                min(image.width, int(max(box[2] for box in boxes)) + 25), min(image.height, int(max(box[3] for box in boxes)) + 25))
        header(c, title, image_id + ' · Detail from original pixels · Before uses base rows; current uses new rows.', number)
        text(c, 'BEFORE', 30, HEIGHT - 119, 12, MUTED, True)
        text(c, 'CURRENT', WIDTH / 2 + 15, HEIGHT - 119, 12, MUTED, True)
        held_base = {row for case in held if case['image_id'] == image_id for row in case['base_rows']}
        held_current = {item['current_row'] for item in maps[image_id]['rows'] if item.get('base_row') in held_base}
        panel(c, image, before, 30, 195, WIDTH / 2 - 45, HEIGHT - 330, held=held_base, crop=crop)
        panel(c, image, after, WIDTH / 2 + 15, 195, WIDTH / 2 - 45, HEIGHT - 330, changed=touched, held=held_current, crop=crop)
        yy = wrap(c, note, 30, 165, WIDTH - 60, 14, 21)
        identity = []
        for item in maps[image_id]['rows']:
            if item.get('base_row') in base_rows or item.get('after_base_row') in base_rows:
                identity.append(f'base {item["base_row"]} → current {item["current_row"]}' if 'base_row' in item else f'added → current {item["current_row"]}')
        wrap(c, 'Row mapping: ' + '; '.join(identity) + '.', 30, yy - 9, WIDTH - 60, 12, 18, MUTED)
        pages.append({'page': number, 'type': 'before_current_detail', 'image_id': image_id,
                      'base_rows': base_rows, 'current_rows': relevant, 'crop': list(crop)})
        c.showPage()

    number += 1
    header(c, 'Three specific row removals', 'Requested decisions · Red dashed boxes show removed base rows only.', number)
    text(c, 'BEFORE · BASE ROW NUMBERS', 30, HEIGHT - 105, 10, MUTED, True)
    text(c, 'CURRENT · NEW ROW NUMBERS', WIDTH / 2 + 15, HEIGHT - 105, 10, MUTED, True)
    for index, (image_id, row) in enumerate([('new_000003',15),('new_000004',15),('new_000033',4)]):
        image, before, after, touched = data(image_id)
        box = before[row-1]['bbox_xyxy']
        crop = (max(0,int(box[0])-25), max(0,int(box[1])-25), min(image.width,int(box[2])+26), min(image.height,int(box[3])+26))
        yy = HEIGHT - 310 - index * 213
        text(c, f'{image_id} · deleted base row {row}', 30, yy + 188, 12, INK, True)
        panel(c, image, before, 30, yy, WIDTH / 2 - 45, 176, deleted={row}, crop=crop, omit_label_rows={row})
        panel(c, image, after, WIDTH / 2 + 15, yy, WIDTH / 2 - 45, 176, changed=touched, crop=crop)
    text(c, 'Renumbering: surviving base 16 → current 15 in new_000003/new_000004; base 5 → current 4 in new_000033.', 30, 94, 10, MUTED)
    wrap(c, 'These removals record the user’s explicit decisions for three identified rows. They do not prove the material is physically absent, and they do not authorize deleting other small or partial fronts.', 30, 69, WIDTH - 60, 11, 16, MUTED)
    pages.append({'page': number, 'type': 'deletion_evidence', 'removed': ['new_000003:15','new_000004:15','new_000033:4']})
    c.showPage(); c.save()
    audit = {'schema_version':1, 'kind':'focused_front_review_r4_pdf', 'review':REVIEW,
             'pdf_filename':output.name, 'pdf_sha256':sha(output), 'pages':pages,
             'changed_images':len(changed_ids), 'operation_counts':dict(counts),
             'all_current_labels_shown_on_each_changed_photo':True,
             'original_image_bytes_hash_verified':True, 'geometry_is_rendered_from_verified_labels':True,
             'generator_sha256':sha(Path(__file__)), 'training_run':False}
    Path(audit_output).write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'pdf':str(output),'pages':len(pages),'bytes':output.stat().st_size,'audit':str(audit_output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-output',type=Path,required=True)
    args=parser.parse_args();build(args.root,args.output,args.audit_output)


if __name__ == '__main__': main()
