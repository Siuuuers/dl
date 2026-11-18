#!/usr/bin/env python3
"""Render every original photograph with its complete r6 annotation set.

This renderer reads accepted labels; it never chooses coordinates. Magenta is
applied independently to box, left point and right point changed from r5.
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
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics

from build_front_review_r4_pdf import (
    INK, MUTED, CYAN, MAGENTA, RED, AMBER, PAPER, WIDTH, HEIGHT,
    register_fonts, text, wrap as base_wrap, ranges,
)
from evaluate_pose import parse_pose_labels
from verify_front_review_r6 import REVIEW, PARENT, read_review
from verify_reconciled_pose_labels import sha


def wrap(c, value, *args, **kwargs):
    """Keep compact audit notes legible in the reader-facing PDF."""
    value = re.sub(r'\b(All|all|Reviewed|reviewed|Inspected|inspected|BASE|base|row|rows|exact|near|occluded)(?=\d)', r'\1 ', value)
    value = re.sub(r'\b(\d+)(fronts?|rows?|base|originals?|visible|cases?|photographs?|photos?|old|px)\b', r'\1 \2', value)
    value = value.replace('near(', 'near (').replace('r5fronts', 'r5 fronts')
    return base_wrap(c, value, *args, **kwargs)


def number_word(value, word):
    return f'{value} {word}'+('' if value == 1 else 's')


def header(c, title, subtitle, page):
    c.setFillColor(white)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    text(c, 'CASSETTE FRONT REVIEW / ROUND 6', 30, HEIGHT - 29, 10, MUTED, True)
    text(c, title, 30, HEIGHT - 60, 24, INK, True)
    text(c, subtitle, 30, HEIGHT - 81, 11, MUTED)
    text(c, f'{page:03d}', WIDTH - 57, 21, 10, MUTED)
    text(c, 'Round 6 / Original camera pixels / Current r6 row numbers unless marked BASE', 30, 21, 9, MUTED)


def panel(c, image, objects, x, y, w, h, *, fields=None, held=None,
          crop=None, deleted=(), labels=True, raw=False):
    fields, held = fields or {}, held or {}
    crop = (0, 0, *image.size) if crop is None else tuple(map(int, crop))
    x0, y0, x1, y1 = crop
    scale = min(w / (x1 - x0), h / (y1 - y0))
    iw, ih = (x1 - x0) * scale, (y1 - y0) * scale
    x += (w - iw) / 2
    y += (h - ih) / 2
    c.drawImage(ImageReader(image.crop(crop)), x, y, iw, ih)
    c.saveState()
    clip = c.beginPath(); clip.rect(x, y, iw, ih); c.clipPath(clip, stroke=0)
    def point(px, py):
        return x + (px - x0) * scale, y + ih - (py - y0) * scale
    def color(row, group):
        if row in deleted: return RED
        if group in fields.get(row, ()): return MAGENTA
        if group in held.get(row, ()): return AMBER
        return CYAN
    shown, changed_fields, visible_points = [], Counter(), 0
    for row, obj in enumerate(objects, 1):
        a, b, d, e = obj['bbox_xyxy']
        if d < x0 or a > x1 or e < y0 or b > y1: continue
        shown.append(row)
        if raw: continue
        changed = fields.get(row, ())
        changed_fields.update(changed)
        c.setStrokeColor(color(row, 'box'))
        c.setLineWidth(.95 if 'box' in changed else .55)
        c.setDash(3, 2) if row in deleted else c.setDash()
        ax, ay = point(a, e)
        c.rect(ax, ay, (d-a)*scale, (e-b)*scale, fill=0, stroke=1)
        c.setDash()
        for side, (px, py, available) in enumerate(obj['keypoints']):
            if not available: continue
            visible_points += 1
            group = 'left' if side == 0 else 'right'
            qx, qy = point(px, py)
            r = 2 if group in changed else 1.5
            c.setFillColor(color(row, group)); c.setStrokeColor(INK); c.setLineWidth(.35)
            if side == 0: c.circle(qx, qy, r, fill=1, stroke=1)
            else: c.rect(qx-r, qy-r, 2*r, 2*r, fill=1, stroke=1)
        if labels:
            value = str(row) + ('x' if row in deleted else '*' if changed else '')
            size = 6.8 if crop == (0, 0, *image.size) else 9
            tw = pdfmetrics.stringWidth(value, 'Review-Bold', size) + 3.5
            tx, ty = point(a, b)
            tx = max(x+1, min(tx, x+iw-tw-1))
            ty = max(y+size+3, min(ty, y+ih-1))
            c.setFillColor(INK); c.rect(tx, ty-size-3, tw, size+3, fill=1, stroke=0)
            tag_color = RED if row in deleted else MAGENTA if changed else AMBER if row in held else CYAN
            text(c, value, tx+1.75, ty-size, size, tag_color, True)
    c.restoreState()
    return {'crop': list(crop), 'displayed_rows': shown, 'objects': len(objects),
            'changed_field_counts': dict(changed_fields), 'visible_points': visible_points,
            'held_current_rows': sorted(held), 'deleted_base_rows': sorted(deleted)}


def definition_page(c, page):
    header(c, 'The corner is a physical feature',
           'The baomai definition is retained. Pixel brightness alone does not identify a corner.', page)
    rules = [
        ('BOX', 'The smallest axis-aligned rectangle enclosing the identifiable visible front and its own front rim. Rack, cast shadow, top surface, projecting support, gripper and neighboring cassettes do not set its outer extent.'),
        ('LEFT / RIGHT', 'The actual two corners at the lower edge of that front, ordered by image X. They are not automatically the corners of the axis-aligned box. A rounded transition is a nominal contour junction, with blur-related uncertainty.'),
        ('MISSING', 'Retain a recognizable front even when one or both corners cannot be located. Each unavailable point is exactly 0 0 0; its marker is absent. Never reconstruct hidden points or whole hidden fronts from stack spacing.'),
        ('OVERLAP', 'A tilted front can require background inside its rectangle. One front visible on both sides of an occluder can require a box spanning that occluder. Neither situation justifies expanding the box beyond visible target material.'),
        ('UNCERTAINTY', 'Amber identifies unresolved fields. In the old-series views, a broad bright band may be a front rim or a projecting support. The original images and wording do not settle ownership, so inherited geometry is retained and flagged.'),
        ('VIEWING AIDS', 'Unmodified source pixels are the evidence. Nearest-neighbor zoom and optional grayscale/autocontrast can expose existing transitions, but cannot recover hidden geometry or decide which physical part owns an edge. No restoration image is a training input.'),
        ('COMPARISON', 'A review of all photos does not certify every inherited label. Any unresolved training photo retains exact PR35 labels in both notebook arms. Development review edits are separate from original validation and both frozen scoring references.'),
    ]
    y = HEIGHT - 139
    for label, body in rules:
        text(c, label, 30, y, 11, MUTED, True)
        y = wrap(c, body, 195, y, WIDTH-230, 14, 21) - 18
    url = 'https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md'
    text(c, 'Primary source: original baomai data README, pinned commit 351eb3cf960c...', 30, 62, 10, MUTED)
    c.linkURL(url, (30, 57, WIDTH-30, 76), relative=0, thickness=0)


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
    for op in operations: op_map[op['image_id']].append(op)
    for case in status['cases']:
        if case['status'] != 'resolved': case_map[case['image_id']].append(case)
    counts = Counter(o['action'] for o in operations)

    def data(iid):
        r = by_id[iid]
        ip = root / 'data' / r['image']
        if sha(ip) != r['image_sha256']: raise ValueError(f'original image hash changed: {iid}')
        im = Image.open(ip).convert('RGB')
        bp = (root / PARENT['path']).parent / before_files[iid]['label_path']
        ap = (root / REVIEW['path']).parent / after_files[iid]['label_path']
        before = parse_pose_labels(bp.read_text(), *im.size, str(bp))
        after = parse_pose_labels(ap.read_text(), *im.size, str(ap))
        edits = {o['base_row']: o for o in op_map[iid] if o['action'] != 'add'}
        fields, held = {}, {}
        for identity in maps[iid]['rows']:
            row, base = identity['current_row'], identity.get('base_row')
            if identity.get('addition_id'): fields[row] = ['box', 'left', 'right']
            elif base in edits: fields[row] = edits[base].get('changed_groups', ['box','left','right'])
            flags = {g for case in case_map[iid] if base in case['base_rows']
                     for g in case.get('fields', ['box','left','right'])}
            if flags: held[row] = sorted(flags)
        return im, before, after, fields, held

    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle('All 157 cassette photographs / Round 6 review')
    c.setAuthor('Siuuuers/dl / original-image annotation review')
    c.setSubject('Complete labels with this pass\'s changed fields highlighted; unresolved ownership recorded.')
    pages = []
    header(c, 'All 157 photographs, one consistent rule',
           'Complete current labeling / Corrections highlighted against the immutable round 5 baseline.', 1)
    text(c, f'{len(op_map)} changed photos', 30, HEIGHT-140, 29, INK, True)
    text(c, f'{counts["replace"]} edited rows / {number_word(counts["add"], "addition")} / {number_word(counts["delete"], "deletion")}',
         445, HEIGHT-136, 20, INK, True)
    y = HEIGHT - 185
    for label, body, col in [
        ('MAGENTA + *', 'Only the field changed in this pass is magenta. A changed point does not recolor its unchanged box or other point.', MAGENTA),
        ('CYAN', 'Inherited field, inspected during this pass. Its color does not claim that the coordinate is exact.', CYAN),
        ('AMBER', 'Unresolved physical ownership or landmark availability. The case ledger records the reason and training treatment.', AMBER),
        ('CIRCLE / SQUARE', 'Image-left / image-right lower-front corner. No marker means that point is unavailable.', MUTED),
        ('RED DASHES', 'Deleted base row, displayed only in a before/detail panel; it is absent from current training labels.', RED),
    ]:
        text(c, label, 30, y, 12, col, True)
        y = wrap(c, body, 230, y, WIDTH-265, 14, 21)-18
    y = wrap(c, f'The source cohort is unchanged: 137 training photos and 20 development photos, including 16 empty scenes. '
             f'The complete review contains {review["expected_counts"]["train"]["objects"]:,} training fronts and '
             f'{review["expected_counts"]["val"]["objects"]:,} development fronts. All 157 images received a fresh visual pass.',
             30, y-7, WIDTH-60, 15, 23)
    y = wrap(c, 'Full-photo pages use CURRENT r6 row numbers. Operations use BASE r5 row numbers. '
             'Deletion or addition can shift later row numbers; use the recorded row map rather than an older screenshot number.',
             30, y-14, WIDTH-60, 14, 21)
    y = wrap(c, 'Review completion is not physical certification. Uncertain cases remain visible in this PDF. '
             'No training has been run and no model has been promoted by this pass; PR35 remains selected.',
             30, y-14, WIDTH-60, 14, 21)
    text(c, 'Use the PDF bookmarks or the linked photo index to jump to a photograph.', 30, 72, 12, MUTED)
    pages.append({'page':1,'type':'cover'}); c.showPage()
    definition_page(c, 2); pages.append({'page':2,'type':'physical_definition'}); c.showPage()

    header(c, 'Photo index', '* = changed photograph / ? = unresolved case / Numbers at right are PDF pages.', 3)
    per_column = (len(ids)+2)//3
    column_width = (WIDTH-60)/3
    for i, iid in enumerate(ids):
        col, line = divmod(i, per_column)
        x, y = 30+col*column_width, HEIGHT-123-line*12.2
        suffix = (' *' if iid in op_map else '') + (' ?' if iid in case_map else '')
        text(c, iid+suffix, x, y, 9.3, INK, True)
        text(c, f'{after_files[iid]["objects"]} fronts', x+159, y, 9.3, MUTED)
        text(c, str(i+4), x+column_width-43, y, 9.3, MUTED)
        c.linkRect('', 'photo-'+iid, (x,y-3,x+column_width-17,y+10), relative=0, thickness=0)
    pages.append({'page':3,'type':'photo_index'}); c.showPage()

    for i, iid in enumerate(ids):
        number = i+4
        im, before, after, fields, held = data(iid)
        per = Counter(o['action'] for o in op_map[iid])
        split_name = 'TRAIN REVIEW' if by_id[iid]['split']=='train' else 'DEVELOPMENT REVIEW / EXPLORATORY'
        unresolved = len(case_map[iid])
        header(c, iid, f'{split_name} / All {len(after)} current fronts / {number_word(per["replace"], "edit")}, {number_word(per["add"], "addition")}, {number_word(per["delete"], "deletion")} / {unresolved} unresolved '+('case' if unresolved == 1 else 'cases'), number)
        c.bookmarkPage('photo-'+iid); c.addOutlineEntry(iid, 'photo-'+iid, level=0, closed=False)
        shown = panel(c, im, after, 30, 117, WIDTH-60, HEIGHT-213, fields=fields, held=held)
        edited = [o['base_row'] for o in op_map[iid] if o['action']=='replace']
        caption = 'Changed BASE rows: '+ranges(edited)+'.'
        if per['delete']: caption += ' Deleted BASE rows: '+ranges(maps[iid]['deleted_base_rows'])+'.'
        if per['add']: caption += ' Added CURRENT rows: '+ranges(k for k in fields if 'addition_id' in maps[iid]['rows'][k-1])+'.'
        if held: caption += ' Unresolved CURRENT rows: '+ranges(held)+'.'
        if any(not item['base_rows'] for item in case_map[iid]):
            caption += ' A possible additional-front region remains unresolved; see the case ledger.'
        if not after: caption = 'Reviewed empty scene. No identifiable cassette front was found; the empty label file is retained.'
        y = wrap(c, caption, 30, 96, WIDTH-60, 10.5, 15)
        note = coverage.get('photo_notes', {}).get(iid, '')
        if isinstance(note,list): note = ' '.join(note)
        if note: wrap(c, note, 30, y-3, WIDTH-60, 9.5, 14, MUTED)
        pages.append({'page':number,'type':'complete_photo','image_id':iid,**shown,
                      'deleted_base_rows':maps[iid]['deleted_base_rows'],
                      'source_image_sha256':by_id[iid]['image_sha256'],
                      'label_sha256':after_files[iid]['label_sha256']})
        c.showPage()

    for detail in coverage.get('pdf_details', []):
        number = len(pages)+1
        iid, crop = detail['image_id'], detail['crop']
        im, before, after, fields, held = data(iid)
        header(c, detail['title'], iid+' / Original pixels, BASE r5 labels and CURRENT r6 labels.', number)
        pw, yy, ph = (WIDTH-90)/3, 206, HEIGHT-345
        for x, label in [(30,'ORIGINAL'),(45+pw,'BEFORE / BASE'),(60+2*pw,'CURRENT')]:
            text(c, label, x, HEIGHT-119, 12, MUTED, True)
        panel(c, im, before, 30, yy, pw, ph, crop=crop, raw=True)
        panel(c, im, before, 45+pw, yy, pw, ph, crop=crop, deleted=maps[iid]['deleted_base_rows'])
        panel(c, im, after, 60+2*pw, yy, pw, ph, crop=crop, fields=fields, held=held)
        wrap(c, detail['note'], 30, 176, WIDTH-60, 14, 21)
        pages.append({'page':number,'type':'original_before_current_detail','image_id':iid,
                      'crop':crop,'base_rows':detail.get('base_rows',[])})
        c.showPage()
    c.save()
    photos = [p for p in pages if p['type']=='complete_photo']
    assert len(photos)==157 and {p['image_id'] for p in photos}==set(ids)
    assert all(p['displayed_rows']==list(range(1,p['objects']+1)) for p in photos)
    audit = {'schema_version':1,'kind':'all157_front_review_r6_pdf','review':REVIEW,'parent':PARENT,
             'pages':pages,'page_count':len(pages),'complete_photo_pages':157,
             'all_current_rows_displayed':True,'field_level_change_highlighting':True,
             'original_images_unchanged':True,'pdf_sha256':sha(output),'pdf_bytes':output.stat().st_size,
             'visual_layout_qa':'pending separate rendered-page inspection'}
    audit_output = Path(audit_output); audit_output.parent.mkdir(parents=True,exist_ok=True)
    audit_output.write_text(json.dumps(audit,indent=2)+'\n')
    return {k:v for k,v in audit.items() if k!='pages'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root,args.output,args.audit_output),indent=2))


if __name__=='__main__': main()
