#!/usr/bin/env python3
"""Render one focused PDF: this pass's changes and every remaining held row."""

import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from evaluate_pose import parse_pose_labels


W, H = A4
M = 28
INK = colors.HexColor('#182630')
MUTED = colors.HexColor('#536775')
CYAN = colors.HexColor('#12d4e7')
AMBER = colors.HexColor('#ffb530')
GREEN = colors.HexColor('#3ee58a')
MAGENTA = colors.HexColor('#f56bdd')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def bound(root, item):
    path = (root / item['path']).resolve()
    if not path.is_relative_to(root) or sha(path) != item['sha256']:
        raise ValueError('Source identity differs: ' + item['path'])
    return path


def descriptor(root, path):
    return {'path': path.relative_to(root).as_posix(), 'sha256': sha(path)}


def ranges(values):
    values = sorted(set(values))
    result = []
    for value in values:
        if result and value == result[-1][-1] + 1:
            result[-1].append(value)
        else:
            result.append([value])
    return ', '.join(str(group[0]) if len(group) == 1 else f'{group[0]}–{group[-1]}'
                     for group in result) or 'none'


def lines(text, width, size):
    result = []
    for paragraph in text.split('\n'):
        line = ''
        for word in paragraph.split():
            trial = (line + ' ' + word).strip()
            if line and pdfmetrics.stringWidth(trial, 'Review', size) > width:
                result.append(line)
                line = word
            else:
                line = trial
        result.append(line)
    return result


def paragraph(c, text, x, y, width, size=10, leading=14, maximum=None):
    wrapped = lines(text, width, size)
    if maximum and len(wrapped) > maximum:
        wrapped = wrapped[:maximum]
        wrapped[-1] = wrapped[-1].rsplit(' ', 1)[0] + ' …'
    c.setFont('Review', size)
    c.setFillColor(INK)
    for line in wrapped:
        c.drawString(x, y, line)
        y -= leading
    return y


def heading(c, title, subtitle, page, section):
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFont('ReviewBold', 22)
    c.setFillColor(INK)
    c.drawString(M, H - 41, title)
    c.setFont('Review', 9)
    c.setFillColor(MUTED)
    c.drawString(M, H - 60, subtitle)
    c.drawRightString(W - M, H - 20, section)
    c.setStrokeColor(colors.HexColor('#d8e2e9'))
    c.line(M, 30, W - M, 30)
    c.setFont('Review', 7.5)
    c.drawString(M, 17, '18 November 2026 / Round 2 / Saved labels; no model predictions')
    c.drawRightString(W - M, 17, str(page))
    c.bookmarkPage(f'page-{page}')
    c.addOutlineEntry(f'{page}. {title}', f'page-{page}', 0)


def crop_for(boxes, size):
    iw, ih = size
    x0 = max(0, int(min(b[0] for b in boxes)) - 18)
    y0 = max(0, int(min(b[1] for b in boxes)) - 18)
    x1 = min(iw, int(max(b[2] for b in boxes)) + 19)
    y1 = min(ih, int(max(b[3] for b in boxes)) + 19)
    if x1 - x0 < 180:
        mid = (x0 + x1) / 2
        x0 = max(0, min(iw - 180, int(mid - 90)))
        x1 = min(iw, x0 + 180)
    if y1 - y0 < 95:
        mid = (y0 + y1) / 2
        y0 = max(0, min(ih - 95, int(mid - 47)))
        y1 = min(ih, y0 + 95)
    return [x0, y0, x1, y1]


def image_panel(c, source, crop, objects, selected, held, top, caption):
    c.setFillColor(INK)
    c.setFont('ReviewBold', 10)
    c.drawString(M, top, caption)
    image = source.crop(crop)
    scale = min((W - 2 * M) / image.width, 257 / image.height)
    width, height = image.width * scale, image.height * scale
    left, bottom = (W - width) / 2, top - 12 - height
    buffer = BytesIO()
    image.save(buffer, 'JPEG', quality=96, subsampling=0)
    buffer.seek(0)
    c.drawImage(ImageReader(buffer), left, bottom, width, height)
    def point(x, y):
        return left + (x - crop[0]) * scale, bottom + height - (y - crop[1]) * scale
    c.saveState()
    path = c.beginPath()
    path.rect(left, bottom, width, height)
    c.clipPath(path, stroke=0)
    for row in sorted(selected):
        obj = objects[row - 1]
        x0, y0, x1, y1 = obj['bbox_xyxy']
        tint = AMBER if row in held else CYAN
        c.setStrokeColor(tint)
        c.setLineWidth(.8)
        c.setDash(3, 2) if row in held else c.setDash()
        c.rect(*point(x0, y1), (x1 - x0) * scale, (y1 - y0) * scale, stroke=1, fill=0)
        c.setDash()
        points = obj['keypoints']
        if all(p[2] == 2 for p in points):
            c.setStrokeColor(GREEN)
            c.setLineWidth(.55)
            c.line(*point(*points[0][:2]), *point(*points[1][:2]))
        for p, colour in zip(points, (GREEN, MAGENTA)):
            if p[2] == 2:
                c.setFillColor(colour)
                c.setStrokeColor(INK)
                c.setLineWidth(.3)
                c.circle(*point(*p[:2]), 1.65, stroke=1, fill=1)
        font_size = max(5.2, min(9, (y1 - y0) * scale * .7))
        tx, ty = point(max(x0, crop[0]), max(y0, crop[1]))
        tx += 1
        ty -= font_size + 1
        text = str(row)
        tw = pdfmetrics.stringWidth(text, 'ReviewBold', font_size)
        c.setFillColor(INK)
        c.rect(tx - .7, ty - .6, tw + 1.4, font_size + 1.4, fill=1, stroke=0)
        c.setFont('ReviewBold', font_size)
        c.setFillColor(tint)
        c.drawString(tx, ty, text)
    c.restoreState()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    candidate_path = (root / args.candidate).resolve()
    candidate = read(candidate_path)
    parent_path = bound(root, candidate['parent'])
    parent = read(parent_path)
    status_path = bound(root, candidate['review_status'])
    status = read(status_path)
    review_path = bound(root, candidate['review'])
    review = read(review_path)
    reference_path = bound(root, candidate['development_reference'])
    reference = read(reference_path)
    source_path = root / 'data/manifest.json'
    if sha(source_path) != candidate['source_manifest_sha256']:
        raise ValueError('Original source manifest differs')
    records = {r['id']: r for r in read(source_path)['records']}
    current_files = {f['image_id']: f for f in candidate['files']}
    parent_files = {f['image_id']: f for f in parent['files']}
    dev_files = {f['image_id']: f for f in reference['files']}
    grouped = defaultdict(list)
    for op in candidate['operations']:
        grouped[op['image_id']].append(op)
    holds = defaultdict(set)
    reasons = defaultdict(list)
    for case in status['cases']:
        if case['status'] != 'resolved':
            key = ('train', case['image_id'])
            holds[key].update(case['held_current_rows'])
            reasons[key].append(case['atlas_note'])
    for item in reference['remaining_current_rows']:
        key = ('dev', item['image_id'])
        holds[key].add(item['row'])
        reasons[key].append(item.get('reason', 'Front identity or exact visible extent remains unresolved.'))
    inherited_status = read(bound(root, parent['review_status']))
    old_holds = defaultdict(set)
    for case in inherited_status['cases']:
        if case['status'] != 'resolved':
            old_holds[case['image_id']].update(case['held_current_rows'])
    extras = {'new_000030': [30], 'new_000041': [15]}
    pages = []
    sources = {}
    def load_objects(iid, directory, file_info, key):
        p = directory / file_info['candidate_label' if key == 'train' else 'label_path']
        expected = file_info['candidate_label_sha256' if key == 'train' else 'label_sha256']
        if sha(p) != expected:
            raise ValueError('Label identity differs: ' + str(p))
        sources[str(p.relative_to(root))] = expected
        return parse_pose_labels(p.read_text(), *records[iid]['original_size'], iid)
    for iid, ops in sorted(grouped.items()):
        before = load_objects(iid, parent_path.parent, parent_files[iid], 'train')
        after = load_objects(iid, candidate_path.parent, current_files[iid], 'train')
        selected_before = {op['base_row'] for op in ops if op['base_row'] is not None} | set(extras.get(iid, []))
        mapping = {m['base_row']: m['candidate_row'] for m in current_files[iid]['row_map'] if m.get('base_row')}
        selected_after = {mapping[row] for row in selected_before if mapping.get(row) is not None}
        boxes = [before[row - 1]['bbox_xyxy'] for row in selected_before] + [after[row - 1]['bbox_xyxy'] for row in selected_after]
        pages.append({'kind': 'change', 'image_id': iid, 'before': before, 'after': after,
                      'selected_before': selected_before, 'selected_after': selected_after,
                      'before_holds': old_holds[iid], 'after_holds': holds[('train', iid)],
                      'crop': crop_for(boxes, records[iid]['original_size']), 'operations': ops})
    for (split, iid), rows in sorted(holds.items()):
        if not rows:
            continue
        objects = (load_objects(iid, candidate_path.parent, current_files[iid], 'train') if split == 'train'
                   else load_objects(iid, reference_path.parent, dev_files[iid], 'dev'))
        boxes = [objects[row - 1]['bbox_xyxy'] for row in rows]
        if split == 'train' and iid in ('new_000030', 'new_000041'):
            old = load_objects(iid, parent_path.parent, parent_files[iid], 'train')
            boxes += [old[row - 1]['bbox_xyxy'] for row in rows]
        pages.append({'kind': 'hold', 'split': split, 'image_id': iid, 'after': objects,
                      'selected_after': rows, 'after_holds': rows,
                      'crop': crop_for(boxes, records[iid]['original_size']),
                      'notes': list(dict.fromkeys(reasons[(split, iid)]))})
    for offset, page in enumerate(pages, 3):
        page['page'] = offset
        record = records[page['image_id']]
        p = root / 'data' / record['image']
        if sha(p) != record['image_sha256']:
            raise ValueError('Original pixels differ: ' + str(p))
        sources[str(p.relative_to(root))] = record['image_sha256']
    pdfmetrics.registerFont(TTFont('Review', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('ReviewBold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(args.output), pagesize=A4, invariant=1, pageCompression=1)
    c.setTitle('Cassette front refinements — 18 November 2026, round 2')
    heading(c, 'Cassette front refinements', 'Only this pass’s changes and the remaining amber rows', 1, 'ROUND 2')
    counts = candidate['operation_counts']
    paragraph(c, f"{counts.get('replace', 0)} rows refined. {counts.get('delete', 0)} row deleted.", M, H - 112, W - 2*M, 20, 27)
    paragraph(c, f"{candidate['expected_counts']['train']['objects']:,} fronts in the complete 137-photo training review. The separate development reference remains {reference['counts']['objects']} fronts in 20 photos.", M, H - 179, W - 2*M, 12, 18)
    y = paragraph(c, 'The following pages compare the previous saved boxes with the new boxes. Only the rows being discussed are overlaid; other labels remain in the dataset. The later amber section shows an original crop above its current saved annotation.', M, H - 251, W - 2*M, 11, 17)
    y = paragraph(c, 'Cyan: no recorded open hold for that row. Amber dashed: a remaining physical identity or extent question. Green and magenta points: the actual lower-left and lower-right front corners when locatable. No hidden corner is inferred.', M, y - 20, W - 2*M, 11, 17)
    y = paragraph(c, 'A box follows visible front-face material. It may need expansion at one edge and trimming at another. Equal cassette dimensions do not imply equal image dimensions under perspective.', M, y - 20, W - 2*M, 11, 17)
    y = paragraph(c, f"{status['remaining_all_case_groups']} training case groups remain across {status['remaining_all_training_images']} photos. The development reference has {len(reference['remaining_current_rows'])} held rows across {len({x['image_id'] for x in reference['remaining_current_rows']})} photos. These counts are not an accuracy score.", M, y - 20, W - 2*M, 11, 17)
    y = paragraph(c, 'Notebook scope: detect currently visible fronts and locatable corners. The controlled experiment retains exact PR35 labels in both arms for every photo with a training hold, including 030, 034 and 041. This review does not promote a model or certify every annotation.', M, y - 20, W - 2*M, 11, 17)
    paragraph(c, 'Review revision: ' + candidate['candidate_id'], M, 65, W - 2*M, 8, 11)
    c.showPage()
    heading(c, 'Choose a review page', 'Current saved row numbers appear on the image overlays.', 2, 'INDEX')
    for kind, x in [('change', M), ('hold', W/2 + 8)]:
        c.setFont('ReviewBold', 11); c.setFillColor(INK)
        c.drawString(x, H - 99, 'This pass’s changes' if kind == 'change' else 'Remaining amber rows')
        y = H - 125
        for item in [p for p in pages if p['kind'] == kind]:
            c.setFont('ReviewBold', 9)
            c.drawString(x, y, str(item['page']))
            c.drawString(x + 23, y, item['image_id'])
            c.setFont('Review', 8)
            label = (str(len(item['operations'])) + (' operation' if len(item['operations']) == 1 else ' operations') if kind == 'change'
                     else item['split'] + ': ' + ranges(item['selected_after']))
            c.drawString(x + 116, y, label)
            c.linkRect('', f"page-{item['page']}", (x, y-3, x+247, y+10), relative=0, thickness=0)
            y -= 20
        if y < 90:
            raise ValueError('Index exceeds one page; paginate explicitly')
    paragraph(c, 'A label can have a valid visible-front box while neither actual cassette corner can be located. The small aperture boundaries belong to the gripper, not to the cassette corners.', M, 83, W-2*M, 9, 13)
    c.showPage()
    audit_pages = []
    for item in pages:
        iid, record = item['image_id'], records[item['image_id']]
        with Image.open(root / 'data' / record['image']) as im:
            image = im.convert('RGB')
        heading(c, iid, 'Only selected rows are overlaid. Original pixels are unchanged.', item['page'], 'CHANGES' if item['kind'] == 'change' else 'AMBER / ' + item['split'].upper())
        if item['kind'] == 'change':
            image_panel(c, image, item['crop'], item['before'], item['selected_before'], item['before_holds'], H-91, 'Before — previous review')
            image_panel(c, image, item['crop'], item['after'], item['selected_after'], item['after_holds'], H-378, 'Now — current saved labels')
            selected = ranges(item['selected_before'])
            text = f'Previous rows: {selected}. Current rows: {ranges(item["selected_after"])}.'
            if iid == 'new_000049':
                text += ' Former row 18 is deleted as you confirmed. Later rows shift up by one; stable historical identities remain in the manifest.'
            elif iid in ('new_000030', 'new_000041'):
                text += ' The upper box now follows definite aperture material. Pale shoulder ownership is still unresolved, so that upper row remains amber. The lower front is retained unchanged.'
            elif iid == 'new_000073':
                text += ' Rows 8/12 divide the third and fourth fronts. The dark and bright parts of the fourth aperture face belong together; the interior shadow is not another cassette seam.'
            elif iid in {f'new_{n:06}' for n in range(94,100)}:
                text += ' Restore the omitted upper portions of rows 4/15 while trimming unrelated margins. Both front-corner pairs are retained.'
            else:
                text += ' Extents follow the observed front contours. Existing corner tokens remain unchanged; this is not a uniform shrink or padding adjustment.'
        else:
            image_panel(c, image, item['crop'], [], [], [], H-91, 'Original crop — no overlay')
            image_panel(c, image, item['crop'], item['after'], item['selected_after'], item['after_holds'], H-378, 'Current saved labels — unresolved rows in amber')
            text = 'Rows ' + ranges(item['selected_after']) + '. ' + ' '.join(item['notes'])
        paragraph(c, text, M, 155, W-2*M, 9, 12.5, maximum=7)
        paragraph(c, 'Crop in original pixels: ' + str(item['crop']) + '. Detailed reasons and row mappings are in the repository review ledger.', M, 53, W-2*M, 7.5, 10)
        audit_pages.append({k:v for k,v in item.items() if k not in ('before','after','operations','selected_before','selected_after','before_holds','after_holds')}
                           | {'rows_before': sorted(item.get('selected_before', [])), 'rows_after': sorted(item['selected_after'])})
        c.showPage()
    c.save()
    audit = {'schema_version':1, 'kind':'focused_front_refinement_review',
             'pdf_filename':args.output.name, 'pdf_sha256':sha(args.output), 'pdf_bytes':args.output.stat().st_size,
             'page_count':len(pages)+2, 'changed_photo_pages':len(grouped), 'held_photo_pages':len(pages)-len(grouped),
             'candidate':descriptor(root,candidate_path), 'parent':descriptor(root,parent_path),
             'status':descriptor(root,status_path), 'review':descriptor(root,review_path),
             'development_reference':descriptor(root,reference_path), 'generator':descriptor(root,Path(__file__).resolve()),
             'sources':[{'path':p,'sha256':h} for p,h in sorted(sources.items())], 'pages':audit_pages,
             'limits':['Only this correction pass and current held rows are shown; this is not a complete all-photo atlas.',
                       'Before/after panels show saved annotations, not predictions. Cyan is not certification.',
                       'Manual soft-edge bounds retain a few original pixels of uncertainty.']}
    args.audit.parent.mkdir(parents=True,exist_ok=True)
    args.audit.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ('pdf_filename','pdf_sha256','pdf_bytes','page_count','changed_photo_pages','held_photo_pages')},indent=2))


if __name__ == '__main__':
    main()
