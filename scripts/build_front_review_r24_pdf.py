#!/usr/bin/env python3
"""Render all nonempty new-series R24 labels and the accepted R23-to-R24 delta."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

from PIL import Image
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from build_front_review_r16_pdf import load_inputs
from build_front_review_r9_pdf import (
    WIDTH, HEIGHT, INK, MUTED, CYAN, MAGENTA, TAG_SIZE, TAG_HEIGHT,
    digest, read_json, register_fonts, text, wrap, ranges, place_tags, intersects,
)

REVIEW = 'data/reviews/fronts-2026-11-23-r24'
LEDGER = 'results/review/front-review-20261123-r24'
GREEN = HexColor('#20e6a3')
PROVENANCE = ('Every nonempty new-series photograph appears once in the atlas with all current R24 rows. '
              'Highlights identify only the saved R23-to-R24 changes. Old-series and empty photographs '
              'are omitted from this PDF, not deleted from the dataset. Original PNG pixels are embedded losslessly.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_descriptor(root, path):
    return {'path': path.relative_to(root).as_posix(), 'sha256': digest(path)}


def prepare(root):
    data = load_inputs(root, root / REVIEW, root / LEDGER, all_nonempty=True)
    require((data['round_number'], data['parent_round']) == (24, 23), 'Wrong review rounds')
    data['ids'] = sorted(i for i, f in data['files'].items() if i.startswith('new_') and f['rows'])
    audit_path = root / LEDGER / 'point-review.json'
    require(source_descriptor(root, audit_path) in data['manifest']['review_evidence'], 'Point ledger hash differs')
    audit = read_json(audit_path)
    data['point_changes'] = {}
    for item in audit['rows']:
        iid, row = item['image_id'], item['row']
        require(data['files'][iid]['raw'][row - 1] == item['after_line'], 'Saved row differs from point ledger')
        changed = {}
        for side in ('left', 'right'):
            entry = item['point_reviews'][side]
            if entry['decision'] == 'corrected':
                changed[side] = entry
        require(set(changed) == data['edits'].get(iid, {}).get(row, set()), 'Point ledger and exact delta disagree')
        data['point_changes'].setdefault(iid, {})[row] = changed
    require(set(data['point_changes']) == set(data['edits']), 'Point ledger does not cover every edited photograph')
    require(sum(len(v) for v in data['point_changes'].values()) == 22, 'Expected 22 amended rows')
    require(sum(len(s) for r in data['point_changes'].values() for s in r.values()) == 26,
            'Expected 26 amended points')
    for iid, edits in data['edits'].items():
        require(iid in data['ids'] and set(edits) == set(data['point_changes'][iid]), 'Edited row coverage differs')
        require(all(set(groups) <= {'left', 'right'} for groups in edits.values()), 'R24 only changes points')
    require(not data['deleted'], 'R24 should not delete rows')
    data['audit'] = audit
    data['audit_path'] = audit_path
    rendering = read_json(root / LEDGER / 'rendering-manifest.json')
    require(rendering['point_review'] == source_descriptor(root, audit_path), 'Comparison-sheet ledger differs')
    data['comparison_sheets'] = {a['image_id']: a for a in rendering['artifacts']}
    require(set(data['comparison_sheets']) == set(data['point_changes']), 'Comparison-sheet coverage differs')
    for entry in data['comparison_sheets'].values():
        require(digest(root / entry['path']) == entry['sha256'], 'Accepted comparison-sheet hash differs')
    data['summary'] = {
        'included_photos': len(data['ids']),
        'included_fronts': sum(len(data['files'][i]['rows']) for i in data['ids']),
        'excluded_old_photos': sum(i.startswith('old_') for i in data['files']),
        'excluded_empty_new_photos': sum(i.startswith('new_') and not v['rows'] for i, v in data['files'].items()),
        'changed_photos': len(data['point_changes']), 'changed_rows': 22, 'changed_points': 26,
        'left_point_changes': 17, 'right_point_changes': 9,
        'geometry_modified_by_renderer': False,
    }
    return data


def page_header(c, title, subtitle, page, total):
    c.setFillColor(white)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    text(c, 'CASSETTE FRONT REVIEW / ROUND 24', 30, HEIGHT - 28, 10, MUTED, True)
    text(c, title, 30, HEIGHT - 60, 24, INK, True)
    text(c, subtitle, 30, HEIGHT - 80, 10.5, MUTED)
    text(c, 'R23 -> R24 / Original pixels / Saved one-based row numbers', 30, 18, 8.7, MUTED)
    c.setFont('Review', 9)
    c.setFillColor(MUTED)
    c.drawRightString(WIDTH - 30, 18, f'{page} / {total}')
    if page != 2:
        c.linkRect('', 'index', (30, HEIGHT - 33, 340, HEIGHT - 17), thickness=0)
        text(c, 'Tap the review heading to return to the photo index', WIDTH - 375, HEIGHT - 28, 8.5, MUTED)


def legend(c, y, compact=False):
    c.setFillColor(INK)
    c.roundRect(30, y - 11, WIDTH - 60, 28 if compact else 39, 4, fill=1, stroke=0)
    size = 9 if compact else 11
    text(c, 'CYAN: current, unchanged', 44, y, size, CYAN, True)
    text(c, 'GREEN + *: corrected in R24', 332, y, size, GREEN, True)
    text(c, 'MAGENTA: previous R23 point', 655, y, size, MAGENTA, True)
    text(c, 'L circle / R square', 970, y, size, white)


def load_image(root, data, iid):
    record = data['records'][iid]
    path = root / 'data' / record['image']
    require(digest(path) == record['image_sha256'], f'Original PNG hash differs: {iid}')
    with Image.open(path) as original:
        image = original.convert('RGB')
    require(list(image.size) == record['original_size'], f'Original dimensions differ: {iid}')
    return image


def cover(c, root, data, pages, appendix, total):
    page_header(c, 'All labeled new-series photographs',
                'Complete current R24 atlas, with the preceding pass highlighted and four accepted comparison sheets.', 1, total)
    c.bookmarkPage('overview')
    c.addOutlineEntry('Overview and R24 changes', 'overview', level=0)
    summary = data['summary']
    for x, value, label in [(30, summary['included_photos'], 'nonempty new-series photos'),
                            (365, f"{summary['included_fronts']:,}", 'current cassette fronts'),
                            (760, '26 / 22 / 4', 'changed points / rows / photos')]:
        text(c, value, x, HEIGHT - 145, 34, INK, True)
        text(c, label, x, HEIGHT - 169, 12, MUTED)
    wrap(c, PROVENANCE, 30, HEIGHT - 205, WIDTH - 60, 12, 17)
    legend(c, HEIGHT - 263)
    text(c, 'THE FOUR CHANGED PHOTOGRAPHS / tap an atlas or comparison page', 30, HEIGHT - 303, 11, MUTED, True)
    width, gap = (WIDTH - 60 - 3 * 18) / 4, 18
    for col, iid in enumerate(sorted(data['point_changes'])):
        x, y = 30 + col * (width + gap), HEIGHT - 485
        image = load_image(root, data, iid)
        c.drawImage(ImageReader(image), x, y, width, width * image.height / image.width)
        text(c, iid, x, y - 25, 16, INK, True)
        rows = data['point_changes'][iid]
        changes = sum(len(v) for v in rows.values())
        text(c, f'{len(rows)} rows / {changes} points', x, y - 45, 11, MUTED)
        text(c, 'Rows: ' + ranges(rows), x, y - 63, 10.5)
        text(c, f'Atlas p. {pages[iid]}', x, y - 89, 12, INK, True)
        c.linkRect('', 'photo-' + iid, (x, y - 94, x + 115, y - 77), thickness=0)
        text(c, f'Compare p. {appendix[iid]}', x + 123, y - 89, 12, INK, True)
        c.linkRect('', 'comparison-' + iid, (x + 120, y - 94, x + width, y - 77), thickness=0)
    note_y = 214
    notes = [
        'Every atlas page shows all saved boxes and visible endpoints, with an unmarked original thumbnail for context.',
        'A star marks a changed row. Green rings mark the exact saved R24 coordinates; magenta marks their previous positions.',
        'Row identities are unchanged. Boxes, classes, visibility, object counts and the two accepted standard rows are preserved.',
        'Photo 010 is a development-review image: its revised points do not alter training labels or the frozen scoring references.',
        'The full photo index is on page 2. PDF bookmarks provide direct navigation to each photo and comparison sheet.',
    ]
    for line in notes:
        note_y = wrap(c, line, 30, note_y, WIDTH - 60, 11, 16) - 8
    text(c, 'Color records edit status; it does not certify every inherited annotation.', 30, 60, 10, MUTED)


def photo_index(c, data, pages, total):
    page_header(c, 'Photo index', '* = changed in R24 / Numbers at right are PDF pages / Tap a name to jump', 2, total)
    c.bookmarkPage('index')
    c.addOutlineEntry('Photo index', 'index', level=0)
    per_column = (len(data['ids']) + 2) // 3
    width = (WIDTH - 60) / 3
    for n, iid in enumerate(data['ids']):
        col, row = divmod(n, per_column)
        x, y = 30 + col * width, HEIGHT - 120 - row * 17
        changed = iid in data['point_changes']
        text(c, iid + (' *' if changed else ''), x, y, 11.2, INK, True)
        text(c, str(len(data['files'][iid]['rows'])) + ' fronts', x + 170, y, 10.5, MUTED)
        text(c, pages[iid], x + width - 49, y, 11.2, INK, True)
        c.linkRect('', 'photo-' + iid, (x, y - 4, x + width - 20, y + 12), thickness=0)
    text(c, f"{data['summary']['excluded_empty_new_photos']} empty new-series photos and "
         f"{data['summary']['excluded_old_photos']} old-series photos omitted from this requested atlas.",
         30, 55, 10.5, MUTED)


def draw_annotated_photo(c, image, rows, changes):
    area = (30, 132, WIDTH - 60, HEIGHT - 244)
    scale = min(area[2] / image.width, area[3] / image.height)
    iw, ih = image.width * scale, image.height * scale
    x, y = area[0] + (area[2] - iw) / 2, area[1] + (area[3] - ih) / 2
    bounds = (x, y, x + iw, y + ih)
    c.drawImage(ImageReader(image), x, y, iw, ih)

    def point(px, py):
        return x + px * scale, y + ih - py * scale

    specs, protected, small_fronts = [], [], []
    c.saveState()
    clip = c.beginPath()
    clip.rect(x, y, iw, ih)
    c.clipPath(clip, stroke=0)
    for row, v in enumerate(rows, 1):
        left, top = (v[1] - v[3] / 2) * image.width, (v[2] - v[4] / 2) * image.height
        right, bottom = (v[1] + v[3] / 2) * image.width, (v[2] + v[4] / 2) * image.height
        qx, qy = point(left, bottom)
        bw, bh = (right - left) * scale, (bottom - top) * scale
        c.setStrokeColor(CYAN)
        c.setLineWidth(.45)
        c.rect(qx, qy, bw, bh, stroke=1, fill=0)
        for side, start in (('left', 5), ('right', 8)):
            if not v[start + 2]:
                continue
            px, py = point(v[start] * image.width, v[start + 1] * image.height)
            protected.append((px, py))
            corrected = side in changes.get(row, {})
            radius = 3.2 if corrected else 1.6
            c.setStrokeColor(GREEN if corrected else CYAN)
            c.setLineWidth(1.0 if corrected else .65)
            if side == 'left':
                c.circle(px, py, radius, fill=0, stroke=1)
            else:
                c.rect(px - radius, py - radius, 2 * radius, 2 * radius, fill=0, stroke=1)
            if corrected:
                ox, oy = point(*changes[row][side]['before_pixel_xy'])
                protected.append((ox, oy))
                c.setStrokeColor(MAGENTA)
                c.setLineWidth(.8)
                c.line(ox - 1.5, oy - 1.5, ox + 1.5, oy + 1.5)
                c.line(ox - 1.5, oy + 1.5, ox + 1.5, oy - 1.5)
                c.setStrokeColor(GREEN)
                c.setLineWidth(.55)
                c.circle(px, py, 5.5, stroke=1, fill=0)
        label = str(row) + ('*' if row in changes else '')
        tw = pdfmetrics.stringWidth(label, 'Review-Bold', TAG_SIZE) + 5
        if bw <= tw * 2 and bh <= TAG_HEIGHT * 2:
            small_fronts.append((qx, qy, qx + bw, qy + bh))
            protected.extend((qx + bw * fx, qy + bh * fy) for fx in (0, .5, 1) for fy in (0, .5, 1))
        specs.append({'row': row, 'label': label, 'anchor': list(point(left, top)), 'width': tw,
                      'box_height': max(0, bh), 'changed': row in changes})
    c.restoreState()
    tags, collisions, clipped = place_tags(specs, bounds, protected)
    require(not any(intersects(region, tag['rect']) for region in small_fronts for tag in tags),
            'A row tag hides a small front')
    require(not any(tag['rect'][0] - 1 <= px <= tag['rect'][2] + 1 and
                    tag['rect'][1] - 1 <= py <= tag['rect'][3] + 1 for tag in tags for px, py in protected),
            'A row tag hides a corner marker')
    for tag in tags:
        color = GREEN if tag['changed'] else CYAN
        if tag['moved']:
            ax, ay = tag['anchor']
            ax, ay = max(x + .6, min(ax, x + iw - .6)), max(y + .6, min(ay, y + ih - .6))
            box = tag['rect']
            tx, ty = max(box[0], min(ax, box[2])), max(box[1], min(ay, box[3]))
            c.setStrokeColor(INK)
            c.setLineWidth(1.25)
            c.line(ax, ay, tx, ty)
            c.setStrokeColor(color)
            c.setLineWidth(.55)
            c.line(ax, ay, tx, ty)
    for tag in tags:
        x0, y0, x1, y1 = tag['rect']
        color = GREEN if tag['changed'] else CYAN
        c.setFillColor(INK)
        c.setStrokeColor(color)
        c.setLineWidth(.45)
        c.rect(x0, y0, x1 - x0, y1 - y0, fill=1, stroke=1)
        text(c, tag['label'], x0 + 2.5, y0 + 2.7, TAG_SIZE, color, True)
    return {'image_bounds_pt': list(bounds), 'displayed_rows': list(range(1, len(rows) + 1)),
            'tags': tags, 'tag_overlap_pairs': collisions, 'clipped_tag_rows': clipped,
            'protected_small_fronts': len(small_fronts), 'row_tags_cover_no_corner_markers': True}


def atlas_page(c, root, data, iid, number, appendix, total):
    item = data['files'][iid]
    changes = data['point_changes'].get(iid, {})
    split = 'Training review' if item['split'] == 'train' else 'Development review'
    subtitle = f'{split} / {len(item["rows"])} current fronts / ' + (
        f'{len(changes)} amended rows / {sum(len(v) for v in changes.values())} changed points' if changes
        else 'No saved changes from R23')
    page_header(c, iid, subtitle, number, total)
    c.bookmarkPage('photo-' + iid)
    c.addOutlineEntry(iid + (' - changed' if changes else ''), 'photo-' + iid, level=0)
    legend(c, HEIGHT - 101, compact=True)
    image = load_image(root, data, iid)
    visual = draw_annotated_photo(c, image, item['rows'], changes)
    c.drawImage(ImageReader(image), 30, 34, 128, 72)
    text(c, 'Unmarked original', 178, 100, 10, MUTED, True)
    if changes:
        text(c, 'Changed rows: ' + ranges(changes), 178, 82, 10.5, INK, True)
        text(c, 'Left: ' + ranges(r for r, s in changes.items() if 'left' in s) +
             '   /   Right: ' + ranges(r for r, s in changes.items() if 'right' in s), 178, 65, 10.1)
        text(c, f'Original / R23 / R24 magnified comparison: page {appendix[iid]}', 178, 47, 10.2, INK, True)
        c.linkRect('', 'comparison-' + iid, (176, 42, 735, 60), thickness=0)
    else:
        text(c, 'All current labels retained exactly from R23.', 178, 80, 10.5)
        text(c, 'No new annotation decisions are made by this rendering.', 178, 62, 10.2, MUTED)
    missing = {side: [r for r, v in enumerate(item['rows'], 1) if not v[start + 2]]
               for side, start in (('left', 5), ('right', 8))}
    text(c, 'MISSING ENDPOINTS / no marker drawn', 805, 100, 9.3, MUTED, True)
    y = 83
    for side in ('left', 'right'):
        y = wrap(c, side.capitalize() + ': ' + ranges(missing[side]), 805, y, WIDTH - 835, 8.8, 11.7)
    require(y >= 30, f'Endpoint note exceeds footer: {iid}')
    return {**visual, 'page': number, 'image_id': iid, 'split': item['split'],
            'label_sha256': item['sha256'], 'original_image_sha256': data['records'][iid]['image_sha256'],
            'changed_fields': {str(r): sorted(s) for r, s in changes.items()}, 'missing_point_rows': missing}


def appendix_page(c, root, data, iid, page, total):
    page_header(c, iid + ' / accepted corner comparison',
                'Original pixels beside previous R23 labels and current R24 labels. Exact saved comparison sheet.', page, total)
    c.bookmarkPage('comparison-' + iid)
    c.addOutlineEntry(iid + ' - magnified comparison', 'comparison-' + iid, level=0)
    entry = data['comparison_sheets'][iid]
    with Image.open(root / entry['path']) as source:
        image = source.convert('RGB')
    area = (30, 50, WIDTH - 60, HEIGHT - 150)
    scale = min(area[2] / image.width, area[3] / image.height)
    iw, ih = image.width * scale, image.height * scale
    c.drawImage(ImageReader(image), area[0] + (area[2] - iw) / 2,
                area[1] + (area[3] - ih) / 2, iw, ih)
    return {'page': page, 'image_id': iid, 'source': entry['path'], 'sha256': entry['sha256']}


def verify_pdf(root, output, data, pages, appendix, total):
    import fitz
    from pypdf import PdfReader
    checks = []
    reader = PdfReader(output)
    refs = {(page.indirect_reference.idnum, page.indirect_reference.generation): n + 1
            for n, page in enumerate(reader.pages)}
    targets = []
    for ref in reader.pages[1].get('/Annots', []):
        destination = ref.get_object()['/Dest'][0]
        targets.append(refs[(destination.idnum, destination.generation)])
    require(targets == list(pages.values()), 'Saved photo index differs')
    with fitz.open(output) as document:
        require(len(document) == total, 'Saved PDF page count differs')
        for iid, page_number in pages.items():
            page = document[page_number - 1]
            require(iid in page.get_text(), f'Saved page identity differs: {iid}')
            sources = page.get_images(full=True)
            require(len(sources) == 1, f'Expected a reused lossless source image: {iid}')
            pix = fitz.Pixmap(document, sources[0][0])
            image = load_image(root, data, iid)
            rgb = image.tobytes()
            require((pix.width, pix.height, pix.n) == (*image.size, 3) and pix.samples == rgb,
                    f'Saved pixels differ from original PNG: {iid}')
            require(len(page.get_image_rects(sources[0][0])) == 2, f'Unmarked original thumbnail missing: {iid}')
            checks.append({'image_id': iid, 'page': page_number,
                           'embedded_rgb_sha256': hashlib.sha256(rgb).hexdigest(),
                           'lossless_original_and_unmarked_thumbnail_verified': True})
        require(len(document.get_toc()) == total, 'PDF bookmark coverage differs')
        for iid, page_number in appendix.items():
            page = document[page_number - 1]
            require(iid in page.get_text(), f'Comparison page identity differs: {iid}')
            images = page.get_images(full=True)
            require(len(images) == 1, 'Expected one accepted comparison sheet')
            pix = fitz.Pixmap(document, images[0][0])
            with Image.open(root / data['comparison_sheets'][iid]['path']) as original:
                require(pix.samples == original.convert('RGB').tobytes(), 'Accepted comparison pixels changed')
    return checks


def build(root, output, receipt_path):
    register_fonts()
    data = prepare(root)
    pages = {iid: n + 3 for n, iid in enumerate(data['ids'])}
    appendix = {iid: len(data['ids']) + n + 3 for n, iid in enumerate(sorted(data['point_changes']))}
    total = 2 + len(pages) + len(appendix)
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle('R24 cassette fronts - all nonempty new-series photographs')
    c.setAuthor('Siuuuers/dl')
    c.setSubject(PROVENANCE)
    cover(c, root, data, pages, appendix, total)
    c.showPage()
    photo_index(c, data, pages, total)
    c.showPage()
    atlas_receipts = []
    for iid, number in pages.items():
        atlas_receipts.append(atlas_page(c, root, data, iid, number, appendix, total))
        c.showPage()
    appendix_receipts = []
    for iid, number in appendix.items():
        appendix_receipts.append(appendix_page(c, root, data, iid, number, total))
        c.showPage()
    c.save()
    pixel_checks = verify_pdf(root, output, data, pages, appendix, total)
    require(all(digest(data['files'][i]['path']) == data['files'][i]['sha256'] for i in data['ids']),
            'A label changed during rendering')
    receipt = {'schema_version': 1, 'kind': 'complete-r24-new-series-atlas',
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'source_revision': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
               'pdf': str(output), 'pdf_sha256': digest(output), 'size_bytes': output.stat().st_size,
               'page_count': total, 'page_size': 'A3 landscape', 'summary': data['summary'],
               'scope': PROVENANCE, 'included_image_ids': data['ids'],
               'photo_pages': pages, 'comparison_pages': appendix,
               'review_manifest': source_descriptor(root, root / REVIEW / 'manifest.json'),
               'parent_review': data['parent'], 'point_ledger': source_descriptor(root, data['audit_path']),
               'builder': source_descriptor(root, Path(__file__).resolve()),
               'all_current_labels_exact_operation_replay': True,
               'all_delta_points_match_saved_ledger': True,
               'all_original_pixels_embedded_losslessly': True,
               'all_photo_index_links_verified': True,
               'old_series_and_empty_images_not_in_atlas': True,
               'rendering_did_not_change_labels': True,
               'visual_review_status': 'pending representative-page inspection',
               'pages': atlas_receipts, 'appendices': appendix_receipts, 'pixel_checks': pixel_checks}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('pdf', 'pdf_sha256', 'size_bytes', 'page_count', 'summary')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    build(args.root.resolve(), output, args.receipt.resolve() if args.receipt else output.with_suffix('.render.json'))
