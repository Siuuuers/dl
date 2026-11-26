"""Build the six-card review directly from original images, not sheet screenshots."""
import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--image-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root, image_root, output = args.root.resolve(), args.image_root.resolve(), args.output.resolve()
sys.path.insert(0, str(root / 'scripts'))
from evaluate_pose import parse_pose_labels

manifest_path = root / 'data/reviews/fronts-2026-11-26-r38/manifest.json'
manifest = json.loads(manifest_path.read_text())
files = {f['image_id']: f for f in manifest['files']}
review = json.loads((output / 'review.json').read_text())
cases = {c['case']: c for c in review['cases']}

font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font_bold_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
fonts = {size: ImageFont.truetype(font_path, size) for size in (12, 14, 15, 16)}
heading = ImageFont.truetype(font_bold_path, 17)
title = ImageFont.truetype(font_bold_path, 22)
cyan = (0, 185, 205)
magenta = (233, 0, 103)
orange = (235, 135, 0)
ink = (27, 42, 56)

items = [
    (49, 'ALREADY MATCHED TARGET', 'IoU 0.430 to R38 row42; existing match p53.',
     'The same visible front has another box extending rightward into the jaw.'),
    (29, 'MISSED TARGET: BOX MISMATCH', 'IoU 0.488 to R38 row55; fixed matching gate 0.50.',
     'The front has a high-confidence proposal. Its box fails the accepted extent convention.'),
    (24, 'EXCLUDED TARGET REGION', 'IoU 0.780 to explicitly excluded R37 row30.',
     'The accepted exclusion remains unchanged. Orange shows the removed reference box.'),
    (50, 'BACKGROUND: TOP FIXTURE', 'No overlap with accepted cassette-front boxes.',
     'White rack rail above the mesh; the adjacent 090 frame exposes its mounting bolts.'),
    (59, 'BACKGROUND: OUTSIDE RACK', 'No overlap with accepted cassette-front boxes.',
     'One of four remaining outputs on this same blue/gray background object.'),
    (42, 'UNCERTAIN: CLIPPED STRIP', 'No overlap; predicted height is 9.02 source pixels.',
     'A clipped front or fixture/rim is unresolved. No annotation is added.'),
]

def wrap(draw, text, font, width):
    lines, current = [], ''
    for word in text.split():
        candidate = f'{current} {word}'.strip()
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

card_w, card_h = 720, 390
gap, outer, top_h = 18, 18, 92
canvas_w = outer * 2 + card_w * 2 + gap
canvas_h = top_h + card_h * 3 + gap * 2 + outer
canvas = Image.new('RGB', (canvas_w, canvas_h), (232, 237, 241))
draw = ImageDraw.Draw(canvas)
draw.text((outer, 12), 'Six examples from the 64 other-unmatched outputs', fill=ink, font=title)
draw.text((outer, 43), 'Original pixels at left; annotated crop at right. Cyan: accepted R38. Magenta: prediction. Orange: excluded reference.', fill=ink, font=fonts[15])
draw.text((outer, 65), 'Primary pIDs preserved. Source crop xyxy coordinates shown. Nearest-neighbor enlargement; accepted labels unchanged.', fill=ink, font=fonts[14])

card_manifest = []
for slot, (case_id, category, metric, finding) in enumerate(items):
    case = cases[case_id]
    x = outer + (slot % 2) * (card_w + gap)
    y = top_h + (slot // 2) * (card_h + gap)
    card = Image.new('RGB', (card_w, card_h), 'white')
    cd = ImageDraw.Draw(card)
    cd.rectangle((0, 0, card_w - 1, card_h - 1), outline=(190, 200, 208), width=1)
    cd.rectangle((1, 1, card_w - 2, 31), fill=(240, 244, 247))
    assert cd.textlength(category, font=heading) < card_w - 24
    cd.text((12, 6), category, fill=ink, font=heading)
    identity = f"{case['image_id']}  |  primary p{case['primary_prediction_id']}  |  confidence {case['prediction']['confidence']:.3f}"
    cd.text((12, 38), identity, fill=ink, font=fonts[16])
    cd.text((12, 61), metric, fill=ink, font=fonts[15])
    crop = case['crop']
    coord = f'Source crop xyxy = ({crop[0]}, {crop[1]}, {crop[2]}, {crop[3]})'
    cd.text((12, 82), coord, fill=(70, 80, 90), font=fonts[14])
    cd.text((12, 105), 'ORIGINAL', fill=(70, 80, 90), font=fonts[12])
    cd.text((card_w // 2 + 6, 105), 'ANNOTATED', fill=(70, 80, 90), font=fonts[12])

    source = Image.open(image_root / f"{case['image_id']}.png").convert('RGB')
    raw = source.crop(crop)
    panel_w, panel_h = 342, 216
    scale = min(panel_w / raw.width, panel_h / raw.height)
    size = (int(math.floor(raw.width * scale)), int(math.floor(raw.height * scale)))
    raw_scaled = raw.resize(size, Image.Resampling.NEAREST)
    annotation = raw_scaled.copy()
    ad = ImageDraw.Draw(annotation)
    sx, sy = size[0] / raw.width, size[1] / raw.height

    def translated(box):
        return [(box[0] - crop[0]) * sx, (box[1] - crop[1]) * sy,
                (box[2] - crop[0]) * sx, (box[3] - crop[1]) * sy]

    def intersects(box):
        return not (box[2] < crop[0] or box[0] > crop[2] or box[3] < crop[1] or box[1] > crop[3])

    ref = parse_pose_labels((manifest_path.parent / files[case['image_id']]['label_path']).read_text(), source.width, source.height)
    for row, gold in enumerate(ref, 1):
        box = gold['bbox_xyxy']
        if intersects(box):
            b = translated(box)
            ad.rectangle(b, outline=cyan, width=1)
            if row == case['best_reference_row']:
                label = f'R38 r{row}'
                ad.text((max(1, b[0] + 2), max(1, b[1] + 2)), label, fill=cyan, font=fonts[12], stroke_width=1, stroke_fill=(0,0,0))
    if case_id == 24:
        excluded = next(e for e in case['exclusion_history_overlaps'] if e['base_reference'] == 'R37' and e['base_row'] == 30)
        ad.rectangle(translated(excluded['bbox_xyxy']), outline=orange, width=3)
    ad.rectangle(translated(case['prediction']['bbox_xyxy']), outline=magenta, width=2)
    for px, py, confidence in case['prediction']['keypoints']:
        if px is not None and py is not None and crop[0] <= px <= crop[2] and crop[1] <= py <= crop[3]:
            px, py = (px - crop[0]) * sx, (py - crop[1]) * sy
            ad.ellipse((px - 3, py - 3, px + 3, py + 3), fill=magenta)

    top = 124 + (panel_h - size[1]) // 2
    for panel_x, im in [(12, raw_scaled), (card_w // 2 + 6, annotation)]:
        card.paste(im, (panel_x + (panel_w - size[0]) // 2, top))
    lines = wrap(cd, finding, fonts[14], card_w - 24)
    assert len(lines) <= 2
    for line_no, line in enumerate(lines):
        cd.text((12, 347 + line_no * 18), line, fill=ink, font=fonts[14])
    canvas.paste(card, (x, y))
    card_manifest.append({'slot': slot + 1, 'case': case_id, 'category': category, 'image_id': case['image_id'], 'primary_prediction_id': case['primary_prediction_id'], 'source_crop_xyxy': crop, 'card_xyxy': [x,y,x+card_w,y+card_h], 'rendered_crop_size': list(size), 'resampling': 'nearest'})

output.mkdir(parents=True, exist_ok=True)
canvas.save(output / 'representative_cases.png')
(output / 'representative_cases.layout.json').write_text(json.dumps({'size': list(canvas.size), 'cards': card_manifest}, indent=2) + '\n')
print(f'Rendered all {len(card_manifest)} cards at {canvas.width} x {canvas.height}.')
