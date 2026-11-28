#!/usr/bin/env python3
"""Reproduce the saved review panels from verified originals and numeric context.

This renderer writes only PNG review panels in --output. It does not edit labels,
models, decisions, or the frozen evaluation archives. Pillow is required.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def panel(image, roi, objects, title, predicted=False, tags=True):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
    crop = image.crop(roi)
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST)
    canvas = Image.new('RGB', (crop.width, crop.height + 32), 'white')
    canvas.paste(crop, (0, 32))
    draw = ImageDraw.Draw(canvas)
    for record in objects:
        obj = record['object']
        coords = [(obj['bbox_xyxy'][0] - roi[0]) * 3, (obj['bbox_xyxy'][1] - roi[1]) * 3 + 32,
                  (obj['bbox_xyxy'][2] - roi[0]) * 3, (obj['bbox_xyxy'][3] - roi[1]) * 3 + 32]
        color = '#00e5ff' if predicted else '#00ff60'
        if record.get('status') == 'duplicate':
            color = '#ffbf00'
        elif record.get('status') == 'extra':
            color = '#ff88ff'
        draw.rectangle(coords, outline=color, width=2)
        if tags:
            name = ('P' if predicted else 'R') + str(record['index'] + 1).zfill(3)
            if predicted:
                name += ' ' + {'matched': 'M', 'duplicate': 'D', 'extra': 'E'}[record['status']]
            tx = max(0, min(coords[0] + 2, crop.width - 66))
            ty = max(32, min(coords[1] + 2, crop.height + 18))
            draw.rectangle((tx, ty, tx + 65, ty + 12), fill='#222222')
            draw.text((tx + 1, ty), name, fill=color, font=small)
        for px, py, flag in obj['keypoints']:
            available = flag > 0.5 if predicted else flag == 2
            if available:
                xx, yy = (px - roi[0]) * 3, (py - roi[1]) * 3 + 32
                draw.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), fill=color, outline='black')
    draw.rectangle((0, 0, crop.width, 31), fill='white')
    draw.text((5, 8), title, fill='black', font=font)
    return canvas


def joined(panels):
    canvas = Image.new('RGB', (sum(p.width for p in panels) + 20 * (len(panels) - 1), panels[0].height), '#eeeeee')
    xpos = 0
    for image in panels:
        canvas.paste(image, (xpos, 0))
        xpos += image.width + 20
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', type=Path, required=True, help='Directory containing the seven original new_*.png files')
    parser.add_argument('--evidence-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads((args.evidence_dir / 'input-provenance.json').read_text())
    context = json.loads((args.evidence_dir / 'comparison-context.json').read_text())
    images = {}
    for image_id, metadata in provenance['images'].items():
        path = args.image_dir / (image_id + '.png')
        data = path.read_bytes()
        if len(data) != metadata['bytes'] or hashlib.sha256(data).hexdigest() != metadata['sha256']:
            raise ValueError('Original image differs: ' + image_id)
        images[image_id] = Image.open(path).convert('RGB')
        if images[image_id].size != (1280, 720):
            raise ValueError('Original image dimensions differ: ' + image_id)
    args.output.mkdir(parents=True, exist_ok=True)
    for case in context['cases']:
        name, roi, image = case['case_id'], case['region_xyxy'], images[case['image']]
        crop = image.crop(roi)
        crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST).save(args.output / (name + '-raw.png'))
        joined([panel(image, roi, case['references'][ref], ref + ' references')
                for ref in ('supplied664', 'frozen666', 'r50')]).save(args.output / (name + '-references.png'))
        joined([panel(image, roi, case['predictions'][arm], arm + ' two-view predictions', True)
                for arm in ('original_labels', 'reviewed_labels')]).save(args.output / (name + '-predictions.png'))
        if name == 'case-08':
            # Separate the thin border hypothesis so dense labels cannot hide it.
            target = next(r for r in case['references']['r50'] if r['row_1based'] == 12)
            matched = next(r for r in case['predictions']['reviewed_labels'] if r['position_1based'] == 53)
            duplicate = next(r for r in case['predictions']['reviewed_labels'] if r['position_1based'] == 55)
            joined([panel(image, roi, [target], 'r50 R012: one available corner', tags=False),
                    panel(image, roi, [matched], 'Reviewed-label P053: matched', True, False),
                    panel(image, roi, [duplicate], 'Reviewed-label P055: duplicate', True, False)]).save(args.output / (name + '-target-comparison.png'))
    print('Verified seven originals and wrote 25 review PNGs; no labels or models changed.')


if __name__ == '__main__':
    main()
