#!/usr/bin/env python3
"""Render the accepted new_000038 review from pinned original pixels.

No model inference, normalized-label rewriting, source-image alteration, or
hidden-point estimation is performed. All writes are confined to --out-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--font', type=Path,
                        default=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_bytes = args.spec.read_bytes()
    spec = json.loads(spec_bytes)
    if spec.get('status') != 'accepted_incremental' or spec.get('review_status') != 'accepted_reviewed':
        raise ValueError('This renderer requires an explicitly accepted incremental review.')
    if spec['image_id'] != 'new_000038':
        raise ValueError('This renderer is scoped to new_000038.')
    for key, expected_hash in [('source_image_path', 'source_image_sha256'),
                               ('source_label_path', 'source_label_sha256')]:
        data = (args.repo_root / spec[key]).read_bytes()
        if digest(data) != spec[expected_hash]:
            raise ValueError(f'Source hash mismatch: {spec[key]}')
    image_path = args.repo_root / spec['source_image_path']
    original = Image.open(image_path).convert('RGB')
    if list(original.size) != spec['image_size_px']:
        raise ValueError('Original image dimensions do not match the accepted record.')
    if not args.font.is_file():
        raise FileNotFoundError('Supply an installed TrueType font with --font.')
    font = lambda size: ImageFont.truetype(str(args.font), size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    colors = [(255, 192, 35), (0, 224, 255), (175, 255, 85)]
    ink = (25, 31, 40)
    background = (250, 250, 250)
    accepted = {row['final_row']: row for row in spec['row_adjudications']}

    def crop(config: dict) -> Image.Image:
        box = config['crop_xyxy']
        scale = config['scale']
        raw = original.crop(box)
        return raw.resize((raw.width * scale, raw.height * scale), Image.Resampling.NEAREST)

    def xy(point, config):
        box, scale = config['crop_xyxy'], config['scale']
        return (point[0] - box[0]) * scale, (point[1] - box[1]) * scale

    def mark(draw, row, color, config, number=True):
        bounds = row['bbox_xyxy_px']
        bounds_px = (*xy(bounds[:2], config), *xy(bounds[2:], config))
        draw.rectangle(bounds_px, outline='black', width=5)
        draw.rectangle(bounds_px, outline=color, width=2)
        for point in row['keypoints_px']:
            if point[2] != 2:
                continue
            x, y = xy(point[:2], config)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline='black', width=4)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
        if number:
            tx, ty = xy((478, (bounds[1] + bounds[3]) / 2 - 2), config)
            draw.text((tx, ty), str(row['final_row']), font=font(20), fill=color,
                      stroke_width=2, stroke_fill='black')

    config = spec['figures']['three_fronts']
    raw = crop(config)
    marked = raw.copy()
    draw = ImageDraw.Draw(marked)
    for row_id, color in zip([1, 16, 17], colors):
        mark(draw, accepted[row_id], color, config)
    canvas = Image.new('RGB', (raw.width * 2 + 60, raw.height + 276), background)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 13), config['title'], font=font(27), fill=ink)
    draw.text((20, 50), 'Original pixels', font=font(19), fill=ink)
    draw.text((40 + raw.width, 50), 'Accepted after root review', font=font(19), fill=ink)
    canvas.paste(raw, (20, 83))
    canvas.paste(marked, (40 + raw.width, 83))
    y = raw.height + 102
    lines = [
        'Row 1: upper front; box [488,592,529,615]; left (490,612).',
        'Row 16: middle front; box [490,614,518,632]; left (491,629).',
        'Row 17: lowest front; box [492,631,513,650]; left (493,647).',
        'All right endpoints remain missing. Each point belongs to its own cassette.',
        'Ordinary placement uncertainty: about 2 px at corners and 3 px at box boundaries.',
        'Observed original pixels only; no hidden corners or model predictions used.',
    ]
    for line in lines:
        draw.text((20, y), line, font=font(17), fill=ink)
        y += 27
    three_path = args.out_dir / config['filename']
    canvas.save(three_path)

    config = spec['figures']['bottom_front']
    raw = crop(config)
    marked = raw.copy()
    mark(ImageDraw.Draw(marked), accepted[18], colors[1], config, number=False)
    canvas = Image.new('RGB', (raw.width + 40, raw.height * 2 + 190), background)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), config['title'], font=font(23), fill=ink)
    draw.text((20, 49), 'Original crop above; accepted visible-front box below.', font=font(17), fill=ink)
    canvas.paste(raw, (20, 85))
    canvas.paste(marked, (20, raw.height + 105))
    y = raw.height * 2 + 122
    draw.text((20, y), 'Row 18: box [309,700,405,720]; upper boundary about ±3 px.', font=font(16), fill=ink)
    draw.text((20, y + 25), 'Both lower corners are outside the image: both slots remain missing.', font=font(16), fill=ink)
    bottom_path = args.out_dir / config['filename']
    canvas.save(bottom_path)

    index = {
        'schema_version': 1,
        'review_status': 'accepted_reviewed',
        'spec_filename': args.spec.name,
        'spec_sha256': digest(spec_bytes),
        'renderer_filename': Path(__file__).name,
        'renderer_sha256': digest(Path(__file__).read_bytes()),
        'source_image_path': spec['source_image_path'],
        'source_image_sha256': spec['source_image_sha256'],
        'source_label_path': spec['source_label_path'],
        'source_label_sha256': spec['source_label_sha256'],
        'pillow_version': PILLOW_VERSION,
        'font_filename': args.font.name,
        'font_sha256': digest(args.font.read_bytes()),
        'figures': [
            {**spec['figures'][key], 'sha256': digest(path.read_bytes()),
             'output_size_px': list(Image.open(path).size)}
            for key, path in [('three_fronts', three_path), ('bottom_front', bottom_path)]
        ],
        'no_source_mutations': True,
        'model_predictions_used': False,
    }
    index_path = args.out_dir / 'new_000038-accepted-render-index.json'
    index_path.write_text(json.dumps(index, indent=2) + '\n')
    print(three_path)
    print(bottom_path)
    print(index_path)


if __name__ == '__main__':
    main()
