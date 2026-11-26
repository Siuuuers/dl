#!/usr/bin/env python3
"""Draw an existing 1280x720 original and its two-corner YOLO TXT labels.

This is a label preview, not inference or physical acceptance. Row numbers are
one-based TXT lines, which can differ from immutable source rows after deletion.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from review_front_mining_r42 import pose


def preview(image_path, labels_path, output, row=None, scale=1):
    original = Image.open(image_path).convert("RGB")
    if original.size != (1280, 720):
        raise ValueError("Use the original 1280x720 photograph, not a preview with a footer")
    lines = Path(labels_path).read_text().splitlines()
    rows = [pose(line, f"TXT row {i + 1}") for i, line in enumerate(lines)]
    if row is not None and not 1 <= row <= len(rows):
        raise ValueError("Requested TXT row does not exist")
    if row is None:
        crop = (0, 0, 1280, 720)
    else:
        a = rows[row - 1]
        crop = (max(0, int((a[1]-a[3]/2)*1280)-12),
                max(0, int((a[2]-a[4]/2)*720)-12),
                min(1280, int((a[1]+a[3]/2)*1280)+13),
                min(720, int((a[2]+a[4]/2)*720)+13))
    tile = original.crop(crop)
    tile = tile.resize((tile.width*scale, tile.height*scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (tile.width, tile.height+34), "#f4f7f8")
    canvas.paste(tile, (0, 34))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    d.text((8, 9), f"{Path(image_path).stem} | TXT " + (f"row {row}" if row else f"{len(rows)} rows"), fill="#14303c", font=font)
    def xy(x, y):
        return ((x-crop[0])*scale, (y-crop[1])*scale+34)
    for i, a in enumerate(rows, 1):
        if row is not None and i != row:
            continue
        x1, y1 = xy((a[1]-a[3]/2)*1280, (a[2]-a[4]/2)*720)
        x2, y2 = xy((a[1]+a[3]/2)*1280, (a[2]+a[4]/2)*720)
        d.rectangle((x1, y1, x2, y2), outline="#00d9e8", width=max(1, scale//2))
        d.text((x1, max(34, y1-15)), str(i), fill="white", stroke_width=1, stroke_fill="#12242a", font=font)
        for k, color in ((5, "#25ed67"), (8, "#168cff")):
            if a[k+2] == 0:
                continue
            x, y = xy(a[k]*1280, a[k+1]*720)
            r = max(2, scale)
            d.ellipse((x-r, y-r, x+r, y+r), outline=color, width=max(1, scale//2))
    output = Path(output)
    if output.resolve() in (Path(image_path).resolve(), Path(labels_path).resolve()):
        raise ValueError("The preview must not overwrite an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {"rows": len(rows), "shown_txt_row": row, "output": str(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--row", type=int)
    p.add_argument("--scale", type=int, choices=range(1, 9), default=1)
    a = p.parse_args()
    print(preview(a.image, a.labels, a.output, a.row, a.scale))


if __name__ == "__main__":
    main()
