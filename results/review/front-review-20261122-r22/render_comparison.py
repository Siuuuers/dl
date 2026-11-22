#!/usr/bin/env python3
"""Render a compact comparison from original pixels and saved r21/r22 labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_pose import parse_pose_labels

CYAN, MAGENTA, INK = "#41e1ed", "#ff36d1", "#132b38"
CASES = [
    ("new_000055", [17], [195, 518, 340, 583],
     "A dark-edge cue can mislead: restore this cassette's own projecting upper lip."),
    ("new_000100", [48], [684, 396, 790, 449],
     "Glare needs contour review: trim the box and move the right point in X and Y."),
    ("new_000094", [23], [330, 593, 487, 666],
     "Restore the clipped upper face: ymin = 605; keep ymax near 655.99."),
    ("new_000096", [37, 38], [482, 459, 625, 536],
     "Remove the two right landmarks; retain both cassette boxes and row identities."),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    records = {r["id"]: r for r in json.loads((ROOT / "data/manifest.json").read_text())["records"]}
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    small = ImageFont.truetype("DejaVuSans.ttf", 14)
    bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 21)
    title = ImageFont.truetype("DejaVuSans-Bold.ttf", 29)
    canvas = Image.new("RGB", (1512, 1450), "#f5f7f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 17), "R22 | visible cassette material and actual lower corners", font=title, fill=INK)
    draw.text((24, 62), "Original image pixels. Cyan: retained fields. Magenta: amended fields. Missing points have no marker.", font=font, fill=INK)
    evidence = []
    for ci, (iid, rows, region, caption) in enumerate(CASES):
        y = 107 + ci * 326
        draw.text((24, y), f"{iid}  |  row" + ("s " if len(rows) > 1 else " ") + ", ".join(map(str, rows)), font=bold, fill=INK)
        draw.text((24, y + 29), caption, font=font, fill=INK)
        rec = records[iid]
        image_path = ROOT / "data" / rec["image"]
        assert sha(image_path) == rec["image_sha256"]
        original = Image.open(image_path).convert("RGB")
        crop = original.crop(region)
        zoom = min(480 // crop.width, 235 // crop.height)
        assert zoom >= 1
        raw = crop.resize((crop.width * zoom, crop.height * zoom), Image.Resampling.NEAREST)
        paths = [ROOT / f"data/reviews/fronts-2026-11-22-r{n}" / rec["label"] for n in (21, 22)]
        lines = [p.read_text().splitlines() for p in paths]
        objects = [parse_pose_labels(p.read_text(), *rec["original_size"]) for p in paths]
        for column, label in enumerate(("ORIGINAL", "R21", "R22")):
            left, top = 24 + column * 496, y + 79
            draw.text((left, y + 58), label, font=small, fill=INK)
            panel = raw.copy()
            pen = ImageDraw.Draw(panel)
            if column:
                version = column - 1
                for row in rows:
                    obj = objects[version][row - 1]
                    a, b = (list(map(float, text[row - 1].split())) for text in lines)
                    changes = {g for g, sl in (("box", slice(1, 5)), ("left", slice(5, 8)), ("right", slice(8, 11))) if a[sl] != b[sl]}
                    def xy(x, py):
                        return ((x - region[0]) * zoom, (py - region[1]) * zoom)
                    box_color = MAGENTA if version and "box" in changes else CYAN
                    x0, y0, x1, y1 = obj["bbox_xyxy"]
                    pen.rectangle((*xy(x0, y0), *xy(x1, y1)), outline=box_color, width=2)
                    for side, (px, py, visible) in zip(("left", "right"), obj["keypoints"]):
                        if not visible:
                            continue
                        color = MAGENTA if version and side in changes else CYAN
                        xx, yy = xy(px, py)
                        marker = (xx - 3, yy - 3, xx + 3, yy + 3)
                        (pen.ellipse if side == "left" else pen.rectangle)(marker, fill=color, outline=INK, width=1)
            canvas.paste(panel, (left, top))
        evidence.append({"image_id": iid, "rows": rows, "region_xyxy": region,
                         "integer_nearest_zoom": zoom, "source_image_sha256": sha(image_path),
                         "source_crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                         "label_sha256": {f"r{n}": sha(p) for n, p in zip((21, 22), paths)}})
    draw.text((24, 1422), "Each panel uses the same crop. Box extrema and lower-corner landmarks are reviewed independently.", font=small, fill=INK)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    receipt = {"schema_version": 1, "file_name": args.output.name, "sha256": sha(args.output),
               "original_pixels_only": True, "annotations_from_saved_labels": True, "cases": evidence}
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": receipt["sha256"]}))


if __name__ == "__main__":
    main()
