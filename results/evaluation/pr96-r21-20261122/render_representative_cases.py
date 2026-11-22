#!/usr/bin/env python3
"""Render unchanged PR96 candidate outputs against two pinned references."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CASES = (("new_000057", 12), ("new_000090", 41),
         ("new_000010", 32), ("new_000011", 58))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    sensitivity = json.loads((evidence / "reference-sensitivity.json").read_text())
    model = sensitivity["models"]["candidate"]
    cases = {(c["image_id"], c["primary_row"]): c for c in
             model["fixed_primary_miss_cases"] + model["primary_match_but_r21_miss_cases"]}
    records = {r["id"]: r for r in json.loads((args.root / "data/manifest.json").read_text())["records"]}
    output = evidence / "representative-cases"
    output.mkdir(exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    inventory = []
    for identity in CASES:
        case = cases[identity]
        path = args.root / "data" / records[identity[0]]["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["image_sha256"]
        image = Image.open(path).convert("RGB")
        objects = [case[k] for k in ("primary_gold", "r21_gold", "prediction")]
        bounds = [o["bbox_xyxy"] for o in objects]
        left = max(0, math.floor(min(b[0] for b in bounds)) - 16)
        top = max(0, math.floor(min(b[1] for b in bounds)) - 20)
        right = min(image.width, math.ceil(max(b[2] for b in bounds)) + 16)
        bottom = min(image.height, math.ceil(max(b[3] for b in bounds)) + 16)
        scale, gap, header, footer = 3, 12, 64, 24
        crop = image.crop((left, top, right, bottom))
        width, height = crop.width * scale, crop.height * scale
        figure = Image.new("RGB", (3 * width + 4 * gap, height + header + footer), "#111827")
        draw = ImageDraw.Draw(figure)
        title = f"{identity[0]}  primary row {identity[1]} -> r21 row {case['r21_row']}"
        draw.text((gap, 7), title, font=font, fill="white")
        draw.text((gap, 30), "Yellow: reference box / visible landmarks    Magenta: same candidate output", font=small, fill="#e5e7eb")
        for column, label in enumerate(("Original pixels", "Frozen primary", "Pre-return r21")):
            x = gap + column * (width + gap)
            figure.paste(crop.resize((width, height), Image.Resampling.NEAREST), (x, header))
            if column:
                gold = case["primary_gold" if column == 1 else "r21_gold"]
                for obj, colour, is_prediction in ((gold, "#fde047", False), (case["prediction"], "#ff43cc", True)):
                    box = [(x + (px - left) * scale, header + (py - top) * scale)
                           for px, py in ((obj["bbox_xyxy"][0], obj["bbox_xyxy"][1]), (obj["bbox_xyxy"][2], obj["bbox_xyxy"][3]))]
                    draw.rectangle(box, outline=colour, width=2)
                    for px, py, visible in obj["keypoints"]:
                        if visible < .5 if is_prediction else visible != 2:
                            continue
                        cx, cy = x + (px - left) * scale, header + (py - top) * scale
                        draw.ellipse((cx-4, cy-4, cx+4, cy+4), outline=colour, width=2)
                iou = case["primary_iou" if column == 1 else "r21_iou"]
                label += f"  IoU {iou:.6f}"
            draw.text((x, header + height + 4), label, font=small, fill="white")
        target = output / f"{identity[0]}-primary-row{identity[1]}.png"
        figure.save(target)
        inventory.append({"image_id": identity[0], "primary_row": identity[1], "r21_row": case["r21_row"],
                          "path": str(target.relative_to(evidence)), "source_image_sha256": case["image_sha256"],
                          "crop_xyxy": [left, top, right, bottom], "scale": scale,
                          "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                          "primary_iou": case["primary_iou"], "r21_iou": case["r21_iou"]})
    (evidence / "representative-image-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
