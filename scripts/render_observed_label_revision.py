#!/usr/bin/env python3
"""Render original-pixel evidence for the fixed-geometry label review; no model."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.absolute()
    if output.exists():
        raise ValueError("choose a fresh figure directory")
    review_path = root / "results/review/fixed-geometry-20261114/accepted-review.json"
    review = json.loads(review_path.read_text())
    candidate = root / "data/candidates/fixed-geometry-labels-2026-11-14"
    manifest = json.loads((candidate / "manifest.json").read_text())
    changes = {c["image_id"]: c for c in manifest["changes"]}
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 17)
    jobs = [(image, image + "-all", (0, 0, 1280, 720), 1, None)
            for image in sorted({op["image_id"] for op in review["operations"]})]
    jobs.extend([
        ("new_000006", "new006-right-jaw", (685, 460, 825, 575), 4, [51, 52, 60]),
        ("new_000006", "new006-thumb-partial", (440, 525, 580, 615), 4, [53]),
        ("new_000012", "new012-corner30", (305, 603, 375, 658), 5, [30]),
    ])
    output.mkdir(parents=True)
    index = {"review_sha256": sha(review_path), "candidate_manifest_sha256": sha(candidate / "manifest.json"),
             "font_sha256": sha(font_path), "images": [], "pixels": "Original PNGs, nearest-neighbor enlargement; no generated content."}
    for image_id, name, crop, scale, selected in jobs:
        source = root / f"data/images/train/{image_id}.png"
        original = Image.open(source).convert("RGB")
        ops = [op for op in review["operations"] if op["image_id"] == image_id
               and (selected is None or op["original_row"] in selected)]
        for op in ops:
            if sha(source) != op["source_image_sha256"]:
                raise ValueError("source image changed")
        w, h = (crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale
        raw = original.crop(crop).resize((w, h), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (w * 3 + 32, h + 96), "#f6f7f9")
        draw = ImageDraw.Draw(canvas)
        for panel, title in enumerate(("Original pixels", "Previous review candidate", "Accepted review revision")):
            xoff = panel * (w + 16)
            canvas.paste(raw, (xoff, 34))
            draw.text((xoff + 4, 8), title, font=font, fill="#15202b")
            if panel == 0:
                continue
            for op in ops:
                text = op["before_text"] if panel == 1 else op["after_text"]
                if text is None:
                    continue
                t = list(map(float, text.split()))
                cx, cy, bw, bh = t[1:5]
                b = [(cx - bw/2)*1280, (cy - bh/2)*720, (cx + bw/2)*1280, (cy + bh/2)*720]
                def xy(x, y):
                    return (xoff + (x - crop[0])*scale, 34 + (y - crop[1])*scale)
                color = "#ef982c" if panel == 1 else "#00b785"
                if op["field"] in ("bbox", "delete_object"):
                    draw.rectangle([xy(b[0], b[1]), xy(b[2], b[3])], outline=color, width=2)
                    draw.text(xy(b[0]+1, b[1]+1), str(op["original_row"]), font=font,
                              fill=color, stroke_width=1, stroke_fill="#15202b")
                for start in (5, 8):
                    if t[start+2] != 2:
                        continue
                    sx, sy = t[start]*1280, t[start+1]*720
                    if not (crop[0] <= sx < crop[2] and crop[1] <= sy < crop[3]):
                        continue
                    px, py = xy(sx, sy)
                    draw.line((px-5, py, px+5, py), fill=color, width=2)
                    draw.line((px, py-5, px, py+5), fill=color, width=2)
        caption = f"{image_id} | Original row IDs | Crop {crop} | Only accepted operations shown"
        draw.text((6, h+46), caption, font=font, fill="#15202b")
        note = ("Row53: conservative right cap only; exact A/B boxes remain under review."
                if name.endswith("thumb-partial") else
                "Existing corner refinement; X token and v2 retained; Y 640.88 to 634 px."
                if name.endswith("corner30") else
                "Hidden corners remain missing. Unshown or retained bounds are not newly certified.")
        draw.text((6, h+70), note, font=font, fill="#15202b")
        target = output / f"{name}.png"
        canvas.save(target)
        index["images"].append({"path": target.name, "sha256": sha(target), "image_id": image_id,
                                "source_image_sha256": sha(source), "crop_xyxy_px": list(crop), "scale": scale,
                                "operation_original_rows": [o["original_row"] for o in ops]})
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"figures": len(index["images"]), "output": str(output)}))


if __name__ == "__main__":
    main()
