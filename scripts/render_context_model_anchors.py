#!/usr/bin/env python3
"""Show original pixels and direct PR35 right-point predictions, with no geometry layer."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.model_inputs.read_bytes()
    bundle = json.loads(source)
    image_id = "new_000010"
    identity = next(item for item in bundle["images"] if item["image_id"] == image_id)
    path = args.repo_root/f"data/images/val/{image_id}.png"
    raw = path.read_bytes()
    digest = lambda value: hashlib.sha256(value).hexdigest()
    if digest(raw) != identity["image_sha256"]:
        raise ValueError("Original image identity mismatch")
    manifest_path = args.repo_root/"data/references/visible-front-2026-11-11/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    entry = next(item for item in manifest["files"] if item["image_id"] == image_id)
    label_bytes = (manifest_path.parent/entry["label_path"]).read_bytes()
    if digest(label_bytes) != entry["label_sha256"]:
        raise ValueError("Reference label identity mismatch")
    rows = label_bytes.decode().splitlines()
    points = []
    for row in (23, 24):
        record = next(item for item in bundle["rows"] if item["image_id"] == image_id and item["row"] == row)
        if not record["matched"]:
            raise ValueError("Required direct model match absent")
        fields = list(map(float, rows[row-1].split()))
        if fields[10] != 2:
            raise ValueError("Required right reference is not observed")
        reference = [fields[8]*identity["width"], fields[9]*identity["height"]]
        prediction = record["prediction"]["keypoints"][1]
        error = sum((a-b)**2 for a, b in zip(reference, prediction[:2]))**.5
        points.append({"row": row, "reference_xy": reference, "prediction_xy": prediction[:2],
                       "prediction_confidence": prediction[2], "error_px": error})

    crop, scale = [850, 390, 955, 515], 4
    panel_width, panel_height = (crop[2]-crop[0])*scale, (crop[3]-crop[1])*scale
    background, ink, muted = "#111821", "#f3f5f8", "#bac4d0"
    reference_color, prediction_color = "#20dba2", "#63b9ff"
    canvas = Image.new("RGB", (panel_width*2+72, panel_height+280), background)
    draw = ImageDraw.Draw(canvas)
    font = lambda size: ImageFont.truetype("DejaVuSans.ttf", size)
    draw.text((24, 18), "Visible right corners: PR35 predicts them", font=font(25), fill=ink)
    draw.text((24, 57), "new_000010 / rows 23 and 24 / saved current-model predictions", font=font(17), fill=muted)
    patch = Image.open(path).convert("RGB").crop(crop).resize((panel_width, panel_height), Image.Resampling.NEAREST)
    for x, title in ((24, "ORIGINAL PIXELS"), (48+panel_width, "REFERENCE + PR35")):
        draw.text((x, 96), title, font=font(16), fill=muted)
        canvas.paste(patch, (x, 124))
    origin = (48+panel_width, 124)
    for record in points:
        for key, color in (("reference_xy", reference_color), ("prediction_xy", prediction_color)):
            x = origin[0]+(record[key][0]-crop[0])*scale
            y = origin[1]+(record[key][1]-crop[1])*scale
            bounds = (x-8, y-8, x+8, y+8)
            if key == "reference_xy":
                draw.rectangle(bounds, outline=color, width=3)
            else:
                draw.ellipse(bounds, outline=color, width=3)
        x = origin[0]+18
        y = origin[1]+(record["reference_xy"][1]-crop[1])*scale-28
        text = f"row {record['row']}"
        bounds = draw.textbbox((x, y), text, font=font(17))
        draw.rectangle((bounds[0]-4, bounds[1]-2, bounds[2]+4, bounds[3]+2), fill=background)
        draw.text((x, y), text, font=font(17), fill=ink)
    y = panel_height+149
    draw.rectangle((27, y, 43, y+16), outline=reference_color, width=3)
    draw.text((53, y-4), "Observed reference", font=font(18), fill=ink)
    draw.ellipse((347, y, 363, y+16), outline=prediction_color, width=3)
    draw.text((373, y-4), "Direct PR35 prediction", font=font(18), fill=ink)
    footer = [
        f"Distance to reference: row 23 = {points[0]['error_px']:.2f} px; row 24 = {points[1]['error_px']:.2f} px.",
        "No geometric estimates are shown in this figure.",
        "The reference labels are unchanged; model confidence is not a visibility guarantee.",
        "Original crop [850, 390, 955, 515], enlarged 4x by nearest-neighbor sampling.",
    ]
    for line in footer:
        y += 26
        draw.text((24, y), line, font=font(16), fill=muted)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir/"context-model-direct-right-20261112.png"
    index_path = args.output_dir/"context-model-direct-right-index-20261112.json"
    if output.exists() or index_path.exists():
        raise FileExistsError("Choose a fresh output directory")
    canvas.save(output)
    index = {"image_id": image_id, "source_image_sha256": digest(raw),
             "reference_label_sha256": digest(label_bytes), "model_inputs_sha256": digest(source),
             "renderer_sha256": digest(Path(__file__).read_bytes()), "pillow_version": Image.__version__,
             "source_crop_xyxy": crop, "nearest_neighbor_scale": scale,
             "figure": output.name, "figure_size": list(canvas.size), "figure_sha256": digest(output.read_bytes()),
             "geometry_estimates_shown": False, "right_points": points}
    index_path.write_text(json.dumps(index, indent=2)+"\n")
    print(json.dumps(index))


if __name__ == "__main__":
    main()
