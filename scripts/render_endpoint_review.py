#!/usr/bin/env python3
"""Render fixed original-pixel endpoint comparisons; no inference or new review."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from evaluate_pose import parse_pose_labels


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "data/candidates/endpoint-visibility-2026-11-12"
OUTPUT = ROOT / "results/verification/baomai-endpoint-candidate-20261112/figures"
CASES = (("new_000031", (555, 402, 775, 542), (22, 23, 24)),
         ("new_000023", (460, 505, 650, 640), (33, 34, 35)))
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
COLORS = {"left": "#10b759", "right": "#e832ad"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    return ImageFont.truetype(str(FONT), size)


def main():
    manifest_path = CANDIDATE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    review_path = CANDIDATE / manifest["review"]["path"]
    if digest(review_path) != manifest["review"]["sha256"]:
        raise ValueError("candidate review hash differs")
    review = json.loads(review_path.read_text())
    decisions = {(d["image_id"], d["original_row"]): d for d in review["decisions"]
                 if d.get("operation") == "endpoint_amendment"}
    changes = {item["image_id"]: item for item in manifest["changes"]}
    if digest(ROOT / "data/manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("dataset manifest hash differs")
    records = {item["id"]: item for item in json.loads((ROOT / "data/manifest.json").read_text())["records"]}
    canvas = Image.new("RGB", (1520, 1020), "#f5f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), "Original pixels and endpoint annotation proposal", font=font(29), fill="#17212b")
    draw.text((24, 64), "Two fixed crops at 2× nearest-neighbor scale. Annotation marks only; no model predictions.",
              font=font(19), fill="#465360")
    draw.text((24, 99), "L = lower-left", font=font(18), fill=COLORS["left"])
    draw.text((204, 99), "R = lower-right", font=font(18), fill=COLORS["right"])
    draw.text((405, 99), "Row IDs use original numbering. Missing endpoints have no marker.",
              font=font(18), fill="#465360")
    raw_panels, provenance = [], []
    for case_index, (image_id, roi, rows) in enumerate(CASES):
        record, change = records[image_id], changes[image_id]
        image_path = ROOT / "data" / record["image"]
        source_path = ROOT / "data" / change["source_label"]
        candidate_path = CANDIDATE / change["candidate_label"]
        for path, expected in ((image_path, record["image_sha256"]),
                               (source_path, change["source_label_sha256"]),
                               (candidate_path, change["candidate_label_sha256"])):
            if digest(path) != expected:
                raise ValueError(f"input hash differs: {path}")
        with Image.open(image_path) as source_image:
            if list(source_image.size) != record["original_size"] or source_image.mode != "RGB":
                raise ValueError("expected the original RGB image and dimensions")
            crop = source_image.crop(roi).resize(((roi[2]-roi[0])*2, (roi[3]-roi[1])*2),
                                                 Image.Resampling.NEAREST)
        source_rows = source_path.read_text().splitlines()
        candidate_rows = candidate_path.read_text().splitlines()
        source_objects = parse_pose_labels(source_path.read_text(), *record["original_size"])
        candidate_objects = parse_pose_labels(candidate_path.read_text(), *record["original_size"])
        row_map = {item["original_row"]: item["candidate_row"] for item in change["row_map"]}
        for row in rows:
            decision = decisions[image_id, row]
            if (source_rows[row-1] != decision["source_row_text"] or
                    candidate_rows[row_map[row]-1] != decision["candidate_row_text"]):
                raise ValueError("row differs from the explicit amendment review")
        anchor = 136 + case_index * 430
        draw.text((24, anchor), f"{image_id}.png  ·  original rows {', '.join(map(str, rows))}",
                  font=font(21), fill="#17212b")
        missing_by_panel = {}
        for column, heading in enumerate(("Original crop · no marks", "Original endpoint labels", "Endpoint candidate labels")):
            left, top = 24 + column * 496, anchor + 32
            draw.rounded_rectangle((left, top, left+480, top+368), radius=10, fill="white", outline="#d6dce3")
            draw.text((left+16, top+10), heading, font=font(19), fill="#17212b")
            x, y = left + (480-crop.width)//2, top+42
            canvas.paste(crop, (x, y))
            if column == 0:
                raw_panels.append(((x, y, x+crop.width, y+crop.height), crop))
                caption = f"Source crop: x{roi[0]}–{roi[2]}, y{roi[1]}–{roi[3]}"
            else:
                objects = source_objects if column == 1 else candidate_objects
                missing = []
                for row in rows:
                    obj = objects[(row if column == 1 else row_map[row])-1]
                    for side, point in zip(("left", "right"), obj["keypoints"]):
                        if point[2] == 0:
                            missing.append(f"{side[0].upper()}{row}")
                            continue
                        px, py = x + (point[0]-roi[0])*2, y + (point[1]-roi[1])*2
                        color = COLORS[side]
                        draw.ellipse((px-4, py-4, px+4, py+4), outline="#ffffff", width=3)
                        draw.ellipse((px-3, py-3, px+3, py+3), outline=color, width=2)
                        label = f"{row} {side[0].upper()}"
                        bounds = draw.textbbox((px+8, py-16), label, font=font(16))
                        draw.rectangle((bounds[0]-2, bounds[1]-1, bounds[2]+2, bounds[3]+1), fill="#ffffff")
                        draw.text((px+8, py-16), label, font=font(16), fill=color)
                caption = "Missing: " + ", ".join(missing)
                missing_by_panel[heading] = missing
            draw.text((left+16, top+330), caption, font=font(17), fill="#465360")
        provenance.append({"image_id": image_id, "original_rows": list(rows), "crop_xyxy": list(roi),
                           "source_image": {"path": image_path.relative_to(ROOT).as_posix(), "sha256": digest(image_path)},
                           "source_label": {"path": source_path.relative_to(ROOT).as_posix(), "sha256": digest(source_path)},
                           "candidate_label": {"path": candidate_path.relative_to(ROOT).as_posix(), "sha256": digest(candidate_path)},
                           "missing_slots": missing_by_panel})
    draw.text((24, 986), "Fixed label comparison from the recorded assistant review. No new visual adjudication or human sign-off.",
              font=font(17), fill="#465360")
    for box, expected in raw_panels:
        if canvas.crop(box).tobytes() != expected.tobytes():
            raise ValueError("unannotated crop pixels were altered")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image_output = OUTPUT / "endpoint-annotation-comparison.png"
    canvas.save(image_output, optimize=True)
    metadata = {"kind": "original_pixel_annotation_comparison", "source_revision": manifest["source_revision"],
                "candidate_manifest_sha256": digest(manifest_path), "candidate_review_sha256": digest(review_path),
                "scale": 2, "resampling": "nearest-neighbor", "dimensions": list(canvas.size),
                "original_crop_panels_pixel_exact": True, "marker_colors": COLORS, "cases": provenance,
                "annotations_not_predictions": True, "missing_markers_drawn": False,
                "descriptive_ranges_drawn": False, "new_visual_adjudication": False,
                "human_signoff": False, "inference_or_training": False,
                "renderer_sha256": digest(Path(__file__)), "font_sha256": digest(FONT),
                "output": {"path": image_output.relative_to(ROOT).as_posix(), "sha256": digest(image_output)}}
    (OUTPUT / "figure-provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"image": str(image_output), "sha256": digest(image_output), "pixel_exact_raw_panels": 2}))


if __name__ == "__main__":
    main()
