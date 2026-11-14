#!/usr/bin/env python3
"""Render four post hoc examples from the frozen PR62 prediction records."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pose_scoring import _box_iou


CASES = (
    {"image_id": "new_000008", "row": 17, "probe_case": "new_000008-row017-left",
     "crop": [650, 415, 930, 555], "title": "A  Synthetic occlusion: a recovered covered corner  |  new_000008:17"},
    {"image_id": "new_000008", "row": 11, "probe_case": None,
     "crop": [490, 410, 770, 550], "title": "B  Clean image: both corner estimates move upward  |  new_000008:11"},
    {"image_id": "new_000056", "row": 3, "probe_case": None,
     "crop": [500, 270, 780, 410], "title": "C  Box matching: an emitted pair loses its scored match  |  new_000056:3"},
    {"image_id": "new_000147", "row": None, "probe_case": None,
     "crop": [0, 0, 300, 150], "title": "D  Empty scene: a new false proposal on the background  |  new_000147"},
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def font(size, bold=False):
    from matplotlib.font_manager import FontProperties, findfont
    return ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")), size)


def dashed(draw, a, b, fill, width=2):
    length = math.dist(a, b)
    if not length:
        return
    for offset in range(0, math.ceil(length), 14):
        start, end = offset / length, min(offset + 8, length) / length
        draw.line([(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) for t in (start, end)], fill=fill, width=width)


def panel(original, crop, truth, prediction, mask=None, diagnostic=False):
    image = original.copy()
    if mask:
        image.paste((127, 127, 127), tuple(mask))
    image = image.crop(tuple(crop)).resize((550, 275), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    sx, sy = image.width / (crop[2] - crop[0]), image.height / (crop[3] - crop[1])
    def xy(point):
        return ((point[0] - crop[0]) * sx, (point[1] - crop[1]) * sy)
    def box(value, color, broken):
        x0, y0 = xy(value[:2]); x1, y1 = xy(value[2:])
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            if broken:
                dashed(draw, a, b, color)
            else:
                draw.line([a, b], fill=color, width=2)
    if truth:
        box(truth["bbox_xyxy"], "#05e0e5", True)
        for point in truth["keypoints"]:
            if point[2] == 2:
                x, y = xy(point)
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="black", width=4)
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="#05e0e5", width=2)
    if prediction:
        box(prediction["bbox_xyxy"], "#ff7639", diagnostic)
        points = [xy(p) for p in prediction["keypoints"]]
        draw.line(points, fill="#ff7639", width=2)
        for x, y in points:
            draw.line([(x - 6, y - 6), (x + 6, y + 6)], fill="black", width=5)
            draw.line([(x - 6, y + 6), (x + 6, y - 6)], fill="black", width=5)
            draw.line([(x - 6, y - 6), (x + 6, y + 6)], fill="#ff7639", width=3)
            draw.line([(x - 6, y + 6), (x + 6, y - 6)], fill="#ff7639", width=3)
    return image


def select(spec, role, raw, scores, truth):
    if spec["probe_case"]:
        target = next(t for t in scores[role]["probe_scores"]["target_scores"] if t["case_id"] == spec["probe_case"])
        record = next(r for r in raw[role]["probe"] if r["case_id"] == spec["probe_case"])
        index = target["match"]["prediction_index"]
        return record["predictions"][index], index, target["matched"], "canonical_match"
    records = raw[role]["clean"][spec["image_id"]]
    if spec["row"] is None:
        assert len(records) == (1 if role == "candidate" else 0)
        return (records[0], 0, False, "empty_scene_output") if records else (None, None, False, "no_output")
    target = scores[role]["clean"]["objects"][f'{spec["image_id"]}:{spec["row"]}']
    if target["matched"]:
        index = target["match"]["prediction_index"]
        return records[index], index, True, "canonical_match"
    index = max(range(len(records)), key=lambda i: _box_iou(truth["bbox_xyxy"], records[i]["bbox_xyxy"]))
    return records[index], index, False, "highest_box_iou_diagnostic_only"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    root, evidence = args.root.resolve(), args.evaluation.resolve()
    inputs, scores = read(evidence / "inputs.json"), read(evidence / "scores.json")
    raw = {role: read(evidence / (role + ".json")) for role in ("current", "control", "candidate")}
    assert raw["current"]["clean"] == raw["control"]["clean"]
    assert all(a["predictions"] == b["predictions"] for a, b in zip(raw["current"]["probe"], raw["control"]["probe"]))
    canvas = Image.new("RGB", (1800, 1650), "white")
    draw = ImageDraw.Draw(canvas)
    ink, muted = "#152234", "#455368"
    draw.text((50, 25), "PR62 occlusion experiment: four illustrative cases", font=font(34, True), fill=ink)
    draw.text((50, 75), "Selected after the fixed development comparison. All errors are in original-image pixels.", font=font(22), fill=muted)
    draw.text((50, 110), "Cyan circles / dashed box: source reference     Orange crosses / box: emitted prediction", font=font(22), fill=muted)
    positions = (50, 625, 1200)
    for x, title in zip(positions, ("Original + source reference", "PR35 / matched control", "Occlusion-trained candidate")):
        draw.text((x, 155), title, font=font(25, True), fill=ink)
    index = {"selection": "Post hoc illustrative cases; all fixed-cohort outcomes remain in the numerical evidence.",
             "coordinate_system": "Original 1280 x 720 image pixels; crops enlarged with nearest-neighbour sampling.",
             "model_sha256": {role: raw[role]["model_sha256"] for role in raw},
             "current_control_raw_predictions_equal": True,
             "source_files_sha256": {name: digest(evidence / name) for name in
                 ("inputs.json", "scores.json", "current.json", "control.json", "candidate.json", "verification.json")},
             "renderer_sha256": digest(__file__), "cases": []}
    for number, spec in enumerate(CASES):
        top = 207 + number * 351
        draw.text((50, top), spec["title"], font=font(23, True), fill=ink)
        image_path = root / "data/images/val" / (spec["image_id"] + ".png")
        reference = inputs["clean_references"][spec["image_id"]]
        assert digest(image_path) == reference["image_sha256"]
        original = Image.open(image_path).convert("RGB")
        truth = reference["objects"][spec["row"] - 1] if spec["row"] else None
        case = next(c for c in inputs["probe_cases"] if c["case_id"] == spec["probe_case"]) if spec["probe_case"] else None
        mask = case["mask_rect_xyxy"] if case else None
        if mask:
            assert inputs["probe_protocol"]["masks"]["fill_rgb"] == [127, 127, 127]
            masked = original.copy()
            masked.paste((127, 127, 127), tuple(mask))
            assert hashlib.sha256(masked.tobytes()).hexdigest() == case["input_rgb_sha256"]
        item = {**spec, "image_path": image_path.relative_to(root).as_posix(), "image_sha256": digest(image_path),
                "reference": truth, "mask_rect_xyxy": mask,
                "masked_input_rgb_sha256": case["input_rgb_sha256"] if case else None, "models": {}}
        canvas.paste(panel(original, spec["crop"], truth, None), (positions[0], top + 35))
        source_note = "Known coordinates before masking" if mask else ("No target in this empty reference image" if truth is None else "Frozen visible-front reference")
        draw.text((positions[0], top + 316), source_note, font=font(20), fill=muted)
        for column, role in enumerate(("current", "candidate"), 1):
            prediction, array_index, matched, basis = select(spec, role, raw, scores, truth)
            diagnostic = basis == "highest_box_iou_diagnostic_only"
            canvas.paste(panel(original, spec["crop"], truth, prediction, mask, diagnostic), (positions[column], top + 35))
            errors = [math.dist(p[:2], q[:2]) for p, q in zip(truth["keypoints"], prediction["keypoints"])] if truth and prediction else None
            overlap = _box_iou(truth["bbox_xyxy"], prediction["bbox_xyxy"]) if truth and prediction else None
            if number == 0:
                note = f"Covered left error: {errors[0]:.2f} px"
            elif number == 1:
                note = f"Left / right error: {errors[0]:.2f} / {errors[1]:.2f} px"
            elif number == 2:
                note = f"IoU {overlap:.4f} | " + ("matched; both errors < 5 px" if matched else "unmatched diagnostic; errors < 5 px")
            else:
                note = f"False output; box confidence {prediction['confidence']:.4f}" if prediction else "No output"
            draw.text((positions[column], top + 316), note, font=font(19 if number == 2 else 20), fill=muted)
            item["models"][role] = {"array_index_zero_based": array_index, "canonical_matched": matched,
                "selection_basis": basis, "prediction": prediction, "endpoint_errors_px": errors, "box_iou": overlap}
        index["cases"].append(item)
    draw.line((50, 1624, 1750, 1624), fill="#d2d9e3", width=1)
    draw.text((50, 1630), "Fixed thresholds: box 0.13, endpoint 0.50, matching IoU 0.50. Row C's diagnostic does not count as a scored success.", font=font(16), fill=muted)
    output = root / "docs/images/occlusion-results-20261114.png"
    canvas.save(output)
    index["figure_sha256"] = digest(output)
    (output.parent / "occlusion-results-20261114-index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
