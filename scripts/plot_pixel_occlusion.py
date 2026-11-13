#!/usr/bin/env python3
"""Reproduce the disclosed example crops from a completed pixel-occlusion probe."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
from probe_pixel_occlusion import altered, mask_rect, read, require, sha


def digest(path):
    return sha(Path(path).read_bytes())


def render(args):
    run, image_root = args.run_dir.resolve(), args.image_root.resolve()
    status = read(run / "status.json")
    require(status["status"] == "complete" and status["completed_predictions"] == 94, "run must be complete")
    for name, expected in status["files"].items():
        require(Path(name).name == name and digest(run / name) == expected, "run artifact hash mismatch")
    protocol, cases = read(run / "protocol.json"), read(run / "prepared-cases.json")
    references, scores = read(run / "reference-objects.json"), read(run / "scores.json")["target_scores"]
    preparation = read(run / "preparation.json")
    require(digest(Path(__file__).with_name("probe_pixel_occlusion.py")) == preparation["runner_sha256"], "mask implementation differs")
    require(digest(run / "protocol.json") == status["protocol_sha256"], "protocol mismatch")
    raw = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    records, case_map = {r["case_id"]: r for r in raw[1:]}, {c["case_id"]: c for c in cases}
    lookup = {(s["image_id"], s["row"], s["condition"]): s for s in scores}
    first, last = protocol["targets"][0], protocol["targets"][-1]
    selection = [(first["image_id"], first["row"], "Prospective: first frozen target"),
                 (last["image_id"], last["row"], "Prospective: last frozen target"),
                 ("new_000008", 17, "Post hoc: largest covered-endpoint error")]
    covered = [s for s in scores if s["condition"] in ("left", "right") and s[s["condition"]]["available"]]
    worst = max(covered, key=lambda s: s[s["condition"]]["error_px"])
    require((worst["image_id"], worst["row"], worst["condition"]) == ("new_000008", 17, "left"), "disclosed failure selection changed")
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    regular, bold = fonts / "DejaVuSans.ttf", fonts / "DejaVuSans-Bold.ttf"
    font = lambda size, heavy=False: ImageFont.truetype(str(bold if heavy else regular), size)
    W, H, margin, gap, panel_w, row_h, start_y = 1800, 1308, 30, 14, 424, 370, 145
    canvas = Image.new("RGB", (W, H), "#F5F7FA")
    draw = ImageDraw.Draw(canvas)
    ink, muted, reference_color, predicted_color = "#14273A", "#4C5968", "#00B5E5", "#FF9B37"
    draw.text((margin, 19), "What changes when known corner pixels are removed?", font=font(32, True), fill=ink)
    draw.text((margin, 63), "PR35 · 94 full-image predictions · 4 familiar scenes · synthetic gray masks · no training", font=font(21), fill=muted)
    draw.text((margin, 96), "Crops are for display only. Inference used each complete 1280 × 720 image.", font=font(19), fill=muted)
    draw.text((1020, 97), "×  Known original coordinate", font=font(19, True), fill="#007A9D")
    draw.text((1360, 97), "○  Matched PR35 prediction", font=font(19, True), fill="#A65200")
    panels, source_images = [], {}
    titles = {"intact": "Intact control", "left": "Left corner masked", "right": "Right corner masked", "center": "Central-mask control"}
    for row_index, (image_id, row, reason) in enumerate(selection):
        ref = references[image_id]
        image_path = image_root / (image_id + ".png")
        require(digest(image_path) == ref["image_sha256"], "original image hash mismatch")
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
        require(rgb.size == (ref["width"], ref["height"]), "image dimensions mismatch")
        source_images[image_id] = {"file_sha256": ref["image_sha256"], "label_sha256": ref["label_sha256"]}
        target = ref["objects"][row - 1]
        box = target["bbox_xyxy"]
        crop = [max(0, math.floor(box[0] - 36)), max(0, math.floor(box[1] - 36)),
                min(rgb.width, math.ceil(box[2] + 36)), min(rgb.height, math.ceil(box[3] + 36))]
        top = start_y + row_index * row_h
        draw.text((margin, top), f"{image_id} · reference row {row}    {reason}", font=font(22, True), fill=ink)
        for column, condition in enumerate(titles):
            score = lookup[image_id, row, condition]
            case, record = case_map[score["case_id"]], records[score["case_id"]]
            rect = None if condition == "intact" else mask_rect(box, condition, rgb.size, protocol["masks"])
            require(rect == case["mask_rect_xyxy"], "recreated mask differs")
            pixels = altered(rgb, rect, protocol["masks"]["fill_rgb"])
            require(sha(pixels.tobytes()) == case["input_rgb_sha256"] == record["input_rgb_sha256"], "recreated input differs")
            x, y = margin + column * (panel_w + gap), top + 37
            draw.rounded_rectangle((x, y, x + panel_w, y + 322), radius=7, fill="white", outline="#D6DFE8")
            draw.text((x + 12, y + 9), titles[condition], font=font(20, True), fill=ink)
            patch = pixels.crop(tuple(crop))
            scale = min((panel_w - 20) / patch.width, 196 / patch.height)
            size = (round(patch.width * scale), round(patch.height * scale))
            display = patch.resize(size, Image.Resampling.LANCZOS)
            sx, sy = size[0] / patch.width, size[1] / patch.height
            ox, oy = x + (panel_w - size[0]) // 2, y + 42 + (196 - size[1]) // 2
            local = ImageDraw.Draw(display)
            xy = lambda p: ((p[0] - crop[0]) * sx, (p[1] - crop[1]) * sy)
            if score["pair_available"]:
                line = [xy(score[s]["raw_prediction"]) for s in ("left", "right")]
                local.line(line, fill="black", width=4)
                local.line(line, fill=predicted_color, width=2)
            for point in target["keypoints"]:
                a, b = xy(point)
                for color, width in (("black", 6), (reference_color, 3)):
                    local.line([(a - 6, b - 6), (a + 6, b + 6)], fill=color, width=width)
                    local.line([(a - 6, b + 6), (a + 6, b - 6)], fill=color, width=width)
            if score["matched"]:
                prediction = record["predictions"][score["match"]["prediction_index"]]
                require(all(prediction["keypoints"][i] == score[s]["raw_prediction"] for i, s in enumerate(("left", "right"))), "score/prediction mismatch")
                for side in ("left", "right"):
                    if score[side]["available"]:
                        a, b = xy(score[side]["raw_prediction"])
                        local.ellipse((a - 8, b - 8, a + 8, b + 8), outline="black", width=5)
                        local.ellipse((a - 8, b - 8, a + 8, b + 8), outline=predicted_color, width=3)
            canvas.paste(display, (ox, oy))
            for index, side in enumerate(("left", "right")):
                point = score[side]
                p = point["raw_prediction"]
                label = f"{side[0].upper()}: {point['error_px']:.2f} px · confidence {p[2]:.3f}" if point["available"] else f"{side[0].upper()}: not displayed"
                if point["available"] and not (crop[0] <= p[0] < crop[2] and crop[1] <= p[1] < crop[3]):
                    label += " · off crop"
                draw.text((x + 12, y + 246 + index * 22), label, font=font(17), fill=ink)
            caption = f"Canonical box IoU {score['match']['iou']:.3f}" if score["matched"] else "No canonical box match"
            draw.text((x + 12, y + 294), caption, font=font(16, not score["matched"]), fill=muted)
            panels.append({"image_id": image_id, "row": row, "condition": condition, "selection": reason,
                           "case_id": case["case_id"], "input_rgb_sha256": case["input_rgb_sha256"],
                           "case_record_sha256": sha(json.dumps(case, sort_keys=True, separators=(",", ":")).encode()),
                           "prediction_record_sha256": sha(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()),
                           "crop_xyxy": crop, "display_xywh": [ox, oy, *size], "mask_rect_xyxy": rect,
                           "canonical_match": score["match"], "known_coordinate_state": score["point_states"]})
    draw.text((margin, 1266), "Crosses remain known from the original; they are not observations through a mask. Rings/lines use endpoint confidence ≥0.50.", font=font(18), fill=muted)
    for output in (args.output.resolve(), args.index.resolve()):
        require(not output.is_relative_to(run) and not output.is_relative_to(image_root), "output would modify source artifacts")
        output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format="PNG", optimize=True)
    index = {"schema_version": 1, "selection": "First and last targets fixed before results; third row selected after results as the largest error on the covered side. Selection does not change metrics.",
             "posthoc_maximum": {"image_id": worst["image_id"], "row": worst["row"], "condition": worst["condition"], "covered_error_px": worst["left"]["error_px"]},
             "sources": {name: digest(run / name) for name in ("status.json", "protocol.json", "preparation.json", "prepared-cases.json", "predictions.jsonl", "scores.json", "reference-objects.json")},
             "source_images": source_images, "panels": panels, "plot_script_sha256": digest(__file__),
             "renderer": {"Pillow": pillow_version, "crop_padding_px": 36, "display_resampling": "LANCZOS",
                          "fonts": {p.name: digest(p) for p in (regular, bold)}},
             "output": {"file": args.output.name, "sha256": digest(args.output), "width": W, "height": H}}
    args.index.write_text(json.dumps(index, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"figure_sha256": index["output"]["sha256"], "panels": len(panels)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "image-root", "output", "index"):
        parser.add_argument("--" + name, type=Path, required=True)
    render(parser.parse_args())
