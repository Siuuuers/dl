#!/usr/bin/env python3
"""Render exact original-pixel crops and explicitly distinguished geometry proposals."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {"reference": "#20dba2", "linear": "#63b9ff", "homography": "#ffad52"}
BG, INK, MUTED = "#111821", "#f3f5f8", "#bac4d0"
SCALE = 3


def font(size):
    return ImageFont.truetype("DejaVuSans.ttf", size)


def marker(draw, xy, kind, color):
    x, y = xy
    r = 7
    if kind == "reference":
        draw.rectangle((x-r, y-r, x+r, y+r), outline=color, width=2)
    elif kind == "linear":
        draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=2)
    else:
        draw.line((x-r, y-r, x+r, y+r), fill=color, width=3)
        draw.line((x-r, y+r, x+r, y-r), fill=color, width=3)


def label(draw, xy, text):
    x, y = xy
    bounds = draw.textbbox((x, y), text, font=font(15))
    draw.rectangle((bounds[0]-3, bounds[1]-2, bounds[2]+3, bounds[3]+2), fill=BG)
    draw.text((x, y), text, font=font(15), fill=INK)


def make_figure(original, crop, title, subtitle, records, footer, hidden):
    width, height = (crop[2]-crop[0])*SCALE, (crop[3]-crop[1])*SCALE
    canvas = Image.new("RGB", (2*width+72, height+290), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), title, font=font(26), fill=INK)
    draw.text((24, 57), subtitle, font=font(17), fill=MUTED)
    for x, name in ((24, "ORIGINAL PIXELS"), (48+width, "GEOMETRY OVERLAY")):
        draw.text((x, 96), name, font=font(15), fill=MUTED)
    patch = original.crop(crop).resize((width, height), Image.Resampling.NEAREST)
    canvas.paste(patch, (24, 124))
    canvas.paste(patch, (48+width, 124))
    ox, oy = 48+width, 124

    def project(point):
        return ox+(point[0]-crop[0])*SCALE, oy+(point[1]-crop[1])*SCALE

    # Donors are rendered once and remain independent of the withheld targets.
    donors = {d["row"]: d for record in records for d in record["donors"]}
    for row, donor in donors.items():
        pair = [project(point) for point in donor["xy"]]
        draw.line((*pair[0], *pair[1]), fill="#dce2e8", width=1)
        for x, y in pair:
            draw.ellipse((x-4, y-4, x+4, y+4), fill="#ffffff", outline="#222222", width=1)
        label(draw, (pair[1][0]-120, pair[1][1]-24), f"anchor r{row}")

    for record in records:
        if hidden:
            observed = {check["slot"]: check["reference_xy"]
                        for check in record["methods"]["homography"]["visible_endpoint_checks"]}
            for point in observed.values():
                marker(draw, project(point), "reference", COLORS["reference"])
        else:
            for point in record["target_xy_for_scoring_only"]:
                marker(draw, project(point), "reference", COLORS["reference"])
        for method in ("linear", "homography"):
            result = record["methods"][method]
            if result["raw_pair_xy"] is not None:
                for point in result["raw_pair_xy"]:
                    marker(draw, project(point), method, COLORS[method])
        if hidden:
            point = record["methods"]["homography"]["raw_pair_xy"][0]
            px, py = project(point)
            label(draw, (px+20, py-5), f"r{record['row']}: hidden left")

    legend_y = 140+height
    entries = [("reference", "Visible reference" if hidden else "Withheld reference"),
               ("linear", "Linear estimate"), ("homography", "Perspective estimate")]
    for i, (kind, text) in enumerate(entries):
        x = 36+i*310
        marker(draw, (x, legend_y+11), kind, COLORS[kind])
        draw.text((x+17, legend_y), text, font=font(17), fill=INK)
    y = legend_y+36
    for line in footer:
        draw.text((24, y), line, font=font(16), fill=MUTED)
        y += 25
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evidence_bytes = args.evaluation.read_bytes()
    evaluation = json.loads(evidence_bytes)
    identities = {entry["image_id"]: entry for entry in evaluation["provenance"]["source_files"]}

    def original(image_id):
        path = args.repo_root/identities[image_id]["image_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != identities[image_id]["image_sha256"]:
            raise ValueError(f"Original image identity mismatch: {image_id}")
        return Image.open(path).convert("RGB")

    visible = next(case for case in evaluation["lanes"]["interpolation"]["cases"]
                   if case["case_id"] == "new_000008:14")
    hidden = [case for case in evaluation["hidden_cases"]
              if case["case_id"] in ("new_000010:23", "new_000010:24")]
    assert len(hidden) == 2
    error = visible["methods"]
    figures = [
        ("context-corners-holdout-20261112.png", "new_000008", [690, 310, 910, 445],
         "Can neighbors recover a known corner pair?",
         "new_000008 / row 14: target coordinates and box withheld; rows 13 and 15 supply the anchors.",
         [visible],
         [f"Maximum endpoint error: linear {error['linear']['maximum_endpoint_error_px']:.2f} px; "
          f"perspective {error['homography']['maximum_endpoint_error_px']:.2f} px.",
          "Manual stack membership and physical row rank are supplied. This is a coordinate holdout, not an occluded-image test.",
          "Original crop [690, 310, 910, 445], enlarged 3x with nearest-neighbor sampling. No image enhancement."], False),
        ("context-corners-hidden-20261112.png", "new_000010", [735, 345, 955, 515],
         "Hidden corners: useful proposals, still unverified",
         "new_000010 / rows 23 and 24: extend the pattern from rows 21 and 22; the gripper covers both left corners.",
         hidden,
         [f"Hidden-left method disagreement: row 23 = {hidden[0]['method_disagreement']['endpoint_distances_px'][0]:.2f} px; "
          f"row 24 = {hidden[1]['method_disagreement']['endpoint_distances_px'][0]:.2f} px.",
          "The visible right corners are checks only. Their agreement cannot certify the hidden left corners.",
          "No hidden-left ground truth is available. These proposals are not accepted training or reference labels.",
          "Original crop [735, 345, 955, 515], enlarged 3x with nearest-neighbor sampling. No image enhancement."], True),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {"evaluation_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
             "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "pillow_version": Image.__version__, "figures": []}
    for filename, image_id, crop, title, subtitle, records, footer, is_hidden in figures:
        output = args.output_dir/filename
        if output.exists():
            raise FileExistsError(output)
        figure = make_figure(original(image_id), crop, title, subtitle, records, footer, is_hidden)
        figure.save(output)
        index["figures"].append({"filename": filename, "image_id": image_id,
                                "source_image_sha256": identities[image_id]["image_sha256"],
                                "source_crop_xyxy": crop, "nearest_neighbor_scale": SCALE,
                                "size": list(figure.size), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    index_path = args.output_dir/"context-corners-figure-index-20261112.json"
    with index_path.open("x") as stream:
        json.dump(index, stream, indent=2)
        stream.write("\n")
    print(json.dumps(index))


if __name__ == "__main__":
    main()
