#!/usr/bin/env python3
"""Show unchanged upstream keypoints over verified source pixels.

These figures are new displays of the saved baomai TXT annotations, not
copies of the upstream editor's exported previews. No prediction is read.
"""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version


COLORS = {"left": "#35e975", "right": "#45b5ff", "pair": "#fa4f5d"}
INK, MUTED, PAPER = "#18232e", "#536271", "#ffffff"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size, bold=False):
    return ImageFont.truetype(
        str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size
    )


def panel_points(draw, origin, scale, crop, annotation):
    points = []
    for side in ("left", "right"):
        x, y = annotation[side + "_xy_px"]
        points.append((origin[0] + (x - crop[0]) * scale,
                       origin[1] + (y - crop[1]) * scale))
    draw.line(points, fill=COLORS["pair"], width=2)
    for side, (x, y) in zip(("left", "right"), points):
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=INK)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=COLORS[side])


def badge(draw, xy, text):
    x, y = xy
    f = font(18, True)
    bounds = draw.textbbox((0, 0), text, font=f)
    width = bounds[2] - bounds[0] + 16
    draw.rounded_rectangle((x - width / 2, y - 15, x + width / 2, y + 15),
                           radius=5, fill=INK)
    draw.text((x, y - 1), text, font=f, fill=PAPER, anchor="mm")


def render(case, source, output):
    crop, scale = case["crop_xyxy"], case["display_scale"]
    panel = source.crop(crop)
    panel = panel.resize((panel.width * scale, panel.height * scale),
                         Image.Resampling.NEAREST)
    margin, gap, top = 24, 24, 126
    width = 2 * panel.width + 2 * margin + gap
    height = top + panel.height + 89
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 15), case["title"], font=font(25, True), fill=INK)
    draw.text((margin, 51), case["image"] + "  |  original TXT rows " +
              ", ".join(str(a["row_1_based"]) for a in case["annotations"]),
              font=font(18), fill=MUTED)
    left = (margin, top)
    right = (margin + panel.width + gap, top)
    for origin in (left, right):
        canvas.paste(panel, origin)
    draw.text((left[0], 96), "Original pixels", font=font(18, True), fill=INK)
    draw.text((right[0], 96), "Original baomai point pairs", font=font(18, True), fill=INK)
    for a in case["annotations"]:
        panel_points(draw, right, scale, crop, a)
        bx, by = case["face_id_text_positions_px"][str(a["row_1_based"])]
        badge(draw, (right[0] + (bx - crop[0]) * scale,
                     right[1] + (by - crop[1]) * scale), str(a["row_1_based"]))
    y = top + panel.height + 15
    for x, color, text in ((margin, COLORS["left"], "Left point"),
                           (margin + 165, COLORS["right"], "Right point"),
                           (margin + 345, COLORS["pair"], "Same front")):
        draw.ellipse((x, y + 5, x + 10, y + 15), fill=color)
        draw.text((x + 18, y), text, font=font(16), fill=INK)
    draw.text((margin, y + 30),
              f"Unchanged labels; {scale}x nearest-neighbor view. Red lines join saved pairs; they are not traced contours.",
              font=font(14), fill=MUTED)
    canvas.save(output, format="PNG", optimize=False)
    return list(canvas.size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    manifest_path = args.repo_root / "data/manifest.json"
    if digest(manifest_path) != spec["manifest_sha256"]:
        raise ValueError("Original manifest does not match the pinned upstream source")
    manifest = json.loads(manifest_path.read_text())
    records = {r["id"]: r for r in manifest["records"]}
    inputs = {}
    images = {}
    for case in spec["cases"]:
        image_id = case["image"]
        record = records[image_id]
        for kind in ("image", "label"):
            relative = case[kind + "_path"]
            path = args.repo_root / relative
            expected = case[kind + "_sha256"]
            if digest(path) != expected or record[kind + "_sha256"] != expected:
                raise ValueError(f"Source hash mismatch: {relative}")
            inputs[relative] = expected
        source = Image.open(args.repo_root / case["image_path"]).convert("RGB")
        if list(source.size) != spec["source_dimensions"]:
            raise ValueError(f"Unexpected original dimensions: {image_id}")
        images[image_id] = source
        lines = (args.repo_root / case["label_path"]).read_text().splitlines()
        for a in case["annotations"]:
            line = lines[a["row_1_based"] - 1]
            if line != a["original_yolo_line"]:
                raise ValueError(f"Changed original row: {image_id}:{a['row_1_based']}")
            values = [Decimal(v) for v in line.split()]
            if len(values) != 11 or values[7] != 2 or values[10] != 2:
                raise ValueError("Examples must have two original visible endpoints")
            for side, offset in (("left", 5), ("right", 8)):
                pixels = [float(values[offset] * source.width),
                          float(values[offset + 1] * source.height)]
                if pixels != a[side + "_xy_px"]:
                    raise ValueError("Saved coordinate decoding differs from the original TXT")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for case in spec["cases"]:
        path = args.output_dir / case["output_file"]
        size = render(case, images[case["image"]], path)
        outputs.append({"file": path.name, "sha256": digest(path), "dimensions": size,
                        "case_id": case["id"], "crop_xyxy": case["crop_xyxy"],
                        "display_scale": case["display_scale"],
                        "rows_1_based": [a["row_1_based"] for a in case["annotations"]]})
    index = {
        "source_commit": spec["source_commit"],
        "meaning": "New displays of unchanged original TXT annotations on original PNG pixels",
        "source_manifest_sha256": digest(manifest_path),
        "spec_sha256": digest(args.spec),
        "renderer_sha256": digest(Path(__file__)),
        "pillow_version": pillow_version,
        "font_sha256": {name: digest(FONT_DIR / name) for name in
                         ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")},
        "verified_original_sources": inputs,
        "outputs": outputs,
    }
    (args.output_dir / "baomai-corner-examples-index-20261112.json").write_text(
        json.dumps(index, indent=2) + "\n")
    print(json.dumps({"verified_source_files": len(inputs), "figures": outputs}, indent=2))


if __name__ == "__main__":
    main()
