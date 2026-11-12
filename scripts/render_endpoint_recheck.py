#!/usr/bin/env python3
"""Show the user-requested right-corner recheck using exact original pixels."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "results/review/baomai-endpoint-recheck-20261112"
CANDIDATE = ROOT / "data/candidates/endpoint-visibility-2026-11-12"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = json.loads((REVIEW / "adjudication.json").read_text())
    manifest = json.loads((CANDIDATE / "manifest.json").read_text())
    review_path = CANDIDATE / manifest["review"]["path"]
    review = json.loads(review_path.read_text())
    assert sha(review_path) == manifest["review"]["sha256"]
    assert review["evidence"]["sha256"] == sha(REVIEW / "adjudication.json")
    image_path = ROOT / evidence["source_image"]["path"]
    assert sha(image_path) == evidence["source_image"]["sha256"]
    image = Image.open(image_path).convert("RGB")
    assert list(image.size) == evidence["source_image"]["size"]
    change = next(c for c in manifest["changes"] if c["image_id"] == "new_000023")
    label_path = CANDIDATE / change["candidate_label"]
    assert sha(label_path) == change["candidate_label_sha256"]
    point = evidence["root_adjudication"]["right_original_pixels"]
    row = label_path.read_text().splitlines()[32].split()
    assert [float(row[8]) * image.width, float(row[9]) * image.height, int(row[10])] == [*point, 2]

    canvas = Image.new("RGB", (1232, 650), "#f5f6f8")
    draw = ImageDraw.Draw(canvas)
    font = lambda size: ImageFont.truetype(str(FONT), size)
    draw.text((24, 20), "new_000023: the upper right corner is revised", fill="#17212b", font=font(27))
    draw.text((24, 61), "Original row 33: accept (612, 549). Rows 34 and 35: retain missing behind the finger.",
              fill="#465360", font=font(18))
    panels = [(24, (450, 505, 660, 640), 2, "Context · original pixels"),
              (464, (590, 530, 635, 570), 8, "Junction · original pixels"),
              (844, (590, 530, 635, 570), 8, "Same pixels · accepted point")]
    raw_panels, records = [], []
    for index, (left, crop_box, scale, title) in enumerate(panels):
        crop = image.crop(crop_box)
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
        draw.text((left, 110), title, fill="#17212b", font=font(18))
        canvas.paste(crop, (left, 148))
        box = (left, 148, left + crop.width, 148 + crop.height)
        if index < 2:
            raw_panels.append((box, crop))
        else:
            px = left + (point[0] - crop_box[0]) * scale
            py = 148 + (point[1] - crop_box[1]) * scale
            draw.ellipse((px-6, py-6, px+6, py+6), outline="white", width=3)
            draw.ellipse((px-4, py-4, px+4, py+4), outline="#e832ad", width=2)
            draw.line((px+7, py-7, px+47, py-48), fill="#e832ad", width=2)
            draw.text((px+49, py-66), "R33", fill="#e832ad", font=font(18))
        records.append({"crop_xyxy": list(crop_box), "scale": scale,
                        "marks_drawn": index == 2, "resampling": "nearest-neighbor"})
    draw.text((24, 432), "The two lower junctions are covered.", fill="#465360", font=font(17))
    draw.text((464, 482), "Small dark return, left of the brighter tool contour.", fill="#465360", font=font(17))
    draw.text((24, 538), "Moderate-confidence adjudication; manual placement range: x610–614, y547–551.",
              fill="#17212b", font=font(19))
    draw.text((24, 572), "Two original-only reviewers disagreed on this upper junction. The dissent is retained in the review.",
              fill="#465360", font=font(17))
    draw.text((24, 607), "Annotation review only. No model prediction, hidden-corner reconstruction or new training result.",
              fill="#465360", font=font(17))
    for box, crop in raw_panels:
        assert canvas.crop(box).tobytes() == crop.tobytes()
    output = REVIEW / "figures"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "new_000023_right_corner_recheck.png"
    canvas.save(path, optimize=True)
    provenance = {"kind": "original_pixel_right_corner_recheck", "source_image_sha256": sha(image_path),
        "adjudication_sha256": sha(REVIEW / "adjudication.json"), "candidate_manifest_sha256": sha(CANDIDATE / "manifest.json"),
        "candidate_label_sha256": sha(label_path), "point_original_pixels": point, "original_row": 33,
        "panels": records, "unmarked_panels_pixel_exact": True, "annotation_not_prediction": True,
        "inference_or_training": False, "renderer_sha256": sha(Path(__file__)), "font_sha256": sha(FONT),
        "output": {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}}
    (output / "figure-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"figure": str(path), "sha256": sha(path), "raw_panels_pixel_exact": True}))


if __name__ == "__main__":
    main()
