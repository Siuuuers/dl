#!/usr/bin/env python3
"""Render the four saved R22/R23 right-corner corrections on original pixels."""

from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("front-face-corner-comparison.png")
REVIEWS = {
    "r22": ROOT / "data/reviews/fronts-2026-11-22-r22",
    "r23": ROOT / "data/reviews/fronts-2026-11-22-r23",
}
CASES = (
    ("new_000001", 27, (488, 518)),
    ("new_000002", 25, (426, 543)),
    ("new_000002", 26, (430, 571)),
    ("new_000002", 27, (432, 600)),
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def main():
    manifests = {key: json.loads((path / "manifest.json").read_text())
                 for key, path in REVIEWS.items()}
    assert manifests["r23"]["source_revision"] == "c1d68bc6ed388077f238c09b74f6a8e15a1375a5"
    assert manifests["r23"]["parent"]["sha256"] == digest(REVIEWS["r22"] / "manifest.json")
    entries = {key: {item["image_id"]: item for item in manifest["files"]}
               for key, manifest in manifests.items()}
    scale, panel_w, panel_h = 4, 344, 232
    margin, gap, header_h, row_h, footer_h = 24, 16, 118, 292, 76
    width = margin * 2 + panel_w * 3 + gap * 2
    height = header_h + len(CASES) * row_h + footer_h
    canvas = Image.new("RGB", (width, height), "#101b25")
    draw = ImageDraw.Draw(canvas)
    title_font, body_font, small_font = font(26), font(19), font(16)
    draw.text((margin, 18), "Front-face corners | four saved R23 corrections",
              font=title_font, fill="#f1f5f9")
    draw.text((margin, 54), "1280 × 720 source images · original pixels enlarged 4× · right landmarks only",
              font=body_font, fill="#b3c1ce")
    for column, label in enumerate(("Original pixels", "R22 · previous", "R23 · corrected")):
        draw.text((margin + column * (panel_w + gap), 88), label,
                  font=body_font, fill=("#dbe7f0", "#ff80cf", "#67efbb")[column])
    evidence = []
    for index, (image_id, row, expected) in enumerate(CASES):
        image_path = ROOT / "data/images/train" / f"{image_id}.png"
        image_hash = digest(image_path)
        labels, points = {}, {}
        for revision, directory in REVIEWS.items():
            item = entries[revision][image_id]
            assert item["split"] == "train", (revision, image_id, "split")
            assert image_hash == item["source_image_sha256"], (revision, image_id, "image hash")
            label_path = directory / item["label_path"]
            assert digest(label_path) == item["label_sha256"], (revision, image_id, "label hash")
            labels[revision] = label_path.read_text().splitlines()[row - 1].split()
            assert len(labels[revision]) == 11
            values = list(map(float, labels[revision]))
            assert values[10] == 2, (revision, image_id, row, "visibility")
            points[revision] = (values[8] * 1280, values[9] * 720)
        assert labels["r22"][:8] == labels["r23"][:8]
        assert labels["r22"][10] == labels["r23"][10]
        assert all(abs(a - b) < 0.00001 for a, b in zip(points["r23"], expected))
        original = Image.open(image_path).convert("RGB")
        assert original.size == (1280, 720)
        old_x, old_y = points["r22"]
        left, top = round(old_x) - 70, round(old_y) - 43
        crop_box = (left, top, left + panel_w // scale, top + panel_h // scale)
        crop = original.crop(crop_box).resize((panel_w, panel_h), Image.Resampling.NEAREST)
        row_top = header_h + index * row_h
        draw.text((margin, row_top + 5), f"{image_id} · row {row} · right corner",
                  font=body_font, fill="#edf3f8")
        coordinate_text = (f"({points['r22'][0]:.0f}, {points['r22'][1]:.0f})"
                           f" → ({points['r23'][0]:.0f}, {points['r23'][1]:.0f}) px")
        draw.text((margin + 640, row_top + 6), coordinate_text,
                  font=small_font, fill="#b3c1ce")
        panel_top = row_top + 39
        for column, revision in enumerate((None, "r22", "r23")):
            panel_left = margin + column * (panel_w + gap)
            canvas.paste(crop, (panel_left, panel_top))
            if revision is not None:
                x, y = points[revision]
                cx = panel_left + (x - left) * scale
                cy = panel_top + (y - top) * scale
                color = "#ff4dc5" if revision == "r22" else "#20e6a3"
                # A hollow ring keeps the saved-coordinate centre unobscured.
                draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline=color, width=2)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    draw.line((cx + 8 * dx, cy + 8 * dy,
                               cx + 12 * dx, cy + 12 * dy), fill=color, width=1)
        evidence.append({"image_id": image_id, "row": row,
                         "source_image_sha256": image_hash,
                         "r22_right_px": points["r22"], "r23_right_px": points["r23"],
                         "source_crop_xyxy": crop_box})
    footer_top = header_h + len(CASES) * row_h
    draw.text((margin, footer_top + 6), "Markers follow saved labels exactly. Front-face corner ≠ outer side silhouette.",
              font=body_font, fill="#dbe7f0")
    draw.text((margin, footer_top + 35), "Only four right-point coordinates change; boxes, left points, visibility and object identities stay unchanged.",
              font=small_font, fill="#b3c1ce")
    canvas.save(OUT, optimize=True)
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "image_size": canvas.size,
                      "source_and_label_hashes_verified": True,
                      "saved_coordinates_verified": True,
                      "unchanged_fields_verified": True,
                      "evidence": evidence}, indent=2))


if __name__ == "__main__":
    main()
