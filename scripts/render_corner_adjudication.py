#!/usr/bin/env python3
"""Render exact original pixels and the two frozen adjudication decisions."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    review_path = args.review_root / "results/review/corner-adjudication-20261112/review.json"
    review = json.loads(review_path.read_text())
    source_path = args.repo_root / review["source_image"]["path"]
    assert sha(source_path) == review["source_image"]["sha256"]
    parent = args.review_root / "data/references/visible-front-corners-2026-11-12/labels/val/new_000010.txt"
    new = args.review_root / "data/references/visible-front-adjudication-2026-11-12/labels/val/new_000010.txt"
    assert sha(parent) == review["parent_reference"]["target_label_sha256"]
    assert sha(new) == review["new_reference"]["target_label_sha256"]
    before, after = parent.read_text().splitlines(), new.read_text().splitlines()
    changes = {c["row_one_based"]: c for c in review["accepted_changes"]}
    assert set(changes) == {28, 32}
    for row, change in changes.items():
        assert before[row - 1] == change["before_row_text"]
        assert after[row - 1] == change["after_row_text"]
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    regular, bold = font_dir / "DejaVuSans.ttf", font_dir / "DejaVuSans-Bold.ttf"
    font = lambda size, strong=False: ImageFont.truetype(str(bold if strong else regular), size)
    source = Image.open(source_path).convert("RGB")
    page = Image.new("RGB", (1100, 930), "white")
    draw = ImageDraw.Draw(page)
    ink, muted, orange, green = "#18232e", "#536271", "#f3a343", "#27df76"
    draw.text((24, 18), "One corner refined; one marked unlocatable", fill=ink, font=font(26, True))
    draw.text((24, 58), "new_000010  |  original pixels and frozen reference decisions", fill=muted, font=font(18))
    columns = (24, 384, 744)
    for x, label in zip(columns, ("Original pixels", "Previous label", "Adjudicated label")):
        draw.text((x, 101), label, fill=ink, font=font(19, True))
    cases = [
        {"row": 28, "crop": [911, 477, 950, 503], "scale": 8, "top": 174,
         "title": "28L: exposed lower-left wax turn", "caption": "Left = (923, 492), v=2"},
        {"row": 32, "crop": [442, 635, 483, 678], "scale": 7, "top": 518,
         "title": "32L: lower fragment beside the hand", "caption": "Left = unlocatable (0 0 0)"},
    ]
    for case in cases:
        row, box, scale, top = case["row"], case["crop"], case["scale"], case["top"]
        draw.text((24, top - 39), case["title"], fill=ink, font=font(19, True))
        crop = source.crop(box)
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
        lefts = []
        for column in columns:
            left = column + (332 - crop.width) // 2
            lefts.append(left)
            page.paste(crop, (left, top))
        change = changes[row]
        for column_index, point, color in (
                (1, change["before_pixel_triplet"], orange),
                (2, change["serialized_pixel_triplet"], green)):
            x, y, visible = point
            if not visible:
                continue
            x = lefts[column_index] + (x - box[0]) * scale
            y = top + (y - box[1]) * scale
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=ink)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        foot = top + crop.height + 12
        draw.text((columns[0], foot), f"{scale}x nearest-neighbor enlargement", fill=muted, font=font(15))
        x, y, _ = change["before_pixel_triplet"]
        draw.text((columns[1], foot), f"Left = ({x:.2f}, {y:.2f}), v=2", fill=ink, font=font(16))
        draw.text((columns[2], foot), case["caption"], fill=ink, font=font(16))
    draw.text((24, 877), "The object and its box remain. A missing point records an unreliable location, not an absent object.",
              fill=muted, font=font(16))
    draw.text((24, 903), "No model prediction or inferred hidden coordinate is shown. Magnification adds no new image detail.",
              fill=muted, font=font(15))
    assert not args.output_dir.exists() or not any(args.output_dir.iterdir()), "Use an empty output directory"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "corner-adjudication-20261112.png"
    page.save(output)
    index = {"source_image_sha256": sha(source_path), "parent_label_sha256": sha(parent),
             "new_label_sha256": sha(new), "review_sha256": sha(review_path),
             "renderer_sha256": sha(Path(__file__)), "pillow_version": pillow_version,
             "font_sha256": {p.name: sha(p) for p in (regular, bold)}, "cases": cases,
             "output": {"file": output.name, "sha256": sha(output), "dimensions": list(page.size)}}
    (args.output_dir / "corner-adjudication-figure-index-20261112.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index["output"]))


if __name__ == "__main__":
    main()
