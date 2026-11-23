#!/usr/bin/env python3
"""Render four focused original/r23/saved-r24 comparison sheets with hash receipts."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REVIEWS = {"r23": "data/reviews/fronts-2026-11-22-r23/manifest.json",
           "r24": "data/reviews/fronts-2026-11-23-r24/manifest.json"}
LEDGER = "results/review/front-review-20261123-r24"
GROUPS = {"new_000010": ((15, 16, 17, 18, 19),),
          "new_000012": ((4, 5, 6, 7, 8, 9),),
          "new_000014": ((21, 22, 23, 24, 25), (31, 32)),
          "new_000016": ((18, 19), (24,), (25, 26, 27))}
STANDARDS = {("new_000010", 16), ("new_000016", 24)}
COLORS = {"r23": "#ff4dc5", "changed": "#20e6a3", "retained": "#46ddee"}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(root, path):
    return {"path": str(path.relative_to(root)), "sha256": digest(path)}


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def decode(line, size):
    tokens = line.split()
    require(len(tokens) == 11, "Expected two-point YOLO pose labels")
    v = list(map(float, tokens))
    w, h = size
    return {"tokens": tokens,
            "box": [(v[1] - v[3] / 2) * w, (v[2] - v[4] / 2) * h,
                    (v[1] + v[3] / 2) * w, (v[2] + v[4] / 2) * h],
            "left": [v[5] * w, v[6] * h, int(v[7])],
            "right": [v[8] * w, v[9] * h, int(v[10])]}


def main(root):
    root = root.resolve()
    manifests = {rev: json.loads((root / path).read_text()) for rev, path in REVIEWS.items()}
    require(manifests["r24"]["parent"] == descriptor(root, root / REVIEWS["r23"]),
            "r24 parent manifest differs")
    source_path = root / "data/manifest.json"
    require(all(manifest["source_manifest_sha256"] == digest(source_path)
                for manifest in manifests.values()), "Source dataset manifest differs")
    records = {item["id"]: item for item in json.loads(source_path.read_text())["records"]}
    entries = {rev: {item["image_id"]: item for item in manifest["files"]}
               for rev, manifest in manifests.items()}
    audit_path = root / LEDGER / "point-review.json"
    require(descriptor(root, audit_path) in manifests["r24"]["review_evidence"],
            "Saved point audit hash differs")
    audit = json.loads(audit_path.read_text())
    decisions = {(item["image_id"], item["row"]): item for item in audit["rows"]}
    standards = {(item["image_id"], item["row"]): item for item in audit["preserved_standards"]}
    rendered_rows = {(iid, row) for iid, groups in GROUPS.items() for rows in groups for row in rows}
    require(set(standards) == STANDARDS and rendered_rows == set(decisions) | STANDARDS,
            "Comparison coverage differs from the focused review")
    out_dir = root / LEDGER
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for image_id, groups in GROUPS.items():
        record = records[image_id]
        image_path = root / "data" / record["image"]
        source_hash = digest(image_path)
        original = Image.open(image_path).convert("RGB")
        require(original.size == tuple(record["original_size"]) == (1280, 720), "Image size differs")
        labels, label_sources, points = {}, {}, {}
        for rev, manifest_path in REVIEWS.items():
            item = entries[rev][image_id]
            require(item["source_image_sha256"] == source_hash == record["image_sha256"],
                    f"Original image hash differs: {image_id}")
            require(item["split"] == record["split"], f"Split differs: {image_id}")
            label_path = (root / manifest_path).parent / item["label_path"]
            require(digest(label_path) == item["label_sha256"], f"Saved label hash differs: {rev}/{image_id}")
            labels[rev] = label_path.read_text().splitlines()
            label_sources[rev] = descriptor(root, label_path)
            points[rev] = {row: decode(labels[rev][row - 1], original.size)
                           for rows in groups for row in rows}
        require(len(labels["r23"]) == len(labels["r24"]), "Instance count differs")
        evidence = []
        for rows in groups:
            for row in rows:
                key = (image_id, row)
                old, new = points["r23"][row], points["r24"][row]
                require(old["tokens"][:5] == new["tokens"][:5] and
                        all(old["tokens"][i] == new["tokens"][i] == "2" for i in (7, 10)),
                        f"Non-XY fields differ: {key}")
                if key in standards:
                    require(labels["r23"][row - 1] == labels["r24"][row - 1] == standards[key]["line"],
                            f"Accepted reference changed: {key}")
                    scope = "accepted_standard"
                else:
                    decision = decisions[key]
                    require(decision["before_line"] == labels["r23"][row - 1] and
                            decision["after_line"] == labels["r24"][row - 1], f"Saved row/audit mismatch: {key}")
                    require(decision["source_image_sha256"] == source_hash, f"Audit source differs: {key}")
                    scope = decision["scope"]
                    for side in ("left", "right"):
                        check = decision["point_reviews"][side]
                        for rev, field in (("r23", "before_pixel_xy"), ("r24", "after_pixel_xy")):
                            require(all(abs(a - b) < 1e-6 for a, b in zip(points[rev][row][side][:2], check[field])),
                                    f"Saved coordinate/audit mismatch: {key}/{side}")
                evidence.append({"row": row, "scope": scope,
                                 "r23_left_xyv": old["left"], "r23_right_xyv": old["right"],
                                 "r24_left_xyv": new["left"], "r24_right_xyv": new["right"]})
        # Separate nearby stacks; use the same crop in all three columns.
        scale, margin, gap, header_h, footer_h = 2, 24, 16, 146, 86
        crops = []
        for rows in groups:
            boxes = [points["r23"][row]["box"] for row in rows]
            box = (max(0, math.floor(min(b[0] for b in boxes)) - 12),
                   max(0, math.floor(min(b[1] for b in boxes)) - 12),
                   min(original.width, math.ceil(max(b[2] for b in boxes)) + 12),
                   min(original.height, math.ceil(max(b[3] for b in boxes)) + 12))
            crop = original.crop(box).resize(((box[2] - box[0]) * scale, (box[3] - box[1]) * scale),
                                             Image.Resampling.NEAREST)
            crops.append((rows, box, crop))
        panel_w = max(crop.width for _, _, crop in crops)
        width = margin * 2 + panel_w * 3 + gap * 2
        height = header_h + sum(crop.height + 54 for _, _, crop in crops) + footer_h
        canvas = Image.new("RGB", (width, height), "#101b25")
        draw = ImageDraw.Draw(canvas)
        title_font, body_font, small_font = font(26), font(18), font(16)
        draw.text((margin, 16), f"{image_id} | lower front-face corners", font=title_font, fill="#f1f5f9")
        scope_note = ("Validation image: review labels only; trainer validation and frozen scoring references stay unchanged."
                      if record["split"] == "val" else
                      "Training image: both endpoints reviewed in the named rows" +
                      (" and adjacent row 5 (marked +)." if image_id == "new_000012" else "."))
        draw.text((margin, 54), scope_note, font=small_font, fill="#b3c1ce")
        draw.text((margin, 80), "Magenta: r23 saved   |   Green: r24 changed XY   |   Cyan: r24 retained XY", font=body_font, fill="#dbe7f0")
        for col, name in enumerate(("Original pixels", "R23 | previous saved labels", "R24 | current saved labels")):
            draw.text((margin + col * (panel_w + gap), 114), name, font=body_font, fill="#edf3f8")
        top, crop_receipts = header_h, []
        for rows, box, crop in crops:
            def suffix(row):
                if (image_id, row) in STANDARDS:
                    return "*"
                return "+" if decisions[(image_id, row)]["scope"] == "adjacent_context" else ""
            names = [str(row) + suffix(row) for row in rows]
            caption = f"Rows {', '.join(names)}  |  source crop {box}  |  original pixels enlarged {scale}x"
            draw.text((margin, top + 8), caption, font=small_font, fill="#b3c1ce")
            crop_top = top + 36
            for col, rev in enumerate((None, "r23", "r24")):
                panel_left = margin + col * (panel_w + gap)
                canvas.paste(crop, (panel_left, crop_top))
                if rev is None:
                    continue
                for row in rows:
                    for side, short in (("left", "L"), ("right", "R")):
                        x, y, _ = points[rev][row][side]
                        changed = points["r23"][row][side] != points["r24"][row][side]
                        color = COLORS["r23" if rev == "r23" else ("changed" if changed else "retained")]
                        cx, cy = panel_left + (x - box[0]) * scale, crop_top + (y - box[1]) * scale
                        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=color, width=2)
                        label = f"{row}{short}" + suffix(row)
                        label_x = cx + 8 if side == "left" else cx - draw.textlength(label, font=small_font) - 8
                        draw.text((label_x, cy - 21), label, font=small_font, fill=color,
                                  stroke_width=1, stroke_fill="#101b25")
            crop_receipts.append({"rows": list(rows), "source_crop_xyxy": box, "integer_enlargement": scale})
            top += crop.height + 54
        draw.text((margin, top + 8), "Markers use saved coordinates. Boxes, classes, visibility and row identities are unchanged.", font=small_font, fill="#dbe7f0")
        uncertainty = ("Colour records edit status. Placement ambiguity: about 1-2 px; shadowed right corners 25-27: about 2-3 px."
                       if image_id == "new_000016" else
                       "Colour records edit status; it is not a certainty score. Rounded, blurred boundaries retain about 1-2 px of ambiguity.")
        draw.text((margin, top + 33), uncertainty, font=small_font, fill="#b3c1ce")
        draw.text((margin, top + 58), "* User-accepted reference, retained exactly." if any((image_id, r) in STANDARDS for g in groups for r in g)
                  else ("+ The single adjacent-context row; source crops and hashes are in rendering-manifest.json."
                        if image_id == "new_000012" else
                        "All source crops and label hashes are recorded in rendering-manifest.json."), font=small_font, fill="#b3c1ce")
        output = out_dir / f"{image_id}-front-face-comparison.png"
        canvas.save(output, optimize=True)
        artifacts.append({**descriptor(root, output), "image_size": list(canvas.size), "image_id": image_id,
                          "split": record["split"], "source_image": descriptor(root, image_path),
                          "label_sources": label_sources, "crop_groups": crop_receipts, "point_evidence": evidence})
    receipt = {"schema_version": 1, "kind": "saved_front_review_comparison", "review_id": manifests["r24"]["review_id"],
               "renderer": descriptor(root, Path(__file__).resolve()),
               "source_manifests": {rev: descriptor(root, root / path) for rev, path in REVIEWS.items()},
               "point_review": descriptor(root, audit_path), "source_and_label_hashes_verified": True,
               "saved_coordinates_verified": True, "unchanged_fields_verified": True,
               "accepted_standards_retained": True, "artifacts": artifacts}
    receipt_path = out_dir / "rendering-manifest.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"manifest": str(receipt_path.relative_to(root)),
                      "outputs": [item["path"] for item in artifacts], "saved_coordinates_verified": True}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().root)
