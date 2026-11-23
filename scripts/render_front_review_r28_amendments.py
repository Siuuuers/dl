#!/usr/bin/env python3
"""Render the complete accepted R27-to-R28 change set from original pixels."""
import argparse
from collections import Counter
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.pdfgen import canvas

from render_front_review_r27_amendments import (
    WIDTH, HEIGHT, MARGIN, INK, MUTED, RULE, PALE, BEFORE, AFTER,
    require, sha, register_fonts, text, para, field_changes, bounds_for, draw_panel,
)

GROUPS = [
    ("new_000008", [5]), ("new_000008", [11]), ("new_000008", [23]),
    ("new_000010", [14, 33]), ("new_000010", [26]),
    ("new_000011", [49]), ("new_000016", [26, 27]),
    ("new_000017", [18]), ("new_000028", [59]), ("new_000055", [23]),
    ("new_000056", [12]), ("new_000058", [7]), ("new_000058", [18]),
    ("new_000059", [7]), ("new_000059", [10]), ("new_000087", [53]),
    ("new_000088", [52]), ("new_000089", [50]),
    ("new_000090", [44, 45]), ("new_000094", [9]),
]
REASONS = {
    ("new_000008", 5): "Move the left point six pixels right, onto the visible rounded turn. The requested direction is supported; a literal twelve-pixel move would put the point farther inside the face. Preserve its y-value and the box.",
    ("new_000008", 11): "Correct the upper boundary, ymin: the old top crosses the cassette above. Keep ymax at 501; the predicted lower edge at about 506.66 includes rack below. This is an explicit upper-edge correction, despite the original feedback naming ymax.",
    ("new_000008", 23): "Move the left point to the curved front turn between the saved and predicted locations. Retain ymax at 372: the predicted lower bound around 379.30 reaches into the next front or separation below.",
    ("new_000010", 14): "Rows 14 and 33: include the thin cassette strip visible to the left of the gripper. Column boundaries and front levels support connecting this strip with the aperture fragment. The box spans the occluder between observed parts; this correspondence is a scene-topology inference. Both unavailable corners remain unavailable. Preserve row 14's preferred xmax at 747.",
    ("new_000010", 26): "Include the visible side rims, then move the right point to the lower-front turn. The box's outer rim and the keypoint need not coincide. Retain ymax near 430; the predicted 435.88 extends into the next cassette. Keep the existing top as well.",
    ("new_000011", 49): "Move the right point from the face interior onto the rounded turn. Extend xmax to contain the visible outer rim. The other saved coordinates remain supported by the original pixels.",
    ("new_000016", 26): "Row 26: tighten xmax to the visible material boundary indicated by the blue screenshot line. Row 27: move the right point above and right to the rounded turn indicated in orange. Coordinates were checked on the original image; uncertainty is about two pixels. Protected standard row 24 is unchanged.",
    ("new_000017", 18): "Additional TXT-screen case. The saved top includes the preceding front above the dark separator. Move ymin to this cassette's own upper edge. Keep both corners and the lower bound. A neighboring point inside a rectangle is a review cue, not an automatic error.",
    ("new_000028", 59): "Additional TXT-screen case. The old lower bound includes the black rack and background under the slanted front. Move the lower boundary upward to y534, at the lowest supported visible material. Keep the locatable right corner and the exact unavailable-left triplet.",
    ("new_000055", 23): "The frame-clipped front ends near the dark divider around x638-640. Tighten xmax from 651 to 640, with about 2-3 pixels of uncertainty. The suggested x538 would cut away most of this front and was not applied or reinterpreted as xmin.",
    ("new_000056", 12): "The blue screenshot line marks the upper boundary, ymin. Move this boundary down to y437 to exclude the front above. Retain ymax near 475.54: the prediction near 481.12 extends into rack/background. Preserve the R27-corrected left point and unavailable right point.",
    ("new_000058", 7): "Move only the left point rightward from the side/rail onto the visible lower-front turn. Keep the saved right point: the prediction would move it farther inside the face, away from the supported curve.",
    ("new_000058", 18): "Include the upper rim visible at the left, around y400-402. This changes ymin to 401, not ymax. The rim slopes downward to the right, so an axis-aligned box also includes nearby context there. Keep the lower edge at 445 and the exact unavailable-left triplet.",
    ("new_000059", 7): "Move the left point rightward from the side shadow toward the front's visible rounded turn. Preserve the y-value, box and right point.",
    ("new_000059", 10): "Move the left point slightly rightward onto the rounded front edge. The remaining coordinates are retained; no global offset is applied to neighboring fronts.",
    ("new_000087", 53): "Move the right point from inside the front onto its visible lower-right turn. The existing box already contains the new point, so no box change is needed.",
    ("new_000088", 52): "Place both points on their own visible lower-front turns. The left point moves slightly inward from the side boundary; the right point moves right and slightly upward from the lower shadow. Keep the box.",
    ("new_000089", 50): "Move the left point left and upward onto the visible lower-left turn. Expand xmin to include that same rim and contain the point. The gripper-hidden right corner remains unavailable.",
    ("new_000090", 44): "Rows 44-45: move both endpoints onto the physical front turns, using the original contour rather than copying the prediction. Row 45's lower edge also moves down by one pixel; this is a modest-confidence refinement within the image's edge softness.",
    ("new_000094", 9): "Additional TXT-screen case. The old top includes the preceding cassette. Set ymin from this front's upper-left rim. Its upper edge slopes downward to the right; a rectangular box can still contain a neighboring point without crossing the physical contour at every x-position.",
}


def footer(pdf, page, pages):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH - MARGIN, 31)
    text(pdf, "R27 TO R28  /  ORIGINAL-PIXEL ANNOTATION REVIEW  /  23 NOV 2026", MARGIN, 18, 7, MUTED)
    pdf.setFont("Amend", 7)
    pdf.drawRightString(WIDTH - MARGIN, 18, f"{page} / {pages}")


def cover(pdf, inventory, pages):
    text(pdf, "ANNOTATION EVIDENCE  /  COMPLETE ACCEPTED CHANGE SET", MARGIN, HEIGHT - 40, 9, MUTED, True)
    text(pdf, "R28 | Annotation amendments", MARGIN, HEIGHT - 79, 27, INK, True)
    y = para(pdf, "Every changed row appears as the same original crop, its saved R27 annotation, and the accepted R28 amendment. Surrounding fronts and rack pixels provide context. The accompanying notebook prepares the next paired training run.", MARGIN, HEIGHT - 98, WIDTH - 2 * MARGIN, 10.5, 15)
    cards = [(23, "changed rows"), (15, "photographs"), (5, "training rows"), (18, "development rows")]
    cw = (WIDTH - 2 * MARGIN - 36) / 4
    for i, (number, label) in enumerate(cards):
        x = MARGIN + i * (cw + 12)
        pdf.setFillColor(PALE)
        pdf.roundRect(x, y - 75, cw, 58, 4, fill=1, stroke=0)
        text(pdf, str(number), x + 12, y - 44, 23, INK, True)
        text(pdf, label, x + 12, y - 64, 9, MUTED)
    y -= 98
    text(pdf, "How to read this review", MARGIN, y, 11, INK, True)
    yy = para(pdf, "Coordinates use the original 1280 x 720 image: <b>x grows right; y grows down.</b> Points mark the lower-front turns. Their y-values are generally near <b>ymax</b>; the distance to ymin also includes the front height. Original contours decide each change. Predictions, brightness and TXT rules only guide inspection.", MARGIN, y - 15, 366, 9, 13)
    text(pdf, "TXT screening: useful flags, then pixels", MARGIN, yy - 23, 11, INK, True)
    yy = para(pdf, "Screened all 4,447 objects across 157 images. Of 4,288 objects with a locatable corner, the median upper gap was 24.93 px and the median lower gap 2.83 px. Large bottom gaps or foreign points inside a box flagged 879 unique rows. These are not 879 proven errors. Eight selected training cases received visual review: three amended, five retained. This nonrandom sample does not estimate the dataset error rate.", MARGIN, yy - 38, 366, 9, 13)
    yy = para(pdf, "<b>33 decisions:</b> 25 requested rows plus 8 screen cases. The 10 retentions are recorded in the ledger. new_000009:55 awaits the intended prediction crop; its label stays unchanged.", MARGIN, yy - 17, 366, 9, 13)
    rx = 443
    text(pdf, "Changed photos and rows", rx, y, 11, INK, True)
    ty = y - 20
    by_image = {}
    for row in inventory["changed_rows"]:
        by_image.setdefault((row["image_id"], row["split"]), []).append(row["row"])
    for (image_id, split), rows in sorted(by_image.items()):
        text(pdf, image_id.replace("new_000", ""), rx, ty, 8.4, INK, True)
        text(pdf, "train" if split == "train" else "dev", rx + 42, ty, 8, MUTED)
        text(pdf, ", ".join(map(str, sorted(rows))), rx + 85, ty, 8.4, INK)
        ty -= 14
    bottom = min(yy, ty) - 17
    bottom = para(pdf, "<b>Preserved:</b> all object identities, all 11 prior standards, class/visibility tokens, unavailable-point zeros, image splits, and frozen evaluation references. No additions or deletions. Typical visual uncertainty is 1-3 px; exact lines, source hashes and per-field decisions are in the ledger. Training performance remains unmeasured for R28.", MARGIN, bottom, WIDTH - 2 * MARGIN, 8.3, 12)
    require(bottom > 43, "Cover overlaps footer")
    footer(pdf, 1, pages)
    pdf.showPage()


def case_page(pdf, group, rows, image, page, pages, crop_dir):
    image_id, ids = group
    title = "row " + str(ids[0]) if len(ids) == 1 else "rows " + ", ".join(map(str, ids))
    split = "TRAINING" if rows[0]["split"] == "train" else "DEVELOPMENT"
    text(pdf, f"{split}  /  R27 TO R28", MARGIN, HEIGHT - 33, 8, MUTED, True)
    text(pdf, f"{image_id} | {title}", MARGIN, HEIGHT - 62, 21, INK, True)
    bounds = bounds_for(rows)
    crop = image.crop(bounds)
    crop_name = image_id + "-rows-" + "-".join(map(str, ids)) + ".png"
    crop.save(crop_dir / crop_name)
    changes = [change for row in rows for change in field_changes(row)]
    panel_h = min(233, 360 - 17 * len(changes))
    panel_w = (WIDTH - 2 * MARGIN - 24) / 3
    top = HEIGHT - 104
    heights = []
    for i, (label, mode, color) in enumerate((("Original pixels", None, MUTED),
                                             ("R27 | previous annotation", "before_line", BEFORE),
                                             ("R28 | accepted amendment", "after_line", AFTER))):
        x = MARGIN + i * (panel_w + 12)
        text(pdf, label, x, top + 10, 9, color, True)
        heights.append(draw_panel(pdf, crop, bounds, rows, mode, x, top, panel_w, panel_h))
    y = top - max(heights) - 23
    text(pdf, "Changed coordinates | original pixels", MARGIN, y, 10, INK, True)
    text(pdf, "Visible basis for the amendment", 446, y, 10, INK, True)
    ty = y - 20
    columns = [MARGIN + 4, MARGIN + 40, MARGIN + 130, MARGIN + 258]
    pdf.setFillColor(PALE)
    pdf.rect(MARGIN, ty - 4, 389, 17, fill=1, stroke=0)
    for x, label in zip(columns, ("Row", "Field", "R27", "R28")):
        text(pdf, label, x, ty + 1, 8, MUTED, True)
    ty -= 17
    for row, field, before, after in changes:
        for x, value in zip(columns, (str(row), field, before, after)):
            text(pdf, value, x, ty, 8.4, INK)
        pdf.setStrokeColor(RULE)
        pdf.setLineWidth(.35)
        pdf.line(MARGIN, ty - 5, MARGIN + 389, ty - 5)
        ty -= 17
    py = para(pdf, escape(REASONS[(image_id, ids[0])]), 446, y - 13, WIDTH - MARGIN - 446, 9, 13)
    py = para(pdf, "Rings mark locatable lower-front corners. Unavailable corners are not drawn. Coordinate display is rounded; the saved TXT retains the exact values recorded in the ledger.", 446, py - 11, WIDTH - MARGIN - 446, 8, 11.5, MUTED)
    end_y = min(ty, py) - 13
    text(pdf, "Same crop and scale in all panels. Context crop: " + str(bounds) + " (x0, y0, x1, y1).", MARGIN, end_y, 7.5, MUTED)
    require(end_y > 43, f"Case page overlaps footer: {image_id}:{ids}")
    footer(pdf, page, pages)
    pdf.showPage()
    return {"page": page, "image_id": image_id, "rows": ids, "crop_xyxy": bounds,
            "original_crop_file": crop_name, "original_crop_sha256": sha(crop_dir / crop_name),
            "changed_coordinates_display": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    ledger_path = root / "results/review/front-review-20261123-r28/row-review.json"
    manifest_path = root / "data/reviews/fronts-2026-11-23-r28/manifest.json"
    ledger = json.loads(ledger_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    rows = [row for row in ledger["rows"] if row["action"] == "replace"]
    by_id = {(row["image_id"], row["row"]): row for row in rows}
    expected = [(image_id, row) for image_id, ids in GROUPS for row in ids]
    require(len(by_id) == len(rows) == len(expected) == 23 and set(by_id) == set(expected), "Every changed row must appear once")
    require(Counter(row["split"] for row in rows) == {"train": 5, "val": 18}, "Intervention scope differs")
    files = {item["image_id"]: item for item in manifest["files"]}
    images = {}
    for row in rows:
        image_id = row["image_id"]
        path = root / "data/images" / row["split"] / (image_id + ".png")
        require(sha(path) == row["source_image_sha256"] == files[image_id]["source_image_sha256"], "Source image drift: " + image_id)
        label_path = manifest_path.parent / files[image_id]["label_path"]
        require(sha(label_path) == files[image_id]["label_sha256"], "R28 label drift")
        require(label_path.read_text().splitlines()[row["row"] - 1] == row["after_line"], "R28 saved row differs")
        images[image_id] = Image.open(path).convert("RGB")
        require(images[image_id].size == (1280, 720), "Unexpected original dimensions")
    require(len(images) == 15, "Changed photo count differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.crop_dir.mkdir(parents=True, exist_ok=True)
    register_fonts()
    inventory = {
        "schema_version": 1, "kind": "complete_r28_annotation_amendment_pdf",
        "reference_before": "fronts-2026-11-23-r27", "reference_after": manifest["review_id"],
        "ledger": {"path": str(ledger_path.relative_to(root)), "sha256": sha(ledger_path)},
        "review_manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha(manifest_path)},
        "renderer": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": sha(__file__)},
        "shared_render_helpers": {"path": "scripts/render_front_review_r27_amendments.py", "sha256": sha(root / "scripts/render_front_review_r27_amendments.py")},
        "summary": {"changed_rows": 23, "changed_photos": 15, "training_rows": 5, "development_rows": 18, "added_rows": 0, "deleted_rows": 0, "case_pages": len(GROUPS)},
        "changed_rows": rows, "pages": [],
        "verification": {"original_images_hash_verified": True, "every_changed_row_exactly_once": True, "saved_r28_lines_verified": True, "unavailable_corners_not_drawn": True, "predictions_not_drawn": True},
    }
    pdf = canvas.Canvas(str(args.output), pagesize=(WIDTH, HEIGHT), pageCompression=1, invariant=1)
    pdf.setTitle("R28 Annotation Amendments - Original Pixel Review")
    pdf.setAuthor("Siuuuers/dl project review")
    pages = len(GROUPS) + 1
    cover(pdf, inventory, pages)
    for page, group in enumerate(GROUPS, start=2):
        image_id, ids = group
        inventory["pages"].append(case_page(pdf, group, [by_id[(image_id, row)] for row in ids], images[image_id], page, pages, args.crop_dir))
    pdf.save()
    inventory["pdf"] = {"filename": args.output.name, "sha256": sha(args.output), "pages": pages, "bytes": args.output.stat().st_size}
    target = args.inventory or root / "results/review/front-review-20261123-r28/amendment-pdf-inventory.json"
    target.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"pdf": str(args.output), **inventory["pdf"], "inventory": str(target)}, indent=2))


if __name__ == "__main__":
    main()
