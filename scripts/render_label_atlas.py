#!/usr/bin/env python3
"""Render all frozen photos with current labels as a zoomable review PDF.

Photos keep their native resolution and use JPEG quality 96 for a manageable
review copy. Boxes, landmarks, lines and row numbers are PDF vectors. Original
PNGs remain the pixel authority; this script never invokes a prediction model.
"""
import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import textwrap

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from evaluate_pose import parse_pose_labels

PAGE_W, PAGE_H = landscape(A3)
MARGIN = 30
INK = colors.HexColor("#16232e")
MUTED = colors.HexColor("#556675")
PAPER = colors.HexColor("#f5f7fa")
CYAN = colors.HexColor("#16d9ec")
AMBER = colors.HexColor("#ffb632")
GREEN = colors.HexColor("#48f293")
MAGENTA = colors.HexColor("#ff69dd")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def wrapped(c, text, x, y, width, size=11, leading=15, color=INK):
    c.setFont("Atlas", size)
    c.setFillColor(color)
    for paragraph in text.split("\n"):
        words, line = paragraph.split(), ""
        for word in words:
            trial = (line + " " + word).strip()
            if line and pdfmetrics.stringWidth(trial, "Atlas", size) > width:
                c.drawString(x, y, line)
                y -= leading
                line = word
            else:
                line = trial
        if line:
            c.drawString(x, y, line)
        y -= leading
    return y


def note_excerpt(text, width, size=9, maximum_lines=4):
    """Fit a footer without shrinking the photo; retain full notes in the audit."""
    def line_count(value):
        count, line = 1, ""
        for word in value.split():
            trial = (line + " " + word).strip()
            if line and pdfmetrics.stringWidth(trial, "Atlas", size) > width:
                count += 1
                line = word
            else:
                line = trial
        return count
    if line_count(text) <= maximum_lines:
        return text, False
    words = text.split()
    suffix = " ... Further field details remain in the review record linked on page 2."
    while words and line_count(" ".join(words) + suffix) > maximum_lines:
        words.pop()
    return " ".join(words) + suffix, True


def heading(c, title, subtitle, page, section):
    c.setFillColor(colors.white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("AtlasBold", 23)
    c.drawString(MARGIN, PAGE_H - 38, title)
    c.setFont("Atlas", 10)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, PAGE_H - 57, subtitle)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 36, section)
    c.setStrokeColor(colors.HexColor("#dce3e9"))
    c.line(MARGIN, 34, PAGE_W - MARGIN, 34)
    c.setFont("Atlas", 9)
    c.drawString(MARGIN, 19, "Cassette label atlas  |  14 November 2026  |  Review labels; no model predictions")
    c.drawRightString(PAGE_W - MARGIN, 19, str(page))


def jpeg_reader(path):
    buffer = BytesIO()
    with Image.open(path) as original:
        original.convert("RGB").save(buffer, format="JPEG", quality=96, subsampling=0)
    buffer.seek(0)
    return ImageReader(buffer)


def annotation(c, objects, left, bottom, width, height, image_size, held_rows, crop=None, selected=None):
    iw, ih = image_size
    x0, y0, x1, y1 = crop or (0, 0, iw, ih)
    sx, sy = width / (x1 - x0), height / (y1 - y0)
    def xy(x, y):
        return left + (x - x0) * sx, bottom + height - (y - y0) * sy
    c.saveState()
    path = c.beginPath()
    path.rect(left, bottom, width, height)
    c.clipPath(path, stroke=0)
    for row, obj in enumerate(objects, 1):
        if selected is not None and row not in selected:
            continue
        bx1, by1, bx2, by2 = obj["bbox_xyxy"]
        if bx2 < x0 or bx1 > x1 or by2 < y0 or by1 > y1:
            continue
        color = AMBER if row in held_rows else CYAN
        x, y = xy(bx1, by2)
        c.setStrokeColor(color)
        c.setLineWidth(max(0.7, min(1.25, sx)))
        c.setDash(3, 2) if row in held_rows else c.setDash()
        c.rect(x, y, (bx2 - bx1) * sx, (by2 - by1) * sy, fill=0, stroke=1)
        c.setDash()
        points = obj["keypoints"]
        if all(point[2] == 2 for point in points):
            c.setStrokeColor(GREEN)
            c.setLineWidth(0.75)
            c.line(*xy(*points[0][:2]), *xy(*points[1][:2]))
        for point, tint in zip(points, (GREEN, MAGENTA)):
            if point[2] != 2:
                continue
            px, py = xy(*point[:2])
            radius = 1.8 if crop is None else 3
            c.setFillColor(tint)
            c.setStrokeColor(INK)
            c.setLineWidth(0.5)
            c.circle(px, py, radius, fill=1, stroke=1)
        size = 5.4 if crop is None else 11
        label = str(row)
        tx, ty = xy(max(bx1, x0) + 1.5 / sx, max(by1, y0) + (size + 2) / sy)
        tw = pdfmetrics.stringWidth(label, "AtlasBold", size)
        c.setFillColor(INK)
        c.rect(tx - 1.2, ty - 1.1, tw + 2.4, size + 2, fill=1, stroke=0)
        c.setFillColor(color)
        c.setFont("AtlasBold", size)
        c.drawString(tx, ty, label)
    c.restoreState()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=Path("data/references/visible-front-adjudication-2026-11-12"),
                        help="Separately versioned development reference directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    root, candidate = args.root.resolve(), args.candidate.resolve()
    source_path = root / "data/manifest.json"
    source, train = read(source_path), read(candidate / "manifest.json")
    status = read(args.status)
    reference_dir = args.reference if args.reference.is_absolute() else root / args.reference
    reference = read(reference_dir / "manifest.json")
    if sha(source_path) != train["source_manifest_sha256"] or sha(source_path) != reference["source_manifest_sha256"]:
        raise ValueError("source manifests disagree")
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    for name, filename in (("Atlas", "DejaVuSans.ttf"), ("AtlasBold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    changes = {item["image_id"]: item for item in train["changes"]}
    val_files = {item["image_id"]: item for item in reference["files"]}
    held, notes = defaultdict(set), defaultdict(list)
    for case in status["cases"]:
        if case["status"] == "resolved":
            continue
        iid = case["image_id"]
        mapping = changes.get(iid, {}).get("row_map", [])
        by_original = {item["original_row"]: item["candidate_row"] for item in mapping
                       if item["original_row"] is not None}
        by_addition = {item.get("addition_id"): item["candidate_row"] for item in mapping
                       if item["original_row"] is None}
        for row in case.get("held_original_rows", []):
            target = by_original.get(row) if mapping else row
            if target is not None:
                held[iid].add(target)
        for addition in case.get("held_addition_ids", []):
            target = by_addition.get(addition)
            if target is not None:
                held[iid].add(target)
        notes[iid].append(case["atlas_note"])
    for case in reference["queue_dispositions"]:
        if case["disposition"] != "retained_unresolved":
            continue
        iid = case["image_id"]
        target = val_files[iid]["retained_source_to_revised_row_one_based"][case["source_row_one_based"] - 1]
        if target is not None:
            held[iid].add(target)
        notes[iid].append("Row %s: %s" % (target, " ".join(case["unresolved_issues"])))
    for case in reference.get("new_review_unresolved", []):
        if "uncertainty" in case["decision"] or "unresolved" in case["decision"]:
            held["new_000010"].update(case["rows"])
            notes["new_000010"].append("Rows %s: %s" % (", ".join(map(str, case["rows"])), case["reason"]))
    records = sorted(source["records"], key=lambda record: (record["split"] != "train", record["id"]))
    if len(records) != 157 or len({record["id"] for record in records}) != 157:
        raise ValueError("all 157 distinct frozen photos are required")
    items = []
    for record in records:
        iid, split = record["id"], record["split"]
        image_path = root / "data" / record["image"]
        if sha(image_path) != record["image_sha256"]:
            raise ValueError(f"original photo hash differs: {iid}")
        if split == "val":
            meta = val_files[iid]
            label_path = reference_dir / meta["label_path"]
            expected = meta["label_sha256"]
        elif iid in changes:
            meta = changes[iid]
            label_path = candidate / meta["candidate_label"]
            expected = meta["candidate_label_sha256"]
        else:
            label_path = root / "data" / record["label"]
            expected = record["label_sha256"]
        if sha(label_path) != expected:
            raise ValueError(f"label hash differs: {iid}")
        objects = parse_pose_labels(label_path.read_text(), *record["original_size"], iid)
        if not held[iid] <= set(range(1, len(objects) + 1)):
            raise ValueError(f"review flag targets absent row: {iid}")
        items.append({"record": record, "image_path": image_path, "label_path": label_path,
                      "objects": objects, "label_sha256": expected})
    focuses = status.get("focus_pages", [])
    first_photo = 5 + len(focuses)
    page_numbers = {item["record"]["id"]: first_photo + index for index, item in enumerate(items)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(args.output), pagesize=(PAGE_W, PAGE_H), pageCompression=1, invariant=1)
    c.setTitle("Cassette label atlas - all 157 working photos - 14 November 2026")
    c.setAuthor("Siuuuers/dl - assisted annotation review")
    c.setSubject("Current training candidate and separate validation development reference; not predictions or a certified release")
    c.bookmarkPage("cover")
    c.addOutlineEntry("Read first", "cover", 0)
    heading(c, "Cassette label atlas", "All 157 photos in the frozen working dataset", 1, "REVIEW COPY")
    c.setFillColor(PAPER)
    c.roundRect(MARGIN, 505, PAGE_W - 2 * MARGIN, 230, 10, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("AtlasBold", 50)
    c.drawString(58, 660, "157 photos. Every current label.")
    wrapped(c, "137 training photos + 20 validation photos. Each photo has its own full page, with zoomable boxes, two lower-front landmarks, current row numbers, and visible review flags.", 60, 615, PAGE_W - 120, 19, 28)
    y = wrapped(c, "This is a review atlas of saved annotations. It is not model output, a confidence display, or a claim that every label is correct. Amber marks known unresolved rows. Unflagged rows are not automatically certified.", 60, 466, PAGE_W - 120, 16, 23)
    y = wrapped(c, "The next GPU run remains pending. Training may proceed only after the remaining material questions are resolved and the training handoff verifies its exact inputs. Changing label files does not change the existing model weights.", 60, y - 20, PAGE_W - 120, 14, 21)
    y = wrapped(c, "Training candidate: " + train["candidate_id"] + "\nValidation source: " + reference["reference_id"] + " (separate development reference).", 60, y - 20, PAGE_W - 120, 12, 18)
    wrapped(c, "Scope: the 157 working-set photos, including every empty-label photo. Excluded final-test, gap and unaccepted upstream images are outside this frozen inventory. Photos are native-resolution JPEG review copies; the original PNGs remain authoritative for individual-pixel decisions.", 60, 168, PAGE_W - 120, 11, 17, MUTED)
    c.showPage()
    c.bookmarkPage("guide")
    c.addOutlineEntry("How to read the labels", "guide", 0)
    heading(c, "How to read the labels", "Use image ID + current row number when reporting a correction", 2, "GUIDE")
    legend = [(CYAN, "Solid cyan box", "No recorded review question for this row. Zero, one or two actual corners may be locatable; corner count does not determine box colour."),
              (AMBER, "Dashed amber box", "A recorded identity, extent or landmark question remains for this row. Some fields may already be corrected."),
              (GREEN, "Green point", "Saved lower-left front corner, ordered by image x. This point is marked locatable (v=2) in the label."),
              (MAGENTA, "Pink point", "Saved lower-right front corner, ordered by image x. Missing or unreliable corners (v=0) are not drawn."),
              (GREEN, "Green connecting line", "Both saved corners have v=2. No confidence threshold is involved in this ground-truth preview.")]
    y = 728
    for color, title, description in legend:
        c.setFillColor(color)
        c.roundRect(48, y - 15, 19, 19, 3, fill=1, stroke=0)
        c.setFont("AtlasBold", 15)
        c.setFillColor(INK)
        c.drawString(82, y, title)
        y = wrapped(c, description, 82, y - 24, PAGE_W - 140, 12, 18) - 25
    y = wrapped(c, "Target: one individual cassette front, with the two actual corners along its lower edge. A visible partial front is still an object even if neither actual corner can be located. A roof, rack edge, gripper edge or an inferred hidden corner is not a substitute.", 48, y, PAGE_W - 96, 13, 20)
    y = wrapped(c, "Known physical constraints help identify surface ownership and separate instances. Equal physical size does not imply equal pixel size under perspective. One box may span an occluder when separate visible patches belong to the same front; its outer bounds still follow observed material. A confirmed cassette count does not make its hidden actual corners visible.", 48, y - 12, PAGE_W - 96, 13, 20)
    wrapped(c, "Row IDs shown here are current output TXT row numbers. Historical IDs are preserved in the candidate row map; deletions can shift displayed numbers. Photo notes summarize recorded questions. Complete per-field reasons and reproducible provenance are in the review record below.", 48, 115, PAGE_W - 96, 11, 17, MUTED)
    c.setFont("Atlas", 10)
    c.drawString(48, 60, "Review record: github.com/Siuuuers/dl/pull/63")
    c.linkURL("https://github.com/Siuuuers/dl/pull/63", (48, 54, 480, 74), relative=0, thickness=0)
    c.showPage()
    for part in range(2):
        c.bookmarkPage(f"index{part}")
        c.addOutlineEntry(f"Photo index {part + 1}", f"index{part}", 0)
        heading(c, "Photo index", "Click an image name to open its page. Asterisk = at least one recorded review question.", part + 3, f"{part + 1} / 2")
        subset = items[part * 84:(part + 1) * 84]
        for index, item in enumerate(subset):
            col, row = divmod(index, 28)
            x, y = 42 + col * 379, 735 - row * 23
            iid, split = item["record"]["id"], item["record"]["split"]
            c.setFillColor(INK)
            c.setFont("Atlas", 12)
            c.drawString(x, y, iid + (" *" if notes[iid] else ""))
            c.setFont("Atlas", 10)
            c.setFillColor(MUTED)
            c.drawString(x + 146, y, "TRAIN" if split == "train" else "VAL / DEV")
            c.drawRightString(x + 337, y, str(page_numbers[iid]))
            c.linkRect("", iid, (x, y - 4, x + 345, y + 14), relative=0, thickness=0)
        c.showPage()
    by_id = {item["record"]["id"]: item for item in items}
    for offset, focus in enumerate(focuses, 5):
        iid = focus["image_id"]
        item = by_id[iid]
        focus_key = f"focus-{iid}-{offset}"
        c.bookmarkPage(focus_key)
        c.addOutlineEntry(focus["title"] + " | " + iid, focus_key, 0)
        heading(c, focus["title"], iid + " | Original pixels beside the current labels", offset,
                focus.get("section", "FOCUSED QUESTION"))
        crop = focus["crop_xyxy_px"]
        iw, ih = crop[2] - crop[0], crop[3] - crop[1]
        raw = Image.open(item["image_path"]).convert("RGB").crop(crop)
        panel_w = (PAGE_W - 90) / 2
        scale = min(panel_w / iw, 430 / ih)
        w, h = iw * scale, ih * scale
        bottom = 725 - h
        for panel in range(2):
            left = 30 + panel * (panel_w + 30)
            c.setFont("AtlasBold", 13)
            c.setFillColor(INK)
            c.drawString(left, 744, "Original PNG crop" if panel == 0 else "Current saved labels")
            c.drawImage(ImageReader(raw), left, bottom, w, h)
            if panel:
                annotation(c, item["objects"], left, bottom, w, h, item["record"]["original_size"], held[iid], crop, focus.get("current_rows"))
        y = wrapped(c, focus["observation"], 35, bottom - 32, PAGE_W - 70, 15, 22)
        wrapped(c, focus["question"], 35, y - 18, PAGE_W - 70, 17, 25)
        c.setFont("Atlas", 11)
        c.setFillColor(MUTED)
        c.drawString(35, 76, f"Full photo: page {page_numbers[iid]}. Crop bounds in original pixels: {crop}.")
        c.linkRect("", iid, (35, 67, 370, 93), relative=0, thickness=0)
        c.showPage()
    audit_pages = []
    last_split = None
    for item in items:
        record, objects = item["record"], item["objects"]
        iid, split, page = record["id"], record["split"], page_numbers[record["id"]]
        c.bookmarkPage(iid)
        if split != last_split:
            split_key = "photo-section-" + split
            c.bookmarkPage(split_key)
            c.addOutlineEntry("Training photos" if split == "train" else "Validation / development photos", split_key, 0)
            last_split = split
        c.addOutlineEntry(iid, iid, 1)
        visible = sum(point[2] == 2 for obj in objects for point in obj["keypoints"])
        summary = f"{len(objects)} objects | {visible} locatable corners | {2 * len(objects) - visible} missing corners | Original 1280 x 720"
        heading(c, iid, summary, page, "TRAINING CANDIDATE" if split == "train" else "VALIDATION / DEVELOPMENT")
        image_w = PAGE_W - 2 * MARGIN
        image_h = image_w * record["original_size"][1] / record["original_size"][0]
        bottom = PAGE_H - 72 - image_h
        c.drawImage(jpeg_reader(item["image_path"]), MARGIN, bottom, image_w, image_h)
        annotation(c, objects, MARGIN, bottom, image_w, image_h, record["original_size"], held[iid])
        c.setFont("AtlasBold", 10)
        c.setFillColor(INK)
        row_text = ", ".join(map(str, sorted(held[iid]))) or "none recorded"
        c.drawString(MARGIN, bottom - 18, "Review rows: " + row_text)
        c.setFont("Atlas", 10)
        c.setFillColor(MUTED)
        c.drawRightString(PAGE_W - MARGIN, bottom - 18, "Index / bookmarks")
        c.linkRect("", "index0", (PAGE_W - 160, bottom - 24, PAGE_W - MARGIN, bottom - 7), relative=0, thickness=0)
        if not objects:
            text = "Empty-label photo. No cassette instances are annotated in this saved label file. Included deliberately in the complete review inventory."
        elif notes[iid]:
            text = " ".join(notes[iid])
        else:
            text = "No unresolved case is recorded for this image in the loaded review ledgers. This status is not independent certification of every box or landmark."
        excerpt, abbreviated = note_excerpt(text, PAGE_W - 2 * MARGIN)
        end_y = wrapped(c, excerpt, MARGIN, bottom - 37, PAGE_W - 2 * MARGIN, 9, 12, MUTED)
        if end_y < 45:
            raise ValueError(f"review notes overflow footer: {iid}")
        audit_pages.append({"page": page, "image_id": iid, "split": split,
                            "source_image_path": record["image"], "source_image_sha256": sha(item["image_path"]),
                            "label_path": item["label_path"].relative_to(root).as_posix(),
                            "label_sha256": item["label_sha256"], "objects": len(objects),
                            "visible_endpoints": visible, "missing_endpoints": 2 * len(objects) - visible,
                            "held_current_rows": sorted(held[iid]), "original_size": record["original_size"],
                            "full_review_notes": text, "footer_abbreviated": abbreviated})
        c.showPage()
    c.save()
    audit = {"schema_version": 1, "kind": "all_photo_label_atlas", "pdf_filename": args.output.name,
             "pdf_sha256": sha(args.output), "pdf_bytes": args.output.stat().st_size,
             "page_count": first_photo + len(items) - 1, "photo_pages": len(items),
             "training_candidate": {"path": (candidate / "manifest.json").relative_to(root).as_posix(), "sha256": sha(candidate / "manifest.json")},
             "validation_reference": {"path": (reference_dir / "manifest.json").relative_to(root).as_posix(), "sha256": sha(reference_dir / "manifest.json")},
             "review_status_sha256": sha(args.status), "generator_sha256": sha(Path(__file__)),
             "rendering": {"photo_resize": False, "photo_jpeg_quality": 96, "photo_jpeg_subsampling": 0,
                           "annotations": "PDF vectors", "focus_crops": "lossless original PNG pixels",
                           "model_inference": False, "lines_require_both_saved_v2": True},
             "pages": audit_pages}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in ("pdf_filename", "pdf_sha256", "pdf_bytes", "page_count", "photo_pages")}, indent=2))


if __name__ == "__main__":
    main()
