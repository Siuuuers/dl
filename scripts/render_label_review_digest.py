#!/usr/bin/env python3
"""Make a focused review PDF from a verified full label atlas and its audit."""

import argparse
from collections import Counter, defaultdict
from copy import copy
from datetime import date
import hashlib
from io import BytesIO
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from render_label_atlas import PAGE_W, PAGE_H, INK, MUTED, PAPER, wrapped


MARGIN = 42
TRAIN_NOTES = {
    "new_000005": "Top-border remnant: extent and actual corner.",
    "new_000034": "Top-border extent and left-corner identity; two case groups.",
    "new_000049": "Top strip: preceding front or next-front molding?",
    "new_000094": "Bottom strip: next front or the lip of row 22?",
    "new_000115": "Top strip: preceding front or next-front molding?",
    "new_000117": "Top strip: preceding front or next-front molding?",
    "new_000121": "Blurred top strip: separate front or molding?",
    "new_000128": "Two top strips: separate preceding fronts or molding?",
}
DEVELOPMENT_NOTES = {
    "new_000007": "Two top-edge strips: separate fronts or adjacent lips?",
    "new_000008": "Bottom-clipped surface: front extent versus roof/rack.",
    "new_000009": "Top-clipped front: exact extent and soft junction.",
    "new_000010": "Bottom-clipped sliver: exact front-box extent.",
    "new_000011": "Top strip: separate front or roof/lip of row 16?",
    "new_000057": "Top strip: distinct-front identity and extent.",
    "new_000059": "Top strip: separate front or upper material of row 2?",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def bound(root, item):
    path = (root / item["path"]).resolve()
    require(path.is_relative_to(root), "source path escapes repository")
    require(sha(path) == item["sha256"], "source hash differs: " + item["path"])
    return path


def image_hashes(page):
    """Compare decoded pixel payloads; copying must not re-encode review crops."""
    hashes = []
    def walk(resources):
        objects = resources.get("/XObject", {})
        objects = objects.get_object() if hasattr(objects, "get_object") else objects
        for value in objects.values():
            obj = value.get_object()
            if obj.get("/Subtype") == "/Image":
                hashes.append(hashlib.sha256(obj.get_data()).hexdigest())
            elif obj.get("/Subtype") == "/Form":
                walk(obj.get("/Resources", {}))
    walk(page["/Resources"])
    return Counter(hashes)


def register_fonts():
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    for name, filename in (("Atlas", "DejaVuSans.ttf"), ("AtlasBold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))


def footer(c, page, review_date):
    c.setStrokeColor(colors.HexColor("#dce3e9"))
    c.line(MARGIN, 34, PAGE_W - MARGIN, 34)
    c.setFillColor(MUTED)
    c.setFont("Atlas", 9)
    c.drawString(MARGIN, 19, "Cassette label corrections  |  " + review_date + "  |  Review labels; no model predictions")
    c.drawRightString(PAGE_W - MARGIN, 19, str(page))


def heading(c, title, subtitle, page, review_date):
    c.setFillColor(colors.white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("AtlasBold", 31)
    c.drawString(MARGIN, PAGE_H - 57, title)
    wrapped(c, subtitle, MARGIN, PAGE_H - 83, PAGE_W - 2 * MARGIN, 12, 17, MUTED)
    footer(c, page, review_date)


def table_row(c, image_id, rows, atlas_page, note, x, y, width, height):
    c.setStrokeColor(colors.HexColor("#e1e7ed"))
    c.line(x, y - height + 4, x + width, y - height + 4)
    c.setFillColor(INK)
    c.setFont("AtlasBold", 11)
    c.drawString(x, y - 18, image_id)
    c.setFont("Atlas", 11)
    c.drawString(x + 117, y - 18, ", ".join(map(str, rows)))
    c.drawRightString(x + 202, y - 18, str(atlas_page))
    end_y = wrapped(c, note, x + 218, y - 17, width - 218, 10.5, 14, MUTED)
    require(end_y >= y - height + 3, "hold-table note exceeds row: " + image_id)


def intro_pages(review_date, train, reference, focuses, pages, training_holds, development_holds, train_groups):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=(PAGE_W, PAGE_H), pageCompression=1, invariant=1)
    total_replaced = train["operation_counts"].get("replace", 0) + reference["operation_counts"].get("replace", 0)
    total_added = train["operation_counts"].get("add", 0) + reference["operation_counts"].get("add", 0)
    heading(c, "Cassette label corrections",
            review_date + "  /  Focused review of saved annotations  /  " + str(len(focuses) + 2) + " pages", 1, review_date)
    y = PAGE_H - 126
    c.setFillColor(INK)
    c.setFont("AtlasBold", 27)
    c.drawString(MARGIN, y, f"{total_replaced} rows refined. {total_added} visible fronts added.")
    wrapped(c, "Original crops and current labels appear side by side on the following pages. "
            "This digest covers selected corrections; the separate full atlas contains all 157 photos.",
            MARGIN, y - 27, PAGE_W - 2 * MARGIN, 12.5, 18)
    card_y, card_h, gap = 558, 82, 20
    card_w = (PAGE_W - 2 * MARGIN - gap) / 2
    cards = [("TRAINING REVIEW", train["expected_counts"]["train"], train["operation_counts"]),
             ("SEPARATE DEVELOPMENT REFERENCE", reference["counts"], reference["operation_counts"])]
    for index, (title, counts, operations) in enumerate(cards):
        x = MARGIN + index * (card_w + gap)
        c.setFillColor(PAPER)
        c.roundRect(x, card_y, card_w, card_h, 7, fill=1, stroke=0)
        c.setFont("AtlasBold", 10)
        c.setFillColor(MUTED)
        c.drawString(x + 16, card_y + 61, title)
        c.setFont("AtlasBold", 21)
        c.setFillColor(INK)
        c.drawString(x + 16, card_y + 33, f"{counts['objects']:,} fronts in {counts['images']} photos")
        c.setFont("Atlas", 10.5)
        c.drawString(x + 16, card_y + 14,
                     f"This pass: {operations.get('replace', 0)} replaced rows + {operations.get('add', 0)} additions. "
                     "Historical versions remain reproducible.")
    c.setFont("AtlasBold", 17)
    c.drawString(MARGIN, 527, "Choose a correction")
    c.setFont("Atlas", 10)
    c.setFillColor(MUTED)
    c.drawRightString(PAGE_W - MARGIN, 527, "Full-atlas numbers point to the complete photo, not this digest.")
    x = [MARGIN, MARGIN + 58, MARGIN + 210, MARGIN + 344, PAGE_W - MARGIN]
    c.setFillColor(INK)
    c.setFont("AtlasBold", 10)
    for xx, text in zip(x[:4], ("HERE", "IMAGE", "LABEL SET", "REVIEW FOCUS")):
        c.drawString(xx, 502, text)
    c.drawRightString(x[4], 502, "FULL ATLAS")
    c.setStrokeColor(colors.HexColor("#ccd6df"))
    c.line(MARGIN, 493, PAGE_W - MARGIN, 493)
    for i, focus in enumerate(focuses):
        yy = 475 - i * 24
        c.setFillColor(INK)
        c.setFont("AtlasBold", 11)
        c.drawString(x[0], yy, str(i + 3))
        c.drawString(x[1], yy, focus["image_id"])
        c.setFont("Atlas", 10.5)
        c.setFillColor(MUTED)
        c.drawString(x[2], yy, "Training" if pages[focus["image_id"]]["split"] == "train" else "Development")
        c.setFillColor(INK)
        title = focus["title"]
        require(pdfmetrics.stringWidth(title, "Atlas", 10.5) < x[4] - x[3] - 70, "focus title is too wide")
        c.drawString(x[3], yy, title)
        c.drawRightString(x[4], yy, str(pages[focus["image_id"]]["page"]))
    wrapped(c, "Cyan means no open recorded hold. Amber marks an unresolved row. "
            "Green and magenta dots are the observed left and right lower-front corners; a hidden corner has no dot.",
            MARGIN, 131, PAGE_W - 2 * MARGIN, 11, 16, MUTED)
    wrapped(c, "A box follows visible front-face material, including separated fragments of one front. "
            "It does not automatically cover the roof, the whole cassette body, or a guessed hidden extent.",
            MARGIN, 88, PAGE_W - 2 * MARGIN, 11, 16)
    c.showPage()

    heading(c, "Remaining boundary questions",
            "Current row numbers and full-atlas photo pages. These recorded holds remain separate from the resolved corrections in this digest.",
            2, review_date)
    wrapped(c, "Most remaining questions involve tiny fragments at an image edge: whether the pixels show a separate front, "
            "a molded lip, or other material. A known cassette count does not reveal an occluded corner or establish a hidden front.",
            MARGIN, PAGE_H - 125, PAGE_W - 2 * MARGIN, 13, 19)
    gap, half = 34, (PAGE_W - 2 * MARGIN - 34) / 2
    sections = [(MARGIN, "TRAINING", f"{train_groups} case groups across {len(training_holds)} photos", training_holds, TRAIN_NOTES),
                (MARGIN + half + gap, "DEVELOPMENT", f"{sum(len(v) for v in development_holds.values())} held rows across {len(development_holds)} photos", development_holds, DEVELOPMENT_NOTES)]
    for left, name, count, holds, descriptions in sections:
        c.setFillColor(INK)
        c.setFont("AtlasBold", 16)
        c.drawString(left, 636, name)
        c.setFont("Atlas", 12)
        c.setFillColor(MUTED)
        c.drawString(left, 614, count)
        c.setFillColor(INK)
        c.setFont("AtlasBold", 9)
        for xx, text in ((left, "IMAGE"), (left + 117, "ROWS"), (left + 164, "ATLAS"), (left + 218, "BOUNDARY TO REVIEW")):
            c.drawString(xx, 584, text)
        c.setStrokeColor(colors.HexColor("#ccd6df"))
        c.line(left, 575, left + half, 575)
        for n, (iid, rows) in enumerate(sorted(holds.items())):
            table_row(c, iid, rows, pages[iid]["page"], descriptions.get(iid, "See the full-atlas note for the recorded uncertainty."),
                      left, 570 - n * 48, half, 48)
    c.setFillColor(PAPER)
    c.roundRect(MARGIN, 84, PAGE_W - 2 * MARGIN, 93, 7, fill=1, stroke=0)
    wrapped(c, "How to read the status", MARGIN + 16, 152, PAGE_W - 2 * MARGIN - 32, 14, 18)
    wrapped(c, "Cyan is not a confidence score or certification. The training ledger counts two separate questions for row 15 in "
            "new_000034, so its nine case groups are not nine different photos. The development reference counts held rows. "
            "These units must not be compared as an accuracy score.", MARGIN + 16, 128, PAGE_W - 2 * MARGIN - 32, 11, 16, MUTED)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--digest-audit", type=Path, required=True)
    args = parser.parse_args()
    root, atlas = args.root.resolve(), args.atlas.resolve()
    require(args.output.resolve() != atlas, "digest must not replace the full atlas")
    audit = load(args.audit)
    require(sha(atlas) == audit["pdf_sha256"] and atlas.stat().st_size == audit["pdf_bytes"],
            "full atlas differs from its completed audit")
    require(audit["kind"] == "all_photo_label_atlas" and audit["photo_pages"] == 157,
            "expected the complete working-set atlas")
    source = load(bound(root, audit["source_manifest"]))
    train_path = bound(root, audit["training_candidate"])
    reference_path = bound(root, audit["validation_reference"])
    train, reference = load(train_path), load(reference_path)
    status = load(bound(root, audit["review_status"]))
    require(sha(root / "scripts/render_label_atlas.py") == audit["generator_sha256"], "atlas generator hash differs")
    require(train["review_status"] == audit["review_status"] and train["development_reference"] == audit["validation_reference"],
            "atlas inputs are not the candidate's pinned status and development reference")
    for manifest in (train, reference):
        bound(root, manifest["review"])
        for item in manifest.get("review_sources", []):
            bound(root, item)
    bound(root, train["annotation_policy"])
    bound(root, train["parent"])
    bound(root, reference["parent_reference"])
    records = {record["id"]: record for record in source["records"]}
    pages = {item["image_id"]: item for item in audit["pages"]}
    require(len(pages) == len(audit["pages"]) == 157 and set(pages) == set(records), "photo inventory differs")
    for iid, item in pages.items():
        record = records[iid]
        require(item["split"] == record["split"] and item["source_image_path"] == record["image"]
                and item["source_image_sha256"] == record["image_sha256"], "photo provenance differs: " + iid)
        require(sha(root / "data" / item["source_image_path"]) == item["source_image_sha256"], "original image differs: " + iid)
        bound(root, {"path": item["label_path"], "sha256": item["label_sha256"]})
    focuses = audit["focus_pages"]
    require(focuses == [{"page": n, **focus} for n, focus in enumerate(status["focus_pages"], 5)], "focus list differs from review status")
    require(len({focus["page"] for focus in focuses}) == len(focuses), "duplicate focus source page")
    training_holds, development_holds = defaultdict(set), defaultdict(set)
    pending = [case for case in status["cases"] if case["status"] != "resolved"]
    for case in pending:
        training_holds[case["image_id"]].update(case["held_current_rows"])
    for case in reference["remaining_current_rows"]:
        development_holds[case["image_id"]].add(case["row"])
    training_holds = {key: sorted(value) for key, value in training_holds.items()}
    development_holds = {key: sorted(value) for key, value in development_holds.items()}
    for holds in (training_holds, development_holds):
        for iid, rows in holds.items():
            require(rows == pages[iid]["held_current_rows"], "held rows differ from full atlas: " + iid)
    review_date = date.fromisoformat(train["review_date"]).strftime("%d %B %Y").lstrip("0")
    register_fonts()
    intro = PdfReader(intro_pages(review_date, train, reference, focuses, pages, training_holds, development_holds, len(pending)))
    full = PdfReader(atlas)
    require(len(full.pages) == audit["page_count"], "full atlas page count differs")
    writer = PdfWriter()
    for page in intro.pages:
        writer.add_page(page)
    writer.add_outline_item("Correction index", 0)
    writer.add_outline_item("Remaining boundary questions", 1)
    crossmap = []
    for short_page, focus in enumerate(focuses, 3):
        old_page, iid = focus["page"], focus["image_id"]
        page = copy(full.pages[old_page - 1])
        require(abs(float(page.mediabox.width) - PAGE_W) < 0.01 and abs(float(page.mediabox.height) - PAGE_H) < 0.01,
                "focus page size differs from A3 landscape")
        # Full-photo destinations are outside this digest; do not clone their page trees.
        page.pop("/Annots", None)
        layer = BytesIO()
        c = canvas.Canvas(layer, pagesize=(PAGE_W, PAGE_H), pageCompression=1, invariant=1)
        c.setFillColor(colors.white)
        c.rect(0, 0, PAGE_W, 33, fill=1, stroke=0)
        footer(c, short_page, review_date)
        c.setFillColor(colors.white)
        c.rect(30, 69, PAGE_W - 60, 19, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("Atlas", 10)
        c.drawString(35, 76, f"Full atlas photo: page {pages[iid]['page']}. Full atlas focus: page {old_page}. "
                     f"Crop bounds in original pixels: {focus['crop_xyxy_px']}.")
        c.save()
        layer.seek(0)
        page.merge_page(PdfReader(layer).pages[0])
        writer.add_page(page)
        writer.add_outline_item(f"{iid}: {focus['title']}", short_page - 1)
        crossmap.append({"digest_page": short_page, "atlas_focus_page": old_page,
                         "atlas_photo_page": pages[iid]["page"], "image_id": iid,
                         "title": focus["title"], "current_rows": focus.get("current_rows"),
                         "crop_xyxy_px": focus["crop_xyxy_px"]})
    writer.add_metadata({"/Title": "Cassette label corrections - " + review_date,
                         "/Author": "Siuuuers/dl - assisted annotation review",
                         "/Subject": "Selected original crops and reviewed labels; separate current holds; no model predictions"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        writer.write(stream)
    digest = PdfReader(args.output)
    require(len(digest.pages) == len(focuses) + 2, "digest page count differs")
    require("Cassette label corrections" in digest.pages[0].extract_text(), "digest title missing")
    for entry in crossmap:
        original_images = image_hashes(full.pages[entry["atlas_focus_page"] - 1])
        require(original_images and original_images == image_hashes(digest.pages[entry["digest_page"] - 1]),
                "copied focus image pixels differ: " + entry["image_id"])
        entry["decoded_image_sha256"] = sorted(original_images.elements())
    report = {
        "schema_version": 1, "kind": "focused_label_review_digest",
        "pdf_filename": args.output.name, "pdf_sha256": sha(args.output),
        "pdf_bytes": args.output.stat().st_size, "page_count": len(digest.pages),
        "intro_pages": 2, "focus_pages": len(focuses),
        "source_atlas": {"filename": atlas.name, "sha256": audit["pdf_sha256"], "page_count": len(full.pages)},
        "source_atlas_audit": {"path": args.audit.resolve().relative_to(root).as_posix(), "sha256": sha(args.audit)},
        "training_candidate": audit["training_candidate"], "development_reference": audit["validation_reference"],
        "review_status": audit["review_status"], "generator_sha256": sha(Path(__file__)),
        "source_hashes_verified_before_copying": True, "verified_photo_source_count": len(pages),
        "page_crossmap": crossmap,
        "remaining_holds": {
            "training_case_groups": len(pending), "training_photos": len(training_holds),
            "training_rows_by_photo": training_holds,
            "development_rows": sum(len(value) for value in development_holds.values()),
            "development_photos": len(development_holds), "development_rows_by_photo": development_holds},
        "rendering": {"intro": "ReportLab vector text", "focus": "Copied verified PDF page content; no image re-encoding",
                      "overlays": "Digest footer numbers and explicit full-atlas photo cross-references",
                      "out_of_digest_links": "Removed full-photo link annotations; printed atlas cross-references retained",
                      "copied_image_payloads_verified_identical": True,
                      "bookmarks": len(focuses) + 2},
        "limits": ["Selected corrections only; use the complete atlas for all 157 photos.",
                   "Cyan is not certification. Remaining holds are listed on digest page 2.",
                   "No model training, evaluation or promotion is claimed by this document."]}
    args.digest_audit.parent.mkdir(parents=True, exist_ok=True)
    args.digest_audit.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("pdf_filename", "pdf_sha256", "pdf_bytes", "page_count", "focus_pages")}, indent=2))


if __name__ == "__main__":
    main()
