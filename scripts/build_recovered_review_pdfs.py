#!/usr/bin/env python3
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
from io import BytesIO
from pathlib import Path
import sys

import fitz
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject, TextStringObject
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


INK = colors.HexColor('#182630')
MUTED = colors.HexColor('#536775')
TEAL = colors.HexColor('#087f8c')
LIGHT = colors.HexColor('#edf5f6')
SOURCE_ATLAS_SHA = 'e5a34c8f8670b2b1ca67c770fca5ad21782942fb212f04384e09ac154b4c4a6f'
SOURCE_CHANGES_SHA = '53d913f6e62bf8ca5eb573a088f4f9c2857dc0744e0ede1b5c949ffe50297ca3'
SOURCE_PNG_SHA = '5b13f5c4d0c5ab94fd5954000a299cd7dc2c1ec486cc6d1d13a623f8fcc0094c'
AFTER_LINE = '0 0.40429688 0.77361111 0.05078125 0.03888889 0.38050781 0.78863889 2 0.00000000 0.00000000 0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrapped(c, text, x, y, width, size=11, leading=16, color=INK, font='Review'):
    c.setFont(font, size)
    c.setFillColor(color)
    for para in text.split('\n'):
        line = ''
        for word in para.split():
            trial = (line + ' ' + word).strip()
            if line and pdfmetrics.stringWidth(trial, font, size) > width:
                c.drawString(x, y, line)
                y -= leading
                line = word
            else:
                line = trial
        if line:
            c.drawString(x, y, line)
        y -= leading
    return y


def header(c, width, height, title, subtitle, page, tag='RECOVERED REVIEW'):
    margin = 28 if width < 700 else 30
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont('ReviewBold', 22 if width < 700 else 25)
    c.drawString(margin, height - 41, title)
    c.setFont('Review', 9 if width < 700 else 11)
    c.setFillColor(MUTED)
    c.drawString(margin, height - 61, subtitle)
    c.setFont('Review', 8.5)
    c.drawRightString(width - margin, height - 20, tag)
    c.setStrokeColor(colors.HexColor('#d8e2e9'))
    c.line(margin, 30, width - margin, 30)
    c.setFont('Review', 7.5 if width < 700 else 9)
    c.drawString(margin, 17, '18 November 2026 / Recovered review / Saved annotations; no model predictions')
    c.drawRightString(width - margin, 17, str(page))


def build_page(size, callback):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=size, invariant=1, pageCompression=1)
    callback(c, *size)
    c.showPage()
    c.save()
    buffer.seek(0)
    return PdfReader(buffer)


def cover_changes(c, w, h):
    header(c, w, h, 'Cassette labels: recovered review', 'Round 3 recovered, with one additional box refinement', 1)
    y = wrapped(c, '889 rows revised', 28, h - 120, w - 56, 28, 34, font='ReviewBold')
    y = wrapped(c, '12 deleted / 0 added', 28, y - 5, w - 56, 17, 25)
    y = wrapped(c, '59 training photos and 5 development photos changed. The inventory stays at 3,789 training fronts and 662 development fronts across all 157 photos.', 28, y - 22, w - 56, 12, 18)
    y = wrapped(c, 'What is new here', 28, y - 27, w - 56, 15, 22, font='ReviewBold')
    y = wrapped(c, 'The saved Round 3 PDFs were recovered. Their vector annotations were converted back to 11-field pose labels and checked against the displayed geometry. Original authoring bytes and the missing original ledger are not claimed recovered.', 28, y - 3, w - 56, 11, 17)
    y = wrapped(c, 'A fresh original-image check refines new_000101 row 52 from [481,543,550,574] to [485,543,550,571]. Its left corner and missing right corner are unchanged. Page 50 shows all three changed rows in this photo; page 75 gives the enlarged follow-up.', 28, y - 12, w - 56, 11, 17)
    y = wrapped(c, 'How to use this copy', 28, y - 24, w - 56, 15, 22, font='ReviewBold')
    y = wrapped(c, 'Existing review pages and page numbers are preserved. The index on pages 3-4 remains clickable. Page 76 records recovery evidence and its limits. Only new_000007 rows 7 and 14 remain recorded as amber (page 6).', 28, y - 3, w - 56, 11, 17)
    y = wrapped(c, 'Cyan means no recorded open hold, not a correctness certificate. Green and pink dots mark independently locatable actual lower-front corners. The two-point task and canonical missing corners are preserved.', 28, y - 12, w - 56, 11, 17)
    y = wrapped(c, 'Review only: the v2 training inputs and evaluation references are unchanged. No new training release, model promotion, or application-readiness claim follows from this recovery. The completed PR67 run must be judged on its own pinned inputs.', 28, y - 12, w - 56, 10.5, 16)
    if y < 47:
        raise ValueError('changes cover overflow')


def cover_atlas(c, w, h):
    header(c, w, h, 'Cassette label atlas', 'Recovered Round 3 review, including the new_000101 row 52 refinement', 1)
    y = wrapped(c, '157 photos. Every reviewed label.', 30, h - 125, w - 60, 30, 40, font='ReviewBold')
    y = wrapped(c, '137 training photos / 3,789 fronts     20 development photos / 662 fronts', 30, y - 12, w - 60, 20, 28)
    y = wrapped(c, 'All photo pages, empty scenes, current row numbers, and existing navigation remain in place. New_000101 is updated on page 79. The recovery appendix is page 166.', 30, y - 26, w - 60, 15, 23)
    y = wrapped(c, 'This is a new review copy', 30, y - 30, w - 60, 19, 27, font='ReviewBold')
    y = wrapped(c, 'The Round 3 vector geometry was recovered to usable 11-field pose labels. This does not recover the original authoring bytes or the missing original ledger. One additional, separately recorded source-image review trims new_000101 row 52 to [485,543,550,571], preserving its existing corner tokens.', 30, y - 4, w - 60, 14, 22)
    y = wrapped(c, 'The other original review pages remain as documented history. Their Round 3 footers refer to that source review. Their saved geometry agrees with the recovered labels within the measured rendering precision; it is not a claim of physical or single-pixel accuracy.', 30, y - 18, w - 60, 14, 22)
    y = wrapped(c, 'Review state and model scope', 30, y - 30, w - 60, 19, 27, font='ReviewBold')
    y = wrapped(c, 'Only new_000007 rows 7 and 14 remain recorded as amber, on page 146. Cyan means no recorded hold. Boxes describe visible cassette fronts, including their front lips; the two points remain actual lower-front corners when locatable. Unseen corners stay missing.', 30, y - 4, w - 60, 14, 22)
    y = wrapped(c, 'The v2 experiment keeps its earlier training labels and predefined evaluation references. This recovered review is not a training release, model result, or deployment certificate. Review these PDFs before accepting the new annotation revision.', 30, y - 18, w - 60, 14, 22)
    y = wrapped(c, 'Image provenance: retained photo pages use native-resolution JPEG review copies. Page 79 uses the verified original PNG. Original PNGs remain the authority for pixel-level decisions; boxes and landmarks remain vector overlays.', 30, y - 25, w - 60, 11, 17, MUTED)
    if y < 47:
        raise ValueError('atlas cover overflow')


def appendix(c, w, h, page, atlas, receipt):
    margin = 28 if w < 700 else 30
    width = w - margin * 2
    header(c, w, h, 'Recovery provenance and limits', 'A new review revision, with the source PDFs preserved', page)
    size, leading = (10.5, 16) if w < 700 else (14, 22)
    y = h - 103
    paragraphs = [
        ('What was recovered', 'The saved Round 3 PDFs preserve all 157 photos and their vector annotations. Their geometry was converted to one class and two lower-front corners per row. Class, normalized bounds, canonical missing states, point order, and point containment were checked. Original label-file bytes and the original operation ledger remain unavailable.'),
        ('Measured agreement', 'Independent before/after panel comparisons covered 1,817 row appearances. The largest geometric disagreement was 0.000719 original-image pixels. The largest discrepancy when preserving matching parent tokens was 0.000696 pixels. These numbers describe PDF conversion consistency, not physical annotation accuracy or camera calibration.'),
        ('Fresh follow-up: new_000101 row 52', 'The original rectangle included rack/support material at the left and beneath the white front lip. The new nominal box is [485,543,550,571] pixels. The left corner remains approximately (487.05,567.82); the right remains missing. Reviewed edge intervals are x484-487 on the left and y569-572 below. They are manual uncertainty estimates, not calibrated confidence intervals.'),
        ('Inventory and remaining review', 'The resulting review has 889 replacements, 12 deletions, and no additions relative to the earlier parent: the recorded 888 Round 3 replacements plus the new row 52 refinement. Counts stay at 3,789 training fronts and 662 development fronts; all 16 empty photos remain. The only recorded amber rows are new_000007:7,14.'),
        ('Training and evaluation stay pinned', 'This work does not modify the v2 training candidate or either of its predefined evaluation references. No improved model, full-dataset certification, or application reliability is established by the annotation recovery. Physical acceptance and any later training selection remain separate decisions.'),
    ]
    for title, text in paragraphs:
        y = wrapped(c, title, margin, y, width, size + 1.5, leading + 2, font='ReviewBold')
        y = wrapped(c, text, margin, y - 2, width, size, leading)
        y -= 12 if w < 700 else 20
    hash_size = 7.3 if w < 700 else 9.5
    y = wrapped(c, 'Source PDF SHA-256', margin, y, width, size, leading, font='ReviewBold')
    for name, digest in [('Changes r3', SOURCE_CHANGES_SHA), ('Atlas r3', SOURCE_ATLAS_SHA)]:
        y = wrapped(c, name + ': ' + digest, margin, y, width, hash_size, hash_size + 4, MUTED)
    y = wrapped(c, 'Verified new101 PNG: ' + SOURCE_PNG_SHA, margin, y - 4, width, hash_size, hash_size + 4, MUTED)
    y = wrapped(c, 'PDF source: preserved r3 documents. Supporting records: r3-reconciliation.json, recovery manifest, and new101-row52-physical-review.json. Full review and conversion records are supplied with this revision.', margin, y - 7, width, hash_size, hash_size + 4, MUTED)
    if y < 43:
        raise ValueError('appendix overflow: ' + str(y))


def replace_page_contents(writer, index, generated):
    target = writer.pages[index]
    replacement = generated.pages[0]
    target[NameObject('/Contents')] = replacement.raw_get('/Contents').clone(writer)
    target[NameObject('/Resources')] = replacement['/Resources'].clone(writer)


def remove_old_credentials(writer):
    root = writer.root_object
    root.pop(NameObject('/AF'), None)
    names = root.get('/Names')
    if names:
        names = names.get_object()
        names.pop(NameObject('/EmbeddedFiles'), None)


def links_snapshot(path):
    d = fitz.open(path)
    result = []
    for i, page in enumerate(d):
        result.append([(link.get('kind'), link.get('page'), link.get('uri')) for link in page.get_links()])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=Path, default=Path.cwd(), help='Root containing dl/, recovered_pdfs/, audit_r3_originals/, and audit_r3_recovery/.')
    ap.add_argument('--current-labels', type=Path, help='Authoritative final review labels directory containing train/ and val/.')
    ap.add_argument('--current-manifest', type=Path, help='Review manifest to bind in the audit; defaults to manifest.json beside --current-labels. Does not change PDF geometry.')
    args = ap.parse_args()
    b = args.workspace
    out = b / 'output/pdf'
    out.mkdir(parents=True, exist_ok=True)
    work = b / 'audit_annotations/pdf_recovery_build'
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(b / 'dl/scripts'))
    import render_label_atlas as atlas_renderer
    import render_front_refinement_review as change_renderer
    from evaluate_pose import parse_pose_labels
    for name, filename in [('Review', 'DejaVuSans.ttf'), ('ReviewBold', 'DejaVuSans-Bold.ttf'), ('Atlas', 'DejaVuSans.ttf'), ('AtlasBold', 'DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, '/usr/share/fonts/truetype/dejavu/' + filename))
    sources = b / 'recovered_pdfs'
    source_changes = sources / 'cassette-front-changes-2026-11-18-r3.pdf'
    source_atlas = sources / 'cassette-label-atlas-2026-11-18-r3.pdf'
    png = b / 'audit_r3_originals/new_000101.png'
    for path, expected in [(source_changes, SOURCE_CHANGES_SHA), (source_atlas, SOURCE_ATLAS_SHA), (png, SOURCE_PNG_SHA)]:
        if sha(path) != expected:
            raise ValueError('source hash differs: ' + str(path))
    source_labels = b / 'audit_r3_recovery/recovered_final/labels/train/new_000101.txt'
    current_lines = source_labels.read_text().splitlines()
    if len(current_lines) != 53:
        raise ValueError('unexpected new101 row count')
    old_line52 = current_lines[51]
    current_lines[51] = AFTER_LINE
    current_text = '\n'.join(current_lines) + '\n'
    if args.current_labels:
        current_label_path = args.current_labels / 'train/new_000101.txt'
        final_lines = current_label_path.read_text().splitlines()
        if (len(final_lines) != 53 or final_lines[:51] != current_lines[:51]
                or final_lines[52:] != current_lines[52:]
                or final_lines[51].split()[0] != old_line52.split()[0]
                or final_lines[51].split()[5:] != old_line52.split()[5:]):
            raise ValueError('published follow-up labels change more than row52 box tokens')
        current_text = current_label_path.read_text()
    else:
        current_label_path = work / 'new_000101-followup.txt'
        current_label_path.write_text(current_text)
    current = parse_pose_labels(current_text, 1280, 720, str(current_label_path))
    if max(abs(a-b) for a,b in zip(current[51]['bbox_xyxy'],[485,543,550,571])) > 0.00002:
        raise ValueError('published row52 box differs from accepted nominal bounds')
    parent_path = b / 'dl/data/candidates/sequence-refined-fronts-2026-11-18-r2/labels/train/new_000101.txt'
    parent = parse_pose_labels(parent_path.read_text(), 1280, 720, str(parent_path))
    recovered = parse_pose_labels(source_labels.read_text(), 1280, 720, str(source_labels))
    if current[51]['keypoints'] != recovered[51]['keypoints']:
        raise ValueError('keypoint changed unexpectedly')
    current_manifest_path = args.current_manifest
    if current_manifest_path is None and args.current_labels:
        current_manifest_path = args.current_labels.parent / 'manifest.json'
    if current_manifest_path is not None and not current_manifest_path.is_file():
        raise ValueError('current review manifest is missing: ' + str(current_manifest_path))
    image = Image.open(png).convert('RGB')
    rc, ra = PdfReader(source_changes), PdfReader(source_atlas)
    size_c = tuple(float(x) for x in (rc.pages[0].mediabox.width, rc.pages[0].mediabox.height))
    size_a = tuple(float(x) for x in (ra.pages[0].mediabox.width, ra.pages[0].mediabox.height))

    def changes50(c, w, h):
        header(c, w, h, 'new_000101', 'Rows 28 and 48 from Round 3; row 52 from the fresh source-image review', 50, 'TRAIN / CHANGE')
        crop = [460, 210, 885, 592]
        change_renderer.image_panel(c, image, crop, parent, {28, 48, 52}, set(), h - 91, 'Before | parent labels')
        change_renderer.image_panel(c, image, crop, current, {28, 48, 52}, set(), h - 378, 'Current | recovered review plus row 52 refinement')
        text = ('Rows 28 and 48 retain the recovered Round 3 refinements. Row 52 now excludes the left rack margin and the support band below its visible front: [481,543,550,574] becomes [485,543,550,571]. Its left actual corner and missing right corner are preserved. Row numbers and instance counts are unchanged. See page 75 for the enlarged row 52 comparison.')
        y = wrapped(c, text, 28, 155, w - 56, 9, 13)
        wrapped(c, 'Verified original PNG; only these three rows are overlaid. Crop [460,210,885,592] in original pixels.', 28, 49, w - 56, 7.5, 10, MUTED)
        if y < 60:
            raise ValueError('page50 note overflow')

    def detail75(c, w, h):
        header(c, w, h, 'new_000101 | row 52', 'Fresh review of the previously unrecorded requested row', 75, 'FOLLOW-UP DETAIL')
        crop = [465, 525, 580, 590]
        change_renderer.image_panel(c, image, crop, recovered, {52}, set(), h - 91, 'Before | recovered Round 3 geometry')
        change_renderer.image_panel(c, image, crop, current, {52}, set(), h - 378, 'Current | trim only the left and lower excess')
        text = ('Nominal box: [485,543,550,571] original pixels. The observed left transition is around x484-487; the lower front/lip transition is around y569-572. These are manual review ranges, not subpixel accuracy guarantees. Keep the existing left corner near (487.05,567.82) and the canonical missing right corner. No hidden full width is inferred.')
        y = wrapped(c, text, 28, 155, w - 56, 9, 13)
        wrapped(c, 'The point tokens and object count are unchanged. This follow-up was not part of the original r3 ledger.', 28, 49, w - 56, 7.5, 10, MUTED)
        if y < 60:
            raise ValueError('page75 overflow')

    def atlas79(c, w, h):
        header(c, w, h, 'new_000101', '53 objects | 103 locatable corners | 3 missing corners | Original 1280 x 720', 79, 'TRAINING LABEL REVIEW')
        width = w - 60
        height = width * 720 / 1280
        bottom = h - 72 - height
        c.drawImage(ImageReader(image), 30, bottom, width, height)
        atlas_renderer.annotation(c, current, 30, bottom, width, height, (1280, 720), set())
        wrapped(c, 'Review rows: none recorded', 30, bottom - 18, width, 10, 14, font='ReviewBold')
        text = ('Rows 28 and 48 retain the recovered Round 3 corrections. Fresh review also trims row 52 to [485,543,550,571] pixels, removing rack/support overreach while preserving the visible left corner and missing right corner. Other rows are inherited recovery geometry, not newly certified physical labels. This page uses the verified original PNG and the current recovered-review labels.')
        y = wrapped(c, text, 30, bottom - 39, width, 9, 13, MUTED)
        if y < 43:
            raise ValueError('atlas79 overflow')

    outputs = []
    for kind, reader, source, size, replacements, appendices in [
        ('changes', rc, source_changes, size_c, {0:cover_changes,49:changes50}, [detail75, lambda c,w,h: appendix(c,w,h,76,False,None)]),
        ('atlas', ra, source_atlas, size_a, {0:cover_atlas,78:atlas79}, [lambda c,w,h: appendix(c,w,h,166,True,None)])]:
        writer = PdfWriter()
        writer.clone_document_from_reader(reader)
        for index, callback in replacements.items():
            replace_page_contents(writer, index, build_page(size, callback))
        if kind == 'changes':
            stream = ContentStream(writer.pages[2].get_contents(), writer)
            found = False
            changed = 0
            for operands, operator in stream.operations:
                if operator == b'Tj' and str(operands[0]) == 'new_000101':
                    found = True
                elif found and operator == b'Tj':
                    if str(operands[0]) != 'train / 2 edits':
                        raise ValueError('unexpected new101 index sequence')
                    operands[0] = TextStringObject('train / 3 edits')
                    found = False
                    changed += 1
            if changed != 1:
                raise ValueError('expected one index update')
            writer.pages[2][NameObject('/Contents')] = writer._add_object(stream)
        for callback in appendices:
            new = build_page(size, callback)
            writer.add_page(new.pages[0])
        if kind == 'changes':
            writer.add_outline_item('75. new_000101 row 52 follow-up', 74)
            writer.add_outline_item('76. Recovery provenance and limits', 75)
        else:
            writer.add_outline_item('166. Recovery provenance and limits', 165)
        remove_old_credentials(writer)
        writer.add_metadata({'/Title':'Cassette ' + ('front changes' if kind == 'changes' else 'label atlas') + ' - recovered review - 18 November 2026', '/Author':'Siuuuers/dl - assisted annotation review', '/Subject':'Review-only PDF geometry recovery plus new101 row52 correction; v2 experiment unchanged', '/Creator':'build_recovered_review_pdfs.py'})
        filename = 'cassette-front-changes-2026-11-18-recovered.pdf' if kind == 'changes' else 'cassette-label-atlas-2026-11-18-recovered.pdf'
        path = out / filename
        with path.open('wb') as f:
            writer.write(f)
        before_links = links_snapshot(source)
        after_links = links_snapshot(path)
        if before_links != after_links[:len(before_links)]:
            raise ValueError('existing PDF links changed: ' + filename)
        new_doc = fitz.open(path)
        old_doc = fitz.open(source)
        if len(new_doc) != len(old_doc) + len(appendices):
            raise ValueError('page count mismatch')
        old_toc = old_doc.get_toc()
        if new_doc.get_toc()[:len(old_toc)] != old_toc:
            raise ValueError('existing bookmarks changed')
        outputs.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'pages':len(new_doc),'existing_page_numbers_preserved':True,'existing_link_count':sum(len(x) for x in before_links),'existing_links_preserved':True,'existing_bookmarks_preserved':True,'replaced_pages':[x+1 for x in replacements],'text_amended_pages':[3] if kind == 'changes' else [],'appended_pages':list(range(len(old_doc)+1,len(new_doc)+1)),'current_new101_label_sha256':sha(current_label_path)})
    audit = {'kind':'recovered_review_pdf_update','source_changes_sha256':SOURCE_CHANGES_SHA,'source_atlas_sha256':SOURCE_ATLAS_SHA,'source_new101_png_sha256':SOURCE_PNG_SHA,'recovered_new101_label_sha256':sha(source_labels),'current_new101_label_sha256':sha(current_label_path),'current_new101_label_path':str(current_label_path),'new101_row52_before_line':old_line52,'new101_row52_after_line':current_text.splitlines()[51],'counts':{'train_fronts':3789,'development_fronts':662,'replacements':889,'deletions':12,'additions':0},'outputs':outputs,'limits':['This is a new review revision, not recovery of original r3 authoring bytes.','Only new101 row52 is freshly physically refined; other rows preserve the recovered review.','Source r3 PDFs remain immutable; existing page numbers/navigation remain intact.','The v2 training and scoring inputs are unchanged.'],'generator_sha256':sha(Path(__file__))}
    audit['current_review_manifest_path'] = str(current_manifest_path) if current_manifest_path else None
    audit['current_review_manifest_sha256'] = sha(current_manifest_path) if current_manifest_path else None
    audit['runtime'] = {'python':sys.version.split()[0], **{name:importlib.metadata.version(name) for name in ['PyMuPDF','pypdf','reportlab','Pillow']}}
    (out / 'recovered-pdf-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
