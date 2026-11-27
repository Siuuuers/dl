#!/usr/bin/env python3
"""Render the focused R50 amendment using exact original pixels and vector labels."""
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import render_front_review_r46 as base

W, H, M, WIDTH = base.PAGE_W, base.PAGE_H, base.MARGIN, base.WIDTH


def header(c, title, subtitle):
    base.page_header(c, title, subtitle)


def text(c, value, x, top, width, size=9.5, height=100):
    return base.para(c, value, x, top, width, size, leading=size * 1.3, max_height=height)


def table(c, headers, rows, top):
    widths = [WIDTH * .45] + [WIDTH * .1375] * 4
    x = M
    c.setFillColor(base.INK)
    c.rect(M, top - 27, WIDTH, 27, fill=1, stroke=0)
    for label, width in zip(headers, widths):
        base.para(c, escape(label), x + 7, top - 6, width - 14, 8,
                  color=colors.white, bold=True, max_height=20)
        x += width
    top -= 29
    for index, row in enumerate(rows):
        c.setFillColor(base.PALE if index % 2 == 0 else colors.white)
        c.rect(M, top - 25, WIDTH, 25, fill=1, stroke=0)
        x = M
        for value, width in zip(row, widths):
            text(c, escape(str(value)), x + 7, top - 6, width - 14, size=8.5, height=18)
            x += width
        top -= 25
    return top


def render_pdf(manifest, ledger, report, originals, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(W, H), invariant=1, pageCompression=1)
    c.setTitle('R50 cassette-front amendment: new_000010 row029')
    c.setAuthor('Cassette vision project review')
    op = ledger['operations'][0]
    im = Image.open(Path(originals) / 'new_000010.png').convert('RGB')

    header(c, 'R50 | One clarified bottom boundary',
           'new_000010 / R38 row 029 | R48 predecessor to R50 proposal | 27 November 2026, Hong Kong')
    text(c, '<b>Physical bottom: y = 720 to nominal y = 714.</b> The user identifies the lower band as another cassette. '
         'The retained left corner at (325, 714) supports the visible lower-left junction. '
         'The right corner stays unavailable.', M, H - 80, WIDTH, size=10.3, height=44)

    region = (300, 640, 480, 720)
    gap = 12
    width = (WIDTH - 2 * gap) / 3
    top = H - 143
    for index, (label, line, changes) in enumerate([
            ('Original pixels', None, ()),
            ('R48: bottom held at 720', op['before_line'], ()),
            ('R50: bottom at 714', op['after_line'], ('ymax',))]):
        base.image_panel(c, im, region, M + index * (width + gap), top,
                         width, 150, label, line, changes)
    y = top - 146
    text(c, '<font color="#08a5bd">Cyan: retained box</font> &nbsp; '
         '<font color="#d21d9e">Magenta: amended bottom</font> &nbsp; '
         '<font color="#09b442">Green: unchanged visible left corner</font>',
         M, y, WIDTH, size=8.5, height=20)
    y -= 27
    text(c, '<b>Decision.</b> ' + escape(op['reason']), M, y, WIDTH, size=9.1, height=64)
    y -= 78
    iw = 225
    ih = iw * 720 / 1280
    bottom = y - 12 - ih
    text(c, 'Full original / reviewed location', M, y, iw, size=8.1, height=15)
    c.drawImage(ImageReader(im), M, bottom, iw, ih)
    c.setStrokeColor(base.MAGENTA)
    c.setLineWidth(.8)
    c.rect(M + region[0] * iw / 1280,
           bottom + (720 - region[3]) * iw / 1280,
           (region[2] - region[0]) * iw / 1280,
           (region[3] - region[1]) * iw / 1280, fill=0, stroke=1)
    x = M + iw + 20
    text(c, '<b>What is preserved</b><br/>The top, both sides, instance identity, '
         'visible left-point tokens and exact missing-right slot <b>0 0 0</b> remain unchanged. '
         'This is one development-label amendment; the training labels are unchanged.<br/><br/>'
         '<b>Coordinate convention</b><br/>The original image is 1280 x 720. Y increases downward. '
         'The physical bottom is stored <b>ymax</b>. Nominal visual uncertainty is about 2 px; '
         'serialization precision is not physical accuracy.', x, y, WIDTH - iw - 20, size=9, height=135)
    base.footer(c, 'User ownership clarification plus original-image review | One field changed | No hidden point inferred')
    c.showPage()

    header(c, 'What the amendment changes in evaluation',
           'Same PR137 cached predictions, same matcher and thresholds; R48 and R50 labels scored separately')
    text(c, 'This replay changes the reference boundary only. It performs no training or fresh inference. '
         'R50 keeps all 654 targets and all 1,220 locatable endpoints. '
         'The complete error atlas retains the original R38 case identities and states its current R50 status.',
         M, H - 80, WIDTH, size=10, height=43)
    reports = report['reports']
    columns = [('r48', 'reference'), ('r50', 'reference'), ('r48', 'candidate'), ('r50', 'candidate')]
    rows = []
    for label, key in [('Missed targets', 'fn'), ('All unmatched outputs', 'fp'),
                       ('Formal duplicate subset', 'duplicate_fp'), ('Other unmatched outputs', 'other_fp')]:
        rows.append([label] + [reports[v][arm]['detection'][key] for v, arm in columns])
    rows.append(['Both corners within 5 px'] + [
        f"{reports[v][arm]['corners_by_locatable_count']['2']['within5']} / 583" for v, arm in columns])
    y = table(c, ['Postclip metric', 'Ref / R48', 'Ref / R50', 'Cand / R48', 'Cand / R50'], rows, H - 138)
    unchanged = not report['assignment_changes']
    text(c, '<b>Interpretation.</b> ' + (
        'Every target keeps its matched/missed state and assigned prediction identity. '
        'A better-defined annotation does not itself make a fixed model learn.' if unchanged else
        'Some assignment outcomes change under the revised reference; the JSON identifies them. '
        'These are reference-version effects on fixed predictions.'), M, y - 13, WIDTH, size=9.2, height=35)

    parent = Path(__file__).resolve().parents[1] / 'data/reviews/fronts-2026-11-27-r48'
    parent_ledger = json.loads((parent / 'audit-ledger.json').read_text())
    held = next(r for r in parent_ledger['rows'] if r['stable_id'] == 'new_000089:R38:041')
    image = Image.open(Path(originals) / 'new_000089.png').convert('RGB')
    top = y - 65
    base.image_panel(c, image, (562, 417, 680, 480), M, top, 280, 160,
                     'Remaining hold: new_000089 / R041 right edge', held['after_line'])
    right = M + 300
    text(c, '<b>One development ownership hold remains.</b><br/>'
         'The thin strip above the rounded gripper may be this front\'s owned upper rim '
         'or the preceding cassette\'s underside. The retained right bound is 664.14; '
         'the right corner stays missing. This round provides no new evidence to settle it.<br/><br/>'
         '<b>Next measured step</b><br/>Finish the interrupted proposal-stage diagnosis. '
         'Inspect whether errors arise in decoded geometry, confidence, NMS, postclip suppression '
         'or assignment. The new error atlas adds visual types and targeted actions while '
         'keeping unobserved causes explicit.', right, top, WIDTH - 300, size=9.1, height=155)
    base.footer(c, 'Assistant-reviewed reference proposal | R38 and prior releases preserved | PR35 remains selected')
    c.showPage()
    c.save()
    index = dict(schema_version=1, round=50, pages=2,
                 pdf_name=output.name, pdf_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                 amended_target=dict(stable_id=op['stable_id'], page=1),
                 remaining_hold=dict(stable_id='new_000089:R38:041', field='xmax', page=2))
    output.with_suffix('.index.json').write_text(json.dumps(index, indent=2) + '\n')
    return index
