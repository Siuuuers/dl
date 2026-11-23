# Complete R24 new-series photo atlas

23 November 2026, Asia/Hong_Kong.

**The delivered PDF contains all 110 nonempty `new_000*` photographs and all
4,179 current cassette-front rows.** It renders the accepted R24 labels and
highlights only the preceding R23-to-R24 changes: 26 points in 22 rows across
four photographs. It does not make new annotation decisions.

The file is `front-review-r24-all-nonempty-new.pdf`, supplied with the chat
handoff. It has 116 A3 landscape pages and preserves original PNG pixels
losslessly. The 31 old-series photographs and 16 empty new-series photographs
are outside this PDF's requested scope; they remain in the dataset.

## Navigation

| Section | PDF pages |
| --- | --- |
| Overview and changed-photo links | 1 |
| Clickable full photo index | 2 |
| All 110 current nonempty new-series photos | 3-112 |
| Accepted original / R23 / R24 comparison sheets | 113-116 |

| Photograph | Full image | Magnified comparison |
| --- | ---: | ---: |
| `new_000010` | 12 | 113 |
| `new_000012` | 14 | 114 |
| `new_000014` | 16 | 115 |
| `new_000016` | 18 | 116 |

Cyan marks retained current annotations. Green rings and starred row tags mark
R24 corrections; magenta marks their previous R23 point positions. Circle and
square markers distinguish left and right endpoints. An unmarked original
thumbnail is included on each atlas page and reuses the full-resolution image.
Color identifies edit status, not physical certainty. Missing points are not
invented or drawn. Saved one-based row identities and both accepted standards
remain unchanged.

Photo 010 is development-only. Its revised review points do not enter training
or replace the original trainer validation or frozen scoring references.

## Reproduce the PDF

In a checkout containing the accepted R24 labels and hydrated original images,
with Pillow, ReportLab, PyMuPDF and pypdf available:

```bash
python3 scripts/build_front_review_r24_pdf.py --output output/front-review-r24-all-nonempty-new.pdf --receipt output/front-review-r24-all-nonempty-new.render.json
```

The renderer checks source and label identities, exact operation replay,
complete row coverage, index links and decoded RGB pixels embedded in the
saved PDF. The [delivered rendering receipt](../results/review/front-review-20261123-r24/complete-atlas-rendering.json)
records page mappings, pixel checks and completed visual inspection. The
rendered original is 148,941,681 bytes, SHA-256
`6ff2a34971900b497a58fd52e8be3b3ef2dc6b213cfc8205b1665b2dc134a0cb`.

The [annotation handoff](front-review-r24.md) and
[completed PR99 model evaluation](../results/evaluation/pr99-r24-20261123/README.md)
remain separate records. The photo atlas shows annotations, not model output.
