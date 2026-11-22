# Reproduce the r22 PDF checks

The completed results are in `../pdf-independent-verification.json` and `../pdf-visual-review.json`. Both identify the delivered PDF by SHA256 and retain all verification results without requiring the original scratch directory. The PDF and full rendered PNGs are not stored in Git.

The vector checker uses Python with PyMuPDF (`fitz`), Pillow and pypdf. The representative renderer also needs Poppler's `pdftoppm` on PATH. Use the repository data and review snapshots identified in the receipts, and download the final PDF identified by SHA256 `f0d30051a59a1341fe2f93975822b046f9ec2a04580df363b48f90fe3cce05e5`.

From the repository root, supply the downloaded PDF path and an output directory outside the checkout:

```bash
python -B results/review/front-review-20261122-r22/pdf-qa/check_vectors.py \
  --pdf /path/to/front-review-r22-all-labeled-photos.pdf \
  --report /path/to/qa-output/vector-report.json

python -B results/review/front-review-20261122-r22/pdf-qa/render_representative.py \
  --report /path/to/qa-output/vector-report.json \
  --output-dir /path/to/qa-output/rendered
```

The checker infers the repository root from its location; `--root` can override it. It checks every photo page against actual saved labels and original RGB image bytes, including coordinates, field colours, marker availability, row identities, footer lists, links, bookmarks and clipping. The generated report supplies the rendering plan.

Rendering produces four compact-index pages, eight representative photo pages and two detail crops. A new rendering is not a new visual approval: inspect every planned PNG before issuing a new visual receipt. The committed visual receipt records the completed inspection and the precise PNG digests inspected in this session. Raster encoding can depend on the Poppler environment; source PDF, saved-label and embedded-pixel checks provide the primary exact-content verification.

The packaged script copies differ from the completed-run copies only in portable path arguments and a corrected selection-purpose caption for photo015. Assertions are unchanged. No additional QA was run during packaging.
