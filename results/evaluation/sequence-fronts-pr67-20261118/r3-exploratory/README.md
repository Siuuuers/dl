# Exploratory replay on recovered r3 labels

**This separate diagnostic leaves the preregistered PR67 comparison unchanged.** It does not train a model, run new inference, change a threshold, or promote a checkpoint. The same three saved prediction files were scored once against the recovered r3 development labels.

Recovery manifest SHA-256: `6e70cb78b7ae6a71d9a61021c42a94bc7c24ccb2eaeeba673ba326c2434aa3d7`.

The recovered reference has **20 images, 662 fronts, 1,226 locatable corners, 98 missing corners and three empty images**. Corner-availability groups contain **20 zero-corner, 58 one-corner and 584 two-corner fronts**. It is a PDF geometry recovery, not a byte-identical restoration of the original authoring files or an independent physical certification. The held cases `new_000007` rows 7 and 14 remain unresolved.

## Measured outcomes

| Model | Matched | Missed | Unmatched outputs | Duplicate subset | Both corners ≤5 px | Both ≤10 px | Miss-free nonempty scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| PR35 | 648 | 14 | 80 | 32 | 533/584 | 582/584 | 9/17 |
| PR67 control | 646 | 16 | 83 | 30 | 531/584 | 583/584 | 7/17 |
| PR67 candidate | 641 | 21 | 105 | 28 | 532/584 | 584/584 | 7/17 |

Each model has zero outputs on the three empty images and one error-free nonempty image out of 17. These observations concern this reused development set only.

| Locatable corners | Objects | PR35 misses | Control misses | Candidate misses |
|---|---:|---:|---:|---:|
| Zero | 20 | 7 | 7 | 10 |
| One | 58 | 6 | 8 | 11 |
| Two | 584 | 1 | 1 | 0 |

The candidate still has **5 more misses and 22 more unmatched outputs than its control**, and **7 more misses and 25 more unmatched outputs than PR35**. Its 584/584 result at 10 px remains restricted to the two-locatable-corner subgroup. The ranking and model-selection recommendation do not change.

## Why the miss counts fell

Relative to the frozen 666-front primary reference, the four removed reference rows are:

- `new_000009:8`
- `new_000011:15`
- `new_000057:1`
- `new_000059:1`

All four were previously unmatched for all three models. After applying the recovery's explicit row maps, **no model gained or lost any matched physical reference identity**. Each model's miss count falls by exactly four because those reference targets were deleted; the predictions and matched counts remain unchanged. This is an annotation effect, not a learned improvement.

Five development label files differ in r3. The changes do not resolve the partial-front extent disagreements identified in the original comparison. In particular, `new_000010:31/32`, `new_000056:15`, and `new_000090:41/42` still contribute the candidate's scored losses. Whether the broad reference extent is supported by separated visible pieces of the same front remains an ownership question; narrow boxes are not automatically correct.

Use the row maps when discussing r3. Primary-reference `new_000009:53` is now r3 row 52, and primary `new_000009:56` is now r3 row 55. The latter was **not** the deleted row. Primary `new_000011:61` is now r3 row 60. Treating row numbers as permanent object identities would misreport these cases.

Against control, the r3 candidate gains current-row matches `009:52`, `056:14`, `087:15`, and loses `008:33`, `009:55`, `010:31`, `010:32`, `011:60`, `056:15`, `090:41`, `090:42`. These are the same physical gain/loss cases as in the fixed comparison. The 087:15 IoU-boundary effect and the new keyboard output in 089 also remain.

## Reproduce from the committed development labels

Run from this repository's root with Python 3.11 or newer and the original 20 validation PNGs materialized through Git LFS. The saved prediction files, fixed `comparison.json` and scoring scripts are included in the checkout. This replay uses the Python standard library and needs no checkpoint download, GPU or new inference.

The diagnostic is bound to the **historical PDF recovery**, whose immutable manifest is `results/review/r3-recovery-20261118/manifest.json` with SHA-256 `6e70cb78b7ae6a71d9a61021c42a94bc7c24ccb2eaeeba673ba326c2434aa3d7`. The current review bundle additionally includes the later `new_000101:52` training-only correction. Its manifest is a different artifact and must not be passed to this replay. Its 20 development label files are still byte-identical to the historical recovery, so the following command assembles a temporary historical view using only those files and the historical manifest. Every copied development label is checked against that manifest before scoring.

```bash
set -e
pr67_evidence="$PWD/results/evaluation/sequence-fronts-pr67-20261118"
r3_replay="$(mktemp -d "${TMPDIR:-/tmp}/dl-pr67-r3-replay.XXXXXX")"

git lfs pull --include='data/images/val/*.png' --exclude=''
mkdir -p "$r3_replay/reference/labels"
cp results/review/r3-recovery-20261118/manifest.json \
  "$r3_replay/reference/manifest.json"
cp -R data/reviews/fronts-2026-11-18/labels/val \
  "$r3_replay/reference/labels/val"

python -B "$pr67_evidence/r3-exploratory/replay_r3.py" \
  --root . \
  --manifest "$r3_replay/reference/manifest.json" \
  --prediction-root "$pr67_evidence" \
  --output "$r3_replay/metrics.json"

cmp "$pr67_evidence/r3-exploratory/metrics.json" "$r3_replay/metrics.json"
```

A successful `cmp` exits with status zero and prints nothing. The temporary view is only a development-reference input; it is not a reconstruction of the historical training tree or a new training release. If later development labels change, the replay will reject them rather than silently measure a different reference.

## Reproduce after recovering the original PDFs again

The same replay also accepts the fresh output directory of `scripts/recover_pdf_review.py`. Use the **original r3 PDFs**, identified below and recorded in the immutable recovery manifest. Updated PDFs that include the `new_000101:52` follow-up are different inputs. The repository records the original PDF hashes, but does not embed the PDF files or a public download URL; obtain the original saved files before running this alternative. Their basenames must be preserved because those names are part of the manifest.

| Original PDF | SHA-256 |
|---|---|
| `cassette-label-atlas-2026-11-18-r3.pdf` | `e5a34c8f8670b2b1ca67c770fca5ad21782942fb212f04384e09ac154b4c4a6f` |
| `cassette-front-changes-2026-11-18-r3.pdf` | `53d913f6e62bf8ca5eb573a088f4f9c2857dc0744e0ede1b5c949ffe50297ca3` |

The manifest records the recovery runtime, so exact regeneration requires **Python 3.12.14 and PyMuPDF 1.26.6** as well as the unchanged recovery script. The script SHA-256 is `1099b7ced5e9dfd99b7eb450a4bb0a29ee7025d8924c856bb70b210ba49fc232`; it pins the parent inputs from source commit `9304d4feb8aac481638ff08829dfb6e86e437c95`. With `uv` available, the following commands create the exact PDF environment. Set `r3_pdf_dir` to the directory containing the two original PDFs. Run from the repository root; keep `pr67_evidence` and `r3_replay` from the preceding block, or initialize them there first.

```bash
r3_pdf_dir=/absolute/path/to/original-r3-pdfs
uv venv --python 3.12.14 "$r3_replay/pdf-env"
uv pip install --python "$r3_replay/pdf-env/bin/python" 'PyMuPDF==1.26.6'

"$r3_replay/pdf-env/bin/python" -B scripts/recover_pdf_review.py \
  --root . \
  --atlas "$r3_pdf_dir/cassette-label-atlas-2026-11-18-r3.pdf" \
  --changes "$r3_pdf_dir/cassette-front-changes-2026-11-18-r3.pdf" \
  --output "$r3_replay/recovered-from-pdf"

cmp results/review/r3-recovery-20261118/manifest.json \
  "$r3_replay/recovered-from-pdf/manifest.json"

python -B "$pr67_evidence/r3-exploratory/replay_r3.py" \
  --root . \
  --manifest "$r3_replay/recovered-from-pdf/manifest.json" \
  --prediction-root "$pr67_evidence" \
  --output "$r3_replay/metrics-from-pdf.json"

cmp "$pr67_evidence/r3-exploratory/metrics.json" \
  "$r3_replay/metrics-from-pdf.json"
```

Use a new recovery directory and a new metrics output path. Leave this historical recovery output before the separate training-only follow-up patch; the current review bundle and its manifest describe that subsequent operation elsewhere.

`replay_r3.py` verifies the historical manifest, all 20 recovered label hashes, original image hashes, saved prediction hashes and scoring helper identities. It calls the existing matcher and corner scorer at unchanged confidence 0.13, matching IoU 0.50, point cutoff 0.50 and 5/10 original-pixel radii. The saved inference already used NMS 0.70; replay does not run NMS again. `metrics.json` contains the results; `identity-reconciliation.json` records how the r3 current rows map to the frozen reference identities.

The original `comparison.json` remains the fixed 666/663-reference experiment. This recovered-reference replay remains exploratory and provides no automatic release or model-promotion approval.
