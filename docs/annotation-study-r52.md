# R52: reproduce the annotation study on CPU

The [four-cell notebook](../notebooks/baomai_annotation_study_r52.ipynb) reproduces
the completed PR146 result audit and the R52 supplementary annotation review.
It creates both PDF reports and ranked review CSVs from pinned original images,
saved predictions and versioned labels. **Select a standard CPU runtime and run
all four code cells.** A GPU is not needed.

## What the notebook does

1. Checks out one immutable source commit. The sparse checkout is reapplied on
   every run, so an incomplete earlier setup can recover missing paths. A changed
   or incompatible source folder is preserved; choose a fresh `BAOMAI_R52_BASE`
   if you want another run.
2. Installs the two small rendering dependencies and verifies the PR146 and
   PR137 return archives, the pinned R44 supplementary-image capsule, 20 original
   development PNGs and 12 supplementary JPEGs. Only those images are fetched;
   no training initializer or full image collection is needed.
3. Independently replays all four scoring references, all 507 supplementary rows
   and the before/after four-side checks. It then renders the amendment and
   result/thesis reports. A completed compatible run can be verified and reused.
4. Creates and offers `baomai-annotation-study-r52.zip` for download. The archive
   includes the two PDFs, scores, case inventory, visual-observation ledger,
   exact annotation proposals, ranked CSVs, source files and hash receipts.

The notebook does not load a model, perform fresh inference, train, change
thresholds, merge a PR or promote a checkpoint. It records an already completed
training result and a new annotation proposal. The R52 side refinements are
approximately 2 px on one supplementary photo; a training gain has not been
measured. A later label-only training pilot is a separate decision after review.

## Read the outputs

| Output | Purpose |
| --- | --- |
| `review/reports/r52-cassette-amendments.pdf` | Every changed row, all 12 frame overviews, context/ownership rules and uncertainty. |
| `review/reports/r52-results-and-thesis.pdf` | PR146 scores, all remaining misses, recovered targets, corner failures, all unmatched outputs and a thesis plan. |
| `review/audit/` | Independently replayed numerical evidence and case identities. |
| `review/geometry/priority-before.csv`, `priority-after.csv` | A queue for original-image review; these flags are not confirmed annotation errors. |
| `review/geometry/audit-ledger.json` | Before/after lines, stable source IDs, accepted proposal reasons, and protected holds. |
| `review/review-receipt.json` and `inventory.json` | Exact provenance and package integrity checks. |

`outNNN` denotes a candidate's one-based position after postclip suppression;
`PNNN` denotes its retained primary prediction ID. These can differ. A source
annotation row such as `frame_0160:source:015` is a different identity. The added
`frame_0128:add:lower_left_top:001` row has no historical source-row number.

All 507 rows received agent-assisted visual review, with 10 rows / 11 side
amendments. The geometry scan flags 370 rows before and 368 after; the strict
interior subset is 38 before and 36 after. Neither reduction is a measured
quality improvement or review-efficiency result. Human adjudication, timed
review and an unflagged sample belong to the proposed thesis experiment.

## Run from a local checkout

The same runner can be called without a notebook:

```bash
python -B scripts/run_annotation_study_r52.py prepare --root . --work /path/outside/repository/r52
python -B scripts/run_annotation_study_r52.py review --root . --work /path/outside/repository/r52
python -B scripts/run_annotation_study_r52.py package --root . --work /path/outside/repository/r52
```

Optional `--archive`, `--prior-archive`, `--r44-archive`, `--originals` and
`--supplementary` arguments reuse existing downloads after exact hash checks.
Use the same path arguments for the prepare and review phases. The notebook
fetches the immutable R51 source Git tree required by the independent audit.
For a local shallow checkout, make that tree available as described in the
[result audit](partial-classification-r51-results.md).

The report renderer also accepts explicit paths and consumes an already passed
audit. Outputs stay outside the source checkout. Failed work is retained for
inspection; completed outputs are reused only when source, evidence and file
hashes agree.

The source pin and execution receipt are recorded in the publication evidence
under `results/review/annotation-study-r52-20261127/`. The notebook is a
reproduction and review tool, not evidence of an independently collected test
set. See the [thesis proposal](thesis-plan-r52.md) for that distinction.

## Published source and execution check

The notebook pins source commit
`d43dee1bb9e2fa837bf3713a0a7f13f467c48dec`. Its SHA-256 is
`2b4d88c030a37157be843b96ad86771cd6922e986728ec302797c8e6fc2f01c1`
(10,843 bytes). The source revision stays fixed when the notebook is subsequently
published in a newer commit.

All four actual code cells passed locally on CPU using a fresh checkout fetched
from GitHub and an existing, exactly verified input cache. The test replayed the
audit, produced both PDFs and built the evidence ZIP. A second execution began
with the R52 review directory deliberately omitted from an otherwise clean sparse
checkout: the first cell restored it, all four cells passed again, and the
completed-review receipt and bundle hash were unchanged. These are execution
checks of the notebook code against published source, not a claim of a Colab GPU
session or new model evaluation. The
[execution receipt](../results/review/annotation-study-r52-20261127/notebook-publication-check.json)
records the exact scope and output hashes.
