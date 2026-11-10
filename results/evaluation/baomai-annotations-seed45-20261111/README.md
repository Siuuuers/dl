# Reviewed-annotation seed45 evaluation evidence

Report date: 2026-11-11, Asia/Hong_Kong. Execution records use UTC.

This capsule preserves the completed three-model evaluation of PR32's matched
seed45 training pair. It does not contain a seed46 result. The working model
remains epoch173 while the predeclared seed46 confirmation is prepared.

## Identity

- GPU return: PR32, commit `074f4d28736fab8b82baf7e56e6ad551997ae3c6`.
- Return ZIP: 39,774,041 bytes; SHA-256 `598c6e1177401690e0057fcaafa260a41e798b216d1c59309b652a94d518293e`.
- Training source: `1c5d2e48ab26c578c5f6660010d6ecfb33231eb8`.
- Clean inference and replay source: `ed4ee162738685bff93d6cad1402d95e128bceb4`.
- Candidate manifest SHA-256: `28fc8f5da95605089d61fb062132200ed296c3a10b216f1a61dbda1798ea4ba7`.
- Recipe: `reviewed-annotations-seed45-v1`.
- Three fresh model runs, 180 unique image/model/threshold records, 360 per-image
  reference evaluations, and 18 reference-specific aggregates.
- Original and PR30 revised references are reported separately. The latter has
  661 objects, 1,228 visible endpoints and 584 two-corner-eligible objects.

All files inside this ZIP are text. `inventory.json` hashes every other payload
file and deliberately excludes itself. Original camera images, model weights
and the returned GPU ZIP are not duplicated. The returned weights remain in
PR32; their byte validation is recorded in the return audit. The separate
resource-trial receipt/log was not returned and is not independently verified.

## Replay the complete official summary

Use Python 3.12.14 for closest numerical reproduction. Saved-prediction scoring
needs no ML packages and performs no training, inference or weight loading.
Extract this ZIP into a fresh directory. Obtain a clean pinned checkout and its
original validation images through Git LFS:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Siuuuers/dl.git dl-replay
GIT_LFS_SKIP_SMUDGE=1 git -C dl-replay checkout --detach ed4ee162738685bff93d6cad1402d95e128bceb4
git -C dl-replay lfs pull --include='data/images/val/*.png' --exclude=''
python dl-replay/scripts/rescore_annotation_comparison.py \
  --root dl-replay \
  --evidence /absolute/path/to/extracted/evidence/seed-45 \
  --candidate data/candidates/annotations-2026-11-10/manifest.json \
  --recipe reviewed-annotations-seed45-v1 \
  --reference-dir dl-replay/data/references/visible-front-2026-11-10 \
  --evaluation-revision ed4ee162738685bff93d6cad1402d95e128bceb4 \
  --output /absolute/path/to/new-comparison.json
```

The output path must not exist. Use exactly this source for seed45 replay.
The later seed46 recipe registry changes `pose_checkpoint.py` bytes; the strict
source audit correctly refuses to replay these old receipts under that source.

Compare the complete output with
`evidence/seed-45/comparison-original-and-pr30.json`. Exclude only the top-level
`/created_at_utc` and `/command`. Check that these three fields identify your
new checkout, then normalize the original and new checkout locations:

- `/revised_reference/root`: `<REVISED_REFERENCE_ROOT>`.
- `/procedure/original_label_path_base`: `<REPOSITORY_ROOT>`.
- `/procedure/revised_label_path_base`: `<REVISED_REFERENCE_ROOT>`.

All other values must match exactly. No numerical tolerance, source identity,
stored runtime, prediction timing, training timestamp, or result field is
excluded. The external `verification.json` records the actual fresh extraction,
clean detached replay checkout, full semantic equality and exact exclusion rules.

## Independent numerical audit

`independent/independent_annotation_score.py` separately implements label parsing,
IoU, confidence-ordered one-to-one matching, endpoint availability, Euclidean
errors and quantiles using only the standard library. It imports no production
scoring or matching functions. Its command is:

```bash
python /absolute/path/to/extracted/independent/independent_annotation_score.py \
  --root dl-replay \
  --evidence /absolute/path/to/extracted/evidence/seed-45 \
  --reference-dir dl-replay/data/references/visible-front-2026-11-10 \
  --official-comparison /absolute/path/to/extracted/evidence/seed-45/comparison-original-and-pr30.json \
  --output /absolute/path/to/new-independent-audit.json
```

The preserved independent audit agrees on all counts and identities. Its largest
floating-point difference from the production calculation is 1.11e-14. The
independent numerical comparison permits 1e-10 absolute arithmetic tolerance;
the official full-summary replay above requires exact numerical equality.

## Layout and limits

- `evidence/seed-45/`: saved predictions, scores, quality outputs, complete
  comparison, and small checkpoint evidence bundles with receipts and logs.
- `audit/`: independent return integrity, input, training and runtime audit.
- `independent/`: independent scorer, numerical audit and primary case records.
- `interpretation-review.json`: threshold/radius transitions, coordinates,
  common-endpoint diagnostics and separate reference effects.
- `original-pixel-review.json`: the three original-image inspection records.
- `execution.json`, `logs/`: actual command identities, exit codes and outputs.
- `baseline-continuity.json`: fresh M0 versus historical payload continuity.
- `run_evaluation.py`, `package_evidence.py`: exact orchestration sources.

Original scratch paths in execution records are preserved as provenance; the
commands above are the portable replay interface. The compact capsule alone
cannot repeat returned-weight byte checks. Recorded remote inventories are not
independent hardware attestation. One seed on a repeatedly used 20-image
development split does not establish statistical significance, independent-test
generalization or a physical acceptance tolerance. The 579/584 result applies only
to two-corner-eligible objects; nineteen of the candidate's 24 misses have fewer
than two locatable reference corners. Old box-only seeds are not pooled here.
