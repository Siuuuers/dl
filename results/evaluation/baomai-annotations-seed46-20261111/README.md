# Reviewed-annotation seed46 confirmation evidence

Report date: 2026-11-11. Execution records use UTC.

This capsule preserves the completed evaluation of PR35's seed46 control and
reviewed-annotation candidate, together with a fresh epoch173 baseline. Both arms
completed 30 epochs from the same epoch173 initialization. Evaluation uses each
arm's final `last.pt`, not trainer-selected `best.pt`.

The fixed confirmation does not repeat a strict primary miss reduction against
the matched control: both miss 24 PR30 reference objects at confidence 0.25,
while extras decrease from 58 to 52. Seed45 and seed46 remain separate paired
comparisons. Epoch173 remains the working default. The completed result supports
targeted error/supervision review rather than another seed sweep.

## Identity and actual hardware

- GPU return: PR35, commit `8e8bb99cabcb44ed0ec0acd14db911aa44013fd8`.
- Return ZIP: 39,776,456 bytes; SHA-256 `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91`.
- Training source: `23d7db97ca626f5591dbd26e699c0f23d318ee73`.
- Clean inference and replay source: `e4bb7019bcffd80b08cab28c4877dd562ee1d796`.
- Exact source tree: `39cc5696320384ce3f1679a70d3d5d51ceccd463`.
- Candidate manifest SHA-256: `28fc8f5da95605089d61fb062132200ed296c3a10b216f1a61dbda1798ea4ba7`.
- Recipe: `reviewed-annotations-seed46-v1`.
- Recipe SHA-256: `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`.
- Three fresh model runs, 180 unique image/model/threshold records, 360 per-image
  reference evaluations, and 18 reference-specific aggregates.
- Original and PR30 revised references are reported separately. The latter has
  661 objects, 1,228 locatable endpoints and 584 two-corner-eligible objects.

The returned training receipts identify **NVIDIA A100-SXM4-40GB** for both seed46
arms, with identical recorded runtimes and backend settings within the pair.
The PR describes later L4 sessions restoring and packaging completed Drive arms;
those sessions are not additional L4 training or partial-training resumption.
The PR34 notebook's completed-arm restoration path returns before training and
preserves the source, recipe and final-checkpoint checks.

Seed45 trained on L4. The paired comparisons remain internally matched, but
cross-seed differences cannot be attributed to seed alone because hardware also
changed. Remote receipts and notebook consistency are evidence of provenance,
not independent hardware or clock attestation. The separate resource-trial
receipt/log was not returned and is not independently reconstructed.

## Archive contents and omissions

Every ZIP member is a regular UTF-8 text file. `inventory.json` hashes and sizes
every other payload file and deliberately excludes itself. ZIP member names,
file types, UTF-8 decoding, CRCs, sizes and hashes are checked before fresh
extraction. The archive contains no original camera pixels, model weights,
returned GPU archive, virtual environment, package cache or application config.
The weights remain in PR35; their byte validation is recorded in the return audit.

The independent publication report review is intentionally external to this
capsule. Early `seed46-case-analysis.json` and `interpretation-run.log` are omitted;
the final `seed46-case-details.json`, interpretation summary and execution record
are preserved. `independent/seed45-comparison.json` retains the exact prior summary
for cross-seed interpretation and queue regeneration. Replaying seed45 itself
requires its earlier capsule and its own pinned source; its predictions are not
duplicated here.

## Replay the complete official summary

The recorded replay uses **Python 3.12.14**. Saved-prediction scoring uses only the
standard library and performs no training, inference or weight loading. Extract
the ZIP into a fresh directory, then obtain a clean pinned checkout and original
images through Git LFS:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Siuuuers/dl.git dl-replay
GIT_LFS_SKIP_SMUDGE=1 git -C dl-replay checkout --detach e4bb7019bcffd80b08cab28c4877dd562ee1d796
git -C dl-replay lfs pull --include='data/images/**' --exclude=''
python dl-replay/scripts/rescore_annotation_comparison.py \
  --root dl-replay \
  --evidence /absolute/path/to/extracted/evidence/seed-46 \
  --candidate data/candidates/annotations-2026-11-10/manifest.json \
  --recipe reviewed-annotations-seed46-v1 \
  --reference-dir dl-replay/data/references/visible-front-2026-11-10 \
  --evaluation-revision e4bb7019bcffd80b08cab28c4877dd562ee1d796 \
  --output /absolute/path/to/new-comparison.json
```

The output path must not exist. Keep the checkout clean and use this exact source:
the strict receipt/source audit checks script bytes, not just compatible behavior.
The verified fresh extraction used a new detached worktree, all 157 original PNGs
hash-verified and hardlinked from the original checkout, and a normal LFS clean
filter stat refresh. The index had to remain the exact source tree, with empty
Git status before and after replay.

Compare the complete replay output with
`evidence/seed-46/comparison-original-and-pr30.json`. Exclude only the top-level
`/created_at_utc` and `/command`. Check the actual original and new root values
before normalizing exactly these three fields:

- `/revised_reference/root`: `<REVISED_REFERENCE_ROOT>`.
- `/procedure/original_label_path_base`: `<REPOSITORY_ROOT>`.
- `/procedure/revised_label_path_base`: `<REVISED_REFERENCE_ROOT>`.

All remaining fields must match exactly, including every numerical result,
source identity, stored runtime, prediction timing and training timestamp. There
is **no numerical tolerance and no recursive removal of similarly named fields**.
The `semantic` function in `package_evidence.py` implements these checked rules.
The external `verification.json` records the fresh extraction, exact equality and
all exclusions; `packaging-commands.json` preserves the local execution transcript.

## Fresh M0 continuity is a different comparison

Fresh seed46 M0 uses the same epoch173 checkpoint and recorded CPU runtime as the
earlier M0 run, but its floating-point prediction payload is not bitwise identical.
The largest box-coordinate and keypoint-coordinate changes are each
**0.0001220703125 px**. Prediction counts/order, endpoint-availability flags and
discrete scoring/quality outcomes are preserved. Some conditional error quantiles
change by small amounts. These observations are documented separately in
`baseline-continuity.json` and `baseline-revised-continuity.json`.

Those measured differences between separate inference executions do not relax the
archive replay check: rescoring the exact newly stored predictions must reproduce
the full current summary exactly under the limited metadata rules above.

## Independent numerical scoring and case interpretation

`independent/independent_annotation_score.py` independently implements label
parsing, direct-area IoU, confidence-ordered one-to-one matching, endpoint
availability, Euclidean errors and quantiles with the standard library. It imports
no production matching or scoring functions:

```bash
python /absolute/path/to/extracted/independent/independent_annotation_score.py \
  --root dl-replay \
  --evidence /absolute/path/to/extracted/evidence/seed-46 \
  --reference-dir dl-replay/data/references/visible-front-2026-11-10 \
  --official-comparison /absolute/path/to/extracted/evidence/seed-46/comparison-original-and-pr30.json \
  --output /absolute/path/to/new-independent-audit.json
python /absolute/path/to/extracted/independent/interpret_seed46.py \
  --root dl-replay \
  --evidence /absolute/path/to/extracted/evidence/seed-46 \
  --reference-dir dl-replay/data/references/visible-front-2026-11-10 \
  --audit /absolute/path/to/extracted/independent/seed46-numerical-audit.json \
  --seed45-summary /absolute/path/to/extracted/independent/seed45-comparison.json \
  --output /absolute/path/to/new-case-details.json
```

The preserved independent audit agrees on all counts and identities across all
360 per-image reference evaluations and 18 aggregates. Its largest arithmetic
difference from production is **1.099120794378905e-14**; this independently written
calculation permits 1e-10 absolute arithmetic tolerance. That separate tolerance
does not apply to the exact official replay.

## Regenerate the targeted review queue

```bash
python /absolute/path/to/extracted/review-queue/generate_review_queue.py \
  --repo dl-replay \
  --seed45-summary /absolute/path/to/extracted/independent/seed45-comparison.json \
  --seed46-summary /absolute/path/to/extracted/evidence/seed-46/comparison-original-and-pr30.json \
  --case-details /absolute/path/to/extracted/independent/seed46-case-details.json \
  --output /absolute/path/to/new-review-queue
```

This deterministically produces `review-queue.json` and its README from verified
source bytes, stored miss sets, the PR30 row map and final case details. The fresh
extraction also regenerates both files byte-for-byte. It does not adjudicate labels.

The queue contains all **24 current candidate misses** plus two separate diagnostic
objects, with the paired detection loss attached to its existing miss record:
**26 unique objects** in total. Of the misses, **21** persist across the five related
M0/seed45/seed46 model variants, and all **19 previously unresolved reference rows**
remain misses. These variants are not independent trials or evidence of label truth.
The 24 misses include eight zero-corner, ten one-corner and six two-corner references;
18 of them lie outside the complete-pair metric's population.

## Layout and practical limits

- `evidence/seed-46/`: saved predictions, scores, quality results, complete summary,
  and small training/checkpoint evidence bundles with receipts and logs.
- `audit/`: archive, source, training, runtime and notebook-persistence review.
- `independent/`: final independent scorer, case code/results and prior seed45 summary.
- `review-queue/`: reproducible ranked objects, prior uncertainty and special cases.
- `runtime/`: text dependency records, probes and continuity script only.
- `original-pixel-review.json`: recorded inspection of original images.
- `alternate-proposal-diagnostics.json`: official versus alternate proposal details.
- `decision-review.json`: decision context and qualifications.
- `execution.json`, `logs/`: actual evaluation commands, exits and outputs.
- `run_evaluation.py`, `package_evidence.py`: exact orchestration sources.

Original scratch paths in execution records are retained as provenance; use the
portable commands above for a new workspace. This text capsule cannot redo weight
hash checks without separately obtaining the returned weights. No new inference,
training, checkpoint loading or reference annotation changes occur during replay.

These 20 repeatedly used development images and their endpoints are not independent
test trials. Small paired effects do not establish generalization or statistical
significance. Ten pixels is a provisional Euclidean radius in the original image,
not a calibrated physical tolerance. Reference ambiguity remains explicit; predictions
do not settle object identity or hidden corner locations. Old box-only training
interventions are not pooled into the present matched-pair result.
