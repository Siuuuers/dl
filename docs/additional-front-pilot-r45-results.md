# R45 returned results: one supplementary photo was trained

**PR133 contains completed training of `frame_0383`. The outcome is mixed, so PR35 remains selected.** The candidate improves several counts against the frozen 666-front reference but regresses against the later R38 annotations on the same 20 development photographs. These are separate reference definitions, not independent test sets.

## What entered training

The [returned PR133 archive](https://github.com/Siuuuers/dl/pull/133) uses reconstructed source [`ea4ac955`](https://github.com/Siuuuers/dl/tree/ea4ac955f2dfaf1add24c243599715f802eb41a7), published through the [rebuilt notebook](https://github.com/Siuuuers/dl/blob/c90b3bf31bedc21ce6402da3018b4c7143f1dd8c/notebooks/baomai_additional_front_pilot_r45_rebuilt.ipynb). It does not use the lost original R45 scripts preserved by draft PR132.

| Measured item | Reference | Candidate |
|---|---:|---:|
| Training photographs | 137 | 138 |
| Training fronts | 3,784 | 3,827 |
| Locatable training corner slots | 6,837 | 6,923 |
| Missing training corner slots | 731 | 731 |
| Supplementary `frame_*` photographs | 0 | 1: `frame_0383` |
| Epochs | 30 | 30 |
| Observed batches | 1,050 | 1,050 |
| Observed optimizer updates | 78 | 78 |
| Image presentations | 4,110 | 4,140 |

The shared 137 photographs have byte-identical effective R37 labels in both arms. Both independently started from the pinned epoch173 initializer with the same seed46 recipe, batch4, 1280-pixel input, AdamW, FP32 and Tesla T4 runtime. The trial was four batches and was discarded before both full runs. The final comparison uses each arm's `last.pt`.

Both original validation splits have the same 20 photographs / 664 original training-validation labels. None entered training. Evaluation rescored them against frozen666, historical663 and R38's 654 accepted fronts. Only candidate training contains `frame_0383`; it is never a test image. The other eleven R44 photographs remain outside both train and validation splits.

Earlier documented training used the 137 `new_*` / `old_*` photographs, while this separate supplementary `frame_*` pool was used here for inference and annotation review. That is the scope of the earlier statement that the frame photos had not been trained. It was not a claim about every upstream project or undocumented historical run. PR133 now changes that answer for exactly **one** supplementary photograph. See the [earlier training-lineage record](front-review-r44-training-plan.md).

## A/B result at the unchanged postclip operating point

Inference is unchanged: box confidence 0.13, ordinary NMS IoU 0.70, postclip NMS with strict IoU greater than 0.70, matching IoU at least 0.50, endpoint confidence at least 0.50. Each table entry is reference to candidate.

| Evaluation reference | Misses | All unmatched predictions | Duplicate subset | Other unmatched | Both corners within 5px | Both within 10px |
|---|---:|---:|---:|---:|---:|---:|
| Frozen666, primary historical gate | 36 to 33 | 99 to 98 | 25 to 22 | 74 to 76 | 517 to 518 / 584 | 573 to 575 / 584 |
| Historical663, diagnostic | 32 to 30 | 98 to 98 | 25 to 22 | 73 to 76 | 517 to 518 / 584 | 573 to 575 / 584 |
| R38 current accepted654 | 11 to 13 | 86 to 90 | 22 to 18 | 64 to 72 | 571 to 569 / 583 | 582 to 581 / 583 |

Duplicates are already included in all unmatched predictions. Thus the R38 candidate's 90 unmatched predictions comprise 18 duplicates plus 72 other unmatched predictions. Do not add duplicates to the unmatched total again.

Raw R38 scores are 11 to 13 misses, 94 to 98 unmatched, 30 to 26 duplicates and 64 to 72 other unmatched. The existing postclip pass removes eight predictions per arm without changing matches or corner scores. Both arms return zero detections on all three empty scenes.

The denominators include all reference objects with the stated number of locatable corners, including missed objects. Under R38, all 583 two-corner objects were matched by both models. The single-corner group falls from 46/54 matched to 44/54; its within-5px success falls from 41/54 to 40/54. The zero-corner group stays 14/17 matched. A small change in conditional endpoint error alone would hide this detection loss.

These counts compare the two **fresh R45 arms**. Earlier PR123/R40 reference counts belong to a different completed model and must not be substituted into this comparison.

## Which misses changed

Rows below are one-based rows of the named evaluation reference, not supplementary-photo source-row identities.

| Reference | Image / row | Candidate outcome | Best retained box IoU, reference to candidate |
|---|---|---|---:|
| R38 | `new_000009` / 51 | Newly missed | 0.5381 to 0.4916 |
| R38 | `new_000010` / 30 | Recovered | 0.3780 to 0.6145 |
| R38 | `new_000011` / 59 | Newly missed | 0.7860 to 0.4702 |
| R38 | `new_000089` / 48 | Newly missed | 0.5342 to 0.4973 |
| Frozen666 | `new_000057` / 7 | Recovered | See cached per-image scores |
| Frozen666 | `new_000087` / 15 | Recovered | See cached per-image scores |
| Frozen666 | `new_000089` / 43 | Recovered | See cached per-image scores |
| Historical663 | `new_000010` / 14 | Newly missed | See cached per-image scores |

All four changed R38 objects have exactly one locatable corner. The three new misses still have overlapping retained predictions, but their boxes fall below the fixed 0.50 matching criterion. For example, `new_000089:48` retains a 0.834-confidence box at IoU 0.4973. Lowering confidence does not correct the geometry of that existing box. The returned cache contains final predictions; it cannot by itself determine the pre-NMS cause of every other missed target.

The candidate's thirteen R38 misses are:

| Image | Reference rows |
|---|---|
| `new_000008` | 33, 34 |
| `new_000009` | 51, 54 |
| `new_000010` | 33 |
| `new_000011` | 55, 56, 57, 59, 60 |
| `new_000088` | 51 |
| `new_000089` | 48 |
| `new_000090` | 39 |

## Decision for the next round

Keep the model architecture and operating thresholds unchanged. Continue qualifying whole supplementary photographs and correcting physical front ownership and corner locations. One added image with 43 fronts is still one capture condition, and all 86 of its corner slots are locatable; it is not a targeted missing-corner treatment.

The annotation follow-up should preserve R44/R45 history, record before/after coordinates and remaining holds, and export only fully qualified whole images. If the next annotation pass leaves the same sole training candidate and unchanged `frame_0383` labels, repeating the same training treatment would not measure those new amendments. A CPU audit/export notebook is useful immediately; a new training comparison should require a changed, frozen treatment first.

Frozen666 remains the primary historical gate, historical663 remains diagnostic, and R38 describes the current accepted geometry. The disagreement is a reason to inspect the changed objects, not to select whichever reference looks most favorable. PR133 does not contain a same-runtime comparison with selected PR35 and does not establish independent-session reliability. There is no automatic checkpoint promotion.

## Reproducible CPU verification

The [audit result](../results/evaluation/pr133-r45-20261127/cpu-audit.json) records a successful check of all 334 inventory entries, the label lineage and completed receipts, final checkpoint bytes, source/runtime records, all 40 image-role postclip outputs and all twelve score reports. The archive has 335 members including its inventory and SHA-256 `0babe043f8358f7ceed3e4e0e570421585690ebf79496c689ba84c1b5b25869a`.

From the repository root, download the pinned two chunks and replay without a GPU or ML package installation:

```bash
python3 scripts/audit_additional_front_pilot_r45_results.py \
  --download-dir /tmp/baomai-pr133 \
  --output /tmp/baomai-pr133/cpu-audit.json
```

For an existing reconstructed ZIP, replace `--download-dir` with `--archive /path/to/baomai-additional-front-pilot-r45-rebuilt-results.zip`.

This verifies the returned evidence and recomputes metrics from saved predictions. It does not perform new model inference or training. Checkpoint hashes bind the recorded files and receipts; they do not independently demonstrate which model generated a prediction without rerunning inference. The audit pins the source scoring helpers and all three reference manifests, so an accidental reference or evaluator change fails explicitly.

The automatic GitHub review failure on PR133 was accompanied by a code-review usage-limit message. It was not an identified defect in the result files.
