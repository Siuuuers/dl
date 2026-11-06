# Proposed training-label overlay: dedup-2026-11-06

This directory contains five replacement **training label files** and a manifest for a proposed removal of eleven redundant physical-instance rows. It is derived from the frozen `Siuuuers/dl` reference at `1c201e35a33727ac0ae59f1d11a3d7fbb72b5ae7` after original-image inspection by primary and independent agents. Its status is **proposed**, with no human approval or measured performance benefit claimed.

Read the [review report](../../../docs/training-pilot-2026-11-06.md) and [full evidence](../../../docs/training-pilot-2026-11-06.json) before applying it. The proposal is newly derived; it is not PR #2's unrecovered historical candidate.

## Verify without modifying data

From the repository root:

```bash
python3 scripts/verify_pose_candidate.py
```

The verifier checks every reference label hash and the exact candidate deletion, including row text, counts and retained-row mapping. It must exit with status 0 and JSON `status: "verified"`. That result certifies the recorded file transformation, not the visual truth of annotations or image-byte availability.

| Source label | Remove original one-based rows | Retain corresponding original rows | Objects before → after |
| --- | --- | --- | --- |
| `labels/train/new_000019.txt` | 56, 57, 58, 59, 60 | 63, 64, 65, 66, 67 | 68 → 63 |
| `labels/train/new_000075.txt` | 1, 2, 3 | 26, 24, 25 | 26 → 23 |
| `labels/train/new_000102.txt` | 15 | 54 | 54 → 53 |
| `labels/train/new_000112.txt` | 35 | 38 | 38 → 37 |
| `labels/train/new_000051.txt` | 21 | 22 | 23 → 22 |

All remaining rows are byte-for-byte unchanged, including boxes, endpoint coordinates and visibility states. The effective full training set would have 3,761 objects, 6,839 visible endpoints and 683 unlocatable endpoints. Validation remains 20 images, 664 objects, 1,227 visible endpoints and 101 unlocatable endpoints. Source labels, checkpoint, image paths and split remain unchanged.

## Before training

This directory is an **overlay**, not a complete training dataset. No candidate YAML is supplied. A separate staging step must create a separate paired `images/` and `labels/` tree, retain unchanged labels for all other images, apply only these five replacements, and verify actual source image bytes. Confirm the trainer's resolved label paths: a YAML pointing at the original image tree can still select the original labels. Preserve the frozen reference and do not edit its hashes to accommodate a candidate.

The separate staging step is now implemented in `scripts/stage_pose_training.py`; follow the [training preparation guide](../../../docs/prepare-pose-training.md). It builds and verifies complete reference/candidate copies without modifying this overlay. The [adopted annotation policy](../../../docs/annotation-policy.md) guides future edits; this candidate remains deletion-only and does not claim a complete migration of every retained box to that policy.

The remaining overlapping rows in `new_000030` (6/29) and `new_000123` (32/33) identify distinct neighboring fronts and must remain. IoU alone is not a deletion rule. Retained coordinates are not newly certified to a pixel tolerance, and hidden corners must not be inferred from a prediction.
