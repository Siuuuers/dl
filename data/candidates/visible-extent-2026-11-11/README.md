# Reviewed visible-front extents

**Status: annotation candidate preserved; matched training completed without model promotion.** The [PR43 evaluation](../../../docs/visible-extent-results-2026-11-11.md) finds no additional reference-object recovery at the declared confidence 0.13; misses change 20→21 and extras 85→87. This cumulative candidate preserves the exact eight-file `annotations-2026-11-10` overlay used for the selected PR35 model and adds a bounded, original-image-supported delta. The [review report](../../../docs/visible-extent-review-2026-11-11.md) supplies the images and uncertainty decisions. The [training handoff](../../../docs/training-visible-extent-2026-11-11.md) remains archived provenance; no rerun is requested.

The selected PR35 checkpoint and its explicit confidence 0.13 proposal command remain current. The completed comparison does not show that this supervision delta improves the operational model. The original-image reasons for its accepted annotation changes remain separately recorded.

## What changes relative to the current training overlay

| Image | Existing rows amended | Addition | Disposition |
| --- | --- | --- | --- |
| `new_000061` | 9, 14, 15 | None | Three bounded horizontal-edge corrections; other source box bounds remain intact. |
| `new_000079` | 34, 15, 16 | None | Exact inheritance of the previously reviewed three-box proposal. |
| `new_000082` | 26 | New row 29 | Align the fourth visible front and append the separately observed fifth front. |
| `new_000094` | 29–32 | None | Align four distinct fronts beside the thumb and remove unsupported box spans into the hand/gripper. |
| `new_000062`, `new_000063` | None | None | Reviewed, but the jaw-gap material does not settle a final observed extent. |

The new delta is **11 bbox replacements, one addition, no deletion and no new visible endpoint**. Every pre-existing class and corner token in these four changed files is preserved. The added row has two canonical `0 0 0` corner slots. Five selected existing rows remain unchanged and uncertain: `061:10`, `062:8/9` and `063:8/9`. The review does not certify their old boxes.

The 079 proposal is included exactly, so these 11 replacements include its three already proposed changes; they are not 11 additional changes on top of 079. This intervention also includes an object addition and should not be described as box-only.

## Exact composition

The cumulative overlay contains **12 training label files**: the prior eight plus `061`, `079`, `082` and `094`. The prior label bytes, manifest change objects and accepted decision objects are preserved exactly. The [standalone 079 proposal](../visible-extent-079-2026-11-11/README.md) and its provenance remain available.

The non-adopted PR40 `partial-fronts-2026-11-11` extension is **not** inherited. Neither are the historical deduplication or box-only candidates. The [manifest](manifest.json) reconstructs every cumulative edit against the frozen original labels; the [review record](review.json) distinguishes inherited decisions, new decisions and uncertainty.

| Effective training variant | Images | Objects | Visible endpoints | Missing endpoints | Empty images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen original | 137 | 3,772 | 6,839 | 705 | 13 |
| Current eight-file overlay | 137 | 3,771 | 6,841 | 701 | 13 |
| This cumulative candidate | 137 | 3,772 | 6,841 | 703 | 13 |

Original validation remains **20 images / 664 objects / 1,227 visible endpoints / 101 missing endpoints / three empty images** in both training arms. Post-training evaluation uses the separately versioned **663-object corrected reference**; those corrected validation labels are not inserted into training or its validation selection.

## Verify and stage

```bash
python -B scripts/verify_pose_annotation_candidate.py \
  --candidate data/candidates/visible-extent-2026-11-11/manifest.json

python -B scripts/stage_pose_training.py \
  --candidate data/candidates/visible-extent-2026-11-11/manifest.json \
  --output ../visible-extent-staged
```

The [mechanical verification](verification.json) checks all frozen label hashes, exact declared edits, full row maps, geometry and totals. The [composition verification](composition-verification.json) records exact inheritance, unchanged endpoint tokens and original-image hashes for the six reviewed images. The [completed staging check](staging-verification.json) additionally verifies all 157 original PNGs and 314 copied images plus 314 copied labels across the original/candidate staged variants, preserving validation. It ran in the recovered working checkout before publication; no training ran. Mechanical verification cannot establish physical identity or image-boundary correctness; those judgments are in the original-image review.

All original files remain intact. This candidate is an assistant-reviewed incremental proposal under the established policy, not supervisor certification, a complete relabeling of the images, or a measured improvement in model accuracy.
