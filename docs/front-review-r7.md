# Round 7: follow up the marked examples, preserve the comparison

The six new screenshots were taken from the delivered **r6 review**. That exact review is preserved in the [published recovery checkpoint](https://github.com/Siuuuers/dl/commit/c8f8104a5c828747b93428579f5e5b8021beb700); this pass uses its row numbers as **BASE r6**. Earlier r5 numbers are not silently reused after r6's deletion. The frozen annotation provenance names local recovery commit `497da71bfdc07b49571a9eee3748b575817fca02`. Its published equivalent has the identical complete Git tree, as recorded in [recovery-checkpoint.json](../results/verification/front-review-r7-20261119/recovery-checkpoint.json). The notebook separately pins the actual published r7 source commit.

**This pass changes seven photographs: 18 existing rows and one restored visible front.** The replacement fields comprise **17 boxes, eight left-point fields and five right-point fields**; these counts overlap. There are no deletions. The complete current review contains **4,451 fronts: 3,788 in training photographs and 663 in development photographs**.

The separately delivered **cassette-front-review-r7.pdf** includes every one of the **157 original photos**, all current labels, and nine enlarged original / r6-before / r7-current comparisons. Magenta highlights only changes made in r7. The previous complete r6 visual review is inherited for unchanged photos; this follow-up does not claim a new per-row visual inspection of all 157 photos.

[Exact operations](../results/review/front-review-20261119-r7/operations.json), [row maps](../results/review/front-review-20261119-r7/row-maps.json), [coverage](../results/review/front-review-20261119-r7/request-coverage.json), [holds](../results/review/front-review-20261119-r7/review-status.json), and the [157-file review manifest](../data/reviews/fronts-2026-11-19-r7/manifest.json) bind the original pixels, decisions, labels and references.

## What the new examples changed

| Original photo | Follow-up decision |
| --- | --- |
| `new_000003` | Correct eight existing rows, restore the fifth small visible front below BASE 55, and remove independently unsupported rack/neighbor extents. Its own seam and visible pale material establish the restored front; stack spacing does not. Move reliably visible corners off the rack. BASE 58's inset left point is unsupported and its physical corner cannot be reliably located through the rail, so it becomes `0 0 0`. |
| `new_000007`, development | Tighten BASE 26 to a conservative source-derived box `[652,230,721,248]`. Copying the approximate blue bottom at about 246 would exclude the existing left point at 247.51. Keep both points and all frozen validation/scoring files unchanged. |
| `new_000017` | Move BASE 58's visible left corner from `(560,550)` to approximately `(558,546)`, at the upper front wall turn. The pale band and extra fragments remain unresolved; do not invent extra instances. |
| `new_000023` | Tighten terminal BASE 27's bottom from 513 to 509, above exposed rack. Larger proposed trims on BASE 24–26 remain held: gray material visibly descends below those proposed main-face bottoms, and its physical ownership remains unresolved. |
| `new_000095` | Refit BASE 10–14 boxes to observed fronts and their own rims. Move BASE 10–13 left points to the lower-front turns. Preserve all existing right points and both BASE 14 points exactly. |
| `new_000114` | Trim BASE 26's right bound from 1033 to 1030 and bottom from 468 to 466. Keep both landmarks and the separately held top boundary. The suggested 419–420 top is not silently accepted while upper-strip ownership remains uncertain. |
| `old_000013` | Honor the user's explicit confirmation: restore the exact original row 1 box tokens. Its top returns from about 15 to 47.56 pixels. Every endpoint and the other seven rows remain unchanged. The entire eight-row review file now equals the original file byte for byte. |

Coordinates refer to the **1280 x 720 original PNG**, not enlarged screenshot pixels. They are nominal estimates on blurred contours, not claims of subpixel physical accuracy.

The restored `new_000003` front is **CURRENT 56, immediately after BASE 55**. The previous BASE 56–59 become CURRENT 57–60. The previous deletion was too broad: a badly placed box did not establish that its underlying front was absent. The old r6 review and its report remain unchanged as the comparison baseline.

## What the box means

The [baomai data README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md) defines two actual lower-front corners, ordered by image X. The linked `yolo-pose` branch uses a different five-point, front/back task and must not redefine this dataset.

Our existing [annotation policy](annotation-policy.md) uses the visible front and its own frontal rim as the box target. Its precise box-boundary details are a project convention informed by the historical upstream editor, not a claim that the current supervisor README specifies every pixel boundary.

Rack, cast shadow, receding top plane and neighboring cassettes do not determine the box's outer extent. Some background inside an axis-aligned rectangle is unavoidable when the front is sloped or partly occluded. Shadow cast **on the target front** does not remove that material from the target. Thus neither an automatic inset nor removal of every dark pixel is justified.

There is no measured evidence here that deliberately adding adjacent rack or cast shadow to the labels improves recognition. Changing labels defines a hypothesis for the next controlled comparison; it does not establish a model gain.

## Held cases and training composition

The inherited case ledger contains **51 cases across 49 photographs**, including **44 training photographs** and five development photographs. All 31 old-series training photographs remain held for the broader front-rim/support interpretation. The user's `old_000013` box adjudication is honored without claiming it settles every endpoint or other old photograph.

Every whole held training photograph keeps **exact PR35 labels in both experiment arms**. This preserves the already established comparison rule. It does not certify those inherited labels, turn the photo into an empty negative, or erase supported review edits from the PDF.

Consequently, the atlas and training candidate have different counts:

| Artifact | Training photos | Training fronts | Validation/development fronts |
| --- | ---: | ---: | ---: |
| Complete r7 review | 137 | 3,788 | 663, exploratory review |
| Fresh PR35-label reference arm | 137 | 3,771 | Original 664, unchanged |
| R7 training candidate | 137 | 3,780 | Original 664, unchanged |

Both arms retain 13 empty training scenes. **79 training photos differ in label bytes**. Against the prepared r6 training candidate, the accepted follow-up changes training labels on `new_000003` and `new_000095`; the other edits are held-photo or development-review changes. In particular, the original `old_000013` training file was already being used, so restoring its review box does not alter training supervision.

Keep the **666-front primary reference**, **663-front historical reference**, and **663-front current development review** distinct. The two 663 counts do not imply equal label bytes. The original 20-photo validation split participated in previous model selection; it is not an unseen final test.

## PR72 result: retain PR35

The [complete PR72 audit](../results/evaluation/pr72-20261119/README.md) verifies the returned r5 archive and freshly infers PR35, the returned control and candidate with identical fixed settings. The unchanged 666-front comparison is:

| Model | Scored misses | Unmatched outputs | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: |
| Retained PR35 | 18 | 80 | 533/584 | 582/584 |
| Fresh r5 control | 20 | 83 | 531/584 | 583/584 |
| PR72 r5 candidate | 30 | 105 | 537/584 | 580/584 |

The candidate gains six five-pixel pair successes against its control but loses ten matches and produces 22 more unmatched outputs. The ten-pixel pair score also falls. The separate historical reference agrees on the worse matching coverage. **PR35 remains selected.**

A scored miss is failure to make a one-to-one box match at IoU 0.50, not proof that no proposal exists. Some strong proposals fail because their extents differ from the fixed reference. Unmatched outputs are not all physically verified background detections. The 584 pair denominator retains eligible missed fronts as failures; the 666 detection denominator includes fronts with zero or one locatable corner. See the saved matching traces rather than interpreting one score alone.

## Run the next notebook

[Open r7 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r7-20261119/notebooks/baomai_front_review_r7.ipynb). Select **Runtime > Change runtime type > GPU**, then **Runtime > Run all**. Return the final results ZIP, including unfavorable results.

The notebook pins an actual committed source revision and the exact [r7 protocol](../experiments/front-review-2026-11-19-r7.json). Both full arms independently initialize from original epoch173: **seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW, learning rate 0.0001**, with the final epoch selected in advance. The short resource trial does not initialize either full run. Completed work is backed up through the established Drive recovery flow.

This tests the accumulated eligible supervision package against the PR35-label control, including whole-photo holds. It is not an isolated ablation of one box or a direct r6-versus-r7 training comparison. R7 has not been trained; no automatic promotion is enabled.

## Verification

```sh
python3 scripts/verify_front_review_r7.py
python3 scripts/check_front_review_r7.py --output /tmp/front-review-r7-check
python3 scripts/build_front_review_r7_pdf.py \
  --output /path/to/cassette-front-review-r7.pdf \
  --audit-output /path/to/r7-pdf-audit.json
```

[Verification records](../results/verification/front-review-r7-20261119/) cover actual two-arm staging, all 157 original image hashes, unchanged original labels, exact PR35 labels on all 44 held photos, unchanged fixed references and the exact `old_000013` restoration. PDF records separately bind all current labels and this pass's field highlighting. Mechanical consistency does not replace physical judgment or demonstrate better unseen-session performance.
