# Round 8: cassette extrema and physical lower corners

Date: **19 November 2026, Asia/Hong_Kong**. **PR35 remains selected.** This pass amends annotations after the five marked examples and audits the returned [PR74 r7 experiment](https://github.com/Siuuuers/dl/pull/74). The next notebook is prepared after the amendment; r8 GPU training has not been run.

## What changed

All **157 original photographs** and their inherited rows received a fresh visual pass. Changed contours and uncertain regions were examined in enlarged original-pixel views. The complete review makes **1,618 row replacements across 93 photographs**, with **0 additions and 0 deletions**. Within replacements, **1,573 box fields, 482 left-point fields and 533 right-point fields** change; these categories overlap.

Every changed box records four original-pixel extrema. The target is the visible front and its own frontal rim. Continuous rails are recognized by their connection through gaps into the mesh. A cassette rim turns and ends with its own front. Image color alone cannot distinguish them. Some boxes contract to remove rack margins; others expand to recover an omitted own rim. The two landmarks follow the physical lower-front turns independently of the AABB.

The [boundary guide](annotation-boundaries-r8.md) gives the explicit interpretation. The one-class/two-corner task and unchanged original PNGs are preserved. Inspection filters do not become training images. Four review witnesses do not add four training keypoints.

| Marked source image | Main evidence used in this pass |
| --- | --- |
| `new_000001` | Side rails continue into the mesh. Row22's top and right bounds expand while its bottom tightens; the right landmark follows the actual lower turn. |
| `new_000003` | The transverse support and continuous dividers are separate from the cassette fronts. Prior visible-fragment identities remain stable. |
| `new_000005` | Follow the dark divider outside the pale front through the stack. A physical lower turn is separate from the rectangle corner. |
| `new_000007` | Preserve sloping own rims and the unavoidable background wedges inside their axis-aligned envelope. |
| `new_000008` | Interpret the screenshot's blue/yellow/orange hints using original pixels; inspect every front's own upper and lower contour. |

The [exact operations](../results/review/front-review-20261119-r8/operations.json), [row map](../results/review/front-review-20261119-r8/row-maps.json), [coverage ledger](../results/review/front-review-20261119-r8/request-coverage.json) and [case ledger](../results/review/front-review-20261119-r8/review-status.json) preserve the before/after identities, field changes, reasons and evidence. The inspection-artifact inventory records the rendered views used; the original image hashes and immutable label versions are the reproducible coordinate authority.

## Complete review, explicit uncertainty

The displayed review contains **3,788 training fronts and 663 development fronts**. It retains 137 training and 20 development photographs, including 13 empty training scenes and 3 empty development scenes. These empty-rack photographs were reviewed; uncertainty is never converted into an empty label.

There remain **89 unresolved cases**, involving **64 training photos and 6 development photos**. All 31 old-series training photographs retain their holds. The broad pale lower flange is connected to the cassette; the remaining landmark question is its outer turn versus the front-wall junction. That distinction is recorded explicitly instead of calling the flange rack. `old_000013`'s complete eight-row review file remains exactly the user-confirmed original bytes.

Some gripper-adjacent strips or gray projections remain ambiguous even when one neighboring rail is clear. Supported changes do not silently certify the other fields. Nominal pixel coordinates and many decimal places in normalized text do not establish subpixel physical accuracy. **`full_dataset_certified` remains false.**

## PR74 result: retain PR35

The [complete PR74 audit](../results/evaluation/pr74-20261119/README.md) verifies the exact returned archive, source and notebook pins, both completed 30-epoch A100 arms, all indexed members and fresh same-runtime CPU inference. Fixed scoring and saved-prediction replay are retained in [fixed-comparison.json](../results/evaluation/pr74-20261119/fixed-comparison.json).

| Primary reference: 666 fronts | Matches | Scored misses | Unmatched outputs | Both corners ≤5 px | Both ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 648 | 18 | 80 | 533/584 | 582/584 |
| Fresh r7 control | 648 | 18 | 80 | 533/584 | 582/584 |
| PR74 r7 candidate | 639 | 27 | 95 | 537/584 | 580/584 |

The candidate adds nine scored misses and 15 unmatched outputs against its fresh control. It gains four five-pixel pair successes but loses two ten-pixel successes. The required historical 663-front comparison also worsens: control/candidate matches 648/640, misses 15/23 and unmatched outputs 80/94. These are separate frozen references, with no retroactive relabeling.

A scored miss is a failure to get a one-to-one box match at IoU 0.50. `new_000010:15` retains a strong candidate proposal at IoU 0.4973, just below the cutoff. `new_000057:11` has a strong proposal while the frozen box extends into rack below the visible front. `new_000009:56` has no overlapping output in the saved post-threshold set. These observations distinguish extent disagreement from absent retained detections; they do not rewrite the score or establish that the candidate is better overall.

PR35 and the fresh r7 control have exactly equal ordered decoded prediction arrays on all 20 fixed images at this operating point, despite different checkpoint byte hashes. Equality on these images does not establish equality elsewhere. The 584 pair denominator includes eligible missed fronts; it is not complete-scene accuracy. Unmatched outputs are not all independently adjudicated background errors.

## The next notebook

[Open the r8 notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r8-20261119/notebooks/baomai_front_review_r8.ipynb) · [Notebook file](../notebooks/baomai_front_review_r8.ipynb) · [Protocol](../experiments/front-review-2026-11-19-r8.json)

Select a GPU, then **Run all**, and return the final ZIP. The notebook pins a real source commit and the exact protocol. It downloads/verifies inputs, runs a four-batch resource check, trains both full arms independently from the original epoch173 checkpoint, and saves complete results with Drive backup in Colab.

- Seed 46; 30 epochs; batch 4; input 1280; FP32; AdamW; learning rate 0.0001; final-epoch selection fixed in advance.
- Candidate/control supervision: **3,770/3,771 training fronts**, with label bytes differing on **59 training photographs**.
- Every one of the **64 held training photos** retains exact PR35 labels in both arms. Supported review edits on a held photo are visible in the atlas but excluded from the treatment difference. The photo remains labeled.
- Both arms retain all 137 training photographs, 13 empty training scenes and original 20-photo/664-front training validation.
- Fixed 666-target primary and 663-target historical references stay unchanged. The current development review is independently versioned and exploratory.

This tests the eligible accumulated supervision package against PR35 labels. It does not isolate a single correction. No seed, epoch, threshold or NMS search is introduced. No training or automatic promotion is performed during preparation. Repeatedly used development scenes cannot establish unseen-session or robot-positioning reliability.

## PDF and verification

The separately delivered **cassette-front-review-r8.pdf** contains all 157 full-photo pages, a physical definition, a four-extrema example, the PR74 result, a linked index and enlarged original/before/current details. **Magenta marks only field changes from r7; cyan denotes an unchanged field; amber records unresolved fields.** The original image underlay is unchanged. See the [PDF delivery receipt](../results/review/front-review-20261119-r8/pdf-delivery.json) and [rendered-page QA](../results/review/front-review-20261119-r8/pdf-visual-qa.json).

`verify_front_review_r8.py` replays exact operations and checks the schema, point containment, visibility slots, source hashes, all 157 coverage entries, four-extrema metadata and original `old_000013` bytes. `check_front_review_r8.py` stages the actual comparison arms and verifies all originals, held-photo equality, frozen validation and both references. The notebook's exact input preflight is also run in a clean checkout of its published source. These mechanical checks bind the saved decisions; they do not independently establish material ownership.

Review manifest SHA-256: `07721c3e5ded57ae289650b9e0a96d424a793d595d769f42809c129e95061ca2`. Protocol SHA-256: `96426b9d0ba7a07b902698e320a82449c051c6790c4bac25dda709d2f181b7d3`. Detailed [preparation verification](../results/verification/front-review-r8-20261119/verification.json) is separate from future GPU results.
