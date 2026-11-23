# R24 model failures: complete inventory for R25 preparation

This ledger replays the **saved PR99 R24 candidate predictions** against the **unchanged accepted R24 exploratory reference**. It runs no model inference or training and makes no annotation change. All rows below are **one-based R24 label rows**; prediction indices are **zero-based**. The [companion JSON](model-failure-inventory.json) retains full precision, source hashes, every selected prediction and the complete 57-other-output list.

**R25 annotation amendments are complete and its notebook is being pinned.** Sections 1–7 preserve the completed R24-reference inventory. Section 8 separately scores the unchanged saved models against the finalized R25 reference; it is not an R25-trained-model result. See the [explanation and next-step report](../../../docs/model-improvement-after-r24.md).

## 1. Counts and scoring meaning

| Quantity | R24 candidate |
| --- | ---: |
| Reference fronts / saved outputs | 662 / 736 |
| Matched / missed targets | 646 / 16 |
| Misses with zero / one / two locatable corners | 7 / 9 / 0 |
| Both-corner targets returned | 584 / 584 |
| Both corners within 5 px / 10 px | 557 / 584; 583 / 584 |
| Failed 5 px pairs: geometric / confidence | 27 / 0 |
| Other returned one-corner endpoints above 5 px | 10 |
| Matched locatable endpoints withheld by confidence | 1 |
| Duplicate / other unmatched outputs | 33 / 57 |
| Outputs on the three empty images | 0 |

A target first needs a same-class retained box with IoU at least 0.50 under the confidence-ordered assignment. A locatable endpoint also needs confidence at least 0.50 and a finite coordinate inside the image. The 5 px pair rule requires **both** endpoint distances to pass. These are separate tests. Operational inference remains 1280, box confidence 0.13 and NMS 0.70.

The 27 failed pairs contain 28 returned endpoints above 5 px. Together with the 10 one-corner errors, that accounts for all 38 returned endpoints outside 5 px. The nine locatable endpoints on unmatched targets plus the single confidence-withheld endpoint account for all 10 missing locatable endpoints. The model also returns 65 of 75 unavailable reference corners on matched objects; their physical coordinate errors cannot be measured from these references.

## 2. Every missed target: all 16

Ten targets have some retained overlap below IoU 0.50; six have none. No qualifying retained box is lost to a competing assignment. An incidental overlap with a neighboring front is not an associated detection. With only the saved filtered outputs, absence cannot be attributed specifically to confidence, NMS or decoded geometry.

| R24 target | Locatable corners | Best IoU | Best overlapping output / box confidence | Original-pixel pattern |
| --- | ---: | ---: | --- | --- |
| [new_000008:33](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L33) | 1 | 0.074066 | 11 / 0.922946 | Right border clips a front |
| [new_000008:34](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L34) | 1 | 0.065351 | 29 / 0.764837 | Right border clips a front |
| [new_000008:36](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L36) | 0 | 0.000000 | None | Bottom-border fragment |
| [new_000009:55](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000009.txt#L55) | 0 | 0.000000 | None | Small fragment in jaw gap |
| [new_000009:56](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000009.txt#L56) | 1 | 0.000000 | None | Thin strip beside jaw |
| [new_000009:57](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000009.txt#L57) | 0 | 0.000000 | None | Small fragment in jaw gap |
| [new_000010:14](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000010.txt#L14) | 0 | 0.385671 | 35 / 0.165105 | Jaw gap; available proposal extends into occlusion |
| [new_000010:33](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000010.txt#L33) | 0 | 0.220174 | 31 / 0.247091 | Jaw gap; available proposal extends into occlusion |
| [new_000010:34](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000010.txt#L34) | 0 | 0.007614 | 31 / 0.247091 | Jaw gap; available proposal extends into occlusion |
| [new_000011:55](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000011.txt#L55) | 1 | 0.482170 | 55 / 0.786640 | Strip beside jaw; overwide retained box |
| [new_000011:56](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000011.txt#L56) | 1 | 0.453501 | 62 / 0.261832 | Strip beside jaw; overwide retained box |
| [new_000011:57](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000011.txt#L57) | 1 | 0.336395 | 60 / 0.337293 | Strip beside jaw; overwide retained box |
| [new_000011:60](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000011.txt#L60) | 0 | 0.072347 | 67 / 0.179956 | Small fragment in jaw gap |
| [new_000053:1](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000053.txt#L1) | 1 | 0.314293 | 21 / 0.137279 | Thin top-border fragment |
| [new_000086:57](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000086.txt#L57) | 1 | 0.000000 | None | 8–9 px strip beside thumb |
| [new_000088:57](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000088.txt#L57) | 1 | 0.000000 | None | 8–9 px strip beside thumb |

For `011:55–57`, the unmatched alternatives have left-corner distances approximately **1.96 / 3.36 / 1.43 px**, but their boxes extend beyond the visible fragment. Those distances diagnose a box-extent problem; they do not create official corner successes. `008:33`, `008:34` and `009:55` are matched by PR35 and the R24 control but missed by both R23 and R24 candidates on this same reference. They remain useful preservation cases.

The [completed sixteen-case diagnosis](../../evaluation/pr99-r24-20261123/partial-front-failure-queue.md) contains every original-image link, accepted box, original-pixel observation and full R24 atlas page.

## 3. Every two-corner pair outside 5 px: all 27

All 27 have a matched box and **both endpoints above the confidence cutoff**. Every failure below is geometric relative to accepted R24 points. Distances and endpoint confidences are rounded to six decimals for readability; the JSON retains the original floating-point values. L/R mean the existing left/right endpoint slots.

| R24 target | Output index | Left error, px | Left confidence | Right error, px | Right confidence | Failed side |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [new_000008:4](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L4) | 15 | 5.777379 | 0.961027 | 2.747650 | 0.960185 | L |
| [new_000008:5](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L5) | 23 | 4.699471 | 0.966889 | 8.954227 | 0.947148 | R |
| [new_000008:6](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L6) | 8 | 6.657898 | 0.950726 | 3.910485 | 0.947057 | L |
| [new_000008:10](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L10) | 27 | 1.475107 | 0.906842 | 12.502088 | 0.908112 | R |
| [new_000008:11](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L11) | 28 | 5.787298 | 0.930279 | 6.661615 | 0.924092 | L/R |
| [new_000008:18](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L18) | 6 | 6.598788 | 0.950847 | 0.992791 | 0.964221 | L |
| [new_000008:19](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L19) | 10 | 8.190125 | 0.958982 | 0.936715 | 0.955346 | L |
| [new_000008:23](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L23) | 19 | 5.575502 | 0.945732 | 1.056571 | 0.938991 | L |
| [new_000008:29](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L29) | 4 | 7.101177 | 0.956625 | 2.439584 | 0.954189 | L |
| [new_000010:26](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000010.txt#L26) | 1 | 2.885728 | 0.955821 | 5.537635 | 0.964796 | R |
| [new_000011:4](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000011.txt#L4) | 53 | 2.506448 | 0.898420 | 5.636536 | 0.904055 | R |
| [new_000011:49](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000011.txt#L49) | 1 | 1.263643 | 0.907106 | 5.696460 | 0.922254 | R |
| [new_000053:15](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000053.txt#L15) | 0 | 1.797996 | 0.945823 | 5.061789 | 0.972655 | R |
| [new_000055:21](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000055.txt#L21) | 22 | 2.995931 | 0.889220 | 6.218010 | 0.895591 | R |
| [new_000056:22](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000056.txt#L22) | 12 | 5.309900 | 0.950937 | 1.368805 | 0.945222 | L |
| [new_000058:1](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000058.txt#L1) | 17 | 5.634599 | 0.952325 | 3.483013 | 0.944848 | L |
| [new_000058:4](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000058.txt#L4) | 1 | 5.127560 | 0.943910 | 3.556051 | 0.954165 | L |
| [new_000058:7](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000058.txt#L7) | 4 | 5.151527 | 0.944713 | 3.056980 | 0.941387 | L |
| [new_000058:10](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000058.txt#L10) | 6 | 5.995879 | 0.920586 | 2.460443 | 0.956401 | L |
| [new_000059:1](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000059.txt#L1) | 15 | 4.244804 | 0.955862 | 5.186857 | 0.947773 | R |
| [new_000059:2](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000059.txt#L2) | 16 | 4.416121 | 0.948213 | 5.094034 | 0.952627 | R |
| [new_000059:7](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000059.txt#L7) | 2 | 5.965110 | 0.937088 | 1.646902 | 0.926794 | L |
| [new_000087:53](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000087.txt#L53) | 12 | 0.996665 | 0.963996 | 5.110117 | 0.962883 | R |
| [new_000088:52](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000088.txt#L52) | 51 | 1.180651 | 0.840121 | 5.410488 | 0.865304 | R |
| [new_000090:19](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000090.txt#L19) | 9 | 1.072037 | 0.940114 | 5.108484 | 0.954597 | R |
| [new_000090:44](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000090.txt#L44) | 40 | 3.926448 | 0.946006 | 6.266003 | 0.942004 | R |
| [new_000090:45](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000090.txt#L45) | 28 | 1.845895 | 0.940185 | 6.084854 | 0.944805 | R |

**The only failed 10 px pair is `new_000008:10`:** its right error is **12.502088 px** with endpoint confidence **0.908112**; its left error is 1.475107 px. A high endpoint score therefore does not certify pixel accuracy. Original crops of all 27 cases were inspected; soft/rounded front transitions and adjacent lips/rack edges are visible in this set, but this inspection does not prove a single training cause or authorize moving the reference toward the model.

The 27 pairs span 11 images. Nine are in photo008; four in 058; three each in 059 and 090; two in 011; one each in 010, 053, 055, 056, 087 and 088.

## 4. Additional one-corner errors and the confidence failure

These cases are excluded from the two-corner pair denominator because only one corner is locatable. They still matter for practical point placement.

| R24 target | Locatable side | Error, px | Endpoint confidence | Classification |
| --- | --- | ---: | ---: | --- |
| [new_000008:1](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L1) | right | 12.262111 | 0.382982 | Below 0.50; distance is diagnostic only |
| [new_000008:8](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L8) | left | 6.080248 | 0.918074 | Geometric error; endpoint returned |
| [new_000008:35](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L35) | left | 5.896019 | 0.933212 | Geometric error; endpoint returned |
| [new_000009:49](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000009.txt#L49) | left | 6.539848 | 0.740473 | Geometric error; endpoint returned |
| [new_000009:50](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000009.txt#L50) | left | 6.644929 | 0.708506 | Geometric error; endpoint returned |
| [new_000009:51](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000009.txt#L51) | left | 7.192536 | 0.627399 | Geometric error; endpoint returned |
| [new_000009:52](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000009.txt#L52) | left | 6.252726 | 0.546573 | Geometric error; endpoint returned |
| [new_000055:23](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000055.txt#L23) | left | 5.802410 | 0.938737 | Geometric error; endpoint returned |
| [new_000056:12](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000056.txt#L12) | left | 5.519084 | 0.935578 | Geometric error; endpoint returned |
| [new_000058:18](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000058.txt#L18) | right | 5.657611 | 0.949257 | Geometric error; endpoint returned |
| [new_000089:50](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000089.txt#L50) | left | 5.922174 | 0.822223 | Geometric error; endpoint returned |

`008:1` is already matched, but its only locatable (right) endpoint has confidence **0.382982**, so the scorer correctly treats it as unavailable. Its saved coordinate is also **12.262111 px** from the reference. Lowering the endpoint cutoff would expose that inaccurate coordinate; it would not make it a 10 px success.

## 5. Every duplicate output: all 33

A duplicate here is an **unmatched output that overlaps an already matched target at IoU ≥0.50**. It counts as an extra output, not as a missed target. The target association uses the highest qualifying reference IoU (then lowest row number); the JSON records every qualifying target if more than one exists.

| Image | Already matched R24 row | Duplicate output index | Box confidence | IoU with that target |
| --- | ---: | ---: | ---: | ---: |
| `new_000007` | 1 | 49 | 0.789084 | 0.886965 |
| `new_000007` | 4 | 50 | 0.781431 | 0.667313 |
| `new_000007` | 10 | 53 | 0.704320 | 0.579516 |
| `new_000007` | 7 | 54 | 0.679748 | 0.766367 |
| `new_000007` | 47 | 57 | 0.184099 | 0.611798 |
| `new_000007` | 47 | 58 | 0.159165 | 0.511474 |
| `new_000008` | 11 | 31 | 0.724068 | 0.942387 |
| `new_000008` | 3 | 39 | 0.141603 | 0.797663 |
| `new_000009` | 12 | 53 | 0.713315 | 0.635826 |
| `new_000009` | 8 | 56 | 0.580746 | 0.640870 |
| `new_000009` | 51 | 59 | 0.412622 | 0.641522 |
| `new_000009` | 50 | 62 | 0.388139 | 0.640760 |
| `new_000009` | 50 | 66 | 0.343982 | 0.613001 |
| `new_000009` | 52 | 69 | 0.220602 | 0.647762 |
| `new_000010` | 29 | 33 | 0.199249 | 0.502774 |
| `new_000011` | 4 | 54 | 0.803736 | 0.734684 |
| `new_000011` | 58 | 67 | 0.179956 | 0.664661 |
| `new_000055` | 23 | 27 | 0.138049 | 0.869572 |
| `new_000056` | 14 | 23 | 0.162368 | 0.754838 |
| `new_000059` | 18 | 18 | 0.444430 | 0.681145 |
| `new_000059` | 18 | 19 | 0.348846 | 0.631560 |
| `new_000087` | 48 | 56 | 0.622129 | 0.829321 |
| `new_000087` | 43 | 58 | 0.237452 | 0.590186 |
| `new_000088` | 51 | 56 | 0.675809 | 0.623260 |
| `new_000088` | 48 | 57 | 0.314204 | 0.550374 |
| `new_000089` | 42 | 52 | 0.656355 | 0.746149 |
| `new_000089` | 41 | 59 | 0.272220 | 0.860355 |
| `new_000089` | 47 | 64 | 0.177849 | 0.671904 |
| `new_000090` | 42 | 48 | 0.745565 | 0.736282 |
| `new_000090` | 40 | 49 | 0.610778 | 0.548473 |
| `new_000090` | 41 | 54 | 0.332797 | 0.769543 |
| `new_000090` | 50 | 56 | 0.291955 | 0.511729 |
| `new_000090` | 50 | 57 | 0.203727 | 0.592109 |

Duplicates occur in 12 images: 007 (6), 008 (2), 009 (6), 010 (1), 011 (2), 055 (1), 056 (1), 059 (2), 087 (2), 088 (2), 089 (3), 090 (5). Counts refer to outputs; repeated rows can have more than one extra output.

At `011:4`, the canonical output 53 has errors **2.506448 / 5.636536 px**; duplicate output 54 has **1.391504 / 3.633758 px** and slightly lower box confidence, **0.803736 versus 0.812371**. This is a saved-output ranking example. Selecting the closer alternative with reference labels would be an oracle operation, so no score or runtime behavior is changed. Stronger NMS could also keep the higher-ranked, less accurate pair. The other failed-pair duplicate case, `008:11`, has a substantially worse alternative (13.297221 / 28.852695 px).

The other **57** unmatched outputs are also recorded in JSON. Some are partial-front extent disagreements, so they must not all be called nonexistent objects. Two current, original-pixel background examples are `089` output **51**, on the bright top rack support (confidence **0.686437**, no reference overlap), and `090` output **50**, on the lower-right keyboard/bezel region (confidence **0.501085**, no reference overlap). Both use zero-based prediction IDs; background outputs have no target row.

## 6. Requested annotation reviews are a different list

| User-requested R25 review | Split | What the completed R24 model evidence says |
| --- | --- | --- |
| `008:35` | Development | Matched; its only locatable left endpoint is 5.896019 px away on R24. Its completed R25 physical amendment is accounted for separately in section 8. |
| `010:18` | Development | Already passes R24 with left/right errors 2.751796 / 0.597099 px. A score pass does not settle the requested physical corner placement. |
| `010:19` | Development | Already passes R24 with left/right errors 2.507165 / 0.605535 px. Same distinction. |
| `012:3`, `012:28` | Training | No PR99 validation prediction exists for these rows. Their pixel-based amendments can affect training if admitted by the existing eligibility rules. |

The supplied good-label standards `008:18–21` and `008:28–32` are visual annotation references, not model-success claims. The model still misses 5 px on the left at standards `008:18`, `008:19` and `008:29` (6.598788, 8.190125 and 7.101177 px). All other supplied standard pairs pass that radius. Model predictions never decide which physical label should be accepted.

## 7. Verification and preserved scope

The existing scorer was replayed against all 20 saved arrays. Every matching trace, the entire aggregate detection record and the entire corner-quality record equal the preserved PR99 report. The 16 miss identities equal the completed diagnostic queue; all 27 failed pairs and all 33 duplicates are accounted for. Original-image and accepted-reference hashes match. No new inference, training, threshold change or accepted-label change was made. Later R25 results need a separately named reference and matched comparison; this R24 inventory remains fixed.

## 8. Current baseline on the completed R25 reference

The R25 annotation amendment is complete: eight left points across eight rows in three images. Four belong to development (`008:35`, `010:18/19/20`); four belong to training (`012:1/2/3/28`). The ambiguous right point at `012:28` is retained. The [R25 review](../../../docs/front-review-r25.md) records the physical decisions. Its [reference manifest](../../../data/reviews/fronts-2026-11-23-r25/manifest.json) is pinned to SHA-256 `de80c21ed4b148efe6446e66b8a15f75c3b6363daf9c03d0b88a29e187483517`.

**These are the same previously saved model outputs, scored against the repaired reference. No R25 model has been trained or evaluated here.** The R25 notebook is being pinned for the new matched comparison. This subsection does not rewrite the original R24 tables above.

| Unchanged saved model | R25 matches / misses | Unmatched outputs | Both within 5 px / 584 | Both within 10 px / 584 |
| --- | ---: | ---: | ---: | ---: |
| PR35 | 625 / 37 | 103 | 543 | 573 |
| R24 control | 624 / 38 | 105 | 539 | 573 |
| R23 candidate | 645 / 17 | 95 | 557 | 583 |
| R24 candidate | 646 / 16 | 90 | 557 | 583 |

All target associations, misses, extras and duplicate identities stay unchanged because the reference boxes did not change. PR35 and the control each lose one five-pixel pair at `010:20` relative to their R24-reference scores (544→543 and 540→539). R23 and R24 remain at 557. This is entirely a reference-coordinate effect.

### The four changed development rows, with the unchanged R24 candidate

| Target | R24 left error, px | R25 left error, px | Right error, px | Left / right confidence | Status on R25 |
| --- | ---: | ---: | ---: | --- | --- |
| `new_000008:35` | 5.896019 | 1.871916 | Unlocatable | 0.933212 / 0.944087 | Locatable left now ≤5 px; right remains unlocatable |
| `new_000010:18` | 2.751796 | 2.473627 | 0.597099 | 0.956494 / 0.960262 | Both endpoints remain ≤5 px |
| `new_000010:19` | 2.507165 | 0.665292 | 0.605535 | 0.938974 / 0.930300 | Both endpoints remain ≤5 px |
| `new_000010:20` | 2.851306 | 2.257313 | 1.622320 | 0.945606 / 0.948295 | Both endpoints remain ≤5 px |

### Which rows are currently poor under R25?

For the unchanged R24 candidate, **all 16 misses in section 2, all 27 five-pixel pair failures in section 3, all their endpoint errors, and all 33 duplicates in section 5 remain current**. The sole ten-pixel pair failure remains `008:10`. The nine currently returned one-corner errors above five pixels are **008:8; 009:49,50,51,52; 055:23; 056:12; 058:18; 089:50**. Their distances and confidences are unchanged from section 4. The separate low-confidence case remains **008:1**.

**008:35 is no longer a five-pixel endpoint failure under the corrected reference.** Its unchanged predicted left coordinate is now 1.871916 px from the accepted point rather than 5.896019 px. The model did not move or learn; the reference was repaired. The four amended development rows and their before/after errors for all four models are preserved in JSON.

Verification binds all R25 validation label files to the pinned manifest, confirms only the four declared left-point changes, and confirms unchanged boxes, classes, visibility, counts and all 80 saved-model matching traces. No inference, training or label mutation was performed by this scoring step.

## 9. Where selected PR35 still fails on the current R25 reference

PR35 remains selected under the frozen 666-front primary decision. On the **separate current R25 reference**, its saved outputs match 625/662 fronts, miss **37**, and place both corners within 5 px on **543/584** eligible fronts; the saved R24 candidate reaches **557/584** with 16 misses. These are existing checkpoints, not an R25-trained result. R25 has not been trained. The frozen-reference counts and row identities must not be substituted for the current rows below.

The table accounts for **all 37 PR35 misses and all 41 failed two-corner targets** on R25. Seven of those 41 fail through unmatched boxes and 34 through geometric error; the missed and failed-pair columns overlap and must not be added together. Rows are one-based current R25 annotation rows. A dash means no failure in that column.

| Photograph | PR35 missed rows | PR35 two-corner targets failing 5 px |
| --- | --- | --- |
| `new_000007` | 1, 8, 9, 14, 15, 30 | 1, 7, 8, 9, 13, 14, 15, 16, 30 |
| `new_000008` | 1, 5, 36 | 4, 5, 6, 18, 19, 21, 25, 27, 28, 29, 30, 31 |
| `new_000009` | 50, 51, 52, 56, 57 | 12 |
| `new_000010` | 14, 24, 31, 32, 33, 34 | 15, 20, 26, 28 |
| `new_000011` | 55, 56, 57, 58, 59, 60 | 4 |
| `new_000053` | 1 | — |
| `new_000055` | — | 21 |
| `new_000056` | 16 | 4, 18, 19, 22 |
| `new_000058` | — | 4 |
| `new_000059` | — | 2, 6, 11 |
| `new_000086` | 57 | — |
| `new_000087` | — | 53 |
| `new_000088` | 57 | — |
| `new_000089` | 42, 43 | 34 |
| `new_000090` | 39, 40, 41, 42, 43 | 44, 45, 46 |

The largest miss concentrations are photos 007, 010 and 011, with six each. Photos 008 and 007 contain 12 and nine failed pairs, respectively. Four matched two-corner targets have a returned endpoint beyond 10 px:

| Current target | PR35 endpoint error beyond 10 px |
| --- | --- |
| `007:13` | Left 14.596 px; right 14.356 px |
| `008:18` | Left 10.286 px |
| `008:19` | Left 13.040 px |
| `008:29` | Left 10.559 px |

The three listed photo008 left endpoints have confidence 0.944–0.957 despite these errors. Accepted annotation standards remain physical label references; high model confidence and annotation correctness do not guarantee accurate predictions.

**Thirteen current targets are missed by PR35, the R24 control and the R24 candidate:** `008:36`; `009:56,57`; `010:14,33,34`; `011:55,56,57,60`; `053:1`; `086:57`; `088:57`. These shared cases concentrate on clipped borders and narrow occluded fragments. Conversely, PR35 matches `008:33,34` and `009:55`, which the R24 candidate misses; retain these three as preservation cases.

Sources: [preserved comparison](../../evaluation/pr99-r24-20261123/fixed-comparison.json), `references.current_r24_exploratory.models.*.scenes`, supplies missed identities; section 8 verifies unchanged R25 box associations. The [existing inventory JSON](model-failure-inventory.json), `finalized_r25_reference_baseline.models.pr35`, supplies all current pair-failure identities and endpoint errors. This section summarizes those saved records without new inference, scoring or annotation changes.
