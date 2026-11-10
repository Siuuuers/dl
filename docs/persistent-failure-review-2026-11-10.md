# Persistent misses: review the reference before another training experiment

**Decision, 2026-11-10: keep epoch173, box confidence 0.25 and NMS IoU 0.70.** The seed-43/44 study is complete, and its [published conclusion](box-repeatability-comparison-2026-11-10.md) stands. This follow-up completes original-image review of all 33 persistent reference-object misses. It identifies specific annotation and matching concerns alongside remaining model limitations. It does not introduce a new model, acceptance tolerance or accuracy result.

The most useful next task is a **separately versioned reference-correction proposal**, starting with questionable instance identity and duplicate representations, then supported box-extent issues. Review analogous training annotations independently. Correcting measurement labels and improving a trained model are different operations; neither implies the other succeeded.

## What this review examined

The review starts from [PR #28 at `843c6a6`](https://github.com/Siuuuers/dl/commit/843c6a6f37073f585e61f130e7b17551bea6f5df) and its [33-row queue](../results/evaluation/baomai-box-repeatability-a100-001/failure-review-locations.json). All five evaluated models miss these same reference rows at the fixed primary confidence. The rows span 16 original 1280×720 validation PNGs: 12 rows have two locatable reference corners, 12 have one and nine have none.

Original image and label bytes were checked against the queue's hashes. Review used unannotated originals/crops and then overlays, with neighboring reference rows inspected when identity was in question. A separate assistant independently reviewed all 12 two-corner cases. An independent numerical calculation checked the saved predictions and matching identities. These are assistant review findings, not human or supervisor approval.

The [per-row evidence](persistent-failure-review-2026-11-10.json) preserves original one-based label rows, source hashes, crop coordinates, observations, overlap proposals and numerical diagnostics. Original source images remain linked through their dataset paths. The categories below organize review work; they are not exclusive physical causes or accepted corrections.

| Diagnostic group | Rows | What it means |
| --- | ---: | --- |
| Box-extent disagreement with two locatable corners | 12 | A visually corresponding M0 proposal exists, but its rectangle fails the fixed IoU match |
| Border truncation or thin fragments | 10 | Frame clipping, adjacent-instance confusion and front inclusion need separate inspection |
| Hand or gripper occlusion | 7 | Some visible fragments are missed; some predicted hidden-side corners lie on foreground material |
| Questionable reference-instance identity | 2 | A rectangle straddles two already-labeled fronts without clearly isolating another front |
| Possible duplicate reference representation | 2 | Neighboring reference rows appear to describe the same fronts with different visibility labels |
| **Total** | **33** | The persistent queue is fully reviewed; exact label amendments remain unaccepted |

## 1. A matching miss can contain useful corner predictions

For all 12 priority rows, the selected M0 proposal concerns the same apparent visible cassette front in the image. Its box confidence is **0.851449–0.943355**, but its overlap with the reference rectangle is only **IoU 0.352873–0.487340**, below the fixed 0.50 gate. Each proposal's highest overlap among all reference rows is the intended queued row, and none of these proposals is assigned elsewhere.

Under the unchanged evaluator, each pair contributes one false negative and one extra prediction. The two counters therefore do not necessarily describe two unrelated physical mistakes. The official counts remain valid for the frozen reference and matching rule; the image review explains what some of those counts represent.

The 24 corresponding lower-corner distances are **0.294191–3.946102 original-image pixels**. All are finite, in-frame and above endpoint confidence 0.50. These distances use visually supported associations outside the official matcher. They are **unscored diagnostics**, not recovered true positives. In particular, do not add 12 to the published 521/584 both-corner success count, subtract 12 from the official misses/extras, or claim improved model accuracy.

| Original image | Original row | M0 confidence | IoU | Unscored left / right distance, px |
| --- | ---: | ---: | ---: | --- |
| `new_000054` | 17 | 0.872679 | 0.434291 | 3.946102 / 1.024018 |
| `new_000056` | 1 | 0.851449 | 0.470831 | 3.188521 / 1.637558 |
| `new_000056` | 4 | 0.851982 | 0.471002 | 2.217943 / 0.526463 |
| `new_000086` | 4 | 0.917480 | 0.352873 | 2.470766 / 1.875889 |
| `new_000086` | 15 | 0.937820 | 0.418100 | 1.237857 / 1.598827 |
| `new_000087` | 4 | 0.871986 | 0.475609 | 1.629061 / 1.708967 |
| `new_000088` | 4 | 0.860200 | 0.425020 | 2.456757 / 0.294191 |
| `new_000088` | 15 | 0.907617 | 0.456639 | 1.911018 / 2.429897 |
| `new_000089` | 4 | 0.876138 | 0.441268 | 0.337544 / 2.404751 |
| `new_000089` | 15 | 0.943355 | 0.487340 | 2.271557 / 3.249801 |
| `new_000090` | 4 | 0.923662 | 0.380143 | 1.502704 / 1.489266 |
| `new_000090` | 15 | 0.901161 | 0.451892 | 2.258473 / 3.366923 |

The image-backed interpretation has three parts:

- `new_000054` row 17 and `new_000056` row 1 have reference upper extents reaching into apparent rack/background above the cassette region. M0 also overhangs the lower contour, so copying its box would not be an accepted correction.
- `new_000056` row 4 includes more gray upper material in the reference. The front-versus-roof/rim boundary needs an explicit image-backed decision before assigning replacement coordinates.
- The remaining nine rows in `new_000086`–`new_000090` use thin lower-band references while M0 spans a taller cassette region. The lower-corner correspondences are supported; the exact upper front boundary remains unaccepted. These similar rack views are not nine independent failure modes or certified independent scenes.

Review the existing originals for these opposite extent patterns: [new_000054](../data/images/val/new_000054.png), original row 17, and [new_000090](../data/images/val/new_000090.png), original row 4. The per-row JSON supplies source crop coordinates and exact reference/proposal geometry. Neither rectangle is an adjudicated replacement.

## 2. Some counted misses require checking object identity

### Two rectangles straddle already-labeled fronts

In `new_000058` row 15 and `new_000059` row 16, both reference corners are unavailable. Each central rectangle crosses the seam between two visible stacked faces, already represented by rows 13/14 and 14/15 respectively. The original image does not clearly establish an additional front isolated by the suspicious rectangle.

This is a reference-identity concern. It does not justify automatically deleting the row, filling hidden corners, or teaching the model to produce another box at that location. Adjudicate the neighboring rows as a group and record the instance decision under the [one-front/one-row policy](annotation-policy.md).

### Matching competition in `new_000086`

Inspect rows **47/58** and **48/59** together. Each pair appears to describe the same visible front, but rows 47/48 mark both corners unavailable whereas rows 58/59 include a visible right corner.

Row 59 has a qualifying M0 proposal with confidence **0.811387** and IoU **0.663891**. The matcher assigns that proposal to row 48, where its IoU is higher: **0.849391**. The same row-48 assignment occurs for all five models at all three recorded thresholds. Matching competition is numerically confirmed; duplicate identity and the correct visibility record still require a separately accepted annotation decision.

Inspect the existing originals: [new_000058](../data/images/val/new_000058.png), rows 13–15, and [new_000086](../data/images/val/new_000086.png), rows 47/58 and 48/59. The per-row evidence preserves the geometry and assignments. Overlap alone is insufficient to prove duplicate physical identity; original pixels and neighboring rows supply the review context.

## 3. Real model limitations remain visible

This review does not turn every miss into a label problem. In `new_000008` row 33, a clipped right-edge front and its left corner remain visible, while the maximum-overlap M0 proposal follows a lower neighboring cassette. In `new_000009` row 57, a small right-side front fragment remains beside the gripper, but no primary M0 box overlaps the reference region. These are plausible residual clipped/occluded-target detection failures; their exact reference extents still need adjudication.

The overlays for `new_000009` row 53, `new_000055` row 26, `new_000086` row 59 and `new_000090` row 43 also show an unsupported hidden-side endpoint on the tool or hand. Preserve the distinction between detecting an identifiable front and being able to locate both physical lower corners. A high confidence value does not certify that a corner is visible or geometrically correct.

The remaining review decisions are listed here, with details and source coordinates in the JSON:

| Original image | Original rows | Review finding |
| --- | --- | --- |
| `new_000007` | 7, 14 | Extremely shallow top strips; the overlapping proposals concern taller/neighboring fronts |
| `new_000008` | 1, 33, 36 | Top, right and bottom truncation; row 33 has a visible target fragment; row 36's front identity is uncertain |
| `new_000009` | 8 | Thin top strip versus the fuller row-9 face below |
| `new_000009` | 53, 56, 57 | Gripper occlusion; row 56's identity is unresolved; rows 53/57 have visible fragments |
| `new_000011` | 15 | Top strip versus the row-16 face below |
| `new_000053` | 1 | Tilted top strip versus the fuller row-2 face |
| `new_000055` | 26 | Hand occlusion and a very short reference extent; hidden-side prediction unsupported |
| `new_000057` | 1 | Top strip versus the fuller row-2 face |
| `new_000058` | 15 | Questionable rectangle across two already-labeled faces |
| `new_000059` | 1, 16 | Top strip plus the repeated questionable central rectangle |
| `new_000086` | 57 | Hand-covered fragment; adjacent proposal does not establish the hidden target |
| `new_000086` | 58, 59 | Possible duplicate representations of rows 47/48; confirmed matching competition for row 59 |
| `new_000088` | 57 | Similar hand-covered fragment; related-view correlation is unresolved |
| `new_000090` | 43 | One exposed corner, one hand-hidden corner; extent mismatch and unsupported hidden-side prediction |

For zero-IoU rows, the queue's numerically selected maximum-overlap proposal is unrelated. Its confidence or corner coordinates must not be interpreted as predictions for that reference instance.

## 4. What can improve without collecting new images?

**First qualify the measurements.** Prepare a reference amendment proposal for the four identity/duplication concerns and the box-extent cases above. Preserve original row identities, before/after text and reasons. Keep unresolved front/roof/rim judgments explicit. Accepting a corrected reference requires a separate version and comparison of every model against the same corrected bytes, with old results retained. A score change from better reference labels is a measurement change, not a trained-model improvement.

**Then review the corresponding training failure modes.** Search the existing training images for upper bounds extending into rack/background, inconsistent thin-band versus visible-front boxes, and duplicate rows with inconsistent missing-corner states. Correct only image-supported cases under the existing target definition. Keep the earlier eleven-row deduplication and 16-edge box candidates separately identified; the repeated study does not support blindly rerunning either intervention until a favorable seed appears.

**Use any training intervention as a testable hypothesis.** Freeze a coherent label change before comparing an unchanged-label control and corrected-label run under matched settings. Keep architecture, initialization and checkpoint choice fixed unless the diagnosed failure justifies changing them. The current review supplies no evidence that a larger network, more epochs or broader augmentation is the necessary next change.

**Improve use while preserving uncertainty.** The folder predictor and browser editor already support the current task. Human checking can catch visible misses and unsupported corners. Any unattended consumer needs an explicit output/abstention policy and acceptance limits; the recorded confidence values cannot provide those guarantees by themselves.

The 320 additional upstream images remain an existing-data opportunity, but the [three-image pilot](additional-annotation-pilot-2026-11-07.md) is quarantined for unresolved annotation fields. Prior use and related frames also prevent treating those images as automatically independent final-test evidence. Reusing development images can improve supervision; it cannot manufacture an untouched test population.

## 5. Verification and limits

- The published evaluation capsule is 2,161,801 bytes with SHA-256 `e4d8c586d2553964bf09b07ffdfa2a1d26326488435f3991375b0cf25c49e0c3`. Its CRC and safe paths were checked, and all 109 payload hashes/sizes match the inventory.
- The queue hash is `f583fa19f84bbd12ba8d9527985291e50d09a39b90319c935ab57c0fb590266d`. Source images and labels for all 33 rows match their recorded hashes. All 12 evaluator source hashes in the parent verification record match this checkout.
- M0 saved predictions have SHA-256 `7b9edb88ca507975bf475d81b66db5b726d42c4ebdc653bf6883e7069612c5a8`. Independent parsing and greedy matching reproduce the priority-image assignments and the row-59 competition. The model hash recorded by those predictions is the original epoch173 identity; this review did not load or re-hash model-weight bytes.
- This work uses saved predictions. No training, fresh inference or full benchmark replay is claimed. The parent study's already completed replay remains in its [verification record](../results/evaluation/baomai-box-repeatability-a100-001/verification.json).
- No reference or training labels, model defaults, scoring rules or published score tables are amended here. The diagnostic categories do not imply 33 defective annotations or 33 independent physical failures. Exact replacement bounds and duplicate deletions remain unaccepted.
- Reused development images, unapproved application tolerances, hidden-corner behavior and possible frame correlation continue to limit readiness claims. The supported use remains human-reviewed localization of supplied still images.

The next review should start from these exact rows and images. It should produce a small, reviewable annotation proposal, not repeat the completed seed experiment or rebuild the existing tools.
