# Baomai annotation policy

Policy ID: **`baomai-visible-front-v1`**. Adopted as the working convention for new and revised annotations in `Siuuuers/dl` on **2026-11-06**, under the user's authorization to research upstream history and adopt reasonable rules. This records project decisions; it does not attribute the added details to the supervisor or certify every frozen annotation as compliant.

## Source priority and adopted decisions

| Source | What we retain | What belongs to a different task |
| --- | --- | --- |
| [Current baomai data README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md) | One `cassette_front` class; two actual lower-front corners ordered by image X; independent visibility; canonical `0 0 0` for an unreliable location; genuine empty scenes. | It does not explicitly settle box extent or a minimum visible-area requirement. |
| [Historical baomai editor instructions](https://github.com/an94-doll/embedding_vision/blob/6f7112a553e1759fc755bda1f18ee56cb34dd315/detect/data/data_last/picture/work/ANNOTATION_EDITOR.md) | The visible front-face box convention and original-image review workflow provide the basis for the local box rule below. | The older frame83/front-lip point convention does not replace the current lower-front target. |
| [yolo-pose annotation instructions](https://github.com/an94-doll/embedding_vision/blob/72a99dafa1b14bfabd101e639b9ddb5f5aa38d52/annotations/README.md) and [schema](https://github.com/an94-doll/embedding_vision/blob/72a99dafa1b14bfabd101e639b9ddb5f5aa38d52/annotations/pose_schema.yaml) | General provenance and review practices only. | Its front/back classes, five object-relative points, geometric midpoint, rounded-corner extrapolations and whole-object boxes are a different task. Do not mix those labels into baomai. |
| [grasp-vision adaptations](https://github.com/an94-doll/embedding_vision/blob/9a6bf2432582af517d789c94eab2ac27e03a304b/provenance/ADAPTATIONS.md) | Provenance confirms that this is a separately adapted project. | Basket/white-box models and their schemas do not redefine the requested epoch173 cassette task. |

The three branch heads were checked remotely on the adoption date. The current baomai source remains authoritative for the target feature. The visible-box and inclusion details below are explicit local decisions informed by the older editor, compatible current examples and the completed image review.

## 1. One identifiable visible front, one row

Annotate each uniquely identifiable visible cassette front exactly once. The target is the physical white front surface; illumination may make it look gray or dark. Do not classify instances using a brightness cutoff.

An instance can remain even if neither target corner is locatable, provided the original image supports its identity and a positive-area visible front extent. Point availability and object presence are separate decisions. The frozen reference already contains 121 training and 21 validation objects with both points missing; that establishes a supported representation, not the correctness of every such object.

Do not create boxes for entirely hidden cassettes inferred only from stack spacing, predictions or another frame. No arbitrary minimum pixel count or visible-area percentage is adopted. A recognizable thin fragment may be a target; an ambiguous fragment remains a recorded review issue.

Strongly overlapping boxes can belong to distinct adjacent fronts. Confirm duplicates from original-image evidence, never from IoU or missing-point flags alone. If an occluder separates two visible portions of one supported instance, keep one instance. If shared identity is uncertain, record that uncertainty instead of automatically joining or deleting rows.

## 2. Box the observed front-face extent

Use the tight axis-aligned rectangle enclosing the visible portions of the target front surface, clipped to the image. Determine its extrema from the target face, not from metal rack, gripper, hand, neighboring cassette, upper surface, projecting tray or other material.

An axis-aligned enclosing rectangle may contain background or occluder pixels between visible portions, or alongside a slanted face. That is compatible with the rule. Do not extend its outer bounds to guessed hidden face limits. Do not split a single face into multiple object rows merely to exclude an occlusion gap.

Keep box and point judgments independent during review. A point outside its box calls for checking the image and target definition; do not automatically enlarge the box to accommodate an incorrect point or snap a correct point to an incorrect box.

## 3. Locate the two actual lower-front corners

First identify the physical lower-front corner pair, then order the pair by image X. Do not substitute an upper lip, tray corner, occluder boundary, brightest stripe or arbitrary lowest image edge. The target is a physical feature, so perspective can tilt its connecting line.

On 2026-11-14 the user emphasized the front face of the **whole white wax block**. This is the individual cassette, not the connected silhouette of a stack. The [existing complete-front example](baomai-corner-examples-2026-11-12.md#2-a-clear-complete-front-new_000008-row-20) and [PR56 guidance audit](lower-front-guidance-audit-2026-11-12.md) already settle this feature. At a shared seam, preserve ownership by the front above rather than substituting the next front's upper shoulder. The [corrected three-front review](target-clarification-2026-11-14.md) applies that rule; it does not redefine the separate visible-extent box convention.

Each endpoint is decided independently:

- Use `v=2` when the actual corner is visible and reliably locatable from the original pixels.
- Use exactly `0 0 0` when it is occluded, outside the image, or cannot be reliably located because of ambiguity, blur or glare.
- Keep a visible right endpoint in the right slot even when the left endpoint is missing, and vice versa. The zero sentinel must not be used to reorder the surviving point.
- Compare X order only when both endpoints are visible. Never infer an absent endpoint from a predicted point, box edge, symmetry or an extended hidden line.

Mild softness or a rounded visible junction does not automatically require a missing label. A reliably identifiable actual junction can be marked. Where the image supports materially different candidate junctions, retain uncertainty rather than importing yolo-pose's extrapolated-corner convention. A visibility label is not a numerical guarantee of subpixel accuracy.

## 4. Coordinates and mechanical acceptance

Use the original 1280×720 PNG, with one 11-field row per instance:

```text
0 cx cy width height left_x left_y left_v right_x right_y right_v
```

Coordinates are normalized by original width/height. Require finite values, class 0, positive box width/height, and visibility only 0 or 2. Preserve `kpt_shape: [2, 3]`, `flip_idx: [1, 0]`, the class name and point names from the frozen data configuration.

Box edges use the closed image rectangle, so a right or bottom box edge may equal normalized 1. Visible point coordinates use the same half-open domain as the existing evaluator: `0 <= x < width`, `0 <= y < height`. A visible point saved at normalized 1 requires review; do not silently clamp or nudge it inward. An actual corner beyond the frame is missing.

Visible points must be on or inside their box. Allow **0.0001 original-image pixels** only when checking box clipping and point containment, to absorb decimal serialization. This is a numerical comparison tolerance, not an annotation-accuracy allowance or deployment requirement. The audited frozen labels have maximum box and point-containment rounding overhangs of approximately 0.0000064 and 0.0000128 px, respectively; no visible endpoint is on the excluded right/bottom boundary.

The staging script enforces schema, canonical missing states, X order and these geometry conditions. It cannot establish physical identity, visible-face tightness, the correct anatomical junction, or human approval. Those require the review record and original image.

## 5. Empty, missing and unresolved are different states

A genuine empty-scene label is an existing zero-length TXT paired with a reviewed original containing no identifiable target front. A missing label file is an error. An unresolved image is not a negative example and must not be turned into an empty file merely to complete an export.

Keep unresolved decisions outside the 11-field label format in the review record. Before releasing a newly annotated dataset, adjudicate affected images or explicitly quarantine them with recorded split/count changes. Do not silently change the frozen train/validation split. For a controlled incremental experiment, preserve legacy annotations and disclose its limited correction scope rather than claiming full migration to this policy.

## 6. Examples already reviewed

| Case | Decision under this policy |
| --- | --- |
| `new_000019`, pairs 56/63 through 60/67 | One row for each physical front; remove only the confirmed redundant representation in the separate candidate. |
| `new_000030`, rows 6/29 | Retain both adjacent fronts. Review row 29's excessive upper box extent separately from instance identity. |
| `new_000123`, rows 32/33 | Retain both fronts; row 33 keeps its visible left point and missing out-of-frame right point. |
| `new_000128`, row 2 | Top truncation does not hide the two visible lower-front corners. |
| `new_000012`, rows 12/25 | Preserve each independently visible left/right endpoint in its proper slot. |
| `new_000012`, row 34 | An identifiable bottom-clipped front may remain with both corners missing. |
| `new_000128`, rows 1/12 | Reassess thin-fragment identity under this policy; there is no automatic size-based deletion. |
| `old_000003` | Gray appearance and rack adjacency do not change the target surface. |
| `new_000131` | The reviewed rack/gripper/support scene remains a genuine empty negative. |

These examples describe semantic decisions from the [training pilot](training-pilot-2026-11-06.md); they do not newly certify every existing box or point.

## 7. Apply changes without changing the experiment's meaning

The [dedup-2026-11-06 candidate](../data/candidates/dedup-2026-11-06/README.md) removes eleven redundant rows and preserves every retained byte. It remains a **deduplication-only candidate**, not a fully reannotated implementation of this policy. Do not add box migrations to it silently; use a separately identified candidate if further image-supported edits are made.

Keep reference images, labels, split and checkpoint intact. Preserve original row IDs, source hashes, reason, reviewer status and changed-row text for each correction. The supervised HTML editor already supplies the needed drawing and saving operations; saving a file alone does not establish semantic acceptance. Run the mechanical checks and retain original-image review evidence.

Validation has already been used for development and epoch selection. Any correction of its reference labels must be separately versioned and reported, not folded into a score comparison unnoticed. Independent final-test scenes and application tolerances are still required for an application-level acceptance claim.

## 8. Context-derived proposals

Reference visibility rules in sections 1–7 do not impose an observed-only inference restriction. The established live/folder contract permits direct model estimates of obscured corners when endpoint confidence and image-bounds gates pass; it connects the two displayed predictions. These outputs are predictions, not certified physically visible points. PR55 retains direct PR35 output and keeps contextual geometry as a separate review tool; PR56 deliberately calls the diagnostic states SHOWN/NOT SHOWN. See the [recovered decision history](target-clarification-2026-11-14.md#3-q3-the-established-inference-behavior-already-permits-estimates).

The user has requested investigation of hidden-corner estimates from neighboring cassettes and rack geometry. The [context-corner pilot](context-corners-2026-11-12.md) implements that proposal with separately recorded `estimated_from_context` coordinates, source anchors, method, physical assumptions and review status. A proposal may be useful even when the corner cannot be directly seen.

Sections 1–7 continue to define the observed reference and current training labels. They do not prohibit a separate experiment with inferred coordinates. Keep an estimated point distinguishable from a visible `v=2` point; retain uncertainty when rank, plane or object identity is unresolved. This pilot approves no inferred point for training and does not replace a canonical missing slot.

Adopting inferred points as training targets would require an explicitly versioned candidate contract, including occlusion/provenance states, coordinate uncertainty, box/point compatibility and evaluation against evidence independent of the estimates. Native `v=1` support alone does not establish accuracy or give inferred coordinates a lower training weight.
