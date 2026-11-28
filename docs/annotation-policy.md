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

The Baomai and yolo-pose heads and the cited source files were checked again on 28 November 2026. The current Baomai instructions preserve each reliably locatable lower endpoint independently; they specify no minimum object size or visible-area fraction. The [R67 source audit](thesis/research-partial-inclusion-r67.md) distinguishes these upstream statements from this project's include-identifiable-partials decision. The [R67 image review](thesis/partial-inclusion-r67.md) resolves four historical exclusions by restoring their visible fronts in a separate evaluation reference and records three unresolved regions. The policy ID is unchanged because the general target definition is unchanged.

## 1. One identifiable visible front, one row

Annotate each uniquely identifiable visible cassette front exactly once. The target is the physical white front surface; illumination may make it look gray or dark. Do not classify instances using a brightness cutoff.

An instance can remain even if neither target corner is locatable, provided the original image supports its identity and a positive-area visible front extent. Point availability and object presence are separate decisions. The frozen reference already contains 121 training and 21 validation objects with both points missing; that establishes a supported representation, not the correctness of every such object.

Do not create boxes for entirely hidden cassettes inferred only from stack spacing, predictions or another frame. No arbitrary minimum pixel count or visible-area percentage is adopted. A recognizable thin fragment may be a target; an ambiguous fragment remains a recorded review issue.

Strongly overlapping boxes can belong to distinct adjacent fronts. Confirm duplicates from original-image evidence, never from IoU or missing-point flags alone. If an occluder separates two visible portions of one supported instance, keep one instance. If shared identity is uncertain, record that uncertainty instead of automatically joining or deleting rows.

## 2. Box the observed front-face extent

Use the tight axis-aligned rectangle enclosing the visible portions of the target front surface, clipped to the image. Determine its extrema from the target face, not from metal rack, gripper, hand, neighboring cassette, upper surface, projecting tray or other material.

An owned molded frontal rim or small tab continuous with that front is part of its observed extent. A broad receding/perforated upper plane is a different surface. Trace the material contour and surface orientation before deciding whether a gray strip is shaded cassette material or rack. Excluding rack does not mean removing every gray pixel or cutting the box to the brightest inner stripe.

### Small fronts and neighboring shadows

The R48 review keeps the same material-ownership rule at every object size. A shadow cast **onto the target's own front** changes its appearance, not its ownership; that shaded front remains included. This remains true when the shadow is cast by a neighboring cassette or the rack. Judge the physical surface receiving the shadow, not the identity of the object casting it. A shadow falling on rack/background, or the material of a gripper or neighboring cassette, must not enlarge the target's outer bounds. A thin owned molded frontal cap remains included even when it is gray. Small size alone is not an exception. This receiver-based clarification is explicit in R52.

In [R50](front-review-r50.md), the user's material-ownership clarification resolves
`new_000010 / R38 row029`: the lower band belongs to the next cassette. Its
physical bottom becomes nominal `ymax=714`, near its retained left corner. This
is a row-specific decision supported by ownership and the visible junction, not
a general rule to copy a corner's Y coordinate to every box. The unseen right
corner remains `0 0 0`.

The full input image already supplies surrounding context. When a separate crop-based experiment needs a larger context window, expand the input crop and preserve the original target geometry through an explicit coordinate transform. Do not enlarge the ground-truth box merely to feed context to the network. Soft image edges can carry nominal visual uncertainty; numerical serialization to many decimals is not evidence of that physical accuracy.

An axis-aligned enclosing rectangle may contain background or occluder pixels between visible portions, or alongside a slanted face. That is compatible with the rule. Do not extend its outer bounds to guessed hidden face limits. Do not split a single face into multiple object rows merely to exclude an occlusion gap.

Keep box and point judgments independent during review. A point outside its box calls for checking the image and target definition; do not automatically enlarge the box to accommodate an incorrect point or snap a correct point to an incorrect box.

### Use neighboring cassettes to identify the repeated front

The R47 review makes the existing neighboring-contour guidance operational. Compare the target with a nearby cassette whose front, molded rim and lower turns are clear. Match the physical surfaces and their contour continuity, allowing for perspective, pose and occlusion. Then locate the target's own visible edge. A neighboring lower corner can clarify approximately where the next front begins, but is not a coordinate to copy and does not reveal an entirely hidden point.

In `frame_0160`, source row 034's left lower corner provides useful context for row 037's upper frontal rim. The visible rim on row 037 must still determine its nominal top; the older box included the continuing rack above. This comparison can resolve surface ownership even when the transition is softened by a few pixels of blur. Normal edge softness should carry a nominal uncertainty, rather than automatically holding an otherwise identifiable front. Keep a field held when materially different surface owners, instance identities or hidden extents remain plausible, and record those alternatives.

For a cassette to the **right** of a rack upright, excluding the rack may mean moving the cassette's **left** boundary rightward: increase `xmin`. If that boundary instead cuts the cassette's own outward molded rim, move it leftward: decrease `xmin`. Inspect each row; do not apply one offset to an entire column. The same reasoning applies independently to the other three bounds. Gray material continuous with the front remains included; an adjoining gray rack strip or neighboring white cassette does not set the target's extrema.

See the [R47 original-image review](front-review-r47.md) for the explicit row decisions, original/R46/R47 comparisons and whole-photograph gates.

### Check all four sides against neighboring corners

The [R52 review](front-review-r52.md) extends the earlier
`rank_front_geometry_candidates.py` screening with a read-only
[`audit_front_geometry_r52.py`](../scripts/audit_front_geometry_r52.py).
It records other instances' locatable corners relative to the left, top, right
and bottom boundaries, distinguishing strict containment from near-edge contact.
Distances are measured in original-image pixels, and the available source-row
identity is retained. Missing `0 0 0` corners are not evidence of contact.

A foreign corner touching or entering a rectangle is a **review cue**, not proof
of an impossible physical configuration. Perspective, slanted surfaces and
occlusion can make the tight axis-aligned envelopes of distinct objects overlap.
The neighboring point may itself be wrong. Inspect the unmarked original and
the wider stack before accepting a change; never shrink a box, move a point,
delete an instance or lift a hold automatically to satisfy this check.

Prioritize strict containment and materially overextended sides over routine
stack contacts. A lower flag count is not a correctness metric. This check also
cannot discover a wholly unannotated cassette for which no box or corner exists;
full-frame review and review of unflagged regions remain necessary. Nominal
boundary refinements and their visual uncertainty belong in a versioned change
ledger, independently of whether the diagnostic count decreases.

## 3. Locate the two actual lower-front corners

First identify the physical lower-front corner pair, then order the pair by image X. Do not substitute an upper lip, tray corner, occluder boundary, brightest stripe or arbitrary lowest image edge. The target is a physical feature, so perspective can tilt its connecting line.

Each endpoint is decided independently:

- Use `v=2` when the actual corner is visible and reliably locatable from the original pixels.
- Use exactly `0 0 0` when it is occluded, outside the image, or cannot be reliably located because of ambiguity, blur or glare.
- Keep a visible right endpoint in the right slot even when the left endpoint is missing, and vice versa. The zero sentinel must not be used to reorder the surviving point.
- Compare X order only when both endpoints are visible. Never infer an absent endpoint from a predicted point, box edge, symmetry or an extended hidden line.

Mild softness or a rounded visible junction does not automatically require a missing label. A reliably identifiable actual junction can be marked. Where the image supports materially different candidate junctions, retain uncertainty rather than importing yolo-pose's extrapolated-corner convention. A visibility label is not a numerical guarantee of subpixel accuracy.

### Bright turn, darker underside, and lateral shadow

Recorded explicitly after the user's R46 feedback on 27 November 2026 (Asia/Hong_Kong): a visible lower-front corner often appears as a relatively bright turn, with darker material or shadow below it and a soft gray/shadowed region beside it. At a left corner the side region is usually to its left; at a right corner it is usually to its right. This is a useful local observation, not an invariant lighting rule or a physical law.

Follow the same front's lower edge to the side turn, then use this brightness pattern to support that interpretation. Inspect an unmarked original crop and the surrounding cassette/rack before moving the point. A highlight, cast shadow, internal stripe, or edge belonging to the next cassette can mimic the pattern. Do not choose the brightest pixel, apply a fixed upward/inward offset, or invent a hidden corner. A nominal outline can make the interpretation reviewable; it remains a diagnostic aid, not a segmentation training mask.

The earlier [R25 guidance](annotation-boundaries-r25.md) already treated brightness and shadow as supporting cues. The explicit underside/lateral pattern and the new examples are documented in the [R46 review](front-review-r46.md).

## 4. Coordinates and mechanical acceptance

Use the original 1280×720 photograph (PNG in the frozen collection; JPEG in the additional `frame_*` collection), with one 11-field row per instance:

```text
0 cx cy width height left_x left_y left_v right_x right_y right_v
```

Coordinates are normalized by original width/height. Require finite values, class 0, positive box width/height, and visibility only 0 or 2. Preserve `kpt_shape: [2, 3]`, `flip_idx: [1, 0]`, the class name and point names from the frozen data configuration.

The image origin is at the upper left: right increases X and down increases Y. Moving a point **up decreases Y**. Moving the top box boundary down increases `ymin`; moving its right boundary right increases `xmax`. The corner coordinates are judged independently of the rectangle's intersections.

In the R48 feedback, the user names physical **top** as `ymax` and physical **bottom** as `ymin`. The review records that wording, then explicitly translates top to stored `ymin` and bottom to stored `ymax`; it does not reverse the file format. Prediction numbers are also versioned: the old R47 PDF's P numbers denote PR133 candidate outputs, which equal PR137 reference outputs in the saved cache. The new R48 error PDF's P numbers denote PR137 candidate outputs. They are not interchangeable row identities.

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

For R48, predictions locate review regions but remain unchanged evidence. Amend the appropriate versioned reference or training proposal only when the original pixels support it. Multiple unmatched prediction cues can refer to one existing physical front; they do not authorize one new annotation row per prediction. A fixed-cache rescore after a reference amendment measures label-version sensitivity, not additional model learning. See [the R48 amendment decisions](front-review-r48.md).

## 8. Context-derived proposals

The user has requested investigation of hidden-corner estimates from neighboring cassettes and rack geometry. The [context-corner pilot](context-corners-2026-11-12.md) implements that proposal with separately recorded `estimated_from_context` coordinates, source anchors, method, physical assumptions and review status. A proposal may be useful even when the corner cannot be directly seen.

Sections 1–7 continue to define the observed reference and current training labels. They do not prohibit a separate experiment with inferred coordinates. Keep an estimated point distinguishable from a visible `v=2` point; retain uncertainty when rank, plane or object identity is unresolved. This pilot approves no inferred point for training and does not replace a canonical missing slot.

Adopting inferred points as training targets would require an explicitly versioned candidate contract, including occlusion/provenance states, coordinate uncertainty, box/point compatibility and evaluation against evidence independent of the estimates. Native `v=1` support alone does not establish accuracy or give inferred coordinates a lower training weight.
