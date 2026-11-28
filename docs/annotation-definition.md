# Cassette-front annotation definition

**Label one identifiable visible cassette front, fit its visible extent, and mark each actual lower corner only when that corner can be located.**

This is the compact operating definition of the existing **`baomai-visible-front-v1`** [policy](annotation-policy.md). It preserves the accepted target and file format. It adds a repeatable review procedure; it does not relabel images or certify every historical label. The [research note](thesis/research-annotation-definition.md) explains the supporting literature and what still needs measurement.

**Frozen benchmark scope.** The general definition below and the inherited reviewed evaluation inventory are not identical. The [R38 record](front-review-r38.md) preserves five case-specific user exclusions, including a visibly present partial front at the former `new_000009:52`. These exclusions are retained in the later reviewed reference; they are not a general size rule or evidence that the excluded fronts are background. The [bounded R65 photo review](thesis/annotation-review-r65.md) documents this distinction with the original pixels and exact historical row identity. Existing reference versions remain frozen, and an extra prediction means an unmatched output under the named reference, not necessarily a nonexistent physical cassette.

## 1. Decide what belongs to the front

The target is the cassette's front face, including its own visible molded frontal rim or small frontal tab. Include its shaded material. Exclude the receding upper plane, projecting tray, rear surface, rack, gripper, hand and neighboring cassettes.

Use contour continuity and surface orientation to establish ownership. Color and brightness support that judgment but do not decide it: a gray strip may be shaded cassette material or metal rack. A shadow changes the appearance of the surface receiving it, not which object owns that surface. A supposed “cap” is included only when it is part of the frontal rim; the word alone does not establish ownership.

One physical front receives one row, even if an occluder separates its visible portions. Overlap between two rectangles does not establish a duplicate. A wholly hidden cassette suggested only by stack spacing receives no row.

## 2. Fit the visible front, then locate its corners

**Box:** find the leftmost, topmost, rightmost and bottommost visible material belonging to this front. These four extrema define the smallest axis-aligned enclosing rectangle. Stop at observed extent or the image boundary; do not extend to an imagined hidden outline. The rectangle may contain background or an occluder between the visible parts. Extra context belongs in an input crop, not in the target box.

**Corners:** follow this front's lower contour to its actual junction with each side. These junctions need not coincide with the rectangle's lower corners. Do not substitute the upper lip, tray, internal groove, highlight, cast-shadow boundary or an occluder's edge. A rounded but identifiable junction can have a nominal location; do not extrapolate ideal straight sides to manufacture a hidden sharp corner.

Left and right mean **image-left and image-right**. Decide each slot independently. If both are available, their X coordinates must be ordered. If one is missing, keep the surviving point in its physical slot.

| Original-image evidence | Annotation action |
| --- | --- |
| Front and both lower corners identifiable | One visible-front box; two observed points with `v=2`. |
| Front identifiable; one or both corners covered, clipped or not reliably locatable | Keep the box; use exactly `0 0 0` for each unavailable corner. |
| Target reaches the image edge | Clip the box to the frame; retain a corner only if it is actually inside the image and locatable. |
| Front is thin or small but uniquely identifiable | Keep it; no automatic size or visible-percentage deletion rule. |
| Two separated visible pieces clearly belong to one front | One box spanning those pieces; no second row for the same object. |
| Material may belong to another front, a support or the rack | Record the alternatives and hold the proposed change for adjudication. |
| No identifiable target exists anywhere in a reviewed image | Keep an empty label file as a genuine negative. |

## 3. Review in a fixed order

1. Open the unmarked original photograph. Identify the front and its neighboring surfaces before showing model predictions.
2. Judge the four box extrema from visible material. Check all four sides, including the shaded rim.
3. Judge the two actual lower corners independently. Use a wider crop to resolve ownership; another frame cannot provide an observed coordinate in this frame.
4. Record uncertainty explicitly: instance identity, surface ownership, hidden extent, or corner location. Ordinary edge softness is a nominal-location issue; competing physical interpretations need adjudication.
5. Sweep the full photograph for omitted fronts and redundant rows. A queue built from predictions or geometry flags cannot establish completeness by itself.
6. Compare the saved overlay with the original. Record the changed fields, reason, immutable source-row identity and reviewer status; then run the existing mechanical checks.

Predictions may identify a region worth reviewing. Agreement with a prediction, a higher score, fewer geometry flags or a prettier rectangle is not evidence that an amendment is physically correct. The original pixels and target definition decide it.

## 4. Keep uncertainty compatible with the implementation

The saved row remains:

```text
0 cx cy width height left_x left_y left_v right_x right_y right_v
```

Use original-image coordinates normalized by width and height. The origin is upper left: X increases rightward and Y downward. The stored top is `ymin`; the bottom is `ymax`. Prefer “top/bottom” in review instructions when an earlier request used these names differently.

The current parser accepts class 0 and point flags **0 or 2**. It has no implemented per-instance ignore field. An unavailable point is not an ignored object: its front still supplies box supervision and is included in detection evaluation. Do not insert `v=1`, a new class, a fake zero-size box or an empty file to encode uncertainty.

For an existing controlled comparison, keep the declared legacy supervision identical in both arms on held photographs and disclose that hold. For newly annotated data, adjudicate the affected photograph or quarantine the whole photograph with its reason and split/count changes recorded. Do not silently remove an ambiguous row while treating the rest of that photograph as exhaustively labeled. Changing an evaluation ignore rule would require a separate, explicit evaluation version.

The existing [label parser](../scripts/evaluate_pose.py), [geometry checks](../scripts/verify_pose_candidate.py) and [candidate verifier](../scripts/verify_pose_annotation_candidate.py) already check format, missing slots, ordering, bounds, containment and recorded changes. Their tiny serialization tolerance is not a claim about annotation accuracy. A containment failure requires image review; never snap a point or enlarge its box automatically to pass it.

## 5. Check whether the definition is repeatable

Before a large new relabeling pass, use a fixed calibration set containing a clear front, shaded rim, rack-adjacent front, occlusion, image-edge truncation, thin partial front and rounded/soft lower corner. Have a second reviewer mark the same originals independently, with old labels and predictions hidden. If only one reviewer is available, a delayed second pass measures that reviewer's repeatability, not agreement between people.

Record disagreements in instance count and ownership separately from coordinate differences. For agreed instances, measure each box-side difference and each available corner's distance in original-image pixels; record availability disagreements separately. Resolve the physical interpretation before averaging coordinates. Add resolved examples to the guidance and keep unresolved alternatives in the ledger.

**This repeatability study has not yet been measured.** Five pixels is an existing model-scoring threshold, not a demonstrated human-label precision or a reason to force uncertain annotations inside that distance.
