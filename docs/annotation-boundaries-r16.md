# Round 16 annotation amendments

This is the bounded follow-up to [r15](annotation-boundaries-r15.md), after the returned [PR88 results](../results/evaluation/pr88-r15-20261122/README.md). The user's 17 requested photos are inspected against the original 1280 × 720 PNGs and the immutable r15 row identities. Unrequested labels and all 31 old-series review files retain their exact r15 bytes.

## Physical target and coordinates

Keep the established target: one visible cassette front and its owned frontal rim per box, with two independently locatable lower front-to-side junctions ordered by image X. A projecting rim may extend beyond its lower junction. A rack, hand or gripper edge cannot establish a cassette corner. An unlocatable point stays exactly `0 0 0`; a locatable point has visibility 2.

Image X increases rightward and image Y downward. A requested physical-top `ymax` becomes image `ymin`, or a physical-bottom `ymin` becomes image `ymax`, only where the original scene supports that interpretation. Cross-row point references align X while preserving the target's own Y and visibility, unless the request explicitly specifies Y and the image supports it. Negative and positive mean movement toward smaller and larger image coordinates. Approximate or directional changes stop at the target front's observed extent; they are bounded visual estimates, not a subpixel measurement claim.

All anchors come from the immutable r15 labels, including anchors also edited in this pass. Photo004 row49's formula uses the two saved source edges. Photo008's abbreviated `15 xmin to 16` means row16's xmin. No edit cascades through an earlier edit in the same pass.

Every clause has a saved disposition. A missing-point sentinel cannot define an edge, a cross-object reference cannot move a corner into another front, and an interior lower junction cannot trim off its owned projecting rim. Unsupported literal changes retain the physically supported saved field, with a specific explanation; this does not automatically hold the entire photo.

## Instance identity and inherited decisions

Delete only the expressly requested r15 row22 in photo066. Preserve its identity in the operation ledger and row map. No instances are added. A dashed D22 marker in the PDF is historical context, not a current training label.

The 17 displayed-case approvals from r14 are inherited through r15 unchanged. This round does not claim new blanket approval or release any held photo. Photo015's undrawn possible extra front remains the sole unresolved instance question. Both training arms keep that photo's exact PR35 labels. The 31 old-series photos remain separately excluded from the annotation treatment, with identical PR35 labels in both arms.

## PDF and training scope

Render exactly the changed nonempty new-series photos, with every current saved row on each. Magenta identifies changed fields and cyan retained fields; amber identifies unresolved displayed fields where present. Requested photos with no supported byte changes remain in the audit and are omitted from the changed-photo PDF. The complete review includes all 157 label files and original photos, including empty scenes.

Eligible training corrections enter the composed candidate. Development-review changes remain a separately named exploratory scoring reference. Original trainer-validation labels and the frozen 666/663 scoring references stay fixed. The PDF renders the saved labels; it is not a separate annotation source. Mechanical replay and geometry checks establish consistency, not infallible physical annotation.

## Next comparison

The PR88 audit retains PR35 as the selected model under the frozen protocol. The r16 trial changes eligible annotations while preserving model architecture, thresholds, scoring policy and numerical recipe. The post-r16 diagnostic must be named separately from the pre-amendment r15 diagnostic used to assess PR88.

Both fresh arms initialize independently from the original epoch173 checkpoint: seed46, 30 epochs, batch4, 1280 input, FP32, AdamW with initial learning rate0.0001, final-epoch selection. The separate resource trial cannot initialize either arm. Preparing this round does not run GPU training or automatically promote a model. Pixel thresholds remain development diagnostics; see [robot-grasp readiness](robot-grasp-readiness.md) for the limits of physical positioning claims.
