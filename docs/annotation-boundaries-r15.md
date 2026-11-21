# Round 15: requested amendments after PR86

Round 15 starts from the saved r14 review at `a928f6beda7a797eeb21fc3961db7f1028555e4c`. Its parent manifest is pinned by SHA-256 in the builder. The saved feedback names24 photographs. Each requested original is inspected with its r14 overlay and relevant detail crops. The audit records exact immutable source rows, original image hashes, before/after lines and each clause’s disposition before replay. Unrequested labels and all31 old-series review files keep their exact r14 bytes.

## Physical target and coordinates

Keep the [r14 physical contract](annotation-boundaries-r14.md): one visible cassette front and its owned frontal rim per box, plus two independently observed lower front-to-side junctions ordered by image X. A projecting rim can extend beyond the lower junction. Rack, hand and gripper edges do not establish cassette corners. A hidden or unlocatable landmark stays exactly `0 0 0`; a locatable point uses visibility2.

Image X increases rightward and image Y downward. “Negative” and “positive” are interpreted as requested movement toward smaller/larger coordinates, not permission for out-of-image coordinates. A physical-top `ymax` request becomes image `ymin`, and a physical-bottom `ymin` request becomes image `ymax`, only where the original scene supports that interpretation. Cross-row point references align X while retaining the target’s own Y/visibility unless Y is explicitly requested and physically supported.

All references use immutable r14 coordinates, including an anchor edited later in the same pass. Photo015’s `xmim` is read as `xmin`; photo017 row59’s self-assignment is a recorded no-op. Approximate requests are bounded by the target front’s own observed material, with actual chosen coordinates documented. Perspective can shift a stack by tier, so an adjacent row cannot automatically define its neighbor’s complete extent.

Unsupported literal clauses remain recorded beside the retained valid coordinates: a missing point cannot become an X=0 anchor; left and right identities cannot be crossed; a cross-object reference cannot extend a box into another stack; an interior lower corner cannot cut away its own projecting upper rim. This decision does not create an automatic whole-photo training hold.

## Instance identity and inherited decisions

Delete only the expressly requested r14 row63 in photo028. Its former identity is recorded in the operation ledger and row map; every surviving row continues to use its r14 identity for this pass. No instances are added. The PDF may draw a dashed D63 historical marker, which is not a current training label.

The17 displayed-case approvals from r14 are inherited unchanged. There is no new blanket approval and no newly released photo in this round. Photo015’s undrawn possible extra front remains the sole unresolved instance question. Its current supported review amendments are saved, while both training arms keep its exact PR35 labels. All31 old-series photos remain separately excluded from the treatment with exact PR35 labels in both arms.

## PDF and training scope

Render exactly the changed nonempty new-series photos, displaying every current saved row on each. Magenta marks changed fields, cyan retained fields, and amber unresolved displayed fields. The undrawn photo015 question stays a text note; no guessed box is drawn. The complete review retains all157 label files and original photos, including empty scenes omitted from this PDF.

Eligible training corrections enter the composed candidate. Development-review changes remain a separately named exploratory scoring reference; original trainer-validation bytes and frozen666/663 scoring references stay fixed. The PDF is a rendering of saved labels, not an alternative data source. Mechanical replay and geometry checks establish file consistency, not infallible physical annotation.

## Next comparison

[PR86’s audit](../results/evaluation/pr86-r14-20261121/README.md) finds mixed, reference-sensitive results. PR35 remains selected. The r15 trial changes the eligible annotations while retaining model architecture, numerical recipe, thresholds and scoring reference policy. A post-amendment diagnostic must be named separately from the pre-amendment r14 diagnostic used to assess PR86.

Both fresh arms initialize independently from the original epoch173 checkpoint and use seed46,30epochs,batch4,1280input,FP32,AdamW with initial learning rate0.0001, and final-epoch selection. The resource trial is separate and cannot become either arm’s initialization. This pass prepares and verifies the notebook; it does not run GPU training or automatically promote a model.
