# Round 11: front boundaries and visible corner amendments

Round 11 starts from the exact round 10 delivery at `8230b8d592c792002bf97405d950264d22292a47`. The user supplied ten marked screenshots and row-specific text covering 58 photographs. The complete review continues to contain all 157 source photographs. Only requested new-series labels may change; all other new labels and all 31 accepted old labels retain their round 10 bytes.

## Physical annotation contract

Keep the [round 10 front-face contract](annotation-boundaries-r10.md): one visible cassette front per instance, an axis-aligned box enclosing its visible front and owned frontal rim, and the two actual lower front corners ordered by image X. The [upstream baomai source and policy audit](annotation-policy.md) remains the authority for landmark meaning. A rear perforated plane, rack member, hand or gripper does not determine a box extreme. A connected shaded frontal rim can belong to the cassette; darkness alone does not distinguish it from the rack.

For each changed box, identify the supported left, top, right and bottom extrema of the visible owned material. For each changed point, identify the actual lower front/side junction. These are separate observations: a point need not coincide with an axis-aligned box corner or an outer rim projection. Do not apply a common shrink, inward offset or universal point-to-box snap.

Visible landmarks use `v=2`; a corner hidden by a gripper, outside the image or not reliably locatable is `0 0 0`, independently for each side. A supported front fragment with two missing corners remains a valid instance. Missing points alone do not require a hold. Object additions and removals require a supported physical separator or evidence that two labels describe the same front.

## Reading the user's directions

All quoted row numbers identify immutable round 10 rows. Operations bind to exact before-lines and parent file hashes. New row numbers are separately recorded after additions and removals. The marked screenshots convey intended boundaries and corner identity; their scaling, line thickness and compression do not establish exact camera coordinates. Every numerical edit uses the original 1280 x 720 PNG. Decimal precision is serialization precision, not a claim of subpixel certainty.

The broad fronts in new124, new127 and new128 make the literal instructions to move `xmax` to the left point physically inconsistent. For example, new127 row18 would shrink from about 200 pixels wide to about 5 pixels and leave the other visible point outside. The grouped boundary-to-point phrases therefore refer to the image-right corner; the corresponding point-to-`xmax` phrases refer to the right point. Separate explicit left-corner corrections retain their stated side. This interpretation was reported to the user and is recorded in the request ledger.

The unspecified `xpoint` in new074 is interpreted as the right corner. All five requested rows have visible right corners; row12's left corner is hidden by the gripper. New072's explicit left-point instruction retains that side. Directional requests that conflict with a supported visible edge receive an explicit disposition rather than an arbitrary displacement.

## Review and training

The PDF contains exactly the photographs with changed label bytes relative to round 10, including photographs with additions or deletions. It displays every current instance on those photographs. Magenta identifies changed fields, cyan inherited fields and amber unresolved fields. Tags show current rows with round 10 identities when numbering changes. Deleted instances are dashed ghosts, and added instances have a plus sign. Unchanged photographs, including unchanged held cases, remain in the complete ledger but are omitted from this PDF.

Unresolved training photographs retain exact PR35 supervision in both comparison arms. The 31 accepted old photographs remain separately excluded from treatment and use exact PR35 supervision in both arms. Original PNGs, split membership, empty scenes, original validation labels and the frozen 666-front primary and 663-front diagnostic scoring references remain fixed. Development-review amendments are separately versioned exploratory labels.

The next notebook uses the established epoch 173 initialization independently for each arm, seed 46, 30 epochs, batch 4, image size 1280, FP32 and AdamW at learning rate 0.0001, with final-epoch selection. Preparing the notebook does not execute new training. The [fresh PR79 audit](../results/evaluation/pr79-r10-20261119/README.md) does not support promotion of the r10 candidate; PR35 remains selected.
