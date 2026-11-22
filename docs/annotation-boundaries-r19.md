# Round 19: cassette fronts, rack boundaries and visible corners

The complete saved r18 review is the immutable parent. Inspect all 40 supplied nonempty photographs in 081–130 against original camera pixels: 35 training and five development photographs 086–090. IDs 091–093 and 104–110 are absent from the prepared source. Preserve every instance and its row identity. Apply only visually supported changes; uncertain fields retain their r18 bytes, without new questions, cases, holds or amber marks.

## Box and point meanings

A box encloses the cassette's visible front material and its owned rim. Exclude surrounding rack bars, gaps, background and fingers. Keep real projecting front material: the most outward visible front edge can lie beyond a lower corner. The box is axis aligned; a sloped front can therefore contain some background within its bounding rectangle. Removing rack means tightening the unsupported boundary, not cutting cassette material to eliminate every background pixel inside the rectangle. Photo 086 row 3 is the user's box example.

Each point marks an independently visible lower corner of the **front face**, ordered by image X. Where the side face forms a shaded triangle, the front corner is its junction with the front, at the right tip in the user's left-corner examples. It is not the outer silhouette of the side face. Photo 128 rows 31–33 are the user's corner examples. Photo 114 row 15's left X is close to row 14's left X; row 16's left corner also lies to the right of its outer silhouette.

The previous round incorrectly moved several front-face corners onto outer side edges. Reinspect all 24 previous point moves against these examples. Restore an older coordinate only where original pixels support it. Neither `xmin = left.x` nor `xmax = right.x` is a universal rule; inspect box boundaries and points separately. Cross-row references use immutable r18 coordinates. Visual estimates do not claim subpixel physical precision.

Known visible points use visibility 2. Preserve every other visibility value and the exact missing-point sentinel. The single user-requested exception is photo 127 row 32: replace its missing left-point sentinel with the visibly supported front-face corner and visibility 2. This restores one endpoint on an existing object and does not add an instance. Never infer a hidden point.

## Saved evidence and display

Save exact before/after lines, source image and label hashes, physical reasons, and explicit dispositions for the user's examples. The complete review contains all 157 annotation files; the training candidate selects their eligible training supervision. Original source labels remain preserved.

The PDF shows only photographs whose saved labels changed, with every current front on each included photograph. Only changed field groups are magenta; retained geometry is cyan. There are no amber or uncertainty markers. The renderer reads saved labels and does not alter them.

## Experiment boundaries

Photo 015's inherited training hold and the separate 31 old-series treatment exclusions retain exact PR35 supervision in both arms. No new hold or exclusion is created. Trainer validation remains the original 20 photographs/664 fronts. Frozen primary scoring remains 666 fronts and historical scoring 663. The current development review is separately pinned exploratory evidence.

The PR92 evaluation uses its pre-return r18 labels; these later r19 changes cannot rewrite that comparison. The next notebook runs two independently initialized arms from original epoch 173 with the existing seed 46, 30 epochs, batch 4, input 1280, FP32 and AdamW recipe and final-epoch selection. Preparing the notebook does not run training or establish an improvement.
