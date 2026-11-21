# Round 18 annotation amendments

The accepted r17 review is the immutable parent. Inspect every supplied nonempty photograph from `new_000081` through `new_000130` against its original camera pixels. The prepared dataset contains 40 photographs in this range: 35 training and five development photographs (086–090). Photos 091–093 and 104–110 are absent from the prepared source; they are not counted as empty scenes or newly imported.

Apply only confident coordinate replacements. Keep unsure fields byte-for-byte unchanged and create no new questions, cases, holds, or amber markings. Preserve all instances, row identities, classes, point visibility values, and annotations outside the requested range. Record the accepted before/after lines and physical reasons against immutable r17 row numbers.

Each box follows its cassette's visible front material and owned rim. The two points locate independently visible lower front-to-side junctions, ordered by image X. A projecting rim may extend beyond its corner. Therefore `xmin = left.x` and `xmax = right.x` are possible corrections, not universal constraints. Move a point to an existing box edge only when that edge is also the visibly supported junction. Never infer a hidden point or use another object as the boundary. Known points retain visibility 2; absent points retain the exact `0 0 0` sentinel. Pixel-based visual estimates do not claim subpixel physical precision.

The PDF contains only photographs with saved label changes. It renders every current front on each included photograph. Only changed field groups are magenta; unchanged geometry is cyan. It displays no uncertainty flags or amber cases. The PDF reads saved labels and cannot adjust their coordinates.

Existing dataset decisions remain inherited. The original photo 015 training hold retains exact PR35 supervision in both arms. The separate 31 old-series treatment exclusions also retain exact PR35 supervision in both arms. No new exclusion is created by this review.

Development corrections are saved in the complete r18 review and can be scored separately as exploratory evidence. Original trainer validation remains 20 photographs / 664 fronts. Frozen primary scoring remains 666 fronts, and historical scoring remains 663 fronts. The PR90 comparison uses its pre-return references; this later annotation pass cannot silently replace them.

The next notebook uses two fresh arms from the original epoch 173 initializer, seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW, and final-epoch selection. The resource trial cannot initialize a full arm. Preparing and validating the notebook does not run GPU training or establish that these annotation changes improve model performance.
