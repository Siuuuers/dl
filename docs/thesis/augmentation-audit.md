# Native augmentation and label survival

## Decision and scope

The completed CPU diagnostic supports retaining the existing augmentation policy. All four notebook cells ran successfully from the published source pin, and the bounded visual review found no concrete inconsistent supervision in the six selected panels. The notebook audits the **existing** data transformation pipeline; it does not train a detector or alter a label. The current model, the prepared quality-focal comparison and the frozen training-label comparison remain unchanged.

The useful question is: **what source fronts and corner observations reach the loss after native augmentation, and which transformed examples need visual review?** A dropped row is not automatically an error. The transformed front might have left the canvas or become too small to identify. Correct native arithmetic also does not establish that the surviving box still follows the visible cassette material precisely.

The [four-cell notebook](../../notebooks/baomai_augmentation_audit_r63.ipynb) uses a CPU runtime. Its implementation is [the audit driver](../../scripts/audit_training_augmentation_r63.py). Publication and execution receipts are kept in [the evidence folder](evidence/augmentation-audit/README.md).

## Why this is a distinct comparison

The existing recipe already uses rotation up to 5 degrees, translation up to 5% of the canvas dimensions, scale from 0.85 to 1.15, horizontal flipping with probability 0.5, and HSV variation. Mosaic, MixUp, CutMix and Copy-Paste are disabled. Rectangular batching, image size 1280 and the left/right slot exchange remain fixed. The precise recipe is defined by `reviewed-incremental-fronts-seed46-v1` in the [training contract](../../scripts/pose_checkpoint.py).

The completed classification-only experiment recorded 118,835 post-augmentation target presentations. The source inventory of 3,999 fronts over 30 passes would contain 119,970 presentations before transformations. Its post-transform partial-target count was 18,626; the source inventory contains 602 partial targets, giving 18,060 source partial presentations. These aggregate differences show that source and consumed inventories differ. They do not identify the individual removed targets or establish a mistake. See [the verified result](../partial-classification-r51-results.md).

Generic covered-corner training has already been tested. That historical treatment retained known source coordinates under artificial pixel corruption, improving the reused synthetic probe but failing the clean-image preservation rule. Its new empty-scene output and clean-image regressions preclude presenting another generic masking run as an untried first step. See [the completed occlusion result](../occlusion-results-2026-11-14.md).

## Fixed CPU protocol

| Item | Specification |
| --- | --- |
| Photographs | The exact 142 PR137 candidate training photographs reused by the quality-focal notebook |
| Source labels | 3,999 fronts, 7,259 known corner slots, 739 unavailable slots; unchanged archived bytes |
| Empty training photographs | 13, retained |
| Reference view | Native letterbox and formatting on the same rectangular training canvas |
| Augmented view | Actual Ultralytics 8.4.182 training dataset and native transforms, using the existing recipe |
| Fixed sampling | Three passes over all 142 photographs; 426 image and 11,997 source-object presentations |
| Seeds | `46 + pass_index * 142 + native_dataset_index`, with indices starting at zero |
| Models and optimization | No model load, neural inference, loss calculation or optimizer update |
| Evaluation photographs | None staged or transformed |
| Review output | At most six original/transformed image panels, chosen by the fixed traversal rule |

This is a new deterministic diagnostic sample. It does not reproduce a historical trainer's random-number stream, and the three transformed copies are not new independent captured photographs. Optional Albumentations availability, package versions and the actual transformation list are recorded. Unexpected extra spatial transforms stop the audit because their row lineage has not been implemented.

The observer logs the actual affine matrix and horizontal-flip decision. It independently recomputes the native filtering predicate and verifies the full final target geometry. Representative empty, nonempty and removed-row samples are also replayed without the observer, requiring identical image and target tensors. Every retained or removed row keeps its original photograph ID, original one-based label row and label hash.

## What the native implementation does

The pinned [augmentation implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/data/augment.py) transforms the four corners of each source axis-aligned box, takes their new axis-aligned extent, clips it to the canvas and filters rows. For this box/keypoint dataset, retained boxes must have width and height strictly greater than 2 pixels, transformed area divided by scaled source-box area strictly greater than 0.10, and aspect ratio strictly below 100. A row can fail more than one condition.

These are box-geometric tests. The area ratio is **not a measured fraction of visible cassette material**. Rotation can enlarge the axis-aligned enclosing box, and an original box may include background or an internal occluder. Without masks or visual review, box clipping cannot recover exact visible-material ownership.

Native keypoint transformation sets availability to zero when a point moves outside its canvas bounds. The [instance implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/utils/instance.py) clips its internal coordinates afterward. The audit preserves the native inclusive upper-bound convention and checks it explicitly. A canonical unavailable source slot remains `0 0 0` in its label file, but its **internal transformed coordinates may be nonzero while its flag remains zero**. That difference alone is not corrupted supervision. A source-unknown point must never become known, and the audit checks that invariant separately from a source-known point becoming cropped out.

## Completed result

The unchanged four-cell notebook completed on CPU from source `87b3f1dfd547bf46fef35b171aebe16d832bd1be`. Its [execution receipt](evidence/augmentation-audit/notebook-execution.json), [numeric result](evidence/augmentation-audit/results/summary.json), [compressed row lineage](evidence/augmentation-audit/results/details.json.gz) and [panel review](evidence/augmentation-audit/visual-review.json) preserve the evidence. The clean published source, five native implementation files, input hashes, native output equivalence and saved aggregate replay all passed. Hosted Colab execution and its browser download interface were not tested.

| Originally known corners | Source presentations | Retained with 0 known | Retained with 1 known | Retained with 2 known | Removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 411 | 397 | 0 | 0 | 14 |
| 1 | 1,395 | 14 | 1,356 | 0 | 25 |
| 2 | 10,191 | 22 | 79 | 10,012 | 78 |
| **Total** | **11,997** | **433** | **1,435** | **10,012** | **117** |

The pipeline retains 11,880 object presentations and clips 368 retained boxes. Of the 117 removed presentations, 108 have zero clipped box area. The remaining nine are thin edge slivers; the largest short side is 4.05 pixels. These measurements do not establish exact visible-material ownership, but they give no demonstrated reason to relax the native filtering thresholds.

Among retained source-complete objects, 101 become partial after cropping: 79 retain one known corner and 22 retain neither. A further 14 originally one-corner objects retain no known corner. No source-unavailable corner becomes known, and all 39 empty-image presentations remain empty. Thus source and consumed partial-target inventories can differ through ordinary geometry, without an annotation edit.

The six selected panels show cropping at the upper or lower image boundary and correctly tracked observed corners; the review found no concrete counterexample in that sample. The raw-source overlay coordinate frame was independently checked for all 426 presentations. This is a bounded visual review, not a validation of every transformed physical contour or proof that the current augmentation is optimal. It supports completing the already prepared model comparisons before adding another occlusion-training arm.

## Why new masks or cropping require a label contract

Random Erasing provides primary evidence that simple rectangular corruption can improve recognition on the benchmarks studied [1]. It does not establish the appropriate label semantics for these visible cassette fronts. If a synthetic mask hides a known lower corner, retaining positive coordinate supervision asks the model to reconstruct known source geometry. Marking the endpoint unavailable asks it to report only the modified image's supported observations. Those are different experiments.

A mask that removes the last visible material at a box edge also changes the correct visible-front extent. Merely clearing the corner flag leaves that box question unresolved. Cropping creates the same issue, and replacing a missing endpoint with the crop boundary would change the physical landmark definition. A mask can also affect neighboring fronts; the old occlusion experiment explicitly measured substantial collateral endpoint coverage.

Simple Copy-Paste updates the masks and boxes of partially occluded instances and removes fully occluded ones [2]. Its instance masks supply information our current box-plus-two-point labels do not provide. Pasting whole box rectangles would also paste background, rack material or neighboring fronts. This is therefore a research direction requiring an additional annotation/geometry contract, not a minimal switch to enable immediately.

## How the result can guide one later experiment

Review the audited examples before changing a threshold or augmentation. If a consistent subset of recognizable partial fronts is removed by the native transformation, a later controlled augmentation-policy comparison can address that specific mechanism. If the source-to-transformed labels are coherent, the audit gives no reason to add a new training run. A changed sampler, partial-target weight, new corner refiner or another training-label arm would repeat questions already covered elsewhere.

A later model comparison must hold source images, labels, initialization, native losses, image/update budget and evaluation references fixed. It must report original zero-, one- and two-known-corner groups, misses, extras, duplicates, one-corner localization, complete-pair success and empty-scene outputs. A pipeline diagnostic alone cannot prove that augmentation caused the current detector's errors or that disabling it will improve recognition.

## Primary sources

[1] Z. Zhong, L. Zheng, G. Kang, S. Li and Y. Yang, “Random Erasing Data Augmentation,” AAAI, 2020, pp. 13001–13008. [Publisher](https://ojs.aaai.org/index.php/AAAI/article/view/7000); [author preprint](https://arxiv.org/abs/1708.04896).

[2] G. Ghiasi et al., “Simple Copy-Paste Is a Strong Data Augmentation Method for Instance Segmentation,” CVPR, 2021, pp. 2918–2928. [Primary paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf).

The papers motivate the comparison and its annotation requirements; neither provides a guarantee for this cassette dataset.
