# R51: isolate partial-target classification weighting

## The question and its evidence

R51 asks whether assigning twice the native classification BCE weight to anchors matched to partially visible cassette fronts improves the detection/corner tradeoff. It is a controlled training experiment, not a claimed improvement. The runnable entry is [the four-cell notebook](../notebooks/baomai_partial_classification_r51.ipynb). It returns `baomai-partial-classification-r51-results.zip`.

The [PR143 CPU audit](../results/evaluation/pr143-r49-20261127/summary.json) verified the returned R49 inference and replayed its cached outputs. Against R38 and the later R50 reference, the PR137 candidate has 15 misses: 10 at the confidence gate, three without a decoded box meeting the matching IoU, and two at ordinary NMS. These are measured pipeline locations, not proven physical causes. Linear Soft-NMS recovers no target, including before postclip filtering. It removes one final duplicate in these two references.

The [bounded cached confidence probe](../results/evaluation/pr143-r49-20261127/confidence-probe/summary.json) examines a different tradeoff; it does not select a new threshold. Lowering confidence from 0.13 to 0.05 changes R38/R50 misses from 15 to 9 and false positives from 84 to 158, with the two-corner 5px count unchanged. Recovering six detections at the cost of 74 more unmatched predictions is evidence to investigate training behavior, not evidence that the lower threshold is preferable.

R51 isolates the classification term from earlier interventions: R30 combined box and classification weighting, while R31 weighted the box term alone. Its factor, inputs and recipe are fixed in advance. The original PR35 selection and frozen666 primary gate remain in place; no model is automatically promoted.

## What both arms actually train

The **reference and candidate both use the exact PR137 candidate image and label bytes**, restored from its audited archive. The new `reference` is a fresh native-equivalent run on 142 images. It is not the historical R47 reference, which used 138 images.

| Item | Reference | Candidate |
| --- | --- | --- |
| Training photographs | 142 | 142, identical bytes |
| Training cassette targets | 3,999 | 3,999, identical labels |
| Locatable/missing corner slots | 7,259 / 739 | 7,259 / 739 |
| Empty training photographs | 13 | 13 |
| Initializer | epoch173 | Same checkpoint bytes |
| Seed / epochs | 46 / 30 | 46 / 30 |
| Image size / batch | 1280 / 4 | 1280 / 4 |
| Arithmetic / optimizer | FP32 / AdamW | FP32 / AdamW |
| Classification multiplier on assigned partial anchors | 1 | 2 |
| Batches / image presentations / optimizer updates | 1,080 / 4,260 / 80 | 1,080 / 4,260 / 80 |

These inputs include the 137 original training PNGs and five supplementary JPEGs: `frame_0146`, `frame_0160`, `frame_0178`, `frame_0371`, and `frame_0383`. **Yes, these five frame photographs were already trained in the PR137 candidate.** R51 uses them in both arms so that changing the training photos cannot explain a difference between the two new runs. Source and archived labels are checked rather than inferred from file names.

Later supplementary amendments are not silently added. In particular, the R48 `frame_0001` candidate is outside this experiment's training set. R50 is a separate development reference and adds no training labels. This experiment therefore does not claim that every latest annotation in the repository has been trained.

The trainer's native validation split stays at **20 photographs / 664 targets**, exactly as in the inherited recipe. Both checkpoints are evaluated using **last.pt**, so an earlier checkpoint is not chosen by one reference's validation score. Final reports separately score **frozen666, historical663, R38 and R50**; these are different labels for the same reused development scenes, not four independent datasets.

## The exact loss change

A partial target has fewer than two nonzero keypoint visibilities in the batch consumed by the loss, after augmentation. Repository annotations use visibility 0 for unavailable corners and 2 for locatable corners. The hook reads the consumed tensor, so an augmentation that clips a corner is handled consistently with the loss's actual inputs.

Native assignment first determines which foreground anchors belong to which ground-truth cassette. For an anchor assigned to a partial target, the candidate multiplies its whole native soft-target classification BCE term by 2. The reference uses factor1 without adding an identity multiplication. Background and fully visible targets retain factor1.

The native assigner, original target scores, original unweighted normalization, native class weights, box loss, keypoint loss, keypoint-visibility loss and RLE loss remain intact. Both native PoseLoss26 branches retain their existing schedule. There is no oversampling, extra-image exposure, modified box multiplier or global library monkeypatch. A training-only criterion is installed after the separate EMA model has been created; serialization uses the native EMA model.

These invariants apply to the native algorithms and their calculations for fixed predictions and a fixed consumed batch. Once the two arms take different updates, their predictions, assignments and soft targets can diverge. Their shared features also allow boxes and corners to change despite keeping the explicit box and corner loss formulas unchanged. The return records each arm's actual assignment statistics separately.

This weighting **does not mathematically guarantee higher confidence**. For a classification logit `z`, sigmoid probability `p` and native soft target `t`, the BCE derivative is `p − t`. Weighting changes that derivative to `w(p − t)` on the selected anchors. If `p < t`, the update pushes upward more strongly; if `p > t`, it pushes downward more strongly. The optimum of one isolated weighted BCE term is still its target. Network capacity, assignment and competing examples determine the final outcome. Treat any recall improvement as a result to measure.

## Run the notebook

Select a T4 GPU in Colab and run all four code cells in order. The notebook separates four tasks:

1. Restore the exact source revision, repair required sparse paths on every setup run, and verify all local Python dependencies against Git and SHA256 pins.
2. Audit the exact PR137 return, restore the original images and epoch173 initializer, and stage both identical datasets. Images retain their original 1280 × 720 bytes. The five JPEGs come from the pinned R44 archive's original-image capsule; no historical review charts are rebuilt.
3. Install Ultralytics 8.4.182 while constraining the existing compatible torch/torchvision pair, then run CPU correctness and real-loader checks.
4. Run a four-batch resource trial, then a fresh reference and a fresh candidate; evaluate each final checkpoint at confidence0.13 with the same native NMS/postclip pipeline, replay the saved metrics, and build the verified return ZIP.

Every command rechecks the pinned source. Both training arms must have identical runtime identities, and the candidate requires a verified completed reference. The resource trial must perform four actual updates with finite losses and gradients; its weights are never used to initialize either arm. Measured optimizer update batch indices must equal the native accumulation schedule, not just have the right final count.

Optional environment paths are `BAOMAI_R51_BASE`, `BAOMAI_R51_ARCHIVE`, `BAOMAI_R51_INPUTS`, and `BAOMAI_R51_BACKUP`. These support a fresh work directory, existing verified downloads, or a persistent backup folder. The notebook does not automatically mount a drive. A local repository mirror can be supplied using `BAOMAI_R51_REPOSITORY`.

Completed trials and arms are verified before reuse and can be copied to the optional backup folder with source, staging, file-hash and runtime receipts. Incomplete runs and conflicting files are preserved; choose a fresh base directory to start over. Interrupted training is not silently resumed or combined across different runtimes. A completed return can restore only its cached evaluation files after all other source, labels and completed runs agree.

## Verification and return evidence

The CPU [loss guard receipt](../results/evaluation/pr143-r49-20261127/partial-classification-gradient-guard.json) checks factor1 loss and full-network gradients against native code, plus selective factor2 gradients in both branches. It covers real mixed examples, a blank example, all-full targets, a consumed-visibility mutation, native class weighting and the 30-epoch branch schedule. This guard uses a small CPU input for differentiation checks and performs zero optimizer updates; it is not a training run or an accuracy measurement.

The real dataset preflight constructs the pinned native datasets/loaders for both roles. Every loaded image and TXT is hash checked, and every cached class, normalized box and keypoint coordinate is compared to the authoritative TXT bytes. It verifies all 142 training and 20 validation photographs per arm and consumes one CPU batch from each loader. The actual GPU trainer repeats those cache/byte checks at pipeline construction.

The hook records consumed image identities, partial/full target counts, assigned and unassigned targets, foreground/background anchor counts, affected classification terms, native and weighted BCE numerators, unchanged denominators and branch schedules. These records must reconcile with measured trainer exposure. A successful run must observe partial classification terms in each native branch; nominal configuration alone is insufficient.

The ZIP contains both final checkpoints and CSV histories; trial and training receipts; source closure and input provenance; every staged training/validation label; all four score references; predictions before and after postclip; per-image detection assignments; recovered and newly missed row identities; corner results stratified by 0/1/2 locatable corners; CPU guard and actual-loader evidence; and a complete member SHA256/size inventory. It omits repeated copies of large original images and the initializer because immutable hashes and source locations already identify them.

The final comparison should report misses, other false positives, duplicates, and corner quality together. Examine recovered and newly missed identities, especially partial fronts, and retain all four reference tables. The same repeatedly reviewed 20 scenes support development conclusions; new independent captures are still needed for generalization claims. With one seed and a fixed small dataset, a favorable R51 result is a candidate for further checking, not sufficient evidence for a broad performance claim.

## Existing R50 receipt behavior

The [real-return serialization regression](../results/evaluation/pr143-r49-20261127/r50-receipt-regression.json) confirms that R50 already normalizes the inherited audit's tuple counts through canonical JSON before saving and comparing its receipt. The canonical real PR137 audit exactly matches the input audit returned in PR143. This is confirmation of an existing fix; R51 makes no R50 source repair and claims no fresh R50 inference result.

## Publication and execution status

The implementation, CPU gradient guard and real-loader preflight have been checked locally. **The R51 four-batch GPU trial and two full training arms have not run here.** Their results will exist only after the notebook completes and its ZIP passes the return audit.

The notebook's `PUBLISH_R51_SOURCE_COMMIT` and `PUBLISH_R51_DRIVER_SHA256` fields are intentional publication placeholders. Publish the implementation commit first, then replace those two fields with that exact source commit and driver SHA256 in a follow-up notebook publication. The source implementation commit stays fixed even if the notebook publication creates a newer commit. Before release, exercise both first setup and resumed setup against the pinned checkout with the real archived inputs.
