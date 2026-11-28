# Controlled comparison of training annotations

28 November 2026, Asia/Hong_Kong.

**Completed:** PR158 returns both paired training arms. The [independent result analysis](training-label-results.md) reproduces all twelve reference/model/view comparisons. Under reviewed evaluation labels, changing only the training labels gives 32 fewer misses, 34 fewer extras and 31 more five-pixel successes with two-view averaging; duplicates rise from 6 to 13. Both older evaluation references favour the supplied-label arm. This is evidence for the frozen label intervention under the declared recipe, with explicit reference dependence.

The [preparation evidence](evidence/training-label-comparison/README.md) preserves the earlier setup checks. The returned run used the PR156 scoring source, which counts clipped zero-area outputs as unmatched extras. The [return analysis](training-label-results.md) records verified source, checkpoint and paired-training identities, exact score replay and an explicitly retained auxiliary progress-log anomaly.

## The question

If two copies of the same supplied detector are fine-tuned on the same photographs with the same settings, how does replacing the supplied training labels with the current effective training labels change their recognition results?

The [completed reference comparison](annotation-comparison.md) changed the labels used to score fixed predictions. This experiment changes the labels used for learning. The evaluation reference stays fixed within each comparison.

## The two arms

| Condition | Supplied-label arm | Reviewed-label arm |
| --- | ---: | ---: |
| Original training photographs | 137 | 137 |
| Additional photographs | 0 | 0 |
| Labeled cassette fronts | 3,772 | 3,784 |
| Known lower corners | 6,839 | 6,837 |
| Unavailable corner slots | 705 | 731 |
| Empty training photographs | 13 | 13 |
| Development photographs | 20 | 20 |
| Starting checkpoint | Supplied epoch 173 | Same checkpoint |
| Seed | 46 | 46 |
| Epochs | 30 | 30 |
| Image presentations | 4,110 | 4,110 |
| Batches, at batch size 4 | 1,050 | 1,050 |
| Expected optimizer updates | 78 | 78 |

Exactly **92 training label files differ numerically**. Both arms use the same image bytes and split. The reviewed labels are the frozen effective original-photo labels from the current detector's training set. The five supplementary photographs are excluded because they do not have a corresponding original supplied-label condition.

The current written annotation definition is a rulebook. This experiment does **not** silently relabel the photographs to match newly clarified rules. A future experiment testing those rules requires a separately reviewed and frozen label release.

## Fixed training procedure

Both arms use the existing 1280-pixel, FP32 AdamW recipe, including the same augmentations and unmodified native pose loss. A discarded four-batch resource trial checks finite loss, finite gradients and actual parameter updates. Each full arm then starts independently from the same checkpoint.

The runner records the actual initial parameter-state hash, per-epoch sample-order hashes, image presentations, and optimizer-update batch indices. These must agree across arms. Both arms use final-epoch weights. The same original trainer-validation labels are used in both runs, and their metrics do not choose the final epoch, a threshold, or a seed.

This is matched fine-tuning from an already trained checkpoint. It is not a comparison of training two models from random initialization. One paired seed gives a bounded pilot result; it does not measure variability across many independent training seeds.

## Fixed evaluation

Each returned model is evaluated on the same 20 photographs using both its original view and the same original/reflected corner averaging method. Both views are filtered before association. Averaging uses only predicted boxes and points; reference coordinates never choose matches or weights.

The reviewed 654-front reference is primary. The untouched supplied 664-front reference and earlier fixed 666-front reference are retained as sensitivity analyses. They describe the same photographs and must not be pooled as additional test images.

The operating point stays at object confidence strictly above 0.25, corner confidence at least 0.50, matching intersection-over-union at least 0.50, and the established suppression at overlap above 0.70. Report missed fronts, extra predictions, duplicates within the extra count, and both known corners within five and ten pixels. Missed complete pairs remain in the corner-success denominator.

## Run and return

Open [the controlled training-label notebook](../../notebooks/baomai_training_label_comparison_r61.ipynb), select a GPU runtime and run all four code cells. The first two cells verify published source, labels, all photograph bytes, and the shared checkpoint. The third trains the pair; the fourth evaluates and verifies the complete evidence.

Both experiment notebooks share `/content/baomai-r61-cache` by default. Running the model-improvement notebook first lets this notebook reuse its already verified original photographs and checkpoint. `BAOMAI_R61_CACHE` can select another shared cache directory; assets occupy its `assets` subfolder. `BAOMAI_TRAINING_LABEL_BASE` separately selects this experiment's source and results directory. For persistence across runtime restarts, point both settings at already mounted persistent locations.

The download is `baomai-training-label-comparison-r61-upload.zip`. To return through GitHub, extract only the outer ZIP and upload the compact results ZIP, return manifest and every small wrapped checkpoint part. The manifest identifies each complete model and every wrapper/member hash. Saved scores and corner associations are independently recomputed before packaging is accepted.

Completed arms can be reused under the same source, inputs and runtime. Incompatible or incomplete work is preserved. No automatic model selection or deployment is performed.

## Verification status

The local input preflight has verified all 157 original photographs, the shared supplied checkpoint, all staged labels, and the exact treatment counts above. Native loaders reproduce every staged label coordinate and count. Sixteen sampled augmented images per arm agree pixel-for-pixel after the same seed reset. Focused checks reproduce the current two-view method's saved scores under all three references, reject missing prediction views, and exercise multipart checkpoint reconstruction and corruption rejection. The notebook code cells are syntax checked.

GPU training and comparative accuracy are complete and independently replayed from saved predictions. The result supports a conditional training-label effect under each fixed reference. These development photographs have already guided annotation review and method selection; a new capture-group evaluation is a separate question.

- [Experiment driver and shared evaluator](../../scripts/run_training_label_comparison_r61.py)
- [Notebook builder](../../scripts/build_training_label_comparison_r61_notebook.py)
- [Focused checks](../../tests/test_training_label_comparison_r61.py)
- [Preflight evidence](evidence/training-label-comparison/README.md)
