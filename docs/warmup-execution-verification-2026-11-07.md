# Bias-warmup implementation: verified and ready for GPU execution

**The isolated recipe and matched-result handoff are implemented. Both bounded CPU trials pass from clean published source `679b87b38e7b0bca5fbd02fb1519a90fb2379541`. Full GPU training and model-quality evaluation remain pending.** Keep the original epoch173 model in use.

Use the [GPU handoff and Colab notebook](gpu-warmup-handoff.md) for the next experiment. The [machine-readable verification](warmup-execution-verification-2026-11-07.json) and [raw evidence ZIP](../results/verification/baomai-warmup-001/local-verification.zip) preserve the implementation checks and actual trial records.

## What changed

The original `RECIPE` is unchanged and remains the CLI default. The new `warmup-bias-lr-1e-4-v1` recipe changes only `warmup_bias_lr` from `0.1` to `0.0001`, and requires reference labels. Both recipes retain the 30-epoch scheduler, original initialization, batch size, image size, augmentation, loss settings and gradient accumulation.

New schema-2 receipts and checkpoint manifests identify the recipe and its canonical configuration hash. Validation checks all 114 resolved trainer arguments, including RLE gain `1.0`. The runner records actual GPU identity, effective memory layout and numerical backend settings after setup without changing them. It also records named learning rates for early real optimizer updates.

The new result packer accepts one fresh original/reference control and one fresh lower-bias/reference run. It checks identical source, runtime and reference input content, with only the declared warmup-rate difference. It preserves final-epoch checkpoints and complete logs in one verified return ZIP. The previous label-comparison packer remains available for its original purpose.

## Actual clean-source CPU trials

Both trials use Python 3.12.14, CPU PyTorch 2.10.0, Ultralytics 8.4.182, four Torch threads, batch 4 and image size 1280. Each retains the declared 30-epoch schedule and stops deliberately after four training batches. The sampled host-memory cap is 8,000 MiB; the wall-time cap is 600 seconds.

| Observed result | Original control | Lower-bias warmup |
| --- | ---: | ---: |
| First bias-group learning rate | 0.1 | 0.0001 |
| Completed batches | 4 | 4 |
| Image presentations | 16 | 16 |
| Actual optimizer updates | 4 | 4 |
| Sampled updates with a changed witness parameter | 4 | 4 |
| Completed epochs | 0 | 0 |
| Parent wall time | 28.50 s | 25.89 s |
| Sampled process-tree peak host RAM | 6,849.80 MiB | 6,802.70 MiB |
| Exit code / receipt status | 0 / `trial-complete` | 0 / `trial-complete` |

Both loaders verified 137/20 images, 3,772/664 objects and 6,839/1,227 visible endpoints. Independent read-back rechecked the original checkpoint bytes, all 628 staged image/label copies per run, both staged YAML identities, logs, recipe declarations, runtime records and all six recorded source-file hashes.

The first four batch image IDs, shapes and object counts match between trials. Their first forward loss and every component are exactly equal before either optimizer update. The only scientific settings difference is `warmup_bias_lr`; the remaining differences are run paths. This supports the intended isolation. Subsequent early losses are preserved in the evidence, but four training batches do not establish improved validation accuracy.

Neither trial executes validation or final checkpoint finalization. Calling the real checkpoint-manifest writer on each trial receipt rejects it as ineligible, writes no manifest, and leaves the receipt unchanged. The observed CPU memory and duration do not qualify a complete GPU run or later validation/finalization phases.

## Compatibility and focused checks

All **96 focused checks** pass on the clean published implementation: 36 evaluation/scoring, 12 corner-quality, 21 checkpoint/provenance, 8 runner/process, 7 original-packer and 12 warmup-packer checks. Successful full-bundle cases in these checks use synthetic fixtures; they are not evidence of completed new model training.

Both real PR #13 checkpoint bundles retain exactly the same serialized selected identities as under historical comparison code `484f5671456f004e4198173537fd847e0a56db90`. No recipe fields were added to legacy identities. The historical full-report replay still belongs to its recorded revision because that replay also checks implementation hashes.

The notebook's code cells compile without executing GPU operations. Two small local subprocess checks exercise interrupted-worker cleanup and a failed launcher leaving a detached child. Packaging and downloading are separate notebook cells so a blocked download can be retried without rerunning training. Fetching the exact source commit into a fresh shallow checkout was verified. Colab setup, CUDA observations and the complete 30-epoch pair still require the user's GPU runtime.

An earlier development probe completed four worker batches but was correctly rejected by the parent after source status changed during concurrent editing. Its failed receipt and raw records are retained separately in the ZIP; the passing results above come from the later clean published checkout.

## Next measurement

Run the notebook's fresh control and warmup recipes in one unchanged GPU session and return `baomai-warmup-results.zip`. Then evaluate M0 and both new final-epoch models together with the unchanged scorer and predeclared five-metric decision rule. The new warmup recipe must pass against both M0 and the matched control before adoption. The present result establishes runnable, traceable experiment support; it does not establish a replacement model.
