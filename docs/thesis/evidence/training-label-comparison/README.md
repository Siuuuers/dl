# Controlled training-label experiment: preparation evidence

The experiment is prepared; **no GPU training result is recorded here**.

The local checks used the real 157 original PNGs and the supplied epoch-173 checkpoint. Every image and checkpoint hash matched its frozen manifest. Both staged arms contain the same 137 training and 20 development photographs. The 92 differing training-label files all differ numerically; the split, images and trainer-validation labels are unchanged.

| File | Evidence |
| --- | --- |
| `plan.json` | Exact two-arm inputs, label intervention, three evaluation references, recipe and equal training budget |
| `preflight.json` | Full asset staging and label-count checks; explicitly records no training or new inference |
| `native-loader-preflight.json` | All native loader label coordinates/counts match the staged files; 16 sampled augmented images per arm match pixel-for-pixel after resetting the same seed |
| `local-validation.json` | Implementation hashes, three focused checks, and four notebook code-cell syntax checks |
| `manifest.json` | Byte counts and SHA-256 inventory for these files |

The checks additionally replay the already saved current-model predictions through the new shared evaluator. The two-view output reproduces five-pixel success of 514/584 on supplied annotations, 525/584 on the earlier corrected annotations, and 569/583 on reviewed annotations. The test uses identical saved inputs in both dummy arm names to test the evaluator; it is not a new trained-model result.

Multipart packaging is exercised with a synthetic byte fixture crossing the 16 MiB split boundary. Independent reconstruction agrees exactly, and a corrupted wrapper is rejected. That fixture is not a model and is not included here.

The source identity in these local receipts distinguishes unpublished development verification from a later clean published-notebook execution. The GPU resource trial, both full training arms and their resulting accuracy remain pending.

See [the experiment guide](../../training-label-comparison.md) for the controlled question and run instructions.
