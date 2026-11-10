# Confirm the annotation intervention with seed 46

**Run one fresh seed 46 control/candidate pair with the same annotations.** The [seed 46 GPU notebook](../notebooks/baomai_reviewed_annotations_seed46.ipynb) verifies the inputs, runs a short resource trial, trains both arms and returns `baomai-reviewed-annotations-seed46-results.zip`. No seed 46 training or accuracy result is available at this preparation stage.

This is the single prospective confirmation described in the [original annotation handoff](training-annotations-2026-11-10.md). Its purpose is to test whether the direction observed with seed 45 repeats. The intervention, initialization, training budget, checkpoint selection, labels and evaluation procedure stay fixed. This is not a combined candidate or a search over seeds.

Implementation revision: `23d7db97ca626f5591dbd26e699c0f23d318ee73`. The notebook pins this immutable implementation and checks the full commit before training.

## Fixed inputs and recipe

| Item | Seed 46 confirmation |
| --- | --- |
| Candidate | `annotations-2026-11-10`, unchanged |
| Candidate manifest SHA-256 | `28fc8f5da95605089d61fb062132200ed296c3a10b216f1a61dbda1798ea4ba7` |
| Recipe | `reviewed-annotations-seed46-v1` |
| Recipe SHA-256 | `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8` |
| Initialization, both arms | Original epoch173, SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` |
| Training | Seed 46, 30 epochs, 1280 image size, rectangular loading, batch 4, FP32 |
| Optimizer | AdamW, nominal `lr0=0.0001`, `warmup_bias_lr=0.0001` |
| Remaining numerical settings | Identical to seed 45 |
| Selection | Finalized epoch-30 `fit/weights/last.pt` |
| Trainer | Ultralytics 8.4.182 |

The candidate still has five additions, six deletions and three retained-row modifications in eight training images. Control totals remain 3,772 objects and 6,839 visible endpoints; candidate totals remain 3,771 and 6,841. Both use the original 137 training images and identical original validation: 20 images, 664 objects and 1,227 visible endpoints. The separately versioned PR30 reference is used after training for reporting.

Both arms start independently from original epoch173. The resource trial and the seed 45 models do not initialize either arm. One pair means two full runs, 60 combined epochs and 8,220 nominal image presentations; those are design counts, not duration estimates.

## GPU workflow

Prefer the same **NVIDIA L4** environment used for seed 45 when available: Python 3.13.16, Torch 2.14.1+cu130, torchvision 0.29.1+cu130 and CUDA 13.0. The two arms must share one unchanged source/runtime. No A100-specific requirement exists. If the provider supplies a different GPU or runtime, its recorded identity remains part of the evidence, and the difference must be disclosed when interpreting results across seeds.

The notebook constrains installation to the installed Torch/torchvision versions while installing the pinned trainer. It downloads the exact 157 original PNGs and original checkpoint through Git LFS, verifies all inputs, then performs four real candidate batches under the full recipe. A successful resource trial requires optimizer updates and an observed parameter change. Both full runs start fresh afterward.

The trial has a 600-second stop budget; each full run has a 21,600-second stop budget. The host-memory budget is recorded separately from GPU VRAM. These limits are not duration promises, and a short trial does not guarantee that final validation or finalization fits. Failed runs remain ineligible. Keep their receipts/logs and use fresh output folders for retries. A source/runtime change requires a fresh matched pair.

Return **`baomai-reviewed-annotations-seed46-results.zip`**. It contains both full-run checkpoint bundles, including `last.pt`, receipts, staging manifests, CSV results and logs. The standard ZIP does not include the separate resource-trial receipt/log; retain those in the workspace. The packager rejects incomplete, mismatched or mixed-seed pairs.

## Evaluate each seed separately

After both seed 46 runs finish, infer fresh M0 epoch173, M1 original-label control and M2 annotation candidate. Use one clean, compatible evaluation revision and the same model/runtime settings for all three. Keep the original and revised-reference results separate. Preserve the previously completed seed 45 evidence and replay it with its own pinned compatible source; changed recipe-registry bytes must not be substituted into its recorded source hashes.

The evaluation contract remains:

- Box confidence **0.25 primary**; **0.15 and 0.40 diagnostic**.
- Image size 1280, rectangular FP32 preprocessing, batch 1, NMS IoU 0.70, `max_det=300`.
- Matching IoU 0.50 and endpoint confidence 0.50.
- Both-corner success at **5 and 10 original-image pixels**, retaining all eligible objects, including misses, in each denominator.
- Missed objects first, accompanied by false positives, missing visible endpoints, hidden-side output availability and common-returned-endpoint errors.

The accepted extra-detection tradeoff remains unspecified. The report therefore requires review and does not automatically promote a model. Retain unfavorable results. Report the control-versus-candidate effect for seed 45 and seed 46 separately; do not pool the same 20 development images across runs as independent new observations, choose the favorable seed, or continue to a seed 47 search. Repeated development evaluation is still not independent-scene validation.

### Returned-evidence entry point

Place the two returned bundles under `../annotation-evidence/seed-46/checkpoints/{reference,candidate}` and produce fresh `m0/`, `m1/`, `m2/` and corresponding `-quality/` directories. Set `ANNOTATION_EVAL_REVISION` to the full clean commit used for this seed's inference and quality runs, then run:

```bash
python scripts/rescore_annotation_comparison.py \
  --root . \
  --evidence ../annotation-evidence/seed-46 \
  --candidate data/candidates/annotations-2026-11-10/manifest.json \
  --recipe reviewed-annotations-seed46-v1 \
  --reference-dir data/references/visible-front-2026-11-10 \
  --evaluation-revision "$ANNOTATION_EVAL_REVISION" \
  --output ../annotation-evidence/seed-46/comparison-original-and-pr30.json
```

The wrapper retains strict candidate, recipe, checkpoint, source and chronology checks. It accepts the two explicitly registered annotation recipes, validates the selected recipe against both full receipts, and reports each invocation as one paired experiment. The historical seed 45 recipe and its hash remain unchanged.

## Preparation verification

The [verification record](../results/verification/baomai-annotations-seed46-20261111/verification.json) binds the delivered notebook to the published implementation. All **70 focused compatibility and integrity checks pass**. All seven code cells and the embedded preflight compile; the exact preflight has executed successfully on all **157 original PNGs** and the original epoch173 checkpoint. An independent read-only review confirms that seed46 changes only the seed and preserves the paired source/runtime/recipe guards. The earlier preparation and review records retain their pre-pin state; the final verification resolves that pin.

This is a prepared handoff. No seed46 GPU training, model inference or full Colab execution has been performed here.
