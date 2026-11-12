# One matched training comparison for the fourteen box corrections

**Completed: retain PR35; no further GPU run is requested.** The matched 30-epoch L4 runs returned in [PR47](https://github.com/Siuuuers/dl/pull/47) pass verification. The [completed evaluation](consistency-results-2026-11-12.md) finds that the fourteen-box candidate does not meet the declared promotion criterion. Keep the selected PR35 checkpoint and its [reviewed-proposal workflow](predict-pose-folder.md#assistant-reviewed-proposals) at **1280 / box confidence 0.13 / NMS 0.70 / endpoint cutoff 0.50**. The notebook, immutable pins and commands below preserve the completed experiment's provenance.

## Completed result

At primary confidence 0.13 on the fixed 663-object reference, with 584 objects eligible for both-corner scoring:

| Model | Matches | Misses | Scored extras | Both corners within 5 px | Both corners within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current PR35 | 643 | 20 | 85 | 529 | 577 |
| L4 old-reviewed control | 641 | 22 | 88 | 527 | 578 |
| L4 consistency candidate | 642 | 21 | 88 | 530 | 578 |

The candidate's one additional scored match over its L4 control is a near-identical existing proposal crossing IoU 0.50. It also loses existing 5 px pair successes. Original-image review establishes no additional recovered front; the candidate retains one more scored miss and three more extras than PR35. The current PR35 checkpoint was trained on an A100, so differences from the L4 control cannot be attributed to the fourteen-box label delta. See the [full result and identity review](consistency-results-2026-11-12.md) for the unchanged-reference comparison, corner tradeoffs and proposal evidence.

## Archived Colab handoff (completed)

The following steps document the completed handoff. The verified ZIP is already available in PR47; use the completed report's replay instructions to inspect it. Another training run is unnecessary.

1. Open the [final handoff notebook](../notebooks/baomai_consistency.ipynb) in [Google Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/consistency-study-20261112/notebooks/baomai_consistency.ipynb).
2. Select **Runtime → Change runtime type → GPU**, then **Runtime → Run all**. Use the same GPU/runtime for the whole pair and approve the Drive mount so completed work can be recovered.
3. Let the notebook verify the inputs, complete its four-batch resource trial, and run both 30-epoch arms. It then verifies and downloads **`baomai-consistency-seed46-results.zip`**.
4. Return that complete ZIP. Keep the receipt, staging manifest, log, results CSV and checkpoint manifest beside each model; the notebook packages these together.

The dedicated Drive folder is `MyDrive/baomai-consistency-seed46-20261112`. Completed bundles and the final ZIP are backed up there. The notebook restores and verifies compatible completed arms before starting a missing run; when both arms are complete, it skips the trial and training. It can also recover a matching completed resource trial. An interrupted local arm is preserved and rejected as incomplete: use a fresh `RUN_BASE` in the notebook and recover the completed Drive artifacts. A changed runtime cannot silently be combined with a recovered arm.

The notebook checks out implementation commit [`14561b7f01cc89c4472e11ffb37b3943ea3e34f9`](https://github.com/Siuuuers/dl/commit/14561b7f01cc89c4472e11ffb37b3943ea3e34f9). Its protocol SHA-256 is `da692f2261fae441dfa7d68b3f86ed850acbb861c75e86e8c54f40b7184e540b`. That immutable implementation contains the code, fixed protocol and PR45 labels. The later handoff notebook supplies the completed source pin; use the notebook linked above, rather than the intentionally unpinned draft inside the implementation commit. Both training receipts must name the same clean implementation source.

## The question tested

The six-image review found boxes that omitted exposed front material or included hidden width and neighboring structures. Four accepted replacements expand to visible material; ten remove unsupported or non-target extent. This experiment tests that **combined fourteen-box change**. It cannot separately attribute an outcome to expansions or contractions.

| Role | Training labels | Initialization and selected output |
| --- | --- | --- |
| `old_reviewed` | Exact eight-file `annotations-2026-11-10` overlay selected for PR35 | Fresh original epoch 173; finalized epoch 30 `last.pt` |
| `consistency_reviewed` | The same eight files plus PR45's fourteen replacements in three further files | Fresh original epoch 173; finalized epoch 30 `last.pt` |

Both inner runner arms are `candidate`, because both use an annotation overlay. Neither full run resumes PR35, a PR43 checkpoint, the resource trial or the other arm. A fresh control is needed to compare the label change within this run's source and runtime; the existing PR35 model was also evaluated as the operational comparator after return.

| New-delta training image | Accepted original row numbers, one-based | Replacements |
| --- | --- | ---: |
| `new_000018` | 24, 25, 26, 32, 33, 34 | 6 |
| `new_000034` | 55, 56, 57 | 3 |
| `new_000049` | 9, 14, 24, 25, 26 | 5 |

The reviewed `old_000007`, `old_000057` and `new_000132` remain unchanged. Nine held complete-box proposals, unresolved identities and every new-delta class/corner token remain unchanged. No object is added or removed. The candidate exactly inherits the selected PR35 overlay and does not accumulate the non-adopted PR40 or PR43 extensions.

The target remains an identifiable **observed cassette front and its two actual lower-front corners**. The accepted expansions reach exposed pixels; they are not estimates of full hidden boxes, concealed corner positions or physical dimensions. Missing corners remain missing. The [annotation policy](annotation-policy.md) and [PR45 adjudication](../results/review/baomai-training-consistency-20261111/adjudication.json) retain those distinctions.

## Fixed recipe and data

The [protocol](../experiments/consistency-2026-11-12.json) declares experiment `consistency-seed46-v1` before training. It uses the existing `reviewed-annotations-seed46-v1` recipe: **seed 46, 30 epochs, batch 4, image size 1280, rectangular batches, FP32, AdamW, learning rate 0.0001 and bias warmup 0.0001**, with the existing losses and augmentations. Ultralytics stays pinned to 8.4.182. The four-real-batch resource trial uses that full recipe schedule, stays separate from both full initializations and cannot produce an eligible selected checkpoint. The notebook requires CUDA.

The architecture, all 157 original images and original 137/20 split remain fixed. Both effective training variants contain **137 images, 3,771 objects, 6,841 visible endpoints, 701 missing endpoint slots and 13 empty images**. Only the fourteen box coordinates differ between their effective datasets.

Original in-training validation remains **20 images / 664 objects / 1,227 visible endpoints / 101 missing slots / three empty images**. The separately corrected **663-object reference** is reserved for post-training scoring. Its labels are never substituted into training validation. These are different versioned references and their denominators stay separate.

## Predeclared improvement criterion

The completed evaluation ran both returned models and the current PR35 comparator in the same inference environment. Primary box confidence stays **0.13**; **0.15** is the sole historical diagnostic. Keep input size 1280, NMS 0.70, matching IoU 0.50, endpoint cutoff 0.50 and the frozen [663-object reference](../data/references/visible-front-2026-11-11/manifest.json). Report the original-reference results separately.

Report missed objects and extras, every gained/lost object identity, both-corner successes within 5 and 10 px, zero/one/two-locatable-corner groups, points returned on unlocatable sides, miss-free scenes, all three empty-image outputs and review workload. Inspect changed matches and newly added proposals against the original images. Preserve inherited reference uncertainty, including the separate `011:61` caveat.

The declared criterion asks for **supported additional recovery without losing previously supported matches or existing 5/10 px pair successes, and without adding empty-scene detections**. Additional scored extras require explicit review. A score increase confined to uncertain references, an IoU-boundary crossing or reassignment among existing proposals does not establish newly recovered physical objects. Retain PR35 unless the measured evidence supports a separate documented change. No threshold, seed, epoch or annotation-subset search follows automatically.

The [completed PR43 result](visible-extent-results-2026-11-11.md) already showed why more faithful labels and improved learned weights are separate claims. This is a different fixed label candidate, not a rerun of that completed experiment. One matched pair can reveal its effect under this recipe. Repeated measurements on the same 20 development views cannot establish reliability on unseen capture sessions.

## Return validation and preparation evidence

The new packager uses the existing completed-checkpoint validators and adds this study's exact candidate identities and fourteen-row contract. It checks clean source, matching runtime/settings, the frozen image/label staging inventory, unchanged validation, correct roles and completed epoch 30 selection before creating a ZIP. It directly verifies the local validation PNGs; the earlier notebook input and staging checks verify all 157 actual originals. It then reads the archive back and verifies its inventory and hashes.

```bash
python -B scripts/package_consistency_results.py \
  --protocol experiments/consistency-2026-11-12.json \
  --old-reviewed /path/to/old_reviewed \
  --consistency-reviewed /path/to/consistency_reviewed \
  --output /path/to/baomai-consistency-seed46-results.zip
```

Use the same clean pinned implementation that produced both runs. Add `--verify-existing` to verify an existing ZIP without rewriting it. The notebook runs this command for you.

The [preparation verification record](consistency-preparation-verification-2026-11-12.json) separates actual input/provenance checks from synthetic packaging and recovery tests. The focused check command is:

```bash
python -B scripts/check_package_consistency_results.py
```

These preparation checks load no trained model and perform no GPU training or inference. They established input integrity and workflow behavior before the experiment. The later [completed evaluation](consistency-results-2026-11-12.md) separately verifies and measures the returned models; it retains PR35. The next no-data work is a bounded review of unreviewed training material for consistency and completeness, with unresolved cases preserved and no automatic training or parameter search.
