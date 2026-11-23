# PR103: returned R26 candidate evaluation

**Retain PR35.** The returned R26 candidate does not establish a replacement on the frozen primary benchmark. Against the unchanged R26 exploratory review, R26 and R25 miss exactly the same 15 targets; R26 adds one unmatched output and has one more both-corner success within 5 pixels. This is a small, mixed change, with a measured training-runtime confound.

This directory preserves the returned-model result **before any R27 annotation amendment**. The visual report diagnoses the returned **R26 candidate**, not PR35. Subsequent R27 annotation changes are cross-referenced on affected PDF cases without changing the R26 scores, row identities, or selection.

## Verified return

[PR103](https://github.com/Siuuuers/dl/pull/103), head `c3cda55dd92dd2c0d85454351f0e06865b5eab73`, returns `results/gpu/baomai-front-review-r26-a100-001/baomai-front-review-r26-results.zip`.

| Item | Verified identity |
| --- | --- |
| Archive | 39,784,715 bytes; SHA-256 `105e86cc793378f1a47dd7cebc49ef84170d4f00ca9bec8c1e5f4e467df4b69d` |
| Training source | `5c0a378dccfa16392eb3951531312efa3378889b` |
| Notebook commit | `cb035d381568ded57917d274385e0fd560c6d6f6` |
| Notebook SHA-256 | `37fe6bb66ce1c4d833b228466b3bce24769c42fa1b383d1384f465ca49561d4d` |
| Protocol SHA-256 | `232f81cdf6d367fc9ece902e01b4ab5a03da5ae0898f5f75abae25ef5001372f` |
| R26 control checkpoint | `fcc2708241cbdd57a0e6e1a0440cd5928ca829eb764cba9770281447446fee61` |
| R26 candidate checkpoint | `a2e71037a9995ed864ec21ab5c03a719e200301766788b88765c745638e7ad9a` |

The ZIP inventory and all 12 payload hashes passed. Both arms completed 30 epochs, 1,050 batches, 4,110 processed training images, and 78 optimizer updates, with exit code 0 and the preselected final-epoch checkpoint. Actual staged images, labels, exclusions, original initializer, source hashes, notebook pins, recipe, and paired runtime were checked. The 20-image/664-front original trainer validation remained unchanged. No training was repeated. See [return-audit.json](return-audit.json).

## Same-runtime comparison

All five checkpoints were freshly inferred on the same 20 original development images with one CPU environment: Ultralytics 8.4.182, PyTorch 2.14.1+cpu, four threads, input 1280, box confidence 0.13, endpoint confidence 0.50, NMS IoU 0.70, and matching IoU 0.50. All three empty scenes have zero outputs for every model. The scorer is unchanged; there was no threshold, seed, or epoch search.

“Both within” counts targets with two locatable reference corners whose two returned predictions are within the stated original-image pixel radius. Eligible missed targets remain in the denominator, which is 584 for each of these references. “Unmatched outputs” describes the scorer assignment outcome; it does not independently establish that every output is background.

### Frozen primary reference: 666 fronts

| Model | Misses | Unmatched outputs | Duplicate subset | Both within 5 px / 584 | Both within 10 px / 584 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 18 | 80 | 32 | 533 | 582 |
| R25 control, L4 | 20 | 83 | 29 | 531 | 583 |
| R26 control, A100 | 18 | 80 | 32 | 533 | 582 |
| R25 candidate, L4 | 37 | 108 | 29 | 518 | 574 |
| **R26 candidate, A100** | **36** | **108** | **27** | **519** | **573** |

R26 has 18 more misses and 28 more unmatched outputs than PR35, with fewer both-corner successes at both radii. Against R25, it gains the frozen-reference match `new_000089:49`, gains two and loses one 5-pixel success, and loses the 10-pixel success `new_000008:20`. These differences do not support a general improvement claim.

### Historical diagnostic reference: 663 fronts

| Model | Misses | Unmatched outputs | Both within 5 px / 584 | Both within 10 px / 584 |
| --- | ---: | ---: | ---: | ---: |
| Retained PR35 | 15 | 80 | 533 | 582 |
| R25 control | 17 | 83 | 531 | 583 |
| R26 control | 15 | 80 | 533 | 582 |
| R25 candidate | 34 | 108 | 518 | 574 |
| R26 candidate | 32 | 107 | 519 | 573 |

This reference has its own manifest and row identities. Its totals must not be merged with the frozen primary or current review.

### Unchanged R26 exploratory review: 661 fronts

| Model | Misses | Unmatched outputs | Duplicate subset | Both within 5 px / 584 | Both within 10 px / 584 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 38 | 105 | 20 | 543 | 573 |
| R25 control | 40 | 108 | 15 | 539 | 573 |
| R26 control | 38 | 105 | 20 | 543 | 573 |
| R25 candidate | 15 | 91 | 33 | 557 | 582 |
| **R26 candidate** | **15** | **92** | **33** | **558** | **582** |

R25 and R26 have identical matched target identities on this same reference. R26 gains 5-pixel successes at `new_000059:2` and `new_000090:19` and loses `new_000090:46`; their 10-pixel success identities are identical. A changed annotation reference can change scores without changing model predictions. The more favorable current-review ranking is useful diagnostic evidence, but it does not replace the preregistered frozen result or provide a new-session test.

The returned R26 candidate matches all 584 targets with two locatable corners. Its remaining 15 misses comprise six targets with no locatable corners and nine with one locatable corner:

| Photo | Missed one-based R26 rows |
| --- | --- |
| `new_000008` | 33, 34 |
| `new_000009` | 55, 56, 57 |
| `new_000010` | 14, 33, 34 |
| `new_000011` | 55, 56, 57, 60 |
| `new_000053` | 1 |
| `new_000086` | 57 |
| `new_000088` | 57 |

The concentration in fronts with fewer locatable corners is an observed error pattern. It does not establish whether any individual old annotation, physical target, training example, or suppression decision is responsible.

## What changed between R25 and R26

The actual candidate training inventories differ in exactly one label file: `new_000013`, rows 29-31. Three left corners changed; the left box bounds of rows 29 and 30 also changed. Class, visibility, right-corner tokens, object counts, original images, and trainer validation are preserved. Development-review changes did not enter training. The reference/control inventories are identical across rounds.

The recorded recipe and software versions also agree: original epoch173 initializer, seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW initial learning rate 0.0001, and final-epoch selection. Among the recorded runtime fields, the GPU name and capability differ: R25 used NVIDIA L4 (8.9), R26 used NVIDIA A100-SXM4-40GB (8.0). The only changed recorded source files add R26 to the verifier/hash registry; the training implementation is otherwise unchanged in those recorded files.

**The unchanged controls nevertheless differ in 706 of 879 saved tensors.** Their scores change too: on frozen 666, misses change from 20 to 18 and unmatched outputs from 83 to 80. The R26 control reproduces all 20 PR35 prediction arrays exactly in the fresh common inference runtime, excluding timing fields. See [crossround-input-verification.json](crossround-input-verification.json) and [control-output-verification.json](control-output-verification.json).

Consequently, R25-to-R26 candidate differences cannot be assigned solely to the three corrected rows. Running inference in one environment removes an inference mismatch but does not erase the different training histories. R26 versus its own A100 control tests accumulated eligible supervision; it is not an isolated three-row trial. These single runs also do not estimate run-to-run variability or reliability on unseen capture sessions.

## Complete visual inventory

[candidate-error-inventory.json](candidate-error-inventory.json) retains all 646 matched targets, all 15 missed targets, and all 92 unmatched outputs with source hashes and full-precision geometry. The PDF contains 61 cases on 32 pages:

- All 15 scored misses, with the best retained overlapping box where available and its actual assignment/IoU explanation.
- All 32 matched corner-failure targets, covering 32 returned locatable-corner errors over 5 pixels and one locatable corner not returned at the fixed threshold. The latter is `new_000008:1`.
- The eight worst remaining matched box overlaps below 0.70, without repeated corner-failure targets.
- The six highest-confidence unmatched outputs, labelled by the scorer as duplicate or other unmatched.

The PDF uses the returned R26 candidate and unchanged R26 labels. Rows subsequently amended in R27 carry a cross-reference to the separate annotation report; a scored disagreement with an old label is not automatically a physical model error. Unlocatable reference corners have no fabricated coordinate. No corner distance is assigned to a missed target or unmatched output. Saved post-NMS outputs do not identify a raw suppression cause.

## Evidence and reproduction

- [fixed-comparison.json](fixed-comparison.json): all five models, three independently identified references, per-photo results, match traces, gained/lost targets, and corner-success identities.
- [return-audit.json](return-audit.json): archive, source, notebook, runtime, training receipt, initializer and staged-input checks.
- [inference/](inference/): fresh prediction arrays, original-reference scorer outputs, completion records, logs and actual inference commands.
- [checkpoint-receipts/](checkpoint-receipts/): exact checkpoint manifests, training/staging receipts, epoch CSVs, and preserved logs. Weight bytes remain in their source archives.
- `audit_return.py`, `run_pr103_inference.py`, `measure_pr103.py`, `verify_crossround_inputs.py`, and their adaptation patches preserve the small changes to the established PR101 pipeline.
- [replay_evidence.py](replay_evidence.py) reproduces the fixed comparison from saved predictions without inference or training. [replay-verification.json](replay-verification.json) records the exact-byte check.
- [render_pr103_failures.py](../../../scripts/render_pr103_failures.py) renders the PDF and verifies every image/reference hash and match trace. [pdf-verification.json](pdf-verification.json) records its layout and coverage checks.

The `inference/eval-*/summary.json` files score the original **664-front trainer reference** as part of the established inference harness. Use `fixed-comparison.json` for the separately versioned 666/663/661 results above.

The next useful work is to resolve physically supported annotation disagreements, preserve this unamended result, and verify whether any accepted amendments change eligible training inputs. A later training hypothesis should be stated separately from development-label corrections. If the next comparison is intended to examine only incremental supervision, keep the actual accelerator and software environment fixed across rounds and compare the unchanged controls. No additional inference tuning or model promotion follows from this return.
