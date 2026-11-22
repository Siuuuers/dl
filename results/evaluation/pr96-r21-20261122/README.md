# PR96 / r21 returned-model audit

**Recommendation: retain PR35. No model was promoted.** The returned r21 candidate has more matching misses and fewer both-corner successes on the unchanged 666-front primary reference than either the fresh control or retained PR35. Its ranking reverses on the separately pinned pre-return r21 review. Exact identity tracing and original-pixel inspection show substantial reference-box sensitivity and residual rack/occluder errors. The 35 primary-reference misses must not be described as 35 physically absent detector outputs.

No r22 annotations were loaded. The prior PR94 evidence remains unchanged.

## Fixed measurements

All three models used the same CPU inference runtime and operating point: input 1280, box confidence 0.13, endpoint confidence 0.50, NMS IoU 0.70, matching IoU 0.50, FP32, maximum 300 outputs, and four threads. The pair-success denominator is 584 fronts with two locatable corners; eligible missed objects remain failures. An unmatched output is a matching outcome, not an independently confirmed background false positive. Duplicates are a subset of unmatched outputs.

| Reference | Model | Matched | Unmatched targets | Unmatched outputs | Duplicates | Both corners <=5 px | Both corners <=10 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen 666 primary | Retained PR35 | 648 | 18 | 80 | 32 | 533/584 | 582/584 |
| Frozen 666 primary | Fresh r21 control | 646 | 20 | 83 | 29 | 531/584 | 583/584 |
| Frozen 666 primary | r21 candidate | 631 | 35 | 100 | 27 | 518/584 | 574/584 |
| Historical 663 | Retained PR35 | 648 | 15 | 80 | 32 | 533/584 | 582/584 |
| Historical 663 | Fresh r21 control | 646 | 17 | 83 | 29 | 531/584 | 583/584 |
| Historical 663 | r21 candidate | 632 | 31 | 99 | 27 | 518/584 | 574/584 |
| Pre-return r21 / 662 exploratory | Retained PR35 | 625 | 37 | 103 | 21 | 545/584 | 573/584 |
| Pre-return r21 / 662 exploratory | Fresh r21 control | 624 | 38 | 105 | 16 | 541/584 | 573/584 |
| Pre-return r21 / 662 exploratory | r21 candidate | 648 | 14 | 83 | 27 | 555/584 | 583/584 |

All three models return zero outputs on all three empty scenes. On the primary reference, the candidate misses 13 zero-corner, 14 one-corner, and 8 two-corner fronts. On the r21 exploratory reference it misses 7 zero-corner and 7 one-corner fronts; all 584 two-corner fronts match. Candidate miss-free nonempty scenes are 4/17 on the primary reference and 10/17 on the r21 review. These reused development images do not establish reliability in unseen capture sessions.

Against the fresh control on the primary reference, the candidate gains 2 and loses 17 matched identities. Against PR35, it gains 2 and loses 19. Exact identities, best overlaps, predictions, and matching traces are retained in [fixed-comparison.json](fixed-comparison.json). The primary and historical references retain 8 and 13 unresolved rows respectively, included in their scores. The r21 ledger has no unresolved development rows; that status does not certify every physical contour.

### What changed from the PR94 r20 candidate?

On the same frozen primary reference, the r21 candidate retains 631 matches and 35 unmatched targets. Unmatched outputs decrease from 103 to 100 and duplicates from 29 to 27. Both-corner successes within 5 px decrease from 520 to 518, while the 10 px count stays 574. These are mixed changes and do not establish a replacement model.

PR35 and the newly trained control each reproduce the PR94 prediction arrays exactly on all 20 images, despite new checkpoint-file hashes for the newly packaged control. The r21 candidate changes predictions on all 17 nonempty images; only the 3 empty-image prediction arrays remain identical. This is direct output comparison, not a claim that checkpoint serialization bytes are identical.

Do not compare the old r20 exploratory 5 px count with the new r21 exploratory count as a pure training effect: the endpoint reference coordinates also changed between those review versions. The frozen-reference comparison above avoids that reference change.

## Why the reference ranking changes

[reference-sensitivity.json](reference-sensitivity.json) follows instance identities through the hash-bound recovery and every archived review row map, replaying the corresponding edits. It does not assume equal row numbers across versions or infer identity from nearest-box matching. Of 666 primary identities, 660 survive and 6 are removed; 2 distinct identities are added, giving 662 r21 targets.

The candidate's 35 primary-reference misses divide into:

- **16 reference-box crossings:** the exact same prediction has IoU below 0.5 against the fixed box and at least 0.5 against the r21 box for the retained identity.
- **6 deleted primary identities:** these no longer appear in r21.
- **13 retained identities still unmatched** in r21.

One previously matched retained identity becomes unmatched after tightening, while both added identities match. Thus `631 + 16 - 1 + 2 = 648` matches, without changing any model output. This diagnoses the metric's sensitivity; it does not independently prove either reference correct.

Representative original-pixel inspection supports that distinction:

| Same r21 candidate output | Fixed-primary to r21 IoU | Observation |
| --- | ---: | --- |
| 057 primary row 12 to r21 row 11 | 0.473685 to 0.767729 | A visible output lies on the same cassette front. The fixed box extends into dark material below it. The unchanged corner errors are 0.338 px and 0.723 px. |
| 090 row 41 | 0.470050 to 0.578489 | The fixed box includes a broad rack strip. The tighter reference gives a match, but the prediction still extends into that strip and puts its right point over rack material. Only the left reference corner is locatable. |
| 010 row 32 | 0.185174 to 0.538225 | A hand occludes most of the front. The fixed box spans the hand; the r21 box covers a much narrower visible sliver. Both corners are unlocatable, and the prediction still extends into the occluder. |
| 011 primary row 58 to r21 row 57 | 0.513749 to 0.284208 | The converse case: reference tightening exposes an oversized output extending into the gray rack. The locatable left point is 2.636 px from the r21 point. |

The four raw/overlay triptychs are in [representative-cases](representative-cases), with source hashes and crop coordinates in [representative-image-inventory.json](representative-image-inventory.json), and observations in [representative-visual-notes.json](representative-visual-notes.json). A box match does not certify a complete physical contour or a valid hidden corner.

The historical reference also differs in geometry, not just object count. In 010 row 14, the same candidate output has IoU 0.604139 against the historical box and 0.385712 against the primary box. This explains the extra historical match in the table; the comparison is recorded in [historical-reference-difference.json](historical-reference-difference.json).

## Returned archive and paired training

The audit reads [PR96](https://github.com/Siuuuers/dl/pull/96) at head `c67a753bd7b35560ea61555c0927eba7e9aca21c`. Its seven changed files return the ZIP, two chunk ZIPs, checksums and documentation. The full ZIP was available as an ordinary Git blob: 39,783,675 bytes, SHA256 `b3acac74a82dc5927242aa73b96b4c33978fee06aa1b1678b6382b237c91b802`. CRC, safe unique paths, the exact 13-member inventory, and all 12 indexed payload hashes and sizes passed. Chunk reconstruction was unnecessary because the exact full archive was directly available.

Training source is `c4fb76afe280552ed13ed3c9b78608deceb7b3c4`; protocol SHA256 is `6e79440caa989de5327531ebde2242aa3ba7f25de7497ebac8542ee766400122`. The notebook at `41dd2be835f5b0e30b10e15431c6795cb00f7388` has SHA256 `1de82d9dbfb53e331048f132d96df01272e2e732d48b1454bbf12c820b005f7a` and pins that source and protocol exactly.

Both arms independently load the original epoch-173 initializer and complete 30 epochs, 1,050 batches, 4,110 image presentations, and 78 optimizer steps, exiting with code 0. They use recipe `reviewed-incremental-fronts-seed46-v1`, seed 46, batch 4, input 1280, FP32, AdamW at initial learning rate 0.0001, final-epoch checkpoint selection, and the same NVIDIA L4 runtime. Wall times are 460.968554938 s for the control and 456.98501135 s for the candidate. PR35 trained on an A100, so its difference from the fresh control cannot be attributed solely to training labels.

All 157 original images, initialization weights, staged source bytes, source-script hashes, and completed-arm receipts were verified. Exactly 92 training-label files differ between arms. Original 664-front trainer validation is identical. Photo 015 and the 31 old-series photos retain exact PR35 supervision in both arms. Control training contains 3,771 fronts / 6,841 locatable endpoints; candidate training contains 3,784 fronts / 6,838 locatable endpoints.

| Checkpoint | SHA256 |
| --- | --- |
| Retained PR35 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Fresh control | `3b6054b0da543752ba7202995bce8834dc5daf02ea7e2463ea409da365e6acc6` |
| R21 candidate | `a14467eb1e15258ab8a5f473e1f1abd600766e11d068dba4cd041714384de1d4` |

No training was repeated. CPU inference uses Python 3.12.14, Torch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0, and OpenCV 5.0.0.93. The isolated inference checkout records 158 materialized LFS image/weight paths as modified because it has no LFS clean filter; each LFS oid, size, and original manifest hash was reverified. Source scripts and annotation files are unchanged. [Inference provenance](inference-provenance.json) retains the runtime and repetition checks.

## Reproduction

- [audit_return.py](audit_return.py) verifies the exact original ZIP and extracted bundles against a checkout of the pinned source with hydrated inputs. The pinned notebook commit must also be available as a Git object.
- [run_pr96_inference.py](run_pr96_inference.py) repeats the three fixed CPU inferences when complete checkpoint bundles and the pinned runtime are supplied. No training or threshold search is performed.
- [measure_pr96.py](measure_pr96.py) reproduces all three reference scores from the saved predictions and text receipts without loading model weights. Supply `--root`, the three prediction paths, the three `--*-manifest` paths, and a fresh `--output`.
- [reference_sensitivity.py](reference_sensitivity.py) reproduces identity and IoU tracing using `--root`, `--evidence`, and a fresh `--output`.
- [render_representative_cases.py](render_representative_cases.py) recreates the four comparison figures from the pinned source images and saved sensitivity cases.

[measurement-adaptation.patch](measurement-adaptation.patch) records the minimal PR94-to-PR96 pin/import changes. Logs are byte-preserved as `train-log.txt`; the scoring runner restores their original `train.log` name in a temporary directory for strict checkpoint validation. [replay-verification.json](replay-verification.json) confirms that the saved predictions and receipts reproduce the comparison byte-for-byte. Raw checkpoint binaries and return archives are not duplicated in this evidence directory.
