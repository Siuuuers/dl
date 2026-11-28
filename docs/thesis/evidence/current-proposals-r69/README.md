# R69: complete proposals from the current reviewed checkpoint

This completed diagnostic captures the frozen PR158 reviewed-label model before confidence filtering and suppression, for all 20 development photographs in both views. It closes the R68 evidence gap for this checkpoint: every decoded box, score and two-corner prediction is available, including candidates that never reach the saved output. No training, optimizer update, label change or threshold search is involved.

The [four-cell notebook](../../../../notebooks/baomai_current_proposals_r69.ipynb) executed unchanged from published source commit `41d0e6937c695377373cc55071ffb780345870ee`. This was local CPU execution in IPython with the source, checkpoint, photographs and returned archives preseeded and hash checked. It was not a hosted Colab run, and live download paths were not exercised. The [execution receipt](notebook-execution.json), [executed notebook](executed-notebook.ipynb), four cell logs, and [independent verification](independent-verification.json) record the completed run. Independent verification passed for the full proposals, reconstruction, measurements, source closure and package.

## What the measurements establish

The main [matrix](matrix.csv) has 40 rows: archived L4 and fresh CPU outputs, five output stages, and four unchanged references (`r50`, `r67`, `supplied664`, `frozen666`). R67 contains 658 reference fronts, including 583 with two known corners. The [generated results](results.md) give the full stage comparison; [summary.json](summary.json) records configuration, input hashes, runtime and source provenance.

Under R67, the fresh completed two-view output has 18 misses, 50 extras, 14 of those extras classified as duplicates, and complete-pair results of 572/583 within five pixels and 583/583 within ten pixels. The archived output has the same miss, total-extra and pair counts, with 13 duplicates. An unmatched output is an extra under this reference; that category alone does not prove that it depicts no physical cassette front.

### Where reference support is lost

IoU eligibility remains at least 0.50; confidence must be strictly above 0.25. Native and postclip NMS retain the original 0.70 overlap rule. The original and reflected views are scored separately here; the completed two-view method still retains original boxes and uses the existing mutual association to average accepted corners.

| Fresh R67 view | Matched | Lost at confidence | No eligible decoded box | Lost at native NMS |
|---|---:|---:|---:|---:|
| Original | 640 | 12 | 5 | 1 |
| Reflected | 646 | 10 | 1 | 1 |

Neither view has an additional R67 loss at postclip suppression or final matching contention. The [per-target stage ledger](target-stages.json.gz) preserves candidate identities, eligible counts and exact first-lost stages. The [combined-view ledger](combined-view-support.json.gz) records support across both responses to the same photograph.

The fixed decoded pools support the following maximum one-to-one matches. These bounds equal simple target coverage in this run; they are diagnostic limits under the fixed IoU relation, not deployable selections or promised recovery.

| R67 view | All decoded candidates | Above 0.25 | After native NMS | After postclip NMS |
|---|---:|---:|---:|---:|
| Original | 653 | 641 | 640 | 640 |
| Reflected | 657 | 647 | 646 | 646 |

See [stage-coverage.json.gz](stage-coverage.json.gz) for all references and counts. Lowering a confidence cutoff cannot supply an original-view box for the five references with no IoU-eligible raw candidate. The support calculation also says nothing about which corner estimates would be useful after such a change.

### The three restored partial fronts

All three focused restorations remain missed at the unchanged settings. Best overlap and highest eligible score describe different candidate selections. Their complete box dimensions, area ratios, corner estimates and stage survival are in [focused-targets.json](focused-targets.json).

| R67 target | View | Best decoded IoU | Highest score among IoU-eligible boxes | Known-corner result for that highest-score eligible box |
|---|---|---:|---:|---|
| `new_000009/R055` | Original | 0.595507 | 0.01454296 | Accepted; 2.62 px error |
| `new_000009/R055` | Reflected | 0.690462 | 0.00772521 | Unavailable: corner confidence 0.4843 |
| `new_000009/R056` | Original | 0.446675 | No eligible box | No eligible candidate to assess |
| `new_000009/R056` | Reflected | 0.525817 | 0.08451592 | Unavailable: corner confidence 0.4942 |
| `new_000053/R022` | Original | 0.589806 | 0.00016235 | Accepted; 2.08 px error |
| `new_000053/R022` | Reflected | 0.749237 | 0.00014884 | Accepted; 5.47 px error |

Endpoint availability still requires a finite point inside the image and confidence at least 0.50. No error is scored for an unavailable endpoint. For example, the reflected candidate for `new_000009/R056` has an eligible box but does not provide an accepted known corner. These observations describe the current computation; they do not establish why training assigned a low detection score or whether a changed loss would correct it.

## Runtime differences and duplicate interpretation

The 666 changed target records across runtime, stage and reference are matched-output index changes only. Detection status, known-corner availability and five-/ten-pixel statuses are unchanged. The analogous 116 records after the fixed corner rule are also index changes only. Repeated stage/reference records are not independent changed physical fronts. Coordinates and scores can differ even when these discrete outcomes agree; neither hardware nor software equivalence is assumed.

One unmatched output in `new_000087` crosses the reference-IoU boundary for `R048`: its archived IoU is 0.432788 and its fresh IoU is 0.500214. R048 is already matched by another output, so this changes one other-extra classification to a duplicate while total extras stay fixed. The fresh native NMS witness records anchor 9210 suppressing 9211. Anchor 9211 is the closest fresh raw box to the archived output, but the unavailable archived decoded tensors prevent assigning an archived anchor identity. The [boundary ledger](cross-runtime-duplicate-boundaries.json.gz) retains this distinction; the [independent receipt](independent-verification.json) also separates geometric statistics over all reference-paired matches from the narrower changed-index population.

For all 14 fresh R67 two-view duplicate competitions, mutual prediction IoU lies between 0.47444 and 0.69991. None exceeds the unchanged NMS threshold of 0.70. Two boxes can therefore both match the same annotation at IoU 0.50 without suppressing each other at prediction-to-prediction IoU 0.70. The [duplicate ledger](duplicate-competitions.json.gz) records mutual overlap, scores, reference overlaps and known-corner outcomes.

## Companion fixed corner rule

The same notebook runs the separate [corner-suppression experiment](../corner-suppression-r69/README.md) on the saved R68 outputs: five methods, four references and three prespecified variants. It is packaged separately under `corner-suppression/`; it does not change the collected raw tensors.

The fixed qualified rule suppresses the lower-score output only when mutual box IoU exceeds 0.60, both outputs supply two accepted corners, and both corresponding corner distances are at most five pixels. On the archived reviewed-label output, R67 misses/extras/duplicates change from 18/50/13 to 18/44/11. On the fresh CPU output they change from 18/50/14 to 18/44/12. Both retain the 572/583 and 583/583 pair results. The same original output indices are retained in all 20 photographs, and fresh baseline-to-rule target outcomes are unchanged across all four references. See [qualified-rule-runtime-continuity.json.gz](qualified-rule-runtime-continuity.json.gz).

This is a model-specific development result. The supplied-label arm loses one partial front under the same rule, so the rule has not earned general deployment or automatic promotion. The [research note](../../research-detection-decisions-r69.md) discusses what these findings justify next.

## Archive, extraction and reproducibility

The complete result archive is preserved in [five ordered binary parts](repository-storage.md), with exact sizes and hashes in the [storage manifest](repository-storage.json). Concatenating them reconstructs the byte-identical ZIP produced by the notebook; [independent storage verification](repository-storage-verification.json) checks all parts, the reconstructed archive and all 35 member hashes. This is the sole repository copy of the large raw tensor archive. The ZIP contains the complete diagnostic outputs, `source.json`, `decoded-proposals.npz` and the companion results. Compact tables and useful ledgers are also provided beside this guide for inspection.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `baomai-current-proposals-r69-results.zip` | 39,331,256 | `cb6a80abb90d792cb241beb93c13312f3bdcbab9c5631c831523df8cce96ca48` |
| ZIP member `decoded-proposals.npz` | 29,296,868 | `c26ccba90d28a6dab9c2f6bd1049b19374d0ddbf60594377ab9e81332454e2b6` |

The NPZ contains 40 complete FP32 tensors, each shaped `[1, 11, 19320]`, with no quantization. It remains as a standalone file in the actual execution workspace. [inventory.json](inventory.json) is the original complete run inventory and includes this archive member; the missing standalone NPZ beside this guide is intentional. No generated inventory or numeric result was rewritten for this storage layout.

First reconstruct the ZIP using [repository-storage.md](repository-storage.md). From the repository root, this then extracts and verifies the member without running inference:

```python
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

evidence = Path("docs/thesis/evidence/current-proposals-r69")
archive = Path("r69-extracted") / "baomai-current-proposals-r69-results.zip"
assert sha256(archive.read_bytes()).hexdigest() == (
    "cb6a80abb90d792cb241beb93c13312f3bdcbab9c5631c831523df8cce96ca48"
)
with ZipFile(archive) as package:
    raw = package.read("decoded-proposals.npz")
assert sha256(raw).hexdigest() == (
    "c26ccba90d28a6dab9c2f6bd1049b19374d0ddbf60594377ab9e81332454e2b6"
)
destination = Path("r69-extracted")
destination.mkdir(exist_ok=True)
(destination / "decoded-proposals.npz").write_bytes(raw)
```

The [protocol](protocol.json), [source-bound package receipt](package-manifest.json), [native suppression traces](traces.json.gz), [saved predictions](predictions.json.gz), [reference labels](reference-labels.json.gz), and [per-image measurements](per-image.json.gz) support replay and case inspection. Native equal-score decisions are observed and validated against remaining maximum scores; a display tie-break never substitutes for the recorded native decision. The notebook binds 23 project modules, 22 reference inputs, checkpoint and image hashes, preprocessing, and runtime source/native-library hashes.

The 20 photographs remain development evidence. This collection does not add independent captures, measure annotation repeatability, isolate hardware as the cause of a numerical difference, or prove that a training intervention will recover the lost fronts.
