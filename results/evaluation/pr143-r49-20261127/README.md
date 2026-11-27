# PR143: completed R49 proposal diagnosis

27 November 2026, Asia/Hong_Kong.

[PR143](https://github.com/Siuuuers/dl/pull/143) returns an actual completed
inference diagnosis on a Tesla T4. It uses the existing PR137 candidate
checkpoint, not a newly trained model. The fresh native hard-NMS outputs exactly
reproduce the saved PR137 candidate outputs on all 20 photographs.

The CPU audit independently reconstructs native greedy selection, box
coordinates, confidence and both keypoints from every returned FP32 tensor.
It also reconstructs fixed linear Soft-NMS, both postclip arms, all 18 original
score tables and all 15 original missed-target traces. Six additional tables
score the same arrays against the current R50 reference. No model inference or
training is run by this audit.

## Measured result

| Reference | Hard final: misses / unmatched / duplicates | Soft final: misses / unmatched / duplicates | Both corners within 5 px, hard and soft |
| --- | --- | --- | --- |
| Frozen666 primary | 34 / 91 / 17 | 34 / 90 / 17 | 511/584 |
| Historical663 | 30 / 90 / 17 | 30 / 89 / 17 | 511/584 |
| R38 diagnostic | 15 / 84 / 14 | 15 / 83 / 13 | 566/583 |
| R50 proposal | 15 / 84 / 13 | 15 / 83 / 12 | 566/583 |

There are no recovered or newly missed targets, and every matched decoded-anchor
identity stays the same in both raw and postclip comparisons under all four
reference versions. Both-corner 10 px success is also unchanged: 575/584 on the
two historical references and 582/583 on R38/R50. These versions reuse the same
20 development images; they are not independent tests. PR35 remains selected
under the existing frozen-reference protocol.

The only final output removed by Soft-NMS is **new_000090 / hard P060**, decoded
anchor **10313**, with hard confidence **0.1364926845**. It is a formal duplicate
of the already matched R041 region on R38/R50 (best IoU 0.6425869). It is an
other-unmatched output on frozen666/historical663 (best IoU 0.3878316).

## Raw and postclip stages

| R38 pipeline state | Retained outputs | Misses | Unmatched | Duplicate subset |
| --- | ---: | ---: | ---: | ---: |
| Native hard NMS, before postclip filter | 732 | 15 | 93 | 22 |
| Native hard NMS, after postclip filter | 723 | 15 | 84 | 14 |
| Linear Soft-NMS, before postclip filter | 745 | 15 | 106 | 34 |
| Linear Soft-NMS, after postclip filter | 722 | 15 | 83 | 13 |

Soft-NMS adds 14 decoded proposal identities before postclip filtering and
removes one native identity. All 14 added identities are then suppressed by
the unchanged postclip filter. However, **raw Soft-NMS already recovers no
target**, so the result cannot be explained solely by the second hard filter.
The returned raw arm already measures the no-postclip case; repeating the same
inference solely to obtain that comparison is unnecessary.

## What the miss trace establishes

Both R38 and R50 have the same stage counts:

- **10 confidence-gate cases:** decoded boxes with matching IoU exist, but
  every qualifying box has confidence at or below the strict 0.13 cutoff.
- **3 geometry cases:** no decoded box reaches matching IoU 0.50. The best
  overlaps are approximately 0.4906 (`new_000009/R051`), 0.3065
  (`new_000011/R057`) and 0.4027 (`new_000011/R060`).
- **2 ordinary-NMS cases:** at least one qualifying box passes confidence
  but is removed by ordinary NMS (`new_000089/R048`, `new_000090/R039`).

These are measured pipeline locations, not proof of the learning cause.
Confidence, localization, target assignment and annotation extent interact;
the trace alone does not prove that changing a loss will improve the model.

## Evidence files

- [summary.json](summary.json): experiment identity, runtime, input/reference
  hashes, aggregate scores and stage counts.
- [reference-replay.json](reference-replay.json): all four references and six
  saved/hard/soft views, with visibility strata, per-image counts and paired
  recovered/newly missed row identities.
- [missed-cases.json](missed-cases.json): reconstructed R38 and R50 target traces,
  best decoded box and every IoU-eligible proposal with exact score and stages.
- [filter-delta.json](filter-delta.json): added/removed decoded identities,
  postclip suppressors, removed-output reference status and assignment changes.
- [verification.json](verification.json): all completed integrity/replay checks.

Individual endpoint arrays and full prediction lists are already in the pinned
return archive. These text reports omit duplicate endpoint arrays while
retaining aggregate error statistics and the evidence needed for the result.
Prediction P numbers are one-based within their named stage; decoded anchor
indices are zero-based identities within each image's returned tensor.

## Reproduce

```bash
python scripts/audit_front_diagnosis_r49_results.py \
  --archive /path/to/baomai-r49-diagnosis-results.zip \
  --prior-archive /path/to/baomai-additional-front-pilot-r47-results.zip \
  --originals /path/to/verified-original-pngs \
  --output /tmp/pr143-r49-audit
```

Omit the archive arguments to download the pinned returns into `--download-dir`.
Original-image byte verification is optional for the numerical replay and is
reported explicitly; the published audit checked all 20 originals. The script
preserves an incompatible existing output directory instead of silently replacing
its reports. NumPy is required; Torch and Ultralytics are not imported.

The prior audit's tuple-valued counts are normalized to their JSON representation
before comparing the archived receipt. Lists and tuples otherwise differ in
Python even when the serialized contract is identical.

PR143 head: `a59f07dce21c5aedcb088c725dc14225a53a301c`.
Return: 15,631,804 bytes; SHA-256
`892c8ee79ff470990bb47dc59f947ad6bd2577a86bc9827e9dd57f81c56213f3`.
Returned R49 source: `8cf5a2afc9fe28ba0f4f4596c27944761a790b76`.
Returned script SHA-256:
`188df92036d5dc8fd8e13fc8d8815728a56c33ced654de2cf4716788eac47811`.

Timing in the returned bundle covers suppression and result construction only.
The NumPy Soft-NMS and native path are different implementations; their timings
do not establish an optimized deployment-speed comparison or end-to-end latency.
