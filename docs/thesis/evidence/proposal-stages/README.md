# Current missed-front proposal-stage audit

This completed CPU analysis traces all 18 missed reviewed fronts through the existing trained checkpoint at the retained confidence threshold, strictly greater than 0.25. It reuses the full decoded proposal tensors returned in PR #143. It performs no neural-network inference, training, label editing, threshold search, or model promotion.

## Result and scope

The mutually exclusive partition is **12 confidence-gated fronts, three native-NMS losses, and three fronts with no decoded box attaining reference IoU 0.50**. Extra postclip suppression and retained assignment contention explain none of these 18 misses. All 18 have incomplete corner references: 13 have one available corner and five have none.

The existing postclip pass already removes six duplicates without losing a reference match: 693 native outputs produce 636 matches and 57 extras including 11 duplicates; 687 postclip outputs produce the same 636 matches and 51 extras including five duplicates. Duplicates are a subset of extras.

The three NMS cases are `new_000089` reviewed rows 43 and 48, and `new_000090` reviewed row 39. In each case, a higher-score candidate suppresses an eligible alternative but remains an unmatched extra under the fixed reviewed reference. The suppressors' target IoUs are approximately 0.468857, 0.495072, and 0.493702. This describes the observed ranking and suppression behavior. It does not establish which candidate is physically correct or prove that changing NMS will improve the complete detector.

The retained method's two-view corner result remains 569/583 complete pairs within five pixels. This audit traces its original-view detection boxes; it does not replace the averaged-corner measurement with an unaveraged replay.

## Files

- [summary.json](summary.json): source identities, actual runtime differences, all continuity checks, all 20 per-image candidate-stage summaries, and the closed miss partition.
- [cases.json](cases.json): every missed reference, zero-based indices and one-based rows, best decoded candidate, highest-score eligible candidate, and every above-threshold eligible candidate lost to native NMS with its actual suppressor. Historical 0.13 matches remain separately identified.
- [results.md](results.md): concise interpretation, candidate-stage table, and the complete 18-case table.
- [manifest.json](manifest.json): byte counts and SHA-256 hashes for those three files, plus the analysis and frozen helper source hashes.

No execution timestamps or absolute machine paths are included in these analytical files. The script records Python and NumPy versions to identify the replay environment.

## Published notebook execution

All four actual code cells in [baomai_proposal_audit_r63.ipynb](../../../../notebooks/baomai_proposal_audit_r63.ipynb) completed unchanged, sequentially in one IPython namespace, from a new source/cache/work directory. The notebook pins published source commit `bb1d74cce4739a72ad0f7d9f227ca1e4cac37c7c`. Its input SHA-256 is `249423cd5f8cbaedd066c88a4e943693bcff618f2b2bdf4f150080f01384baf8`.

This was a local CPU execution with Python 3.12.14, NumPy 2.3.5, and IPython 9.18.1. It did not exercise a hosted Google Colab runtime or Colab's browser download interface; the non-Colab `FileLink` fallback completed.

- [notebook-execution.json](notebook-execution.json) records all four cell outcomes, unchanged cell-source hashes, execution durations, log hashes, source cleanliness, package verification, and the precise environment.
- [executed-notebook.ipynb](executed-notebook.ipynb) preserves the unchanged cell sources and actual text/rich outputs.
- [baomai-proposal-audit-r63-results.zip](baomai-proposal-audit-r63-results.zip) contains the proposal-stage and box-geometry results: 109,099 bytes, SHA-256 `c04a501740a2379d59544ffd114ec7c2cdfda75114d797516e988b15e2999d63`.
- `notebook-cell-1.txt` through `notebook-cell-4.txt` contain the actual captured process and cell text output.

Cell four reran both analyses into separate verification directories. All eight analytic files matched their first-run versions byte for byte. An independent execution harness then verified the exact nine-member ZIP set, CRCs, every outer and inner manifest entry, source/helper hashes against the clean published checkout, and equality between packaged, first-run, and verification files. It reconfirmed all 20 historical prediction replays, all 60 current reference/image outcome comparisons, and all 120 geometry score-table replays. No neural-network inference or training occurred.

## Provenance and verified continuity

The decoded input is `baomai-r49-diagnosis-results.zip`, 15,631,804 bytes, SHA-256 `892c8ee79ff470990bb47dc59f947ad6bd2577a86bc9827e9dd57f81c56213f3`, from PR #143 head `a59f07dce21c5aedcb088c725dc14225a53a301c`. Its original driver source commit is `8cf5a2afc9fe28ba0f4f4596c27944761a790b76`. It contains 19,320 decoded proposals per photograph before confidence filtering and native NMS, for 20 photographs.

The current comparison input is the committed [annotation-reference archive](../annotation-reference/baomai-annotation-reference-r60-results.zip), 972,658 bytes, SHA-256 `ce8df4bcf223b6bee49fa67e829a443acb572e4a6aad75fa0f92657a16cc2601`. Both input archives have their complete member sets, CRCs, byte counts, and SHA-256 inventories verified before analysis.

The archives record the same trained-checkpoint SHA-256, `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`, and identical SHA-256 values for all 20 photographs. The replay verifies this recorded identity without loading weights or photograph files. All three reference-label sets are read from the current archive and their label hashes are checked.

The historical tensors came from Python 3.13.16, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, Ultralytics 8.4.182, and a Tesla T4. The current prediction cache came from Python 3.12.14, PyTorch 2.9.0+cpu, torchvision 0.24.0+cpu, and Ultralytics 8.4.182. These runtimes are different.

Before interpreting proposal stages, the analysis requires all 20 historical 0.13 native anchor selections, boxes, scores, corners, postclip predictions, and postclip audits to replay exactly. At 0.25, it verifies native record order and numerical agreement, postclip retained identities, all prediction/reference match identities, all missed/extra/duplicate identities, and both per-corner and complete-pair threshold outcomes across all 60 photograph/reference combinations. The maximum observed coordinate difference is 0.00018310546875 pixels; the maximum box or corner score difference is 3.7550926208496094e-06. Numeric byte identity across runtimes is not claimed.

## Reproduction

The only additional package needed by this audit is NumPy. The current environment completed the analysis with Python 3.12.14 and NumPy 2.3.5. It does not require PyTorch, torchvision, Ultralytics, or a GPU.

Download or reuse the already verified historical archive with the existing acquisition helper:

```python
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from audit_front_diagnosis_r49_results import obtain_archive

archive = obtain_archive(Path("/tmp/r63-proposal-cache"))
print(archive)
```

From the repository root, run:

```bash
python scripts/analyze_proposal_stages_r63.py \
  --root . \
  --r49-archive /tmp/r63-proposal-cache/baomai-r49-diagnosis-results.zip \
  --reference-archive docs/thesis/evidence/annotation-reference/baomai-annotation-reference-r60-results.zip \
  --output /tmp/r63-proposal-analysis
```

The four deterministic output names are `summary.json`, `cases.json`, `results.md`, and `manifest.json`. The script preserves any different existing output instead of overwriting it. Byte-for-byte reruns require the same source and recorded Python/NumPy environment.

The eight focused tests cover strict confidence and NMS boundaries, inclusive reference-IoU eligibility, exact agreement with the scorer's IoU formula, empty candidate/reference sets, repeated proposals for one target, shared candidates across targets, augmenting-path matching, and bounded cross-runtime comparison:

```bash
python -m unittest discover -s tests -p test_proposal_stages_r63.py -v
```

## Interpretation limits

Geometric support decreases from 651 referenced fronts in the complete decoded tensor to 639 above confidence 0.25 and 636 after native NMS. Maximum one-to-one matching confirms those same capacities, accounting for candidates that overlap multiple references. Nine raw candidates are eligible for more than one reviewed reference; none survives the confidence gate. No retained prediction has competing eligible references in this comparison.

An eligible raw proposal is not an independently recovered physical object. The one-to-one bounds are diagnostics, not detector scores or deployable selections. They do not count extra detections, rank candidates, establish reliable corners for incomplete targets, or prove that a confidence change would have a favorable precision/recall tradeoff. The historical 0.13 and 0.05 controls already address that last question separately.

All findings concern the existing checkpoint and repeatedly used development photographs. They do not diagnose the reflected view's absent pre-NMS tensors, future trained checkpoints, or unseen capture sessions. A stage label identifies loss of geometric eligibility within this fixed computation; it is not proof of a unique physical cause or a guaranteed improvement.
