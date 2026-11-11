# Completed proposal-filter investigation

The [full report](../../../docs/proposal-filter-review-2026-11-11.md) selects the existing PR35 seed46 checkpoint at **confidence 0.13** for assistant-reviewed proposals. Image size 1280, NMS 0.70, endpoint cutoff 0.50 and the 663-object reference remain fixed. The [decision record](decision.json) follows the completed numerical and original-image review: one supported additional match, nine additional scored extras, no lost matches or corner-pair successes, and no new empty-image detections.

This is a measured operating-setting change with the same weights. Generic defaults remain epoch173/0.25. PR40's training-label extension is still not selected.

## Evidence capsules

| Capsule | Contents | Size | Members |
| --- | --- | ---: | ---: |
| [trace-evidence.zip](trace-evidence.zip) | Both complete pre-filter tensors, fixed-setting outputs, source/runtime identities, executed trace and single-case replay scripts, diagnostic figure, independent numerical audit | 2,947,589 bytes | 17 |
| [confidence-evidence.zip](confidence-evidence.zip) | Prospective full-set protocol, forty predictions, both reference versions' scores, exact baseline comparison, ten added-proposal review sheets, all executed analysis sources, independent numerical and per-proposal visual reviews | 3,710,085 bytes | 35 |

The [inventory](evidence-inventory.json) records both archive hashes and every member's hash and size. Both archives passed a fresh ZIP CRC/member/hash readback. Separate checksum files are provided for the [trace](trace-evidence.zip.sha256) and [confidence comparison](confidence-evidence.zip.sha256).

The measured confidence summary and original queue retain their initial `pending_visual_review` states as historical records. The later `independent-audit/confidence-013-visual-review.json` inside the confidence capsule covers all ten additions. The separate [decision](decision.json) records the completed review and explicit selected workflow. It does not rewrite the earlier measured evidence.

## Exact source identity

Execution recorded local source commit `c064b564f79a41578f07084b185946d33fe05024`. Its complete Git tree is identical to published commit **[`f1fc541493a1cce2a8d8b55405806eee8915b712`](https://github.com/Siuuuers/dl/commit/f1fc541493a1cce2a8d8b55405806eee8915b712)**: tree `a912171a38c9e41e62d0a0d53df7a77a66071774`. The source-equivalence record is in both capsules. Use the published commit for reproduction; the external scripts retain their actually executed hashes.

The selected weights are the complete PR35 candidate bundle, finalized epoch-30 `last.pt`, SHA-256 `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`. [Download and verification instructions](../../../docs/predict-pose-folder.md#use-the-seed46-candidate-explicitly) identify the immutable archive. The historical comparison input is `run/predictions-current.jsonl` in the [completed PR40 evaluation capsule](../baomai-partial-fronts-results-20261111/evidence.zip), SHA-256 `e08f5dee7f4f08377f97982acb362fe6f7d97066e6b9b4de623d9c598366b465`.

The raw trace made two original-image predictions and one excluded warmup. Its nine inter-op threads differ from the archived evaluation's one; the two final outputs match exactly. The full-set comparison made forty predictions and one excluded warmup, using four intra-op and one inter-op CPU threads, with every fresh 0.15 array exactly matching the archive. Both used pinned Ultralytics 8.4.182, PyTorch 2.10.0+cpu and torchvision 0.25.0+cpu. Network was disabled during both inference runs.

## Reproduce the calculations

Verify the archive checksums, extract into new directories, and retain the original evidence unchanged. The scripts' README files contain full commands. Hydrate the original validation PNGs and retain the pinned repository metadata and labels; use original images, not annotated previews.

For a saved-output replay, make a separate copy of the extracted `confidence/` directory, then use the capsule's executed comparison script:

```bash
python -B /path/to/confidence-evidence/confidence/compare_confidences.py \
  --root /path/to/dl-at-f1fc541 \
  --study /path/to/new-confidence-replay \
  --archive /path/to/pr40-evidence/run
```

This imports the unchanged matching and quality helpers, verifies archived identities, and recalculates the original 664-object and corrected 663-object results without loading a model. It also checks that all twenty 0.15 arrays reproduce the archive and that every higher-confidence output remains exactly unchanged at 0.13. The independent audit separately recomputed eighty score records, eighty quality records, four complete aggregates, twenty-four scene/range aggregate pairs, and all object/group changes.

To repeat fresh inference, follow `confidence/README.md` with a new study directory and the complete verified PR35 bundle. Copy the preserved fixed protocol before running the evaluator and keep its source-equivalence note with the new invocation. The actual invocation records the checkout used. The full-set command fixes `--conf 0.15 0.13`; no additional threshold search is part of reproduction.

For the raw trace, use `trace/trace_raw_proposals.py` with the published equivalent source, the complete PR35 candidate manifest, and the archived current predictions. Its output directory must be new and outside the source checkout. The script aborts if the two final outputs fail exact comparison. The subsequent `trace/replay_013_diagnostic.py` uses the saved tensor and performs no model load or fresh prediction.

Independent audit scripts are preserved as executed, including their original workspace paths and helper dependency. The primary trace and comparison CLIs accept explicit paths for reproduction. No claim is made that the historical audit scripts are general-purpose commands without adjusting those recorded paths.

## Scope

All twenty images are reused development material. Confidence 0.13 was motivated by the earlier single-case trace; fixing a follow-up protocol does not make its evidence independent. Ten added proposals across six images were reviewed by an assistant, without human or supervisor certification or any reference amendment. Review time was not measured. `009:57` remains missed, the inherited annotation uncertainties remain, and complete original-image review is still necessary.
