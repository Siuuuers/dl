# PR62 result evidence and offline replay

**Decision: retain PR35.** The synthetic covered-side ten-pixel score improves 54/60→58/60, while the fixed clean preservation gates fail. Read the [result report](../../../docs/occlusion-results-2026-11-14.md) for the complete interpretation and [comparison figure](../../../docs/images/occlusion-results-20261114.png).

## Provenance

- Training implementation: `b457a4a6975104700d7342fa4b997ea6da7625ad`.
- Original PR61 source/notebook head: `53ccf4ec3bc857d3ab41c3be6f0551447a48f8b0`.
- Returned GPU artifacts: [PR62](https://github.com/Siuuuers/dl/pull/62), commit `52a5be19cb6afdece05637c6e6c86dcc5fccd76d`.
- Return archive SHA-256: `4c94b9ae2496b0ab8846941e7da61eba4931630b805ca7f35a0a9272fc486cad`, 39,788,156 bytes.
- Frozen experiment protocol SHA-256: `c70563034d3ac969ff7837a594d53bc6316fda3897e1557b5d7828be52e94a62`.
- Operational PR35 checkpoint SHA-256: `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`.

The return passes the existing 14-file paired-run verifier. Both roles completed 30 epochs, 4,110 image presentations and 1,050 batches. The candidate applied 1,839 masks, and the control applied none; their planning digests agree. Exact label inventories and recorded runtime/source identities were checked.

Fresh evaluation used 240 image predictions: 20 clean images for each of three models, and 90 masked images for each new role. Three blank warmups are excluded. The 94 historical PR35 probe records were verified and replayed. The four intact probe records for each new role reuse its corresponding clean-image prediction. PR35's fresh clean arrays reproduce the published baseline, and control equals PR35 on every clean/probe raw array.

## Files

| File | Purpose |
| --- | --- |
| [evaluation.zip](evaluation.zip) | All raw model predictions, frozen references and cases, complete scores, replay status, independent checker and original returned metadata; no model weights or source images. |
| [archive.json](archive.json) | Archive SHA-256, byte size and member count. |
| [summary.json](summary.json) | Compact view of all clean/probe aggregates, changed identities, supporting gains, full-image effects and prospective numeric gates. Redundant pooled error arrays remain in the archive's complete summary. |
| [verification.json](verification.json) | Return verification, source/model hashes, actual training exposure and inference runtimes. |
| [replay-verification.json](replay-verification.json) | Exact offline reconstruction of all scores and summaries. |
| [independent-review.json](independent-review.json) | Independent normalized-label parsing, direct IoU matching, endpoint/pair calculations and decision checks. Arithmetic check counts are not independent-example counts. |
| [visual-adjudication.json](visual-adjudication.json) | All nine lost clean five-pixel pairs, the gained box match and the empty-image false proposal. |
| [additional-case-review.json](additional-case-review.json) | Every gained clean match/pair identity and all 17 changed masked pair cases. |

The archive contains `evaluation/` with eight JSON files, `review/` with the independent checker and review records, `return-metadata/` with the twelve non-weight files from PR62, and an `inventory.json` binding every payload member's bytes. The independently computed result agrees with the primary runner; neither checker imports the other's matching implementation.

The core evidence uses twenty reused development images and four probe source images drawn from those twenty. It contains zero new captures and zero independent test images. Masked copies and repeated target observations do not increase the independent-scene count.

## Replay without inference

Use this PR61 source checkout, including the new [evaluation runner](../../../scripts/evaluate_occlusion_results.py), and choose fresh output directories:

```bash
python -m zipfile -e \
  results/verification/baomai-occlusion-results-20261114/evaluation.zip \
  /path/to/pr62-evidence

python -B scripts/evaluate_occlusion_results.py \
  --root . \
  --replay /path/to/pr62-evidence/evaluation \
  --output /path/to/new-pr62-replay
```

The replay checks the frozen protocol, reference/case bytes, saved artifact hashes and evaluator source hashes, then recreates every clean/probe score and summary. It requires no GPU, image originals, model weights, Torch installation or new predictions. It reports `numeric_gates_pass_against_both: false`; manual image adjudication is documented separately and does not relax the failed fixed gates.

The independent standard-library checker can also be run after fetching the exact original PR62 ZIP:

```bash
python /path/to/pr62-evidence/review/independent_check.py \
  --root . \
  --evaluation /path/to/pr62-evidence/evaluation \
  --returned-zip /path/to/baomai-occlusion-seed46-results.zip \
  --output /path/to/new-independent-review.json
```

For fresh inference, supply the restored validation originals, the extracted PR62 return and the complete verified PR35 checkpoint bundle to `evaluate_occlusion_results.py --returned-dir ... --returned-zip ... --current-manifest ... --image-root ... --output ...`. Use the pinned inference runtime in `verification.json`. This is an optional reproduction path, not a request to retrain the models.

The [figure renderer](../../../scripts/render_occlusion_results.py) consumes the saved evaluation directory plus three distinct original PNGs for its four cases. Its [index](../../../docs/images/occlusion-results-20261114-index.json) binds those input images and the exact masked pixels. Its examples are explicitly post hoc illustrations.

## Interpretation boundaries

The lost `new_000056:3` box match explains the two-count locatable-endpoint coverage loss: the candidate still emits both coordinates within five pixels. The new `new_000009:56` match comes from a proposal already present in PR35, with both reference corners missing. These diagnostics explain the canonical score without changing it.

The `new_000147` background proposal is a genuine false box/line under the selected thresholds. `new_000008:11` has a clear clean-corner displacement. Seven other lost five-pixel pairs involve smaller coordinate movements around soft boundaries; keep the numerical regressions without claiming seven large independent physical failures. All gained identities and the useful masked-input effect are retained equally.

No training labels, evaluation labels, frozen protocol, historical scorer, production defaults, model selection or branch merge changed as a result of this result publication.
