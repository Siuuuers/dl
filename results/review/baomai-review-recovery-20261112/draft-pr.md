## Why

PR48 completes the PR47 comparison and retains PR35. The next original-image review found 17 training-box corrections, one duplicate row and six rows with questionable corner supervision. The local review was interrupted, then the execution filesystem restarted before publication. This recovery preserves the explicit supported changes and current readiness findings in a small self-contained record.

## Changes

- Reconstruct 17 box corrections and delete one confirmed both-missing duplicate from pinned PR48 source labels in images 023, 031 and 045.
- Preserve separate endpoint-amended files for images 023 and 031: nine changed slots across six original rows, with three unavailable slots kept missing.
- Include the exact source labels, source identities, row maps, before/after tokens, output hashes and a bounded reconstruction check.
- Record current PR35 metrics together: 643/663 matches, 20 misses, 85 scored extras, 577/584 two-corner successes within 10px, 3/17 miss-free nonempty scenes, and 14/20 misses outside the two-corner denominator.
- Include the recovery report and explicit limitations. This does not claim recovery of the old full evidence capsule or create a training-stager-compatible candidate.

## Verification

All three source-label hashes match the pinned dataset records. Rebuilt box-review files for 023 and 031 exactly match their parent hashes recorded before the restart. Independent comparison verifies all 17 box changes, the single duplicate deletion, all nine endpoint changes, row identities and unchanged unrelated tokens. Geometry checks pass, including canonical missing triples and original-pixel containment tolerance. `python3 -B rebuild_review.py --check` reproduces the exact files and correction record.

Actual original-image inspection and the saved-prediction numerical replay completed before the restart. The recovery did not download or re-inspect the original PNGs, run inference or train a model.

## Decision

Keep PR35 seed46 at 1280 / confidence 0.13 / NMS 0.70 / endpoint cutoff 0.50 for assistant-reviewed proposals. The label amendments improve the documented supervision; their effect on a trained model is unmeasured. No model replacement, new GPU request, parameter search or PR merge accompanies this draft.

Proposed destination: `Siuuuers/dl`, new branch `codex/corner-review-recovery-20261112`, based on PR48 branch `codex/consistency-results-20261112` at `f8d758ade0b721d494306e3002d26a3d56d807cb`. Keep all recovery files together under `results/review/baomai-review-recovery-20261112/`. Publication remains pending explicit approval of this concrete payload.
