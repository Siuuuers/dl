# Historical accepted training corrections: composition audit

This is a source and operation audit, followed by a separately materialized reviewed candidate. It is not a new model result, a complete physical certification of all labels, or a GPU handoff. Original source row numbers are one-based throughout.

## What the selected model learned from

The selected PR35 seed46 model used `annotations-2026-11-10`, the eight-file overlay with 3,771 objects and 6,841 visible endpoint slots. Its exact manifest is pinned in `docs/training-annotations-seed46.md` and its executed loader counts are recorded in `docs/annotation-confirmation-2026-11-11.md`. PR63’s 17-file `remaining-fronts-2026-11-14` overlay has not been used to train a model. It also deliberately leaves several older accepted interventions separate. Therefore, yes: the selected model’s supervision still contains some subsequently identified label errors and omissions. This does not prove that every model prediction is wrong, nor does applying correct labels guarantee that a finite retraining run will improve every metric.

## Acceptance and scope

All 119 unique historical operations below occur in emitted, accepted annotation proposals. Annotation and box records explicitly say `accepted_incremental`. The older eleven dedup deletions are bound to primary visual decisions with independent agreement in `docs/training-pilot-2026-11-06.json`. PR58’s composed boxes replay accepted recovered decisions. The manifests retain `status: proposed`; this means neither human certification nor model promotion has been claimed. Held, withdrawn and generic review observations were not extracted as operations.

Every historical label file reconstructed byte-for-byte from original source rows and its declared changes. There were no source text/hash errors, row-map errors, divergent accepted values for the same original row/field, or delete-and-edit conflicts. This is stronger than concatenating candidate folders, but does not replace original-image adjudication for unresolved cases.

## Accepted operations absent from PR63’s current 17-file overlay

| Historical review | Accepted delta absent from PR63 | Original image / row identities |
|---|---|---|
| Dedup, 2026-11-06 | 11 deletions | 019:56–60; 075:1–3; 102:15; 112:35; 051:21 |
| Box refinement, 2026-11-07 | 16 single-edge boxes | 012:15,16,26,27,37,38; 030:9–11,23–26,29; 128:10,11 |
| Partial fronts, PR40 | 5 additions, 2 deletions, 1 bbox, 1 donor-left transfer | 119:34/36; 120:56/new front; added 122:57,125:53/54,130:59 in their original proposal |
| Visible extent, PR43 | 11 boxes and 1 addition | 061:9,14,15; 079:34,15,16; 082:26 + added front; 094:29–32 |
| Consistency, PR47 | 14 boxes | 018:24–26,32–34; 034:55–57; 049:9,14,24–26 |
| Endpoint visibility, PR50/52 | 10 endpoint slots | 023:33 left/right,34 left,35 left; 031:22–24 both slots |
| Recovered boxes / composition, PR49/58 | 17 boxes and 1 deletion | 023:12–14,29–31,36 boxes and original22 deletion; 031:26–28,37–41 boxes; 045:1,15 boxes |

Total absent from current PR63: **90 field operations across 23 images**: 59 bbox replacements, 14 deletions, six added fronts, and eleven endpoint-slot changes (ten endpoint amendments plus the 119 donor transfer). All current PR63 operations are retained by the reconciliation.

## Counts

| Dataset | Training objects | Visible slots | Missing slots | Changed files versus original |
|---|---:|---:|---:|
| Frozen original | 3,772 | 6,839 | 705 | 0 |
| Selected PR35 model supervision | 3,771 | 6,841 | 701 | 8 |
| Existing PR63 remaining-fronts candidate | 3,779 | 6,846 | 712 | 17 |
| Historical accepted reconciliation only | 3,771 | 6,847 | 695 | 39 |
| Reconciliation plus independently adjudicated new038 repair | **3,774** | **6,849** | **699** | **40** |

All variants keep 137 training images and 13 empty training labels. Original in-training validation remains 20 images / 664 objects / 1,227 visible / 101 missing / three empty. The separately adjudicated development reference remains 20 / 663 / 1,227 / 99 / three empty. Reference changes never enter the training overlay.

## Identity and composition dependencies

1. `new_000023` deletes original22 as a duplicate of original36. Original33/34/35 endpoint changes become output32/33/34; original36 becomes output35. Use original IDs, never an already-shifted line number.
2. `new_000102` deletes original15, retains original54, and independently keeps the PR35 bbox correction on original53. The final rows are original53 → output52 and original54 → output53. This explains why 16 of 17 PR63 label files can remain byte-identical, while the seventeenth changes without losing any PR63 accepted field.
3. Three donor consolidations are inseparable: 067 source25-left → retained12-left; 072 source36-right → retained34-right; 119 source36-left → retained34-left. Donor rows are deleted only while their exact visible triple survives on the retained owner.
4. `new_000120` original56 is a gripper-only row that is deleted. The appended physical front becomes candidate56, but it is a different identity with a stable addition ID. It is not a replacement under the old identity.
5. Keep the two new027 additions in recorded append order, the four new029 additions in recorded order, and all other historical addition IDs. Added objects have no fabricated original row ID.

## Superseded and excluded records

- The standalone 079 three-box candidate is exactly represented in the full visible-extent candidate; count it once.
- The older nine-slot recovered endpoint record lacks the later accepted 023 original33 right point `(612,549)`. The latest ten-slot endpoint candidate supersedes that missing-slot state. This moderate-confidence right point was explicitly accepted after a review that retained one reviewer’s dissent; preserve its uncertainty record.
- PR35 copies inside later candidates are exact inherited copies, not new corrections. The complete PR58 candidate includes the endpoint control; it is not an additional second endpoint intervention.
- New015’s three rack-top/backer additions were withdrawn and remain absent. The nearby-frame structure comparison did not establish cassette identity.
- The earlier proposed new038 upper-to-lowest point move is withdrawn. The new accepted sibling adjudication instead assigns three separate fronts, refines original row1 to its upper owner, appends two lower fronts, and separately appends the bottom-clipped front. This is five new field operations.
- The supplementary `additional-pilot-2026-11-07` is a separate 320-frame intake; all three reviewed sample images remained quarantined. It does not enter the frozen 137-image training cohort.
- Context-inferred corners remain review aids, not canonical visible labels. Four generations of development-reference edits are separately frozen and never composed into training.

## Exact replay versus new visual work

The accepted operations have no unresolved token conflict requiring a new visual decision merely to copy them. The independent original-pixel reviews justify their bounded changes, not the correctness of every unedited field in those images. Existing box acceptance does not certify an unchanged physical corner. The held ledger retains distinct unresolved identities, front/upper-surface boundaries, point-ownership questions, and partial review limitations. It also records which old concerns have been resolved or superseded.

The new038 repair is the sole newly adjudicated local delta in this composition. All right endpoints remain missing; two new left endpoints are marked visible; the clipped front has both missing. The accepted figure and JSON retain approximately two-pixel point and three-pixel box placement uncertainty. No subpixel physical accuracy is asserted.

## Minimal implementation and invariants

`verify_reconciled_pose_labels.py` exposes two commands, `build` and `verify`, with one explicit hash-bound specification. It imports the established parser, geometry checks and count logic; runs the unchanged historical validators in their matching policy contexts; derives all historical after-tokens from the source manifests; and checks the local pixel encoding against the accepted after-tokens. It materializes only the changed training labels plus `manifest.json` and `review.json`. Candidate-root `README.md` is optional prose and is excluded from the exact annotation payload check; other undeclared files are rejected.

The current policy must remain at SHA-256 `74dd943976a4abc22ecbf73551c69f715666060d5961b2bed81e6a9c73a3afc9`. Older strict candidates use the unchanged snapshot `3cda361ac4c5c9ba36b037e7e8968e0629eb3bca8d489c187beb8dd66e208a91` in a disposable verification directory. Neither old manifests nor old validator guards are rewritten.

The new kind is `training_reconciled_overlay`, with `release_ready: false` and `training_recipe_id: null`. The existing training stager/recipes are unchanged and do not accept this new kind. Reconciliation is a concrete data review deliverable; training integration requires a later explicit finalized experiment.

## Files in this review

- The candidate's `review.json` records all 119 historical operations plus five local operations, exact before/after tokens, source records and inherited provenance; no second copy of the operation list is needed.
- `source-inventory.json`: immutable source, policy, code, candidate, review, reference and weight descriptors used by the verifier.
- `summary.json`: historical-only counts, omitted operations and zero-conflict result.
- Candidate `review.json`: the final 124 operations; original-to-candidate row maps are in its bound `manifest.json`.
- `specification.json`: final invocation inputs with stable repository-relative evidence paths.
