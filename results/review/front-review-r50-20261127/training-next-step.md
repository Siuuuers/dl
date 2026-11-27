# Existing-data training follow-up: effective supervision, prior trials, and an excluded target

Read-only audit, 27 November 2026, Asia/Hong_Kong. Source repository: `Siuuuers/dl`, PR140 lineage. The audit itself performed no new inference, label changes or training. Counts below were recomputed from every effective R37 label file, after matching all 137 file hashes to the manifest. Historical metrics are quoted from the preserved notebook summaries, not independently rerun here.

## 1. What PR137 actually trained

Sources: `data/candidates/front-review-2026-11-26-r37/manifest.json`; `docs/additional-front-pilot-r47-results.md`; `results/evaluation/front-errors-r48-20261127/return-verification.json`.

Both PR137 arms use the same **137 effective-R37 original photographs**, with **3,784 fronts, 6,837 locatable corner slots, 731 unavailable slots, and 13 empty scenes**. The full review's 3,786 fronts are not the effective training count.

| Effective original supervision | Photos | Fronts | Two-corner | One-corner | Zero-corner |
|---|---:|---:|---:|---:|---:|
| Accepted R37 review selection | 105 | 3,456 | 2,866 | 454 | 136 |
| Held `new_000015`, exact original/PR35 labels | 1 | 59 | 55 | 3 | 1 |
| Old-series treatment exclusions, exact original/PR35 labels | 31 | 269 | 269 | 0 | 0 |
| **Original effective total** | **137** | **3,784** | **3,190** | **457** | **137** |

The 105 accepted photos include all 13 empty training scenes; 92 nonempty review files differ from original supervision. “Excluded from treatment” means the label update is withheld, **not that the image is absent from training**. The 31 old-series photos are a preserved treatment exclusion; they must not be described as 31 presently unresolved physical cases. `docs/front-review-r12.md` already distinguishes the accepted old series from unresolved training photos.

The held original photo is `new_000015`, unresolved case **`r6-a-04-new_000015`**, region `[700,515,727,558]`: a pale strip in a narrow gripper slot lacks an independently established front boundary. Its original 59-row labels remain effective. The review contains 61 rows, including two additional zero-corner fronts, but the separate undrawn possible-front question keeps the whole photo on its inherited label fallback. Do not release it by bypassing that completeness decision.

Only **three** effective label files differ from their corresponding full R37 review files: `new_000015`, `old_000010`, and `old_000042`. Thus a broad claim that all 32 fallback files contain unapplied changes would be false.

The 31 old IDs are: `old_000003`, `004`, `007`, `010`, `013`, `018`, `021`, `022`, `027`, `028`, `038`, `039`, `042`, `046`, `050`, `051`, `054`, `057`, `061`, `065`, `068`, `069`, `073`, `077`, `079`, `087`, `089`, `093`, `097`, `100`, `101` (all abbreviated suffixes share the `old_000` prefix).

PR137 reference adds `frame_0383`; candidate adds `frame_0146`, `frame_0160`, `frame_0178`, `frame_0371`, `frame_0383`. The seven held supplementary photos are **absent** from both new training sets: `frame_0001`, `frame_0038`, `frame_0074`, `frame_0128`, `frame_0225`, `frame_0280`, `frame_0301`. This exclusion mechanism differs from the original-photo legacy fallback.

| Training/development group | Two-corner | One-corner | Zero-corner |
|---|---:|---:|---:|
| PR137 reference training | 3,233 | 457 | 137 |
| PR137 candidate training | 3,397 | 465 | 137 |
| Added training fronts | 164 | 8 | 0 |
| R38 development targets | 583 | 54 | 17 |
| PR137 candidate misses | 0 | 12 | 3 |

This supports prioritizing partial-front supervision, but does not prove that its frequency or occlusion caused the misses. The latest run **did** train supplementary frame photos. It adds mostly two-corner fronts to a comparison whose remaining misses are all partial.

## 2. Extent consistency: documented caveat, not a proven current cause

The repository documents front-versus-occluder boundary questions. `docs/training-annotations-2026-11-10.md` explicitly records legacy boxes entering roofs or occluders and limits its certification scope. Later reviews corrected many such cases; that old caveat cannot be copied as proof that the same errors persist today.

A concrete later correction is `docs/annotation-boundaries-r34.md`: `new_000034:43` and `new_000040:34,35` had been mistakenly interpreted as jaw-only material in R33. R34 restored the same-instance fragments visible through the gripper opening and kept the hidden right corners unavailable. Those accepted corrections are already inherited by R37; proposing them again would repeat completed work.

Current rule: enclose observed frontal material and the target's own rim/lip. A single axis-aligned box can contain internal gripper pixels between owned visible fragments. A boundary must not extend merely to include gripper, rack, a neighboring front, or an imagined hidden edge. Therefore “exclude every gripper pixel from the rectangle” is not the actual annotation contract.

The current manifest explicitly says `full_dataset_certified: false` and `full_source_training_readiness: review_only`. This warrants a bounded training-label audit. It does not establish a particular erroneous row, quantify label noise, or prove causation of the present failures without original-image review.

## 3. Avoid repeating completed mechanisms as new ideas

| Trial | Verified scope/evidence | Recorded result or limit |
|---|---|---|
| **R29 sampling, PR107** | Completed seed46 L4 pair, 30 epochs, 1,050 batches, 78 updates, 4,110 presentations per arm. Nine partial-rich training photos sampled twice per epoch with rotating omissions; identical R28 labels. | The fetched PR107 return README records completion and provenance, explicitly makes no accuracy claim. No numerical scoring comparison was found in the restored text during this bounded audit. Do not call it untried, successful, or negative without recovering its evaluation. |
| **R31 partial-box factor2, PR110** | Same R31 labels, native factor1 versus factor2 for already-assigned partial-target box-regression terms; seed46. Summary preserved in `notebooks/baomai_partial_box_replication_r32.ipynb`, opening markdown cell. | Frozen666 misses **36→35**, unmatched **108→111**, pair success unchanged. R31 misses **15→12**, 5px pair successes **572→570**. A trade-off, not a clean improvement. |
| **R32 replication, PR113** | Same box-loss treatment, prespecified seed47. Summary preserved in `notebooks/baomai_front_review_r33.ipynb`, opening markdown cell. | Frozen666 misses **32→33**, unmatched **105→100**, 5px pair successes **524→522**. Failed the clean-improvement screen; partial-box weight escalation explicitly stopped. |

PR107 head: `0c1556bae12dd90e7be26fa24aca599a4c49f62f`; returned ZIP SHA-256 `62d1932ab4681ab239709e80ceb4c84d4339e87ac74f7da5554203c53394edbd`. PR110 head: `3758b084929bcde9a423fc22d7953353bb44eb2d`. PR113 head: `257ce13706586607447cc8c4d948a35def4a9eb7`. Read-only GitHub provenance is saved in `prior-training-provenance.json` beside this note. The PR bodies are return provenance; the notebook paragraphs above are the source of the quoted score comparisons.

R29's nine already-tested sampling photos are `new_000012`, `018`, `023`, `031`, `044`, `045`, `094`, `116`, `128` (all abbreviated suffixes share `new_000`). Another generic proposal to oversample these same partial-rich photos is not a new mechanism.

## 4. Smallest defensible next training study

**First finish the interrupted R49 diagnosis using the R50 successor; do not manufacture training gains from a notebook checkout repair.** Then, if diagnosis and original-pixel review support a supervision hypothesis, use the same existing image inventory and native loss in both arms, changing only a small, explicitly reviewed set of eligible training labels concerning partial-front ownership or extent. Freeze the same development reference and operating point for both models. Record actual exposure and initialization. Preserve all visibility-zero corners.

This is a specific **label-delta comparison**, not another generic sampling or box-weight increase. The delta must contain new supported training corrections; current development-only amendments, already accepted R34 corrections, and unresolved whole-photo holds do not qualify. Review suitable complete training scenes already available, including the previously sampled cohort, without changing their sampling merely because they contain partial fronts.

If no new supported training-label delta is found, **there is no justified new training pair yet**. Complete inference diagnosis and the error report using existing data. Resolving `new_000015` could eventually release two already-reviewed zero-corner additions, but the remaining ownership question cannot be declared solved by a desire for more training examples. New captures remain useful for genuinely new partial-front coverage and eventual untouched-session evaluation.

## 5. `new_000009` excluded-front identity: preserve the user's decision

Exact source: `results/review/front-review-20261126-r38/row-review.json`, stable ID **`fronts-2026-11-26-r37/new_000009:52`**. The user-directed deletion removed a visible partial front with box **`[600,490,615,504]`**, left corner **`(601,502,v=2)`**, and unavailable right corner. The recorded reason explicitly says the original pixels support a small partial front and that its removal changes the chosen reference population; it is not a background or duplicate declaration or a general size cutoff.

The independently observed sliver in current PR137-candidate outputs P057/P065/P066 is geometrically associated with this excluded region. Their IoUs with the old rectangle are **0.330254 / 0.409067 / 0.220604**, respectively; none passes the 0.50 match gate even against that old box. The wider boxes also include gripper material.

**Preserve the exclusion. Do not propose a new target or describe these outputs as recovered detections.** The interpretation is “responses around a visible but explicitly excluded reference front, with extent overreach,” not “previously unknown missed annotation.” Stable prediction-number mapping is retained separately from raw primary IDs58/66/67.

Detailed machine-readable identity record: `new009-exclusion-identity.json`. The earlier independent zero-overlap audit has been updated with this provenance and preserve-deletion action for all three cases.
