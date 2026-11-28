# Where the remaining cassette detections disappear

28 November 2026, Asia/Hong_Kong.

The [CPU notebook](../../notebooks/baomai_proposal_audit_r63.ipynb) combines a full-proposal replay with a box-geometry diagnostic. All four actual code cells have completed unchanged from clean published source in a fresh local CPU run. It needs NumPy and the archived decoded tensors, not a GPU, model weights or original photographs. The [execution evidence](evidence/proposal-stages/README.md) records both exact analytical replays and independent verification of the nine-member result package. The hosted Colab download interface was not exercised.

## Why this is useful

The current two-view method has 18 missed fronts under the reviewed reference. Its detection boxes come from the adapted detector's original view; corner averaging does not change them. The recent common-runtime cache contains only proposals already passing confidence and native suppression, so that cache alone cannot locate all earlier losses.

An older returned archive contains full FP32 decoded tensors from the same adapted checkpoint and the same 20 photographs. Reusing it distinguishes confidence filtering, native NMS, final suppression and reference matching without another inference run. Coordinate scaling and clipping are replayed and checked; clipping is not isolated as a separate loss category. This extends the completed [error audit](detection-errors.md); it does not repeat the older 0.13-to-0.05 confidence experiment as a new improvement.

## Fixed comparison

| Setting | Value |
| --- | --- |
| Current object confidence | Strictly greater than 0.25 |
| Ordinary and final NMS overlap | 0.70 |
| Reference matching IoU | At least 0.50 |
| Output cap | 300 per photograph |
| Reference versions | Supplied 664, earlier fixed 666, reviewed 654 fronts |
| New inference or training | None |
| Annotation changes | None |

The exact historical 0.13 native and final outputs must replay before the current 0.25 analysis. Model and image SHA-256 identities must agree across the older full-tensor archive and the recent reference archive. The current operating-point detection assignments and scored outcomes must also agree. CPU/GPU coordinate and score differences are measured separately; agreeing outcomes are not described as byte-identical inference.

## Interpretation

The completed replay locates the 18 reviewed-reference misses as 12 with no eligible proposal above the confidence cutoff, three whose above-cutoff eligible proposals are lost during native NMS, and three with no decoded box reaching matching IoU. These stages partition the current misses, not their physical or learning causes. A missed reference may have several weak or suppressed candidates, and a single candidate can overlap more than one reference.

The audit therefore retains individual proposal identities, suppression competitors and reference assignments. Geometric eligibility and one-to-one matching bounds are diagnostic quantities; using reference coordinates to choose a proposal would not be an implementable inference rule.

This strengthens the motivation for the prepared [classification-loss comparison](quality-focal-experiment.md). It does not establish that its loss change will improve ranking, or that lowering a threshold will improve the balance of misses and extras. The earlier lower-threshold replay recovered targets while adding many other outputs.

## Box geometry and the next model choice

The second analysis describes the size of matched and missed reference fronts and compares each missed front with its greatest-IoU retained prediction where overlap is positive. It reports signed edge differences, center/size differences and reference-defined corner availability under all three annotation versions.

A greatest-IoU association is a review cue. It does not prove that the candidate depicts the same physical fragment. The evidence does not justify automatically shrinking predictions to reference boxes or using these associations at runtime. The [geometry evidence](evidence/box-geometry/README.md) and [research note](research-next-models.md) explain why a tiny-object distance loss needs more support before another GPU comparison.

## Run and return

Open the notebook in a standard CPU runtime and run all four code cells. It fetches an immutable source revision, verifies the approximately 15.6 MB earlier archive, reproduces the two analyses, runs them again in separate verification directories and compares their saved bytes. The downloaded `baomai-proposal-audit-r63-results.zip` contains the two analyses and their source/input identities. Rerunning a fully completed CPU analysis is optional.

The 20 photographs remain development data. This diagnostic supplies no independent capture-group result, new model accuracy claim or robotic-arm localization measurement.
