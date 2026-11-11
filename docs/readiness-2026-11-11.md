# PR30 readiness snapshot on the supplied cassette images

**Later follow-up:** the user delegated assistant review, and the [completed low-confidence review](low-confidence-review-2026-11-11.md) added two omitted fronts in a separate 663-object reference and selected candidate confidence 0.15 for reviewed proposals. The tables below remain the historical PR30 661-object snapshot; do not combine their denominators with the new reference.

**Use the system as a human-reviewed offline prototype. Retain epoch173 as the working default and seed46 M2 as an explicitly selected experimental candidate. The evidence does not establish unattended reliability on unseen scenes.** No new training is justified merely by repeating the completed annotation experiment.

Date: **2026-11-11, Asia/Hong_Kong**. This report summarizes saved PR36 results; it performs no new inference, training or annotation review. The [machine-readable calculation](../results/verification/baomai-readiness-20261111/readiness.json) preserves every image count, source hash and denominator. The [completed confirmation](annotation-confirmation-2026-11-11.md) remains the experiment record.

## What works, and what the headline hides

The revised PR30 reference contains **20 images, 661 objects, 1,228 locatable corners and 584 objects with both corners locatable**. Three images are empty. All primary comparisons use confidence **0.25**, NMS IoU **0.70**, matching IoU **0.50**, endpoint confidence **0.50**, and 1280 rectangular FP32 inference.

| Primary measure | Epoch173 | Original-label control | Reviewed-annotation candidate |
| --- | ---: | ---: | ---: |
| Misses / 661 objects | 27 | 24 | 24 |
| Extra detections | 54 | 58 | 52 |
| Object recall | 95.92% | 96.37% | 96.37% |
| Object precision | 92.15% | 91.65% | 92.45% |
| Missing locatable corners / 1,228 | 26 | 23 | 22 |
| Both corners within 5 px / 584 | 528 | 527 | 529 |
| Both corners within 10 px / 584 | 575 | 578 | 577 |

Recall is matched objects divided by reference objects; precision is matched objects divided by returned detections. Both-corner success includes missed eligible objects in its denominator. However, **577/584 = 98.80% describes the two-locatable-corner subgroup**, not overall accuracy.

| Locatable reference corners | Objects | Epoch173 misses | Control misses | Candidate misses | Candidate miss rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two | 584 | 7 | 6 | 6 | 1.03% |
| One | 60 | 12 | 10 | 10 | 16.67% |
| Zero | 17 | 8 | 8 | 8 | 47.06% |

**Eighteen of the candidate's 24 misses fall outside the two-corner denominator.** The current [annotation policy](annotation-policy.md) retains every identifiable visible front with supported positive-area extent, including partial fronts with one or neither corner locatable. Entirely hidden or ambiguous identities are not invented. No difficult object is removed to improve these rates.

Ten pixels remains a **provisional Euclidean radius for each corresponding corner in the original 1280 × 720 image**. It is not annotation slack, millimetre accuracy, or a complete acceptance gate. The stricter 5 px result remains visible. Candidate coordinates are returned on 64/68 unlocatable sides of matched objects; this measures disagreement with reference availability, not measurable error at an unseen corner.

## Whole scenes still need review

The following fractions use only the **17 nonempty images**. “No detection errors” means FN=FP=0; it does not establish accurate corners or complete task success.

| Scene measure | Epoch173 | Control | Candidate |
| --- | ---: | ---: | ---: |
| No missed objects | 4/17 | 3/17 | 3/17 |
| No detection errors | 1/17 | 0/17 | 0/17 |
| Maximum misses in one image | 5 | 5 | 5 |
| Maximum extras in one image | 14 | 13 | 12 |

Both maxima occur in `new_000009`. The candidate's miss-free images are `new_000010`, `new_000054` and `new_000058`; epoch173 additionally has `new_000087`. Epoch173's sole detection-error-free nonempty image is `new_000058`. All three models produce zero extras on the three empty images, reported separately.

Thus better aggregate counts have not produced more entirely correct scenes. These are revised-reference results; earlier original-reference scene conclusions must not be substituted.

## What another change would need to accomplish

Against epoch173, the candidate improves the seven chosen primary headline measures in the confirmation report. Against its matched control, misses tie, extras fall by six, 5 px pair successes increase by two and 10 px pair successes decrease by one. Corner-error changes are mixed. Additional training contributes to the baseline comparison; the annotation-specific strict miss reduction did not repeat. Seed45 used L4 and seed46 used A100, so between-experiment variation is not solely a seed effect.

The already measured candidate confidence **0.15** gives **21 misses and 78 extras**: three recovered matches at the cost of 26 additional extras. Recall becomes 96.82%, precision 89.14%; 10 px pair success remains 577/584, and miss-free scenes remain 3/17. Keep 0.25 unless the review workload justifies this explicit tradeoff.

The [26-object queue](../results/evaluation/baomai-annotations-seed46-20261111/review-queue.md) retains all nineteen unresolved reference dispositions. Persistence across related models does not prove a label wrong. Continue only from a concrete image-supported correction or failure hypothesis; do not start another seed search. The candidate can be exercised through the [folder workflow](predict-pose-folder.md) without promoting it.

The twenty development images already participated in model selection and repeated review. Correlated objects and frames are not independent trials. Required success rates, an acceptable extra-proposal workload and unseen-scene evidence remain unset. These limits permit useful work on the supplied images while limiting claims about future inputs.

## Reproduce this summary

The generator verifies the published capsule SHA-256, its comparison-member hash against the official report, and the revised-reference manifest hash. It reads **`comparison-original-and-pr30.json` → `revised`**, sums its per-image counts, and checks them against both frozen aggregates and the official seed46 summary. It never uses the separate original-reference `per_image.json` files. No model weights or images are loaded.

```bash
python3 results/verification/baomai-readiness-20261111/summarize_readiness.py --check
```

Omit `--check` to regenerate the deterministic JSON, or use `--output /tmp/readiness.json` for a separate copy. The capsule, member, official summary, reference manifest and generator hashes are recorded in the JSON.
