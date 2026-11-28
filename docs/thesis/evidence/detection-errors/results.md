# Cached detection error audit

All six baseline rows reproduce exactly; no inference or training was performed.

| Reference | Method | Missed | Extras | Duplicate subset | Other IoU 0.10–<0.50 | Other IoU <0.10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| supplied664 | original | 39 | 56 | 11 | 38 | 7 |
| supplied664 | proposed | 46 | 69 | 8 | 52 | 9 |
| frozen666 | original | 26 | 41 | 11 | 25 | 5 |
| frozen666 | proposed | 34 | 55 | 9 | 39 | 7 |
| r50 | original | 34 | 61 | 5 | 49 | 7 |
| r50 | proposed | 18 | 51 | 5 | 36 | 10 |

## Current method under the reviewed reference

| Locatable reference corners | Fronts | Missed |
| --- | ---: | ---: |
| 0 | 17 | 5 |
| 1 | 54 | 13 |
| 2 | 583 | 0 |

## Interpretation limits

- Overlap strata are geometric review cues, not physical background, label-error or object-identity verdicts.
- The native cache was already filtered at confidence 0.25, ordinary NMS 0.70 and max_det 300, then clipped. It cannot expose discarded native proposals or lower-score responses.
- No eligible cached candidate does not establish that the network produced no candidate. New pre-filter decoding would be needed to separate confidence, native NMS and proposal failures.
- Pipeline stages identify where eligible records disappear within the saved cache only; they do not establish a physical cause.
- Three annotation references describe the same repeatedly inspected 20 development photographs and are not independent test sets.
- Corner availability is reference-defined. A missing corner does not remove its front from detection evaluation.
- Reflected-view complementarity is an annotation-based diagnostic, not an implementable oracle or evidence that merging views improves the result.
- This replay changes no labels, predictions, model, matching rule, confidence cutoff or deployment setting.
