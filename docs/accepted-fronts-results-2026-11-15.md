# PR64 accepted-fronts measurement (15 November 2026)

**Decision: retain PR35.** This is a measured, completed 3-model CPU-inference comparison using verified checkpoint bytes, not a training-loss proxy.

[Successful GitHub Actions run](https://github.com/Siuuuers/dl/actions/runs/43717526943) · [Full evidence artifact: 60 prediction arrays, primary + diagnostic reference scores and training audit](https://github.com/Siuuuers/dl/actions/runs/43717526943/artifacts/14162004655)

## Fixed comparison

Models: PR35 seed46 checkpoint, fresh PR64 control (original PR35 labels), and PR64 treatment (accepted-fronts label revision). Both fresh models trained on the same NVIDIA L4, independently from epoch173 for 30 epochs. Evaluation uses pinned Ultralytics 8.4.182 on CPU, confidence 0.13, NMS IoU 0.70, endpoint confidence 0.50, matching IoU 0.50, images at size 1280, and Euclidean original-pixel error radii 5/10. Primary 2026-11-12 and separate diagnostic 2026-11-14 reference both contain 20 photos, 663 objects, 584 pairs with two locatable corners and three empty scenes.

### Primary 2026-11-12 reference

| Metric | PR35 | PR64 control | PR64 candidate |
|---|---:|---:|---:|
| Matched cassettes | **643** | 641 | 639 |
| Misses (lower better) | **20** | 22 | 24 |
| Unmatched proposals (lower better) | **85** | 88 | 107 |
| Duplicate proposals | 31 | **28** | 31 |
| Other unmatched proposals | **54** | 60 | 76 |
| Locatable endpoints returned / 1,227 | **1,207** | 1,205 | 1,205 |
| Both actual corners within 5 px / 584 | **529** | 527 | 528 |
| Both actual corners within 10 px / 584 | 577 | 578 | **579** |
| Predictions on three empty photos | 0 | 0 | 0 |
| Nonempty scenes without a miss / 17 | **3** | **3** | 2 |

Candidate against matched control: **+2 misses, +19 extras, +1 five-pixel pair, +1 ten-pixel pair**. Its five-pixel result hides six gained and five lost identities. Candidate against retained PR35: **+4 misses, +22 extras, −1 five-pixel pair, +2 ten-pixel pairs**. No 17-photo nonempty scene is completely free of misses and unmatched proposals for any model.

### Diagnostic 2026-11-14 reference

| Metric | PR35 | PR64 control | PR64 candidate |
|---|---:|---:|---:|
| Matched | **648** | 646 | 643 |
| Misses | **15** | 17 | 20 |
| Extras | **80** | 83 | 103 |
| Both corners within 5 px / 584 | **533** | 531 | 532 |
| Both corners within 10 px / 584 | 582 | **583** | **583** |
| Empty-scene outputs | 0 | 0 | 0 |

The second reference modifies nine box extents. Its absolute scores are **not** a model improvement relative to the first reference, but the ranking stays unfavorable to the candidate in both.

## Where the regression occurs

Measured differences in misses/extras, **candidate minus matched PR64 control**, on the primary reference:

| Photo | Miss change | Unmatched-proposal change |
|---|---:|---:|
| `new_000010` | +2 | +9 |
| `new_000090` | 0 | +8 |
| `new_000089` | +1 | +5 |
| `new_000008` | +1 | +2 |
| `new_000009` | −2 | −4 |
| `new_000007` | 0 | −2 |

These are fixed box-association outcomes. Unmatched proposals may be duplicate-looking or near real cassettes and are not individually certified background hallucinations. The extra count is concentrated in 010/090/089 and merits original-image inspection before attributing a physical failure mode.

The PR64 candidate's mean last-five-epoch validation box/pose losses were 1.45078/0.01327 versus 1.41642/0.01219 for its control. Training losses are secondary, not substitutes for detection/coordinate scoring.

## Provenance and decision constraints

PR64 ZIP SHA-256 `700597d31088c516f8ac6ebd1692883878fc95c7d5dccc548100188769c715ee`; PR35 ZIP SHA-256 `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91`. PR35 checkpoint `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`; fresh control `92ab01f57fa5029976d3929ff20af0c37d132aa9abe2b433943d58998e47bb5b`; treatment `2461f93dea1bec63754c08a2918f9077780820d164a1b330334b862d9653654d`.

**Do not promote the treatment.** This is one seed on repeatedly used, temporally related development images, with 13 unresolved reference rows and unresolved physical-label questions. Neither 10-pixel success nor good training losses demonstrate accurate naturally hidden coordinates or reliability on unseen video. Keep PR65 video-continuity corrections separate; define any future correction experiment before training.
