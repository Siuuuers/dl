# Round 6 annotation pipeline verification

**Result: verified. No training was run.** The executable check staged the actual reference and candidate datasets, compared their bytes, and tested two invalid artifacts in an isolated copy. Its exact inputs and source-code hashes are recorded in [verification.json](verification.json). The complete file inventory is in [staging-manifest.json](staging-manifest.json).

| Check | Observed result |
| --- | --- |
| Original camera images | All 157 PNGs matched their source hashes and dimensions. |
| Training cohort | Both arms retained 137 photos, including 13 empty scenes. |
| Training fronts | PR35 reference: 3,771; r6 candidate: 3,779. |
| Training validation | All 20 photos retained the exact original label bytes in both arms: 664 fronts. |
| Held training photographs | All 44 retained exact PR35 label bytes in both arms, including all 31 old-series photos. |
| Separately reviewed development labels | Seventeen development label files differ from the original source labels. None was substituted into staged validation. |
| Fixed scoring references | The 666-front primary and 663-front historical references retained their previous hashes and counts. |
| Earlier review verifiers | Round 4 and round 5 both passed. |
| Tampered candidate with recomputed label hash | Rejected because it differed from the exact reviewed/held composition. |
| Stale field-highlighting metadata | Rejected because the declared changed fields differed from the label tokens. |

The r6 verifier also requires complete visual-review coverage for the 157-photo cohort, valid immutable base-row identities, valid uncertainty regions, and exact row-map replay. These checks establish consistency and provenance. They do not prove that a physical boundary interpretation is correct, that unresolved labels are certified, or that a trained model will improve.

## Reproduce

Hydrate the repository's original camera PNGs through Git LFS, then run from the checkout:

```sh
python3 scripts/check_front_review_r6.py --output /tmp/front-review-r6-verification
```

The output directory must be new and outside the checkout. The checker creates both staged datasets and `verification.json`; it leaves original labels unchanged. The staging manifest records the actual output location, creation time, and dataset-YAML hashes, so a later run will produce a different staging-manifest hash even when all scientific inputs and checks agree.

The checker does not load model weights or execute a trainer. Weight verification and the pinned source checkout are additional notebook preflight checks before any training run.
