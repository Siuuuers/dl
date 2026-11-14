# Missing corners, atlas colours, and training readiness

> Historical review snapshot. The [complete visible-front sweep and prepared handoff](complete-label-review-2026-11-14.md) supersedes the current-status counts in this report; its original evidence remains preserved.

**Subsequent focused review:** the [new037/new038 comparison](box-extent-comparison-2026-11-14.md) adds a row-1 extent/landmark hold without changing labels. It supersedes this snapshot's current-status and atlas counts; the missing-corner rule and new130 closure below remain valid.

**A supported visible cassette front can have a solid cyan box with zero, one or two locatable lower-front corners.** Corner availability and review status are separate decisions. This revision clarifies the atlas legend and closes one overly cautious missing-corner hold. It changes no training label, source image, validation reference or model weight.

## What the requirements mean

The [upstream baomai data README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md#L31-L39) defines the two actual lower-front corners, requires `0 0 0` when a point cannot be reliably located, and retains a visible endpoint independently of the other. The [local annotation policy](annotation-policy.md#1-one-identifiable-visible-front-one-row) explicitly permits an identifiable visible front with both points missing. Its visible-front box convention is an adopted project decision informed by the historical upstream editor; the current upstream README does not itself fully specify box extent.

| Accepted annotation | Box in the atlas | Saved points | Connecting line |
| --- | --- | --- | --- |
| Front and visible extent established; both actual corners locatable | Solid cyan | Both endpoints `v=2` | Yes |
| Front and visible extent established; only one actual corner locatable | Solid cyan | One `v=2`; the other exactly `0 0 0` | No |
| Front and visible extent established; neither actual corner locatable | Solid cyan | Both endpoints exactly `0 0 0` | No |
| A recorded identity, observed-extent or landmark decision is still unresolved | Dashed amber for the affected existing row | Preserve its current saved states pending adjudication | Only if both saved states are `v=2`; this does not resolve the review question |

One visible corner does **not** mean a `v=1` label. Each of the two slots is independently `v=2` or `v=0` under this contract. Do not move a point onto a gripper edge, box corner or inferred hidden junction to complete a pair.

Cyan has a precise bookkeeping meaning: no unresolved row-level question is recorded in the loaded review ledgers. It does not certify every field independently or prove that every object in the photo has been labelled. A possible omitted instance can remain an image-level question without any existing amber row; `new_000096` is such a case. The atlas index marks these photos with an asterisk.

The atlas displays saved reference annotations, so it has no model confidence threshold. The existing [prediction display contract](target-clarification-2026-11-14.md) separately thresholds predicted endpoints and draws a line when both pass its display checks. A high prediction score does not prove actual physical visibility. This revision changes neither display behaviour nor model confidence.

## One hold closed correctly, with no coordinate changes

In `new_000130-052`, original/current rows **54 and 55** already had accepted front identities and observed boxes. The remaining question concerned two proposed visible right-corner coordinates. The [independent image review](../results/review/aperture-labels-20261114/late-independent-review.json) did not establish those bright/dark junctions as the actual lower-front corners. Both saved slots in both rows were already canonical missing values.

[Policy section 3](annotation-policy.md#3-locate-the-two-actual-lower-front-corners) supplies a finished annotation decision for precisely this situation: retain `0 0 0` when the actual corner is not reliably locatable. Rejecting an unsupported visible-point proposal does not require leaving a correct missing-point state unresolved indefinitely.

The [new status revision](../results/review/visibility-readiness-20261114/review-status.json) therefore changes this case from partial to resolved within the observed-reference scope, removes rows 54/55 from its held-row list, and records `accepted_keep_canonical_missing`. Its inherited lower-front row and previously rejected duplicate proposal remain unchanged. Every other case object is identical to the frozen parent status.

The [bounded semantic audit](../results/review/visibility-readiness-20261114/corner-only-hold-audit.json) found no other remaining case that could be closed solely by accepting existing missing slots. Other held cases have an independent question about front count, material ownership, observed boundaries or an existing visible-point assignment.

## Exact current counts

The unchanged candidate has **137 training photos, 3,770 objects, 6,854 locatable endpoint slots, 686 missing slots and 13 empty photos**. Counts describe annotations, not model performance.

| Training atlas colour | Zero locatable corners | One locatable corner | Two locatable corners | Total rows |
| --- | ---: | ---: | ---: | ---: |
| Solid cyan | 77 | 435 | 3,157 | 3,669 |
| Dashed amber | 32 | 33 | 36 | 101 |
| Total | 109 | 468 | 3,193 | 3,770 |

Thus **512 cyan boxes have fewer than two locatable corners**. Conversely, 36 amber rows already contain two saved visible corners. The colour is not a corner-count filter.

Of the original 87 review groups, **29 are resolved, 38 are partial and 20 are held**. Including the separately discovered new096 question, **59 groups across 46 training photos remain unresolved**. Amber existing rows occur in 35 of those photos; the remaining image-level questions must still be considered. These are review-work counts, not error rates or a claim that every held annotation is wrong.

## Decision on the next training run

**Do not launch a full corrected-dataset run from the present review candidate.** Its manifest remains `release_ready:false` and `training_recipe_id:null`. Missing corners are not the blocker. The blockers are remaining material annotation decisions and the absence of a registered training contract for this candidate kind.

The current kind is `training_observed_review_composition`. It is absent from the recipe-kind mapping in [`scripts/pose_checkpoint.py`](../scripts/pose_checkpoint.py) and from the verifier dispatcher in [`scripts/stage_pose_training.py`](../scripts/stage_pose_training.py). The dedicated review composer deliberately verifies an unreleased candidate. Renaming the kind or flipping readiness flags would not establish a valid training contract.

Do not rerun the [PR62 protocol](../experiments/occlusion-training-2026-11-13.json) to train these repairs: it is pinned to `annotations-2026-11-10`. That would reuse the older supervision.

The concrete path to the next run is:

1. Resolve material identity, count and boundary questions, or explicitly quarantine affected photos with recorded train-set changes. Correct missing endpoint states may remain missing.
2. Freeze the chosen supervision, provide its verifier/staging support, and register a recipe that names the intended control and candidate. The generic runner's reference arm is not automatically the actual PR35 supervision.
3. Use [`scripts/train_pose_comparison.py`](../scripts/train_pose_comparison.py) for a bounded trial and then the specified GPU comparison. Keep the existing development reference and acceptance criteria fixed, and retain PR35 until the comparison justifies replacement.

[Policy section 5](annotation-policy.md#5-empty-missing-and-unresolved-are-different-states) also permits a **separately documented controlled incremental experiment** that preserves legacy annotations and discloses its limited correction scope. Such an experiment need not claim complete reannotation, but still needs an explicit dataset/recipe contract and compatible staging support. This revision creates neither a release contract nor such an experimental recipe.

The still-useful physical question on PDF page 5 is narrower than cassette presence: in `new_000030`, does the horizontal transition near original-image y=482 separate two cassette fronts, or lie within one front? Cassette material behind the opening is already confirmed; actual front count and observed bounds remain open.

## Atlas revision and verification

The updated **163-page PDF** still includes all **157 working photos**: 137 training candidate pages and 20 pages from the separately versioned development reference. Page 2 explains the colour/visibility distinction; page 4 removes new130's review asterisk; page 99 shows its two accepted missing-corner rows in cyan. All photo and label identities, counts and coordinates are unchanged.

The [atlas audit](../results/review/visibility-readiness-20261114/atlas-audit.json), [bounded PDF QA](../results/review/visibility-readiness-20261114/atlas-qa.json) and [verification receipt](../results/review/visibility-readiness-20261114/verification.json) record the revision. Original validation remains 20 images / 664 objects; the separate development reference remains 20 images / 663 objects. They have not been conflated or rewritten.

```bash
python -B scripts/compose_observed_pose_review.py --root . verify \
  --candidate data/candidates/aperture-reviewed-labels-2026-11-14

python -B scripts/render_label_atlas.py --root . \
  --candidate data/candidates/aperture-reviewed-labels-2026-11-14 \
  --status results/review/visibility-readiness-20261114/review-status.json \
  --output /path/outside/repo/cassette-label-atlas-2026-11-14.pdf \
  --audit /path/outside/repo/atlas-audit.json
```

Candidate manifest SHA-256 remains `565570c24165b59ad3ee2368ebd80a9c9ff4bb25d96da87bcdee93602bd75983`.

No training, inference, checkpoint replacement or model-performance gain occurred in this revision.
