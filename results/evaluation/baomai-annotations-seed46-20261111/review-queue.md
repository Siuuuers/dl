# Seed46 targeted review queue

This queue contains **26 unique objects**: all **24 candidate misses** against the PR30 revised reference at confidence 0.25, one additional 10 px pair diagnostic, and one additional gained detection. The lost detection is already one of the 24 misses and appears only once.

## What deserves attention

**21 of 24 misses persist across all five compared variants.** All **19 previously retained-unresolved reference rows** remain candidate misses; 18 of those are in the persistent group. The remaining misses are two previously amended rows (new_000008:33 and new_000009:57) and three rows outside the previous 33-row queue (new_000009:52, new_000011:61, new_000087:15).

Five variants means epoch173 plus the seed45 and seed46 control/candidate pairs. These models and reused views are related; persistence is a review priority, not five independent trials or proof that a label is correct. The older diagnosis used the seed43/44 family; only its documented visual observations and PR30 dispositions are carried forward.

The 24 current misses have 8 with 0 locatable corners, 10 with 1 locatable corner, 6 with 2 locatable corners. A high two-corner success rate therefore cannot summarize this miss queue.

## Ranked objects

M = missed by the fixed one-to-one reference matcher; D = matched. C = control; A = reviewed-annotation candidate. Rows are one-based **revised-reference rows**. The JSON also supplies the original row from the manifest map, exact box/corner coordinates, source hashes, visibility, prior reasoning and transition details.

| Rank | Image:row | Tier | Locatable corners | M0 | C45 | A45 | C46 | A46 | Prior disposition / extra role |
| ---: | --- | ---: | ---: | :---: | :---: | :---: | :---: | :---: | --- |
| 1 | new_000007:7 | 1 | 0 | M | M | M | M | M | Unresolved |
| 2 | new_000007:14 | 1 | 0 | M | M | M | M | M | Unresolved |
| 3 | new_000008:36 | 1 | 0 | M | M | M | M | M | Unresolved |
| 4 | new_000009:8 | 1 | 1 | M | M | M | M | M | Unresolved |
| 5 | new_000009:53 | 1 | 1 | M | M | M | M | M | Unresolved |
| 6 | new_000009:56 | 1 | 0 | M | M | M | M | M | Unresolved |
| 7 | new_000011:15 | 1 | 0 | M | M | M | M | M | Unresolved |
| 8 | new_000053:1 | 1 | 1 | M | M | M | M | M | Unresolved |
| 9 | new_000055:26 | 1 | 1 | M | M | M | M | M | Unresolved |
| 10 | new_000056:4 | 1 | 2 | M | M | M | M | M | Unresolved |
| 11 | new_000057:1 | 1 | 0 | M | M | M | M | M | Unresolved |
| 12 | new_000059:1 | 1 | 0 | M | M | M | M | M | Unresolved |
| 13 | new_000086:15 | 1 | 2 | M | M | M | M | M | Unresolved |
| 14 | new_000086:57 | 1 | 1 | M | M | M | M | M | Unresolved |
| 15 | new_000088:15 | 1 | 2 | M | M | M | M | M | Unresolved |
| 16 | new_000088:57 | 1 | 1 | M | M | M | M | M | Unresolved |
| 17 | new_000089:15 | 1 | 2 | M | M | M | M | M | Unresolved |
| 18 | new_000090:15 | 1 | 2 | M | M | M | M | M | Unresolved |
| 19 | new_000008:33 | 1 | 1 | M | M | M | M | M | Amended |
| 20 | new_000009:57 | 1 | 1 | M | M | M | M | M | Amended |
| 21 | new_000011:61 | 1 | 0 | M | M | M | M | M | Outside prior queue |
| 22 | new_000008:1 | 2 | 1 | M | M | M | D | M | Unresolved; detection loss |
| 23 | new_000009:52 | 2 | 1 | M | D | D | M | M | Outside prior queue |
| 24 | new_000087:15 | 2 | 2 | D | D | D | M | M | Outside prior queue |
| 25 | new_000090:43 | 3 | 1 | M | M | M | M | D | Amended; detection gain |
| 26 | new_000007:15 | 3 | 2 | M | D | D | D | D | Outside prior queue; 10 px pair loss |

Tier 1 contains the 21 persistent misses. Tier 2 contains three variable misses; tier 3 contains the two additional diagnostics.

## Three paired transition cases

- **new_000007:15:** the official control match has corner errors **5.760 / 1.508 px**; the candidate match has **14.596 / 14.356 px**. The candidate also returns a higher-IoU proposal at slightly lower confidence. The fixed confidence-ordered matcher selects the first proposal, so inspect duplicate/assignment competition together with top-boundary corner geometry. This is a detected-object localization diagnostic, not another detection miss.
- **new_000008:1:** the candidate loses a control box match on an already unresolved top-clipped reference. The control's endpoint confidences are below 0.50, so the box-match loss does not create a further lost returned visible endpoint. This object is recorded once in tier 2.
- **new_000090:43:** the candidate gains a match as overlap moves from **IoU 0.499094 to 0.502481** across the fixed 0.50 rule. PR30 amended this row's bottom edge. Preserve that correction and its residual uncertainty; the narrow score transition does not establish a newly discovered physical object.

## Next work

Keep epoch173 as the working default and stop seed searching. Use this queue for targeted supervision and error review: carry unresolved reference issues forward, inspect the small set of changed detections and duplicate/assignment behavior, and audit analogous training examples only for a specific remaining policy inconsistency. Any next intervention must be separately identified and justified from pixels. Do not request a seed47 sweep or alter reference labels to make predictions score better.

Start with original images and the existing annotation policy. For unresolved rows, record what question remains unanswerable from the available pixels instead of repeating a supposed correction. For clear targets, inspect whether the failure is due to box extent, confidence, duplicate assignment, occlusion or corner localization. Use those concrete phenotypes to review analogous examples in the existing training split. The prior eight-file training overlay has already been tested; repeating it with another seed does not resolve the identified supervision questions.

No validation images enter training, no hidden corner coordinates are invented, and no reference is silently removed or changed based on predictions. Any future model-change claim needs an appropriate evaluation separate from repeatedly reviewed examples. Seed45 L4 versus seed46 A100 also prevents calling cross-seed differences a pure seed effect.

## Reproduce

The generator uses only the Python standard library, verifies all 20 original PNG and label hashes plus the revised labels and row mappings, and checks that the two M0 miss sets agree. It reads existing predictions only to verify their hashes against the independent case audit; it performs no inference, checkpoint loading or label edits.

```bash
python generate_review_queue.py \
  --repo /path/to/dl \
  --seed45-summary /path/to/seed45/evidence/seed-45/comparison-original-and-pr30.json \
  --seed46-summary /path/to/evidence/seed-46/comparison-original-and-pr30.json \
  --case-details /path/to/independent/seed46-case-details.json \
  --output /path/to/review-queue
```

The seed46 summary's sibling m0/m1/m2 directories must contain their predictions.jsonl files. Outputs are `review-queue.json` and this README. All source identities and the generator hash are recorded in the JSON. The queue is a read-only triage artifact and performs no new visual adjudication.
