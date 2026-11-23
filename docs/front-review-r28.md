# R28: annotation amendments and the next training notebook

23 November 2026, Asia/Hong_Kong.

**R28 accepts 23 coordinate amendments across 15 photographs and prepares a
fresh R26-versus-R28 training comparison. PR35 remains selected. R28 has not
been trained or scored yet.** The previous returned result is preserved in
[PR103](https://github.com/Siuuuers/dl/pull/103) and its
[evaluation report](../results/evaluation/pr103-r26-20261123/README.md).

## Deliverables

- `r28-annotation-amendments.pdf`: 21 pages, with all 23 changed rows shown once
  in 20 original/R27/R28 cropped comparisons. Nearby fronts and rack provide
  context. Retained-only cases are recorded in the ledger.
- [R28 Colab notebook](../notebooks/baomai_front_review_r28.ipynb): select a GPU
  runtime, choose **Run all**, and return `baomai-front-review-r28-results.zip`.
- [Exact decisions](../results/review/front-review-20261123-r28/row-review.json),
  [immutable review](../data/reviews/fronts-2026-11-23-r28/manifest.json),
  [training-input delta](../results/review/front-review-20261123-r28/training-input-delta.json),
  and [fixed experiment](../experiments/front-review-2026-11-23-r28.json).

The PDF inventory records its exact bytes, source hashes and complete coverage:
[amendment inventory](../results/review/front-review-20261123-r28/amendment-pdf-inventory.json).
It is an annotation report. A new missed/error report follows the GPU return;
there are no new R28 predictions to inspect yet.

## What changed

All 25 requested identities were inspected: 23 textual rows and two additional
rows from the `016` screenshot. The `056:12` screenshot overlaps a textual
request. Eight additional training cases came from a predefined TXT screen.
Across these 33 decisions, 23 rows change and 10 remain exact. No objects are
added, deleted or renumbered. Class tokens, visibility, unavailable-point zeros
and all 11 prior standards are unchanged. The explicit revisits of `008:19`
and `008:29` retain their saved lines.

| Photograph | Split | Changed rows | Main amendment |
| --- | --- | --- | --- |
| 008 | Development | 5, 11, 23 | Two left turns; row 11's upper boundary. |
| 010 | Development | 14, 26, 33 | Include exposed cassette strips/rims; row 26's right turn. |
| 011 | Development | 49 | Right turn and outer rim. |
| 016 | Training | 26, 27 | Screenshot-guided right boundary and right turn. |
| 017 | Training | 18 | Remove the preceding front from the upper box extent. |
| 028 | Training | 59 | Remove rack/background below the visible front. |
| 055 | Development | 23 | Frame-clipped right material boundary near x640. |
| 056 | Development | 12 | Screenshot-guided upper boundary at y437. |
| 058 | Development | 7, 18 | Left turn; include row 18's own upper rim. |
| 059 | Development | 7, 10 | Left turns. |
| 087 | Development | 53 | Right turn. |
| 088 | Development | 52 | Both lower-front turns. |
| 089 | Development | 50 | Left turn and its outer rim. |
| 090 | Development | 44, 45 | Both turns; row 45's lower boundary by one pixel. |
| 094 | Training | 9 | Upper boundary at this front's own rim. |

This is **5 changed training rows and 18 development rows** in R27-to-R28.
Fourteen rows have a box change; 16 endpoints change across the 23 rows.
Typical qualitative edge uncertainty is 1-3 original pixels. Decimal storage
does not establish subpixel physical accuracy. Row `090:45`'s one-pixel box
refinement is explicitly less certain than the larger corrections.

### Important interpretations of the feedback

Coordinates are original 1280 x 720 pixels, with x increasing right and y
increasing down. `ymin` is the top and `ymax` the bottom.

- `056:12`: the blue line marks **ymin**, now 437. Keep ymax near 475.54;
  the predicted 481.12 includes rack/background. The R27 left-point correction
  and missing right point remain exact.
- `008:11`: change **ymin to 456**, keeping ymax at 501. The old top crosses
  the previous cassette; the predicted bottom goes too low.
- `058:18`: change **ymin to 401**, including the visible upper rim at the
  left. Keep ymax at 445. The tilted rim makes a rectangular box include
  context on the right.
- `008:5`: left x moves **978 to 984**, preserving y15. The requested
  direction is supported, but a literal +12 would move inside the front.
- `055:23`: xmax becomes **640**, with approximately 2-3 px uncertainty.
  x538 would remove much of the front. It was neither applied nor silently
  reinterpreted as xmin.
- `010:14/33`: xmin becomes **598/599**, including the thin strip left of
  the gripper. Visible column boundaries and front levels support associating
  that strip with the aperture fragment. This correspondence is an occlusion
  inference, not observation through the gripper. The rectangle spans the
  occluder between observed fragments; no hidden corner is invented. Row 14's
  xmax remains 747.
- `010:26`: xmin 936, xmax 1096, right point (1093,427.5). Keep ymax near 430;
  the predicted 435.88 reaches into the next front.
- `016:26/27`: xmax 1108 to 1105 and right point (1084,398) to (1089,394.5),
  respectively, checked against the original pixels and both screenshot cues.

All other supported point refinements and retained fields are documented in
the exact ledger and PDF. Prediction coordinates are inspection cues, not
automatic annotation replacements. Brightness alone is also insufficient:
shadowed cassette material can be grey and a real corner can straddle a dark
boundary.

### One unresolved identity: `009:55`

The current gold file has 57 rows. Gold row 55 is the gripper-aperture fragment
at approximately x682-720, y471-494. Both one-based and zero-based
interpretations of prediction numbers 55 and 60 instead identify the left
exposed stack around x594-660. No retained PR103 prediction overlaps the
gold 55 patch. Its label therefore stays byte-identical. A marked crop of the
intended prediction is needed before moving that physical object's bounds;
this does not block the other amendments or training notebook.

## TXT geometry screening

The [screening script](../scripts/rank_front_geometry_candidates.py) reads
the immutable R27 TXT files. It computes, for available lower-front points,

```text
bottom_gap = ymax - max(available point y)
upper_gap  = min(available point y) - ymin
height     = ymax - ymin
```

A bottom gap above `max(5 px, 0.25 * height)` is a review candidate. The second
rule flags another object's locatable point more than 2 px inside all four
box sides. It records the point's owning row and relative position. These
are declared inspection thresholds, not physical laws or calibrated error
classifiers. Unavailable zero triplets are excluded from point calculations.

The screen covers 157 files / 4,447 objects / 8,062 available endpoints.
There are 159 objects without any locatable corner. Across the other 4,288,
the median upper gap is 24.93 px and the median bottom gap is 2.83 px. Thus
the lower-front points are generally a better reference for **ymax** than
for ymin. The front's height contributes to the upper gap.

There are 256 large-bottom-gap flags and 648 foreign-point-overlap flags,
covering 879 distinct objects. They are not 879 confirmed mistakes.
Perspective, tilted fronts, clipping and occlusion can make both flags
legitimate. In particular, a point of the upper cassette may lie inside an
axis-aligned rectangle around a sloping lower cassette.

The predefined eight-case training sample gives three amendments
(`017:18`, `028:59`, `094:9`) and five retentions (`006:56`, `012:2`,
`014:8`, `018:1`, `029:42`). Each received original-pixel review. The five
retentions preserve tilted, clipped or ambiguous extents. This nonrandom
sample does not measure dataset error prevalence or screening precision.
See the [full screen](../results/review/front-review-20261123-r28/geometry-screen.json).

## The actual training experiment

R27 was reviewed but never trained. Therefore the relevant comparison is
**fresh R26 supervision versus fresh R28 supervision**, both in one GPU
session and independently initialized from original epoch173. Reusing a
historical R26 checkpoint as the reference would mix label changes with a
different training history. The earlier L4/A100 difference makes a paired
run particularly useful.

The exact treatment is **10 rows across 9 training photos**:
`013:27`, `014:35`, `016:26`, `016:27`, `017:18`, `028:59`, `049:3`,
`051:20`, `083:12`, `094:9`. This includes the seven untrained R27 rows,
with the two `016` rows refined again, plus three newly corrected rows.
The byte comparison identifies six box rows, one left endpoint and four
right endpoints. No object or visibility difference is introduced between
the arms.

| Item | Fixed treatment |
| --- | --- |
| Reference arm | Exact eligible R26 training supervision. |
| Candidate arm | Eligible R28 training supervision. |
| Initial checkpoint | Original epoch173, independently loaded for each arm. |
| Recipe | Seed 46; 30 epochs; batch 4; image 1280; FP32; AdamW; initial lr 0.0001. |
| Selection | Final epoch, fixed in advance. |
| Runtime | Same recorded GPU/software for both arms and resource trial; A100 recommended. |
| Training population | 137 photos / 3,784 objects, including 13 empty scenes, in each arm. |
| Preserved treatment exclusions | Photo 015 and 31 old-series photos retain identical R26 labels in both arms; these photos are still present. |
| Trainer validation | Original 20 photos / 664 fronts, byte-identical between arms. |
| Post-return primary reference | Frozen 666 fronts. |
| Historical diagnostic | Frozen 663 fronts. |
| Current-label diagnostic | R28's 661 development fronts, reported separately as exploratory. |

The complete review snapshot has 3,786 training objects. The staged comparison
has 3,784 because it preserves the established held/excluded supervision.
The training-input report makes that distinction explicit; no new deletion
is introduced by R28. Development amendments affect label-based diagnostics,
not the labels used to train these models.

### Running the notebook

1. Open `notebooks/baomai_front_review_r28.ipynb` in Colab.
2. Choose a GPU runtime; A100 is recommended. Choose **Run all**.
3. Complete the Google Drive authorization when Colab requests it, so verified
   completed bundles and the input-delta report are recoverable.
4. Let the notebook fetch its exact pinned source, download and hash-check
   the images/initializer, verify labels, run the resource trial and train
   both arms. Keep the pair in the same recorded runtime. A failed check
   stops the run with its actual error.
5. Keep the final `baomai-front-review-r28-results.zip` and return it for
   evaluation. The ZIP includes both final checkpoints, staged input
   identities, the embedded exact input delta, training receipts and logs.

Do not tune confidence, epochs or seeds after seeing these results. Compare
both new arms and retained PR35 using the same inference/evaluation settings;
inspect gained and lost identities, missed targets, unmatched outputs,
duplicates, empty scenes and both-corner success within 5/10 original pixels.
Eligible misses remain in denominators. Conflicting frozen/current-reference
rankings remain a reason for further adjudication, not automatic promotion.

## Next improvement after this run

The immediate purpose is to measure whether these ten training changes help.
More annotation micro-adjustments should follow specific errors, not a target
number of changes. The highest-value data addition remains a new capture
session containing gripper-adjacent partial fronts, clipped small fronts,
tilted/shadowed turns, rack/hand lookalikes and ordinary/empty scenes. Reserve
that session before training. Keep identifiable hard fronts and honest
missing-point visibility rather than deleting them to improve a score.

This remains RGB front detection with two image-plane corners. Image-coordinate
agreement alone does not establish metric 3-D or robot grasp accuracy.

## Reproduction

Original images must be materialized from LFS before replay. Their hashes are
verified. Run from the repository root:

```bash
python scripts/front_review_r28.py --root .
python scripts/front_review_r28_training.py --root .
python scripts/render_front_review_r28_amendments.py --root . --output /tmp/r28-annotation-amendments.pdf --crop-dir /tmp/r28-crops
```

The two builders refuse to overwrite an existing immutable snapshot with
different bytes. The source-pinned notebook is generated after the source
commit exists. GPU training is the outstanding experiment; CPU preflight,
dataset staging/loader checks and initializer-forward verification establish
input readiness only.
