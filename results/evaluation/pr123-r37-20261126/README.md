# PR123 / R37 fixed-inference review

PR123's R37-label candidate preserves the same 15 missed R37 fronts and the same
19 matched rows with an available-corner error above five pixels as its native
R36-label control. It removes one unmatched output (94 → 93), with 31 duplicates
in both arms. These results do not justify replacing the selected PR35 model.

The main error catalogue retains the preserved R37 reference. R38's five
user-directed exclusions and three corrections are reported in a separate
cached-reference replay. All 137 training label files remain byte-identical in
R38; there is no R38-trained model or new inference experiment in this delivery.

## Exact returned run and verification

- [PR123](https://github.com/Siuuuers/dl/pull/123), artifact revision
  `6707967049c548396745e66127691028c17c20a4`.
- Artifact directory: `results/gpu/baomai-front-review-r37-l4-001`.
- [Training source](https://github.com/Siuuuers/dl/tree/86dea96b5294568dcc96f8912f6949d12433380c):
  `86dea96b5294568dcc96f8912f6949d12433380c`.
- Full archive: 39,799,383 bytes, 15 verified members, SHA256
  `41039efb8643f97fc35beb29c71679fd10d0430cdaad43fafa790849d10dac58`.
- Archive manifest SHA256:
  `aa63672bc58e590550fd60ac4d20addda64c646f8c363f390566b9ad3aaeeed9`.
- Protocol SHA256:
  `46a8edcafc41d39f14d230534ef34a7499aa82191d0352d75ae64766f600c9c4`.
- R36-label reference checkpoint SHA256:
  `69b5ee75152523aef294cba79727310ab24b12b81d482a5db0df3716417f9d57`.
- R37-label candidate checkpoint SHA256:
  `51d8a60f95b1b76097513d8b07732de5d5deed5e6e17251a28ddd28e6be1ebc1`.

Both arms passed the exact-source native pair verification: independent
epoch-173 initialization, native Pose26 loss, seed 46, 30 completed epochs,
1,050 batches, 78 optimizer updates, and 4,110 image presentations per arm.
The accepted training delta is two existing rows in two training images:
`new_000006:53` changes a box boundary and `new_000034:35` changes the left
point. No targets, classes, or visibility flags were added or removed.

`archive-audit.json` records wrapper, part, full-archive, member, checkpoint,
receipt, protocol, and native-verification evidence. The initializer's frozen
identity is checked in the model card, protocol, staging, and both receipts;
initializer bytes were not downloaded this round because inference uses the
returned trained checkpoints.

Fresh inference used the exact published source checkout. Its original cache
headers report 157 materialized PNG paths as Git-modified. This was reconciled
explicitly in `inference-source-audit.json`: every image's actual SHA256 and size
match its committed LFS pointer and original manifest, all 47 checked source
files match the pinned source blobs, and no dirty code, label, or protocol path
was found. Original receipt and prediction-header bytes were preserved.

## Fixed common-runtime inference

Both checkpoint arms were inferred freshly on all 20 development images,
including the three empty images `new_000146`, `new_000147`, and `new_000148`.
The core runtime matches the previous saved caches: Python 3.12.14, PyTorch
2.10.0 CPU, Ultralytics 8.4.182, four CPU threads, FP32, full-frame size 1280,
rectangular preprocessing, and batch size one. Box confidence is 0.13,
one-to-many NMS IoU is 0.70, maximum detections are 300, endpoint confidence is
0.50, and matching IoU is 0.50. No new confidence, NMS, or prediction-head probe
was performed.

The PR123 reference reproduces the PR121 candidate's prediction arrays exactly
on all 20 images: same output counts and indices, coordinates, and confidence
values. The different checkpoint hashes do not imply different predictions.
`comparison.json` stores the direct reproduction check and per-model cache and
runtime identities.

## Preserved-reference results

“Unmatched” includes duplicates. The pair columns count objects for which both
available reference corners are within the stated original-image pixel radius;
eligible missed objects remain in the denominator.

| Reference | Model | Missed | Unmatched | Duplicate subset | Both ≤5 px | Both ≤10 px |
|---|---|---:|---:|---:|---:|---:|
| R37, 659 fronts | Native control | 15 | 94 | 31 | 570/583 | 582/583 |
| R37, 659 fronts | Native candidate | 15 | 93 | 31 | 570/583 | 582/583 |
| R37, 659 fronts | Selected PR35 | 43 | 112 | 15 | 546/583 | 569/583 |
| Frozen 666 | Native control | 36 | 108 | 29 | 518/584 | 573/584 |
| Frozen 666 | Native candidate | 35 | 106 | 29 | 517/584 | 573/584 |
| Frozen 666 | Selected PR35 | 18 | 80 | 32 | 533/584 | 582/584 |
| Historical 663 | Native control | 32 | 107 | 29 | 518/584 | 573/584 |
| Historical 663 | Native candidate | 31 | 105 | 29 | 517/584 | 573/584 |
| Historical 663 | Selected PR35 | 15 | 80 | 32 | 533/584 | 582/584 |

The primary R37 candidate returns 737 outputs: 644 matches plus 93 unmatched
outputs, of which 31 are duplicates and 62 are other unmatched outputs. All
583 fronts with two available corners are matched. Among the 76 partial fronts,
61 are matched; three misses have zero available corners and twelve have one.
Seven misses have best output IoU below 0.10; eight have a returned overlapping
box below the fixed 0.50 matching gate. These geometric groups identify review
priorities, not definitive physical causes. All three empty scenes return zero
outputs in both arms.

The confidence-ordered scorer is unchanged. An independent maximum-cardinality
check on its same-class IoU-eligible graph finds zero additional possible
matches for all five checkpoint records, all 20 images, and every separately
scored reference. Greedy assignment is therefore not hiding extra matches in
these results. An unmatched output is not automatically a hallucination, and
a duplicate prediction does not establish a duplicate reference annotation.

Two paired frozen-reference crops explain the small metric changes. For frozen
`new_000089:43`, best IoU changes from control p54 at 0.499002 to candidate p55 at
0.506051, crossing the fixed 0.50 gate. For frozen `new_000010:20`, matched p15's
right-corner error changes from 4.626013 to 5.324199 pixels, crossing five pixels
while staying below ten. Neither case establishes a robust improvement on new
acquisition sessions. Prediction IDs belong to their named cache and arm.

## Separate R38 reference-only replay

R38 manifest SHA256:
`0921081930f49340ff34290d5752be2d669f084f4d1518e2019060a5600c43b5`.
R38 has 654 fronts, 1,220 available and 88 unavailable endpoints, and the same
583 two-corner pairs. Stable R37 row IDs and all subsequent row shifts are
validated through the exact row map and operations ledger.

The five exclusions are old R37 `new_000009:52`, `new_000010:30`,
`new_000053:1`, `new_000086:57`, and `new_000088:57`. They are user-directed
reference exclusions, not confirmed duplicate annotations. Four were candidate
misses; `new_000010:30` was matched to candidate p29. Holding every output fixed,
the candidate therefore changes from 644 to 643 matches, 15 to 11 misses, and
93 to 94 unmatched outputs. No retained R37 row gains or loses a candidate
match. Deleting the four missed references is not detector recovery.

Three retained rows are corrected. `new_000007:47` changes top 388 → 385;
`new_000008:7` changes bottom 33 → 23; `new_000010:29` changes top
679.9117968 → 666 and left point (320.9100032,717.2699976) → (325,714).
For the unchanged candidate p28 on `new_000010:29`, left-point error changes
5.131271 → 0.500253 pixels because the reference moves. This removes one corner
error row (19 → 18); the full-pair numerators and denominator stay unchanged.

Two existing unmatched outputs change category from duplicate to other
unmatched: `new_000007` p59 and `new_000010` p34. The candidate's duplicate count
therefore changes 31 → 29 while other unmatched changes 62 → 65. No output was
suppressed. The PR35 replay has a separate retained-row gate crossing at
`new_000008:7` (p36 IoU 0.659023 → 0.459319); its miss total changes 43 → 40,
not by a simple count of all excluded rows. `r38-reference-delta.json` records
the complete row-level attribution for all five checkpoints.

## Input context and cropped review panels

The model already receives full 1280×720 source frames containing rack and
neighboring-object context. The actual training receipts sample tensors of
shape `[4,3,736,1280]` after rectangular preprocessing. Review-PDF crops are a
display choice and are not the network's training input. The audit at
`results/review/front-review-20261126-r38/input-context-audit.json` binds source
routing, receipts, batch shapes, and augmentation settings. Geometric
augmentation can resample or clip context; the evidence does not claim every
original background pixel always survives or that the model explicitly learns
rack geometry.

## Reproduce the cached scores and PDF

The default scoring replay requires only the Python standard library and the
committed JSONL/manifest/TXT inputs. It does not load images, Torch, model
weights, or network resources:

```bash
python3 -S scripts/score_pr123_review.py
```

The importable API is
`replay(root, prediction_paths, reference_manifests)`, returning
`{'comparison': ..., 'reference_delta': ...}` without writing files.
`default_prediction_paths()` supplies current reference/candidate, two PR121
caches, and PR35. `REFERENCES` contains frozen666, historical663, R36, and R37;
add the explicit R38 manifest to request its reference replay. The CLI includes
R38 when its finalized manifest is present. Each input path must resolve inside
the repository root. The scorer records cache, manifest, and helper-code hashes
and verifies the recorded label, image-identity, operations, and row-map hashes.
The pinned replay notebook additionally checks these inputs against its frozen
contract before executing the scorer.

The full PDF also requires the verified original PNGs, Pillow, and ReportLab:

```bash
python3 scripts/render_pr123_failures.py --output /tmp/pr123-r37-misses-and-errors.pdf
```

The 42-page PDF has 36 preserved-R37 pages followed by paired threshold,
full-frame context, separate R38 reference-change, and provenance appendices.
Its primary inventory covers all 15 misses, 19 matched corner-error rows, and
93 unmatched outputs exactly once per category. `pdf-inventory.json` records
source-image and PDF hashes, page locations, original row and prediction IDs,
crop bounds, and appendix lineage. `pdf-qa.json` records full-page rendering,
text-bound checks, inventory checks, and visual review. Original images,
trained weights, and generated PDFs are not duplicated into this Git directory.
