# R43 — one frame per review page, with the latest corner decisions

## Open the review

Download [review.html](../results/review/front-mining-r43/review.html) and open it in a browser. It is self-contained, including the original photos, and works offline. The [review ZIP](../results/review/front-mining-r43/review.zip) also contains the exact labels, amendment ledger, previous evidence and verifier.

The optional [CPU notebook](../notebooks/baomai_front_review_r43.ipynb) downloads the pinned package, checks it, displays the same page and offers the ZIP. No GPU run is needed to view these amendments.

### What changed in the page

- **One frame per page.** All its rows appear together. Previous/Next at the top and bottom moves between frames.
- **All rows is the default.** Unchanged rows remain beside amended rows. A row jump scrolls to an existing card; it does not paginate rows.
- Every card has **Original pixels / Working baseline / Previous R42 / Latest R43** panels. Its expandable history retains the upstream row text, earlier decision, exact values and unresolved fields.
- Magenta in R42 means a change from the working baseline. Magenta in R43 means a change since R42, so an older correction is not presented as a new one.
- The whole-frame overlay can show upstream labels, the working baseline, R42 or R43. It can also be switched off.
- The five earlier prediction-comparison figures remain under their respective frames. They are explicitly **historical R42 diagnostics**, not scores recalculated against R43 labels.

The page retains **508 stable identities across 12 frames**: 507 current fronts plus the earlier deleted identity. In `frame_0128`, source row 43 remains a deletion record, and the earlier added front retains its separate identity. A new correction does not renumber or erase another front.

## PR129 return

[PR129](https://github.com/Siuuuers/dl/pull/129), head `3644cd90434d755da9ace3c77cbf193bf0e3ba87`, returned the existing R42 CPU review package. Its ZIP is 4,181,563 bytes, SHA256 `716f81f9b72fffccfb52120d90b30a50b3d605ebd649736ed1f5ad189b4cb2ef`. It is byte-identical to the R42 publication, with all 65 members verified. It supplies the previous review state, rather than new model predictions.

R43 checks the ten requested source rows against their original JPEGs. Each received independent assistant visual review and a further original-pixel cross-check. Full-frame completeness and the other rows are inherited from R42; this targeted pass does not claim a new exhaustive review of all 507 fronts.

## Requested corner and boundary decisions

Coordinates below are nominal **original-image pixels**, `(x, y)`, on 1280 × 720 images. Stored normalized values retain eight decimal places for reproducibility, not because the blurred physical edge is known to subpixel precision.

| Frame / immutable source row | Field | R43 decision | Physical reason |
|---|---|---|---|
| `frame_0001` / 1 | Left point | `(587, 207)` | Move from the dark vertical strip toward the visible lower-front/side turn. |
| `frame_0001` / 36 | Left point | `(701, 598)` | Small refinement from an inset lower-edge position to the rounded turn. Necessity is weaker than the larger corrections; the old point is not a proven hard error. Existing `ymin` hold remains. |
| `frame_0146` / 1 | Left point | `(574, 201)` | Move up/right from the shaded seam to the owned front junction. |
| `frame_0160` / 24 | Left point | `(418, 465)` | Move up/slightly right from the diffuse lower seam to the visible turn. |
| `frame_0160` / 31 | Left point | **Retain `(408.8658944, 563.9668992)`** | The considered `(410, 563)` point and the existing point occupy the same blurred, shaded turn. Pixels do not establish that the 1.49-pixel change is better. |
| `frame_0001` / 43 | Right point | `(856, 657)` | Place the point at the lower-front turn before the continuing rack strip. Hidden left point remains exactly `0 0 0`. |
| `frame_0001` / 43 | Right box boundary | `xmax ≈ 863.281 → 859` | Remove the extra outer rack strip while retaining the owned upper rim, which extends farther right than the lower corner. Existing `xmin` hold remains. |
| `frame_0160` / 1 | Right point | `(538, 301)` | Move left from the rack-side strip to the physical turn. |
| `frame_0160` / 26 | Right point | `(661, 475)` | Follow the lower rim outward to its rounded right-side junction. This is a modest nominal refinement. |
| `frame_0160` / 37 | Right point | `(530, 613)` | Move left/up from the dark rack gap to the owned front junction. Existing `ymin` hold remains. |
| `frame_0383` / 10 | Right point | `(644, 303)` | Small inward refinement to the shaded outer turn, retaining its molded rim rather than snapping to the bright stripe. |

This means **9 amended rows, 1 reviewed-and-retained row, 9 changed corner slots, and 1 changed box boundary**. Twenty serialized coordinate tokens change; no visibility flags, identities or current front counts change. All 498 unrequested row records are preserved exactly. See the [amendment ledger](../data/reviews/additional-fronts-2026-11-26-r43/amendments.json) for before/after text, nominal support ranges and reasons.

### The right border is `xmax`

Image x increases to the right: `xmin` is the left border and `xmax` is the right border. For row 43, “move the right borderline left” is implemented as decreasing **`xmax`**. Moving `xmin` left would expand the box on the opposite side. The original left bound is preserved to serialization precision; its separate unresolved material boundary is not silently approved.

## How the prior guidance locates a corner

The [labeling guide](front-labeling-guide.md), [R33 guidance published in PR115](https://github.com/Siuuuers/dl/blob/d4631e07af44efc3c35d84f57f55c10c6ff06a65/docs/annotation-boundaries-r33.md), and [R36 guidance published in PR120](https://github.com/Siuuuers/dl/blob/a0db36122c4c03a7fa3a9115525113dc2a21636b/docs/annotation-boundaries-r36.md) support the same physical target:

1. Identify one cassette's visible front and its own molded frontal rim. Separate it from the broad upper plane, the neighboring front, rack and gripper.
2. Follow the actual lower-front edge and the side until they turn into one another. A rounded edge has a small observable interval; it need not contain one uniquely measurable mathematical point.
3. Use contour continuity, plane changes and occlusion junctions to establish ownership. A dark strip that continues vertically through gaps and into the base is evidence of rack structure. A short shaded rim that continues into the same front can belong to the cassette.
4. Use color as supporting evidence. Shaded plastic can be gray; metal can be bright. A global white/gray threshold, brightest-stripe rule or fixed inward offset cannot establish the corner.
5. Judge the box and corner separately. An owned upper rim may determine `xmax` beyond the lower corner. Do not automatically snap a corner to the box corner, or a box boundary to the point.
6. A visible, locatable corner uses `v=2`. A hidden or unlocatable physical corner remains `0 0 0`; neighboring spacing, symmetry and model predictions do not supply its annotation coordinate.

**Rack pixels can fall inside a correct tight axis-aligned box. Rack material must not determine its outer extent.** A slanted face or visible owned patches around an occluder can require a rectangle containing other pixels. Removing all interior rack/gripper pixels by shrinking the rectangle would cut off real front material.

The R33 `new_000088` row 43 example retains an owned upper strip despite jaw pixels inside its box. The R36 `new_000010` row 24 example includes the front's pale rim through about x912, ending the box at x913 before the continuing rack separator. Both concern material ownership, rather than a blanket “exclude every gray pixel” rule.

For R43 `frame_0001` row 43, the owned upper rim supports nominal `xmax=859`; the lower corner is around `(856,657)`. Those two different x coordinates are consistent with the same guidance.

## What `frame_*` means

| Collection | Original filenames | Size | Role here |
|---|---|---:|---|
| Frozen baomai working images | 126 `new_*` and 31 `old_*` PNGs | 157 | Existing 137/20 train/development split and prior project work. |
| Additional upstream `grasp-vision` pool | `frame_*.jpg` | 320 | Existing photos under `box_in_basket/white_box_pose`; selected cases are being reviewed for our visible-front policy. |
| Current review | 12 selected `frame_*` originals | 12 | Three earlier pilot revisits and nine disagreement-mining cases, with their source identities retained. |

The stems already exist in the [pinned upstream manifest](https://github.com/an94-doll/embedding_vision/blob/9a6bf2432582af517d789c94eab2ac27e03a304b/box_in_basket/white_box_pose/manifest.json). They were not created by renaming `new_*`/`old_*`. The numbers are source identifiers with gaps; `frame_0383` does not mean there are 383 photos in this 320-image package.

**We have the actual photos.** All 12 original JPEGs are embedded in the HTML and included under `originals/` in the ZIP, with their hashes verified. An exported PNG showing the same decoded image is not another capture.

The latest exact-overlap check covers all 320 pool images against all 157 frozen originals: zero identical file hashes and zero identical decoded RGB images. This does not rule out cropped/resized/lossy variants, adjacent captures or shared sessions. The selected review cases are development material; no independent test status is assigned.

The 2026-11-26 upstream check still found `baomai` at `351eb3c`, `grasp-vision` at `9a6bf24`, and `yolo-pose` at `72a99da`. These match the earlier pins; no new branch-head content was found. The package's October 28 timestamp is an export timestamp, not a verified capture date. [Provenance evidence](../data/reviews/additional-fronts-2026-11-26-r43/provenance.json).

## Status and verification

The three complete assistant-reviewed candidates remain **`frame_0146`, `frame_0371`, `frame_0383`**, totaling 129 fronts. Nine whole frames remain held, with the same 55 held source rows. Correcting a point on a held frame does not resolve a different uncertain boundary or make the whole frame eligible for export.

No training split has been assigned to these candidates. This round creates no inference, training or model promotion, and does not establish that unmatched predictions improved. Its purpose is a clearer, cumulative review and a small set of physically supported annotation decisions.

The builder verifies the PR129 archive identity, reconstructs and checks all 65 R42 members, replays only the ten requested decisions, validates geometry and unchanged holds, checks the candidate-export set, and binds the HTML to the exact current and previous manifests. See [verification](front-review-r43-verification.json).

The viewer was exercised through DOM interaction and real image/canvas decoding across all 12 frames. Browser layout screenshots were not produced because the installed Chromium launcher has no browser executable. Desktop and narrow-screen layouts are defined as four and two columns respectively; browser/CSS visual verification remains the stated limit of the automated check.

### Rebuild

```bash
python scripts/build_front_review_r43.py \
  --return-archive /path/to/PR129-returned.zip \
  --amendments data/reviews/additional-fronts-2026-11-26-r43/amendments.json \
  --output results/review/front-mining-r43/review.zip \
  --manifest-output data/reviews/additional-fronts-2026-11-26-r43/manifest.json \
  --html-output results/review/front-mining-r43/review.html

python scripts/build_front_review_r43.py \
  --check-capsule results/review/front-mining-r43/review.zip
```

The capsule also carries its three Python source files so the same check works after extraction, with `python review.py --check-capsule /path/to/review.zip` and only the Python standard library.
