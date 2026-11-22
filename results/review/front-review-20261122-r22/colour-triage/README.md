# Colour-assisted inspection evidence

This cue ranks suspicious horizontal margins for manual inspection. It does not
identify material ownership or modify annotation geometry. The source is the
immutable r21 review; every original image and label is hash checked.

All 70 supplied photos in 001–080 have a whole-photo overlay and a manual scan.
The initial contact sheets show at most six flagged edges per image, balanced
by side. That display cap is not a review cap: the complete edge scores are in
`photo-scores.json`, and recurring rail patterns were traced beyond the cards.
The exact physical decisions are in the parent directory's `visual-audit.json`.

The cue compares brightness near each box edge with nearby interior material,
then weights local contrast by its vertical continuity and relative chroma.
It flagged 33 of the 37 user-named rack edges and none of four tested edges on
094:31–32. Those examples helped choose the threshold; this is a small
calibration check, not a general accuracy estimate. The four missed named
edges are recorded in `summary.json`.

Shaded own walls, notches and projecting rims often produce high scores.
Glare and blurred separators can produce low scores. Grey pixels inside an
axis-aligned box are not automatically a labeling error: a sloping visible
front does not fill every pixel of its enclosing rectangle.

## Reproduction

From a checkout with the original images hydrated:

```bash
python -B results/review/front-review-20261122-r22/colour-triage/reproduce.py --output /absolute/fresh/inspection-directory
```

This writes the full per-edge measurement JSON, whole-photo overlays and raw
versus saved-label contact sheets. Original pixels are only cropped or enlarged
by integer nearest-neighbour scaling for inspection. No annotation is written.
The archived score table and source hashes allow a reproducibility check without
storing all preview images in Git.
