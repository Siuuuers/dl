# Figure captions

The concise current report uses only Figure 1. Figures 2–4 and their measured data remain historical assets from the earlier report presentation.

## Figure 1 — Cassette front and lower corners

The original photograph and an enlarged cassette front. The blue rectangle marks the labelled front box; orange points mark the two labelled lower corners. The enlargement uses the same source pixels.

File: `figure-1-target.png`

## Figure 2 — Effect of masking part of the cassette front

Corner-pair accuracy of the reviewed-label model for the same 30 fronts in four photographs. A successful pair has both corner errors within the stated distance from their annotated positions in the intact photograph. All distances are measured in original-image pixels. Centre masking also reduces five-pixel success, although the two target-corner neighbourhoods remain uncovered.

File: `figure-2-occlusion.png`

## Figure 3 — Gains and losses after changing the image input

Changes in successful corner pairs after contrast enhancement or whole-image enlargement. A pair succeeds when both corner errors are at most five original-image pixels. Counts use the same 584 fronts with two annotated corners in the fixed reference. Contrast enhancement gained 7 successes and lost 12; the larger input gained 17 and lost 25.

File: `figure-3-input-changes.png`

## Figure 4 — Corner-score cutoff and accepted-pair quality

The same model predictions evaluated against the fixed and reviewed references at different corner-score cutoffs. A confirmed pair belongs to a matched front with two annotated corners and has both corner errors within five original-image pixels. The horizontal axis divides confirmed pairs by all fronts with two annotated corners; the vertical axis divides confirmed pairs by all accepted outputs. An accepted pair has two finite coordinates inside the original image and both corner scores at or above the cutoff. The model's box score must also exceed 0.20. Labels show the tested cutoffs; lines connect the measured points. At 0.99, no pairs are accepted, so the vertical ratio is undefined.

File: `figure-4-pair-tradeoff.png`

# Reproduction and verification notes

These notes identify the experiment records used to build the figures. They are source documentation and are not part of the report captions.

- `figure-data-r57.json` retains every measured value, annotated target coordinate, case identifier and source hash from `../figures/figure-data-r56.json`. Only its schema and presentation labels change, with two additional provenance fields recording the original metrics hash and the purpose of the revision.
- The measured-source mappings are unchanged: Figure 1 uses the reviewed-reference record for source image `new_000007.png`, row 28; Figure 2 uses the earlier PR35 annotation-study candidate; Figure 3 uses the two R53 input comparisons; Figure 4 uses the R51 weighted-training candidate and the recorded strict box-score cutoff greater than 0.20. Figure 2 and Figure 4 therefore do not depict the later R53 input-test baseline.
- Source image SHA-256: `bd0bbad3d205341c2bf35c2ff61ccce00761602c3580f74ab20a048ef29f7147`.
- Original metrics SHA-256: `767dd7ab4b3f5aace902cf9c580e68c1d695244e648a59c52025e78e5a4e953c`.
- `figure-render-manifest.json` records the script, metrics and output-image hashes. Images are 1950 pixels wide, corresponding to 6.5 inches at 300 dots per inch.
- All four images were visually checked. The long contrast-enhancement label fits within the canvas; Figure 4's two-line horizontal label and cutoff annotations are legible without overlap.
- Figure 4 is a development-set diagnostic of agreement with annotations. Its fractions are not a calibrated probability of physical correctness for automated use.

From the repository root, build the figures with a local copy of the original photograph:

```bash
python scripts/build_thesis_figures_r57.py \
  --original-image /path/to/new_000007.png
```

The figure builder checks the photograph's SHA-256 and dimensions before rendering. No model training or inference is required.
