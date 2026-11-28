# R56 thesis figures: captions and reproducibility

The figure numbering follows report appearance: target definition, simulation, R53 experiments, then the R55 automatic-pair diagnostic. All four PNGs are 6.5 inches wide at 300 dpi, with black labels on a white background. The plotted values are copied from measured result records; no fitted curves, simulated measurements, or uncertainty bars have been added. Figure 1 includes an original photograph and an enlarged display crop.

## Figure 1 — Cassette-front and corner target definition

**File:** `figure-1-target.png`

**Caption:** Figure 1. Cassette-front target definition in the original 1280 × 720 image `new_000007.png`. Panel (a) shows the complete captured scene; panel (b) enlarges the same pixels around R50 reference row 028. The blue rectangle is the annotated cassette-front bounding box, and the orange points are the two annotated actual lower-front corners, ordered by image x coordinate. Both reference points have visibility value 2. The connecting orange segment is a visual guide between those points. The inset is a display crop and was not used as a separate model input. This example illustrates annotation semantics; it contains no model predictions and was not selected as a successful detection result.

**Exact annotation:** box `(723.0, 116.20999998, 793.5000000000001, 135.99999997199998)`; lower-left point `(724.7900032, 135.2200032)`; lower-right point `(792.0700032000001, 131.9899968)`, in original-image pixels. The near-integer floating-point differences arise from the stored normalized-label conversion.

**Provenance:** the original image has SHA-256 `bd0bbad3d205341c2bf35c2ff61ccce00761602c3580f74ab20a048ef29f7147`. The annotation was extracted from the verified R54 input snapshot, SHA-256 `b6059a3810e29ba3afb245f1585704521c2633d8caca6fddd908be015e249125`. `figure-data-r56.json`, section `figure1`, stores the complete source annotation used by the renderer. No image enhancement is applied.

## Figure 2 — Controlled digital occlusion

**File:** `figure-2-occlusion.png`

**Caption:** Figure 2. PR35 corner-pair success under controlled digital occlusion of 30 fixed targets from four familiar development photographs. The intact source images provide the known original corner coordinates. Separate gray masks cover the left corner, right corner, or center of the front; the model receives the altered complete image. A pair is successful only when its object is matched and both predicted corners are within the stated Euclidean tolerance of the original reference coordinates. The denominator remains all 30 designated targets in every condition. The counts at 5/10 original-image pixels are 27/30 and 30/30 for intact inputs, 21/30 and 28/30 for left masks, 22/30 and 26/30 for right masks, and 22/30 and 30/30 for center masks. All 30 targets match in each condition. The center mask removes meaningful image content and is a location control, not a harmless sham. These fixed-cohort counts have no uncertainty bars; 30 targets and their altered copies do not constitute independent deployment scenes. Artificial masks, collateral occlusion, and repeated source images limit transfer to naturally hidden corners.

**Provenance:** measured experiment `pixel-occlusion-20261113-v1`, `results/verification/baomai-pixel-occlusion-20261113/summary.json`, field `target_conditions`; projection SHA-256 `f556e0663f8844ed65587a76c8b9e08006d795bf09fe9c699e98f2444b6c3b23`. The complete original replay archive `evaluation.zip` is 737,981 bytes with SHA-256 `f2dfbc58479ec48172137ceec5b07c5d74f5fb60428dd0c42bd12590d43a780e`. The [published protocol and runner](https://github.com/Siuuuers/dl/commit/3647d0937ab036974cda4867815a8fe20aef07d7) preceded masked inference. `figure-data-r56.json`, section `figure2`, retains the measured target counts. This figure summarizes an earlier completed diagnostic; it is not a new R56 experiment.

## Figure 3 — Paired R53 gains and losses

**File:** `figure-3-r53-changes.png`

**Caption:** Figure 3. Paired changes in both-corner success at 5 original-image pixels for the R53 CLAHE and larger-input experiments against the primary frozen666 reference. All 584 reference fronts with two eligible corners remain in the denominator, including missed objects. A gain is a reference front that fails the identity control but succeeds under the treatment; a loss is a control success that fails under the treatment. Relative to the 1280-pixel identity control, CLAHE gains seven and loses twelve successes, for a net change of −5 (511 to 506 successes). Scaling the inference input from 1280 to 1600 gains seventeen and loses twenty-five successes, for a net change of −8 (511 to 503). Bars extend left for losses and right for gains, with absolute counts labelled. The same checkpoint, images, matching rule, and postprocessing policy are used within each paired experiment. These observations come from the same development photographs and do not establish generalization to an independent test set. The figure reports pair localization; it does not by itself summarize every detection acceptance criterion.

**Provenance:** [PR149](https://github.com/Siuuuers/dl/pull/149) and [PR150](https://github.com/Siuuuers/dl/pull/150), returned `cases.json` files. Their SHA-256 hashes are respectively `4208adcb2fdfc9a0e0b59de37421fa09ff61675b1ebd03d28c5f57f32541d326` and `d13191aa79efbc5e3d71480e213180e7812cc3590dc5d8f37d8b764f41129397`. The return archives have SHA-256 `1e1dde4528729deced36a7943f1a3fc68e4e4993e05c0289e10fcf03f2cfa56d` and `b1638b1eaf5e3877a77efb840c342bd1dfe7783ca5951041ed37b692df59abca`. `figure-data-r56.json`, section `figure3`, contains all named gained/lost reference-row identities, not only aggregate counts.

## Figure 4 — Reference support and eligible-pair coverage

**File:** `figure-4-pair-tradeoff.png`

**Caption:** Figure 4. R55 post hoc reference-supported pair fraction versus eligible-pair recall at a tolerance of 5 original-image pixels for the candidate with box score strictly greater than 0.20. Each plotted point applies the same endpoint-score cutoff q to both corners, with finite, in-bounds coordinates required. An offered pair is reference-supported only if its prediction has a one-to-one box match under the fixed association rule, both reference corners are eligible, and both endpoint errors are at most 5 pixels. The vertical axis divides supported pairs by all offered pairs; the horizontal axis divides supported pairs by all eligible two-corner reference fronts (584 for frozen666, 583 for R50). Labels identify the tested q values 0.50, 0.70, 0.80, 0.90 and 0.95. Lines connect measured settings only and do not imply interpolation or a selected optimum. At q = 0.99 no pairs are offered, so the supported fraction is undefined and that setting is omitted. The two curves evaluate the same images and predictions against different annotation references. Their difference demonstrates reference dependence; neither curve is a calibrated estimate of physical automatic-use accuracy.

**Exact plotted counts:**

| q | Offered pairs | Frozen666 supported / 584 | R50 supported / 583 |
| --- | ---: | ---: | ---: |
| 0.50 | 709 | 511 | 567 |
| 0.70 | 678 | 511 | 567 |
| 0.80 | 650 | 511 | 567 |
| 0.90 | 514 | 424 | 472 |
| 0.95 | 116 | 92 | 111 |

**Provenance:** `r55-analysis/automatic-pair-results.json`, candidate `candidate_box0.2`, SHA-256 `52d51e955c44b4cad24368424cfd43b3302455381311959c13dbbab2d6ff45b8`; verified input snapshot SHA-256 `b6059a3810e29ba3afb245f1585704521c2633d8caca6fddd908be015e249125`. `figure-data-r56.json`, section `figure4`, stores all plotted numerators, denominators, fractions, and the q = 0.99 omission rule. These are the R55 candidate results and must not be presented as the R53 scale checkpoint's results.

## Portable build

From a repository checkout with Python, Matplotlib and Pillow installed:

```bash
python scripts/build_thesis_figures_r56.py \
  --original-image /path/to/new_000007.png
```

The original image must have the exact hash above. The script locates `docs/thesis/figures/figure-data-r56.json` relative to its own repository path, so the current working directory is immaterial. The optional `--metrics PATH` and `--output DIRECTORY` arguments select another metrics file and output folder. The renderer verifies the image hash, image dimensions, two visible reference corners, 1950-pixel output width, and 300-dpi metadata. It writes a manifest containing source-script, metrics, and output-image hashes. The plotted measurements are already in the small metrics JSON; regenerating the figures does not require model weights, network access, or another inference run.

The figures were built with the primary runtime. Where that runtime is available, the equivalent command is:

```bash
"$CODEX_PRIMARY_RUNTIME_PYTHON" scripts/build_thesis_figures_r56.py \
  --original-image /path/to/new_000007.png
```

All four rendered figures were inspected individually for text clipping, overlapping labels, data consistency, color clarity and figure order. Captions should accompany the images in the report; they supply the denominator, reference, and interpretation limits that a compact plot cannot show completely.
