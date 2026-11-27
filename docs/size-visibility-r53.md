# R53: is target size enough to explain the misses?

27 November 2026. Descriptive CPU analysis of verified PR146 saved predictions; no new model run.

The R50 reference contains 654 fronts on the same 20 development photographs. The two arms have 15 and 11 misses after the existing postclip step. Every per-image assignment and aggregate score was recomputed and checked against the return.

## Corner availability

| Locatable lower-front corners | Targets | Reference misses | Candidate misses |
| --- | ---: | ---: | ---: |
| 0 | 17 | 3 | 2 |
| 1 | 54 | 12 | 9 |
| 2 | 583 | 0 | 0 |

Corner availability is not a measured occlusion fraction. It is nevertheless necessary to stratify it: a size-only explanation could hide the fact that all misses occur in fronts with unavailable corners.

## Visible-front box height

These fixed descriptive bands use original-image pixels, not input-tensor pixels or COCO area categories. They were selected after previous outcomes were known; do not treat the table as a preregistered causal test.

| Height (px) | Targets | Reference misses | Candidate misses |
| --- | ---: | ---: | ---: |
| <8 | 0 | 0 | 0 |
| 8–<16 | 14 | 5 | 5 |
| 16–<32 | 416 | 7 | 4 |
| >=32 | 224 | 3 | 2 |

### Cross the bands with corner availability

| Corners | Height (px) | Targets | Reference misses | Candidate misses |
| ---: | --- | ---: | ---: | ---: |
| 0 | <8 | 0 | 0 | 0 |
| 0 | 8–<16 | 2 | 1 | 1 |
| 0 | 16–<32 | 12 | 2 | 1 |
| 0 | >=32 | 3 | 0 | 0 |
| 1 | <8 | 0 | 0 | 0 |
| 1 | 8–<16 | 6 | 4 | 4 |
| 1 | 16–<32 | 32 | 5 | 3 |
| 1 | >=32 | 16 | 3 | 2 |
| 2 | <8 | 0 | 0 | 0 |
| 2 | 8–<16 | 6 | 0 | 0 |
| 2 | 16–<32 | 372 | 0 | 0 |
| 2 | >=32 | 205 | 0 | 0 |

## What this does and does not suggest

A single increase from inference size 1280 to 1600 is a reasonable diagnostic on the frozen PR137 checkpoint. It tests whether a changed sampling scale helps these specific instances. It does not prove that size caused the misses, reveal a hidden corner, or add information that was absent in the capture. Compare exact recovered and newly missed rows, unmatched outputs, corner error in original pixels, latency and memory. Keep the CLAHE contrast test separate.

The current data already make corner availability an essential reporting stratum. Do not select a resolution solely from the pooled recall or call 654 related targets 654 independent trials. New capture sessions remain necessary for external validation.

## Reproduce

```bash
python scripts/analyze_size_visibility_r53.py \
  --archive /path/to/baomai-partial-classification-r51-results.zip \
  --output /tmp/r53-size-visibility.json \
  --markdown /tmp/r53-size-visibility.md
```

The script verifies the exact archived bytes, all four reference manifests and label hashes, then replays R50 assignments. It makes no network calls. The full JSON retains every target's geometry and both match identities.

[Machine-readable analysis](../results/review/thesis-r53-20261127/size-visibility.json) · [Prior result audit](partial-classification-r51-results.md) · [Thesis](thesis.md)
