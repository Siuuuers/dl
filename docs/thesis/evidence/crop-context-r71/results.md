# Fixed crop context and scale comparison

The frozen reviewed-label model was run on the same 20 development photographs. The fresh full-image control passed the saved CPU matching and corner-outcome checks.

Each candidate adds the same two overlapping full-height crops. The first keeps the original image scale. The second enlarges each crop by 5/3. Artificial-edge boxes are rejected before whole-record suppression.

The first candidate changes context and adds candidate exposure. The second adds enlargement to the same crop geometry.

| Reference | Method | Missed | Extras | Duplicates | Both corners within 5 px | Both within 10 px | Empty-scene outputs |
|---|---|---:|---:|---:|---:|---:|---:|
| r50 | baseline | 15 | 51 | 14 | 572/583 | 583/583 | 0 |
| r67 | baseline | 18 | 50 | 14 | 572/583 | 583/583 | 0 |
| supplied664 | baseline | 50 | 76 | 12 | 513/584 | 566/584 | 0 |
| frozen666 | baseline | 38 | 62 | 14 | 523/584 | 575/584 | 0 |
| r50 | crop_context_768 | 14 | 52 | 11 | 572/583 | 581/583 | 0 |
| r67 | crop_context_768 | 17 | 51 | 11 | 572/583 | 581/583 | 0 |
| supplied664 | crop_context_768 | 49 | 77 | 11 | 507/584 | 563/584 | 0 |
| frozen666 | crop_context_768 | 37 | 63 | 13 | 517/584 | 572/584 | 0 |
| r50 | crop_context_scale_1280 | 14 | 107 | 44 | 566/583 | 581/583 | 0 |
| r67 | crop_context_scale_1280 | 17 | 106 | 44 | 566/583 | 581/583 | 0 |
| supplied664 | crop_context_scale_1280 | 46 | 129 | 38 | 506/584 | 565/584 | 0 |
| frozen666 | crop_context_scale_1280 | 34 | 115 | 40 | 516/584 | 574/584 | 0 |

## Paired changes under r67

- crop_context_768: 1 recovered fronts and 0 newly missed fronts. Five-pixel pair gains/losses: 1/1. Ten-pixel pair gains/losses: 0/2.
- crop_context_scale_1280: 2 recovered fronts and 1 newly missed fronts. Five-pixel pair gains/losses: 4/10. Ten-pixel pair gains/losses: 0/2.

The package contains every native crop record, pre-clamp corner coordinate, boundary decision, mapped point, pooled source, suppression decision and paired target outcome. All four references use fresh matching. This comparison does not train a model or select a deployment default.
