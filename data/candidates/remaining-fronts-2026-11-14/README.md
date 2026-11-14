# Remaining-original annotation candidate

This proposed revision extends the exact [border-fronts parent](../border-fronts-2026-11-13/README.md) with accepted findings from the remaining 57-original review. **PR35 remains selected; this candidate has not been trained.**

[Whole-original review report](../../../docs/training-review-2026-11-14.md) · [Review evidence and adjudication](../../../results/review/remaining-training-20261114/README.md)

## Accepted changes

Coordinates use original 1280×720 pixels; box order is `xmin,ymin,xmax,ymax`. Rows refer to original source identities. The six additions append after existing rows, preserving their identity and bytes.

| Image | Operation | Accepted geometry |
| --- | --- | --- |
|`new_000029`|Append rows 43–46|Boxes `[683,470,726,499]`,`[699,491,719,518]`,`[700,513,715,537]`,`[702,534,716,549]`; all eight endpoint slots missing.|
|`new_000121`|Append row 57|Box `[568,508,605,551]`; both endpoints missing.|
|`new_000129`|Append row 50|Box `[700,516,729,541]`; both endpoints missing; approximately 4px upper-bound uncertainty.|
|`new_000033`|Replace rows 41/42 left slots only|Observed left points `(613,453,v2)`/`(617,477,v2)`; right slots remain missing; approximately 2px placement uncertainty.|
|`new_000037`|Replace row 11 bbox tokens|Only `ymax→471`; other bounds and endpoints preserved.|
|`new_000040`|Replace row 14 bbox tokens|Only `xmax→655`; other bounds and endpoints preserved.|
|`old_000042`|Replace row 10 bbox tokens|Box `[575,294,873,327]`; both visible endpoint token triplets preserved.|

The source rows and explicit operations are bound in [manifest.json](manifest.json); accepted physical reasons, uncertainty and review dependence are in [review.json](review.json). The exact new_000033 coordinate confirmation followed exposure to root suggestions and is not a blind independent mark. Approximate manual uncertainty is not a calibrated confidence interval.

The `new_000015` top-strip additions were withdrawn; the `new_000038` extent question and old-image top-bound proposals remain held. No hidden corner is inferred from rack spacing, another image, a model prediction or an occluder edge. All twelve slots on the six added rows are exactly `0 0 0`.

## Composition and counts

All ten parent label files are byte-identical; all ten inherited change records and fifteen accepted review decisions are preserved semantically exactly. Seven new complete label files produce a seventeen-file overlay. No other historical annotation candidate is silently combined.

| Effective dataset | Images | Objects | Visible endpoints | Missing slots | Empty images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Border-fronts parent training |137|3,773|6,844|702|13|
| Proposed training revision |137|3,779|6,846|712|13|
| Original training-validation split |20|664|1,227|101|3|

The separate 663-object development reference is unchanged. Documented review membership is not certification of every original box, point or object identity. The new front rows increase missing-slot counts because their existence is visible while their actual corners are obscured.

## Verify

From the repository root:

```bash
python scripts/verify_pose_annotation_candidate.py \
  --candidate data/candidates/remaining-fronts-2026-11-14/manifest.json
```

The existing unmodified verifier checks all 157 original label identities, declared edits, row maps, missing-slot representation, geometry, retained bytes and effective counts. It does not adjudicate physical image content. Source revision is `1b71e8158e0594ee4c72ba845fa7b514a78b9e0d`.

This is reviewed supervision for a future defined comparison. It introduces no training recipe, GPU run, reference relabeling, inference-default change or model-improvement claim.
