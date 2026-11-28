# Quality-focal preparation evidence

These receipts verify the implementation and inputs before paired GPU training. They contain no new trained-candidate accuracy result.

- `loss-guard.json`: actual supplied weights differentiated on two verified training photographs and one synthetic empty scene. Exponent zero matches native losses and all 498 parameter-gradient tensors exactly. Exponent two preserves fixed-prediction assignments, soft targets and nonclassification gradients while its classification derivative agrees with an independent analytic calculation. Missing coordinates, empty targets, class weights and native branch scheduling are checked. No optimizer step is made.
- `cpu-loader-preflight.json`: both actual 142-photo training sets and their 20-photo development sets are staged and read by the native loader. Image and label identities, parsed targets, loader exposure and the matched budget are checked.
- `native-score-floor-check.json`: eight supplied-model CPU inference calls on one populated and one empty development scene, including reflected views, compare native score floors 0.001 and 0.25. Records retained above 0.25 and their common post-clipping suppression are identical in these selected checks. This is not a claim that all future candidate outputs have been checked.
- `local-validation.json`: source hashes and local verification scope. The source base and dirty-tree status identify a development check; the clean published notebook execution is recorded separately when completed.

The seven detection-ranking checks and three controlled-label/shared-scoring/package checks pass. The shared scorer replays all three actual annotation-reference results. Its multipart package test crosses the 16 MiB boundary, verifies exact reconstruction and rejects corruption.

The candidate changes the classification objective, including its aggregate scale. The experiment does not isolate focal weighting from a uniform classification-strength reduction. Both complete arms must start independently from the supplied checkpoint; the disposable GPU resource trial cannot supply their initial weights.

See the [experiment guide](../../quality-focal-experiment.md) for the scientific question and interpretation.
