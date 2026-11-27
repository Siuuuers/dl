# Corner quality diagnostic protocol

28 November 2026, Asia/Hong_Kong.

This exploratory CPU replay uses saved original, contrast-enhanced, and enlarged-image predictions. It does not run a model, fit a predictor, choose a score threshold, change labels, or promote a model.

The ordinary evaluation uses the fixed original scorer at box confidence greater than 0.13, matching intersection over union at least 0.50, and endpoint confidence at least 0.50. All four annotation references are retained separately.

For every reference front with two locatable corners, partition the unchanged-view result into: missed box; matched box with an endpoint failing finite-coordinate, in-bounds or score checks; two returned endpoints with worst error above 5 pixels; or both endpoints within 5 pixels. Denominators retain all eligible fronts.

For matched fronts with both endpoint proposals passing the fixed checks, compare ranking signals against worst-corner error. The fixed signals are box score, minimum endpoint score, and reference box overlap. Reference overlap is available only during evaluation and is not a deployable input. Use Spearman rank correlation and the area under the receiver operating characteristic curve for identifying error above 5 pixels, with ties worth one half. Do not fit a threshold or calculate a statistical significance claim from these correlated scenes.

To test a reference-free agreement feature, associate each original prediction with each alternative view using mutual-best same-class predicted-box overlap at least 0.50. The association does not consult reference boxes or points. The agreement signal is the maximum corresponding-corner displacement across the two alternative views. Only rows with usable point pairs and both associations have this signal. Report the resulting subset size and recompute all ranking measures on that same subset. Report each alternative-view association's consistency with its independently computed reference assignment; an unmatched assignment is not automatically a physically wrong association.

These fixed definitions are written before executing the diagnostic. The work is post hoc because the underlying experiments and their headline results have already been seen. The same development photographs are reused, and reference dependence must remain visible.

## Follow-up diagnostic amendments

After the initial pooled ranking was calculated, failures were observed to cluster by image. The same features were therefore also evaluated using only within-image failing-successful pairs, reporting both comparison-count-weighted and image-macro AUROC. This is an exploratory robustness check; no threshold or feature sign was fitted.

The view association was prediction-only from the start. A follow-up audit records the number of forward/reverse candidates meeting the same fixed overlap threshold, exact maximum-overlap ties, missing associations, unavailable alternative pairs, and reference-assignment agreement after association. Ambiguous candidates are retained in the common cohort.

A crop feasibility calculation was added with factor 1.5 fixed before running it. Predicted boxes expand about their centre, minima are floored, maxima are ceiled, and bounds are image-clipped. The unchanged control and weighted candidate are analyzed separately under the same native >0.13 box cutoff. Both complete-reference misses and matched unavailable pairs remain in overall accounting. Crop containment is not a model-accuracy experiment.

The main analysis replays all eight saved baselines. Independently written rank-sum calculations verify all 28 pooled pairwise-AUROC entries. No GPU inference, training, annotation edit, model edit, or threshold selection was performed.
