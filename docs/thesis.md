# Final Year Project I workspace

Start with the **[thesis folder](thesis/README.md)** and **[current report](thesis/report.md)**. The report explains the task for undergraduate Electronic Engineering readers, compares with the original supplied baomai model, and cites external academic or professional sources. Acknowledgements are omitted.

The compact **[annotation definition](annotation-definition.md)** explains visible-front ownership, box extent, actual lower corners and unresolved cases. Its [research basis](thesis/research-annotation-definition.md) separates established rules from the proposed repeatability measurement.

The next GPU runs are prepared in this order:

1. **[Model improvement with Quality Focal Loss](thesis/quality-focal-experiment.md)** — change the classification objective while keeping data and the rest of training fixed.
2. **[Controlled training-label comparison](thesis/training-label-comparison.md)** — keep the original 137 photographs and training settings fixed while changing only the training labels.

The guides link to their notebooks and completed preparation checks. GPU accuracy results remain pending.

The [PR155 residual correction result](thesis/residual-results.md) is complete and audited: 564/583 reviewed complete pairs within five pixels and 582/583 within ten. It remains below the proposed two-view method's 569/583 and 583/583. Its slight mean-error improvement over the unrefined control is retained alongside the five-pixel regression.

The [annotation-reference experiment](thesis/annotation-comparison.md) adds the untouched supplied 664-front labels, which differ from the earlier fixed 666-front labels. With the same saved predictions, five-pixel success changes from 521 to 514/584 under supplied labels, 534 to 525/584 under earlier fixed labels, and 555 to 569/583 under reviewed labels. This measures evaluation-reference sensitivity, not the isolated effect of training-label changes.

The **[completed CPU notebook](../notebooks/baomai_annotation_reference_r60.ipynb)** reproduces all six method/reference combinations. All four actual code cells have already completed from published source. No model inference, training or annotation editing is required.

The Word equations now use centered native display math with separate right-aligned numbers and automatic height. Exact presentation counts and checks are in the [validation record](thesis/report-validation-fyp-i.json).

Outputs are used automatically, so missed fronts, extra predictions, duplicates and coordinate accuracy remain separate measurements. Current evidence comes from repeatedly examined development photographs. The selected deployment model and live defaults remain unchanged. The [source map](thesis/report-source-map.md) keeps technical provenance separate from the report.
