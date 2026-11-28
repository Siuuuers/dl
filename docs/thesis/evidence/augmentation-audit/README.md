# Augmentation-audit evidence

The [protocol](../../augmentation-audit.md) and [four-cell CPU notebook](../../../../notebooks/baomai_augmentation_audit_r63.ipynb) inspect native training transformations without training or inference.

All four actual notebook code cells completed unchanged in one local CPU IPython session on 2026-11-28. The fresh source checkout was pinned to `87b3f1dfd547bf46fef35b171aebe16d832bd1be`; it remained clean after execution. The [execution receipt](notebook-execution.json) records cell identities, logs, runtime versions, package membership, source hashes and independent output verification. [The executed notebook](executed-notebook.ipynb) preserves its source and text output. Its six image outputs are linked to their saved panel files with verified hashes, avoiding duplicate image bytes. The user notebook itself displays the panels inline when run.

## Result

Across 426 image presentations, native augmentation retained 11,880 of 11,997 source-object presentations and removed 117. Of the removed boxes, 108 had zero clipped area and nine retained thin edge slivers, with maximum short side 4.05 pixels. Of the retained originally complete objects, 79 became one-corner and 22 became zero-corner targets. No source-unavailable corner became known; all 39 empty-image presentations stayed empty.

The [six-panel visual review](visual-review.json) found no concrete inconsistent supervision in its bounded sample. The current augmentation policy is retained. This audit does not train or score a model and does not show that a changed augmentation would improve recognition. Removed or clipped training rows are review candidates, not proven label errors or demonstrated causes of model failure.

## Saved evidence

- [Input/source plan](plan.json): frozen training membership, label hashes, recipe and published source closure.
- [Readable result](results/results.md) and [numeric summary](results/summary.json).
- [Compressed row lineage](results/details.json.gz): every source row, native filter decision, transformed box/point flags, 426 image records and 142 unaugmented controls.
- [Return manifest](return-manifest.json): exact hashes and sizes for the plan and all result members.
- [Cell 1](notebook-cell-1.txt), [cell 2](notebook-cell-2.txt), [cell 3](notebook-cell-3.txt), [cell 4](notebook-cell-4.txt) logs.

The 4,960,176-byte return ZIP was verified independently (SHA256 `e225bcd570fde8f7cc8f12551799a46200cc4bda6b7ccc2933d69823a63b5aac`). Its members are stored once here; the ZIP is not duplicated in Git. Hosted Colab and the Colab browser-download interface were not exercised. The non-Colab download link completed successfully.
