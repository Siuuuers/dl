# R31 partial-front box-only loss

Follow-up to [PR109](https://github.com/Siuuuers/dl/pull/109) and R30 review. Use the [standalone Colab notebook](../notebooks/baomai_partial_box_loss_r31.ipynb). Select L4 or A100, run all cells, preserve Drive output, and return `baomai-partial-box-loss-r31-results.zip`.

Both roles use the same eligible R31 annotations, independent epoch-173 starts, ordinary sampling, seed 46, 30 epochs, batch 4, imgsz 1280, FP32 and AdamW. Candidate weights only assigned partial-target native CIoU and normalized-L1 box terms by 2. Classification and pose objectives remain native. See [protocol](../experiments/partial-box-loss-2026-11-24-r31.json) and [row review](../results/review/front-review-20261124-r31/row-review.json).

R31 review: 13 amended rows, 6 retained; one amended training row, 12 development rows. No changes to splits, identities or visibility. The frozen 666-front primary and historical 663-front reference are unchanged; PR35 remains selected. PR109 R30 review: 16 to 11 misses, 91 to 110 unmatched outputs, 571 to 570 two-corner successes within 5 px. Do not treat reference-only rescoring as learning. No new GPU run or promotion is claimed.

The notebook preserves the exact prepared bytes (SHA-256 `3487253855abe19b4199e420dd7e9717caa7ada4198f8dfd256595d8809ca047`) and reconstructs the fixed local source commit `0a8ff31cadaf0ee0691de73fd103bba5de82b038` from the published R30 parent. The separate crop PDFs remain in the conversation; no original image pixels or weights are added here.
