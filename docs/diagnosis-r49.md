# R49 frozen-model proposal diagnosis and Soft-NMS pilot

Open [baomai_r49_diagnosis.ipynb](../notebooks/baomai_r49_diagnosis.ipynb) in Colab; select a GPU runtime and Run all (CPU is supported, but slower). Return `baomai-r49-diagnosis-results.zip`.

This is an exploratory development-set inference comparison, not training. The exact PR137 candidate checkpoint and its verified archived predictions are restored by the R48 audit. All 20 original development PNGs and the R38, frozen666 (primary), and historical663 reference labels are pinned and verified. These are three annotation versions on the same scenes, not three independent tests.

Trace the union of the 15 saved R38 misses and any fresh hard-NMS misses through decoded geometry, the strict confidence gate, ordinary NMS, postclip suppression, and assignment. Use original-pixel IoU >= .50 and independently available corner slots. Native NMS index reconstruction must exactly match the fresh instrumented predictor output. Saved/fresh differences are reported, not silently attributed to filtering.

The single predeclared intervention replaces ordinary hard NMS with linear Soft-NMS: at overlap > .70, decay remaining scores by (1-IoU), retain score > .13, cap 300 outputs, then use the same postclip filter. Both arms consume the same freshly decoded tensor and preserve the two keypoints attached to each proposal. Compare misses, unmatched, duplicate subset, both-corner 5/10px success, and paired recovered/newly missed rows. The output includes decoded tensors and exact proposal indices so the experiment is reviewable.

Validation before publication: all four notebook code cells compiled; the script's actual notebook reconstruction was executed locally; synthetic Soft-NMS and stage-classification checks passed. Fresh inference, archive retrieval and the complete Colab run were not executed in the authoring environment (GitHub binary/network access unavailable there). The notebook contains runtime and archive integrity checks and may stop if its assumptions do not hold. No model or inference default is changed. PR35 remains selected. No new images, weights, PDFs, or training-label changes are published.

Notebook SHA-256: `3cebaaa67aa76050113c75e963fca086575f7dc5a83181b40f9ef6ba565cd8ef`.
