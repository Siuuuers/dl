# R45 recovery handoff (not runnable)

The R44 audit is already published at `61c04d6a6dcb7e9f01ca99c603148b7f1506766b`. This draft preserves the recovered original R45 notebook, not a replacement implementation or a completed training run.

**Do not run the notebook yet.** Its source revision `60bc0b6a3c0efadd53a21dab0699b39327054396` was not published and cannot currently be retrieved from GitHub. The earlier local 22-file publication patch is unavailable in the current session. GitHub returned HTTP 422 for that revision.

Recovered notebook: `notebooks/recovery/baomai_additional_front_pilot_r45.ipynb` (11,241 bytes, original SHA-256 `4401f23301b75eeff3288f239db34a1c2d34d40ddd128d7dfb236efccaa70251`).

Required missing source includes:
- `scripts/additional_front_pilot_r45.py`
- `scripts/evaluate_additional_front_pilot_r45.py`
- `experiments/additional-front-pilot-2026-11-26-r45.json` (expected SHA-256 `7a4acb537abb611647e82615afe96badba2ec60a5a77218a9b3f28ab30cb80ba`)
- The remaining text-only changes from the original R44/R45 publication patch.

The intended experiment compares 137 original R37 training photos with the same 137 plus the fully reviewed R44 frame_0383 (43 fronts), starting independently from epoch173. It fixes seed 46, batch 4, 1280 input size, FP32, AdamW, 30 epochs, and final last.pt selection. Keep the 20 original development photos outside training. Evaluate misses, unmatched outputs, duplicates, and corner successes together on the same fixed references and settings.

Recover the exact source (or explicitly reconstruct and validate a new version), publish an immutable reachable revision, verify the protocol hash and data preparation, and only then move the notebook into the normal notebook directory and run on GPU. No model selection is changed.

This recovery commit adds text only: the notebook and this note. No images, PDFs, archives, weights, or new annotation changes are included.
