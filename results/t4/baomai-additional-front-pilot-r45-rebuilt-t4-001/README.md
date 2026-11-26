# Reconstructed R45 supplementary-frame pilot T4 results

Provenance for the returned archive:

- Experiment: reconstructed R45 supplementary-frame / additional-front pilot (`r45-rebuilt`). This is a newly reconstructed replacement of a lost original, not the unpublished original notebook.
- Source revision: `ea4ac955f2dfaf1add24c243599715f802eb41a7` ("Reconstruct R45 pilot using published R37 and R44 inputs", 2026-11-26T15:21:18Z / 11:21:18 PM HKT / Asia/Hong_Kong on 26 Nov 2026). The rebuilt notebook does not use the lost original scripts `additional_front_pilot_r45.py` / `evaluate_additional_front_pilot_r45.py` / `experiments/additional-front-pilot-2026-11-26-r45.json`. Evaluator/packager is `scripts/rebuild_additional_front_pilot_r45.py` (the returned archive also stores `source/rebuild_additional_front_pilot_r45.py`). This results pack does not merge, close, or edit earlier PRs, including draft PR 132.
- Notebook: `notebooks/baomai_additional_front_pilot_r45_rebuilt.ipynb`, 13901 bytes, SHA-256 `08230b155237f11c7d8311d7aa22a5df39dd0abec756195448840834c51b693a`, LF-only, published at commit `c90b3bf31bedc21ce6402da3018b4c7143f1dd8c`. Drive copy run unchanged: https://colab.research.google.com/drive/1I8koriRXaUjhKer73XepzMzOcyOA7lyU?authuser=1 title Copy of baomai_additional_front_pilot_r45_rebuilt.ipynb, account 1 gloriawingyin@gmail.com.
- Drive backup path: `/content/drive/MyDrive/baomai-r45-rebuilt/ea4ac955f2dfaf1add24c243599715f802eb41a7/`
- Runtime: one Colab Tesla T4 High-RAM session throughout (Cells 1–11). Hostname `e447b3b11050` was observed in the Colab terminal nvidia-smi, not stored in the receipts (receipts have no `device_name`, `device_actual`, or `hostname` keys, and also have no hours, wall, or created_at fields). NVIDIA-SMI / Driver 580.82.07, CUDA 13.0, Tesla T4 15360 MiB, UUID `GPU-a93bfef5-482c-e458-7745-616de7a3cd83`, High-RAM ~51.0 GB. Receipt and `evaluation/summary.json` runtime block: gpu Tesla T4, torch 2.14.1+cu130, python 3.13.16, cuda 13.0, cudnn 92800, ultralytics 8.4.182. Training logs also record `CUDA:0 (Tesla T4, 14913MiB)`. Settings in receipts: amp false, batch 4, imgsz 1280, optimizer AdamW, seed 46, device 0.
- Roles: reference = exact R37 effective labels, 137 train photos / 3,784 fronts, 4110 presentations. Candidate = those labels plus the added complete R44 `frame_0383` (43 fronts), 138 train photos / 3,827 fronts, 4140 presentations. These train photo, front, and presentation counts were measured from the archive labels and receipts. Both arms independently started from `weights/cassette_lower_epoch173.pt` (staging `initializer` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`; receipt model path `/content/baomai-r45-rebuilt/source/weights/cassette_lower_epoch173.pt`). Recipe in `staging.json`: `reviewed-incremental-fronts-seed46-v1` SHA-256 `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`. 20 original development photos stay outside training (both roles have the same 20 val label files; none of those ids are in train). The added photo is never used as a test image (`frame_0383` is present only in the candidate train split). The only `frame_*` id in staging is candidate-train `frame_0383`; other R44 photos remain held and are not in either role's train or val lists. Resource-trial: status `trial-complete`, role candidate, 4 batches / 16 images / 4 steps (`trial` true), then discarded; Drive backup saved. The trial receipt is in the archive and has no `last.pt` artifact.
- Both 30-epoch arms completed on the same T4. Training logs record: reference `30 epochs completed in 0.208 hours`; candidate `30 epochs completed in 0.210 hours`. Observed session times were about 02:24 AM HKT / Asia/Hong_Kong on 27 Nov 2026 (reference) and about 02:46 AM HKT (candidate). Receipts do not store those clock times. Each 30-epoch receipt: status `complete`, batches 1050, steps 78, trial false, source `ea4ac955f2dfaf1add24c243599715f802eb41a7`, staging_sha256 `3935dc0d359a0ffe503fc71e0521e500294b3bbbe75ae296634277f0b9aa9bec`. Cell 11 evaluate+package finished about 02:49–02:50 AM HKT / Asia/Hong_Kong on 27 Nov 2026; files.download succeeded; the runtime then terminated; Manage Sessions was empty; Reconnect was not clicked.
- Archive: `baomai-additional-front-pilot-r45-rebuilt-results.zip`, 41067308 bytes, SHA-256 `0babe043f8358f7ceed3e4e0e570421585690ebf79496c689ba84c1b5b25869a`. The zip contains 335 members: `inventory.json` plus 334 inventory entries. Members include both 30-epoch `last.pt` files (21704750 bytes each), receipts/progress, logs, `evaluation/summary.json`, `evaluation/predictions.json`, and dataset labels. `staging.json` SHA-256 `3935dc0d359a0ffe503fc71e0521e500294b3bbbe75ae296634277f0b9aa9bec`.

Independently measured archive members (SHA-256 and size). These values were hashed from the extracted zip members and match the hashes recorded in the receipts and `inventory.json`:

- `candidate/fit/weights/last.pt`, 21704750 bytes, SHA-256 `e311400a4db6d912714166e839b136b49bbb5790d3d1d633d484961ac2d0a095`
- `reference/fit/weights/last.pt`, 21704750 bytes, SHA-256 `dfd68ac6d180c3adff2604a69ea5769421f8780f98774f831d08f21acce52002`
- `candidate/receipt.json`, 4111 bytes, SHA-256 `8c9799004c6372422df72ae6d92bc4d060a131258f6b24a32843bb7e77e758b6`
- `reference/receipt.json`, 4111 bytes, SHA-256 `96ff694d43b5bfcd4029f7d070dce09b9797df910acfde5041b3319a7a5fbb47`
- `resource-trial/receipt.json`, 3680 bytes, SHA-256 `f68f3d25048cae3ad1cb9a0dcf48a5971891755899063629fd4f76bb7c1ea319`
- `candidate/progress.json`, 4111 bytes, SHA-256 `8c9799004c6372422df72ae6d92bc4d060a131258f6b24a32843bb7e77e758b6`
- `reference/progress.json`, 4111 bytes, SHA-256 `96ff694d43b5bfcd4029f7d070dce09b9797df910acfde5041b3319a7a5fbb47`
- `resource-trial/progress.json`, 3680 bytes, SHA-256 `f68f3d25048cae3ad1cb9a0dcf48a5971891755899063629fd4f76bb7c1ea319`
- `candidate/fit/results.csv`, 6051 bytes, SHA-256 `cc140e8e83325a276fed8d0bb1c5fa00bf734839afc6ca84d6a7aee6babcbcec`
- `reference/fit/results.csv`, 6049 bytes, SHA-256 `ff18b3f25c05918f33c3493bfc63e63ce711c30b63aee15bb7d5e1fc9c2c8581`
- `candidate.log`, 227504 bytes, SHA-256 `15b4df216b798fd3d7279c58bcd4f62bccddf5d9105262131107962006df956f`
- `reference.log`, 229350 bytes, SHA-256 `f99ee61fe5051ce9091531b5e5ea64cbb49adc41cb8ae58816bec97505b9dc9e`
- `resource-trial.log`, 9952 bytes, SHA-256 `9489c12a1169d105839f6d0b7da417fd1d9c613aa4a4706cafba89dee25e7d5b`
- `evaluation/summary.json`, 2495077 bytes, SHA-256 `943934164c3b445f813a02d3cec395b48344e3b287580dffc2931143704ec242`
- `evaluation/predictions.json`, 1685720 bytes, SHA-256 `060994f91dd4a8853858d45d3bff2758b9c89b4447c2a971f68ca8b293e59767`
- `inventory.json`, 52475 bytes, SHA-256 `87c5f4ad55d96f9175574e82edb82c3781895641895ab2eedf7a4312a2b754a0`
- `staging.json`, 155082 bytes, SHA-256 `3935dc0d359a0ffe503fc71e0521e500294b3bbbe75ae296634277f0b9aa9bec`
- `source/rebuild_additional_front_pilot_r45.py`, 21653 bytes, SHA-256 `e6a61f4bd6aa792c0bb74ee6a6e4c07259c72494211f40cb90402354a2ab2507`
- `datasets/candidate/labels/train/frame_0383.txt`, 4042 bytes, SHA-256 `1848d4dad3185444726de644f5303ee548fc74e08d4361d11c8bd8220fece7b0`

Evaluation (from Cell 11, also recorded in `evaluation/summary.json`; the Cell 11 corner-eval lines below were checked against that summary and are recorded here, not interpreted as a promotion). Inference settings from `evaluation/summary.json`: references frozen666, historical663, r38; imgsz 1280, conf 0.13, iou 0.7, matching_iou 0.5, max_det 300, nms true, batch 1.

```
Verified return archive against current evidence.
frozen666 raw reference misses 36 unmatched 107 duplicates 31 other 76 two corners within5 517 / 584 within10 573
frozen666 raw candidate misses 33 unmatched 106 duplicates 29 other 77 two corners within5 518 / 584 within10 575
frozen666 postclip reference misses 36 unmatched 99 duplicates 25 other 74 two corners within5 517 / 584 within10 573
frozen666 postclip candidate misses 33 unmatched 98 duplicates 22 other 76 two corners within5 518 / 584 within10 575
historical663 raw reference misses 32 unmatched 106 duplicates 31 other 75 two corners within5 517 / 584 within10 573
historical663 raw candidate misses 30 unmatched 106 duplicates 29 other 77 two corners within5 518 / 584 within10 575
historical663 postclip reference misses 32 unmatched 98 duplicates 25 other 73 two corners within5 517 / 584 within10 573
historical663 postclip candidate misses 30 unmatched 98 duplicates 22 other 76 two corners within5 518 / 584 within10 575
r38 raw reference misses 11 unmatched 94 duplicates 30 other 64 two corners within5 571 / 583 within10 582
r38 raw candidate misses 13 unmatched 98 duplicates 26 other 72 two corners within5 569 / 583 within10 581
r38 postclip reference misses 11 unmatched 86 duplicates 22 other 64 two corners within5 571 / 583 within10 582
r38 postclip candidate misses 13 unmatched 90 duplicates 18 other 72 two corners within5 569 / 583 within10 581
Return: /content/baomai-r45-rebuilt/baomai-additional-front-pilot-r45-rebuilt-results.zip SHA256: 0babe043f8358f7ceed3e4e0e570421585690ebf79496c689ba84c1b5b25869a
```

The archive was reassembled by concatenating raw `part00` then `part01` after unzipping the stored chunk zips. `sha256sum -c` passed for the archive, the chunk zips, and the raw parts. `unzip -t` and `ZipFile.testzip()` reported no errors. The committed chunk blobs are the raw bytes, not Git LFS pointers. The uploads/repo directory omits the full ZIP; reconstruct it from the two stored parts before hashing.

This records the returned archive. It makes no accuracy or performance claim. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical/diagnostic only. automatic_promotion is false. This pack does not establish new-session reliability and is not an automatic model promotion. The three reference sets are scored separately (raw and postclip).
