# Partial-classification r51 T4 results

Provenance for the returned archive:

- Experiment: partial-classification-r51, controlled training experiment (`schema` `baomai-r51-partial-classification-v1`). Two matched 30-epoch arms start independently from `weights/cassette_lower_epoch173.pt` (staging `initializer` SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`; receipt model path `/content/baomai-r51-da5a67d4bec8/run/initializer/cassette_lower_epoch173.pt`) with recipe `reviewed-incremental-fronts-seed46-v1` (recipe SHA-256 `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`). Both arms use the same 142 photos / 3999 labels as the PR137 candidate. Shared training budgets: 1080 batches / 80 optimizer updates / 4260 image presentations, seed 46, imgsz 1280, batch 4, FP32 (`amp` false), AdamW. Classification factors: reference=1 / candidate=2 (candidate doubles classification BCE only on anchors assigned to partial cassettes). Staging records `same_images_and_labels` true, `same_initializer` true, `equal_image_exposure` true, `equal_optimizer_updates` true, `native_equivalent_reference` true, `no_label_changes` true, and `automatic_promotion` false. This results pack does not merge, close, or edit earlier PRs, including draft PR 145, results PR 144, results PR 143, PR 140, draft PR 142, draft PR 139, and PR 137. PR35 remains the selected training comparison.
- Source revision: `da5a67d4bec8171246c56c6b03b34ee20fc519e0`, printed as `SOURCE_REVISION` and stored as receipt/`staging.json`/`evaluation/summary.json` `source`. `SCRIPT_SHA256` `382b51e48013ec82b30eb1a44175546ae30ce716bd75882646338ae8b48b0852` (matches inventory and independently measured `source/scripts/run_partial_classification_r51.py`).
- Notebook: unchanged `notebooks/baomai_partial_classification_r51.ipynb` from commit `07d0261e420a918c01a823f8668b00f1472f5f31`, 13475 bytes, SHA-256 `aaaadd3d76cdb8e0744c929c8cd9122f70e066c252d602dba5265bf5e8029663`, LF-only. Drive copy run unchanged: https://colab.research.google.com/drive/1wBq0sIOFkKtWHQVNuHHgWI4Q_yDIM5Uf?authuser=1
- Runtime: one Colab Tesla T4 High-RAM session. A100 High-RAM was requested; Colab connected an authorized T4 High-RAM fallback ("The selected GPU type was not available. You are now connected to a T4."), which was kept. Hostname `611c67e1740b` and the NVIDIA-SMI / UUID below were observed in the run account / Colab terminal; they are not stored in the receipts (receipts have no `device_name`, `device_actual`, `hostname`, or session-time keys). NVIDIA-SMI / Driver 580.82.07, CUDA 13.0; GPU 0 Tesla T4 15360 MiB, UUID `GPU-8b7894d2-fe3d-f91e-f395-a2cc8b1e03e4`; UI RAM 50.99 GB (High-RAM). Receipt and `evaluation/summary.json` runtime block: gpu Tesla T4, torch 2.14.1+cu130, python 3.13.16, cuda 13.0, cudnn 92800, ultralytics 8.4.182. Settings in receipts: amp false, batch 4, imgsz 1280, optimizer AdamW, seed 46, device 0. The runtime was released after the ZIP; Manage Sessions was empty; Reconnect was not clicked.
- Resource-trial: four-batch resource trial first; status `trial-complete`, role candidate, `trial` true, batches 4, images 16, steps 4; `artifacts` is null (no trial `last.pt`). Trial weights were not used to initialize the arms. Both arms status `complete`, `trial` false: reference 1080 batches, 30 epochs, 80 steps, 4260 images; candidate 1080 batches, 30 epochs, 80 steps, 4260 images. Training logs record reference `30 epochs completed in 0.222 hours` and candidate `30 epochs completed in 0.224 hours`. Receipt `staging_sha256` and `staging.json` SHA-256 `dcaf95c7dd1e1d39bffae8bebed50068efbdee5f2cd071a5d166787f32348855`.
- Archive: `baomai-partial-classification-r51-results.zip`, 43528788 bytes, SHA-256 `9538932145e90201d407c339b7da1f73c5c905935853191ab9683b221bd5a09e`. The zip contains 519 members: `inventory.json` plus 518 inventory entries. Members include both 30-epoch `last.pt` files (21704750 bytes each), receipts/progress, logs, `evaluation/summary.json`, `evaluation/predictions.json`, and dataset labels. `unzip -t` and `ZipFile.testzip()` reported no errors. This matches the Colab Cell 4 Return line. Return path: `/content/baomai-r51-da5a67d4bec8/baomai-partial-classification-r51-results.zip`.

Independently measured archive members (SHA-256 and size). These values were hashed from the extracted zip members and match the hashes recorded in the receipts and `inventory.json`:

- `candidate/fit/weights/last.pt`, 21704750 bytes, SHA-256 `b37738005c47eebe489ab5a06723e2e32b3c50cfdfaae88a178518262bd494ab`
- `reference/fit/weights/last.pt`, 21704750 bytes, SHA-256 `0a1546050a56e6b1eab0d8555dcafa0f76378c64193a25648fd8d66ff713a3cb`
- `candidate/receipt.json`, 3654443 bytes, SHA-256 `db5c077ad8478fcf3e09d4957db5f27442d720f3fc2776c9e3f09210f8a00dc4`
- `reference/receipt.json`, 3651850 bytes, SHA-256 `4a3d771499921dc1b5e03c1386f7a7c2c953b187a8cdf2e3f8a9456b1f32a083`
- `resource-trial/receipt.json`, 24012 bytes, SHA-256 `8de778de7c4df30e8e838e955ed1b2dcb4e33c9ecef91673ca1ce3af4a8f8195`
- `candidate/progress.json`, 3654443 bytes, SHA-256 `db5c077ad8478fcf3e09d4957db5f27442d720f3fc2776c9e3f09210f8a00dc4`
- `reference/progress.json`, 3651850 bytes, SHA-256 `4a3d771499921dc1b5e03c1386f7a7c2c953b187a8cdf2e3f8a9456b1f32a083`
- `resource-trial/progress.json`, 24012 bytes, SHA-256 `8de778de7c4df30e8e838e955ed1b2dcb4e33c9ecef91673ca1ce3af4a8f8195`
- `candidate/fit/results.csv`, 6047 bytes, SHA-256 `08ee787f32ae39bde6006b2022bb7592413c9699f663a748607a76185ad94051`
- `reference/fit/results.csv`, 6046 bytes, SHA-256 `66aeb0981d72feb86ae87e5d35bd316c8c8ca7c699469dcf289669214543616e`
- `candidate.log`, 233134 bytes, SHA-256 `fa82856a21378f6d11336e87c91b972a2a614b4b5048ae2f460b20a6fb985d1f`
- `reference.log`, 233124 bytes, SHA-256 `85056527d9837b72415d2fcc10edeba7d9bff5f2c6afdc3de69180034af5c897`
- `resource-trial.log`, 7523 bytes, SHA-256 `b7d3e823248ac554b4a9993421c3d23e63c5f5750bb231adbc29fbea8d455a42`
- `evaluation/summary.json`, 3341663 bytes, SHA-256 `e6514d5e26c79a0f3498d8f6e0370515efcf933691e8d5e3e8bb7dad89ed7317`
- `evaluation/predictions.json`, 1708809 bytes, SHA-256 `5fb49479ccbd93b0f9f538a0ad72e9d410c674f2a576107c1ec46f6507b1dbec`
- `inventory.json`, 80888 bytes, SHA-256 `ecad271a5961f89ea3740a08d07b38e35e880fd4887d6a79c83811587fe4feff`
- `staging.json`, 186624 bytes, SHA-256 `dcaf95c7dd1e1d39bffae8bebed50068efbdee5f2cd071a5d166787f32348855`
- `source/scripts/run_partial_classification_r51.py`, 48277 bytes, SHA-256 `382b51e48013ec82b30eb1a44175546ae30ce716bd75882646338ae8b48b0852`
- `cpu-loader-preflight.json`, 2734 bytes, SHA-256 `42f7276c6be249aa87004f203459a6e8d91475efbf8e9c845fee39484662c6c2`
- `datasets/candidate/labels/train/frame_0146.txt`, 4102 bytes, SHA-256 `533da8e4c77be0ecf16be8957c2e75ede93c29c9dc5e931a6337e8014177e907`
- `datasets/candidate/labels/train/frame_0160.txt`, 4042 bytes, SHA-256 `0fc63ecdceb96ea26ec3ae17eccb4365bb1442325de0c7076e997778e19366e3`
- `datasets/candidate/labels/train/frame_0178.txt`, 4072 bytes, SHA-256 `34545a83102bddfea1de5e64a4a65dabfdb2e7d2a6a21e3e3af0876c9850d5f2`
- `datasets/candidate/labels/train/frame_0371.txt`, 4042 bytes, SHA-256 `83c5ab727ee95c937cbc7f91afe38b832d0b0eb6a8dc25664afeb396f477e2f5`
- `datasets/candidate/labels/train/frame_0383.txt`, 4042 bytes, SHA-256 `1848d4dad3185444726de644f5303ee548fc74e08d4361d11c8bd8220fece7b0`
- `datasets/reference/labels/train/frame_0383.txt`, 4042 bytes, SHA-256 `1848d4dad3185444726de644f5303ee548fc74e08d4361d11c8bd8220fece7b0`

Evaluation recorded in `evaluation/summary.json` (source of truth; postclip tp/fn/fp/duplicate_fp, with two-corner `corners_by_locatable_count.2` within5px / eligible and within10px). Inference settings from that summary: references frozen666, historical663, r38, r50; imgsz 1280, conf 0.13, iou 0.7, matching_iou 0.5, max_det 300, nms true, batch 1. These counts are recorded, not interpreted as a promotion.

```
frozen666 postclip reference 632/34/91/17 both within5px 511 / 584 within10px 575
frozen666 postclip candidate 635/31/120/23 both within5px 511 / 584 within10px 575
historical663 postclip reference 633/30/90/17 both within5px 511 / 584 within10px 575
historical663 postclip candidate 634/29/121/23 both within5px 511 / 584 within10px 575
r38 postclip reference 639/15/84/14 both within5px 566 / 583 within10px 582
r38 postclip candidate 643/11/112/24 both within5px 567 / 583 within10px 582
r50 postclip reference 639/15/84/13 both within5px 566 / 583 within10px 582
r50 postclip candidate 643/11/112/23 both within5px 567 / 583 within10px 582
```

Colab on-screen postclip counts were recorded for cross-check only (tp/fn/fp/duplicate_fp, two-corner within5px). They matched the JSON values above except historical663 reference fn, which printed as 31 on screen; `evaluation/summary.json` records 30. This README uses the JSON value.

Raw counts from `evaluation/summary.json` (clearly labeled; not the on-screen postclip lines; also not a promotion):

```
frozen666 raw reference 632/34/100/24 both within5px 511 / 584 within10px 575
frozen666 raw candidate 635/31/129/30 both within5px 511 / 584 within10px 575
historical663 raw reference 633/30/99/24 both within5px 511 / 584 within10px 575
historical663 raw candidate 634/29/130/30 both within5px 511 / 584 within10px 575
r38 raw reference 639/15/93/22 both within5px 566 / 583 within10px 582
r38 raw candidate 643/11/121/32 both within5px 567 / 583 within10px 582
r50 raw reference 639/15/93/21 both within5px 566 / 583 within10px 582
r50 raw candidate 643/11/121/31 both within5px 567 / 583 within10px 582
```

This pack uses the same method as the r47 / r45 / r37 packs: split the raw ZIP into 20971264-byte parts, wrap each raw part with ZIP_STORED (one member whose name is the raw part filename), and commit the wraps plus `baomai-partial-classification-r51-results.zip.sha256`, `parts.sha256`, `chunks.sha256`, and `README.md`. The full ZIP is not committed and is not stored with Git LFS. Because this archive is 43528788 bytes, which is larger than two 20971264-byte parts (41942528 bytes), a literal 20971264-byte split yields three raw parts (`part00` and `part01` at 20971264 bytes, `part02` at 1586260 bytes) and therefore three stored wraps. The wrap overhead is 202 bytes per part (it depends on the r51 part filename length and matches the r47 202-byte overhead). All three wraps start with `PK\x03\x04`.

Reconstruction: unzip each stored chunk zip, then concatenate the raw parts in order (`part00`, then `part01`, then `part02`):

```
unzip baomai-partial-classification-r51-results.zip.part00.zip
unzip baomai-partial-classification-r51-results.zip.part01.zip
unzip baomai-partial-classification-r51-results.zip.part02.zip
cat baomai-partial-classification-r51-results.zip.part00 \
    baomai-partial-classification-r51-results.zip.part01 \
    baomai-partial-classification-r51-results.zip.part02 \
    > baomai-partial-classification-r51-results.zip
sha256sum -c baomai-partial-classification-r51-results.zip.sha256
sha256sum -c parts.sha256
sha256sum -c chunks.sha256
```

The archive was reassembled that way before this pack was committed. `sha256sum -c` passed for the archive, the raw parts, and the chunk wraps. The reconstructed concatenation hashed to `9538932145e90201d407c339b7da1f73c5c905935853191ab9683b221bd5a09e`. The committed chunk blobs are the raw bytes, not Git LFS pointers. The uploads/repo directory omits the full ZIP; reconstruct it from the three stored parts before hashing.

This records the returned archive. It makes no accuracy or performance claim. It is not an automatic model promotion. PR35 remains selected. Frozen 666 is the primary evaluation reference and 663 is historical/diagnostic only. The four reference sets are scored separately.
