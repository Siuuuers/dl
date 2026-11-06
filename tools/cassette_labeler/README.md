# Cassette labeler

Open `index.html` in desktop Chrome or Edge. If opening with `file://` does not allow folder access, run from the repo root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open <http://localhost:8000/tools/cassette_labeler/>. No dependencies or network connection needed.

1. Click **Open data folder**, choose this repo's `data/` or exported `data_pred/`, and allow read/write access. The folder can contain `train`, `val`, or both under `images/` and `labels/`. PNGs must be downloaded, not Git LFS pointers.
2. Choose a `train/` or `val/` image. Existing labels load automatically; absent or empty labels start with no cassettes.
3. Select a cassette by box or number. Drag green (front_lower_left) and blue (front_lower_right) points onto the actual front-lower corners. Drag inside its box to move only the box; use **Draw / replace box** to redraw its bounds.
4. **Add cassette** starts a new box: drag its bounds, then use **Place left point** and **Place right point**. **Left missing** / **Right missing** write `0 0 0`; placing a point marks it visible (`2`). **Delete cassette** removes the selected row.
5. **Save label** overwrites `data/labels/{train,val}/<image-stem>.txt` in place (or creates it). All cassette rows are saved as `0 cx cy w h left_x left_y left_v right_x right_y right_v`, normalized to the image. Saving no cassettes writes an empty file. Switching images with unsaved changes asks before discarding them.
6. **Label text** shows those exact rows for the current image. Edit them and click **Apply** to update the cassettes; **Save label** writes the file. Invalid text leaves the cassettes unchanged. Pending text survives cassette selection and is protected when leaving the image. Apply it or use **Reset text** before editing graphically, using Undo, or saving. Reset text discards only the unapplied text.
7. **Undo** reverts one actual drag, add, delete, missing-point change, or Apply. Selecting a point or marking an already missing point does not consume it. Undo after saving marks the restored labels unsaved; save again to write them.

## Correct model predictions

In the existing model environment, with the validation PNGs and model weights downloaded, run from the repo root:

```sh
python scripts/export_pose_drafts.py
```

This exports predictions for the validation images into `data_pred/`. Open that folder in the labeler and save corrections there. The exporter stops if `data_pred/` already exists; move the old folder before deliberately starting a new export so corrected labels are retained.

Use the original images plus the matching label files. `data/annotated/` contains human-corrected previews with a footer, not editable model results. Live-camera S-key snapshots use JSON and are not imported by this labeler.

Browser automation uses the same controls and canvas; it needs access to the local repo and directory picker. Saves update only labels, not annotated previews or snapshot hashes in `data/manifest.json` / `data/SHA256SUMS.txt`.

The HTML page does no training or inference and needs no backend, account, framework, or package installation.

## Focused checks

From the repo root:

```sh
node tools/cassette_labeler/check.mjs
python3 scripts/check_export_pose_drafts.py
```

These dependency-free checks cover editing/Undo/file writes and draft-folder preservation. The exporter check uses a fake model; it does not run inference. Browser rendering and native folder permissions require a desktop browser walkthrough.
