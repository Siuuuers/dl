# Cassette labeler

Open `index.html` in desktop Chrome or Edge. If opening with `file://` does not allow folder access, run from the repo root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open <http://localhost:8000/tools/cassette_labeler/>. No dependencies or network connection needed.

1. Click **Open data folder**, choose this repo's `data/`, and allow read/write access. PNGs must be downloaded, not Git LFS pointers.
2. Choose a `train/` or `val/` image. Existing labels load automatically; absent or empty labels start with no cassettes.
3. Select a cassette by box or number. Drag green (front_lower_left) and blue (front_lower_right) points onto the actual front-lower corners. Drag inside its box to move only the box; use **Draw / replace box** to redraw its bounds.
4. **Add cassette** starts a new box: drag its bounds, then use **Place left point** and **Place right point**. **Left missing** / **Right missing** write `0 0 0`; placing a point marks it visible (`2`). **Delete cassette** removes the selected row.
5. **Save label** overwrites `data/labels/{train,val}/<image-stem>.txt` in place (or creates it). All cassette rows are saved as `0 cx cy w h left_x left_y left_v right_x right_y right_v`, normalized to the image. Saving no cassettes writes an empty file. Switching images with unsaved changes asks before discarding them.

Browser automation uses the same controls and canvas; it needs access to the local repo and directory picker. Saves update only labels, not annotated previews or snapshot hashes in `data/manifest.json` / `data/SHA256SUMS.txt`.

No training, inference, Roboflow, backend, auth, extra classes/keypoints, theming, frameworks, packaging, CI, or test suite.
