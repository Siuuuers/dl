# Round 17 annotation amendments

The accepted r16 labels are the immutable parent. This round applies the user's new requests to photos 067-075, 079 and 080 only. All eleven belong to training. Earlier accepted labels, original images and original `data/labels/` bytes are preserved.

Each box follows its cassette's visible front material and owned rim. The two points locate independently visible lower front-to-side junctions, ordered by image X. A missing point is exactly `0 0 0`; a known point has visibility 2. A projecting rim can extend beyond its lower corner. Other objects and unlocatable sentinels cannot supply a boundary.

All row numbers and cross-row anchors refer to r16 before any edit in this round. A request such as `23 xmax to 24` uses row 24's original xmax. Point-to-edge requests transfer the indicated coordinate; cross-row point references transfer X while retaining the target's own Y and visibility unless Y is explicitly requested. Negative X direction means movement leftward to a smaller in-image coordinate, not a negative normalized coordinate. Bounded directional moves follow the visible junction; they are visual estimates rather than subpixel physical measurements.

Delete photo 074's source row 13 and set source row 10's right point to `0 0 0`. Preserve source row identities in the row map: surviving rows 14-20 become current rows 13-19. The original-baomai question about source row 20 is traced by object geometry, not by equating current and original row numbers. It does not authorize another deletion. No new instances are added.

Every requested clause receives a saved disposition. Unsupported or already-satisfied parts retain their saved values with an explicit reason; supported parts on that photo still apply. The review PDF renders the resulting saved labels with changed fields highlighted. Deleted-row markers are display history and never become training labels.

The sole inherited training hold remains photo 015's undrawn possible-front question. It keeps exact PR35 labels in both training arms. The 31 old-series photos remain a separate treatment exclusion and also retain exact PR35 labels in both arms. Inherited displayed-case approvals remain unchanged.

Use the same model, original epoch 173 initialization, seed 46, 30 epochs, batch 4, 1280 input, FP32, AdamW and final-epoch checkpoint selection. The resource trial cannot initialize a full arm. Original trainer validation stays at 20 photos/664 fronts; primary scoring stays at 666 fronts, historical scoring at 663 fronts, and the r16 development review at 662 fronts. PR35 remains selected until actual return evidence supports a change. Preparing a notebook or passing geometry checks does not establish physical annotation perfection or improved model performance.
