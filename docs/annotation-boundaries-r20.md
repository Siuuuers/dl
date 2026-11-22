# Round 20: horizontal rack margins

The accepted r19 PDF and saved labels are the immutable baseline. This pass reviews the 40 supplied nonempty photographs in new_000081–new_000130 and only tightens visually supported horizontal box margins. Photos 091–093 and 104–110 are absent from the prepared source. No original image or source label is changed.

## The same object definition

A box encloses the visible cassette front and its owned rim. Exclude the adjacent rack, fingers, gaps and background from unsupported outer margins. The two keypoints remain the actual lower-left and lower-right front-face corners, ordered by image X; they are independent of the outer side-face silhouette.

The user's shortcut, xmin near left-point X and xmax near right-point X, is a candidate boundary for an ordinary front whose edge meets that corner. Inspect the original pixels before accepting it. Keep the farther visible front edge when an upper rim or sloping front projects beyond the lower corner. An axis-aligned rectangle around a sloping front can still contain background at its corners.

Photo 087 row 5 is the accepted standard and remains byte-identical to r19. Its box already extends about 2.08 pixels beyond each lower corner, so the standard does not imply exact endpoint equality. Photo 086 row 3 also remains unchanged. Photo 087 rows 21–24 are the user's explicit cleanup targets. Their rack-facing edges are tightened using the existing visible front corners and original pixels.

## Deliberately narrow amendments

Only horizontal center and box-width tokens may change. Every object, row identity, class, vertical box field, corner coordinate and visibility token is preserved exactly. Missing corners remain the exact inherited sentinel and never supply an invented boundary. Changes only shrink an existing horizontal interval; ambiguous margins keep their accepted r19 coordinates. No new holds, cases or uncertainty markers are introduced.

Record exact before/after lines, original-image and parent-label hashes, and a physical reason for each accepted change. Save the full review and the training candidate separately. The changed-photo PDF reads those saved labels and highlights changed boxes in magenta; retained fields are cyan. The display never moves label geometry.

## The same experiment

Use the existing two-point Baomai model and paired training recipe. Both arms independently initialize from original epoch 173: seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and initial learning rate 0.0001. Select each final-epoch checkpoint. The small resource trial never initializes either full run.

Retain 137 training photos, including 13 empty scenes. Photo 015 and the 31 old-series photos retain exact PR35 supervision in both arms. Photos 086–090 are development images; their review edits do not enter training or replace the original 20-photo/664-front trainer validation. Keep the frozen 666-front primary and 663-front historical scoring references unchanged. The separately versioned r20 development review is exploratory evidence.

The experiment compares accumulated eligible r20 supervision against PR35 supervision. It does not isolate the effect of this horizontal cleanup or establish performance on a new capture session. No new model is selected merely because labels or the notebook were prepared.
