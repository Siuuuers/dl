# Round 12: requested front and corner amendments

Round 12 starts from the saved round 11 review at `7699d086194abfeaed663fa19a05961b7492b4db`. The user supplied three marked screenshots and row-specific text covering 46 photographs. The complete review retains all 157 source photographs. Only requested new-series labels may change; unrequested new labels and all 31 accepted old labels retain their round 11 bytes.

## Physical annotation contract

Keep the [round 11 contract](annotation-boundaries-r11.md): one visible cassette front per instance, an axis-aligned box enclosing the visible front and its owned frontal rim, and the two actual lower front corners ordered by image X. A rack, rear perforated plane, hand or gripper does not establish a cassette boundary. A shaded frontal rim may belong to the cassette; brightness alone cannot determine ownership.

Observe box extrema and lower front/side junctions independently. Do not apply a universal inward offset or snap every point to a box corner. A point can legitimately lie inside an axis-aligned bounding box when the visible front has a projecting rim or a sloping edge.

A reliably locatable corner uses visibility 2. A hidden, out-of-frame or unlocatable corner is exactly `0 0 0`, independently on each side. A supported visible front fragment remains an instance even when both corners are missing. Add or remove an instance only when a physical separator or duplicate ownership is established.

## Interpreting the feedback

Operation identities bind to immutable round 11 rows, exact before-lines and source file hashes. Prior row maps are checked when a supplied cross-row reference appears to address a different physical front. Current rows and any added or deleted identities are recorded separately. The marked screenshots establish intended changes; original 1280 × 720 PNG pixels establish numerical coordinates.

Image X increases rightward and image Y increases downward. Several notes use `ymin` for the physical bottom and `ymax` for the physical top. Where the original photograph supports that reading, interpret the named physical edge and record the translation explicitly. Do not collapse a front by moving its top to its own lower corner. Cross-row references usually align one coordinate while preserving the target cassette's own corner and extent; they do not authorize copying a point into another instance.

Incomplete clauses and unsupported distant-row references receive explicit request dispositions. They do not by themselves make an otherwise supported photograph uncertain. A training hold is reserved for an actual unresolved physical boundary, point or instance identity. Decimal digits in the label format express serialization precision, not subpixel certainty.

## PDF and training inputs

The PDF contains exactly the photographs whose saved annotation bytes change relative to round 11, with every current instance displayed on each included photograph. Cyan denotes inherited fields, magenta edited fields, and amber unresolved fields. Tags retain parent-row identity when numbering changes; added instances receive a plus sign and removed instances appear as dashed ghosts.

Unresolved training photographs use exact PR35 labels in both comparison arms. The 31 accepted old photographs are separately excluded from treatment and also use identical PR35 labels in both arms. Original PNGs, split membership, empty scenes, original validation labels and the frozen 666-front primary and 663-front diagnostic scoring references remain fixed. Current development-review amendments are versioned separately and must not be presented as an untouched test set.

The notebook uses epoch 173 initialization independently for each arm, seed 46, 30 epochs, batch 4, image size 1280, FP32, AdamW and learning rate 0.0001, with final-epoch selection. Annotation review and notebook preparation do not demonstrate a model improvement or run new training. Compare the returned models under the declared evaluation protocol before deciding whether to promote a candidate.
