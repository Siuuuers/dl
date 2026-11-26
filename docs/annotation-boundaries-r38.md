# R38: focused development-reference amendments

Reviewed 26 November 2026 after PR123 returned the R37 comparison. All eight current requests concern development images: three rows receive coordinate corrections and five rows are excluded by explicit user instruction. Every requested row number names the R37 reference **before** deletion. The inherited `new_000010:24` landmark question remains pending and its complete row is preserved.

## Physical target and coordinates

Keep the established target: observed cassette-front material, including its own molded lip, rim, short protrusion and shaded supporting material. Contour continuity, stacking and occlusion determine ownership; colour is supporting evidence. Exclude separate rack, gripper, background and neighboring cassettes. An enclosing rectangle can contain internal occluder pixels while correctly bounding owned visible material.

Stored original-image x grows rightward and y grows downward. The user's physical `ymax/top` request refers to stored `ymin`; the physical `ymin/bottom` request refers to stored `ymax`. The three accepted amendments are:

| R37 image / row | Accepted R38 correction | Preserved |
|---|---|---|
| new_000010 / 29 | Stored top 679.9118 to 666; observed left point (320.91,717.27) to (325,714) | Left/right/bottom box edges; left visibility 2; unavailable right triplet |
| new_000008 / 7 | Stored bottom 33 to 23 | Other box edges; left point (1221,20), visibility 2; unavailable right triplet |
| new_000007 / 47 | Stored top 388 to 385 | Other box edges; both observed points and visibility |

Coordinates are integer adjudications with approximately 2 original pixels of visual ambiguity. Unedited label tokens remain exact, and the existing opposite y edge is retained when converting a changed y edge back to normalized center/height tokens. Neither rounded predictions nor soft image boundaries provide subpixel physical certainty.

### new_000010 row 29

The old top at 679.9 cuts into the visible frontal material. The original light rim around y666 continues across the front below the broad upper plane, supporting top 666. The left lower-front turn is around (325,714); the old point lies leftward and lower in the dark supporting region. Keep the unseen right corner unavailable. The visible bottom remains clipped by the image boundary at y720.

PR121 candidate p28 and the separate PR123 candidate p28 both support this direction: their top is about 665.8 and their left point is about (324.6,713.7). These are corroborating predictions, not the definition of ground truth. Their other box edges and predicted right point are not substituted into the reference.

### new_000008 row 7

The saved bottom at y33 extends into the broad separator below the clipped front. The owned lower edge slopes and fades from the observed left turn near y20 toward roughly y23 at the right image boundary. Bottom 23 includes that visible extent without forcing the box to coincide with the left point. The left point itself remains unchanged.

PR121 and PR123 candidate p36 both extend to about y34; their box height is not adopted. This correction follows the original contour and the user's ownership direction. No unseen right point is added.

### new_000007 row 47

The upper shaded front edge reaches about y385 near the right side; the old top at 388 cuts that sloping extent. The user's explicit PR121 candidate p57 cue has top 384.77. PR123 p45 and the separate PR123 p57 have a similar top, supporting the same whole-pixel placement. Keep xmin 726 and the existing lower-front points; p57's narrower box and displaced left point are not adopted.

## Five case-specific exclusions

Remove exactly the R37 rows `new_000086:57`, `new_000088:57`, `new_000053:1`, `new_000010:30` and `new_000009:52`. These are explicit user-directed exclusions, not an automatic size cutoff or five confirmed duplicate annotations. In particular, `new_000009:52` visibly contains a small partial front; excluding it does not prove that the region is background. High containment of `new_000053:1` or `new_000010:30` with another reference box does not independently prove duplicate physical ownership.

The original R37 reference, labels and results remain intact. R38 removes five targets, four available endpoints and six unavailable endpoints. Development counts change from 659 to 654 fronts, 1,224 to 1,220 available endpoints, and 94 to 88 unavailable endpoints. The 583 two-corner objects remain. Any lower miss count caused by deleting a target is a reference-population change, not a model recovery.

## Stable identities and the inherited request

The operation ledger binds the parent file hashes, exact before-lines and source images. Stable IDs use `fronts-2026-11-26-r37/<image>:<row>`. The R37-to-R38 row map records retained/replaced rows and explicit null destinations for deleted rows. This matters for `new_000053` (old row 2 becomes current row 1), `new_000010` (old rows 31-34 become 30-33) and `new_000009` (old rows 53-55 become 52-54). Earlier R35 exclusions are never reintroduced.

`new_000010:24` remains R38 row 24 and is retained byte-for-byte. The left point remains unavailable behind the jaw, the observed right point remains (909,491), and the box remains (810,464,913,495). No current request resolves that inherited point-name question.

## Training and evaluation consequence

All 137 reviewed training files remain byte-identical to R37. The existing eligible R37 training manifest is retained; it still describes 3,784 training fronts and the original 664-front trainer-validation reference. No R38 training candidate, new recipe or paired GPU experiment is created for these development-only changes. The complete reviewed training snapshot separately retains 3,786 fronts because held/excluded training supervision is selected by the existing eligible manifest.

Preserve frozen 666, historical 663 and R37 evaluation results. Rescore fixed caches against R38 with explicit reference and row-map identities, distinguishing deleted targets, coordinate-induced matching changes and actual model differences. This reference replay does not show R38 learning or reliability on new camera sessions.

The accepted physical review used the materialized original PNGs. Default review verification still checks their bytes. For a later CPU cache replay, the verifier can instead bind the committed LFS image OIDs to those already-reviewed hashes while checking the same label, ledger, row-map and prediction-cache bytes. That mode requires no image or checkpoint download and does not repeat visual inspection or inference.

Expanding a target box into surrounding rack or environment is not adopted. The model already receives the surrounding image; moving a label edge changes the geometry it is taught to predict. If a context region is later needed, it should have an explicit separate meaning while the physical front and observed-corner definitions remain stable.
