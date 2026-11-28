# Partial-front inclusion: upstream evidence and the R67 decision

28 November 2026, Asia/Hong_Kong. Research base: `Siuuuers/dl` commit
`67024106`. This note separates upstream instructions, external evidence and the
local annotation decision. It does not itself change labels, scores or model
weights.

**Recommendation:** restore a previously excluded front in a new reference
version when the original photograph establishes one distinct, identifiable
visible target. Annotate its observed front extent and only its reliably
locatable lower corners. Keep all earlier reference versions and their results.
Small size, truncation or unavailable corners alone are insufficient reasons to
exclude an identifiable front. This is an explicit local inclusion decision
consistent with the supplied task, not a claim that upstream published a
complete partial-object policy.

## 1. Current upstream source was checked again

Remote `git ls-remote` returned these branch heads on the date above:

| Branch | Current commit |
| --- | --- |
| `baomai` | `351eb3cf960ccc34829e2bc72e1fa4645acc6295` |
| `yolo-pose` | `72a99dafa1b14bfabd101e639b9ddb5f5aa38d52` |

Both agree with the pins already recorded in
[the annotation policy](../annotation-policy.md). Files were read at immutable
commit URLs through the GitHub connector; their retrieved bytes reproduce the
listed Git blob identities. Web retrieval of these particular upstream pages
was unavailable, so the content findings are based on those verified repository
files, not on a search snippet.

| Source | What it actually establishes | What it does not establish |
| --- | --- | --- |
| [U1: current Baomai README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/README.md) | The target is the white cassette front's two actual lower-edge corners, ordered by image X. The upper lip, base/tray and rack corners are different features. | No minimum pixel area, visible percentage or general small-fragment exclusion rule. |
| [U2: current data README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md) | One `cassette_front` class; two points; `v=2` for visible points; `0 0 0` when a point cannot be located reliably. An available opposite endpoint is retained. Occlusion boundaries do not supply guessed corner coordinates. Genuine empty scenes have empty labels. | No detailed visible-versus-amodal box rule or explicit threshold deciding whether a partial front exists. |
| [U3: historical editor instructions](https://github.com/an94-doll/embedding_vision/blob/6f7112a553e1759fc755bda1f18ee56cb34dd315/detect/data/data_last/picture/work/ANNOTATION_EDITOR.md) | Adding an instance starts by drawing the visible front extent. Unclear or out-of-frame points remain `0 0 0`. Original images can be viewed without overlays. Pending, approved and excluded training-image states are distinct. | Its older frame-83 long-edge endpoint convention does not replace the current lower-front target. An image-level training exclusion is not an implemented per-object ignore region. |
| [U4: yolo-pose annotation README](https://github.com/an94-doll/embedding_vision/blob/72a99dafa1b14bfabd101e639b9ddb5f5aa38d52/annotations/README.md) and [U5: schema](https://github.com/an94-doll/embedding_vision/blob/72a99dafa1b14bfabd101e639b9ddb5f5aa38d52/annotations/pose_schema.yaml) | A different task: front/back classes, five object-relative points, a geometric midpoint, rounded-corner extrapolation and whole-object boxes including thickness/protrusions. | These five-point rules do not redefine the Baomai two-lower-corner task. They must not be combined silently with its rows. |

The historical instruction's phrase “可见正面范围” supports the visible-front
interpretation. The more detailed rules about front-surface ownership, shaded
material, tight enclosing extrema and retention of identifiable partials are
local decisions documented in [the current definition](../annotation-definition.md).
They should be presented as such. None of the inspected upstream instructions
says that a front should be deleted merely because one or both lower corners
cannot be marked.

## 2. Three external sources inform this narrow choice

### Missing objects and background supervision: SparseDet

Suri et al. [R1] distinguish annotated foreground, unannotated foreground and
background. Their analysis explains how omitted training objects can be treated
as negative regions; their method separates likely unlabeled foreground and
uses additional learning objectives. The relevant lesson here is to avoid
silently treating a confirmed positive as background. It does not establish that
the cassette model needs SparseDet, that an uncertain fragment is a positive, or
that restoring one label will improve recognition. Manual adjudication of a few
known cases is a smaller intervention than adopting its mining architecture.

**Split matters:** former `new_000009/R052` is a development-reference case.
Restoring a reference row changes evaluation membership and matching; it does
not change the loss of fine-tuning runs that exclude that photograph. The
missing-training-label mechanism is relevant to analogous omissions in training
images, not evidence of the cause of this particular development output.

### Visible extent and hidden extent are different targets: CityPersons

CityPersons [R2] explicitly provides a visible box derived from an instance's
segmentation and a separate aligned full-body box. It also records ignore
regions and gives them defined evaluation/training treatment. This demonstrates
that visible, amodal and ignored are distinct choices requiring explicit
representation. It does not imply that tissue cassettes should use full-body
targets or inherit pedestrian scale/occlusion cutoffs. Here the established
visible-front task is retained: cover observed material belonging to one front,
without guessing its hidden outline.

### Object presence and point availability are distinct: COCO format

The official COCO format [R3] adds keypoints to an object annotation that already
contains a box and identity. It distinguishes unlabeled, labeled-hidden and
labeled-visible keypoints, and explicitly allows object annotations with no
labeled keypoints. This supports separating the existence of a front from the
availability of its landmarks. Baomai has its own stricter working convention:
use only reliably locatable `v=2` points and canonical missing `0 0 0` slots.
COCO's `v=1` state is not permission to invent hidden cassette coordinates or to
add an unsupported flag to the current parser.

## 3. Apply the recommendation without changing what the evidence means

For each previously excluded row, inspect an unmarked original crop and enough
surrounding image to distinguish the front from neighbors, rack and gripper.
The previous row and prediction overlays can then be used for comparison; they
do not determine the answer.

| Image-supported judgment | R67 action |
| --- | --- |
| Distinct front with a positive-area visible extent | Include one object row, even if thin, clipped or partly covered. |
| One lower corner clearly locatable; the other unavailable | Preserve the available point in its physical left/right slot; encode the other as `0 0 0`. |
| Front identity and extent supported; neither corner locatable | Keep box supervision and detection evaluation; encode both missing points. Do not claim a complete-pair coordinate result for this target. |
| Two visible pieces demonstrably belong to one front | One enclosing visible-front box; do not duplicate the instance to avoid the occlusion gap. |
| Existing row already represents the same front | Correct the existing instance or remove a confirmed redundant row; do not restore a second copy. |
| Material ownership or instance identity remains unresolved | Record alternatives and hold the affected new reference decision for adjudication. Do not label the region as proven background. |
| Entire cassette is hidden and inferred only from spacing | Do not create a visible-front row. |

The [R65 image review](annotation-review-r65.md) already documents visible
partial-front material at the location of **R37 `new_000009/R052`**, explicitly
removed in R38. Its recorded historical box is `[600,490,615,504]`, with one
available corner near `(601,502)`. The new user authorization permits resolving
that inherited exception according to the consistent definition. It does not
make every historical row or coordinate correct. Use the current original-image
adjudication to establish the restored geometry; do not blindly copy the old
row or copy a model output. Other excluded rows require their own identity check.

Publish a separately named R67 reference and a field-level change ledger,
preserving the supplied, earlier fixed and reviewed references. Rescore the
same saved predictions against all relevant versions. A change in detection
counts after restoring a reference is a reference effect, not new model
learning. If training labels also change, isolate that intervention in a later
matched experiment and retain the established photograph split.

The current 11-field schema has no per-instance ignore field. Deleting a visible
front neither creates an ignore region nor guarantees that its predictions are
exempt from extra-output counts. No new ignore mechanism is needed for the
confirmed positive cases. Unresolved newly annotated photographs must be
adjudicated or explicitly quarantined with recorded split/count effects; they
must not be exported as exhaustive negative supervision. Any future ignore
contract should be separately implemented and evaluated.

## 4. Source identities and immutable access

The five upstream file hashes below were checked against their returned blob
identities. Full commit-specific URLs are U1–U5 above.

| Source | Bytes | Git blob SHA-1 | SHA-256 |
| --- | ---: | --- | --- |
| U1 | 7,793 | `80dd548b33e7b77ecc5f53b176b1d1725aed53e3` | `b7d545efd536a1d8ef31f01f102303bc9b1badcf480fb44121b4bf59c0badf91` |
| U2 | 3,263 | `7af771c01e2d35f0262afec040d039da679418f3` | `bb05d9c3fb396299227326132d71c4b7d84c0214eebcb819fb310bf172b8c740` |
| U3 | 2,745 | `4342fb7915a2c8ec345a0fb2fc50b011394eb50f` | `9bc2c68b87a4e33f68881a64958d2bb819a877fc5cc8dc095151c303fda2ded7` |
| U4 | 2,390 | `11c6732a38814695640a31e8cb0868cf3d7a8cf6` | `0158677dfd5e216b1eb1ae5333c205f21123889b50c5541a45ae01a85c086720` |
| U5 | 72 | `2f17fc046573d1f5b0c22ad5d9a347c02bb93226` | `dc83ed1384238eab6fd4b6383496b7ad7b2afa3d5c1b73737a42bf144c228638` |

### Research references

[R1] S. Suri, S. Rambhatla, R. Chellappa, and A. Shrivastava, “SparseDet:
Improving sparsely annotated object detection with pseudo-positive mining,” in
*Proc. ICCV*, 2023, pp. 6747–6758, doi: 10.1109/ICCV51070.2023.00623.
[Primary paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Suri_SparseDet_Improving_Sparsely_Annotated_Object_Detection_with_Pseudo-positive_Mining_ICCV_2023_paper.pdf) ·
[IEEE](https://ieeexplore.ieee.org/abstract/document/10378480) ·
[Author-institution publication record](https://pure.johnshopkins.edu/en/publications/sparsedet-improving-sparsely-annotated-object-detection-with-pseu/).
The CVF accepted copy uses pp. 6770–6781; the entry pairs the final publisher's
pagination with its DOI. Sections 1 and 3 establish the omitted-foreground issue
and the method; no published benchmark gain is treated as a cassette result.

[R2] S. Zhang, R. Benenson, and B. Schiele, “CityPersons: A diverse dataset for
pedestrian detection,” *CVPR*, 2017, Section 3.1.
[Primary paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.pdf) ·
[Immutable official annotation format](https://github.com/cvgroup-njust/CityPersons/blob/839c22fb05a16c150cb77f9b73a5c0e9642af21e/annotations/README.txt).
The guide's Git blob is `f956166f0cfabc126e4cb56928fd381d0dad3335`;
934 original bytes, SHA-256
`a45c8470e862bcbb7d270bddb9304e47e6620df19e2c55cc7270324cb2c6e7a7`.

[R3] COCO Consortium, “Data format,” Section 2, “Keypoint Detection.”
[Official documentation](https://cocodataset.org/#format-data) ·
[Immutable official source](https://github.com/cocodataset/cocodataset.github.io/blob/5e1c4da72464b1c6f068df0c02c91e3000ea62c4/dataset/format-data.htm).
Git blob `79b95a164fa8f019e83267d2173cbe22ffe26f35`; 15,380 bytes,
SHA-256 `a15ea326fbb7a7d5adc349aa8bfd615e442886b0c5cccb819cc9d5bce3185897`.

All linked source contents were inspected on 28 November 2026. No external
paper supplies the physical identity of a particular cassette fragment; that
decision belongs to the documented original-image review.
