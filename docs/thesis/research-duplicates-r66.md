# Duplicate and unmatched outputs: evidence before another model

28 November 2026, Asia/Hong_Kong. Research and interpretation supporting the
[frozen output audit](duplicate-audit-r66.md). No default model, annotation or
operating threshold is changed by this note.

**Retrospective update, 28 November 2026:** the [addendum below](#retrospective-addendum-28-november-2026) corrects the prospective Soft-NMS wording using recovered completed experiments and links the subsequent partial-front work. The original analysis remains available as a historical record.

**Decision:** inspect physical instance ownership and the output-selection
trade-off before adding a loss, ranker or detector family. The completed paired
returns and [assignment diagnostic](score-assignment-audit.md) do not support a
general absence of training assignments. Removing the corner term is also not
an established remedy: the frozen box-only counterfactual lowers the mean target
for one-known-corner fronts. Neither observation explains every development
error or reconstructs historical training.

## 1. Evaluation outcomes and physical explanations are different

The scorer matches predictions to references once at box IoU at least 0.50. An
unmatched prediction is an extra; an unmatched reference is a miss. An extra
overlapping an already matched reference at IoU at least 0.50 is counted within
the duplicate subset. NMS instead compares predictions with predictions. Thus
two outputs can both overlap the same reference sufficiently to be a scored
duplicate while their mutual overlap remains below the NMS cutoff.

TIDE [1] distinguishes sources of detection error and estimates their separate
effects on AP. Its discussion of annotation mistakes also shows why a confident
unmatched prediction can correspond to omitted or ambiguously labeled
foreground. The cassette audit borrows this diagnostic distinction; its fixed
threshold counts are not a TIDE oracle-AP analysis. One displaced box can produce
both an extra and a miss, so adding those counts does not count distinct physical
failures. A maximum reference overlap below 0.10 is a review cue, not proof of
background.

The [training-label return](training-label-results.md) contains 13 duplicate
extras under the reviewed reference. Their overlap with the prediction matched
to the same highest-overlap reference is approximately 0.475–0.700, all at or
below the suppression cutoff of 0.70. Twelve concern an incomplete-corner
reference. This is consistent with the rule, but it does not establish that
every geometrically associated output depicts the same physical front.

## 2. What the inexpensive output rules establish

The [R66 audit](duplicate-audit-r66.md) evaluates fixed rules on saved outputs.
For the reviewed-label two-view arm, added post-output hard NMS at 0.50 changes
misses/extras/duplicates from **15/51/13 to 19/23/1**. Its complete-pair counts
remain **572/583 within five pixels and 583/583 within ten pixels**. Requiring
support from the existing mutual original/reflected association changes the
detection counts to **19/27/6**. Duplicates are included within extras in all
three rows.

These outcomes make the trade-off concrete: fewer extras can coincide with lost
partial-front detections, even while complete-pair scores stay unchanged. The
five- and ten-pixel denominators omit references without two available corners.
Neither rule is promoted. The audit preserves the other tested fixed rule,
reference versions, individual changes and matching checks; this note does not
select an operating point from those reused development photographs.

The reflected input is a deterministic transformation of the same photograph.
Agreement between its network response and the original-view response is a
consistency signal, not an independent observation, another camera angle or
evidence revealing an occluded surface. Shared errors can agree. A valid front
can also disappear in one orientation, so a consensus gate may remove a true
detection. This is a reason to measure the gate, not to treat agreement as ground
truth.

An added filter acts only on surviving outputs. It cannot recover proposals
already removed by native NMS or confidence filtering. Similarly, the audit's
maximum-cardinality comparison is an annotation-dependent diagnostic bound,
not an inference rule that may consult reference boxes.

## 3. What the suppression papers would change

**Soft-NMS [2] changes score decay, not the learned detector.** It progressively
reduces overlapping predictions' scores instead of immediately deleting them.
Its linear and Gaussian variants differ; the paper uses Gaussian sigma 0.5 in
its main experiments. A score floor still removes sufficiently decayed outputs.
The method can preserve adjacent objects lost by hard suppression, but is not
inherently a stronger duplicate remover. No cassette benefit follows from the
paper's benchmark gains.

A later single fixed Soft-NMS comparison should begin with verified **pre-NMS
decoded proposals**, retain original and decayed scores separately, and declare
which native/postclip suppression stage is replaced. Applying it to an already
suppressed cache cannot reproduce the method's recovery opportunity. Preserve
the existing operating cutoff and report ranking measures because decay changes
score semantics. The current [stage audit](proposal-stages.md) bounds recovery
from suppression-only changes above the existing confidence cutoff at three
reviewed fronts on that saved candidate pool. It cannot recover the twelve
confidence-gate losses or three targets without matching decoded geometry.

**IoU-Net [3] changes which overlapping box is preferred.** It learns box
localization confidence and uses it to guide suppression; the paper also has a
separate refinement procedure. This addresses the measured cases in which a
worse-ranked geometry suppresses an eligible alternative. It is a conditional,
more costly option because it requires quality supervision and another learned
component. The current native pose targets already combine box and known-corner
quality, so adding a predictor should not be described as introducing
localization awareness for the first time. A box-quality ranker must also retain
good corner coordinates; accurate boxes and accurate task landmarks are
different outcomes.

## 4. One-to-one training has already been tried here

DETR [4] trains a set of predictions through bipartite assignment and removes
conventional NMS from its inference design. YOLOv10 [5] combines one-to-many and
one-to-one training branches, aligns their matching metrics, and uses the
one-to-one branch for inference. These are training mechanisms for discouraging
redundancy, not guarantees that a learned model can never emit two predictions
for one real object. The pinned Pose26 implementation already contains both
branches; its pose-specific rules are established by the inspected source, not
by assuming it reproduces either paper completely.

The [historical PR121 probe](../front-review-r37.md) tested the trained
one-to-one head from the same checkpoint without another training run. At its
then-current confidence cutoff of 0.13, extras fell from 94 to 67, but misses
rose from 15 to 16 and duplicates from 31 to 37. Five-pixel pair successes rose
from 570 to 572, while ten-pixel successes fell from 582 to 581, each out of 583.
One clipped zero-area output was explicitly counted as an invalid extra. These
are historical settings and references, not an R66 comparison. They prevent
presenting an NMS-free head as an unexplored duplicate cure. A full replacement
with DETR would change too many components to isolate the current failure.

## 5. Raw-image review, omissions and the scope exception

Noise-aware evaluation [6] demonstrates that annotation discrepancies can alter
detector rankings, especially for small boxes and strict overlap criteria. It
supports examining repeatability and reference sensitivity; it does not decide
the physical cassette boundary or justify moving a label toward a prediction.

Use the [annotation definition](../annotation-definition.md) and review original
pixels before displaying model/reference overlays. Record instance ownership,
visible extrema, each corner's availability and unresolved alternatives. Then
compare the overlays and preserve a reason for every accepted change. A useful
physical review distinguishes repeated output for one front, distinct adjacent
fronts, wrong extent, omitted reference, background and unresolved ownership.
Keep that judgment separate from the automatic matching category.

The general policy already retains each identifiable partial front once.
However, the [R65 review](annotation-review-r65.md) recovered an inherited
exception: **R37 `new_000009/R052`** was explicitly excluded in R38 despite
visible partial-front material. Its historical box was `[600,490,615,504]` with
one available corner near `(601,502)`. The original row identity and instruction
are preserved in the linked review. This is a frozen inventory exception, not a
general exclusion of partial objects or evidence of background. It remains
unchanged here; a future reference version must explicitly retain, reverse or
otherwise resolve that case-specific scope decision.

Deleting a row does not create an ignore region. The current parser and scorer
have no per-instance ignore field, and an unavailable corner does not ignore its
front. Unresolved new annotations should be adjudicated or the whole photograph
quarantined with its reason and split changes recorded, rather than presenting
an omitted positive as exhaustively annotated background. An actual ignore
mechanism would require a separately declared training/evaluation contract.
This unresolved exception is the specific question; the established requirement
to retain identifiable partial fronts does not need to be reopened.

## Primary sources

[1] D. Bolya, S. Foley, J. Hays, and J. Hoffman, “TIDE: A general toolbox for
identifying object detection errors,” in *Computer Vision—ECCV 2020*, LNCS,
vol. 12348, pp. 558–573, 2020, doi: 10.1007/978-3-030-58580-8_33.
[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480562.pdf) ·
[Publisher](https://link.springer.com/chapter/10.1007/978-3-030-58580-8_33).

[2] N. Bodla, B. Singh, R. Chellappa, and L. S. Davis, “Soft-NMS—Improving object
detection with one line of code,” in *Proc. ICCV*, pp. 5562–5570, 2017,
doi: 10.1109/ICCV.2017.593.
[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.pdf) ·
[IEEE](https://ieeexplore.ieee.org/document/8237855) ·
[Author-institution publication record](https://pure.johnshopkins.edu/en/publications/soft-nms-improving-object-detection-with-one-line-of-code/).
The CVF accepted version instead prints pp. 5561–5569; the entry uses final
publisher pagination with the publisher DOI.

[3] B. Jiang, R. Luo, J. Mao, T. Xiao, and Y. Jiang, “Acquisition of localization
confidence for accurate object detection,” in *Computer Vision—ECCV 2018*, LNCS,
vol. 11218, pp. 816–832, 2018,
doi: 10.1007/978-3-030-01264-9_48.
[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.pdf) ·
[Publisher](https://link.springer.com/chapter/10.1007/978-3-030-01264-9_48) ·
[Accepted-version record](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.php).
The CVF accepted-version record instead lists pp. 784–799; the entry uses the
Springer publication's pp. 816–832 and volume 11218 with its DOI.

[4] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko,
“End-to-end object detection with transformers,” in *Computer Vision—ECCV 2020*,
LNCS, vol. 12346, pp. 213–229, 2020, doi: 10.1007/978-3-030-58452-8_13.
[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf) ·
[Publisher](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_13).

[5] A. Wang, H. Chen, L. Liu, K. Chen, Z. Lin, J. Han, and G. Ding, “YOLOv10:
Real-time end-to-end object detection,” in *Advances in Neural Information
Processing Systems*, vol. 37, pp. 107984–108011, 2024,
doi: 10.52202/079017-3429.
[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf) ·
[Proceedings record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c34ddd05eb089991f06f3c5dc36836e0-Abstract-Conference.html).

[6] J. Murrugarra-Llerena and C. R. Jung, “Noise-aware evaluation of object
detectors,” in *Proc. IEEE/CVF WACV*, pp. 9322–9331, 2025,
doi: 10.1109/WACV61041.2025.00903.
[Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.pdf) ·
[IEEE](https://ieeexplore.ieee.org/abstract/document/10943665).
The CVF accepted version prints pp. 9304–9313; this entry preserves the final
publisher pagination already verified in the manuscript's bibliography audit.

The papers were read from primary full-text sources. Publication records were
checked on 28 November 2026. Sources motivate hypotheses and interpretation;
project performance claims refer to the linked repository evidence.

## Retrospective addendum, 28 November 2026

Section 3's proposed Soft-NMS comparison omitted completed project evidence.
[PR143/R49](../front-diagnosis-r51.md) already replayed hard NMS and linear
Soft-NMS from the same decoded proposals, using confidence strictly above 0.13,
overlap 0.70, cap 300 and the unchanged postclip filter. Under R50, misses stayed
at 15, unmatched outputs fell from 84 to 83 and formal duplicates from 13 to 12.
Five-pixel pair success stayed at 566/583. The primary frozen666 comparison also
recovered no front and removed one unmatched output. Raw Soft-NMS before
postclip filtering recovered no target, so that filter did not erase a measured
recall gain.

The [PR144/R50 confirmation](../../results/evaluation/pr144-r50-20261127/README.md)
found all 20 decoded tensors byte-identical to PR143 and complete prediction and
trace objects identical. These are two returns of the same fixed comparison,
not independent scenes or evidence for a second successful method. A generic
Soft-NMS rerun is therefore not the next experiment. A different variant would
need a distinct stage-specific reason and a declared protocol. The result does
not establish that every Soft-NMS configuration fails.

The later [partial-front review](partial-inclusion-r67.md) resolves four
historical exclusions, including the `new_000009/R052` case discussed in section
5. Three other cases remain held. The [R68 research and completed CPU
comparison](research-partial-detection-r68.md) then separates confidence and
geometry losses and evaluates one fixed union with unused reflected outputs.
For the reviewed-label model under r67, that union reduces misses from 18 to 10
but increases extras from 50 to 60 and duplicates from 13 to 18. All three
focused restored misses remain. Known-corner accuracy on recovered partials and
the weaker benefit under older references prevent interpreting unchanged
complete-pair scores as a complete success. No output rule or model is promoted.
