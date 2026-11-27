# R50: project direction from the lecture and the available evidence

27 November 2026, Asia/Hong_Kong.

**Update after PR143:** the inference question proposed below has now completed.
The [R51 diagnosis](front-diagnosis-r51.md) confirms exact fresh-baseline replay,
no miss recovery from Soft-NMS, and ten confidence / three decoded-geometry / two
NMS cases. A single lower-confidence probe recovers six targets at the cost of
74 additional unmatched outputs. The [R51 notebook](partial-classification-r51.md)
therefore isolates partial-target classification weighting on unchanged training
inputs. This is a distinct ablation after the earlier joint box/classification
and box-only trials, with no guaranteed improvement. Sections below preserve
the lecture interpretation and the pre-return R50 decision context; R49/R50
inference does not need to be repeated.

**The present assignment is to annotate the cassette data and train a model.**
The user says the thesis topic has not been settled. Professor LI Hongsheng's
uploaded detection lecture is background teaching material, not a thesis
specification. The earlier [R48 thesis plan](thesis-plan-r48.md) remains a proposal
for discussion. No new capture session is available now, and useful work on the
existing dataset can continue.

## 1. What has actually been completed

[PR137](additional-front-pilot-r47-results.md) is the latest returned training
experiment. Both arms completed 30 epochs. Its reference uses 138 photographs;
the candidate uses 142, including five supplementary `frame_*` photographs.
Thus the frame photographs have now been used for training. The candidate adds
four photographs and 172 fronts relative to its paired reference.

[PR139](https://github.com/Siuuuers/dl/pull/139) prepares an inference diagnosis and
Soft-NMS experiment. [PR140](https://github.com/Siuuuers/dl/pull/140) repairs the
missing `results/review/front-review-20261126-r38/row-maps.json` checkout dependency.
Neither contains a completed R49 diagnosis archive. In PR140 the sparse-checkout
repair runs only when creating a new repository, so a previously failed checkout
still skips that repair. [R50](diagnosis-r50.md) reapplies the required paths on
every setup run, checks the exact source and all input files, and audits the real
archive and originals before installing the inference runtime.

The [R50 annotation change](front-review-r50.md) is one development-row bottom
boundary. It changes no training input and does not itself teach the model.
The richer [error report](front-error-review-r50.md) explains all original 119
R38 review cases while keeping current R50 score status separate.

## 2. Read the lecture as a guide to questions

Source: the user-supplied `detection.pdf`, *ELEG5760: Machine Learning for
Multimedia Applications - Object Detection in Computer Vision*, Prof. LI
Hongsheng, November 2025. References below use the 39-page PDF's physical page
numbers, starting at 1. Source SHA-256:
`95eff36f47bde94d8029d46f012a55ae48a6fc32563799e1b858f9a66ef037eb`.
The lecture itself, full extracted text and slide images are not added to Git.

| Lecture topic | Pages | Question for this project |
| --- | --- | --- |
| Detection target | 3 | Which visible physical front does one row describe? The two lower corners are an additional project-specific pose target. |
| Intersection over union, IoU | 5 | Is an unmatched box on the wrong object, or on the correct object with the wrong extent? |
| NMS, including Soft-NMS | 6-7 | Did suppression remove a valid nearby cassette, or leave repeated hypotheses for one cassette? |
| Precision, recall and AP | 9-11 | What is the tradeoff between misses and unmatched outputs as confidence changes? Counts at one cutoff are not AP. |
| Box regression | 13-14 | Do consistent physical training targets help the model predict the correct visible extent? |
| Feature pyramids and scale | 21-23 | Are the affected fronts poorly represented at their actual resized size? |
| Hard examples and focal loss | 25, 28-29 | Are specific verified difficult examples underrepresented, and is a loss change justified by evidence? |
| GIoU and DETR assignment | 33-36 | Which part belongs to training, and which part belongs to our evaluation matcher? |

The original YOLO grid example on page 27 is historical. Its 7-by-7 grid and two
boxes per cell do not describe the current pinned YOLO-pose model. Focal loss is
a training loss, not a contrast filter. DETR's training assignment is not the
same as matching retained predictions to references during evaluation. None of
these lecture topics requires a new architecture for this project.

The original [Soft-NMS paper](https://arxiv.org/abs/1704.04503) motivates one small
inference comparison: decay overlapping scores instead of immediately deleting
every box above the overlap cutoff. It does not guarantee improvement on these
cassette scenes. The [Feature Pyramid Networks paper](https://arxiv.org/abs/1612.03144)
explains multiscale feature fusion; simply enlarging the input is a different
intervention. These papers explain mechanisms, while our returned experiments
must establish project-specific benefits.

## 3. The strongest lead is partially visible fronts

Under the unchanged R38 reference, the PR137 candidate misses:

| Independently locatable corners | Targets | Missed |
| --- | ---: | ---: |
| Two | 583 | 0 |
| One | 54 | 12 |
| Zero | 17 | 3 |

The added training fronts are 164 with two corners, eight with one, and none
with zero. The addition therefore supplies mostly the two-corner group, which
has no misses under this reference.
This is evidence for prioritizing partial-front diagnosis, not proof that data
imbalance caused every miss. The visual review shows occlusion or image
truncation in the missed cases; the saved final predictions cannot by themselves
identify which internal pipeline stage lost each target.

An unmatched output is also not automatically background. Some are repeated
boxes, some extend from a real fragment across the gripper, some are keyboard or
rack responses, and some remain ambiguous. Three outputs in `new_000009` cover a
fragment explicitly excluded at the user's direction in R38. Its earlier row52
was recorded as a real visible partial front, not established background. Preserve
that exclusion and record the reference-policy effect; do not add one row for
each output or use them as verified hard negatives.

## 4. Why a small boundary error can matter

IoU compares two rectangles' intersection with their union. For equal-width,
equal-height boxes of height `h`, displaced vertically by `d` with `0 <= d < h`,
their IoU is `(h-d)/(h+d)`. This simplified calculation isolates vertical error;
real cases can also differ in width and ownership.

A height of eight pixels and a three-pixel displacement gives `5/11`, about
0.455. That fails the current 0.50 evaluation match despite covering the same
small physical front. The appropriate response might be better localization or
a clarified reference boundary; raising confidence cannot move a box corner.
Likewise, removing duplicate outputs cannot create an absent decoded proposal.

The physical definition remains the visible front, including its own molded rim
and shaded material. A neighbor, rack or gripper must not define its outer bounds.
A rectangle enclosing the visible parts can contain some occluder pixels between
those extrema. Missing corners stay unavailable. Small size is not permission to
borrow another cassette's shadow to enlarge the target.

## 5. The next notebook answers a limited, useful question

The [four-cell R50 notebook](../notebooks/baomai_r50_diagnosis.ipynb) uses the exact
PR137 candidate checkpoint and 20 verified original development images. It first
checks source and data on CPU. It then traces decoded proposals through the
confidence gate, ordinary NMS, postclip suppression and final assignment.

One prespecified comparison runs ordinary hard NMS versus linear Soft-NMS on the
same freshly decoded tensors: overlap above 0.70 decays score by `1-IoU`, with the
same strict confidence cutoff 0.13, cap 300 and postclip filter. Corners remain
attached to their original proposal. The frozen666 primary reference, historical
663, R38 and R50 proposal are scored separately. Saved/fresh baseline differences
are reported rather than attributed to the filter.

| Observed stage in a future completed trace | Smallest justified next investigation |
| --- | --- |
| No decoded box reaches matching IoU | Review visible extent, training supervision and scale representation. NMS cannot recover a box the model did not propose. |
| Suitable geometry exists below confidence | Study partial-front confidence and examples; assess any threshold change together with added unmatched outputs. |
| Suitable proposal is removed by NMS | Inspect the fixed Soft-NMS comparison and whether nearby true cassettes are preserved. |
| Suitable proposal is removed after clipping | Audit that suppression pair and its physical identities. |
| Suitable proposal survives but is unmatched | Inspect one-to-one assignment competition, reference membership and box identity. |

This inference notebook does not replace the professor's training assignment.
Training has already been performed; diagnosis makes the next training change
specific enough to learn from. There is no need to rerun identical supervision
because a development annotation changed.

## 6. Existing data still offer a training path

The effective R37 core used by PR137 contains 137 photographs and 3,784 fronts.
The [source audit and prior-trial context](../results/review/front-review-r50-20261127/training-next-step.md)
checks all effective label hashes and finds:

| Core training selection | Photographs | Fronts | Two / one / zero corners |
| --- | ---: | ---: | --- |
| Accepted reviewed supervision | 105 | 3,456 | 2,866 / 454 / 136 |
| Held `new_000015`, original fallback | 1 | 59 | 55 / 3 / 1 |
| Old-series treatment exclusions, original fallback | 31 | 269 | 269 / 0 / 0 |

The 31 old photographs still train; the label treatment is excluded. They are
not 31 active physical holds. Only `new_000015`, `old_000010` and `old_000042`
currently differ between effective supervision and the corresponding full R37
review. The two additional proposed zero-corner rows in held `new_000015` do not
authorize bypassing its whole-photo completeness hold.

The next small training study, if the trace and original-image review justify it,
is a newly supported correction to eligible training labels under unchanged
native loss, images, initialization and budget. Review all objects in affected
training photographs so unannotated cassettes are not silently treated as
background. Keep these 20 repeatedly reviewed development images out of that
training amendment.

There are also existing supplementary photographs to finish reviewing. R48 made
`frame_0001` a sixth complete proposal: it would add 43 fronts, 41 with two corners
and two with one, to the 142-photo candidate. That is a possible separate
data-addition experiment, but supplies little direct evidence for the zero-corner
failure group. More images and more partial-front supervision are different
experimental changes.

Do not repeat an earlier intervention under a new name. R29's partial-image
sampling comparison already completed in PR107, although this bounded source
audit did not establish its numerical result. The R32 notebook records R31's
partial-box factor-two trial: frozen666 misses 36 to 35 and unmatched 108 to111,
while R31 strict pairs fall 572 to570. The R33 notebook records the seed47
replication: frozen misses 32 to33, unmatched 105 to100 and strict pairs 524 to522;
it failed the recorded screen. These are historical trial results under their
own references, not pooled current-model evidence.

Historical gripper/fragment annotation mistakes at `new_000034:43` and
`new_000040:34,35` were already corrected in R34 and inherited by R37. They must
not be presented as newly discovered unfinished corrections. The source audit
does not establish that any particular current training row is physically wrong.

## 7. A useful project record while the thesis remains open

A provisional study question is: **Within the available dataset, which verified
annotation and inference or training changes improve visible-front detection and
corner localization under a fixed evaluation protocol?** This is a proposal for
discussion with the professor, not a settled thesis title or a claim of novelty.

The immediate record can contain the target definition, versioned annotations,
actual training inputs, returned model comparisons, complete error taxonomy and
one controlled follow-up. Negative results and unresolved ownership cases belong
in that record as evidence rather than being hidden by changing the reference.

These photographs have repeatedly guided development. Results on them are
development evidence. A random reshuffle now cannot restore an untouched test.
When new capture sessions eventually arrive, their unseen-session performance
can extend the study; their absence does not stop today's annotation, diagnosis
or controlled training work.

The highest-value future questions for the professor are the required thesis
deliverables, whether the two visible corners are part of the assessed task and
their required accuracy, and whether deployment imposes a latency constraint.
Those answers may change later architecture or evaluation choices. None blocks
this review or the next notebook.
