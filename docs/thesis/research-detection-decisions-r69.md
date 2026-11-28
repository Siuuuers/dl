# Choosing between score ranking, instance separation and pose agreement

28 November 2026. This review examines three additional training methods and a smaller inference experiment. **The practical priority is to distinguish a bad score ordering from two responses to one front and from two adjacent fronts.** Each problem requires different evidence. The corner-agreement replay, current-checkpoint proposal collection and full independent numerical verification have completed. The [four-cell notebook execution](evidence/current-proposals-r69/notebook-execution.json) also passed. No new GPU training method is selected.

## 1. What the completed cassette work already establishes

The [R66 duplicate audit](duplicate-audit-r66.md) found 13 formal duplicates for the reviewed-label model under r50. Their overlaps with the retained prediction assigned to the same reference were approximately 0.475–0.700. Their survival is consistent with native suppression above 0.70. Twelve overlap an incomplete-corner reference most strongly. Additional suppression at 0.60 removed 19 extras, including 11 duplicates, but lost one previously matched partial front. Its unchanged complete-pair counts did not reveal that loss.

The [completed R68 union](research-partial-detection-r68.md) provides the opposite tradeoff. Under r67, adding unpaired reflected outputs changed reviewed-model misses from 18 to 10, extras from 50 to 60 and duplicates from 13 to 18. Five-pixel pairs remained 572/583 and ten-pixel pairs remained 583/583. The eight recovered fronts had fewer than two known corners. All three restored tiny focus targets remained missed. This establishes a coverage tradeoff on reused development photographs, not a selected improvement.

Linear Soft-NMS, larger whole-image inference and a trained one-to-one head already have project results. Their history is recorded in [R68](research-partial-detection-r68.md#1-correction-to-the-earlier-research-sequence). They are not new proposals here. Likewise, historical decoded-proposal findings must not be attributed to the newer reviewed-label checkpoint. Its saved post-NMS output cannot establish what existed before filtering and suppression.

Two distinctions guide the new research:

- **Ordering and retention are different.** A better score order can change which competing box survives NMS. It need not remove duplicates whose overlap stays below the suppression threshold and whose scores both remain above the confidence floor.
- **Reference duplication and physical identity are different.** An extra may repeat a front, describe an adjacent omitted fragment or have inaccurate extent. A raw-image identity review is needed before making it a negative training example.

## 2. Rank & Sort Loss: improve ordering without adding an inference head

**External method and result.** Oksuz et al. [1] rank positives above negatives and sort positives by localization quality. A custom gradient procedure handles the ranking objective. Their ATSS ablation without an auxiliary quality head reports 38.7 AP for the baseline, 39.7 with RS alone, 39.8 after score-based weighting and 39.9 after task balancing. The larger Faster R-CNN headline gain combines training changes. Appendix C.6 also reports a change in score scale and raises the COCO inference floor from 0.05 to 0.40 for efficiency. Its ATSS training iteration takes 0.70 s versus 0.44 s in the reported setup. These are external COCO measurements, not cassette predictions.

**Implementation check.** The pinned [official loss](https://github.com/kemaloksuz/RankSortLoss/blob/b41f64df70b1f8e3260e5c04015e0c7b59b339d9/mmdet/models/losses/ranking_losses.py) compares logit differences. The [PAA integration](https://github.com/kemaloksuz/RankSortLoss/blob/b41f64df70b1f8e3260e5c04015e0c7b59b339d9/mmdet/models/dense_heads/rank_based_paa_head.py) uses detached predicted-box IoU as the positive quality, also weights regression by scores and changes task balancing. Its all-negative-batch branch returns zero ranking and sorting losses. Empty images in a mixed batch can still supply negatives. This is a concrete porting concern for our empty-scene training, not evidence that the official implementation crashes.

**Project inference.** The RS classification objective alone is unchanged by a common shift of all logits. Its sigmoid scores are not:

\[
L_{\mathrm{RS}}(z+c\mathbf{1},q)=L_{\mathrm{RS}}(z,q),\qquad
\operatorname{sigmoid}(z+c)\ne\operatorname{sigmoid}(z).
\]

Here z contains classification logits, q contains quality targets and c is one shared offset. The equation explains why improving relative order does not establish the meaning of an absolute confidence cutoff. It is not an invariance claim about the paper's complete score-weighted training recipe.

For a **synthetic illustration**, probabilities 0.20 and 0.10 correspond to logits about −1.386 and −2.197. Adding 2 gives probabilities about 0.649 and 0.451. Their order is unchanged, while both now cross 0.25. These are invented values illustrating the mathematics, not measured cassette scores.

Our native supervision already combines box and available-corner agreement. Replacing it with raw IoU would change the target meaning as well as the loss. A ranking adaptation must state which quality it orders, preserve unknown-corner masking and explicitly handle all-negative batches. It must measure absolute filtering and empty-scene outputs alongside ranking. Better box ordering can still select worse corners.

**Decision.** RS is a conditional option if the current checkpoint repeatedly ranks a geometrically inferior competitor above a usable candidate. It adds no required inference head, but it does require a deliberate training adaptation. It is not a direct one-output-per-front objective and is not a justified response to low scores alone.

## 3. Repulsion Loss: separate different instances during box training

**External method and result.** Wang et al. [2] retain attraction to the assigned box and add two penalties. RepGT penalizes overlap with another reference object using intersection divided by that other reference's area. RepBox penalizes overlap between predicted boxes assigned to different objects. Same-target proposal pairs are excluded from that second penalty. On CityPersons at the same input scale, their reasonable-set log-average miss rate falls from 14.6 to 13.7 with either term and to 13.2 with both. Lower is better. The 10.9 result also uses a larger input scale and is not a loss-only comparison.

**Project inference.** This addresses boxes drifting toward neighboring objects. It does not directly teach the model to discard a second box assigned to the same front. Our visible-fragment rectangles can legitimately overlap. Even two correctly placed rectangles could then incur a repulsion penalty. Minimizing overlap is not equivalent to recovering the specified visible extent.

The useful prerequisite would be a training-target and candidate audit showing repeated drift toward another identifiable front, not merely a count of unmatched outputs. The audit should compare predicted overlap with the overlap already present between the corresponding reference boxes. Missing labels can also turn a real neighbor into an unrepresented object. Neither repulsion term supplies a reference coordinate for an unavailable corner.

**Decision.** Do not make repulsion the next duplicate-removal experiment. Consider it only if neighboring-instance box drift becomes a demonstrated training failure and a controlled formulation preserves legitimate overlap. The checked author paper and [author publication entry](https://www.xloong.wang/) did not provide an author-released implementation link for this work. Third-party ports were not treated as official code.

## 4. Learned NMS: directly learn which response represents an instance

**External method and result.** Hosang, Benenson and Schiele [3] rescore detections jointly from boxes, scores and neighboring relationships. During training, one-to-one matching supplies positives and unmatched responses supply negatives. Its basic model does not use image appearance. On COCO persons minitest, eight-block Gnet raises AP over IoU 0.50:0.95 from 35.5 to 36.7. In the separate multiclass experiment, overall AP rises from 23.5 to 24.3 while person AP falls from 35.5 to 35.1. These different outcomes argue against promising universal gains.

**Implementation check.** The [official network](https://github.com/hosang/gossipnet/blob/ce8837238eb5d486f6a645e64f194f20d2e72863/nms_net/network.py) applies a matching operation to its scores and reference overlaps, then a weighted binary loss. The [repository](https://github.com/hosang/gossipnet/tree/ce8837238eb5d486f6a645e64f194f20d2e72863) uses legacy TensorFlow and a compiled matching operation. It would require a maintained port and an additional inference component.

**Project inference.** This is the most direct of the three training methods for one response per instance. It is also the largest new system commitment. A real but unlabeled front can become an unwanted negative through the matching step. Training needs consistent instance coverage and detector outputs representative of inference. The repeatedly used twenty development photographs are not an independent training set for this module.

Boxes and scores also have an information limit. If a repeated response and two adjacent partial fronts produce indistinguishable geometric inputs, a function of those inputs alone cannot establish which physical explanation is correct. Additional image evidence or reliable landmark relationships may help, but would be an additional design choice. A fixed-box rescorer cannot create geometry absent from its input pool.

**Decision.** Retain learned NMS as a later architecture option. The present evidence does not justify its annotation, training and deployment costs.

## 5. A smaller experiment using the existing corner predictions

**External motivation.** Fang et al.'s RMPE [4] suppresses redundant human poses using corresponding-joint confidence and spatial similarity. Its parameters are optimized on validation data. Table 3(c) reports MPII validation mAP of 75.7 without pose NMS, 79.1 and 79.9 with two previous pose-NMS methods and 80.8 for the full method. This motivates using pose agreement as evidence about redundancy. It does not establish a threshold for two cassette corners.

**Fixed project method.** The [protocol](evidence/corner-suppression-r69/protocol.json) qualifies additional box suppression above 0.60 with agreement of both corresponding accepted predicted corners within five original-image pixels. Each point must have confidence at least 0.50, finite coordinates and a position inside the original image. The rule compares left with left and right with right, using completed saved coordinates with their existing corner averages. Outputs are processed by descending box confidence, with original output index breaking ties. A previously kept same-class output must meet every condition before another output is removed. Survivors retain their coordinates and original relative order. The comparison includes unchanged outputs and the completed unconditional 0.60 rule, without adding the R68 reflected-output union.

The hypothesis is that a neighboring partial front can have an overlapping box while its predicted corners differ. This rule would retain that distinction when the unconditional box rule removes it. It could fail when both responses inherit the same incorrect corner estimates, when a true duplicate has inaccurate corners or when a partial front has no accepted pair. Predicted confidence does not certify physical visibility, and agreement with another prediction is not agreement with an annotation.

This is a small cassette-specific inference rule inspired by pose similarity. It does not reproduce RMPE, implement standard OKS-NMS or create independent observations. Its five-pixel distance is an exploratory design choice borrowed from an existing reporting tolerance, not an independently validated duplicate-identity boundary. It needs no new GPU training.

**Completed result with independent verification.** The [saved-output results](evidence/corner-suppression-r69/results.md) and [full sixty-row matrix](evidence/corner-suppression-r69/matrix.csv) retain all arms. For the reviewed-label model under r67, the rule changes misses/extras/duplicates from **18/50/13 to 18/44/11**. Five-pixel pairs stay 572/583 and ten-pixel pairs stay 583/583. It removes six extras, of which two meet the formal duplicate definition, and preserves the partial front lost by unconditional additional NMS at 0.60. The independent audit reproduced all 60 matrix rows, 1,200 image scores, 64 target-change records, 520 reference-indexed removals and 117 overlap pairs. The [target ledger](evidence/corner-suppression-r69/target-changes.json.gz) shows no reviewed-label target gains, losses, rematches or known-corner changes under any of the four references:

| Reference | Misses, unchanged → qualified | Extras, unchanged → qualified | Duplicate subset, unchanged → qualified |
| --- | ---: | ---: | ---: |
| r50 | 15 → 15 | 51 → 45 | 13 → 11 |
| r67 | 18 → 18 | 50 → 44 | 13 → 11 |
| Supplied664 | 50 → 50 | 76 → 70 | 12 → 10 |
| Frozen666 | 38 → 38 | 62 → 56 | 14 → 12 |

This stability is specific to the reviewed-label arm. The supplied-label arm loses `new_000009:R012`, a one-known-corner front whose previously returned point had error 1.622 px. Its r67 misses change from 50 to 51 and extras from 84 to 81 while duplicates remain six. The historical, native and QFL arms preserve their r67 miss counts and remove three, four and six extras, respectively. Their duplicate counts change by zero, zero and one. Thus preservation of reference coverage depends on the checkpoint.

The four references score the same photographs and outputs. Their agreement here is sensitivity evidence, not four independent tests. Removing six unsupported outputs also does not establish six physical false positives. The reviewed-model result warrants further validation of this small rule. The checkpoint counterexample argues against applying it automatically across models, and no default is changed.

## 6. What the current checkpoint's decoded proposals establish

The separate [decoded-proposal diagnostic](evidence/current-proposals-r69/protocol.json) collected all 19,320 raw candidates for both views of every development photograph from the PR158 reviewed-label checkpoint. This was fresh FP32 CPU inference with frozen parameters. It preserved the existing confidence floor above 0.25, NMS above 0.70 and evaluation matching at IoU at least 0.50. The full tensors, native retained indices, suppression witnesses and coordinate conversions were checked before interpreting stage losses. The [readable results](evidence/current-proposals-r69/results.md) and [stage coverage](evidence/current-proposals-r69/stage-coverage.json.gz) preserve the measurements. No optimizer or annotation change was involved.

Under r67, the original view's 18 misses comprise **12 confidence-gate losses, five targets without any decoded box reaching IoU 0.50 and one native-NMS loss**. The reflected view's twelve misses comprise ten confidence losses, one geometry failure and one native-NMS loss. Reference support equals maximum one-to-one matching at every measured stage:

| Fixed candidate pool | Original view, maximum matches | Reflected view, maximum matches |
| --- | ---: | ---: |
| All decoded boxes | 653/658 | 657/658 |
| Above confidence 0.25 | 641/658 | 647/658 |
| After native NMS | 640/658 | 646/658 |
| After postclip processing | 640/658 | 646/658 |

Consequently, changing only NMS on the same above-confidence original-view boxes can add **at most one** reference match. It cannot restore the twelve targets filtered earlier or create eligible geometry for the other five. This is an annotation-based upper bound on a fixed pool. It does not show that a learned ranking rule can achieve it without losses.

The current original-view focus cases are:

| R67 target | Best raw box IoU | Highest score among IoU-eligible boxes | First missing stage |
| --- | ---: | ---: | --- |
| `new_000009:R055`, right-jaw fragment | 0.595507 | 0.01454296 | Confidence |
| `new_000009:R056`, lowest left band | 0.446675 | No eligible box | Geometry |
| `new_000053:R022`, top wedge | 0.589806 | 0.00016235 | Confidence |

The best-overlap box and the highest-scoring eligible box are separate diagnostic selections. In the reflected view, all three have eligible geometry but remain below the confidence floor. Their highest eligible scores are 0.00772521, 0.08451592 and 0.00014884, respectively. Therefore neither original-view NMS ranking nor the completed two-corner suppression rule solves these three misses. The stage finding locates the computational loss. It does not identify whether appearance learning, target construction or effective training exposure caused it.

### Runtime differences must be described at the right level

The fresh CPU and archived L4 outputs have the same reviewed-reference miss, extra and five-/ten-pixel complete-pair counts. The initial target-change ledger nevertheless contained 666 records across output stages and references. Independent decomposition found that all 666 are output-index changes. Detection status, known-corner acceptance and threshold outcomes remain unchanged. The corresponding matched boxes have IoU at least 0.999541, with maximum paired coordinate difference 0.017609 px.

A separate extra on `new_000087` changes from other-extra to duplicate. Its overlap with r67 `R048` changes from 0.432788 to 0.500214, crossing the duplicate definition's 0.50 boundary. Thus baseline duplicates number 13 in the archive and 14 on the fresh CPU run. Fresh native lineage shows anchor 9210 suppressing 9211, and 9211 resembles the archived box at IoU 0.999562. The archive does not retain raw anchor identity, so this is insufficient to attribute the difference to a specific CPU/GPU mechanism.

The frozen corner rule removes six outputs in each runtime. Its retained-index lists agree. Extras fall from 50 to 44 in both, while duplicates change from 13 to 11 in the archive and from 14 to 12 in the CPU run. Neither runtime has a baseline-to-rule target or known-corner change. This supports continuity of the rule's observed effect without claiming prediction byte identity or that hardware affects only speed. The [runtime-continuity ledger](evidence/current-proposals-r69/qualified-rule-runtime-continuity.json.gz) preserves that comparison.

## 7. Decision from the completed evidence

The current original-view bound gives little scope for a suppression-rank intervention to recover additional detections from boxes already above the confidence floor. RS remains a conditional training idea, not the next justified GPU run. Repulsion and learned NMS require stronger evidence of their respective failure mechanisms. The completed corner rule is the smaller measured candidate for reducing unsupported outputs on the reviewed checkpoint, with its demonstrated checkpoint dependence and incomplete physical adjudication retained.

For duplicate-sensitive rules, retain a pair ledger with box overlap, corresponding predicted-point distances, accepted-point flags and the raw-image identity assessment. Unknown or held regions remain unresolved. Do not use evaluation annotations to decide which records survive at inference. Do not tune a sequence of thresholds until these same photographs look better.

Measure lost and recovered targets separately for zero, one and two known corners. Keep all extras, duplicates, empty-scene outputs and complete-pair results. Report available-corner errors for recovered partial fronts, since the complete-pair denominator excludes them. Recheck older frozen references as sensitivity analyses and preserve the established evaluation thresholds. Independent capture groups remain necessary for a generalization claim.

The corner-qualified rule retains the reviewed-reference matches while removing fewer extras and duplicates than unconditional suppression. This is a measured coverage/cleanup tradeoff, not a solution to all detection errors. Further model complexity should address the established score or geometry failures and earn its cost on independent captures. The availability of another published method is not sufficient reason for another training run.

### Completed reproduction

The [CPU notebook](../../notebooks/baomai_current_proposals_r69.ipynb) ran all four actual code cells unchanged in one local IPython namespace from source [`41d0e6937c695377373cc55071ffb780345870ee`](https://github.com/Siuuuers/dl/tree/41d0e6937c695377373cc55071ffb780345870ee). Its [receipt](evidence/current-proposals-r69/notebook-execution.json) records all cells passing, forty raw forward views, the forty-row proposal/runtime matrix and the sixty-row saved-output companion. The frozen inputs were preseeded locally and checked by the actual cells. This verifies the local execution, not hosted Colab, remote downloads or the Colab download interface. No training, optimizer update, reference amendment or default promotion occurred.

## Primary references and inspected source revisions

The page ranges below identify the linked CVF proceedings copies. Publisher pagination is not combined with a different copy's DOI.

1. K. Oksuz, B. C. Cam, E. Akbas and S. Kalkan, “Rank & Sort Loss for Object Detection and Instance Segmentation,” in *Proc. IEEE/CVF Int. Conf. Computer Vision*, 2021, pp. 3009–3018. [CVF paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.pdf). [Versioned author manuscript](https://arxiv.org/abs/2107.11669v2), including the supplement. The loss-only comparison is Table 9. Inference-floor and runtime qualifications are Appendix C.6. Official repository revision `b41f64df70b1f8e3260e5c04015e0c7b59b339d9`. Inspected loss-file Git blob `ef67f0c0fb5227e85d1bc1a35558ee9f1d2bfdc3` and PAA-head blob `4a5c6e46d235b4253846efbdc6b9ee5d06a4cfad`.

2. X. Wang, T. Xiao, Y. Jiang, S. Shao, J. Sun and C. Shen, “Repulsion Loss: Detecting Pedestrians in a Crowd,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition*, 2018, pp. 7774–7783. [CVF record](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.html). [Versioned author manuscript](https://arxiv.org/abs/1711.07752v2) and [full text](https://arxiv.org/html/1711.07752v2). Section 4 defines the two penalties. Tables 1–3 separate terms, weights and input scale.

3. J. Hosang, R. Benenson and B. Schiele, “Learning Non-Maximum Suppression,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition*, 2017, pp. 4507–4515. [CVF record](https://openaccess.thecvf.com/content_cvpr_2017/html/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.html). [Versioned author manuscript](https://arxiv.org/abs/1705.02950v2), including the supplement. Table 1 concerns person-specific training. The multiclass per-category result is supplementary Table 2. Official repository revision `ce8837238eb5d486f6a645e64f194f20d2e72863`. Inspected network Git blob `5129f4c16d971bc33b9c820a43e194b42650013b` and matching-operation blob `28f5e8f76ac17ada238e746ed5cdd3456c6bd52c`.

4. H.-S. Fang, S. Xie, Y.-W. Tai and C. Lu, “RMPE: Regional Multi-Person Pose Estimation,” in *Proc. IEEE Int. Conf. Computer Vision*, 2017, pp. 2334–2343. [CVF paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Fang_RMPE_Regional_Multi-Person_ICCV_2017_paper.pdf). [Versioned author manuscript](https://arxiv.org/abs/1612.00137v5). Section 3.2 defines parametric pose NMS and Table 3(c) isolates its replacement or removal. The cassette rule borrows the idea of comparing corresponding predicted points, not the paper's model, similarity function or learned parameters.
