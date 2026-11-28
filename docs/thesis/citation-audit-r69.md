# Citation and explanation update

28 November 2026, Asia/Hong_Kong.

This update accompanies the report's completed-work prose revision and the new corner-conditioned suppression comparison. The earlier [citation audit](citation-audit-r64.md) retains the verification of the existing sources. The [bibliography](bibliography-fyp-i.json) is the current numbered record.

## Applied reference

The report adds RMPE as reference [17]. Fang, Xie, Tai and Lu compare predicted human poses when removing repeated detections. Section 3.2 of the [primary CVF paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Fang_RMPE_Regional_Multi-Person_ICCV_2017_paper.pdf) provides the direct connection to this project.

The cassette experiment uses that idea at a smaller scale. Existing boxes must overlap above IoU 0.60, both outputs must have two accepted corners, and corresponding corner distances must each be at most five original-image pixels before the lower-scored output is removed. This fixed condition uses the cassette detector's existing outputs. It does not reproduce RMPE's fitted similarity function, pose merging or full architecture.

The bibliography cites the verified CVF accepted copy, pages 2334–2343, using its URL. It does not attach a publisher DOI with a different page range. Soft-NMS moves to [18] and IoU-Net to [19]. Their existing publication records remain unchanged.

## How the equations support the implementation

| Equation | Project decision explained in the revised prose |
| --- | --- |
| 1, intersection over union | Compare box extent for evaluation matching, suppression and cross-view association, with the role and threshold stated for each operation. |
| 2, precision and recall | Read recovered fronts together with the extra outputs introduced by a change. |
| 3, maximum of two corner errors | Require both lower-edge endpoints to meet the image-pixel tolerance. The new suppression test reuses the distance calculation between predictions to measure their agreement. |
| 4, horizontal reflection | Return mirrored coordinates to the original photograph and exchange the two corner slots before association and averaging. |
| 5, equal coordinate averaging | Combine associated original and reflected estimates without learning combination weights. |
| 6, coordinate correction | Add the local model's predicted displacement to the detector's starting point after converting the displacement to original-image pixels. |

The six formula blocks are unchanged. Their adjacent explanations state the calculation's role and the reason for using it. The final [report validation](report-validation-fyp-i.json) checks native editable equations, source identity and presentation.

## Research retained in the technical note

The [detection-decision note](research-detection-decisions-r69.md) examines Rank & Sort Loss, Repulsion Loss and Learning Non-Maximum Suppression alongside RMPE. These papers help distinguish score ordering, separation of different objects and removal of repeated predictions. The main report cites the applied RMPE connection and keeps the other conditional choices in the research record.

The primary papers establish their methods and their own benchmark results. The cassette measurements come from the separately archived notebook outputs. The research note distinguishes the loss-only Rank & Sort ablation from its full-system results, separates different-target RepBox terms from same-target duplicates, and records the added training component required by learned suppression.
