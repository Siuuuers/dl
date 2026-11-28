# Final Year Project I report requirements

The report is for undergraduate Final Year Project I in the Department of Electronic Engineering, followed by Project II next semester. It introduces the vision task to an unfamiliar reader. Acknowledgements are omitted at the user's request. An institution-specific cover template, word limit and page limit remain unspecified.

| Requirement | Current report |
| --- | --- |
| Abstract, 150–300 words | Section 1. Objectives, methods, results and evidence limits. Exact length is recorded in validation. |
| Acknowledgements | Omitted as requested |
| Table of Contents | Section 2. Every section and subsection, decimal numbering and verified Word page numbers. |
| Introduction | Section 3. Problem, aim and relevant published work. |
| Background | Section 4. Annotations, detection matching, corner error and confidence. |
| Methodology and Implementation | Section 5. Data review, supplied and proposed methods, three local models and evaluation. |
| Simulation Results | Section 6. Ideal-heatmap numerical check. |
| Experimental Results | Section 7. Model comparisons, paired training experiments, output restrictions and reference-inclusion effects. |
| Discussions | Section 8. Contribution, interpretation, annotation sensitivity and further work. |
| Conclusions | Section 9. Supported results and adverse findings. |
| References | Section 10. External academic and professional sources in IEEE first-citation order. |
| Appendices | Section 11. Reference sensitivity and training and coordinate settings. |

All sections receive decimal numbers and contents entries. Major equations are centered, numbered and editable native mathematics, with numbers at the right margin. Their symmetric borderless layout rows use automatic height and remain distinct from the captioned data tables. The one explanatory figure and every data table have descriptive numbered captions. Table widths, padding and type sizes must keep complete headers, including “Photographs” and “Duplicates,” visible without clipping.

## Writing and citation quality

Use clear, connected prose for an undergraduate Electronic Engineering reader unfamiliar with computer vision. Explain what each method receives, what it changes and what is measured before listing settings. Define necessary terms near first use. Keep the explanation focused on this task rather than adding a general neural-network tutorial.

Keep Sections 4, 5 and 7 concise. Section 4 defines the measurements, Section 5 explains the implemented choices and comparisons, and Section 7 presents each result and its implication. Use “earlier annotation versions” for older evaluation labels. Explain decoding as converting a heatmap to a coordinate, with invalid-point handling in the appendix. Keep file identity, archived-record assurances and detailed replay checks in the source map and evidence records. Preserve the relevant thresholds, comparison controls, denominators and adverse results when shortening prose.

Explain each displayed equation immediately through its role in this project: what it calculates, how its result is used and why the choice supports detection or corner localization. Keep the six equation blocks intact. Use the implemented coordinate conventions and thresholds, and avoid detached textbook derivations.

Use no semicolons. Revise sentence structure rather than replacing punctuation mechanically. Lead with completed work, why a method was chosen and what its result means. Describe the completed annotation review and its accepted inclusion decisions directly. Avoid repetitive statements about work being unmeasured, outstanding or subject to future review. Keep the material experimental conditions together in one concise discussion paragraph: the 20 development photographs, one paired seed per controlled training experiment and results under named reference versions. Preserve the difference between confidence-based selection and measured coordinate accuracy without adding a recurring calibration caveat.

Explain project procedures directly, without GitHub links, project-note citations or unexplained experiment codes in the manuscript. The unchanged supplied checkpoint remains the main benchmark. Keep the adapted single-view control visible when assessing a corner method's added contribution. Distinguish supplied, earlier fixed and reviewed evaluation references, and separate changing evaluation labels from changing training labels.

The user's 28 November citation clarification requires a source's idea, mechanism or finding to be explained alongside the project's adaptation and relevant limits. Use IEEE bracketed numbers in first-citation order, reuse numbers consistently and place them before sentence punctuation. Each number identifies one complete work with verified authors, title, venue, year, available pages and DOI or online source and access date. The supplied [IEEE citation guide](http://www.ijssst.info/info/IEEE-Citation-StyleGuide.pdf) could not be fetched. The accessible [official IEEE Reference Guide](https://docs.google.com/document/d/e/2PACX-1vSBoV_hTIURPZKtQkxI6bAONJtjyDLxQIUvCHi-XPzD8NKLcxRfZrpvQLFrjWZ3M6FFR_Obau3R-t63/pub), version 3.28.2025, provides the formatting rules. The [citation audit](citation-audit-r64.md) records source scope and metadata decisions.

Physical units are metric, and image errors use pixels. Use Letter portrait, 25 mm margins, Times New Roman 12-point body text and 1.15 line spacing. Front matter, Introduction, References and Appendices start on fresh pages. The main body flows between sections.

The [validation record](report-validation-fyp-i.json) records final page and word counts, headings, references, artifact hashes, contents page map, equation geometry and all-page visual inspection. An institutional template can be applied later.

## Available requirements and downstream scope

On 28 November 2026, the user identified the supplied baomai guidance and materials as the available requirements. Vision outputs will locate cassettes for robotic arms, with arm control handled in later integration. Physical acceptance tolerance and the downstream message format remain to be specified. The present evaluation measures cassette-front detection and two lower-corner coordinates in image space. Robotic positioning needs a separate physical evaluation.

The inspected [baomai README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/README.md), [annotation guidance](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md), and [viewer](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/scripts/view_d405_yolo.py) distinguish annotation availability from predicted confidence. Unlocatable annotation points use `0 0 0`. The viewer checks each endpoint independently and connects the points only when both pass, while raw predictions retain two coordinate and score slots. A downstream specification would still need to define how unavailable predictions are represented and accepted.

The completed model and training-label comparisons retain their frozen protocols. The clarified application requires no change to those experiments. Later integration can specify its own output and acceptance requirements.

## Reporting hardware and interpreting results

The user prioritizes recognition findings and simplicity, with no latency or hardware constraint. Record GPU models and relevant software and precision settings briefly in the reproducibility appendix. Use matched arms for scientific comparisons. Equal seeds alone do not establish identical training across platforms. CPU model details become relevant if a timing benchmark is added. Keep total extras and their duplicate subset visible alongside improvements in other measures.

The paired experiments use the same seed and final-epoch rule within each pair. Record their experimental conditions once in the discussion and retain the named scoring reference with each result. When annotations change, retain the old scoring reference and provide a PDF containing only changed photographs with exact amendments highlighted. The photographs and target definition must support each edit.

The partial-front review restores four identifiable fronts in a separate 658-front reference. Preserve the 654-front results and report the reference-only replay separately in Section 7.6. Front inclusion remains independent of corner availability. The completed review kept three other regions unannotated because the visible material could not be confidently assigned to separate fronts. Those regions retain ordinary evaluation rules. The amendment PDF contains the three changed photographs with original, before and after crops. Historical coordinates and availability flags remain unchanged.
