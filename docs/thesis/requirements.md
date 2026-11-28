# Final Year Project I report requirements

The report is for undergraduate Final Year Project I in the Department of Electronic Engineering, followed by Project II next semester. It explains the vision task for an unfamiliar reader. Acknowledgements are omitted at the user's request. No institution-specific cover template, word limit or page limit has been supplied.

| Requirement | Current report |
| --- | --- |
| Abstract, 150–300 words | Section 1; objectives, methods, results and evidence limits; exact length verified in the validation record |
| Acknowledgements | Omitted as requested |
| Table of Contents | Section 2; every section and subsection, decimal numbering and verified Word page numbers |
| Introduction | Section 3; problem, aim and relevant published work |
| Background | Section 4; annotations, detection matching, corner error and confidence |
| Methodology and Implementation | Section 5; data review, original/proposed methods, three local models and common evaluation |
| Simulation Results | Section 6; ideal-heatmap numerical check |
| Experimental Results | Section 7; original-model comparison, three local-model returns and paired classification-loss and training-label comparisons |
| Discussions | Section 8; contribution, interpretation, annotation sensitivity and further work |
| Conclusions | Section 9; supported results and adverse findings |
| References | Section 10; external academic/professional sources in IEEE first-citation order |
| Appendices | Section 11; three-reference scoring comparison and training/coordinate settings |

All sections receive decimal numbers and contents entries. Major equations are numbered, centered editable native display mathematics; equation numbers are separate at the right margin. Symmetric borderless layout rows use automatic height and are distinct from the nine captioned data tables. The one explanatory figure and every data table have descriptive numbered captions.

The manuscript contains no GitHub links, project-note citations or unexplained experiment codes. It explains project procedures and results directly. The untouched supplied checkpoint remains the common baseline; supplied, earlier fixed and reviewed evaluation labels remain distinct. Changing evaluation labels is explicitly separated from changing training labels.

The reader is an undergraduate Electronic Engineering reader who is unfamiliar with this vision specialty. Explain what a method receives, what it changes and which measurement it affects before listing implementation settings. Define necessary terms at first use without adding a general neural-network tutorial. The adapted single-view control must remain visible where the added contribution of a corner method is discussed; it does not replace the supplied model as the main benchmark.

The user's 28 November citation clarification requires each citation to support a stated idea, mechanism or finding, with the project's adaptation and any important limit explained nearby. A link alone is not a literature explanation. Use IEEE bracketed numbers in first-citation order, reuse each number consistently, place citations before sentence punctuation and maintain one complete bibliographic work per number. Include verified authors, title, venue, year, pages where available, DOI or an appropriate online source and access date. The supplied [IEEE citation guide](http://www.ijssst.info/info/IEEE-Citation-StyleGuide.pdf) could not be fetched; the accessible current [official IEEE Reference Guide](https://docs.google.com/document/d/e/2PACX-1vSBoV_hTIURPZKtQkxI6bAONJtjyDLxQIUvCHi-XPzD8NKLcxRfZrpvQLFrjWZ3M6FFR_Obau3R-t63/pub), version 3.28.2025, supplies the formatting rules. The [citation audit](citation-audit-r64.md) records source scope and metadata decisions.

Physical units are metric; image errors use pixels. The document uses Letter portrait, 25 mm margins, Times New Roman 12-point body text and 1.15 line spacing. Front matter, Introduction, References and Appendices start on fresh pages; the main body flows between sections.

The [validation record](report-validation-fyp-i.json) is authoritative for final page/word counts, heading/reference counts, artifact hashes, contents page map, equation geometry and all-page visual inspection. An institutional template can be applied later.

## Available requirements and downstream scope

On 28 November 2026, the user clarified that the supplied baomai guidance and materials are the available requirements. The outputs will be sent to robotic arms to locate cassettes; arm control and integration are outside this project. No separate downstream message format or physical acceptance tolerance has been supplied. The study retains cassette-front detection and two lower-corner estimation, its annotation rules, and the declared image-space evaluation. Robot positioning is the intended application; it has not been verified through this vision study.

The inspected [baomai README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/README.md), [annotation guidance](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md), and [viewer](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/scripts/view_d405_yolo.py) distinguish annotation availability from model output. Unlocatable annotation points use `0 0 0`; the viewer checks each predicted endpoint independently and draws a connecting line only when both pass. Its raw prediction records still carry two coordinate/score slots. These conventions do not establish a downstream API rule for missing predictions, a compulsory complete pair for every detection, or a physical acceptance tolerance.

The completed model and training-label comparisons retain their frozen protocols. This clarification required no architecture change, new acceptance stage, or additional interface question before those experiments. Any later downstream specification can define its own output and acceptance requirements explicitly.

## Reporting hardware and new result interpretation

The user prioritizes recognition discoveries and a simple report, with no latency or hardware constraint. Record the GPU model and relevant software/precision settings briefly in the reproducibility appendix. The main scientific comparisons are within matched arms, not cross-GPU performance claims. Equal seeds do not establish identical training across platforms. CPU model details are not needed for an unclaimed timing benchmark. Keep total extras and their duplicate subset visible when a candidate improves other measurements.

The completed paired returns use the same seed and final-epoch rule within each experiment. Their single-seed, development-only scope and dependence on evaluation annotations must remain explicit. If an annotation is amended, retain the old reference for scoring and create a changed-photo-only PDF with the exact amended details highlighted. No model score alone justifies a label change.

The completed partial-front review restores four identifiable visible fronts in a separate 658-front reference. Preserve the original 654-front results and report the reference-only replay separately in Section 7.6. The existing policy continues to define front inclusion independently from corner availability. Three unresolved fragments remain held, without being declared background or evaluator ignore regions. The amendment PDF contains only the three changed photographs, with original, before and after crops; inherited coordinates and visibility flags are unchanged.
