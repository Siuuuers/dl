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
| Experimental Results | Section 7; original-model comparison and all three local-model returns |
| Discussions | Section 8; contribution, interpretation, annotation sensitivity and further work |
| Conclusions | Section 9; supported results and adverse findings |
| References | Section 10; external academic/professional sources in IEEE first-citation order |
| Appendices | Section 11; three-reference scoring comparison and training/coordinate settings |

All sections receive decimal numbers and contents entries. Major equations are numbered, centered editable native display mathematics; equation numbers are separate at the right margin. Symmetric borderless layout rows use automatic height and are distinct from the four captioned data tables. The one explanatory figure and every data table have descriptive numbered captions.

The manuscript contains no GitHub links, project-note citations or unexplained experiment codes. It explains project procedures and results directly. The untouched supplied checkpoint remains the common baseline; supplied, earlier fixed and reviewed evaluation labels remain distinct. Changing evaluation labels is explicitly separated from changing training labels.

Physical units are metric; image errors use pixels. The document uses Letter portrait, 25 mm margins, Times New Roman 12-point body text and 1.15 line spacing. Front matter, Introduction, References and Appendices start on fresh pages; the main body flows between sections.

The [validation record](report-validation-fyp-i.json) is authoritative for final page/word counts, heading/reference counts, artifact hashes, contents page map, equation geometry and all-page visual inspection. An institutional template can be applied later.

## Available requirements and downstream scope

On 28 November 2026, the user clarified that the supplied baomai guidance and materials are the available requirements; no separate downstream interface specification has been supplied. The study therefore retains the existing cassette-front detection and two lower-corner estimation task, its annotation rules, and the declared image-space evaluation. Robot positioning remains application motivation, not a verified integration or grasp-success result.

The inspected [baomai README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/README.md), [annotation guidance](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md), and [viewer](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/scripts/view_d405_yolo.py) distinguish annotation availability from model output. Unlocatable annotation points use `0 0 0`; the viewer checks each predicted endpoint independently and draws a connecting line only when both pass. Its raw prediction records still carry two coordinate/score slots. These conventions do not establish a downstream API rule for missing predictions, a compulsory complete pair for every detection, or a physical acceptance tolerance.

Continue the prepared model and training-label comparisons under their frozen protocols. This clarification requires no architecture change, new acceptance stage, or additional interface question before those experiments. Any later downstream specification can define its own output and acceptance requirements explicitly.
