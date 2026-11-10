# Reviewed validation reference: 2026-11-10

Reference ID: **`baomai-visible-front-2026-11-10`**. This is the assistant-reviewed, user-authorized bounded amendment of the original 20-image validation reference. Use the [completed report](../../../docs/reference-amendment-2026-11-10.md) for decisions, measurements and remaining issues.

The complete label set is in [`labels/val/`](labels/val/). The [manifest](manifest.json) binds every original image and label hash, revised label hash, original-to-revised row mapping, exact before/after text, review reason and all 33 queue dispositions. Original row numbers in the report always mean the frozen source file; deletions can shift rows in the revised files.

| Change | Count |
| --- | ---: |
| Label files with accepted changes | 11 of 20 |
| Retained rows modified | 13 |
| Source rows removed | 4: two invalid instances and two redundant duplicate rows |
| Previously omitted instances added | 1 |
| Persistent queue rows amended | 14 of 33 |
| Persistent queue rows retained with unresolved issues | 19 of 33 |
| Revised instances / locatable endpoints | 661 / 1,228 |

Every existing visible-corner coordinate is preserved. Consolidating the two duplicate pairs transfers each supported right-corner triplet to the retained instance. The added cassette has one newly marked right corner; its left corner stays missing.

This reference is a bounded correction, not a complete policy migration. Unmodified labels and retained box edges can contain legacy defects. Keep the unresolved records with the data; never convert them into empty negatives. These validation images remain development evidence and must not be added to training.

## Open the reviewed labels in the existing HTML editor

Use the existing [cassette labeler](../../../tools/cassette_labeler/README.md). It already supports this exact 11-field YOLO-Pose format; no new editor is needed.

From the repository root, first hydrate the original validation PNGs with Git LFS. Then create a **new working directory outside the checkout**. This command verifies the original images/labels, copies the reviewed labels and the 20 source PNGs, and refuses an existing destination:

```bash
python - <<'PY'
from pathlib import Path
import shutil
import sys
sys.path.insert(0, "scripts")
from evaluate_pose import preflight_dataset

checked = preflight_dataset(Path.cwd(), require_weights=False)
destination = Path("../baomai-reference-review-20261110")
destination.mkdir()
shutil.copytree("data/references/visible-front-2026-11-10/labels", destination / "labels")
(destination / "images/val").mkdir(parents=True)
for record in checked["records"]:
    shutil.copy2(record["image_path"], destination / "images/val" / (record["id"] + ".png"))
print(destination.resolve())
PY
```

Open `tools/cassette_labeler/index.html` in desktop Chrome or Edge. Choose **Open data folder**, select the printed working directory, and use **Show annotations** to compare marks with the original pixels. **Edit selected row**, **Draw / replace box**, **Left missing**, **Right missing**, **Save label**, and **Undo** operate as documented in the editor guide. Further edits belong to a new reference version with updated review evidence and hashes.

The correction work used original-image coordinate review and direct label writes. This session's native browser input service prevented completing an editor load/save walkthrough; no successful GUI save round-trip is claimed.

## Replay the stored predictions

Extract `results/evaluation/baomai-box-repeatability-a100-001/evaluation.zip` into a new directory. With all original validation PNGs hydrated, run from the repository root:

```bash
python scripts/rescore_reference_amendment.py \
  --root . \
  --reference-dir data/references/visible-front-2026-11-10 \
  --evidence /path/to/extracted-evaluation \
  --output /path/to/new-reference-replay.json
```

The standard-library-only script first verifies the saved predictions against their original dataset and checkpoint identities. It then compares the original and revised references using the same five models, three stored confidence thresholds, matcher and endpoint rules. The output path must be new. No model weights are loaded and no inference or training runs.

The [published summary](../../../results/evaluation/baomai-reference-amendment-20261110/summary.json) and [full JSON evidence capsule](../../../results/evaluation/baomai-reference-amendment-20261110/replay.zip) retain both reference versions' results. Metric changes from this replay describe changed annotations and matching populations.
