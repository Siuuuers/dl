#!/usr/bin/env python3
"""Build the slim, immutable-source R46 CPU review and PR133 replay notebook."""
import argparse
import ast
import json
from pathlib import Path
import re


def notebook(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pass a published immutable source commit")
    cells = []
    def cell(kind, source):
        item = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(keepends=True))
        if item["source"]:
            item["source"][-1] = item["source"][-1].rstrip("\n") + "\n"
        if kind == "code":
            item.update(execution_count=None, outputs=[])
            ast.parse(source.strip())
        cells.append(item)
    cell("markdown", """
# R46 - cassette-front review and PR133 result replay

**Choose a CPU runtime, then Run all.** This notebook checks and renders the R46
annotations and replays the completed R45 scores. It does not train a model.

PR133 trained the original 137 photos in both arms, plus **frame_0383 only** in
the candidate. The R46 changes are in held photos; frame_0383 remains byte
identical. Repeating that GPU treatment would not test these amendments.

You receive an amendment PDF, the R45 missed/error PDF, and one review ZIP with
originals, label TXT files, audit records and offline verification scripts.
Held proposal files are review material; only `complete-candidates/labels`
passes the recorded whole-image annotation gate. PR35 remains selected.

The existing R44 HTML remains a historical snapshot. The PDFs and R46 manifest
show the new corrections. Coordinates use x right, y down: **up decreases y**.
""")
    cell("markdown", "## 1. Load the published source\n\nThe source is pinned to one commit. Downloads use fixed content hashes.")
    cell("code", '''
from pathlib import Path
import json, os, subprocess, sys

SOURCE_REVISION = "__REVISION__"
BASE = Path(os.environ.get("BAOMAI_R46_HOME", "/content/baomai-r46"))
REPO, WORK, CACHE = BASE / "source", BASE / "work", BASE / "cache"
BASE.mkdir(parents=True, exist_ok=True)
SPARSE_PATHS = [
    "/scripts/", "/metadata/", "/data/manifest.json", "/data/labels/",
    "/data/candidates/front-review-2026-11-26-r37/",
    "/data/reviews/additional-fronts-2026-11-27-r46/",
    "/data/reviews/fronts-2026-11-26-r38/",
    "/data/references/sequence-refined-reference-2026-11-18/",
    "/data/references/visible-front-completion-2026-11-14/",
]
def run(command, cwd=None):
    subprocess.run([str(x) for x in command], cwd=cwd, check=True)
if not (REPO / ".git").exists():
    if REPO.exists() and any(REPO.iterdir()):
        raise RuntimeError("Preserved nonempty source folder; choose a fresh BAOMAI_R46_HOME")
    run(["git", "init", REPO])
    run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], REPO)
    run(["git", "fetch", "--depth", "1", "--filter=blob:none", "origin", SOURCE_REVISION], REPO)
    run(["git", "sparse-checkout", "init", "--no-cone"], REPO)
    run(["git", "sparse-checkout", "set", *SPARSE_PATHS], REPO)
    run(["git", "checkout", "--detach", SOURCE_REVISION], REPO)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True)
if head != SOURCE_REVISION or dirty:
    raise RuntimeError("Source revision differs or tracked source was edited; preserve it and use a fresh folder")
WORK.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
print("Pinned source:", head)
'''.replace("__REVISION__", revision))
    cell("markdown", """
## 2. Build the annotation review

Only Pillow and ReportLab are installed. The builder checks the exact R44
archive, every R46 edit, missing-point slots, row identities and complete-image
export. The PDF shows original/R43/R44/R46 comparisons, traced examples and all
12 photographs. Checksums and geometry checks do not certify physical truth.
""")
    cell("code", '''
run([sys.executable, "-m", "pip", "install", "--quiet", "Pillow==13.0.0", "reportlab==4.4.9"])
REVIEW = WORK / "review"
AMENDMENTS_PDF = REVIEW / "r46-annotation-amendments.pdf"
run([sys.executable, REPO / "scripts/build_front_review_r46.py",
     "--r44-archive", CACHE / "r44-review.zip", "--download-r44",
     "--output", REVIEW, "--pdf", AMENDMENTS_PDF])
verification = json.loads((REVIEW / "verification.json").read_text())
print("R46 changes:", verification["latest_counts"])
print("Complete-image candidates:", verification["annotation_complete_ids"])
print("Held whole photographs:", len(verification["held_image_ids"]))
''')
    cell("markdown", """
## 3. Replay the completed R45 result

This verifies the returned PR133 archive and both training receipts, then
reproduces the saved scores against R38, frozen666 and historical663 separately.
It downloads about 41 MB of returned data. It does not load weights or run
inference. The result PDF includes all 13 candidate R38 misses and the changed
matching cases. Leave the switch enabled to reproduce both reports.
""")
    cell("code", '''
REPLAY_PR133 = True
RESULTS_PDF = REVIEW / "r45-pilot-results-and-missed-review.pdf"
if REPLAY_PR133:
    PR133 = CACHE / "pr133"
    run([sys.executable, REPO / "scripts/audit_additional_front_pilot_r45_results.py",
         "--download-dir", PR133, "--output", REVIEW / "pr133-cpu-audit.json"])
    run([sys.executable, REPO / "scripts/export_additional_front_pilot_r45_pdf.py",
         "--archive", PR133 / "baomai-additional-front-pilot-r45-rebuilt-results.zip",
         "--images-dir", PR133 / "originals", "--download-images", "--output", RESULTS_PDF])
else:
    print("PR133 replay skipped. Annotation export is still verified.")
''')
    cell("markdown", """
## 4. Save the review files

Download the two PDFs and the single ZIP. No return PR is needed merely to
reproduce the same review. If you make further corrections, preserve the source
row IDs and describe the physical reason so they can become a new version.

For a quick JPEG + TXT check after extracting the ZIP:

```bash
python scripts/preview_pose_labels.py --image originals/frame_0001.jpg --labels proposals/frame_0001.txt --row 1 --scale 6 --output row001.png
```

`--row` means the one-based TXT line. Use `manifest.json` to map it to an immutable
source row. An outline is a review aid, not a segmentation mask; open traces
deliberately leave an unresolved upper edge unclosed.
""")
    cell("code", '''
run([sys.executable, REVIEW / "scripts/build_front_review_r46.py", "--check-output", REVIEW])
sys.path.insert(0, str(REPO / "scripts"))
from build_front_review_r46 import package
ZIP = WORK / "baomai-front-review-r46-results.zip"
receipt = package(REVIEW, ZIP)
print(json.dumps(receipt, indent=2))
deliverables = [AMENDMENTS_PDF, ZIP]
if REPLAY_PR133:
    deliverables.insert(1, RESULTS_PDF)
try:
    from google.colab import files
    for path in deliverables:
        files.download(str(path))
except ImportError:
    for path in deliverables:
        print("Saved:", path)
''')
    return dict(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "baomai_front_review_r46.ipynb", "provenance": []},
        "source_revision": revision, "training": False,
    }, nbformat=4, nbformat_minor=5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = notebook(a.source_revision)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(f"Saved {a.output} ({a.output.stat().st_size} bytes; four CPU code cells)")


if __name__ == "__main__":
    main()
