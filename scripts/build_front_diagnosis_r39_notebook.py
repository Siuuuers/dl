#!/usr/bin/env python3
"""Build a slim CPU notebook that replays the completed R39 diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]
RESULTS = "results/evaluation/pr123-r39-diagnosis-20261126"
REVIEW = "data/reviews/fronts-2026-11-26-r38/manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).lstrip().splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build(root, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Supply the immutable published source commit")
    review = json.loads((root / REVIEW).read_text())
    files = {
        "scripts/diagnose_r39_proposals.py", "scripts/probe_r39_clipped_nms.py", "scripts/evaluate_pose.py",
        "scripts/pose_scoring.py", "scripts/pose_checkpoint.py", "scripts/diagnose_pose_quality.py",
        REVIEW, review["row_maps"]["path"],
        "results/evaluation/pr123-r37-20261126/predictions/candidate.jsonl",
        "results/evaluation/pr123-r37-20261126/comparison.json",
        *(f"{RESULTS}/{name}" for name in (
            "diagnosis.json", "summary.json", "counterfactual-predictions.jsonl", "duplicate-overlap-audit.json",
            "clip-nms-probe.json", "clipped-nms-predictions.jsonl", "clip-nms-visual-review.json")),
        "docs/front-diagnosis-r39.md",
    }
    files.update((Path(REVIEW).parent / item["label_path"]).as_posix()
                 for item in review["files"] if item["split"] == "val")
    frozen_path = "data/references/sequence-refined-reference-2026-11-18/manifest.json"
    frozen = json.loads((root / frozen_path).read_text())
    files.add(frozen_path)
    files.update((Path(frozen_path).parent / item["label_path"]).as_posix()
                 for item in frozen["files"] if item.get("split", "val") == "val")
    hashes = {name: digest(root / name) for name in sorted(files)}
    setup = '''
import hashlib, json, os, sys, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
SOURCE_REVISION = __REVISION__
FILES = __FILES__
RESULTS = __RESULTS__
WORK = Path(os.environ.get("BAOMAI_R39_WORK", str(Path("/content/baomai-r39") if Path("/content").is_dir() else Path.cwd() / "baomai-r39"))).expanduser().resolve()
SOURCE = WORK / "source"
SOURCE.mkdir(parents=True, exist_ok=True)
LOCAL_SOURCE = os.environ.get("BAOMAI_R39_SOURCE_DIR")

def acquire(item):
    relative, expected = item
    target = SOURCE / relative
    if target.exists():
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Preserved changed file: {target}. Choose a new BAOMAI_R39_WORK folder.")
        return
    if LOCAL_SOURCE:
        data = (Path(LOCAL_SOURCE) / relative).read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/Siuuuers/dl/{SOURCE_REVISION}/{relative}"
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f"Pinned source hash differs: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    temporary.write_bytes(data)
    temporary.replace(target)

with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(acquire, FILES.items()))
print(f"Ready: {len(FILES)} pinned evidence/source files. CPU only; no ML installation.")
'''.replace("__REVISION__", repr(revision)).replace("__FILES__", repr(hashes)).replace("__RESULTS__", repr(RESULTS))
    replay = '''
for relative, expected in FILES.items():
    if hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Evidence changed: {relative}")
sys.path.insert(0, str(SOURCE / "scripts"))
from diagnose_r39_proposals import replay_evidence
from probe_r39_clipped_nms import replay_probe

evidence_path = SOURCE / RESULTS / "diagnosis.json"
evidence = json.loads(evidence_path.read_text())
summary = replay_evidence(SOURCE, evidence_path)
expected_summary = json.loads((SOURCE / RESULTS / "summary.json").read_text())
if summary != expected_summary:
    raise RuntimeError("Replayed diagnosis differs from the saved result")
clip_summary = replay_probe(SOURCE, SOURCE / RESULTS / "clip-nms-probe.json")

def show_table(headers, rows):
    text = "| " + " | ".join(headers) + " |\\n| " + " | ".join("---" for _ in headers) + " |\\n"
    text += "\\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except ImportError:
        print(text)

base, probe = summary["baseline"]["detection"], summary["counterfactual"]["detection"]
clipped = clip_summary["references"]["r38"]["probe"]["detection"]
show_table(["Accepted R38: 654 fronts", "Original 0.13", "Extra clipped NMS 0.70", "Exploratory conf 0.05"],
           [[label, base[key], clipped[key], probe[key]] for label, key in (
               ("Outputs", "prediction_count"), ("Matches", "tp"), ("Misses", "fn"),
               ("All unmatched", "fp"), ("Duplicate subset", "duplicate_fp"), ("Other unmatched", "other_fp"))])
mechanism_labels = {
    "no_iou_eligible_decoded_box": "Box overlap below 0.50",
    "confidence_gate": "Low confidence",
    "nms_suppression": "NMS suppression",
    "nms_or_output_limit_unresolved": "Unresolved filtering",
    "retained_assignment": "Matching assignment",
}
show_table(["Where the 11 baseline misses arise", "Count"],
           [[mechanism_labels[key], value] for key, value in summary["mechanism_counts"].items()])
show_table(["Image", "R38 row", "R37 row", "Mechanism", "Recovered at 0.05"],
           [[c["image_id"], c["r38_row"], c["r37_row"], mechanism_labels[c["mechanism"]],
             "Yes" if c["probe_recovers"] else "No"] for c in evidence["missed_cases"]])
print("All 20 baseline output arrays were reproduced exactly during the recorded fresh CPU trace.")
print("This notebook replays saved evidence and scores; it does not repeat that inference.")
print("The exploratory threshold is not adopted. All accepted labels and selected defaults are preserved.")
print("Extra clipped NMS removes nine outputs and preserves every existing matched prediction on R38 and frozen666.")
print(summary["replay_limit"])
'''
    export = '''
import csv, io, zipfile

rows = io.StringIO()
writer = csv.writer(rows)
writer.writerow(["image_id", "r38_row", "r37_row", "stable_id", "mechanism", "eligible_raw_boxes", "highest_eligible_confidence", "recovered_at_0_05"])
for case in evidence["missed_cases"]:
    eligible = case["iou_eligible_proposals"]
    writer.writerow([case["image_id"], case["r38_row"], case["r37_row"], case["stable_id"], case["mechanism"],
                     len(eligible), max((p["confidence"] for p in eligible), default=""), case["probe_recovers"]])
payloads = {
    "summary.json": (json.dumps(summary, indent=2, allow_nan=False) + "\\n").encode(),
    "clip-nms-summary.json": (json.dumps(clip_summary, indent=2, allow_nan=False) + "\\n").encode(),
    "baseline-predictions.jsonl": (SOURCE / "results/evaluation/pr123-r37-20261126/predictions/candidate.jsonl").read_bytes(),
    "missed-fronts.csv": rows.getvalue().encode(),
    "next-steps.md": (SOURCE / "docs/front-diagnosis-r39.md").read_bytes(),
    "source.json": (json.dumps({"revision": SOURCE_REVISION, "files_sha256": FILES,
                               "operation": "saved-evidence replay", "training": False,
                               "fresh_inference": False, "automatic_promotion": False}, indent=2) + "\\n").encode(),
}
for name in ("diagnosis.json", "counterfactual-predictions.jsonl", "duplicate-overlap-audit.json",
             "clip-nms-probe.json", "clipped-nms-predictions.jsonl", "clip-nms-visual-review.json"):
    payloads[name] = (SOURCE / RESULTS / name).read_bytes()
ARCHIVE = WORK / "baomai-front-diagnosis-r39-results.zip"
temporary = ARCHIVE.with_suffix(".tmp")
with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for name, data in sorted(payloads.items()):
        item = zipfile.ZipInfo(name, date_time=(2026, 11, 26, 0, 0, 0))
        item.compress_type = zipfile.ZIP_DEFLATED
        bundle.writestr(item, data)
with zipfile.ZipFile(temporary) as bundle:
    if bundle.testzip() is not None:
        raise RuntimeError("Export archive failed its integrity check")
if ARCHIVE.exists() and ARCHIVE.read_bytes() != temporary.read_bytes():
    temporary.unlink()
    raise RuntimeError("Preserved a different existing archive. Choose a new BAOMAI_R39_WORK folder.")
temporary.replace(ARCHIVE)
print(f"Saved {ARCHIVE.name} ({ARCHIVE.stat().st_size:,} bytes). No new training is needed for this diagnosis.")
if os.environ.get("BAOMAI_R39_DOWNLOAD", "1") != "0":
    try:
        from google.colab import files
        files.download(str(ARCHIVE))
    except ImportError:
        try:
            from IPython.display import FileLink, display
            display(FileLink(str(ARCHIVE)))
        except ImportError:
            print(ARCHIVE)
'''
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "colab": {"name": "baomai_front_diagnosis_r39.ipynb"},
            "baomai": {"source_revision": revision, "training": False, "inference": False,
                       "kind": "completed-raw-proposal-diagnostic-replay", "code_cells": 3}},
        "cells": [
            cell("markdown", """
                # R39 · Understand the remaining missed fronts

                Choose a **standard CPU runtime**, then **Run all**. This small notebook replays the completed PR123 raw-proposal diagnosis against your accepted R38 labels. It downloads only the pinned scripts, labels and numerical evidence; it needs no GPU, model weights, source photos or ML package installation.

                **Main result:** 6 of the 11 misses are confidence-gated, 4 lack a decoded box meeting the fixed IoU criterion, and 1 loses its qualifying box to NMS. An extra NMS pass after clipping removes **8 duplicates and 1 other unmatched output**, preserving every existing match and corner result. The separate 0.05 confidence probe recovers five matches but adds 68 unmatched outputs. Neither experiment automatically changes the selected defaults. Duplicates are already included in unmatched totals.

                R38 amendments and earlier PDFs are accepted. Detection remains separate from corner availability. Existing permission to predict hidden corners is preserved; a predicted point does not become observed training truth.
            """),
            cell("code", setup),
            cell("markdown", """
                ## Replay the evidence

                Recompute baseline and confidence-probe scores from complete 20-image prediction caches, check the accepted labels and original R37 identities, and rebuild mechanism counts from the saved eligible raw proposals. Reapply the small clipped-box NMS function and reproduce both R38 and frozen666 comparisons. Both-corner success remains 570/583 within 5 pixels and 582/583 within 10 pixels on R38.

                The underlying trace used fresh CPU inference and exactly reproduced all accepted baseline arrays. This notebook does not rerun the network. The compact evidence retains all qualifying proposals for the missed targets, but omits full dense tensors; independently establishing that an omitted tensor contains no other qualifying box requires the documented full trace command.
            """),
            cell("code", replay),
            cell("markdown", """
                ## Export the next-step record

                The ZIP includes the mechanism table, stable missed-front identities, both prediction/score evidence and the next-step guide. No labels, thresholds, model selection or training inputs are changed by running it. Use the guide to choose the next focused data or inference experiment.
            """),
            cell("code", export),
        ],
    }
    for index, item in enumerate(notebook["cells"]):
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"r39-cell-{index}", "exec")
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "notebooks/baomai_front_diagnosis_r39.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build(root, args.source_revision), indent=1) + "\n")
    print(json.dumps({"path": str(output), "size_bytes": output.stat().st_size, "sha256": digest(output)}))
