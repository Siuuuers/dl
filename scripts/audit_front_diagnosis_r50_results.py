#!/usr/bin/env python3
"""Audit PR144 through its original pinned R50 verifier and independent R51 replay.

No fresh inference or training. Requires the original clean R50 source checkout,
the PR137/PR143/PR144 ZIPs and the 20 original development PNGs. Existing files
are preserved. The original verifier runs in a separate process/source namespace.
"""
import argparse
from collections import Counter
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import audit_front_diagnosis_r49_results as replay
from predict_pose_folder import postclip_nms

ROOT = Path(__file__).resolve().parents[1]
PR_HEAD = "9e6841592bba4a3053b6e47360b9793b29a6d0ef"
ARCHIVE_SHA = "c22da73bf196121cc9090e19b443ae8aa1d7c89a6de5d18353e0d4857d65014f"
ARCHIVE_BYTES = 15997796
SOURCE = "5aa5c04d91fb6e1b5a5cf9cce85215f7564238aa"
SCRIPT_SHA = "cba4f1df6dcf431c933af1506a4de15718636d9477406eb34779ac4d8ab6ee3e"
NAME = "baomai-r50-diagnosis-results.zip"
need, sha, checked, encoded = replay.need, replay.sha, replay.checked, replay.encoded


def audit(args):
    root, source, work = args.root.resolve(), args.source_root.resolve(), args.work.resolve()
    data = checked(args.archive, ARCHIVE_SHA)
    need(len(data) == ARCHIVE_BYTES, "PR144 archive size differs")
    old_data = checked(args.pr143_archive, replay.ARCHIVE_SHA)
    script = source / "scripts/diagnose_fronts_r50.py"
    checked(script, SCRIPT_SHA)
    common = [sys.executable, "-B", str(script), "--root", str(source), "--work", str(work),
              "--source-revision", SOURCE, "--script-sha256", SCRIPT_SHA,
              "--archive", str(args.prior_archive.resolve()), "--originals-dir", str(args.originals.resolve())]
    subprocess.run([*common, "--preflight-only"], check=True)
    target = work / NAME
    need(not target.exists() or target.read_bytes() == data, "Different existing return preserved")
    if not target.exists():
        target.write_bytes(data)
    subprocess.run([*common, "--verify-return"], check=True)
    records, refs, maps, references = replay.load_references(root)
    geometry = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z, zipfile.ZipFile(io.BytesIO(old_data)) as old:
        returned = json.loads(z.read("summary.json"))
        outputs = json.loads(z.read("predictions.json"))
        traces = json.loads(z.read("trace.json"))
        cases = json.loads(z.read("missed-cases.json"))
        preflight = json.loads(z.read("preflight.json"))
        need(json.loads((work / "preflight.json").read_text()) == preflight, "Original source/input contract differs")
        need(set(outputs) == set(traces) == set(records), "Prediction image membership differs")
        comparisons = {name: {"bytes_equal": z.read(name) == old.read(name),
                             "json_equal": json.loads(z.read(name)) == json.loads(old.read(name))}
                       for name in ("predictions.json", "trace.json", "input-audit.json")}
        need(all(value["json_equal"] for value in comparisons.values()), "PR143 returned computation differs")
        identical_tensors = []
        for identity in sorted(records):
            row, trace = outputs[identity], traces[identity]
            raw_bytes = z.read("decoded/" + identity + ".npy")
            need(raw_bytes == old.read("decoded/" + identity + ".npy"), "PR143 tensor bytes differ: " + identity)
            identical_tensors.append(identity)
            raw = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
            need(list(raw.shape) == trace["shape"] and sha(raw.tobytes()) == trace["raw_tensor_sha256"], "Tensor trace differs")
            geometry[identity] = replay.decoded_geometry(raw, trace["input_shape"])
            boxes, scaled, points, confidence = geometry[identity]
            replay.check_hard_indices(boxes, confidence, row["ordinary_indices"])
            need(replay.reconstructed(row["ordinary_indices"], confidence[row["ordinary_indices"]], scaled, points) == row["hard_raw"], "Native reconstruction differs")
            indices, scores, capped = replay.soft_indices(boxes, confidence)
            need(indices == row["soft_raw_indices"] and capped == trace["soft_cap_reached"]
                 and replay.reconstructed(indices, scores, scaled, points) == row["soft_raw"], "Soft reconstruction differs")
            for arm, key in (("hard", "ordinary_indices"), ("soft", "soft_raw_indices")):
                kept, filtering = postclip_nms(row[arm + "_raw"])
                need(kept == row[arm] and filtering == row[arm + "_audit"], "Postclip replay differs")
                need([row[key][n - 1] for n in filtering["retained_primary_prediction_ids"]] == row[arm + "_indices"], "Postclip anchor map differs")
            need(trace["above_confidence"] == int(sum(confidence > .13)) and trace["native_reconstruction_exact"], "Confidence/native trace differs")
            need(trace["saved_raw_exact"] == (row["hard_raw"] == row["saved"]["raw"])
                 and trace["saved_postclip_exact"] == (row["hard"] == row["saved"]["postclip"]), "Saved/fresh receipt differs")
            checked(args.originals / (identity + ".png"), records[identity]["image_sha256"])
        timings = json.loads(z.read("timing.json"))
        need(len(timings) == 20 and {row["image"] for row in timings} == set(records)
             and all(math.isfinite(row[key]) and row[key] >= 0 for row in timings for key in row if key != "image"), "Invalid returned timing data")
        original_reports = json.loads(old.read("summary.json"))["reports"]
        member_count = len(z.namelist()) - 1
    reports = replay.score_reports(refs, outputs)
    need(reports == returned["reports"], "Independent four-reference score/assignment replay differs")
    need({name: reports[name] for name in original_reports} == original_reports, "PR143 original score tables differ")
    rebuilt_cases = replay.missed_cases(refs, maps, outputs, reports, geometry)
    case_key = lambda row: (row["reference"], row["image"], row["r38_row"])
    need(all(row["row"] == row["r38_row"] for row in cases), "R50 row alias differs")
    need(sorted(({k: v for k, v in row.items() if k != "row"} for row in cases), key=case_key)
         == sorted(rebuilt_cases, key=case_key), "Independent missed-case trace replay differs")
    stages = {name: dict(Counter(row["mechanism"] for row in rebuilt_cases if row["reference"] == name and row["fresh_hard_miss"])) for name in ("r38", "r50")}
    need(stages == returned["fresh_miss_mechanisms"], "Miss-stage aggregate differs")
    previous = json.loads((root / "results/evaluation/pr143-r49-20261127/summary.json").read_text())
    compact = {name: {view: {"detection": {k: v for k, v in reports[name][view]["detection"].items() if k != "endpoint_errors_px"},
                            "corners_by_locatable_count": reports[name][view]["corners_by_locatable_count"]}
                     for view in replay.VIEWS} for name in refs}
    need(compact == previous["reports"], "Earlier R51 PR143/R50 score replay differs")
    delta = replay.filter_delta(outputs, refs, reports)
    need(delta["counts"] == previous["filter_delta_counts"] and delta["matched_anchor_assignment_changes"] == previous["matched_anchor_assignment_changes"], "Earlier proposal identity comparison differs")
    return dict(schema="pr144-r50-return-audit-v1", status="passed", pr=144, pr_head=PR_HEAD,
        archive_sha256=ARCHIVE_SHA, archive_bytes=ARCHIVE_BYTES, source=SOURCE, script_sha256=SCRIPT_SHA,
        auditor_sha256=sha(Path(__file__).read_bytes()), replay_helper_sha256=sha((root / "scripts/audit_front_diagnosis_r49_results.py").read_bytes()),
        original_source_and_input_contract_sha256=preflight["contract_sha256"],
        source_data_files_verified=len(preflight["contract"]["source_files_sha256"]), inventory_members_verified=member_count,
        original_images_verified=20, native_numpy_selection_checks=20, exact_native_output_reconstructions=20,
        exact_soft_output_reconstructions=20, replayed_postclip_filters=40, replayed_score_tables=24, replayed_miss_traces=len(rebuilt_cases),
        runtime=returned["runtime"], references=references, primary_reference="frozen666",
        pr143_comparison=dict(archive_sha256=replay.ARCHIVE_SHA, members=comparisons,
            byte_identical_decoded_tensors=len(identical_tensors), decoded_image_ids=identical_tensors,
            original_18_score_tables_exact=True, r51_r50_replay_6_score_tables_exact=True),
        fresh_saved_exact_images=sum(row["hard"] == row["saved"]["postclip"] for row in outputs.values()),
        miss_stages=stages, filter_delta_counts=delta["counts"], matched_anchor_assignment_changes=delta["matched_anchor_assignment_changes"],
        reports=compact, fresh_inference_by_this_audit=False, returned_experiment_declares_fresh_inference=True,
        training=False, automatic_promotion=False,
        conclusion="PR144 corroborates the PR143/R51 diagnosis exactly on the same checkpoint, image tensors, predictions and score references. It adds no recovered cassette or independent test evidence.",
        timing_limit="Returned timings cover suppression and construction only; timing differences are not an optimized deployment benchmark.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    for name in ("source-root", "work", "archive", "pr143-archive", "prior-archive", "originals", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = encoded(result)
    need(not args.output.exists() or args.output.read_bytes() == data, "Different existing audit preserved")
    args.output.write_bytes(data)
    print("PR144 audit passed:", args.output, "SHA256:", sha(data), flush=True)


if __name__ == "__main__":
    main()
