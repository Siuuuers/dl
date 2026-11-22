#!/usr/bin/env python3
"""Trace fixed-primary/r21 score sensitivity with unchanged PR96 predictions.

Identity comes from hash-bound archived row maps, never nearest-box assignment.
This diagnoses reference dependence; it does not replace the primary result.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(root, descriptor):
    path = root / descriptor["path"]
    assert sha(path) == descriptor["sha256"], str(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, evidence = args.root.resolve(), args.evidence.resolve()
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import parse_pose_labels, preflight_dataset
    from front_review_experiment import apply_operations
    from pose_scoring import _box_iou

    scored = read(evidence / "fixed-comparison.json")
    protocol = read(bound(root, scored["protocol"]))
    primary_descriptor = {"path": protocol["evaluation"]["primary_reference"],
                          "sha256": protocol["evaluation"]["primary_reference_sha256"]}
    current_descriptor = protocol["evaluation"]["current_review_reference"]
    primary_path, current_path = (bound(root, d) for d in (primary_descriptor, current_descriptor))
    primary, current = read(primary_path), read(current_path)
    records = {r["id"]: r for r in preflight_dataset(root, require_weights=False)["records"]}

    def labels(path, manifest):
        result = {}
        for item in manifest["files"]:
            iid = item["image_id"]
            if iid not in records:
                continue
            p = path.parent / item["label_path"]
            assert sha(p) == item["label_sha256"]
            rec = records[iid]
            result[iid] = parse_pose_labels(p.read_text(), rec["width"], rec["height"])
        assert set(result) == set(records)
        return result

    gold = {"primary": labels(primary_path, primary),
            "current_r21_exploratory": labels(current_path, current)}
    chain, path = [], current_path
    while True:
        manifest = read(path)
        if "parent" not in manifest:
            break
        chain.append((path, manifest))
        path = bound(root, manifest["parent"])
    base = read(path)
    recovery_path = bound(root, base["recovery_manifest"])
    recovery = read(recovery_path)
    assert recovery["pinned_parent_manifests"][primary_descriptor["path"]] == primary_descriptor["sha256"]
    base_files = {f["image_id"]: f for f in base["files"]}
    primary_files = {f["image_id"]: f for f in primary["files"]}
    lineage = {}
    for item in recovery["files"]:
        iid = item["image_id"]
        if iid not in records:
            continue
        expected_parent = primary_path.parent / primary_files[iid]["label_path"]
        assert root / item["parent_label_path"] == expected_parent
        assert item["parent_label_sha256"] == sha(expected_parent)
        assert base_files[iid]["label_sha256"] == item["recovered_label_sha256"]
        assert sha(path.parent / base_files[iid]["label_path"]) == item["recovered_label_sha256"]
        lineage[iid] = {r["current_row"]: r["parent_row"] for r in item["row_map"]}
    provenance = [{"path": str(path.relative_to(root)), "sha256": sha(path)},
                  {"path": str(recovery_path.relative_to(root)), "sha256": sha(recovery_path)}]
    for child_path, manifest in reversed(chain):
        parent_path = bound(root, manifest["parent"])
        parent = read(parent_path)
        parent_files = {f["image_id"]: f for f in parent["files"]}
        operations = read(bound(root, manifest["operations"]))
        maps = read(bound(root, manifest["row_maps"]))
        provenance.extend([{"path": str(child_path.relative_to(root)), "sha256": sha(child_path)},
                           manifest["operations"], manifest["row_maps"]])
        for item in manifest["files"]:
            iid = item["image_id"]
            if iid not in records:
                continue
            old_path = parent_path.parent / parent_files[iid]["label_path"]
            new_path = child_path.parent / item["label_path"]
            assert sha(old_path) == parent_files[iid]["label_sha256"]
            assert sha(new_path) == item["label_sha256"]
            expected, row_map = apply_operations(old_path.read_bytes(),
                [op for op in operations if op["image_id"] == iid], iid)
            assert new_path.read_bytes() == expected and row_map == maps[iid]
            lineage[iid] = {r["current_row"]: lineage[iid][r["base_row"]] if "base_row" in r else None
                            for r in maps[iid]["rows"]}
    assert sum(map(len, lineage.values())) == 662
    assert all(len(v) == len(set(v.values())) for v in lineage.values())
    inverse = {iid: {parent: now for now, parent in values.items() if parent is not None}
               for iid, values in lineage.items()}

    report = {"schema_version": 1, "kind": "pr96-r21-reference-definition-sensitivity",
              "script_sha256": sha(__file__), "fixed_comparison_sha256": sha(evidence / "fixed-comparison.json"),
              "references": {"primary": primary_descriptor, "current_r21_exploratory": current_descriptor},
              "identity_method": "Exact immutable recovery and successive review row maps, replayed against hash-bound label bytes.",
              "lineage_provenance": provenance, "current_to_primary_rows": lineage,
              "primary_objects": 666, "retained_primary_objects": sum(len(v) for v in inverse.values()),
              "deleted_primary_rows": {iid: sorted(set(range(1, len(gold['primary'][iid])+1))-set(inverse[iid]))
                                       for iid in records if len(inverse[iid]) != len(gold['primary'][iid])},
              "models": {}, "limits": [
                  "No predictions, checkpoints, thresholds or primary results are changed.",
                  "Lineage establishes the annotation's identity correspondence; visual review is separate.",
                  "IoU crossings establish score sensitivity to geometry, not independent correctness of either reference.",
                  "An unmatched gold row is a matching failure, not necessarily absence of a returned detector output.",
                  "Unmatched predictions include reference disagreements and duplicates; they are not all confirmed background errors.",
                  "Only pre-return r21 labels and frozen666 are used; no r22 labels are loaded."]}
    for role in ("pr35", "reference", "candidate"):
        pred_path = evidence / "inference" / ("eval-"+role) / "predictions.jsonl"
        assert sha(pred_path) == scored["models"][role]["predictions_sha256"]
        predictions = {r["image_id"]: r["predictions"] for r in map(json.loads, pred_path.read_text().splitlines()[1:])}
        counts = Counter()
        cases, reversed_cases = [], []
        for iid, rec in sorted(records.items()):
            traces = {label: scored["references"][label]["matching_traces"][role][iid]
                      for label in gold}
            matches = {label: {r["gold_index"]+1: r for r in trace["matches"]}
                       for label, trace in traces.items()}

            def detail(parent_row, current_row, pred_index):
                old = gold["primary"][iid][parent_row-1]
                new = gold["current_r21_exploratory"][iid][current_row-1] if current_row else None
                predicted = predictions[iid][pred_index] if pred_index is not None else None
                result = {"image_id": iid, "image_sha256": rec["image_sha256"],
                          "primary_row": parent_row, "r21_row": current_row,
                          "primary_gold": old, "r21_gold": new,
                          "prediction_index": pred_index, "prediction": predicted,
                          "primary_iou": _box_iou(old["bbox_xyxy"], predicted["bbox_xyxy"]) if predicted else None,
                          "r21_iou": _box_iou(new["bbox_xyxy"], predicted["bbox_xyxy"]) if new and predicted else None}
                if new:
                    result["box_edges_changed"] = [abs(a-b)>1e-6 for a,b in zip(old["bbox_xyxy"],new["bbox_xyxy"])]
                    result["reference_keypoint_visibility_changed"] = [a[2]!=b[2] for a,b in zip(old["keypoints"],new["keypoints"])]
                    result["reference_keypoint_shifts_px"] = [math.hypot(a[0]-b[0],a[1]-b[1]) if a[2]==b[2]==2 else None
                                                               for a,b in zip(old["keypoints"],new["keypoints"])]
                for label, target in (("primary",old),("r21",new)):
                    result[label+"_corner_errors_px"] = [math.hypot(p[0]-g[0],p[1]-g[1]) if g[2]==2 else None
                        for g,p in zip(target["keypoints"],predicted["keypoints"])] if target and predicted else None
                return result

            for index in traces["primary"]["unmatched_gold_indices"]:
                parent_row = index+1
                now = inverse[iid].get(parent_row)
                matched = matches["current_r21_exploratory"].get(now)
                if now is None:
                    category = "deleted_from_r21_reference"
                    pred_index = None
                elif matched:
                    pred_index = matched["prediction_index"]
                    old_iou = _box_iou(gold["primary"][iid][index]["bbox_xyxy"], predictions[iid][pred_index]["bbox_xyxy"])
                    category = "same_output_crosses_iou_with_reference_box_change" if old_iou < .5 else "matching_assignment_changed"
                else:
                    category = "still_unmatched_in_r21"
                    pred_index = max(range(len(predictions[iid])), key=lambda j: _box_iou(gold["primary"][iid][index]["bbox_xyxy"],predictions[iid][j]["bbox_xyxy"])) if predictions[iid] else None
                case = detail(parent_row,now,pred_index)
                case["category"] = category
                cases.append(case)
                counts[category] += 1
            for index in traces["current_r21_exploratory"]["unmatched_gold_indices"]:
                now = index+1
                parent_row = lineage[iid][now]
                old_match = matches["primary"].get(parent_row)
                if old_match:
                    case = detail(parent_row,now,old_match["prediction_index"])
                    case["category"] = "same_output_falls_below_iou_with_reference_box_change" if case["r21_iou"] < .5 else "matching_assignment_changed"
                    reversed_cases.append(case)
        report["models"][role] = {"predictions_sha256":sha(pred_path),
             "fixed_primary_misses":len(cases), "fixed_miss_categories":dict(counts),
             "fixed_primary_miss_cases":cases, "primary_match_but_r21_miss_cases":reversed_cases,
             "r21_misses":scored["references"]["current_r21_exploratory"]["models"][role]["detection"]["fn"]}
    assert not args.output.exists(), "Choose a fresh output"
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"deleted_primary_rows":report["deleted_primary_rows"],
        "models":{role:{k:v for k,v in model.items() if not k.endswith("cases")}
                  for role,model in report["models"].items()}},indent=2))


if __name__ == "__main__":
    main()
