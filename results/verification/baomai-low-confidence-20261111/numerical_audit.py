import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile


CAPSULE = "results/evaluation/baomai-annotations-seed46-20261111/evaluation.zip"
CAPSULE_SHA = "3dfe3415841b47a605e2fd4b9e490b2c12b348b0014c6c2a8c78c06f323ffc1a"
COMPARISON = "evidence/seed-46/comparison-original-and-pr30.json"
PREDICTIONS = "evidence/seed-46/m2/predictions.jsonl"
REFERENCE = "data/references/visible-front-2026-11-10"
COUNT_KEYS = ("gold_count", "prediction_count", "tp", "fn", "fp", "duplicate_fp", "other_fp", "visible_endpoints", "returned_endpoints", "missing_endpoints")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("numerical-audit.json"))
    args = parser.parse_args()
    root = args.repo.resolve()
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import parse_pose_labels
    from pose_scoring import score_image, _box_iou
    from diagnose_pose_quality import quality_for_image, _returned
    capsule = (root / CAPSULE).read_bytes()
    assert sha(capsule) == CAPSULE_SHA
    with zipfile.ZipFile(root / CAPSULE) as archive:
        comparison_bytes, prediction_bytes = archive.read(COMPARISON), archive.read(PREDICTIONS)
    comparison = json.loads(comparison_bytes)
    records = [json.loads(line) for line in prediction_bytes.decode().splitlines()]
    header = records[0]
    for name, digest in header["source"]["files_sha256"].items():
        assert sha((root / name).read_bytes()) == digest
    predictions = {(r["image_id"], r["threshold"]): r for r in records[1:]}
    official = {(r["image_id"], r["box_confidence"]): r["revised"] for r in comparison["per_image"] if r["model_id"] == "m2"}
    reference_bytes = (root / REFERENCE / "manifest.json").read_bytes()
    assert sha(reference_bytes) == comparison["revised_reference"]["manifest_sha256"]
    files = json.loads(reference_bytes)["files"]
    image_rows, additions, recovered_cases = [], [], []
    label_hashes = {}
    all_retained_exact = True
    reassignments = []
    retained_error_changes = []
    for record in sorted(files, key=lambda x: x["image_id"]):
        image_id = record["image_id"]
        data = (root / REFERENCE / record["label_path"]).read_bytes()
        assert sha(data) == record["label_sha256"]
        label_hashes[image_id] = sha(data)
        width, height = record["width"], record["height"]
        gold = parse_pose_labels(data.decode(), width, height)
        old, new = predictions[(image_id, .25)], predictions[(image_id, .15)]
        assert old["image_sha256"] == new["image_sha256"] == record["image_sha256"]
        old_p, new_p = old["predictions"], new["predictions"]
        retained_exact = new_p[:len(old_p)] == old_p
        assert retained_exact and [p for p in new_p if p["confidence"] >= .25] == old_p
        all_retained_exact &= retained_exact
        assert all(.15 <= p["confidence"] < .25 for p in new_p[len(old_p):])
        scores = {conf: score_image(gold, pred, width, height) for conf, pred in ((.25, old_p), (.15, new_p))}
        qualities = {conf: quality_for_image(gold, pred, width, height, [2, 5, 10]) for conf, pred in ((.25, old_p), (.15, new_p))}
        for conf in (.25, .15):
            expected = official[(image_id, conf)]
            assert all(scores[conf][k] == expected["score"][k] for k in COUNT_KEYS)
            assert scores[conf]["unmatched_prediction_indices"] == expected["score"]["unmatched_prediction_indices"]
            assert scores[conf]["duplicate_prediction_indices"] == expected["score"]["duplicate_prediction_indices"]
            assert qualities[conf] == expected["quality"]
            assert [i + 1 for i in scores[conf]["unmatched_gold_indices"]] == expected["missed_label_rows_one_based"]
        old_matches = {m["prediction_index"]: m for m in scores[.25]["matches"]}
        new_matches = {m["prediction_index"]: m for m in scores[.15]["matches"]}
        for pi, match in old_matches.items():
            if new_matches.get(pi) != match:
                reassignments.append({"image_id": image_id, "prediction_index": pi, "old": match, "new": new_matches.get(pi)})
        old_missed = set(scores[.25]["unmatched_gold_indices"])
        new_missed = set(scores[.15]["unmatched_gold_indices"])
        recovered, lost = sorted(old_missed - new_missed), sorted(new_missed - old_missed)
        assert not lost
        old_err = {}
        new_err = {}
        for destination, score, pred in ((old_err, scores[.25], old_p), (new_err, scores[.15], new_p)):
            for match in score["matches"]:
                for side, (gp, pp) in enumerate(zip(gold[match["gold_index"]]["keypoints"], pred[match["prediction_index"]]["keypoints"])):
                    if gp[2] == 2 and _returned(pp, width, height):
                        destination[(match["gold_index"], side)] = math.hypot(gp[0] - pp[0], gp[1] - pp[1])
        for key, error in old_err.items():
            if new_err.get(key) != error:
                retained_error_changes.append({"image_id": image_id, "gold_index_side": key, "old_error": error, "new_error": new_err.get(key)})
        inverse_rows = {}
        for source_row, revised_row in enumerate(record["retained_source_to_revised_row_one_based"], 1):
            if revised_row is not None:
                inverse_rows.setdefault(revised_row, []).append(source_row)
        for pi in range(len(old_p), len(new_p)):
            pred = new_p[pi]
            match = new_matches.get(pi)
            label = "new_match" if match else ("duplicate_extra" if pi in scores[.15]["duplicate_prediction_indices"] else "other_extra")
            top = sorted(range(len(gold)), key=lambda gi: (-_box_iou(pred["bbox_xyxy"], gold[gi]["bbox_xyxy"]), gi))[:3]
            row = {"case_id": f"{image_id}:p{pi+1}", "image_id": image_id, "image_path": f"data/images/val/{image_id}.png", "reference_label_path": REFERENCE + "/" + record["label_path"], "image_sha256": record["image_sha256"], "prediction_index_zero_based": pi, "prediction_number_one_based": pi+1, "classification": label, "prediction": pred, "top_reference_overlaps": [{"revised_row_one_based": gi+1, "source_rows_one_based": inverse_rows.get(gi+1, []), "iou": _box_iou(pred["bbox_xyxy"], gold[gi]["bbox_xyxy"]), "reference": gold[gi]} for gi in top]}
            if match:
                gi = match["gold_index"]
                sides = []
                for side, (gp, pp) in enumerate(zip(gold[gi]["keypoints"], pred["keypoints"])):
                    returned = _returned(pp, width, height)
                    sides.append({"slot": side, "reference_locatable": gp[2] == 2, "returned_at_endpoint_050": returned, "endpoint_confidence": pp[2], "coordinate_distance_px_if_locatable": math.hypot(gp[0]-pp[0], gp[1]-pp[1]) if gp[2] == 2 else None, "scored_error_px": math.hypot(gp[0]-pp[0], gp[1]-pp[1]) if gp[2] == 2 and returned else None})
                row["match"] = {"revised_row_one_based": gi+1, "source_rows_one_based": inverse_rows.get(gi+1, []), "iou": match["iou"], "locatable_reference_corners": sum(p[2] == 2 for p in gold[gi]["keypoints"]), "reference": gold[gi], "endpoint_details": sides}
                recovered_cases.append(row)
            additions.append(row)
        image_rows.append({"image_id": image_id, "image_sha256": record["image_sha256"], "predictions_025": len(old_p), "predictions_015": len(new_p), "retained_raw_predictions_exact_and_same_prefix": retained_exact, "added_predictions": len(new_p)-len(old_p), "newly_matched_revised_rows_one_based": [i+1 for i in recovered], "lost_matches_revised_rows_one_based": [i+1 for i in lost], "score_025": {k:scores[.25][k] for k in COUNT_KEYS}, "score_015": {k:scores[.15][k] for k in COUNT_KEYS}, "delta_015_minus_025": {k:scores[.15][k]-scores[.25][k] for k in COUNT_KEYS}})
    assert not reassignments and not retained_error_changes
    aggregates = {str(conf): next(m["revised"] for m in comparison["metrics"] if m["model_id"] == "m2" and m["box_confidence"] == conf) for conf in (.25, .15)}
    for conf in (.25, .15):
        for key in COUNT_KEYS:
            assert sum(r["score_025" if conf == .25 else "score_015"][key] for r in image_rows) == aggregates[str(conf)]["score"][key]
    classified = {name:sum(a["classification"] == name for a in additions) for name in ("new_match", "duplicate_extra", "other_extra")}
    out = {"schema_version":1, "kind":"seed46-confidence-delta-numerical-review", "source":{"capsule":CAPSULE,"capsule_sha256":CAPSULE_SHA,"comparison_member":COMPARISON,"comparison_sha256":sha(comparison_bytes),"predictions_member":PREDICTIONS,"predictions_sha256":sha(prediction_bytes),"reference_manifest":REFERENCE+"/manifest.json","reference_manifest_sha256":sha(reference_bytes),"reference_label_sha256":label_hashes,"scorer_sha256":sha((root/"scripts/pose_scoring.py").read_bytes()),"parser_sha256":sha((root/"scripts/evaluate_pose.py").read_bytes()),"quality_sha256":sha((root/"scripts/diagnose_pose_quality.py").read_bytes()),"generator_sha256":sha(Path(__file__).read_bytes())},"settings":{"model_sha256":header["model_sha256"],"inference":header["inference"],"old_box_confidence":.25,"new_box_confidence":.15,"matching_iou":.5,"endpoint_confidence":.5,"reference_view":"PR30 revised"},"summary":{"images":len(image_rows),"images_with_added_proposals":sum(r["added_predictions"]>0 for r in image_rows),"retained_predictions":sum(r["predictions_025"] for r in image_rows),"retained_predictions_exact_same_order":all_retained_exact,"added_predictions":len(additions),"added_classification_counts":classified,"newly_matched_objects":[{"image_id":r["image_id"],"revised_row_one_based":r["match"]["revised_row_one_based"],"source_rows_one_based":r["match"]["source_rows_one_based"],"prediction_number_one_based":r["prediction_number_one_based"]} for r in recovered_cases],"lost_matches":[],"retained_match_reassignments":reassignments,"retained_endpoint_error_changes":retained_error_changes},"per_image":image_rows,"added_proposals":additions,"aggregates":aggregates,"limits":["Scorer labels duplicate/other extras by overlap with the revised reference; these are not visual adjudications of physical identity.","All reported identities, score counts and quality values replay exactly using the unchanged production scorer and reference; this is not an independently implemented scorer.","Original image pixels are not loaded by this numerical script; its image hashes bind the stored predictions and reference records.","The three new matches are measured against the frozen reference; the separate prior 011:61 boundary/identity caveat still applies.","Retained predictions and matches are exact. Quantile changes arise from adding one scored visible endpoint rather than moving retained coordinates.","No inference, training, label amendment, threshold default change or deployment accuracy claim."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out["summary"],indent=2))


if __name__ == "__main__":
    main()
