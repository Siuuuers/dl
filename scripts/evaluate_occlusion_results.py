#!/usr/bin/env python3
"""Evaluate or replay the fixed PR62 comparison without changing its training protocol."""

import argparse
import json
from pathlib import Path
import zipfile

from diagnose_pose_quality import aggregate_quality, quality_for_image
from evaluate_pose import file_sha256, parse_pose_labels
from occlusion_experiment import load_protocol, package_pair
from pose_scoring import aggregate_scores, score_image
from probe_pixel_occlusion import altered, build, endpoint, read, require, score_all, sha, write


ROLES = ("current", "control", "candidate")
RETURN_SHA256 = "4c94b9ae2496b0ab8846941e7da61eba4931630b805ca7f35a0a9272fc486cad"


def frozen_inputs(root):
    context = load_protocol(root)
    evaluation = context["document"]["evaluation"]
    ref_path = root / evaluation["primary_reference"]["path"]
    references = {}
    for item in read(ref_path)["files"]:
        references[item["image_id"]] = {**item, "objects": parse_pose_labels(
            (ref_path.parent / item["label_path"]).read_text(), item["width"], item["height"], item["image_id"])}
    require(len(references) == 20 and sum(len(x["objects"]) for x in references.values()) == 663,
            "clean reference denominator changed")
    spec = evaluation["synthetic_probe"]
    with zipfile.ZipFile(root / spec["archive_path"]) as archive:
        saved = {name: json.loads(archive.read("evaluation/" + name + ".json")) for name in
                 ("prepared-cases", "reference-objects", "scores", "summary")}
        lines = [json.loads(x) for x in archive.read("evaluation/predictions.jsonl").splitlines()]
    probe = read(root / spec["protocol_path"])
    require(lines[0]["model_sha256"] == evaluation["PR35"]["checkpoint_sha256"], "wrong saved PR35")
    scores, summary = score_all(probe, saved["prepared-cases"], saved["reference-objects"], lines[1:])
    require(scores == saved["scores"] and summary == saved["summary"], "published PR60 replay differs")
    baseline = probe["baseline"]
    require(file_sha256(root / baseline["archive_path"]) == baseline["archive_sha256"], "clean baseline hash differs")
    with zipfile.ZipFile(root / baseline["archive_path"]) as archive:
        records = [json.loads(x) for x in archive.read(baseline["predictions_member"]).splitlines()]
    clean = {r["image_id"]: r["predictions"] for r in records if
             r.get("record_type") == "prediction" and r.get("threshold") == .13}
    require(set(clean) == set(references), "published clean baseline image set differs")
    return context, {"clean_references": references, "probe_protocol": probe,
                     "probe_cases": saved["prepared-cases"], "probe_references": saved["reference-objects"]}, clean, lines[1:]


def infer(root, args, output, context, inputs, expected_clean, current_probe):
    from PIL import Image
    from predict_pose_folder import prepare_predictor, verify_trained_checkpoint
    require(file_sha256(args.returned_zip) == RETURN_SHA256, "PR62 returned archive identity differs")
    verified = package_pair(root, args.returned_dir / "control", args.returned_dir / "candidate",
                            args.returned_zip, verify_existing=True)
    images = {}
    for image_id, ref in inputs["clean_references"].items():
        path = args.image_root / (image_id + ".png")
        require(file_sha256(path) == ref["image_sha256"], "clean input hash differs: " + image_id)
        with Image.open(path) as source:
            require(source.format == "PNG" and source.getexif().get(274, 1) == 1, "image encoding changed")
            images[image_id] = source.convert("RGB")
        require(images[image_id].size == (ref["width"], ref["height"]), "input dimensions differ")
    cases, references, _ = build(root, args.image_root, inputs["probe_protocol"])
    require(cases == inputs["probe_cases"] and references == inputs["probe_references"], "PR60 inputs differ")
    verification = {"return_archive": verified, "training_source": "b457a4a6975104700d7342fa4b997ea6da7625ad",
                    "return_commit": "52a5be19cb6afdece05637c6e6c86dcc5fccd76d",
                    "protocol_sha256": context["sha256"], "new_training": False,
                    "new_image_predictions": 240, "reused_PR35_probe_predictions": 94,
                    "warmup_predictions_excluded": 3, "models": {}, "training": {}}
    for role in ("control", "candidate"):
        receipt = read(args.returned_dir / role / "training-receipt.json")
        verification["training"][role] = {k: receipt[k] for k in
            ("runtime", "input_corruption", "source", "epochs_completed", "images_processed", "batches_completed") if k in receipt}
    for role in ROLES:
        manifest = args.current_manifest if role == "current" else args.returned_dir / role / "checkpoint-manifest.json"
        selected = verify_trained_checkpoint(root, manifest)
        if role == "current":
            require(selected["model_sha256"] == context["document"]["evaluation"]["PR35"]["checkpoint_sha256"], "wrong operational model")
        predictor, runtime = prepare_predictor(selected["weights"], "cpu", 4, next(iter(images.values())), .13, .70)
        verification["models"][role] = {"model_sha256": selected["model_sha256"],
                                        "checkpoint_manifest_sha256": file_sha256(manifest), "runtime": runtime}
        clean = {}
        for image_id, rgb in images.items():
            clean[image_id] = predictor(rgb)["predictions"]
            print(json.dumps({"role": role, "clean_images": len(clean)}), flush=True)
        if role == "current":
            require(clean == expected_clean, "PR35 clean raw predictions no longer reproduce")
            probe_records = current_probe
        else:
            probe_records = []
            for case in cases:
                if case["condition"] == "intact":
                    predictions = clean[case["image_id"]]
                else:
                    rgb = altered(images[case["image_id"]], case["mask_rect_xyxy"], inputs["probe_protocol"]["masks"]["fill_rgb"])
                    require(sha(rgb.tobytes()) == case["input_rgb_sha256"], "masked pixel hash differs")
                    predictions = predictor(rgb)["predictions"]
                probe_records.append({"record_type": "prediction", **{k: case[k] for k in
                    ("case_id", "image_id", "condition", "input_rgb_sha256")}, "predictions": predictions})
                if len(probe_records) % 10 == 0:
                    print(json.dumps({"role": role, "probe_records": len(probe_records)}), flush=True)
        write(output / (role + ".json"), {"model_sha256": selected["model_sha256"], "clean": clean, "probe": probe_records})
        del predictor
    verification["PR35_clean_arrays_equal_published"] = True
    verification["source_files"] = {p: file_sha256(root / p) for p in (
        "scripts/evaluate_occlusion_results.py", "scripts/diagnose_pose_quality.py", "scripts/pose_scoring.py",
        "scripts/probe_pixel_occlusion.py", "scripts/evaluate_pose.py", "scripts/predict_pose_folder.py")}
    write(output / "verification.json", verification)


def clean_scores(references, predictions):
    require(set(predictions) == set(references), "clean image set differs")
    images, objects, qualities = {}, {}, []
    for image_id, ref in references.items():
        gold, raw = ref["objects"], predictions[image_id]
        require(all(p["confidence"] >= .13 for p in raw), "unfiltered clean proposal")
        scored = score_image(gold, raw, ref["width"], ref["height"], .5, .5)
        images[image_id] = scored
        qualities.append(quality_for_image(gold, raw, ref["width"], ref["height"], [5, 10]))
        matches = {m["gold_index"]: m for m in scored["matches"]}
        for index, truth in enumerate(gold):
            match = matches.get(index)
            predicted = raw[match["prediction_index"]] if match else None
            points = [endpoint(predicted["keypoints"][side] if predicted else None, point, ref["width"], ref["height"])
                      if point[2] == 2 else None for side, point in enumerate(truth["keypoints"])]
            eligible = all(point is not None for point in points)
            objects[f"{image_id}:{index + 1}"] = {"matched": match is not None, "match": match, "endpoints": points,
                "pair_eligible": eligible, "pair_available": eligible and all(p["available"] for p in points),
                **{f"pair_within_{r}px": eligible and all(p[f"within_{r}px"] for p in points) for r in (5, 10)}}
    totals, quality = aggregate_scores(list(images.values())), aggregate_quality(qualities)
    totals.pop("endpoint_errors_px")
    totals.update(pairs_within_5px=quality["radii"][0]["both_visible_objects_within_radius"],
                  pairs_within_10px=quality["radii"][1]["both_visible_objects_within_radius"],
                  pair_denominator=quality["both_visible_objects"],
                  pairs_available=quality["both_visible_objects_with_both_returned"],
                  empty_outputs=sum(x["prediction_count"] for x in images.values() if x["gold_count"] == 0),
                  nonempty_miss_free_images=sum(x["gold_count"] > 0 and x["fn"] == 0 for x in images.values()),
                  nonempty_images_without_misses_or_extras=sum(x["gold_count"] > 0 and x["fn"] == x["fp"] == 0 for x in images.values()))
    return {"summary": totals, "images": images, "objects": objects}


def changed_flags(candidate, baseline, fields):
    return {field: {"gained": [key for key in candidate if candidate[key][field] and not baseline[key][field]],
                    "lost": [key for key in candidate if not candidate[key][field] and baseline[key][field]]}
            for field in fields}


def summarize(inputs, raw):
    results = {}
    for role in ROLES:
        clean = clean_scores(inputs["clean_references"], raw[role]["clean"])
        probe_scores, probe_summary = score_all(inputs["probe_protocol"], inputs["probe_cases"],
                                                inputs["probe_references"], raw[role]["probe"])
        results[role] = {"clean": clean, "probe_scores": probe_scores, "probe_summary": probe_summary}
    comparisons = {}
    for base in ("current", "control"):
        candidate, baseline = results["candidate"], results[base]
        a, b = candidate["clean"]["summary"], baseline["clean"]["summary"]
        clean_gates = {name: a[name] <= b[name] for name in ("fn", "fp", "duplicate_fp", "empty_outputs")}
        clean_gates.update({name: a[name] >= b[name] for name in ("returned_endpoints", "pairs_within_5px", "pairs_within_10px")})
        clean_changes = changed_flags(candidate["clean"]["objects"], baseline["clean"]["objects"],
                                      ("matched", "pair_available", "pair_within_5px", "pair_within_10px"))
        target_fields = ("matched", "pair_available", "pair_within_5px", "pair_within_10px")
        targets = {role: {f'{t["case_id"]}:{t["row"]}': t for t in results[role]["probe_scores"]["target_scores"]}
                   for role in ("candidate", base)}
        changed = changed_flags(targets["candidate"], targets[base], target_fields)
        covered, supporting = {}, []
        for role in ("candidate", base):
            covered[role] = sum(t[t["condition"]]["within_10px"] for t in targets[role].values()
                                if t["condition"] in ("left", "right"))
        for key, target in targets["candidate"].items():
            side = target["condition"]
            if side not in ("left", "right"):
                continue
            old, new = targets[base][key][side], target[side]
            if new["within_10px"] and not old["within_10px"] and old["available"] and old["error_px"] - new["error_px"] >= 2:
                supporting.append({"case_id": target["case_id"], "object": f'{target["image_id"]}:{target["row"]}',
                                   "error_before_px": old["error_px"], "error_after_px": new["error_px"]})
        probe_gates = {"additional_covered_10px": covered["candidate"] - covered[base] >= 2,
                       "two_objects_improve_at_least_2px": len({x["object"] for x in supporting}) >= 2}
        conditions_a, conditions_b = candidate["probe_summary"]["target_conditions"], baseline["probe_summary"]["target_conditions"]
        for condition in ("intact", "left", "right", "center"):
            ca, cb = conditions_a[condition]["counts"], conditions_b[condition]["counts"]
            fields = ["matched", "pair_available"]
            if condition in ("left", "right"):
                opposite = "right" if condition == "left" else "left"
                fields += [condition + "_within_5px", condition + "_within_10px", opposite + "_within_10px"]
            if condition == "center":
                fields += ["pair_within_5px", "pair_within_10px"]
            probe_gates.update({condition + "/" + field: ca[field] >= cb[field] for field in fields})
        comparisons[base] = {"clean_nonregression": clean_gates, "clean_identity_changes": clean_changes,
                             "probe_gates": probe_gates, "probe_identity_changes": changed,
                             "covered_10px_counts": covered, "supporting_gains": supporting,
                             "numeric_gates_pass": all(clean_gates.values()) and all(probe_gates.values()),
                             "changed_identities_require_image_review": True}
    summary = {"clean": {role: results[role]["clean"]["summary"] for role in ROLES},
               "probe": {role: results[role]["probe_summary"] for role in ROLES}, "comparisons": comparisons,
               "numeric_gates_pass_against_both": all(x["numeric_gates_pass"] for x in comparisons.values()),
               "automatic_model_promotion": False,
               "scope": "Fixed development comparison; artificial masks and repeated scenes do not establish natural occlusion or unseen-scene accuracy."}
    return results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    for name in ("returned-dir", "returned-zip", "current-manifest", "image-root"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    require(not output.exists(), "output must be fresh")
    context, inputs, expected, current_probe = frozen_inputs(root)
    output.mkdir(parents=True)
    if args.replay:
        source = args.replay.resolve()
        require(read(source / "inputs.json") == inputs, "saved references/cases differ from frozen sources")
        status = read(source / "status.json")
        require(status["status"] == "complete", "incomplete evidence")
        for name, digest in status["files"].items():
            require(Path(name).name == name and file_sha256(source / name) == digest, "evidence hash differs")
        verification = read(source / "verification.json")
        require(verification["protocol_sha256"] == context["sha256"] and
                verification["return_archive"]["sha256"] == RETURN_SHA256, "evidence provenance differs")
        for name, digest in verification["source_files"].items():
            require(file_sha256(root / name) == digest, "evaluation helper changed: " + name)
        for name in ("verification.json", *[role + ".json" for role in ROLES]):
            (output / name).write_bytes((source / name).read_bytes())
    else:
        require(all((args.returned_dir, args.returned_zip, args.current_manifest, args.image_root)), "inference needs inputs and all checkpoints")
        infer(root, args, output, context, inputs, expected, current_probe)
    raw = {role: read(output / (role + ".json")) for role in ROLES}
    verification = read(output / "verification.json")
    require(all(raw[role]["model_sha256"] == verification["models"][role]["model_sha256"] for role in ROLES), "raw model binding differs")
    require(raw["current"]["clean"] == expected and raw["current"]["probe"] == current_probe, "PR35 raw arrays differ")
    results, summary = summarize(inputs, raw)
    write(output / "inputs.json", inputs)
    write(output / "scores.json", results)
    write(output / "summary.json", summary)
    if args.replay:
        require(results == read(source / "scores.json") and summary == read(source / "summary.json"), "replayed scores differ")
    files = ["inputs.json", "scores.json", "summary.json", "verification.json", *[r + ".json" for r in ROLES]]
    write(output / "status.json", {"status": "complete", "mode": "replay" if args.replay else "inference",
                                  "files": {name: file_sha256(output / name) for name in files}})
    print(json.dumps({"status": "complete", "replayed": bool(args.replay),
                      "numeric_gates_pass_against_both": summary["numeric_gates_pass_against_both"]}))


if __name__ == "__main__":
    main()
