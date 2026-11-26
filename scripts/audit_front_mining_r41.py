#!/usr/bin/env python3
"""Replay saved R41 records and provenance; never run model inference or train."""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

ARCHIVE_SHA = "c73f3cafd916532e727cd6467f580253fa2dd3778f3f19169eb9abc45972f218"
NOTEBOOK_SHA = "f03151ff2549b936718a75ddc84ea0941fc1b88dbf22624d6898af0d7e1cf827"
SUMMARY_FIELDS = (
    "image_id", "source_image_sha256", "source_label_sha256", "source_rows",
    "source_partial_rows", "primary_outputs", "filtered_outputs", "suppressed",
    "source_unmatched_rows", "source_unmatched_outputs", "source_duplicate_subset", "priority",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_file(path, expected, size=None):
    data = path.read_bytes()
    check(digest(data) == expected, f"SHA256 mismatch: {path}")
    check(size is None or len(data) == size, f"Size mismatch: {path}")
    return path


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(data):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(data, object_pairs_hook=strict_object, parse_constant=reject)


def archive_names(z):
    items = z.infolist()
    names = [item.filename for item in items]
    check(len(names) == len(set(names)), "Duplicate ZIP paths")
    for item in items:
        path = PurePosixPath(item.filename)
        check(not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename,
              f"Unsafe ZIP path: {item.filename}")
        check((item.external_attr >> 16) & 0o170000 != 0o120000,
              f"Symlink ZIP member: {item.filename}")
    check(z.testzip() is None, "ZIP CRC mismatch")
    return set(names)


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def function_subset(cell, names, namespace, notebook_path):
    parsed = ast.parse("".join(cell["source"]))
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    check({node.name for node in selected} == set(names), "Notebook function selection mismatch")
    # Only the inspected pure/replay functions are defined. No notebook top-level execution.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(notebook_path), "exec"), namespace)


class _NotebookPoolCache:
    """Map the notebook's CACHE / 'pool' reads to an explicit CLI input root."""

    def __init__(self, pool_root):
        self.pool_root = pool_root

    def __truediv__(self, child):
        if child != "pool":
            raise ValueError(f"Unexpected notebook cache access during replay: {child}")
        return self.pool_root


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify PR127/R41 artifacts and replay all saved predictions; no model inference or training.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root containing pinned R40 scripts and metadata (default: this script's Git root).")
    parser.add_argument("--notebook", type=Path,
                        help="Original R41 notebook (default: <repo>/notebooks/baomai_front_mining_r41.ipynb).")
    parser.add_argument("--return-archive", type=Path, required=True,
                        help="Original PR127 baomai-front-mining-r41-results.zip.")
    parser.add_argument("--return-root", type=Path, required=True,
                        help="Already extracted return directory; every byte is checked against the ZIP.")
    parser.add_argument("--pool-archive", type=Path, required=True,
                        help="Pinned upstream reviewed320 ZIP.")
    parser.add_argument("--pool-root", type=Path, required=True,
                        help="Extracted pool directory containing images/ and labels/.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination audit JSON, written only after all checks pass.")
    args = parser.parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    if args.notebook is None:
        args.notebook = args.repo / "notebooks/baomai_front_mining_r41.ipynb"
    for name in ("notebook", "return_archive", "return_root", "pool_archive", "pool_root", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    protected_files = {args.notebook, args.return_archive, args.pool_archive}
    protected_roots = (args.return_root, args.pool_root, args.repo / "scripts", args.repo / "data", args.repo / "notebooks")
    if args.output in protected_files or any(args.output == root or root in args.output.parents for root in protected_roots):
        parser.error("--output must be outside the audited input files and source directories")
    return args


def main(argv=None):
    args = parse_args(argv)
    RETURNED, REPO, NOTEBOOK = args.return_root, args.repo, args.notebook
    verify_file(args.return_archive, ARCHIVE_SHA, 41161728)
    inventory = strict_json((RETURNED / "inventory.json").read_text())
    with zipfile.ZipFile(args.return_archive) as z:
        names = archive_names(z)
        check(len(names) == 97 and names == set(inventory) | {"inventory.json"}, "Return member set mismatch")
        check(sum(i.file_size for i in z.infolist()) == 49065846, "Return expanded size mismatch")
        check({p.relative_to(RETURNED).as_posix() for p in RETURNED.rglob("*") if p.is_file()} == names,
              "Extracted member set mismatch")
        for name in names:
            contents = z.read(name)
            check(contents == (RETURNED / name).read_bytes(), f"Extracted bytes mismatch: {name}")
            if name in inventory:
                check(inventory[name] == {"sha256": digest(contents), "bytes": len(contents)},
                      f"Inventory entry mismatch: {name}")

    verify_file(NOTEBOOK, NOTEBOOK_SHA, 29478)
    notebook = strict_json(NOTEBOOK.read_text())
    constants = {}
    for node in ast.parse("".join(notebook["cells"][1]["source"])).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    receipt = strict_json((RETURNED / "receipt.json").read_text())
    for field, constant in {
        "source_revision": "R40", "upstream_revision": "UPSTREAM", "model_return_revision": "PR123",
        "checkpoint_sha256": "WEIGHT_SHA", "pool_archive_sha256": "POOL_SHA",
        "model_return_archive_sha256": "MODEL_ARCHIVE_SHA", "source_scripts_sha256": "SCRIPTS",
    }.items():
        check(receipt[field] == constants[constant], f"Receipt/notebook pin mismatch: {field}")
    for name, sha in constants["SCRIPTS"].items():
        verify_file(REPO / "scripts" / name, sha)
    intake_path = REPO / "docs/additional-frame-intake-2026-11-07.json"
    pilot_path = REPO / constants["PILOT"] / "manifest.json"
    verify_file(intake_path, receipt["intake_sha256"])
    verify_file(pilot_path, receipt["pilot_manifest_sha256"])
    check(receipt["intake_sha256"] == "6754ce0994dfb4f2587117bd213df9896283f87b4dacf0d7a714562d41e8a8c6",
          "Intake pin mismatch")
    pilot_bytes = pilot_path.read_bytes()
    check(hashlib.sha1(f"blob {len(pilot_bytes)}\0".encode() + pilot_bytes).hexdigest()
          == "830b4e45d9a2aceedd7bcd7a22731b57b41070be", "Pilot Git blob pin mismatch")
    intake, pilot = strict_json(intake_path.read_text()), strict_json(pilot_bytes)
    source_records = sorted(intake["reviewed320_records"], key=lambda r: r["image"])
    source_by_id = {Path(r["image"]).stem: r for r in source_records}
    check(len(source_records) == len(source_by_id) == len({r["image_sha256"] for r in source_records}) == 320,
          "Pool records/ID/content uniqueness mismatch")
    check(intake["source"]["commit"] == constants["UPSTREAM"]
          and intake["source"]["zip_sha256"] == constants["POOL_SHA"], "Intake upstream pin mismatch")
    check(pilot["source_commit"] == constants["UPSTREAM"] and not pilot["eligible_image_ids"]
          and tuple(pilot["selected_image_ids"]) == constants["PILOT_IDS"], "Pilot status mismatch")
    pilot_by_id = {r["id"]: r for r in pilot["images"]}
    check(all(r["disposition"] == "quarantined" for r in pilot_by_id.values()), "Pilot disposition mismatch")
    verify_file(args.pool_archive, constants["POOL_SHA"], 43424432)
    with zipfile.ZipFile(args.pool_archive) as z:
        pool_names = archive_names(z)
        for record in source_records:
            for kind in ("image", "label"):
                rel = record[kind]
                matches = [name for name in pool_names if name == rel or name.endswith("/" + rel)]
                check(len(matches) == 1, f"Ambiguous pool ZIP member: {rel}")
                contents = z.read(matches[0])
                check(digest(contents) == record[kind + "_sha256"], f"Pool manifest mismatch: {rel}")
                check(contents == (args.pool_root / rel).read_bytes(), f"Extracted pool mismatch: {rel}")

    sys.path.insert(0, str(REPO / "scripts"))
    import evaluate_pose as evaluator
    import predict_pose_folder as runner
    from pose_scoring import score_image
    from PIL import Image

    for module_name in constants["SCRIPTS"]:
        module = sys.modules[module_name[:-3]]
        check(Path(module.__file__).resolve() == (REPO / "scripts" / module_name).resolve(),
              f"Loaded unpinned module; run the audit in a fresh process: {module_name}")

    check(receipt["inference"] == {**evaluator.INFERENCE, "conf": .13, "endpoint_confidence": .50,
                                 "postclip_nms_iou": .70, "source_comparison_iou": .50},
          "Inference settings mismatch")
    check(receipt["kind"] == "r41-unqualified-upstream-review-mining" and receipt["fresh_inference"] is True
          and receipt["completed_images"] == 320 and receipt["training"] is False
          and receipt["new_approved_training_images"] == 0 and receipt["automatic_promotion"] is False,
          "Receipt scope mismatch")
    runtime = receipt["runtime"]
    check(runtime["ultralytics"] == "8.4.182" and runtime["loaded_kpt_shape"] == [2, 3]
          and runtime["device_actual"] == "cuda:0" and runtime["device_name"] == "Tesla T4"
          and runtime["backend_end2end"] is False and runtime["backend_fp16"] is False,
          "Recorded runtime/pose/backend contract mismatch")
    namespace = {
        "Path": Path, "CACHE": _NotebookPoolCache(args.pool_root), "runner": runner, "evaluator": evaluator,
        "score_image": score_image, "verify_file": verify_file, "ENDPOINT_CONF": .50,
        "PILOT_IDS": constants["PILOT_IDS"],
    }
    function_subset(notebook["cells"][3], {"process_frame"}, namespace, NOTEBOOK)
    function_subset(notebook["cells"][5], {"frame_number", "choose_review"}, namespace, NOTEBOOK)
    details = [strict_json(line) for line in (RETURNED / "predictions.jsonl").read_text().splitlines()]
    check(len(details) == 320 and [d["image_id"] for d in details] == list(source_by_id),
          "Prediction record completeness/order mismatch")
    summaries, suppression_rows = [], []
    endpoint_counts = {"displayed": 0, "below_display_threshold": 0, "outside_image": 0}
    for detail in details:
        name = detail["image_id"]
        record = source_by_id[name]
        width, height = record["width"], record["height"]
        original = {}
        for prediction in detail["predictions"]:
            primary_id = prediction["primary_prediction_id"]
            check(type(primary_id) is int and primary_id not in original, f"Repeated retained ID: {name}")
            original[primary_id] = {k: v for k, v in prediction.items() if k != "primary_prediction_id"}
        for item in detail["postclip_nms"]["suppressed_predictions"]:
            primary_id = item["primary_prediction_id"]
            check(type(primary_id) is int and primary_id not in original, f"Repeated suppressed ID: {name}")
            original[primary_id] = item["prediction"]
            suppression_rows.append({"image_id": name, **{k: v for k, v in item.items() if k != "prediction"}})
        check(set(original) == set(range(1, detail["primary_outputs"] + 1)), f"Primary ID gap: {name}")
        check(0 <= len(original) <= receipt["inference"]["max_det"], f"Primary output count invalid: {name}")
        raw = []
        for primary_id in sorted(original):
            prediction = original[primary_id]
            box = prediction["bbox_xyxy"]
            check(0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height,
                  f"Unclipped or nonpositive prediction box: {name}/{primary_id}")
            check(prediction["confidence"] >= .13, f"Prediction below box threshold: {name}/{primary_id}")
            for point, status in zip(prediction["keypoints"], prediction["endpoint_status"]):
                check(all(isinstance(v, (int, float)) and math.isfinite(v) for v in point),
                      f"Nonfinite endpoint: {name}/{primary_id}")
                check(all(type(v) is bool for v in status.values()), f"Nonboolean endpoint status: {name}/{primary_id}")
                endpoint_counts["displayed"] += int(status["displayed"])
                endpoint_counts["below_display_threshold"] += int(not status["passes_display_threshold"])
                endpoint_counts["outside_image"] += int(not status["in_image"])
            raw.append({k: v for k, v in prediction.items() if k != "endpoint_status"})
        for value in [*detail["stage_ms"].values(), detail["predict_call_wall_ms"]]:
            check(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                  f"Invalid recorded timing: {name}")
        def replay_saved_predictions(_rgb):
            return {"predictions": raw, "stage_ms": detail["stage_ms"],
                    "predict_call_wall_ms": detail["predict_call_wall_ms"]}
        summary, replayed = namespace["process_frame"](record, replay_saved_predictions)
        check(replayed == detail, f"Saved prediction/source-comparison replay mismatch: {name}")
        check(tuple(summary) == SUMMARY_FIELDS, f"Summary schema mismatch: {name}")
        summaries.append(summary)

    selected = namespace["choose_review"](summaries)
    check(len(selected) == len({r["image_id"] for r in selected}) == 12, "Review selection size mismatch")
    check([r["image_id"] for r in selected] == receipt["selected_review_ids"], "Receipt selection mismatch")
    check(csv_bytes(selected) == (RETURNED / "review-queue.csv").read_bytes(), "Review queue CSV byte mismatch")
    priorities = sorted(summaries, key=lambda r: (-r["priority"], -r["source_partial_rows"], r["image_id"]))
    check(csv_bytes(priorities) == (RETURNED / "all-frame-priorities.csv").read_bytes(), "Priority CSV byte mismatch")
    check({p.name for p in (RETURNED / "review").iterdir() if p.is_dir()}
          == set(receipt["selected_review_ids"]), "Review directory membership mismatch")
    detail_by_id = {r["image_id"]: r for r in details}
    selected_audit = []
    total_rows, missing_slots, visible_slots = 0, 0, 0
    for selected_row in selected:
        name = selected_row["image_id"]
        record = source_by_id[name]
        directory = RETURNED / "review" / name
        verify_file(directory / "original.jpg", record["image_sha256"])
        verify_file(directory / "source.txt", record["label_sha256"])
        rgb, meta = runner.decode_image(args.pool_root / record["image"])
        with Image.open(directory / (name + ".png")) as png:
            check(png.format == "PNG" and png.size == rgb.size and png.convert("RGB").tobytes() == rgb.tobytes(),
                  f"Paired PNG is not a lossless source decode: {name}")
        if name in pilot_by_id:
            entry = pilot_by_id[name]
            original_proposal = REPO / constants["PILOT"] / entry["candidate_label"]
            verify_file(original_proposal, entry["candidate_label_sha256"])
            expected_label = original_proposal.read_bytes()
            check(strict_json((directory / "pilot-review.json").read_text()) == entry,
                  f"Pilot metadata mismatch: {name}")
            label_origin = "previous_quarantined_pilot_proposal"
        else:
            expected_label = (args.pool_root / record["label"]).read_bytes()
            label_origin = "unqualified_upstream_source"
        label_path = directory / (name + ".txt")
        check(label_path.read_bytes() == expected_label, f"Provisional label source mismatch: {name}")
        check(strict_json((directory / "comparison.json").read_text()) == detail_by_id[name],
              f"Selected comparison mismatch: {name}")
        labels = evaluator.parse_pose_labels(label_path.read_text(), rgb.width, rgb.height, str(label_path))
        total_rows += len(labels)
        for i, label in enumerate(labels, 1):
            x1, y1, x2, y2 = label["bbox_xyxy"]
            check(-1e-4 <= x1 < x2 <= rgb.width + 1e-4 and -1e-4 <= y1 < y2 <= rgb.height + 1e-4,
                  f"Provisional bbox outside image: {name}/row{i}")
            for x, y, visibility in label["keypoints"]:
                missing_slots += int(visibility == 0)
                visible_slots += int(visibility == 2)
                if visibility == 2:
                    check(0 <= x < rgb.width and 0 <= y < rgb.height,
                          f"Visible provisional endpoint outside image: {name}/row{i}")
                    check(x1 - 1 <= x <= x2 + 1 and y1 - 1 <= y <= y2 + 1,
                          f"Visible provisional endpoint outside own box by more than 1px: {name}/row{i}")
        selected_audit.append({
            **selected_row, "provisional_label_origin": label_origin,
            "provisional_label_sha256": digest(expected_label), "provisional_rows": len(labels),
            "source_jpeg_and_png_pixel_equal": True, "schema_and_mechanical_geometry_pass": True,
        })

    totals = {field: sum(r[field] for r in summaries) for field in SUMMARY_FIELDS[3:]}
    selected_totals = {field: sum(r[field] for r in selected) for field in SUMMARY_FIELDS[3:]}
    audit = {
        "kind": "r42-read-only-r41-saved-output-integrity-audit", "passed": True,
        "return_pr": "https://github.com/Siuuuers/dl/pull/127",
        "return_commit": "1aa89408089ea848ed7557cc33e195d42612f1f5",
        "return_archive": {"sha256": ARCHIVE_SHA, "bytes": 41161728, "members": 97,
                           "uncompressed_bytes": 49065846, "inventory_payloads_verified": 96,
                           "inventory_sha256": digest((RETURNED / "inventory.json").read_bytes()),
                           "crc_paths_and_extracted_bytes_pass": True},
        "original_notebook": {"sha256": NOTEBOOK_SHA, "bytes": 29478,
                              "source": "original Library handoff; not present in R40 Git source"},
        "verified_source_revision": receipt["source_revision"],
        "verified_source_scripts_sha256": receipt["source_scripts_sha256"],
        "verified_intake_sha256": receipt["intake_sha256"],
        "verified_pilot_manifest_sha256": receipt["pilot_manifest_sha256"],
        "pool": {"upstream_revision": receipt["upstream_revision"],
                 "sha256": constants["POOL_SHA"], "bytes": 43424432,
                 "archive_members": len(pool_names), "manifested_payloads_hash_verified": 640,
                 "unique_frame_ids": 320, "unique_source_image_hashes": 320},
        "recorded_runtime": runtime, "verified_inference_settings": receipt["inference"],
        "saved_output_replay": {"records": 320, "complete_intake_order": True,
                                "all_primary_ids_reconstructable": True,
                                "all_primary_boxes_and_endpoint_statuses_valid": True,
                                "postclip_suppression_reproduced_exactly": True,
                                "all_source_comparison_details_reproduced_exactly": True,
                                "all_frame_priority_csv_reproduced_byte_exactly": True,
                                "review_queue_csv_reproduced_byte_exactly": True},
        "old_unqualified_source_disagreement_totals_not_accuracy": totals,
        "selected_old_unqualified_source_disagreement_totals_not_accuracy": selected_totals,
        "primary_endpoint_display_status_counts_not_accuracy": endpoint_counts,
        "suppression_audit": suppression_rows,
        "selected_review": selected_audit,
        "provisional_label_mechanical_totals": {"frames": 12, "rows": total_rows,
                                               "missing_endpoint_slots": missing_slots,
                                               "visible_endpoint_slots": visible_slots,
                                               "geometry_flags": 0},
        "limits": [
            "Saved prediction records were replayed; no new model inference or training was run.",
            "Runtime and checkpoint execution are attested by the returned receipt and checked against notebook pins; weights were not downloaded or re-executed in this audit.",
            "T4 CUDA runtime is authorized for this mining notebook; it is not an exact repeat of a prior CPU benchmark.",
            "Source rows and previous pilot proposals remain unqualified; unmatched and duplicate-subset counts are mining disagreements, not established physical errors or accuracy.",
            "Mechanical geometry success does not establish that a box or endpoint matches a physically visible cassette front.",
            "A suppressed prediction is not automatically a physical duplicate; suppression correctness requires physical review.",
            "Filename bucket spacing and distinct image hashes do not establish independent capture sessions.",
            "No training labels, training eligibility, model promotion or production defaults are approved by this audit.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "frames": 320, "selected_frames": 12,
                      "provisional_rows": total_rows, "totals_not_accuracy": totals,
                      "selected_totals_not_accuracy": selected_totals,
                      "suppression_frames": len({r["image_id"] for r in suppression_rows}),
                      "audit": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
