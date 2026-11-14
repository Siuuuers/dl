#!/usr/bin/env python3
"""Build or verify one reviewed training overlay; never stage or run training.

Historical labels are independently reconstructed and checked by the existing
strict verifiers under their recorded policy snapshots. Exact reviewed tokens
are composed by immutable original row ID. A hash-bound local adjudication can
add explicitly accepted decisions; held cases are preserved as a separate ledger.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile

sys.dont_write_bytecode = True
KIND = "training_reconciled_overlay"
CANDIDATE_ID = "reconciled-labels-2026-11-14"
SLICES = {"bbox": (1, 5), "left_endpoint": (5, 8), "right_endpoint": (8, 11)}
COUNT_KEYS = ("images", "objects", "visible_endpoints", "missing_endpoints", "empty_images")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bound(base, descriptor):
    require(isinstance(descriptor, dict) and set(descriptor) >= {"path", "sha256"}, "invalid bound-file descriptor")
    relative = descriptor["path"]
    require(isinstance(relative, str) and relative, "empty bound-file path")
    path = Path(relative) if Path(relative).is_absolute() else base / relative
    require(path.is_file() and not path.is_symlink(), f"missing or symlinked bound file: {path}")
    require(sha(path) == descriptor["sha256"], f"bound file SHA-256 mismatch: {path}")
    return path.resolve()


def helpers(root):
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import parse_pose_labels
    from verify_pose_candidate import read_reference, check_geometry, totals
    return parse_pose_labels, read_reference, check_geometry, totals


def history_checks(root, inventory, records, source_bytes):
    """Invoke existing strict checks unchanged; old policies only in temp copy."""
    from verify_pose_candidate import verify_candidate
    from verify_pose_box_candidate import verify_box_candidate
    from verify_pose_annotation_candidate import verify_annotation_candidate
    from verify_pose_endpoint_candidate import verify_endpoint_candidate
    from verify_pose_composed_candidate import verify_composed_candidate
    dispatch = {"training_label_overlay": verify_candidate,
                "training_box_overlay": verify_box_candidate,
                "training_annotation_overlay": verify_annotation_candidate,
                "training_endpoint_overlay": verify_endpoint_candidate,
                "training_composed_overlay": verify_composed_candidate}
    for descriptor in inventory["historical_code"]:
        bound(root, descriptor)
    old_policy = bound(root, inventory["historical_policy"])
    current_policy = bound(root, inventory["current_policy"])
    require(current_policy.read_bytes().startswith(old_policy.read_bytes()), "historical policy core changed")
    receipts = []
    with tempfile.TemporaryDirectory(prefix="reconciled-label-history-") as temporary:
        snapshot = Path(temporary)
        def put(relative, payload):
            path = snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        put("data/manifest.json", (root / "data/manifest.json").read_bytes())
        for record in records:
            put("data/" + record["label"], source_bytes[record["id"]])
        put("docs/annotation-policy.md", old_policy.read_bytes())
        for item in inventory["historical_sources"]:
            path = bound(root, item["manifest"])
            proposal = read_json(path)
            relative_folder = Path(item["manifest"]["path"]).parent
            put(relative_folder / "manifest.json", path.read_bytes())
            if "review" in proposal:
                review_path = path.parent / proposal["review"]["path"]
                require(sha(review_path) == proposal["review"]["sha256"], "historical review hash mismatch")
                put(relative_folder / proposal["review"]["path"], review_path.read_bytes())
            for change in proposal["changes"]:
                put(relative_folder / change["candidate_label"], (path.parent / change["candidate_label"]).read_bytes())
        for descriptor in inventory["historical_evidence"]:
            path = bound(root, descriptor)
            put(descriptor["path"], path.read_bytes())
        for item in inventory["historical_sources"]:
            original = bound(root, item["manifest"])
            proposal = read_json(original)
            policy = proposal.get("annotation_policy", {}).get("sha256")
            use_old = policy == inventory["historical_policy"]["sha256"]
            target_root = snapshot if use_old else root
            target = target_root / item["manifest"]["path"]
            receipt = dispatch[proposal["kind"]](target_root, target)
            require(receipt["status"] == "verified" and receipt["candidate_manifest_sha256"] == item["manifest"]["sha256"], "historical verifier identity mismatch")
            receipts.append({"candidate_id": proposal["candidate_id"],
                             "manifest_sha256": item["manifest"]["sha256"],
                             "kind": proposal["kind"], "status": "verified",
                             "policy_context": "isolated_historical_snapshot" if use_old else "live_pinned_policy"})
    return receipts


def historical_operations(root, inventory, by_id, source_bytes):
    """Extract only emitted, accepted changes; deduplicate exact inherited fields."""
    unique = {}
    for item in inventory["historical_sources"]:
        path = bound(root, item["manifest"])
        manifest = read_json(path)
        review = read_json(path.parent / manifest["review"]["path"]) if "review" in manifest else None
        for ci, change in enumerate(manifest["changes"]):
            image_id = change["image_id"]
            require(by_id[image_id]["split"] == "train", "historical operation touches validation")
            source = source_bytes[image_id].decode().splitlines()
            def emit(field, record, ri, collection, before, after, row=None, addition_id=None):
                if manifest["kind"] != "training_label_overlay":
                    require(record.get("review_status") == "accepted_incremental", "unaccepted historical operation")
                else:
                    evidence = read_json(bound(root, inventory["dedup_acceptance"]))
                    accepted = next((x for x in evidence["primary_duplicate_decisions"] if x["image_id"] == image_id), None)
                    require(accepted and accepted["primary_agrees_with_independent_review"] and record in accepted["pairs"], "dedup lacks accepted visual agreement")
                key = f"{image_id}:source:{row}:{field}" if row is not None else f"{image_id}:addition:{addition_id}"
                origin = {"candidate_id": manifest["candidate_id"], "manifest": item["manifest"],
                          "record_pointer": f"/changes/{ci}/{collection}/{ri}",
                          "review_id": record.get("review_id"),
                          "reason": record.get("reason"),
                          "review_status": record.get("review_status", "accepted_visual_duplicate_decision")}
                if review is not None:
                    origin["review"] = {"path": (Path(item["manifest"]["path"]).parent / manifest["review"]["path"]).as_posix(), "sha256": manifest["review"]["sha256"]}
                else:
                    origin["review"] = inventory["dedup_acceptance"]
                op = {"id": key, "image_id": image_id, "source_row": row,
                      "addition_id": addition_id, "field": field,
                      "before_tokens": before, "after_tokens": after,
                      "source_label_sha256": by_id[image_id]["label_sha256"],
                      "source_image_sha256": by_id[image_id]["image_sha256"],
                      "origins": [origin]}
                if row is not None:
                    require(source[row - 1].split() == record.get("source_row_text", record.get("removed_row_text", source[row - 1])).split(), "historical source-row text mismatch")
                    op["source_row_text"] = source[row - 1]
                if field == "add_object":
                    op["append_order"] = record["candidate_row"]
                if field == "delete_object":
                    retained = record.get("retained_original_row", record.get("keep_original_row"))
                    if retained is not None:
                        op["retained_source_row"] = retained
                donor = record.get("donor_endpoints", [])
                if donor:
                    op["atomic_group"] = record["review_id"]
                    op["donor_endpoints"] = donor
                if "endpoints" in record:
                    op["endpoint_review"] = record["endpoints"]
                if key in unique:
                    old = unique[key]
                    require(old["before_tokens"] == before and old["after_tokens"] == after, f"conflicting historical field: {key}")
                    old["origins"].append(origin)
                else:
                    unique[key] = op
            for index, record in enumerate(change.get("pairs", [])):
                row = record["remove_original_row"]
                emit("delete_object", record, index, "pairs", source[row - 1].split(), None, row=row)
            for index, record in enumerate(change.get("deletions", [])):
                row = record["original_row"]
                emit("delete_object", record, index, "deletions", source[row - 1].split(), None, row=row)
            collection = "replacements" if "replacements" in change else "rows"
            for index, record in enumerate(change.get(collection, [])):
                row = record["original_row"]
                before = record["source_row_text"].split()
                after = record["candidate_row_text"].split()
                require(before[0] == after[0], "historical class change forbidden")
                for field, (lo, hi) in SLICES.items():
                    if before[lo:hi] != after[lo:hi]:
                        emit(field, record, index, collection, before[lo:hi], after[lo:hi], row=row)
            for index, record in enumerate(change.get("additions", [])):
                emit("add_object", record, index, "additions", None, record["candidate_row_text"].split(), addition_id=record["addition_id"])
    operations = sorted(unique.values(), key=lambda x: (x["image_id"], x["source_row"] or 10**9, x["id"]))
    require(len(operations) == inventory["expected_unique_operations"], "historical operation count mismatch")
    require(dict(Counter(x["field"] for x in operations)) == inventory["expected_operation_counts"], "historical field counts mismatch")
    return operations


def new_operations(local, descriptor, by_id, source_bytes):
    """Read explicit new review tokens and independently check their pixel encoding."""
    require(local.get("status") == "accepted_incremental", "local adjudication is not accepted")
    image_id = local["image_id"]
    require(image_id == "new_000038" and by_id[image_id]["split"] == "train", "only the declared new038 review is eligible")
    require(local["source_image_sha256"] == by_id[image_id]["image_sha256"] and local["source_label_sha256"] == by_id[image_id]["label_sha256"], "local adjudication source hash mismatch")
    width, height = by_id[image_id]["original_size"]
    source = source_bytes[image_id].decode().splitlines()
    def bbox_tokens(bounds):
        x1, y1, x2, y2 = bounds
        return [f"{(x1+x2)/(2*width):.8f}", f"{(y1+y2)/(2*height):.8f}", f"{(x2-x1)/width:.8f}", f"{(y2-y1)/height:.8f}"]
    def point_tokens(point):
        x, y, visible = point
        require(visible in (0, 2), "invalid new visibility")
        require(visible != 0 or (x == 0 and y == 0), "new missing slot must be canonical zero")
        return [f"{x/width:.8f}", f"{y/height:.8f}", str(visible)]
    result = []
    for index, item in enumerate(local["operations"]):
        require(item.get("status") == "accepted_reviewed"
                and item.get("review", {}).get("accepted_by") == "/root"
                and item["review"].get("human_approved") is False,
                "local operation lacks recorded root review")
        kind = item["kind"]
        row = item.get("source_row")
        if kind == "replace_bbox":
            field, after = "bbox", bbox_tokens(item["bbox_xyxy_px"])
        elif kind == "replace_endpoint":
            field = item["side"] + "_endpoint"
            require(field in SLICES and field != "bbox", "invalid endpoint side")
            after = point_tokens([*item["xy_px"], item.get("visibility", 2)])
        else:
            require(kind == "add_object" and row is None, "unsupported local operation")
            field = "add_object"
            after = ["0", *bbox_tokens(item["bbox_xyxy_px"]), *point_tokens(item["keypoints_px"][0]), *point_tokens(item["keypoints_px"][1])]
        if row is not None:
            require(row == 1, "new038 review only replaces original row1")
            lo, hi = SLICES[field]
            before = source[row - 1].split()[lo:hi]
            require(item.get("source_before_tokens") == source[row - 1].split()
                    and item.get("source_before_text") == source[row - 1],
                    f"local original row differs: {item['id']}")
            recorded_after = item["after_tokens"][lo:hi]
        else:
            before = None
            require(item.get("source_before_tokens") is None, "addition has a source row")
            recorded_after = item["after_tokens"]
        require(recorded_after == after, f"local pixel-to-token encoding differs: {item['id']}")
        require(item["after_text"].split() == item["after_tokens"], "local after text/tokens differ")
        addition_id = item.get("addition_id")
        require(row is not None or isinstance(addition_id, str) and addition_id, "added object lacks stable identity")
        key = f"{image_id}:source:{row}:{field}" if row is not None else f"{image_id}:addition:{addition_id}"
        op = {"id": key, "image_id": image_id, "source_row": row, "addition_id": addition_id,
              "field": field, "before_tokens": before, "after_tokens": after,
              "source_label_sha256": local["source_label_sha256"],
              "source_image_sha256": local["source_image_sha256"],
              "origins": [{"adjudication": descriptor, "record_pointer": f"/operations/{index}",
                           "review_id": item["id"], "review_status": "accepted_incremental",
                           "reason": item.get("reason"), "uncertainty": item.get("uncertainty")}],
              "atomic_group": "new_000038-three-front-ownership-and-border-review"}
        if row is not None:
            op["source_row_text"] = source[row - 1]
        else:
            op["append_order"] = item["final_row"]
        result.append(op)
    require(len(result) == 5 and Counter(x["field"] for x in result) == Counter(bbox=1, left_endpoint=1, add_object=3), "new038 operation scope differs")
    return result


def compose(operations, by_id, source_bytes, parse_labels, geometry, total):
    seen, per_image = set(), defaultdict(list)
    for op in operations:
        require(op["id"] not in seen, "repeated operation identity")
        seen.add(op["id"])
        per_image[op["image_id"]].append(op)
    payloads, changes, effective = {}, [], {}
    for image_id, record in sorted(by_id.items()):
        source = source_bytes[image_id].decode().splitlines()
        rows = {number: line.split() for number, line in enumerate(source, 1)}
        changed_rows, removed, added = set(), set(), []
        for op in per_image[image_id]:
            row, field = op["source_row"], op["field"]
            if field == "delete_object":
                require(row in rows and rows[row] == op["before_tokens"], "delete source mismatch")
                require(row not in removed, "duplicate deletion")
                removed.add(row)
            elif field == "add_object":
                added.append(op)
            else:
                require(row in rows, "replacement source absent")
                lo, hi = SLICES[field]
                require(source[row - 1].split()[lo:hi] == op["before_tokens"], "replace source tokens differ")
                rows[row][lo:hi] = op["after_tokens"]
                changed_rows.add(row)
        require(not (removed & changed_rows), "cannot delete and edit the same original row")
        for op in per_image[image_id]:
            if op.get("retained_source_row") is not None:
                require(op["retained_source_row"] not in removed, "dedup retained identity deleted")
            for donor in op.get("donor_endpoints", []):
                donor_row = donor["donor_original_row"]
                require(donor_row in removed, "donor consolidation deletion absent")
                start = 5 + 3 * donor["donor_endpoint"]
                target = 5 + 3 * donor["candidate_endpoint"]
                require(rows[op["source_row"]][target:target+3] == source[donor_row-1].split()[start:start+3], "donor tokens not preserved exactly")
        lines, row_map = [], []
        for number, line in enumerate(source, 1):
            if number not in removed:
                lines.append(" ".join(rows[number]) if number in changed_rows else line)
            row_map.append({"original_row": number, "candidate_row": None if number in removed else len(lines)})
        added.sort(key=lambda x: x["append_order"])
        require(len({x["append_order"] for x in added}) == len(added), "ambiguous appended addition order")
        for op in added:
            lines.append(" ".join(op["after_tokens"]))
            row_map.append({"original_row": None, "addition_id": op["addition_id"], "candidate_row": len(lines)})
        payload = ("\n".join(lines) + "\n").encode() if lines else b""
        objects = parse_labels(payload.decode(), *record["original_size"], image_id)
        geometry(objects, *record["original_size"], image_id)
        effective[image_id] = objects
        if per_image[image_id]:
            require(record["split"] == "train" and payload != source_bytes[image_id], "non-training or ineffective edit")
            relative = record["label"]
            payloads[relative] = payload
            changes.append({"image_id": image_id, "source_label": relative,
                            "candidate_label": relative, "source_label_sha256": record["label_sha256"],
                            "candidate_label_sha256": digest(payload), "source_image_sha256": record["image_sha256"],
                            "source_objects": len(source), "candidate_objects": len(lines),
                            "operation_ids": sorted(op["id"] for op in per_image[image_id]), "row_map": row_map})
        else:
            require(payload == source_bytes[image_id], "unmodified file serialization differs")
    return payloads, changes, total(list(by_id.values()), effective)


def expected(root, spec_path):
    spec = read_json(spec_path)
    require(spec.get("schema_version") == 1 and spec.get("candidate_id") == CANDIDATE_ID, "unexpected composition specification")
    require(spec.get("release_ready") is False, "release readiness cannot be claimed")
    inventory_path = bound(root, spec["source_inventory"])
    inventory = read_json(inventory_path)
    bound(root, inventory["source_manifest"])
    parse_labels, read_reference, geometry, total = helpers(root)
    _, records, by_id, source_bytes, gold, source_counts = read_reference(root, inventory["source_manifest"]["sha256"])
    for record in records:
        image_id = record["id"]
        geometry(gold[image_id], *record["original_size"], f"source:{image_id}")
        image = root / "data" / record["image"]
        require(image.is_file() and not image.is_symlink() and sha(image) == record["image_sha256"], f"original image hash differs: {image_id}")
        header = image.read_bytes()[:24]
        require(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR" and list(struct.unpack(">II", header[16:24])) == record["original_size"], f"original image dimensions differ: {image_id}")
    for descriptor in inventory["frozen_files"]:
        bound(root, descriptor)
    receipts = history_checks(root, inventory, records, source_bytes)
    historical = historical_operations(root, inventory, by_id, source_bytes)
    local_path = bound(root, spec["local_adjudication"])
    local = read_json(local_path)
    local_ops = new_operations(local, spec["local_adjudication"], by_id, source_bytes)
    operations = sorted(historical + local_ops, key=lambda x: (x["image_id"], x["source_row"] or 10**9, x["id"]))
    unresolved_path = bound(root, spec["unresolved_ledger"])
    unresolved = read_json(unresolved_path)
    payloads, changes, counts = compose(operations, by_id, source_bytes, parse_labels, geometry, total)
    local_payload = payloads[by_id[local["image_id"]]["label"]]
    require(digest(local_payload) == local["expected_result_label_sha256"],
            "local adjudication result hash differs")
    local_lines = local_payload.decode().splitlines()
    require(len(local_lines) == local["result_row_count"], "local result row count differs")
    for adjudication in local["row_adjudications"]:
        require(local_lines[adjudication["final_row"] - 1] == adjudication["after_text"],
                "local final row differs from its accepted adjudication")
    require(counts == spec["expected_counts"], "effective counts differ from frozen specification")
    require(len(changes) == spec["expected_changed_files"] and len(operations) == spec["expected_operations"], "composition scope differs")
    require(counts["val"] == source_counts["val"], "validation changed")
    review = {"schema_version": 1, "candidate_id": CANDIDATE_ID,
              "method": "Exact historical accepted tokens plus one separately accepted original-image adjudication, composed by immutable source row identity.",
              "physical_review_is_human_approval": False, "full_dataset_certified": False,
              "new_model_inference": False, "new_training": False, "release_ready": False,
              "historical_validation": receipts, "historical_unique_operations": len(historical),
              "local_operations": len(local_ops), "operations": operations,
              "unresolved_ledger": spec["unresolved_ledger"],
              "unresolved_status": unresolved.get("status"),
              "source_images_verified": len(records), "source_labels_verified": len(records),
              "source_counts": source_counts, "effective_counts": counts}
    review_payload = encoded(review)
    path_descriptor = spec_path.relative_to(root).as_posix() if spec_path.is_relative_to(root) else str(spec_path)
    manifest = {"schema_version": 1, "kind": KIND, "status": "reviewed_not_released",
                "candidate_id": CANDIDATE_ID, "source_revision": spec["source_revision"],
                "source_manifest_sha256": inventory["source_manifest"]["sha256"],
                "annotation_policy": inventory["current_policy"],
                "specification": {"path": path_descriptor, "sha256": sha(spec_path)},
                "source_inventory": spec["source_inventory"], "local_adjudication": spec["local_adjudication"],
                "unresolved_ledger": spec["unresolved_ledger"],
                "review": {"path": "review.json", "sha256": digest(review_payload)},
                "release_ready": False, "training_recipe_id": None,
                "physical_review_is_human_approval": False, "full_dataset_certified": False,
                "expected_counts": counts, "operation_counts": dict(Counter(x["field"] for x in operations)),
                "changes": changes,
                "limits": ["Reviewed observations remain fallible; exact composition is not independent physical certification.",
                           "Held, withdrawn and superseded geometry is not materialized.",
                           "Original train/validation labels, historical candidates, development references, model weights and historical validators remain unchanged.",
                           "No GPU recipe, model selection or quality gain is implied by this data candidate."]}
    payloads["review.json"] = review_payload
    payloads["manifest.json"] = encoded(manifest)
    return payloads, manifest


def verify_tree(output, payloads):
    """Verify the complete annotation payload; README.md is optional prose."""
    require(output.is_dir() and not output.is_symlink(), "candidate is not a regular directory")
    actual = set()
    for path in output.rglob("*"):
        require(not path.is_symlink(), f"candidate symlink forbidden: {path}")
        if path.is_file():
            relative = path.relative_to(output).as_posix()
            if relative == "README.md":
                continue
            actual.add(relative)
            require(relative in payloads and path.read_bytes() == payloads[relative], f"candidate bytes differ: {relative}")
        else:
            require(path.is_dir(), "nonregular candidate member")
    require(actual == set(payloads), "missing or undeclared candidate file")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("build")
    build.add_argument("--spec", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    verify = modes.add_parser("verify")
    verify.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.mode == "build":
        spec_path, output = args.spec.resolve(), args.output.absolute()
        require(not output.exists() and not output.is_symlink(), "output exists; choose a fresh candidate directory")
        for protected in ("data/labels", "data/images", "data/references", "weights"):
            require(not output.resolve().is_relative_to(root / protected), "output overlaps frozen inputs")
    else:
        output = args.candidate.resolve()
        if output.is_file():
            require(output.name == "manifest.json", "expected candidate directory or manifest.json")
            output = output.parent
        manifest = read_json(output / "manifest.json")
        spec_path = bound(root, manifest["specification"])
    payloads, manifest = expected(root, spec_path)
    if args.mode == "build":
        output.mkdir(parents=True, exist_ok=False)
        for relative, payload in payloads.items():
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    verify_tree(output, payloads)
    result = {"status": "verified", "candidate_id": CANDIDATE_ID, "kind": KIND,
              "candidate": str(output), "candidate_manifest_sha256": sha(output / "manifest.json"),
              "changed_train_files": len(manifest["changes"]), "counts": manifest["expected_counts"],
              "operation_counts": manifest["operation_counts"], "release_ready": False,
              "training_recipe_id": None, "historical_validators_unchanged": True,
              "original_images_and_labels_verified": 157, "validation_unchanged": True,
              "payload_scope": "All annotation, manifest and review bytes; optional root README.md is documentation only.",
              "training_or_inference_performed": False}
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise SystemExit(f"reconciled-label verification failed: {error}")
