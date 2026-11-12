#!/usr/bin/env python3
"""Verify the exact ten-slot endpoint control plus the frozen PR49 box delta.

Historical manifests and validators are never rewritten. Their strict checks run
in a temporary snapshot with the original labels and their pinned old policy.
The live policy is independently pinned and must retain that policy byte-for-byte
before its separately scoped section 8. This is label verification, not image
review, training, or a claim of improved model quality.
"""

import sys
sys.dont_write_bytecode = True

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from evaluate_pose import _relative_file, file_sha256, parse_pose_labels
from verify_pose_annotation_candidate import read_json
from verify_pose_candidate import (COUNT_KEYS, GEOMETRY_EPSILON_PX, check_geometry,
                                   read_reference, require, totals)


COMPOSED_KIND = "training_composed_overlay"
COMPOSED_CANDIDATE = "data/candidates/composed-fronts-2026-11-12/manifest.json"
COMPOSED_CANDIDATE_ID = "composed-fronts-2026-11-12"
COMPOSED_RECIPE_ID = "reviewed-composed-seed46-v1"
SOURCE_SHA256 = "72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926"
CONTROL = {
    "manifest": "data/candidates/endpoint-visibility-2026-11-12/manifest.json",
    "sha256": "9c489bf57a526509de73e40cc73b02117398fba4ec97107bf047aa71e4951c41",
}
PARENT = {
    "manifest": "data/candidates/annotations-2026-11-10/manifest.json",
    "sha256": "28fc8f5da95605089d61fb062132200ed296c3a10b216f1a61dbda1798ea4ba7",
}
FROZEN_SPEC = {
    "path": "results/review/next-training-20261112/combined-operations-spec.json",
    "sha256": "1492c39b5a6b3b890ab6807fc51a1ed1524f1bf57844dd285341b6b2344852b0",
}
RECOVERY = {
    "path": "results/review/baomai-review-recovery-20261112/correction-record.json",
    "sha256": "6f199638487933a04a0f2d67958d06be642ff998bd14657bfc913fe3406e24e3",
}
CURRENT_POLICY = {
    "id": "baomai-visible-front-v1", "path": "docs/annotation-policy.md",
    "sha256": "74dd943976a4abc22ecbf73551c69f715666060d5961b2bed81e6a9c73a3afc9",
}
HISTORICAL_POLICY = {
    "path": "docs/policies/annotation-policy-pr35-2026-11-10.md",
    "sha256": "3cda361ac4c5c9ba36b037e7e8968e0629eb3bca8d489c187beb8dd66e208a91",
    "snapshot_install_path": "docs/annotation-policy.md",
}
# These existing validators and their parsing/geometry helpers stay unchanged.
STRICT_CODE_SHA256 = {
    "scripts/verify_pose_endpoint_candidate.py": "9adcf9805ba9ee6fbe252c05903bab5a997e9e7c5063362d77e1f5fcccdb53d5",
    "scripts/verify_pose_annotation_candidate.py": "0dc3d263b4aad594e1019de8f8318dcd963d9c0c2f4604ad2dee408c83af7979",
    "scripts/verify_pose_candidate.py": "03395b9a82a4c120647a1cac51c89fd214d8cf9b15c04e2196a40f39077d8491",
    "scripts/evaluate_pose.py": "3f80bd09efbe97327b705ffe6ad3e662111f3e2bc71b69b51a9d38d7baa90ce0",
    "scripts/pose_scoring.py": "04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018",
}
COMPOSED_IMAGES = ("new_000023", "new_000031", "new_000045")
CONTROL_TRAIN_COUNTS = dict(zip(COUNT_KEYS, (137, 3771, 6842, 700, 13)))
TREATMENT_TRAIN_COUNTS = dict(zip(COUNT_KEYS, (137, 3770, 6842, 698, 13)))
VAL_COUNTS = dict(zip(COUNT_KEYS, (20, 664, 1227, 101, 3)))


def same(left, right):
    """JSON equality that does not mistake bools for row numbers or counts."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def bound_file(base, descriptor, name, path_key="path"):
    require(isinstance(descriptor, dict), f"{name} descriptor must be an object")
    path = _relative_file(base, descriptor[path_key])
    require(file_sha256(path) == descriptor["sha256"], f"{name} SHA-256 mismatch")
    return path


def regular_tree(folder):
    require(folder.is_dir() and not folder.is_symlink(), f"expected regular directory: {folder}")
    paths = set()
    for path in folder.rglob("*"):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()),
                f"only regular files/directories are allowed: {path}")
        if path.is_file():
            paths.add(path.relative_to(folder).as_posix())
    return paths


def policy_compatibility(root):
    historical = bound_file(root, HISTORICAL_POLICY, "historical policy").read_bytes()
    current = bound_file(root, CURRENT_POLICY, "current policy").read_bytes()
    require(current.startswith(historical), "current policy changed the historical observed-label core")
    suffix = current[len(historical):]
    require(suffix.startswith(b"\n## 8. Context-derived proposals\n")
            and b"Sections 1\xe2\x80\x937 continue to define the observed reference and current training labels." in suffix
            and b"This pilot approves no inferred point for training and does not replace a canonical missing slot." in suffix,
            "policy extension does not preserve observed-label semantics")
    return {
        "historical_core_byte_identical": True,
        "historical_core_sha256": HISTORICAL_POLICY["sha256"],
        "current_policy_sha256": CURRENT_POLICY["sha256"],
        "appended_section": "8. Context-derived proposals",
        "appended_bytes_sha256": hashlib.sha256(suffix).hexdigest(),
        "inferred_points_approved_for_training": False,
    }


def verify_historical_control(root, records, source_bytes):
    """Run the unchanged strict chain against a disposable historical snapshot."""
    parent_path = bound_file(root, PARENT, "exact PR35 parent", "manifest")
    control_path = bound_file(root, CONTROL, "exact ten-slot control", "manifest")
    control = read_json(control_path)
    control_review = read_json(bound_file(control_path.parent, control["review"], "control review"))
    evidence_path = bound_file(root, control_review["evidence"], "current endpoint evidence")
    for relative, expected in STRICT_CODE_SHA256.items():
        require(file_sha256(_relative_file(root, relative)) == expected,
                f"historical strict validator/helper changed: {relative}")
    for folder in (parent_path.parent, control_path.parent):
        regular_tree(folder)
    # pose_checkpoint is copied only as the existing endpoint recipe-ID support
    # dependency. New recipe integration may extend it; the strict validators
    # above remain pinned, and their returned old recipe ID is checked below.
    support = _relative_file(root, "scripts/pose_checkpoint.py")
    with tempfile.TemporaryDirectory(prefix="pose-composed-historical-") as temporary:
        snapshot = Path(temporary)

        def put(relative, payload):
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        put("data/manifest.json", _relative_file(root, "data/manifest.json").read_bytes())
        for record in records:
            put("data/" + record["label"], source_bytes[record["id"]])
        put("docs/annotation-policy.md", bound_file(root, HISTORICAL_POLICY, "historical policy").read_bytes())
        for descriptor, path in ((PARENT, parent_path), (CONTROL, control_path)):
            shutil.copytree(path.parent, snapshot / Path(descriptor["manifest"]).parent)
        put(control_review["evidence"]["path"], evidence_path.read_bytes())
        for relative in STRICT_CODE_SHA256:
            put(relative, _relative_file(root, relative).read_bytes())
        put("scripts/pose_checkpoint.py", support.read_bytes())
        program = (
            "import json,sys; from pathlib import Path; "
            "root=Path(sys.argv[1]); sys.path.insert(0,str(root/'scripts')); "
            "from verify_pose_endpoint_candidate import verify_endpoint_candidate; "
            "print(json.dumps(verify_endpoint_candidate(root,root/sys.argv[2]),allow_nan=False))"
        )
        result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(snapshot), CONTROL["manifest"]],
                                text=True, capture_output=True, timeout=30)
        require(result.returncode == 0,
                "unchanged historical endpoint/PR35 validation failed: " + (result.stderr or result.stdout).strip())
        receipt = json.loads(result.stdout)
    require(receipt.get("status") == "verified"
            and receipt.get("candidate_manifest_sha256") == CONTROL["sha256"]
            and receipt.get("parent_manifest_sha256") == PARENT["sha256"]
            and receipt.get("new_endpoint_edit_counts", {}).get("changed_endpoint_slots") == 10
            and receipt.get("training_recipe_id") == "reviewed-endpoints-seed46-v1"
            and same(receipt.get("counts"), {"train": CONTROL_TRAIN_COUNTS, "val": VAL_COUNTS}),
            "historical control receipt differs from the exact ten-slot control")
    return receipt


def load_frozen_inputs(root):
    spec = read_json(bound_file(root, FROZEN_SPEC, "frozen composition specification"))
    recovered = read_json(bound_file(root, RECOVERY, "frozen recovered decisions"))
    control_path = bound_file(root, CONTROL, "exact ten-slot control", "manifest")
    parent_path = bound_file(root, PARENT, "exact PR35 parent", "manifest")
    control, parent = read_json(control_path), read_json(parent_path)
    require(spec["source_manifest"]["path"] == "data/manifest.json"
            and spec["source_manifest"]["sha256"] == SOURCE_SHA256
            and spec["endpoint_manifest"]["path"] == CONTROL["manifest"]
            and spec["endpoint_manifest"]["sha256"] == CONTROL["sha256"]
            and spec["parent_manifest"]["path"] == PARENT["manifest"]
            and spec["parent_manifest"]["sha256"] == PARENT["sha256"]
            and spec["recovery_record"]["path"] == RECOVERY["path"]
            and spec["recovery_record"]["sha256"] == RECOVERY["sha256"],
            "frozen specification source/control/parent/evidence identity differs")
    return spec, recovered, control, parent


def line_text(line):
    return line.decode("utf-8").rstrip("\r\n")


def replace_tokens(line, start, stop, tokens):
    text = line_text(line)
    ending = line[len(text.encode("utf-8")):]
    require(ending in (b"", b"\n", b"\r\n"), "unsupported original line ending")
    fields = text.split()
    fields[start:stop] = tokens
    return " ".join(fields).encode("utf-8") + ending


def build_expected_composition(root, spec, recovered, control, parent, by_id, source_bytes):
    """Independently derive every overlay byte and audit row from frozen inputs."""
    control_folder = root / Path(CONTROL["manifest"]).parent
    control_changes = {item["image_id"]: item for item in control["changes"]}
    require(len(parent["changes"]) == 8 and len(control_changes) == 10
            and same(control["changes"][:8], parent["changes"])
            and same(spec["unchanged_parent_images"], [item["image_id"] for item in parent["changes"]]),
            "expected eight exact inherited files and ten control files")
    files = {item["image_id"]: item for item in spec["files"]}
    require(list(files) == list(COMPOSED_IMAGES), "frozen composition file inventory differs")
    operations = spec["operations"]
    boxes = [item for item in operations if item["kind"] == "replace_bbox"]
    endpoints = [item for item in operations if item["kind"] == "replace_endpoint"]
    deletions = [item for item in operations if item["kind"] == "delete_confirmed_duplicate"]
    require(len(operations) == 28 and len(boxes) == 17 and len(endpoints) == 10 and len(deletions) == 1,
            "expected exactly seventeen boxes, ten inherited endpoint slots, and one deletion")
    recovered_boxes = {(item["image_id"], item["original_row"]): item["bbox_xyxy_original_pixels"]
                       for item in recovered["box_edits"]}
    require(len(recovered_boxes) == 17
            and set(recovered_boxes) == {(item["image_id"], item["source_row"]) for item in boxes},
            "box operations differ from recovered decision inventory")
    require([(item["image_id"], item["original_row"]) for item in recovered["deletions"]]
            == [("new_000023", 22)], "recovered duplicate deletion differs")
    deleted = deletions[0]
    require(deleted["image_id"] == "new_000023" and deleted["source_row"] == 22
            and deleted["retained_source_row"] == 36 and deleted["retained_candidate_row"] == 35
            and deleted["candidate_row"] is None, "must delete source023 row22 and retain source row36")
    changes = copy.deepcopy(parent["changes"])
    payloads = {}
    for change in parent["changes"]:
        image_id = change["image_id"]
        payloads[image_id] = _relative_file(control_folder, change["candidate_label"]).read_bytes()
    expected_control_endpoint_keys = set()
    for image_id in COMPOSED_IMAGES:
        record, evidence_file = by_id[image_id], files[image_id]
        require(record["split"] == "train"
                and evidence_file["source_image_sha256"] == record["image_sha256"],
                f"composition source image identity differs: {image_id}")
        require(evidence_file["source_label"]["path"] == "data/" + record["label"]
                and evidence_file["source_label"]["sha256"] == record["label_sha256"],
                f"composition source label identity differs: {image_id}")
        bound_file(root, evidence_file["source_label"], "frozen original label")
        original = source_bytes[image_id].splitlines(keepends=True)
        require(all(line.strip() for line in original), "composition needs one nonblank line per source object")
        if image_id in control_changes:
            change = control_changes[image_id]
            control_bytes = _relative_file(control_folder, change["candidate_label"]).read_bytes()
            descriptor = evidence_file["current_endpoint_candidate"]
            require(descriptor["path"] == str(Path(CONTROL["manifest"]).parent / change["candidate_label"])
                    and descriptor["sha256"] == change["candidate_label_sha256"], "control label descriptor differs")
            bound_file(root, descriptor, "current endpoint label")
        else:
            require(evidence_file["current_endpoint_candidate"] is None, "unexpected endpoint control file")
            control_bytes = source_bytes[image_id]
        control_lines = control_bytes.splitlines(keepends=True)
        require(len(control_lines) == len(original), "composed files must preserve original control row identities")
        for number, (before, after) in enumerate(zip(original, control_lines), 1):
            a, b = line_text(before).split(), line_text(after).split()
            require(a[:5] == b[:5], "ten-slot control changed a composed-file box or class")
            for side, start in (("left", 5), ("right", 8)):
                if a[start:start + 3] != b[start:start + 3]:
                    expected_control_endpoint_keys.add((image_id, number, side))
        image_ops = [item for item in operations if item["image_id"] == image_id]
        removed = {item["source_row"] for item in image_ops if item["kind"] == "delete_confirmed_duplicate"}
        row_map, next_row = [], 0
        for number in range(1, len(original) + 1):
            if number not in removed:
                next_row += 1
            row_map.append({"original_row": number, "candidate_row": None if number in removed else next_row})
        expected_spec_map = {"source_rows": len(original), "result_rows": next_row,
                             "source_to_candidate": [{"source_row": row["original_row"],
                                                      "candidate_row": row["candidate_row"]} for row in row_map]}
        require(same(spec["row_maps"][image_id], expected_spec_map), f"frozen full row map differs: {image_id}")
        treatment, box_only = list(control_lines), list(original)
        operation_keys = set()
        for operation in image_ops:
            number, kind = operation["source_row"], operation["kind"]
            require(type(number) is int and 1 <= number <= len(original)
                    and operation["candidate_row"] == row_map[number - 1]["candidate_row"],
                    f"operation source/candidate identity differs: {image_id}")
            key = (number, kind, operation.get("side"))
            require(key not in operation_keys, "duplicate composition operation")
            operation_keys.add(key)
            before, fixed_control = line_text(original[number - 1]).split(), line_text(control_lines[number - 1]).split()
            if kind == "replace_bbox":
                require(operation["before_tokens"] == before[1:5]
                        and operation["bbox_xyxy_px"] == recovered_boxes[(image_id, number)]
                        and operation["evidence_path"] == RECOVERY["path"], "bbox tokens/provenance differ")
                treatment[number - 1] = replace_tokens(treatment[number - 1], 1, 5, operation["after_tokens"])
                box_only[number - 1] = replace_tokens(box_only[number - 1], 1, 5, operation["after_tokens"])
                obj = parse_pose_labels(line_text(treatment[number - 1]), *record["original_size"])[0]
                require(all(abs(a - b) <= GEOMETRY_EPSILON_PX for a, b in
                            zip(obj["bbox_xyxy"], operation["bbox_xyxy_px"])), "bbox pixels differ from normalized tokens")
            elif kind == "replace_endpoint":
                start = {"left": 5, "right": 8}[operation["side"]]
                require(operation["before_tokens"] == before[start:start + 3]
                        and operation["after_tokens"] == fixed_control[start:start + 3],
                        "endpoint operation must preserve exact current ten-slot control tokens")
            else:
                require(kind == "delete_confirmed_duplicate"
                        and operation["original_line"] == line_text(original[number - 1])
                        and all(float(value) == 0 for value in fixed_control[5:]),
                        "duplicate deletion must remove the exact both-missing source row")
        box_bytes = b"".join(line for number, line in enumerate(box_only, 1) if number not in removed)
        require(bound_file(root, evidence_file["box_proposal"], "recorded box proposal").read_bytes() == box_bytes,
                "recovered box-only evidence is not the exact recorded box/deletion operations")
        result_bytes = b"".join(line for number, line in enumerate(treatment, 1) if number not in removed)
        require(hashlib.sha256(result_bytes).hexdigest() == evidence_file["result_sha256"],
                f"reconstructed treatment differs from frozen prototype: {image_id}")
        payloads[image_id] = result_bytes
        replacements, deletion_records = [], []
        for number in range(1, len(original) + 1):
            mapped = row_map[number - 1]["candidate_row"]
            row_operations = [item for item in image_ops if item["source_row"] == number]
            if not row_operations:
                require(treatment[number - 1] == original[number - 1], "unlisted source row changed")
                continue
            audit = {
                "original_row": number, "candidate_row": mapped,
                "source_row_text": line_text(original[number - 1]),
                "control_row_text": line_text(control_lines[number - 1]),
                "review_id": f"composed-{image_id}-source-{number}",
                "review_status": "accepted_incremental",
                "reason": "Exact composition of previously recorded decisions; no new visual approval or inferred coordinate.",
                "frozen_operations": copy.deepcopy(row_operations),
            }
            if mapped is None:
                audit.update(operation="delete_confirmed_duplicate", retained_original_row=36,
                             retained_candidate_row=35)
                deletion_records.append(audit)
            else:
                after = line_text(treatment[number - 1]).split()
                fixed_control = line_text(control_lines[number - 1]).split()
                require(after[0] == fixed_control[0] and after[5:] == fixed_control[5:],
                        "treatment changed class or endpoint tokens relative to control")
                before = line_text(original[number - 1]).split()
                fields = (("bbox", 1, 5), ("left_endpoint", 5, 8), ("right_endpoint", 8, 11))
                audit.update(operation="composed_amendment", candidate_row_text=line_text(treatment[number - 1]),
                             changed_fields=[name for name, a, b in fields if before[a:b] != after[a:b]],
                             changed_fields_from_control=[name for name, a, b in fields if fixed_control[a:b] != after[a:b]],
                             donor_endpoints=[])
                replacements.append(audit)
        changes.append({
            "image_id": image_id, "source_label": record["label"], "candidate_label": record["label"],
            "source_label_sha256": record["label_sha256"],
            "candidate_label_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "source_image_sha256": record["image_sha256"], "source_objects": len(original),
            "candidate_objects": next_row, "deletions": deletion_records,
            "replacements": replacements, "additions": [], "row_map": row_map,
        })
    require(expected_control_endpoint_keys == {(item["image_id"], item["source_row"], item["side"]) for item in endpoints}
            and len(expected_control_endpoint_keys) == 10, "frozen specification omits or adds control endpoint slots")
    return changes, payloads


def expected_review(source_revision, spec, compatibility):
    return {
        "schema_version": 1, "candidate_id": COMPOSED_CANDIDATE_ID,
        "source_revision": source_revision, "control": CONTROL, "historical_parent": PARENT,
        "frozen_specification": FROZEN_SPEC, "recovered_decisions": RECOVERY,
        "annotation_policy": CURRENT_POLICY, "historical_policy": HISTORICAL_POLICY,
        "policy_compatibility": compatibility,
        "method": "Mechanical composition of frozen accepted decisions, with unchanged strict historical validators in an isolated historical-policy snapshot.",
        "operations": spec["operations"], "row_maps": spec["row_maps"],
        "new_visual_review": False, "new_model_inference": False, "new_training": False,
        "inferred_coordinates_used": False,
    }


def verify_composed_candidate(root, candidate):
    root = Path(root).resolve()
    candidate = Path(candidate)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    proposal = read_json(candidate)
    require(type(proposal.get("schema_version")) is int and proposal["schema_version"] == 1
            and proposal.get("kind") == COMPOSED_KIND and proposal.get("status") == "proposed",
            "expected a proposed training_composed_overlay schema 1")
    require(proposal.get("candidate_id") == COMPOSED_CANDIDATE_ID
            and proposal.get("training_recipe_id") == COMPOSED_RECIPE_ID, "composed candidate/recipe identity differs")
    require(isinstance(proposal.get("source_revision"), str)
            and re.fullmatch(r"[0-9a-f]{40}", proposal["source_revision"]), "expected full source revision SHA")
    for field, expected in (("control", CONTROL), ("historical_parent", PARENT),
                            ("frozen_specification", FROZEN_SPEC), ("recovered_decisions", RECOVERY),
                            ("annotation_policy", CURRENT_POLICY), ("historical_policy", HISTORICAL_POLICY),
                            ("historical_validator_sha256", STRICT_CODE_SHA256)):
        require(same(proposal.get(field), expected), f"composed {field} identity differs")
    require(proposal.get("source_manifest_sha256") == SOURCE_SHA256, "composed source manifest identity differs")
    compatibility = policy_compatibility(root)
    require(same(proposal.get("policy_compatibility"), compatibility), "policy compatibility proof differs")
    spec, recovered, control, parent = load_frozen_inputs(root)
    reference_hash, records, by_id, source_bytes, gold, source_counts = read_reference(root, SOURCE_SHA256)
    control_receipt = verify_historical_control(root, records, source_bytes)
    expected_changes, payloads = build_expected_composition(root, spec, recovered, control, parent, by_id, source_bytes)
    require(same(proposal.get("changes"), expected_changes),
            "composed change records/row maps differ from exact frozen operations")
    review_path = bound_file(candidate.parent, proposal["review"], "composed review")
    require(same(read_json(review_path), expected_review(proposal["source_revision"], spec, compatibility)),
            "composed review differs from frozen operations and provenance")
    require(regular_tree(candidate.parent / "labels") ==
            {str(Path(item["candidate_label"]).relative_to("labels")) for item in expected_changes},
            "unexpected or missing composed overlay labels")
    effective = dict(gold)
    for change in expected_changes:
        image_id = change["image_id"]
        path = _relative_file(candidate.parent, change["candidate_label"])
        require(path.read_bytes() == payloads[image_id]
                and file_sha256(path) == change["candidate_label_sha256"],
                f"unlisted rows/bytes changed in composed candidate: {image_id}")
        effective[image_id] = parse_pose_labels(path.read_text(encoding="utf-8"), *by_id[image_id]["original_size"], str(path))
    for record in records:
        check_geometry(effective[record["id"]], *record["original_size"], record["id"])
    counts = totals(records, effective)
    require(same(counts, {"train": TREATMENT_TRAIN_COUNTS, "val": VAL_COUNTS})
            and same(proposal.get("expected_counts"), counts) and counts["val"] == source_counts["val"],
            "composed counts or unchanged validation totals differ")
    train_members = spec["effective_train_members"]
    require(len(train_members) == 137 and len({item["image_id"] for item in train_members}) == 137
            and {item["image_id"] for item in train_members} == {r["id"] for r in records if r["split"] == "train"},
            "frozen effective training inventory differs")
    for item in train_members:
        require(hashlib.sha256(payloads.get(item["image_id"], source_bytes[item["image_id"]])).hexdigest() == item["sha256"],
                f"effective training bytes differ from frozen specification: {item['image_id']}")
    edits = {"changed_train_files": 11, "added_rows": control_receipt["added_objects"],
             "deleted_rows": control_receipt["removed_objects"] + 1,
             "replaced_rows": sum(len(item["replacements"]) for item in expected_changes),
             "donor_endpoints_transferred": control_receipt["merged_endpoints"],
             "net_object_change": counts["train"]["objects"] - source_counts["train"]["objects"]}
    delta = {"changed_train_files": 3, "changed_box_rows": 17, "deleted_rows": 1,
             "changed_endpoint_slots": 0, "added_rows": 0, "net_object_change": -1,
             "net_visible_endpoint_change": 0, "net_missing_endpoint_change": -2}
    require(same(proposal.get("expected_edit_counts"), edits)
            and same(proposal.get("delta_from_control"), delta), "composed edit totals differ")
    return {
        "status": "verified", "kind": COMPOSED_KIND, "candidate_id": COMPOSED_CANDIDATE_ID,
        "candidate_manifest_sha256": file_sha256(candidate), "source_manifest_sha256": reference_hash,
        "declared_source_revision": proposal["source_revision"], "source_revision_git_object_checked": False,
        "reference_labels_verified": len(records), "changed_train_labels": 11,
        "added_objects": edits["added_rows"], "removed_objects": edits["deleted_rows"],
        "replaced_objects": edits["replaced_rows"], "merged_endpoints": edits["donor_endpoints_transferred"],
        "counts": counts, "changed_box_rows": 17, "delta_from_control": delta,
        "control": {"manifest": CONTROL["manifest"], "manifest_sha256": CONTROL["sha256"],
                    "candidate_id": control_receipt["candidate_id"], "kind": control_receipt["kind"],
                    "counts": control_receipt["counts"], "changed_train_labels": 10,
                    "changed_endpoint_slots": 10},
        "parent_manifest_sha256": PARENT["sha256"], "review_sha256": proposal["review"]["sha256"],
        "annotation_policy_sha256": CURRENT_POLICY["sha256"], "policy_compatibility": compatibility,
        "historical_validation": {"isolated_snapshot": True, "original_label_files": 157,
                                  "policy_sha256": HISTORICAL_POLICY["sha256"],
                                  "strict_validator_sha256": STRICT_CODE_SHA256,
                                  "parent_and_control_manifests_unchanged": True},
        "validation_labels_unchanged": True, "parent_labels_and_reviews_preserved": True,
        "unchanged_rows_byte_preserved": True, "full_row_maps_verified": True,
        "endpoint_tokens_preserved_from_control": True, "image_bytes_verified": False,
        "images_staged_by_check": False, "dataset_built": False, "visual_correctness_verified": False,
        "mechanical_geometry_verified": True, "training_recipe_supported": True,
        "training_recipe_id": COMPOSED_RECIPE_ID,
        "code_sha256": {**STRICT_CODE_SHA256,
                        "scripts/verify_pose_composed_candidate.py": file_sha256(Path(__file__).resolve()),
                        "scripts/pose_checkpoint.py": file_sha256(root / "scripts/pose_checkpoint.py")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", type=Path, default=Path(COMPOSED_CANDIDATE))
    args = parser.parse_args()
    try:
        result = verify_composed_candidate(args.root, args.candidate)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, allow_nan=False))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
