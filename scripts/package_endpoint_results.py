#!/usr/bin/env python3
"""Package one fixed PR35-label versus reviewed-endpoint CUDA comparison.

Both inner arms remain candidate. The numerical recipes match, but their IDs
truthfully identify different annotation contracts. No model is loaded here.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

from evaluate_pose import file_sha256, preflight_dataset
from package_gpu_results import INDEX_NAME, collect_arm, copy_member, shared_settings, verify_archive
from pose_checkpoint import (ANNOTATION_CONFIRMATION_RECIPE_ID, ENDPOINT_RECIPE_ID,
    _contained_path, _read_json, _require, _same, recipe_identity)
from train_pose_comparison import source_identity
from verify_pose_annotation_candidate import verify_annotation_candidate
from verify_pose_endpoint_candidate import verify_endpoint_candidate


ROLES = ("old_reviewed", "endpoint_reviewed")
ROLE_RECIPE_IDS = dict(zip(ROLES, (ANNOTATION_CONFIRMATION_RECIPE_ID, ENDPOINT_RECIPE_ID)))
PROTOCOL_KIND = "baomai-endpoint-comparison-protocol"
RESULT_KIND = "baomai-endpoint-comparison-results"
DEFAULT_PROTOCOL = "experiments/endpoint-visibility-2026-11-12.json"
OLD_MANIFEST = "data/candidates/annotations-2026-11-10/manifest.json"
OLD_SHA256 = "28fc8f5da95605089d61fb062132200ed296c3a10b216f1a61dbda1798ea4ba7"
NEW_MANIFEST = "data/candidates/endpoint-visibility-2026-11-12/manifest.json"
NEW_SHA256 = "9c489bf57a526509de73e40cc73b02117398fba4ec97107bf047aa71e4951c41"
CANDIDATE_SOURCE = "5f547d6166099faf39bc5b3bf9c044333d1682f0"
RECIPE_SHA256 = "f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8"
INITIAL_SHA256 = "e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e"
SOURCE_SHA256 = "72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926"
PRIMARY_REFERENCE = "data/references/visible-front-2026-11-11/manifest.json"
PRIMARY_REFERENCE_SHA256 = "2e60b3e81195448cc565a1000b4305e3215baba2bf0ee0c976e6551c2e64937c"
ADJUDICATION = {"path": "results/review/baomai-endpoint-recheck-20261112/adjudication.json",
                "sha256": "da622a47e30895f782cf2ad6b3f5380021eb88690b500aaca0d0ba49845bee5f"}
DELTA_ROWS = {"new_000023": [33, 34, 35], "new_000031": [22, 23, 24]}
ENDPOINT_DELTA = {"changed_train_files": 2, "replaced_rows": 6, "changed_endpoint_slots": 10,
    "visible_to_missing": 3, "missing_to_visible": 4, "visible_coordinate_corrections": 3,
    "retained_missing_slots": 2, "net_visible_endpoint_change": 1, "net_object_change": 0}
EXECUTION = {"actual_runner_arm_for_both_roles": "candidate", "seed": 46, "epochs": 30,
    "independent_epoch173_initialization": True, "resume": False,
    "checkpoint_selection": "final_epoch", "checkpoint_path": "fit/weights/last.pt",
    "resource_trial": {"role": "endpoint_reviewed", "batches": 4, "initializes_full_runs": False},
    "require_same_clean_source": True, "require_same_recorded_gpu_runtime": True,
    "require_same_resolved_numerical_settings": True, "reject_incompatible_restores": True,
    "automatic_promotion": False}
EVALUATION_SETTINGS = {
    "primary_box_confidence": 0.13, "diagnostic_box_confidences": [], "imgsz": 1280,
    "nms_iou": 0.7, "matching_iou": 0.5, "endpoint_confidence": 0.5, "radii_px": [5, 10],
    "primary_reference": PRIMARY_REFERENCE, "primary_reference_sha256": PRIMARY_REFERENCE_SHA256,
    "all_object_denominator": 663, "both_corner_denominator": 584,
    "historical_reference": "data/manifest.json", "references_reported_separately": True,
    "grouped_recall_by_locatable_corner_count": [0, 1, 2],
    "endpoint_visibility_pairs": [["endpoint_reviewed", "old_reviewed"], ["endpoint_reviewed", "PR35"]],
    "unlocatable_slot_comparison": "Use the same reference-unlocatable slots on objects matched by both members of each pair; report numerator, denominator and identities. Also report unconditional outputs and unmatched-object counts.",
    "locatable_coverage": "Report returned/reference-locatable slots over the full reference and the same common-matched objects; report missing returned slots and coordinate errors.",
    "required_results": ["All-object matches, misses, duplicate and other extras, and review workload",
        "Zero/one/two-locatable-corner group recall with complete denominators",
        "Both-corner 5/10px successes over all 584 eligible objects, including failures to detect",
        "Common-matched unlocatable-slot outputs together with locatable-slot coverage",
        "All gained/lost object and pair-success identities, with original-image assignment review",
        "Miss-free nonempty scenes and every output on the three empty images",
        "The exact current PR35 checkpoint at identical inference/reference settings"],
    "PR35": {"source_revision": "8e8bb99cabcb44ed0ec0acd14db911aa44013fd8",
        "archive_path": "results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip",
        "archive_sha256": "df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91",
        "checkpoint_manifest": "candidate/checkpoint-manifest.json",
        "checkpoint_sha256": "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1"},
    "decision": "At the fixed primary confidence 0.13, require supported endpoint-availability or localization improvement against the matched old_reviewed control while preserving object completeness: no additional supported lost matches and no higher missed-object count. Preserve locatable-slot coverage and existing 5px and 10px pair successes; empty scenes must have no added outputs. Inspect extras and all gained/lost identities against original pixels. Newly found objects are not required for an endpoint benefit, but fewer unlocatable-slot outputs caused by missed objects or blanket endpoint suppression do not qualify. Assess replacement of the exact operational PR35 checkpoint separately at these same settings and preservation criteria. IoU/assignment crossings and uncertain labels do not establish physical recovery. No automatic promotion or threshold/seed/epoch/subset search."}


def check_count_delta(old, endpoint):
    expected = json.loads(json.dumps(old))
    expected["train"]["visible_endpoints"] += 1
    expected["train"]["missing_endpoints"] -= 1
    _require(_same(endpoint, expected), "endpoint counts must differ only by +1 visible/-1 missing")


def validate_protocol(root, protocol_path=DEFAULT_PROTOCOL):
    root = Path(root).resolve()
    requested = Path(protocol_path)
    relative = requested.relative_to(root).as_posix() if requested.is_absolute() else requested.as_posix()
    path = _contained_path(root, relative)
    protocol = _read_json(path)
    _require(type(protocol.get("schema_version")) is int and protocol["schema_version"] == 1
             and protocol.get("kind") == PROTOCOL_KIND
             and protocol.get("experiment_id") == "endpoint-visibility-seed46-v1"
             and protocol.get("status") == "prospectively_declared", "fixed endpoint protocol identity differs")
    _require(protocol.get("recipe_sha256") == RECIPE_SHA256 and "recipe_id" not in protocol,
             "protocol needs one numerical recipe hash and distinct per-role recipe IDs")
    _require(protocol.get("initial_checkpoint_sha256") == INITIAL_SHA256
             and protocol.get("reference_manifest_sha256") == SOURCE_SHA256
             == file_sha256(root / "data/manifest.json"), "initial checkpoint or source reference differs")
    _require(protocol.get("candidate_source_revision") == CANDIDATE_SOURCE,
             "protocol endpoint review source revision differs")
    _require(_same(protocol.get("execution"), EXECUTION), "fixed execution settings differ")
    _require(_same(protocol.get("evaluation"), EVALUATION_SETTINGS)
             and file_sha256(_contained_path(root, PRIMARY_REFERENCE)) == PRIMARY_REFERENCE_SHA256,
             "fixed evaluation settings or reference differ")
    _require(protocol.get("delta_train_image_ids") == sorted(DELTA_ROWS)
             and _same(protocol.get("delta_train_rows_1based"), DELTA_ROWS)
             and _same(protocol.get("new_endpoint_edit_counts"), ENDPOINT_DELTA), "fixed endpoint delta differs")
    _require(_same(protocol.get("annotation_adjudication"), ADJUDICATION)
             and file_sha256(_contained_path(root, ADJUDICATION["path"])) == ADJUDICATION["sha256"],
             "rechecked endpoint adjudication differs")
    roles = protocol.get("roles")
    _require(isinstance(roles, dict) and set(roles) == set(ROLES), "protocol roles differ")
    records = {row["id"]: row for row in _read_json(root / "data/manifest.json")["records"]}
    effective, checks, proposals = {}, {}, {}
    for role, location, digest, candidate_id in (
        ("old_reviewed", OLD_MANIFEST, OLD_SHA256, "annotations-2026-11-10"),
        ("endpoint_reviewed", NEW_MANIFEST, NEW_SHA256, "endpoint-visibility-2026-11-12")):
        identity = recipe_identity(ROLE_RECIPE_IDS[role])
        _require(identity["recipe_sha256"] == RECIPE_SHA256, "role numerical recipe changed")
        expected = {"candidate_manifest": location, "candidate_manifest_sha256": digest,
                    "candidate_id": candidate_id, **identity}
        _require(_same(roles[role], expected), f"fixed candidate or recipe identity differs: {role}")
        manifest = _contained_path(root, location)
        _require(file_sha256(manifest) == digest, f"candidate manifest SHA-256 differs: {role}")
        proposals[role] = proposal = _read_json(manifest)
        verifier = verify_annotation_candidate if role == "old_reviewed" else verify_endpoint_candidate
        checks[role] = verifier(root, manifest)
        changes = {item["image_id"]: item for item in proposal["changes"]}
        effective[role] = {image_id: changes.get(image_id, {}).get("candidate_label_sha256", row["label_sha256"])
                           for image_id, row in records.items()}
    endpoint = proposals["endpoint_reviewed"]
    _require(endpoint["source_revision"] == CANDIDATE_SOURCE, "endpoint review source revision differs")
    _require(_same(checks["endpoint_reviewed"]["new_endpoint_edit_counts"], ENDPOINT_DELTA),
             "verified candidate does not contain the fixed ten-slot intervention")
    _require(_same(_read_json(_contained_path((root / NEW_MANIFEST).parent, endpoint["review"]["path"]))["evidence"],
                   ADJUDICATION), "candidate uses another endpoint adjudication")
    rows = {change["image_id"]: [item["original_row"] for item in change["replacements"]]
            for change in endpoint["changes"] if change["image_id"] in DELTA_ROWS}
    _require(_same(rows, DELTA_ROWS), "verified endpoint replacement rows differ")
    check_count_delta(checks["old_reviewed"]["counts"], checks["endpoint_reviewed"]["counts"])
    _require(checks["old_reviewed"]["counts"]["train"] == dict(images=137, objects=3771,
             visible_endpoints=6841, missing_endpoints=701, empty_images=13), "PR35 counts differ")
    changed = sorted(image_id for image_id in records
                     if effective["old_reviewed"][image_id] != effective["endpoint_reviewed"][image_id])
    _require(changed == sorted(DELTA_ROWS) and all(records[i]["split"] == "train" for i in changed),
             "effective labels differ outside the two endpoint training images")
    return {"path": relative, "sha256": file_sha256(path), "document": protocol,
        "records": records, "effective_label_sha256": effective, "candidate_checks": checks,
        "delta_train_image_ids": changed, "delta_train_rows_1based": DELTA_ROWS,
        "delta_verification": {"new_endpoint_edit_counts": ENDPOINT_DELTA,
            "inherited_files_and_change_objects_exact": 8, "inherited_decision_objects_exact": 12,
            "class_boxes_objects_and_validation_preserved": True, "visual_correctness_verified": False}}


def collect_reviewed_run(preflight, run, role, context):
    _require(role in ROLES, "unknown reviewed-label role")
    bundle = collect_arm(preflight, Path(run), "candidate")
    spec = context["document"]["roles"][role]
    for key in ("candidate_manifest", "candidate_manifest_sha256", "candidate_id"):
        _require(bundle["staging"]["source"].get(key) == spec[key], f"{role} staged {key} differs")
    expected = recipe_identity(ROLE_RECIPE_IDS[role])
    _require(all(bundle["identity"].get(key) == value == spec[key] for key, value in expected.items()),
             f"{role} checkpoint recipe differs")
    _require(bundle["identity"].get("initial_checkpoint_sha256") == INITIAL_SHA256,
             f"{role} initialization differs")
    _require(bundle["receipt"]["runtime"]["device"].startswith("cuda:"),
             f"{role} must be a CUDA training run; no CPU fallback")
    return bundle


def check_reviewed_pair(bundles, context):
    _require(set(bundles) == set(ROLES), "both fixed roles are required")
    old, endpoint = (bundles[role] for role in ROLES)
    for name, first, second in (
        ("training source", old["receipt"]["source"], endpoint["receipt"]["source"]),
        ("recorded GPU/runtime", old["receipt"]["runtime"], endpoint["receipt"]["runtime"]),
        ("numerical training settings", shared_settings(old["receipt"]), shared_settings(endpoint["receipt"]))):
        _require(_same(first, second), f"old/endpoint {name} differ")
    _require(old["receipt"]["source"]["working_tree_status"] == "", "training source must be clean")
    varying = {"candidate_id", "candidate_manifest", "candidate_manifest_sha256", "declared_candidate_source_revision"}
    _require(_same({k: v for k, v in old["staging"]["source"].items() if k not in varying},
                   {k: v for k, v in endpoint["staging"]["source"].items() if k not in varying}),
             "old/endpoint shared staging source differs")
    inventories = {}
    for role, bundle in bundles.items():
        variant = bundle["staging"]["variants"]["candidate"]
        _require(_same(variant["counts"], context["candidate_checks"][role]["counts"]),
                 f"{role} counts differ from verified candidate")
        expected = {}
        for image_id, record in context["records"].items():
            expected[record["image"]] = record["image_sha256"]
            expected[record["label"]] = context["effective_label_sha256"][role][image_id]
        inventory = {item["path"]: item["output_sha256"] for item in variant["files"]}
        _require(len(variant["files"]) == len(expected) and inventory == expected
                 and all(item["source_sha256"] == item["output_sha256"] for item in variant["files"]),
                 f"{role} complete candidate inventory differs")
        inventories[role] = inventory
    check_count_delta(old["staging"]["variants"]["candidate"]["counts"],
                      endpoint["staging"]["variants"]["candidate"]["counts"])
    changed = sorted(path for path in inventories["old_reviewed"]
                     if inventories["old_reviewed"][path] != inventories["endpoint_reviewed"][path])
    _require(changed == sorted(context["records"][i]["label"] for i in context["delta_train_image_ids"]),
             "training pair changed images, validation or unapproved labels")


def bundle_rows(bundles):
    rows = []
    for role in ROLES:
        for item in bundles[role]["files"]:
            _require(item["path"].startswith("candidate/"), "unexpected inner bundle prefix")
            rows.append({**item, "path": role + "/" + item["path"].removeprefix("candidate/")})
    return sorted(rows, key=lambda item: item["path"])


def result_index(context, bundles, rows, created_at):
    old = bundles["old_reviewed"]
    return {"schema_version": 1, "kind": RESULT_KIND, "created_at_utc": created_at,
        "experiment_id": context["document"]["experiment_id"],
        "protocol": {"path": context["path"], "sha256": context["sha256"]},
        "packager_sha256": file_sha256(Path(__file__)), "recipe_sha256": RECIPE_SHA256,
        "initial_checkpoint_sha256": INITIAL_SHA256,
        "training_source": old["receipt"]["source"], "runtime": old["receipt"]["runtime"],
        "shared_training_settings": shared_settings(old["receipt"]),
        "delta_train_image_ids": context["delta_train_image_ids"],
        "delta_train_rows_1based": context["delta_train_rows_1based"],
        "new_endpoint_edit_counts": context["document"]["new_endpoint_edit_counts"],
        "roles": {role: {**context["document"]["roles"][role],
            "checkpoint_manifest": f"{role}/checkpoint-manifest.json",
            "checkpoint_identity": bundles[role]["identity"],
            "training_counts": bundles[role]["staging"]["variants"]["candidate"]["counts"]} for role in ROLES},
        "files": [{key: value for key, value in row.items() if key != "source_path"} for row in rows],
        "verification_scope": "Two complete CUDA candidate bundles with distinct annotation/endpoint recipe IDs and identical numerical settings; same clean source/recorded runtime; fixed effective label inventories and +1 visible/-1 missing endpoint delta. Resource-trial artifacts are not included. No model loading, inference, training execution or performance approval by this packager."}


def package_results(root, old_reviewed, endpoint_reviewed, output, *, protocol=DEFAULT_PROTOCOL, verify_existing=False):
    root = Path(root).resolve()
    runs = {}
    for role, value in zip(ROLES, (old_reviewed, endpoint_reviewed)):
        path = Path(value)
        _require(path.is_dir() and not path.is_symlink(), f"{role} needs a nonsymlink run directory")
        runs[role] = path.resolve()
    _require(not any(runs[a].is_relative_to(runs[b]) for a, b in (ROLES, tuple(reversed(ROLES)))),
             "old/endpoint run directories must not overlap")
    output = Path(output)
    _require(not output.is_symlink(), "output must not be a symlink")
    if output.exists() and not verify_existing:
        raise FileExistsError(f"output already exists; use --verify-existing without overwriting: {output}")
    output = output.resolve()
    _require(output.suffix.lower() == ".zip" and not any(output.is_relative_to(path) for path in (root, *runs.values())),
             "output must be a ZIP outside source and run directories")
    context = validate_protocol(root, protocol)
    preflight = preflight_dataset(root, require_weights=False)
    bundles = {role: collect_reviewed_run(preflight, run, role, context) for role, run in runs.items()}
    current = source_identity(root)
    _require(current["working_tree_status"] == "", "packaging requires the clean pinned source checkout")
    _require(all(_same(bundle["receipt"]["source"], current) for bundle in bundles.values()),
             "training source differs from the packaging checkout")
    check_reviewed_pair(bundles, context)
    rows = bundle_rows(bundles)
    if verify_existing:
        with zipfile.ZipFile(output) as archive:
            index_bytes = archive.read(INDEX_NAME)
        created = json.loads(index_bytes).get("created_at_utc")
        _require(isinstance(created, str) and datetime.fromisoformat(created).utcoffset() is not None,
                 "existing result index lacks an aware timestamp")
        index = result_index(context, bundles, rows, created)
        expected = (json.dumps(index, indent=2, allow_nan=False) + "\n").encode()
        _require(index_bytes == expected, "existing result index differs from this verified pair")
        verify_archive(output, expected, rows)
    else:
        index = result_index(context, bundles, rows, datetime.now(timezone.utc).isoformat())
        index_bytes = (json.dumps(index, indent=2, allow_nan=False) + "\n").encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".endpoint-results-", dir=output.parent) as temporary:
            archive_path = Path(temporary) / "results.zip"
            with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                archive.writestr(INDEX_NAME, index_bytes)
                for row in rows:
                    copy_member(archive, row)
            verify_archive(archive_path, index_bytes, rows)
            os.link(archive_path, output)
    return {"status": "verified_existing" if verify_existing else "packaged", "archive": str(output),
            "sha256": file_sha256(output), "size_bytes": output.stat().st_size,
            "files": len(rows) + 1, "roles": index["roles"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--old-reviewed", type=Path, required=True)
    parser.add_argument("--endpoint-reviewed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    try:
        result = package_results(args.root, args.old_reviewed, args.endpoint_reviewed, args.output,
                                 protocol=args.protocol, verify_existing=args.verify_existing)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"Endpoint result package refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
