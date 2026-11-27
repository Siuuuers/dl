#!/usr/bin/env python3
"""Build the R48 supplementary and development annotation proposals on CPU.

Prediction cues are frozen PR133 candidate outputs, also reproduced exactly by
the PR137 reference. The returned model outputs and old references are immutable.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
import build_front_review_r47 as r47
import build_front_review_r46 as r46
import build_front_review_r44 as r44
import build_front_review_r43 as r43
from evaluate_pose import parse_pose_labels
from verify_pose_candidate import check_geometry

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = "data/reviews/additional-fronts-2026-11-27-r48"
DEVELOPMENT = "data/reviews/fronts-2026-11-27-r48"
R47_SHA = "34fe23ff1744243f2a74fb3466bb7c9106b6ad31636f8180658ffe326e26b2ea"
R38 = "data/reviews/fronts-2026-11-26-r38/manifest.json"
R38_SHA = "0921081930f49340ff34290d5752be2d669f084f4d1518e2019060a5600c43b5"
OLD_INDEX = "results/evaluation/front-errors-r47-20261127/error-index.json"
OLD_INDEX_SHA = "04f30942e38bd00f93437f1b111705afac4e4c06139b6bd3b3f9cd48bfe80cbe"
RETURN_SHA = "fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023"
# Canonical encoded() bytes of the complete PR137 reference prediction record.
REFERENCE_PREDICTIONS_SHA = "dbee466505287122543648bf999d688483a5b6c80573976bd31bb0d1678b7990"
REQUESTED_SUPPLEMENT = {f"frame_0001:source:{n:03d}" for n in range(6, 12)}
REQUESTS = (
    ("D01", "new_000010", "P", 29, 29, ("ymax",)),
    ("D02", "new_000090", "R", 47, 47, ("left_point", "right_point")),
    ("D03", "new_000008", "P", 35, 17, ("ymin",)),
    ("D04", "new_000008", "P", 36, 17, ("ymin",)),
    ("D05", "new_000053", "R", 14, 14, ("ymax",)),
    ("D06", "new_000087", "P", 58, 43, ("ymin",)),
    ("D07", "new_000089", "P", 52, 42, ("xmax",)),
    ("D08", "new_000089", "P", 55, 42, ("xmax",)),
    ("D09", "new_000089", "P", 59, 41, ("ymin", "xmax")),
    ("D10", "new_000089", "P", 64, 49, ("ymin",)),
    ("D11", "new_000059", "P", 20, 18, ("ymin",)),
    ("D12", "new_000059", "P", 19, 18, ("ymin",)),
    ("D13", "new_000087", "P", 62, 43, ("ymin",)),
    ("D14", "new_000089", "P", 58, 49, ("ymin",)),
    ("D15", "new_000090", "P", 50, 40, ("ymin",)),
)
REQUIRED_CUES = {
    ("new_000010", 29): {29, 27}, ("new_000090", 47): {32},
    ("new_000008", 17): {35, 36, 38, 26}, ("new_000053", 14): {2},
    ("new_000087", 43): {58, 62, 53}, ("new_000089", 42): {52, 55},
    ("new_000089", 41): {59}, ("new_000089", 49): {64, 58, 57},
    ("new_000059", 18): {20, 19, 18}, ("new_000090", 40): {50, 47},
}
SOURCES = ("build_front_review_r48.py", "render_front_review_r48.py")
FIXED_FILES = {
    "manifest.json", "parents/r47-manifest.json", "parents/r38-manifest.json",
    "parents/r47-error-index.json", "parents/reference-predictions.json",
    "supplementary/audit-ledger.json", "development/audit-ledger.json",
    "supplementary/outlines.json", "development/outlines.json",
    *("source/" + name for name in SOURCES),
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else Path(value).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def checked(path, expected):
    need(sha(path) == expected, "Hash differs: " + str(path))
    return Path(path)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def uncertainty(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def fields_with_reasons(items, allowed, context):
    need(isinstance(items, list) and all(isinstance(x, dict) for x in items), context + ": expected field records")
    names = [x.get("field") for x in items]
    need(all(isinstance(x, str) and x in allowed for x in names)
         and len(names) == len(set(names)), context + ": invalid or repeated field")
    need(all(nonempty(x.get("reason")) for x in items), context + ": missing reason")
    return names


def put(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == payload, "Existing different file preserved: " + str(path))
    else:
        path.write_bytes(payload)


def preserve_points(before, after, allow_visible_coordinates=False):
    a, b = before.split(), after.split()
    need(len(a) == len(b) == 11 and a[0] == b[0], "Label schema changed")
    for start in (5, 8):
        need(a[start + 2] == b[start + 2], "Visibility changed")
        need(float(a[start + 2]) in (0, 2), "Invalid visibility")
        if float(a[start + 2]) == 0 or not allow_visible_coordinates:
            need(a[start:start + 3] == b[start:start + 3], "Unavailable or protected point changed")


def supplementary(previous, ledger):
    need(ledger["previous_manifest_sha256"] == R47_SHA and type(ledger["round"]) is int and ledger["round"] == 48
         and type(ledger["schema_version"]) is int and ledger["schema_version"] == 1,
         "Supplementary review parent differs")
    need(ledger["all_current_rows_inspected"] is True and ledger["human_review_performed"] is False,
         "Supplementary review scope differs")
    old_rows = r43.all_rows(previous)
    current = {sid: row for sid, row in old_rows.items() if row["after_line"] is not None}
    decisions = ledger["rows"]
    need(len(decisions) == len(current) == 507 and {x["stable_id"] for x in decisions} == set(current),
         "Every current supplementary row must appear exactly once")
    decisions = {row["stable_id"]: row for row in decisions}
    frames, tallies = [], Counter()
    files = {}
    for frame in previous["frames"]:
        identity, rows, lines = frame["image_id"], [], []
        for old in frame["rows"] + frame["additions"]:
            if old["after_line"] is None:
                continue
            item = decisions[old["stable_id"]]
            need(item["before_line"] == old["after_line"] and item["image_id"] == identity
                 and type(item["source_row"]) is type(old["source_row"]) and item["source_row"] == old["source_row"]
                 and item["source_image_sha256"] == frame["source_image_sha256"], "Supplementary identity differs")
            need(item["inspected"] is True and item["full_original_inspected"] is True
                 and item["human_review_performed"] is False and nonempty(item["reason"])
                 and uncertainty(item["nominal_visual_uncertainty_px"]), "Missing visual decision")
            need(item["requested_fields"] == ["xmin", "xmax"] and isinstance(item["updates_px"], dict)
                 and set(item["updates_px"]) <= {"xmin", "xmax"},
                 "Supplementary edit exceeds side-boundary scope")
            after = r46.amend_line(item["before_line"], item["updates_px"])
            need(after == item["after_line"], "Supplementary coordinate replay differs")
            preserve_points(item["before_line"], after)
            changes = r44.changed_fields(item["before_line"], after)
            need(changes == item["changed_fields"], "Supplementary changed fields differ")
            removed = fields_with_reasons(item.get("hold_resolutions", []), set(old["unresolved_fields"]), "Supplementary hold resolution")
            holds = [x for x in old["unresolved_fields"] if x not in removed]
            permitted_holds = set(r44.FIELDS) | set(old["unresolved_fields"]) | {"bbox.ymin", "instance_identity", "front_surface_identity"}
            added = fields_with_reasons(item.get("new_holds", []), permitted_holds, "Supplementary new hold")
            need(not (set(added) & (set(holds) | set(removed))), "Repeated or resolved-and-reopened hold")
            holds.extend(added)
            need(item["remaining_holds"] == holds, "Supplementary remaining holds differ")
            row = {**deepcopy(item), "source_row": old["source_row"], "output_row": len(lines) + 1,
                   "unresolved_fields": holds, "requested": old["stable_id"] in REQUESTED_SUPPLEMENT}
            rows.append(row)
            lines.append(after)
            tallies["reviewed_rows"] += 1
            tallies["amended_rows"] += bool(changes)
            tallies["box_boundaries_changed"] += sum(x in ("xmin", "ymin", "xmax", "ymax") for x in changes)
            tallies["held_rows"] += bool(holds)
            tallies["held_fields"] += len(holds)
            tallies["resolved_fields"] += len(removed)
            tallies["new_held_fields"] += len(item.get("new_holds", []))
        payload = ("\n".join(lines) + "\n").encode()
        check_geometry(parse_pose_labels(payload.decode(), 1280, 720, identity), 1280, 720, identity)
        complete = not any(row["unresolved_fields"] for row in rows)
        files[f"supplementary/proposals/{identity}.txt"] = payload
        if complete:
            files[f"supplementary/complete-candidates/labels/{identity}.txt"] = payload
        frames.append(dict(image_id=identity, source_image_sha256=frame["source_image_sha256"],
                           annotation_complete=complete, eligible_for_candidate_export=complete,
                           training_split="unassigned", rows=rows))
    tallies["retained_rows"] = tallies["reviewed_rows"] - tallies["amended_rows"]
    need(ledger["full_originals_inspected"] == [f["image_id"] for f in previous["frames"]], "Original-image scope differs")
    report = dict(previous_manifest_sha256=R47_SHA, audit_sha256=sha(encoded(ledger)), counts=dict(tallies), frames=frames,
                  annotation_complete_ids=[f["image_id"] for f in frames if f["annotation_complete"]],
                  held_image_ids=[f["image_id"] for f in frames if not f["annotation_complete"]],
                  historical_identities=508, corner_coordinates_and_visibility_unchanged=True)
    return report, files


def development(parent_data, index_data, ledger, reference_predictions, label_reader):
    """Pure replay from byte-pinned parents and a callable reading parent labels."""
    need(sha(parent_data) == R38_SHA and sha(index_data) == OLD_INDEX_SHA, "Development pinned parent differs")
    need(sha(encoded(reference_predictions)) == REFERENCE_PREDICTIONS_SHA, "Frozen reference prediction record differs")
    parent, old_index = json.loads(parent_data), json.loads(index_data)
    need(ledger["parent_manifest_sha256"] == R38_SHA and ledger["prediction_index_sha256"] == OLD_INDEX_SHA,
         "Development parent or cue index differs")
    need(ledger["human_review_performed"] is False and type(ledger["round"]) is int and ledger["round"] == 48
         and ledger["schema"] == "baomai-r48-development-adjudications-v1", "Development scope differs")
    need(isinstance(ledger["requests"], list) and isinstance(ledger["rows"], list), "Development ledger lists required")
    request_tuples = []
    for request in ledger["requests"]:
        need(type(request["number"]) is int and type(request["reference_row"]) is int
             and all(nonempty(request[k]) for k in ("request_id", "image_id", "kind", "physical_hint"))
             and isinstance(request["requested_fields"], list)
             and all(isinstance(field, str) for field in request["requested_fields"]), "Invalid request types")
        request_tuples.append((request["request_id"], request["image_id"], request["kind"], request["number"],
                               request["reference_row"], tuple(request["requested_fields"])))
    need(len(request_tuples) == 15 and set(request_tuples) == set(REQUESTS), "Exact fifteen development requests differ")
    need(all(type(r["reference_row"]) is int for r in ledger["rows"]), "Development row numbers must be integers")
    operations = {(r["image_id"], r["reference_row"]): r for r in ledger["rows"]}
    requested_targets = {(r[1], r[4]) for r in REQUESTS}
    need(len(operations) == len(ledger["rows"]) == 10 and set(operations) == requested_targets,
         "Expected exactly the ten requested development targets")
    source_list = [r for r in parent["files"] if r["split"] == "val"]
    source_rows = {r["image_id"]: r for r in source_list}
    need(len(source_rows) == len(source_list) == 20 and sum(r["objects"] for r in source_list) == 654,
         "Pinned development population differs")
    need(set(reference_predictions) == set(source_rows) and all(im in source_rows for im, _ in operations),
         "Development target or prediction photograph is outside the pinned validation set")
    reviewed_images = {im for im, _ in operations}
    need(isinstance(ledger["full_originals_inspected"], list)
         and len(ledger["full_originals_inspected"]) == len(reviewed_images)
         and set(ledger["full_originals_inspected"]) == reviewed_images, "Development original-image scope differs")
    index_predictions = {(x["image"], x["prediction_number"]): x for x in old_index["unmatched_predictions"]}
    for request in ledger["requests"]:
        key = (request["image_id"], request["reference_row"])
        if request["kind"] == "P":
            old = index_predictions.get((key[0], request["number"]))
            need(old is not None and old["shown_row"] == key[1], "Legacy prediction-to-reference mapping differs")
            need(old["prediction"] == reference_predictions[key[0]]["postclip"][request["number"] - 1],
                 "Old request prediction differs from the pinned reference cache")
    files, entries, changed, point_changes, held_rows, held_fields = {}, [], 0, 0, 0, 0
    applied = set()
    for identity, source in sorted(source_rows.items()):
        old_data = label_reader(source)
        need(isinstance(old_data, bytes) and sha(old_data) == source["label_sha256"], "Pinned R38 label differs: " + identity)
        lines = old_data.decode().splitlines()
        for (im, number), op in operations.items():
            if im != identity:
                continue
            applied.add((im, number))
            need(1 <= number <= len(lines) and lines[number - 1] == op["before_line"], "R38 row identity differs")
            need(op["stable_id"] == f"{identity}:R38:{number:03d}" and op["source_image_sha256"] == source["source_image_sha256"]
                 and op["original_pixels_inspected"] is True and op["human_review_performed"] is False
                 and nonempty(op["reason"]) and uncertainty(op["nominal_visual_uncertainty_px"]), "Original-pixel review evidence missing")
            relevant = [r for r in REQUESTS if (r[1], r[4]) == (im, number)]
            need(op["request_ids"] == sorted(r[0] for r in relevant), "Row request coverage differs")
            requested_fields = {field for r in relevant for field in r[5]}
            need(isinstance(op["updates_px"], dict) and set(op["updates_px"]) <= requested_fields,
                 "Development edit exceeds requested fields")
            after = r46.amend_line(op["before_line"], op["updates_px"])
            need(after == op["after_line"], "Development coordinate replay differs")
            preserve_points(op["before_line"], after, allow_visible_coordinates=True)
            actual = r44.changed_fields(op["before_line"], after)
            need(actual == op["changed_fields"], "Development changed fields differ")
            holds = fields_with_reasons(op["held_fields"], requested_fields, "Development hold")
            need(not (set(holds) & set(op["updates_px"])), "Held development field was amended")
            expected_status = ("partial_amendment_with_hold" if actual else "held") if holds else (
                "assistant_proposed_amendment" if actual else "retained")
            need(op["decision_status"] == expected_status, "Development decision status differs")
            changed += bool(actual)
            point_changes += sum(x.endswith("point") for x in actual)
            held_rows += bool(holds)
            held_fields += len(holds)
            need(isinstance(op["prediction_cues"], list), "Development prediction cue list required")
            numbers = [cue["prediction_number"] for cue in op["prediction_cues"]]
            need(all(type(n) is int and 1 <= n <= len(reference_predictions[identity]["postclip"]) for n in numbers),
                 "Prediction cue index is outside its one-based list")
            need(len(numbers) == len(set(numbers)) and set(numbers) == REQUIRED_CUES[(im, number)],
                 "Required requested, context or assigned prediction cue missing")
            named = {r[3] for r in relevant if r[2] == "P"}
            for cue in op["prediction_cues"]:
                number_p = cue["prediction_number"]
                need(cue["role"] == ("requested" if number_p in named else "context_or_assigned"), "Prediction cue role differs")
                need(reference_predictions[identity]["postclip"][number_p - 1] == cue["prediction"],
                     "Frozen prediction cue differs")
            lines[number - 1] = after
        payload = ("\n".join(lines) + "\n").encode() if lines else b""
        objects = parse_pose_labels(payload.decode(), 1280, 720, identity)
        check_geometry(objects, 1280, 720, identity)
        need(len(objects) == source["objects"], "Development instance population changed")
        files[f"development/labels/{identity}.txt"] = payload
        files[f"parents/development-labels/{identity}.txt"] = old_data
        entries.append(dict(image_id=identity, source_image_sha256=source["source_image_sha256"],
                            label_path=f"labels/{identity}.txt", parent_label_sha256=source["label_sha256"],
                            label_sha256=sha(payload), objects=len(objects), split="val"))
    need(applied == set(operations), "A development operation was not applied")
    need(len(entries) == 20 and sum(e["objects"] for e in entries) == 654, "Development inventory differs")
    report = dict(parent_manifest_sha256=R38_SHA, audit_sha256=sha(encoded(ledger)),
                  prediction_index_sha256=OLD_INDEX_SHA, reference_predictions_sha256=REFERENCE_PREDICTIONS_SHA,
                  status="assistant_proposed_development_reference",
                  development_only=True, full_dataset_certified=False, changed_training_rows=0,
                  prediction_cues="PR133 candidate; byte-identical geometry in PR137 reference",
                  requests=ledger["requests"], rows=ledger["rows"], files=entries,
                  counts=dict(requests=15, distinct_reviewed_targets=10, amended_rows=changed,
                              corner_positions_changed=point_changes, objects=654, images=20,
                              held_rows=held_rows, held_fields=held_fields,
                              added_rows=0, deleted_rows=0, visibility_changes=0))
    return report, files


def validate_outlines(supplementary_outlines, development_outlines, supp):
    outlines = supplementary_outlines
    need(type(outlines.get("schema_version")) is int and outlines["schema_version"] == 1
         and type(outlines.get("round")) is int and outlines["round"] == 48
         and outlines.get("training_mask") is False and isinstance(outlines.get("items"), list), "Supplementary outline scope differs")
    rows = {row["stable_id"]: row for frame in supp["frames"] for row in frame["rows"]}
    seen = set()
    for item in outlines["items"]:
        identity = item["stable_id"]
        need(identity in rows and identity not in seen, "Invalid or repeated supplementary outline identity")
        seen.add(identity)
        need(type(item["closed"]) is bool and isinstance(item["vertices_px"], list)
             and len(item["vertices_px"]) >= (3 if item["closed"] else 2), "Outline closure or vertices differ")
        need(all(isinstance(point, list) and len(point) == 2
                 and all(type(v) in (int, float) and math.isfinite(v) for v in point)
                 and 0 <= point[0] < 1280 and 0 <= point[1] < 720 for point in item["vertices_px"]), "Outline coordinates outside original image")
        need(nonempty(item["reason"]) and uncertainty(item["nominal_uncertainty_px"]), "Outline evidence missing")
    outlines = development_outlines
    need(outlines.get("schema") == "baomai-r48-diagnostic-outlines-v1"
         and type(outlines.get("round")) is int and outlines["round"] == 48
         and outlines.get("training_mask") is False and outlines.get("outlines") == [], "Development outline scope differs")
    need(nonempty(outlines.get("interpretation")), "Development outline interpretation missing")


def review_manifest(supp, dev, outline_bytes, source_bytes):
    return dict(schema="baomai-r48-visible-front-amendments-v1", round=48, date="2026-11-27",
                previous_source_revision="d0136ae8b2283d38da953e8f109e542029682678",
                result_archive_sha256=RETURN_SHA, human_review_performed=False,
                new_training=False, predictions_modified=False, frozen_references_modified=False,
                coordinate_semantics="Original1280x720: x right, y down; stored ymin top/ymax bottom. User ymax/top and ymin/bottom are translated explicitly.",
                supplementary=supp, development=dev,
                outlines_sha256={name: sha(data) for name, data in outline_bytes.items()},
                source_sha256={name: sha(data) for name, data in source_bytes.items()})


def required_files(supp, dev, labels):
    originals = {"originals/" + frame["image_id"] + ".jpg" for frame in supp["frames"]}
    originals.update("originals/" + row["image_id"] + ".png" for row in dev["rows"])
    return FIXED_FILES | set(labels) | originals


def construct(root, previous_folder, archive, originals):
    r47.verify_output(previous_folder)
    previous = read(checked(previous_folder / "manifest.json", R47_SHA))
    with zipfile.ZipFile(checked(archive, RETURN_SHA)) as z:
        reference_predictions = json.loads(z.read("evaluation/predictions.json"))["reference"]
    reference_data = encoded(reference_predictions)
    need(sha(reference_data) == REFERENCE_PREDICTIONS_SHA, "Returned reference predictions differ")
    parent_data = checked(root / R38, R38_SHA).read_bytes()
    index_data = checked(root / OLD_INDEX, OLD_INDEX_SHA).read_bytes()
    sl = read(root / SUPPLEMENT / "audit-ledger.json")
    dl = read(root / DEVELOPMENT / "audit-ledger.json")
    supp, files = supplementary(previous, sl)
    dev, extra = development(parent_data, index_data, dl, reference_predictions,
                             lambda entry: ((root / R38).parent / entry["label_path"]).read_bytes())
    files.update(extra)
    labels = dict(files)
    files["parents/r47-manifest.json"] = (previous_folder / "manifest.json").read_bytes()
    files["parents/r38-manifest.json"] = parent_data
    files["parents/r47-error-index.json"] = index_data
    files["parents/reference-predictions.json"] = reference_data
    files["supplementary/audit-ledger.json"] = encoded(sl)
    files["development/audit-ledger.json"] = encoded(dl)
    outline_bytes = {}
    for name, source in (("supplementary", SUPPLEMENT), ("development", DEVELOPMENT)):
        outline_bytes[name] = (root / source / "outlines.json").read_bytes()
        files[f"{name}/outlines.json"] = outline_bytes[name]
    validate_outlines(json.loads(outline_bytes["supplementary"]), json.loads(outline_bytes["development"]), supp)
    for frame in supp["frames"]:
        name = frame["image_id"] + ".jpg"
        files["originals/" + name] = checked(previous_folder / "originals" / name, frame["source_image_sha256"]).read_bytes()
    reviewed = {r["image_id"] for r in dev["rows"]}
    for entry in dev["files"]:
        if entry["image_id"] in reviewed:
            name = entry["image_id"] + ".png"
            files["originals/" + name] = checked(originals / name, entry["source_image_sha256"]).read_bytes()
    source_bytes = {name: (root / "scripts" / name).read_bytes() for name in SOURCES}
    for name, data in source_bytes.items():
        need(data == Path(__file__).with_name(name).read_bytes(), "Executing and bundled source differ: " + name)
        files["source/" + name] = data
    manifest = review_manifest(supp, dev, outline_bytes, source_bytes)
    files["manifest.json"] = encoded(manifest)
    need(set(files) == required_files(supp, dev, labels), "Constructed review membership differs")
    return manifest, files


def source_exports(manifest, files):
    outputs = {}
    for name, data in files.items():
        if name.startswith("supplementary/proposals/") or name.startswith("supplementary/complete-candidates/"):
            outputs[SUPPLEMENT + "/" + name.split("/", 1)[1]] = data
        elif name.startswith("development/labels/"):
            outputs[DEVELOPMENT + "/" + name.split("/", 1)[1]] = data
    outputs[DEVELOPMENT + "/manifest.json"] = encoded({k: v for k, v in manifest["development"].items() if k not in ("rows", "requests")})
    release = dict(round=48, generated_manifest_sha256=sha(files["manifest.json"]),
                   parent_manifest_sha256=R47_SHA,
                   annotation_complete_ids=manifest["supplementary"]["annotation_complete_ids"],
                   held_image_ids=manifest["supplementary"]["held_image_ids"],
                   counts=manifest["supplementary"]["counts"],
                   audit_sha256=sha(files["supplementary/audit-ledger.json"]),
                   outlines_sha256=sha(files["supplementary/outlines.json"]),
                   files={p.removeprefix(SUPPLEMENT + "/"): sha(b) for p, b in outputs.items() if p.startswith(SUPPLEMENT + "/")})
    outputs[SUPPLEMENT + "/release-status.json"] = encoded(release)
    return outputs


def file_members(folder):
    paths = list(Path(folder).rglob("*"))
    need(not any(p.is_symlink() for p in paths), "Review directories must not contain symbolic links")
    return {p.relative_to(folder).as_posix() for p in paths if p.is_file()}


def verify_output(folder):
    folder = Path(folder)
    previous = read(checked(folder / "parents/r47-manifest.json", R47_SHA))
    parent_data = checked(folder / "parents/r38-manifest.json", R38_SHA).read_bytes()
    index_data = checked(folder / "parents/r47-error-index.json", OLD_INDEX_SHA).read_bytes()
    reference_predictions = read(checked(folder / "parents/reference-predictions.json", REFERENCE_PREDICTIONS_SHA))
    sl, dl = (read(folder / name / "audit-ledger.json") for name in ("supplementary", "development"))
    supp, labels = supplementary(previous, sl)
    dev, extra = development(parent_data, index_data, dl, reference_predictions,
                             lambda entry: (folder / "parents/development-labels" / (entry["image_id"] + ".txt")).read_bytes())
    labels.update(extra)
    expected = required_files(supp, dev, labels)
    found = file_members(folder)
    need(found == expected | {"inventory.json"}, "Unexpected or missing review file: " + ", ".join(sorted(found ^ (expected | {"inventory.json"}))))
    inventory = read(folder / "inventory.json")
    need(isinstance(inventory, dict) and set(inventory) == expected, "Inventory does not match the independently replayed membership")
    for name in expected:
        data = (folder / name).read_bytes()
        item = inventory[name]
        need(isinstance(item, dict) and type(item.get("bytes")) is int
             and item == dict(bytes=len(data), sha256=sha(data)), "Review file changed: " + name)
    for name, data in labels.items():
        need((folder / name).read_bytes() == data, "Annotation label replay differs: " + name)
    for name, ledger in (("supplementary", sl), ("development", dl)):
        need((folder / name / "audit-ledger.json").read_bytes() == encoded(ledger), "Noncanonical bundled ledger")
    outline_bytes = {name: (folder / name / "outlines.json").read_bytes() for name in ("supplementary", "development")}
    validate_outlines(json.loads(outline_bytes["supplementary"]), json.loads(outline_bytes["development"]), supp)
    for frame in supp["frames"]:
        checked(folder / "originals" / (frame["image_id"] + ".jpg"), frame["source_image_sha256"])
    for row in dev["rows"]:
        checked(folder / "originals" / (row["image_id"] + ".png"), row["source_image_sha256"])
    source_bytes = {name: (folder / "source" / name).read_bytes() for name in SOURCES}
    for name, data in source_bytes.items():
        need(data == Path(__file__).with_name(name).read_bytes(), "Bundled source differs: " + name)
    manifest = review_manifest(supp, dev, outline_bytes, source_bytes)
    need((folder / "manifest.json").read_bytes() == encoded(manifest), "Complete review manifest replay differs")
    return dict(passed=True, scope="Exact R47/R38 parents, complete annotation-ledger and label replay, whole-photo membership, immutable request/cue identities, original image hashes, outlines, bundled source and exact file inventory; assistant visual proposals are not independent human sign-off.",
                supplementary=supp["counts"], development=dev["counts"],
                verified_inventory_entries=len(expected), reference_predictions_sha256=REFERENCE_PREDICTIONS_SHA,
                new_training=False, predictions_modified=False, frozen_references_modified=False)


def preflight_output(folder, files):
    """Reject stale or incompatible content before any output is written."""
    found = file_members(folder)
    need(found <= set(files) | {"inventory.json"}, "Existing unexpected review files preserved: " + ", ".join(sorted(found - set(files) - {"inventory.json"})))
    inventory = encoded({name: dict(bytes=len(data), sha256=sha(data)) for name, data in sorted(files.items())})
    for name in found:
        expected = inventory if name == "inventory.json" else files[name]
        need((Path(folder) / name).read_bytes() == expected, "Existing different review file preserved: " + name)
    return inventory


def preflight_exports(root, exports):
    for relative in (SUPPLEMENT + "/proposals", SUPPLEMENT + "/complete-candidates/labels", DEVELOPMENT + "/labels"):
        present = {relative + "/" + name for name in file_members(root / relative)}
        need(present <= set(exports), "Existing unexpected annotation export preserved")
    for name, data in exports.items():
        if (root / name).exists():
            need((root / name).read_bytes() == data, "Existing different annotation export preserved: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--r47", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--originals", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--write-source", action="store_true")
    parser.add_argument("--check-output", type=Path)
    args = parser.parse_args()
    if args.check_output:
        print(json.dumps(verify_output(args.check_output), indent=2))
        return
    need(all((args.r47, args.archive, args.originals, args.output)), "Supply --r47 --archive --originals --output")
    if args.pdf:
        need(not args.pdf.resolve().is_relative_to(args.output.resolve()), "The PDF must be outside the review output directory")
    manifest, files = construct(args.root, args.r47, args.archive, args.originals)
    exports = source_exports(manifest, files)
    inventory_data = preflight_output(args.output, files)
    preflight_exports(args.root, exports)
    for name, data in exports.items():
        if args.write_source:
            put(args.root / name, data)
        else:
            need((args.root / name).read_bytes() == data, "Committed annotation export differs: " + name)
    for name, data in files.items():
        put(args.output / name, data)
    if args.pdf:
        from render_front_review_r48 import render_pdf
        render_pdf(args.output, args.pdf)
    put(args.output / "inventory.json", inventory_data)
    print(json.dumps(verify_output(args.output), indent=2))


if __name__ == "__main__":
    main()
