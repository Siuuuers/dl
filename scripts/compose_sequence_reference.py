#!/usr/bin/env python3
"""Version reviewed development labels without changing training validation."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
import json
from pathlib import Path

from compose_sequence_front_review import bound, check_corner_change, descriptor, digest, encoded, one_line
from evaluate_pose import parse_pose_labels
from verify_pose_candidate import check_geometry, read_reference, require, totals


KIND = "review_sequence_reference_refinement"


def compose(root, review_path):
    review = json.loads(review_path.read_text())
    config = review["composition"]
    reference_id = config["reference_id"]
    require(reference_id == "sequence-refined-reference-" + config["review_date"],
            "reference identity must match its review date")
    date.fromisoformat(config["review_date"])
    parent_path = bound(root, config["parent"])
    parent = json.loads(parent_path.read_text())
    source_hash, records, by_id, _, _, _ = read_reference(root, parent["source_manifest_sha256"])
    val_records = sorted((r for r in records if r["split"] == "val"), key=lambda r: r["id"])
    parent_files = {item["image_id"]: item for item in parent["files"]}
    require(len(val_records) == len(parent_files) == 20 and
            set(parent_files) == {r["id"] for r in val_records}, "development inventory differs")
    for source in review.get("review_sources", []):
        bound(root, source)
    grouped, seen = defaultdict(list), set()
    for op in review["operations"]:
        iid, action = op["image_id"], op["operation"]
        require(iid in parent_files and action in ("replace", "delete", "add"),
                "reference operation must target a development photo")
        require(isinstance(op.get("reason"), str) and op["reason"].strip(), "operation needs a reason")
        evidence = op.get("evidence_image_ids", [])
        require(evidence and iid in evidence and all(eid in by_id for eid in evidence),
                "operation must identify the target and available evidence images")
        for eid in evidence:
            bound(root, {"path": "data/" + by_id[eid]["image"], "sha256": by_id[eid]["image_sha256"]})
        if action == "add":
            require(op.get("base_row") is None and op.get("before_line") is None
                    and isinstance(op.get("addition_id"), str) and op["addition_id"].strip(),
                    "addition needs a stable identity and no base row")
            key = (iid, "addition", op["addition_id"])
        else:
            require(type(op.get("base_row")) is int and op["base_row"] > 0, "invalid base row")
            key = (iid, "row", op["base_row"])
        require(key not in seen, "repeated operation on one reference object")
        seen.add(key)
        grouped[iid].append(op)
    payloads, files, parsed, base_parsed, operations = {}, [], {}, {}, []
    mappings = {}
    for record in val_records:
        iid, size = record["id"], record["original_size"]
        prior = parent_files[iid]
        image = root / "data" / record["image"]
        require(digest(image.read_bytes()) == record["image_sha256"], "original pixels differ: " + iid)
        label = parent_path.parent / prior["label_path"]
        before = label.read_bytes()
        require(digest(before) == prior["label_sha256"], "parent reference label differs: " + iid)
        lines = before.decode().splitlines(keepends=True)
        require(all(line.strip() for line in lines), "blank physical label rows are unsupported")
        base_parsed[iid] = parse_pose_labels(before.decode(), *size, iid + " baseline")
        if "row_map" in prior:
            row_map = deepcopy(prior["row_map"])
        else:
            row_map = [{"original_row": original, "candidate_row": current}
                       for original, current in enumerate(prior["retained_source_to_revised_row_one_based"], 1)]
            row_map += [{"original_row": None, "candidate_row": row,
                         "addition_id": parent["reference_id"] + f"-{iid}-addition-{row}"}
                        for row in prior.get("added_revised_rows_one_based", [])]
        require(sorted(m["candidate_row"] for m in row_map if m["candidate_row"] is not None)
                == list(range(1, len(lines) + 1)), "parent reference row map is incomplete")
        identities = {m.get("addition_id") for m in row_map if m["original_row"] is None}
        removed, replacements, additions = set(), {}, []
        for op in grouped[iid]:
            row, action = op.get("base_row"), op["operation"]
            old_fields = None
            if action != "add":
                require(row <= len(lines) and lines[row - 1].rstrip("\r\n") == op["before_line"],
                        f"exact parent row differs: {iid}:{row}")
                old_fields = one_line(op["before_line"], iid)
            else:
                require(op["addition_id"] not in identities, "reused addition identity")
                identities.add(op["addition_id"])
            new_fields = None
            if action != "delete":
                new_fields = one_line(op["after_line"], iid)
                objects = parse_pose_labels(op["after_line"], *size, iid + " operation")
                check_geometry(objects, *size, iid + " operation")
                require(len(objects) == 1, "one output row required")
                require(op.get("before_line") != op["after_line"], "ineffective replacement")
            else:
                require(op.get("after_line") is None, "deletion cannot include output")
            check_corner_change(op, old_fields, new_fields)
            if action == "delete":
                removed.add(row)
            elif action == "replace":
                replacements[row] = op["after_line"] + "\n"
            else:
                additions.append(op)
        result = "".join(replacements.get(i, line) for i, line in enumerate(lines, 1) if i not in removed)
        if additions:
            require(not result or result.endswith("\n"), "a terminating newline is required before additions")
            result += "".join(op["after_line"] + "\n" for op in additions)
        payload = result.encode()
        if not grouped[iid]:
            require(payload == before, "an unreviewed reference file changed")
        for entry in row_map:
            old = entry["candidate_row"]
            entry["base_row"] = old
            entry["candidate_row"] = (None if old is None or old in removed else
                                      old - sum(row < old for row in removed))
        for row, op in enumerate(additions, len(lines) - len(removed) + 1):
            row_map.append({"original_row": None, "base_row": None, "candidate_row": row,
                            "addition_id": op["addition_id"]})
        parsed[iid] = parse_pose_labels(result, *size, iid)
        check_geometry(parsed[iid], *size, iid)
        require(sorted(m["candidate_row"] for m in row_map if m["candidate_row"] is not None)
                == list(range(1, len(parsed[iid]) + 1)), "output reference row map is incomplete")
        mapping = {m["base_row"]: m["candidate_row"] for m in row_map if m["base_row"] is not None}
        mappings[iid] = mapping
        addition_rows = {m.get("addition_id"): m["candidate_row"] for m in row_map if m["original_row"] is None}
        for op in grouped[iid]:
            operations.append({**op, "output_row": (addition_rows[op["addition_id"]] if op["operation"] == "add"
                                                     else mapping[op["base_row"]]),
                               "source_image_sha256": record["image_sha256"],
                               "base_label_sha256": digest(before)})
        output = record["label"]
        payloads[output] = payload
        item = deepcopy(prior)
        item.update(label_path=output, label_sha256=digest(payload), row_count=len(parsed[iid]),
                    parent_label=descriptor(root, label), changed_from_parent=payload != before, row_map=row_map)
        item["retained_source_to_revised_row_one_based"] = [m["candidate_row"] for m in row_map
                                                           if m["original_row"] is not None]
        item["added_revised_rows_one_based"] = sorted(m["candidate_row"] for m in row_map
                                                      if m["original_row"] is None and m["candidate_row"] is not None)
        files.append(item)
    baseline_counts, counts = totals(val_records, base_parsed)["val"], totals(val_records, parsed)["val"]
    require(baseline_counts == parent["counts"], "parent reference totals differ")
    require(counts["images"] == 20 and counts["empty_images"] == 3, "reference photos or negatives changed")
    if "expected_counts" in review:
        require(counts == review["expected_counts"], "reviewed reference totals differ")
    holds = {(item["image_id"], item["row"]): deepcopy(item) for item in parent["remaining_current_rows"]}
    decisions_seen = set()
    for decision in review["decisions"]:
        key = (decision["image_id"], decision["base_row"])
        require(key in holds and key not in decisions_seen, "unknown or repeated reference hold decision")
        require(decision["status"] in ("resolved", "pending") and decision.get("reason"),
                "reference decision needs a status and reason")
        decisions_seen.add(key)
        if decision["status"] == "resolved":
            del holds[key]
        else:
            holds[key]["reason"] = decision["reason"]
    remaining = []
    for (iid, old), item in sorted(holds.items()):
        current = mappings[iid][old]
        require(current is not None, "a held reference object was deleted without a resolution")
        item.update(row=current, parent_row=old)
        remaining.append(item)
    corners = Counter(sum(point[2] == 2 for point in obj["keypoints"])
                      for objects in parsed.values() for obj in objects)
    manifest = {"schema_version": 1, "kind": KIND, "reference_id": reference_id,
                "date": config["review_date"], "policy_id": parent["policy_id"],
                "status": "awaiting_pdf_review", "scope": "Separately versioned development reference; no training-validation replacement.",
                "source_manifest_path": "data/manifest.json", "source_manifest_sha256": source_hash,
                "parent_reference": config["parent"], "review": descriptor(root, review_path),
                "review_sources": review.get("review_sources", []), "counts": counts,
                "baseline_counts": baseline_counts, "objects_by_locatable_corners": dict(sorted(corners.items())),
                "delta_from_parent": {key: counts[key] - baseline_counts[key] for key in counts},
                "files": files, "operations": operations,
                "operation_counts": dict(Counter(op["operation"] for op in operations)),
                "decisions": review["decisions"], "remaining_current_rows": remaining,
                "remaining_rows_are_authoritative": True, "full_dataset_certified": False,
                "limits": ["Past metric reports keep their original reference version; no scores are rewritten.",
                           "This reference contains known unresolved rows and awaits the requested PDF review.",
                           "The same-video scenes are development evidence, not an independent final test."]}
    payloads["manifest.json"] = encoded(manifest)
    return payloads, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    review_path = (args.review if args.review.is_absolute() else root / args.review).resolve()
    require(review_path.is_relative_to(root), "review path escapes repository")
    payloads, manifest = compose(root, review_path)
    output = root / "data/references" / manifest["reference_id"]
    existing = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} if output.exists() else set()
    require(existing <= set(payloads) | {"README.md"}, "unexpected files in reference output")
    if not args.verify_only:
        for relative, payload in payloads.items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    for relative, payload in payloads.items():
        require((output / relative).read_bytes() == payload, "saved reference differs: " + relative)
    print(json.dumps({"reference_id": manifest["reference_id"], "manifest_sha256": digest(payloads["manifest.json"]),
                      "counts": manifest["counts"], "operation_counts": manifest["operation_counts"],
                      "remaining_rows": len(manifest["remaining_current_rows"])}, indent=2))


if __name__ == "__main__":
    main()
