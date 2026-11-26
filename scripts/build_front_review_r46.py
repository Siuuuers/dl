#!/usr/bin/env python3
"""Replay R46 physical-front decisions, render review evidence, and export labels.

CPU only. A passing check establishes consistent files, not human approval or
physical accuracy. Held photographs never enter the complete-candidate export.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import urllib.request
import zipfile

sys.dont_write_bytecode = True

import build_front_review_r43 as r43
import build_front_review_r44 as r44
import review_front_mining_r42 as legacy

ROOT = Path(__file__).resolve().parents[1]
REVIEW = "data/reviews/additional-fronts-2026-11-27-r46"
R44_REVISION = "61c04d6a6dcb7e9f01ca99c603148b7f1506766b"
R44_SHA = "100bfaceb9281bcf8c37c48eaaacf5159df14b0d04dd0ec37d5dcd4fc89c672d"
R44_BYTES = 6599840
R44_MANIFEST_SHA = "6a52279ac5c53452cf7e6a3dbd6ea8f76d4aeae287bd2d2a23ae5b61a4c231bf"
R44_URL = ("https://raw.githubusercontent.com/Siuuuers/dl/" + R44_REVISION
           + "/results/review/front-mining-r44/review.zip")
REQUESTS = {
    "frame_0001:source:001": ["right_point"],
    "frame_0001:source:008": ["left_point"],
    "frame_0001:source:015": ["right_point"],
    "frame_0001:source:023": ["left_point"],
    "frame_0001:source:030": ["left_point"],
    "frame_0160:source:005": ["left_point"],
    "frame_0160:source:039": ["xmin", "left_point"],
    "frame_0038:source:008": ["left_point"],
    "frame_0038:source:009": ["left_point"],
    "frame_0074:source:027": ["left_point"],
    "frame_0074:source:043": ["left_point", "ymax", "xmax"],
    "frame_0128:source:029": ["xmax"],
    "frame_0128:source:039": ["left_point", "xmin"],
    "frame_0128:source:041": ["previous_annotation"],
    "frame_0146:source:025": ["xmax"],
    "frame_0178:source:006": ["xmin", "xmax"],
    "frame_0225:source:003": ["ymin", "left_point"],
}
SCRIPTS = ["build_front_review_r46.py", "build_front_review_r44.py",
           "build_front_review_r43.py", "review_front_mining_r42.py",
           "render_front_review_r44.py", "render_front_review_r43.py",
           "render_front_review_r46.py", "preview_pose_labels.py"]
need, digest = legacy.require, legacy.digest
encoded = r43.json_bytes


def read(path):
    return legacy.load_json(Path(path).read_bytes())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def amend_line(before, updates):
    """Change only named coordinate fields; retain exact other tokens."""
    need(isinstance(updates, dict) and set(updates) <= set(r44.FIELDS), "unknown edit field")
    old = r44.geometry(before)
    box = deepcopy(old)
    for name, value in updates.items():
        vals = value if isinstance(value, list) else [value]
        need(all(type(x) in (int, float) and math.isfinite(x) for x in vals), "nonfinite edit")
        if name.endswith("point"):
            need(isinstance(value, list) and len(value) == 2, "point needs XY")
        else:
            need(type(value) in (int, float), "bound needs number")
        box[name] = value
    tokens = before.split()
    if any(name in updates for name in ("xmin", "xmax")):
        tokens[1] = f"{(box['xmin'] + box['xmax']) / 2560:.8f}"
        tokens[3] = f"{(box['xmax'] - box['xmin']) / 1280:.8f}"
    if any(name in updates for name in ("ymin", "ymax")):
        tokens[2] = f"{(box['ymin'] + box['ymax']) / 1440:.8f}"
        tokens[4] = f"{(box['ymax'] - box['ymin']) / 720:.8f}"
    for name, index in (("left_point", 5), ("right_point", 8)):
        if name in updates:
            need(tokens[index + 2] == "2", "an unlocatable slot cannot acquire a coordinate")
            tokens[index] = f"{box[name][0] / 1280:.8f}"
            tokens[index + 1] = f"{box[name][1] / 720:.8f}"
    after = " ".join(tokens)
    legacy.pose(after, "R46 edited row")
    need(set(r44.changed_fields(before, after)) <= set(updates), "unrequested coordinate changed")
    return after


def load_r44(path):
    raw = Path(path).read_bytes()
    need(len(raw) == R44_BYTES and digest(raw) == R44_SHA, "R44 archive identity")
    payload = legacy.archive_payload(io.BytesIO(raw))
    r44.verify_capsule(payload)
    need(digest(payload["manifest.json"]) == R44_MANIFEST_SHA, "R44 manifest identity")
    old = legacy.load_json(payload["manifest.json"])
    _, _, previous, _ = r44.prior_archive(payload[r44.HISTORY])
    return old, payload, previous


def verify_outlines(outlines, manifest):
    need(outlines.get("schema_version") == 1 and outlines.get("training_mask") is False, "outline scope")
    rows, seen = r43.all_rows(manifest), set()
    for item in outlines["items"]:
        sid = item["stable_id"]
        need(sid in rows and rows[sid]["after_line"] is not None and sid not in seen, "outline identity")
        seen.add(sid)
        need(type(item["closed"]) is bool, "outline closure must be explicit")
        need(len(item["vertices_px"]) >= (3 if item["closed"] else 2), "outline vertices")
        for point in item["vertices_px"]:
            need(isinstance(point, list) and len(point) == 2
                 and all(type(v) in (int, float) and math.isfinite(v) for v in point)
                 and 0 <= point[0] < 1280 and 0 <= point[1] < 720, "outline pixel coordinates")
        need(item["reason"].strip() and type(item["nominal_uncertainty_px"]) in (int, float)
             and math.isfinite(item["nominal_uncertainty_px"]) and item["nominal_uncertainty_px"] > 0, "outline evidence")


def apply_audit(old, audit, outlines):
    need(digest(encoded(old)) == R44_MANIFEST_SHA, "audit predecessor")
    need(audit["schema_version"] == 1 and audit["round"] == 46, "audit schema")
    need(audit["previous_manifest_sha256"] == R44_MANIFEST_SHA, "audit source identity")
    need(audit["human_review_performed"] is False, "no human approval is implied")
    current = {k: v for k, v in r43.all_rows(old).items() if v["after_line"] is not None}
    decisions = audit["rows"]
    need(len(decisions) == 507 and len({r["stable_id"] for r in decisions}) == 507
         and {r["stable_id"] for r in decisions} == set(current), "all current rows exactly once")
    need(audit["full_originals_inspected"] == [f["image_id"] for f in old["frames"]], "full-photo review scope")
    out = deepcopy(old)
    rows = r43.all_rows(out)
    frames = {f["image_id"]: f for f in old["frames"]}
    count = dict(reviewed_rows=507, amended_rows=0, retained_rows=0,
                 corner_coordinate_slots_changed=0, box_boundaries_changed=0,
                 added_rows=0, deleted_rows=0, visibility_changes=0,
                 requested_rows=len(REQUESTS), requested_rows_amended=0,
                 newly_held_fields=0, resolved_fields=0)
    for entry in decisions:
        sid = entry["stable_id"]
        prior, row = current[sid], rows[sid]
        fid = sid.split(":")[0]
        need(entry["image_id"] == fid and entry["source_row"] == prior["source_row"]
             and entry["source_image_sha256"] == frames[fid]["source_image_sha256"], sid + ": source identity")
        need(entry["before_line"] == prior["after_line"], sid + ": predecessor line")
        need(entry["inspected"] is True and entry["reason"].strip(), sid + ": visual record")
        need(entry["requested_fields"] == REQUESTS.get(sid, []), sid + ": request mapping")
        after = amend_line(entry["before_line"], entry["updates_px"])
        need(after == entry["after_line"], sid + ": edited line replay")
        actual = r44.changed_fields(entry["before_line"], after)
        need(actual == entry["changed_fields"], sid + ": changed fields")
        resolutions = entry.get("hold_resolutions", [])
        removed = {x["field"] for x in resolutions}
        need(len(removed) == len(resolutions) and removed <= set(prior["unresolved_fields"]), sid + ": hold resolution scope")
        need(all(x["reason"].strip() for x in resolutions), sid + ": resolution evidence")
        added = entry.get("new_holds", [])
        need(all(x["field"] and x["reason"].strip() for x in added), sid + ": new hold evidence")
        fields = [x for x in prior["unresolved_fields"] if x not in removed]
        reasons = list(prior["unresolved_reasons"]) if fields else []
        for hold in added:
            if hold["field"] not in fields:
                fields.append(hold["field"])
                reasons.append(hold["reason"])
                count["newly_held_fields"] += 1
        if prior.get("latest_update"):
            row.setdefault("review_history", []).append({"round": 44, **deepcopy(prior["latest_update"])})
        row["r44_line"] = entry["before_line"]
        row["after_line"] = after
        row["action"] = "add" if row["source_row"] is None else ("retain" if after == row["before_line"] else "replace")
        row["changed_token_indices"] = r43.changed(row["before_line"], after)
        row["reason"] = prior["reason"] + "\nR46: " + entry["reason"]
        row["unresolved_fields"], row["unresolved_reasons"] = fields, reasons
        row["nominal_visual_uncertainty_px"] = entry["nominal_visual_uncertainty_px"]
        row["latest_update"] = dict(status="amended" if actual else ("held" if fields else "retained"),
            reason=entry["reason"], requested_fields=entry["requested_fields"],
            changed_fields=actual, changed_token_indices=r43.changed(entry["before_line"], after),
            all_current_fields_inspected=True, hold_resolutions=resolutions,
            newly_unresolved_fields=[x["field"] for x in added])
        count["amended_rows"] += bool(actual)
        count["retained_rows"] += not actual
        count["corner_coordinate_slots_changed"] += sum(k in actual for k in ("left_point", "right_point"))
        count["box_boundaries_changed"] += sum(k in actual for k in ("xmin", "ymin", "xmax", "ymax"))
        count["requested_rows_amended"] += sid in REQUESTS and bool(actual)
        count["resolved_fields"] += len(removed)
    out.update(kind="r46-physical-corner-and-front-ownership-review", round=46,
        created_date="2026-11-27", previous_manifest_sha256=R44_MANIFEST_SHA,
        previous_round={"round": 44, "manifest_sha256": R44_MANIFEST_SHA,
                        "latest_counts": old["latest_counts"]}, latest_counts=count,
        audit_sha256=digest(encoded(audit)), outlines_sha256=digest(encoded(outlines)),
        latest_review_scope="Original photographs and every current row inspected; recorded local refinements and explicit field holds. Assistant visual proposals, not human sign-off.")
    # Older notes describe earlier rounds; they must not masquerade as the current gate.
    out["policy_notes"] = ["Historical R42/R43: " + s if s.startswith("The three complete images") else s
                           for s in out["policy_notes"]]
    out["policy_notes"].append("R46: a bright lower-front turn with a darker underside and lateral gray/shadow is supporting evidence. Trace material ownership first; never move points by a global brightness rule. In image coordinates, up decreases y. Review outlines are not segmentation masks.")
    complete, held = [], []
    for frame in out["frames"]:
        frame["unresolved_stable_ids"] = [r["stable_id"] for r in frame["rows"] + frame["additions"] if r["unresolved_fields"]]
        frame["annotation_complete"] = not frame["unresolved_stable_ids"]
        frame["eligible_for_candidate_export"] = frame["annotation_complete"]
        frame["disposition"] = "assistant_review_complete" if frame["annotation_complete"] else "held"
        frame["all_current_rows_inspected_round"] = 46
        frame["full_image_inspected"] = frame["all_baseline_rows_inspected"] = True
        frame["latest_reviewed_stable_ids"] = [r["stable_id"] for r in frame["rows"] + frame["additions"] if r["after_line"] is not None]
        (complete if frame["annotation_complete"] else held).append(frame["image_id"])
    out["annotation_complete_ids"], out["held_image_ids"] = complete, held
    out["policy_notes"].append("Current R46 complete-image candidates: " + (", ".join(complete) or "none") + ". Eligibility is an annotation gate, not a new training result or independent test assignment.")
    r43.recount(out)
    verify_outlines(outlines, out)
    need(set(rows) == set(r43.all_rows(old)), "stable identities changed")
    need(all(rows[k] == v for k, v in r43.all_rows(old).items() if v["after_line"] is None), "deleted history changed")
    return out


def verify_output(folder):
    folder = Path(folder)
    for name in SCRIPTS:
        need((folder / "scripts" / name).read_bytes() == Path(__file__).with_name(name).read_bytes(),
             "bundled source differs: " + name)
    old = read(folder / "history/r44-manifest.json")
    need(digest((folder / "history/r43-manifest.json").read_bytes()) == r44.PREVIOUS_MANIFEST_SHA256,
         "R43 comparison history identity")
    audit, outlines = read(folder / "audit-ledger.json"), read(folder / "outlines.json")
    manifest = apply_audit(old, audit, outlines)
    need((folder / "manifest.json").read_bytes() == encoded(manifest), "manifest replay mismatch")
    source = {}
    ids = {f["image_id"] for f in manifest["frames"]}
    for prefix, suffix in (("originals", ".jpg"), ("source-labels", ".txt"),
                           ("baseline-labels", ".txt"), ("proposals", ".txt")):
        need({p.name for p in (folder / prefix).iterdir()} == {i + suffix for i in ids},
             "unexpected files in " + prefix)
    for frame in manifest["frames"]:
        fid = frame["image_id"]
        for dst, src in (("original.jpg", "originals/" + fid + ".jpg"),
                         ("source.txt", "source-labels/" + fid + ".txt"),
                         (fid + ".txt", "baseline-labels/" + fid + ".txt")):
            source["review/" + fid + "/" + dst] = (folder / src).read_bytes()
        need((folder / "proposals" / (fid + ".txt")).read_bytes() == legacy.candidate_bytes(frame), fid + ": proposal differs")
    expected = {f["image_id"] + ".txt" for f in manifest["frames"] if f["eligible_for_candidate_export"]}
    need({p.name for p in (folder / "complete-candidates/labels").iterdir()} == expected, "complete-image export set")
    for name in expected:
        need((folder / "complete-candidates/labels" / name).read_bytes() == (folder / "proposals" / name).read_bytes(), "candidate bytes")
    report = legacy.verify(manifest, source)
    report.update(round=46, latest_counts=manifest["latest_counts"], previous_archive_sha256=R44_SHA,
                  all_508_stable_identities_preserved=True, missing_slots_and_visibility_preserved=True,
                  human_review_performed=False, new_training=False, html_regenerated=False)
    return report


def package(folder, output):
    folder, output = Path(folder), Path(output)
    need(not output.resolve().is_relative_to(folder.resolve()), "ZIP output must be outside its input directory")
    verify_output(folder)
    entries = {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*"))
               if p.is_file() and p.name != "inventory.json" and "__pycache__" not in p.parts
               and p.suffix != ".pyc"}
    inventory = {n: {"bytes": len(b), "sha256": digest(b)} for n, b in entries.items()}
    entries["inventory.json"] = encoded(inventory)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(entries.items()):
            item = zipfile.ZipInfo(name, (2026, 11, 27, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            z.writestr(item, data)
    payload = legacy.archive_payload(output)
    need(set(payload) == set(entries), "packaged member set")
    return {"path": str(output), "bytes": Path(output).stat().st_size,
            "sha256": digest(Path(output).read_bytes()), "members": len(entries)}


def build(archive, audit_path, outlines_path, folder, pdf=None):
    old, payload, previous = load_r44(archive)
    audit, outlines = read(audit_path), read(outlines_path)
    manifest = apply_audit(old, audit, outlines)
    folder.mkdir(parents=True, exist_ok=True)
    # An existing export may only be rebuilt if its predecessor is the same.
    if (folder / "manifest.json").exists():
        need(read(folder / "manifest.json")["previous_manifest_sha256"] == R44_MANIFEST_SHA, "unrelated output directory")
    for name, data in (("history/r44-manifest.json", payload["manifest.json"]),
                       ("history/r43-manifest.json", previous["manifest.json"]),
                       ("audit-ledger.json", encoded(audit)), ("outlines.json", encoded(outlines)),
                       ("manifest.json", encoded(manifest))):
        write(folder / name, data)
    for name in SCRIPTS:
        write(folder / "scripts" / name, Path(__file__).with_name(name).read_bytes())
    (folder / "complete-candidates/labels").mkdir(parents=True, exist_ok=True)
    for frame in manifest["frames"]:
        fid = frame["image_id"]
        for prefix, suffix in (("originals/", ".jpg"), ("source-labels/", ".txt"), ("baseline-labels/", ".txt")):
            name = prefix + fid + suffix
            write(folder / name, previous[name])
        label = legacy.candidate_bytes(frame)
        write(folder / "proposals" / (fid + ".txt"), label)
        candidate = folder / "complete-candidates/labels" / (fid + ".txt")
        if frame["eligible_for_candidate_export"]:
            write(candidate, label)
        elif candidate.exists():
            candidate.unlink()
    report = verify_output(folder)
    write(folder / "verification.json", encoded(report))
    write(folder / "README.txt", (
        "R46 cassette-front annotation review, 27 November 2026 (Asia/Hong_Kong).\n"
        "proposals contains all 12 review label files, including held images.\n"
        "Only complete-candidates/labels passes the recorded whole-image annotation gate.\n"
        "These files do not overwrite the original 137 training / 20 development split.\n"
        "PR133 actually trained frame_0383; this review is separate and does not train.\n"
        "Source row IDs are stable; TXT line order can differ after a historical deletion.\n"
        "Bright edges/shadows support contour interpretation, not automatic snapping.\n"
        "The existing HTML stays at its R44 snapshot; consult this PDF/manifest for R46.\n"
        "Offline check: python scripts/build_front_review_r46.py --check-output .\n"
        "Quick TXT preview (Pillow): python scripts/preview_pose_labels.py --image originals/frame_0001.jpg --labels proposals/frame_0001.txt --output preview.png\n"
        "Preview row numbers are TXT lines. Stable source IDs and any output-row changes are in manifest.json.\n"
    ).encode())
    if pdf:
        from render_front_review_r46 import render_pdf
        render_pdf(folder, pdf)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--r44-archive", type=Path)
    p.add_argument("--download-r44", action="store_true")
    p.add_argument("--audit-ledger", type=Path)
    p.add_argument("--outlines", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--pdf", type=Path)
    p.add_argument("--zip", type=Path)
    p.add_argument("--check-output", type=Path)
    a = p.parse_args()
    if a.check_output:
        print(json.dumps(verify_output(a.check_output), indent=2))
        return
    need(a.output is not None and a.r44_archive is not None, "supply --output and --r44-archive")
    if a.download_r44 and not a.r44_archive.exists():
        a.r44_archive.parent.mkdir(parents=True, exist_ok=True)
        temp = a.r44_archive.with_suffix(".download")
        urllib.request.urlretrieve(R44_URL, temp)
        need(temp.stat().st_size == R44_BYTES and digest(temp.read_bytes()) == R44_SHA, "download identity")
        temp.replace(a.r44_archive)
    report = build(a.r44_archive, a.audit_ledger or a.root / REVIEW / "audit-ledger.json",
                   a.outlines or a.root / REVIEW / "outlines.json", a.output, a.pdf)
    if a.zip:
        a.zip.parent.mkdir(parents=True, exist_ok=True)
        report["package"] = package(a.output, a.zip)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
