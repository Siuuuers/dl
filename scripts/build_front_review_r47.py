#!/usr/bin/env python3
"""Replay the R47 original-image review and export whole-photo candidates.

This is a CPU annotation workflow. File consistency is not physical proof or
human sign-off. Every released field has a recorded, local visual reason.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import urllib.request
import zipfile

sys.dont_write_bytecode = True
import build_front_review_r43 as r43
import build_front_review_r44 as r44
import build_front_review_r46 as r46
import review_front_mining_r42 as legacy

ROOT = Path(__file__).resolve().parents[1]
REVIEW = "data/reviews/additional-fronts-2026-11-27-r47"
R46_REVIEW = r46.REVIEW
R46_MANIFEST_SHA = "e8fe0d6acaf7ec67165a4554f2cf0af85a3c53e865bcd22ad2334f2e1938e6d8"
R46_SOURCE_REVISION = "d49a55d7c2240bbb18565e643cf01e691d39a7a3"
REQUESTS = {f"frame_0001:source:{n:03d}": ["cassette_rack_boundary_review"]
            for n in list(range(2, 31)) + list(range(34, 43))}
REQUESTS.update({"frame_0160:source:001": ["xmin"],
                 "frame_0160:source:037": ["ymin"]})
SCRIPTS = list(dict.fromkeys(r46.SCRIPTS + ["build_front_review_r47.py",
                                           "render_front_review_r47.py"]))
need, digest, encoded = legacy.require, legacy.digest, r43.json_bytes
read, write = r46.read, r46.write


def predecessor(root, archive):
    old44, payload, prior43 = r46.load_r44(archive)
    prior_audit = read(root / R46_REVIEW / "audit-ledger.json")
    prior_outlines = read(root / R46_REVIEW / "outlines.json")
    old46 = r46.apply_audit(old44, prior_audit, prior_outlines)
    need(digest(encoded(old46)) == R46_MANIFEST_SHA, "R46 predecessor identity")
    return old46, old44, payload, prior43, prior_audit, prior_outlines


def apply_audit(old, audit, outlines):
    need(digest(encoded(old)) == R46_MANIFEST_SHA, "R47 predecessor differs")
    need(audit.get("schema_version") == 1 and audit.get("round") == 47, "R47 audit schema")
    need(audit.get("previous_manifest_sha256") == R46_MANIFEST_SHA, "R47 audit lineage")
    need(audit.get("human_review_performed") is False, "human sign-off must not be implied")
    need(audit.get("all_current_rows_inspected") is True, "review scope")
    need(audit["full_originals_inspected"] == [f["image_id"] for f in old["frames"]], "photograph scope")
    previous = r43.all_rows(old)
    current = {sid: r for sid, r in previous.items() if r["after_line"] is not None}
    decisions = audit["rows"]
    need(len(decisions) == len(current) == 507 and
         {r["stable_id"] for r in decisions} == set(current), "507 current rows exactly once")
    out = deepcopy(old)
    rows = r43.all_rows(out)
    frames = {f["image_id"]: f for f in old["frames"]}
    count = dict(reviewed_rows=507, amended_rows=0, retained_rows=0,
                 corner_coordinate_slots_changed=0, box_boundaries_changed=0,
                 added_rows=0, deleted_rows=0, visibility_changes=0,
                 requested_rows=len(REQUESTS), requested_rows_amended=0,
                 resolved_fields=0, rows_with_resolved_fields=0, newly_held_fields=0)
    for entry in decisions:
        sid = entry["stable_id"]
        prior, row = current[sid], rows[sid]
        fid = sid.split(":")[0]
        need(entry["image_id"] == fid and entry["source_row"] == prior["source_row"]
             and entry["source_image_sha256"] == frames[fid]["source_image_sha256"], sid + ": original identity")
        need(entry["inspected"] is True and entry["reason"].strip(), sid + ": visual record")
        need(entry["requested_fields"] == REQUESTS.get(sid, []), sid + ": requested scope")
        need(entry["before_line"] == prior["after_line"], sid + ": predecessor text")
        after = r46.amend_line(entry["before_line"], entry["updates_px"])
        need(after == entry["after_line"], sid + ": coordinate replay")
        actual = r44.changed_fields(entry["before_line"], after)
        need(actual == entry["changed_fields"], sid + ": changed fields")
        resolutions = entry.get("hold_resolutions", [])
        removed = {item["field"] for item in resolutions}
        need(len(removed) == len(resolutions) and removed <= set(prior["unresolved_fields"]), sid + ": resolution scope")
        need(all(item["reason"].strip() for item in resolutions), sid + ": resolution evidence")
        new_holds = entry.get("new_holds", [])
        need(all(item["field"] and item["reason"].strip() for item in new_holds), sid + ": new hold evidence")
        fields = [field for field in prior["unresolved_fields"] if field not in removed]
        # Old unresolved reasons remain in the exact predecessor history. The
        # current record names only still-open fields, avoiding stale assertions
        # that a resolved field is unresolved.
        reasons = (["R47 still unresolved: " + ", ".join(fields) + ". " + entry["reason"]]
                   if fields else [])
        for item in new_holds:
            need(item["field"] not in removed, sid + ": resolve and reopen conflict")
            if item["field"] not in fields:
                fields.append(item["field"])
                reasons.append(item["reason"])
                count["newly_held_fields"] += 1
        if prior.get("latest_update"):
            row.setdefault("review_history", []).append({"round": 46, **deepcopy(prior["latest_update"])})
        row["r46_line"] = entry["before_line"]
        row["after_line"] = after
        row["action"] = "add" if row["source_row"] is None else ("retain" if after == row["before_line"] else "replace")
        row["changed_token_indices"] = r43.changed(row["before_line"], after)
        row["reason"] = prior["reason"] + "\nR47: " + entry["reason"]
        row["unresolved_fields"], row["unresolved_reasons"] = fields, reasons
        row["nominal_visual_uncertainty_px"] = entry["nominal_visual_uncertainty_px"]
        row["latest_update"] = dict(
            status="amended" if actual else ("resolved" if removed else ("held" if fields else "retained")),
            reason=entry["reason"], summary_reason=entry.get("summary_reason", entry["reason"]),
            requested_fields=entry["requested_fields"],
            changed_fields=actual, changed_token_indices=r43.changed(entry["before_line"], after),
            all_current_fields_inspected=True, hold_resolutions=resolutions,
            newly_unresolved_fields=[item["field"] for item in new_holds])
        count["amended_rows"] += bool(actual)
        count["retained_rows"] += not actual
        count["corner_coordinate_slots_changed"] += sum(k in actual for k in ("left_point", "right_point"))
        count["box_boundaries_changed"] += sum(k in actual for k in ("xmin", "ymin", "xmax", "ymax"))
        count["requested_rows_amended"] += sid in REQUESTS and bool(actual)
        count["resolved_fields"] += len(removed)
        count["rows_with_resolved_fields"] += bool(removed)
    out.update(kind="r47-neighbor-guided-visible-front-review", round=47,
        created_date="2026-11-27", previous_manifest_sha256=R46_MANIFEST_SHA,
        previous_round={"round": 46, "manifest_sha256": R46_MANIFEST_SHA,
                        "source_revision": R46_SOURCE_REVISION, "latest_counts": old["latest_counts"]},
        latest_counts=count, audit_sha256=digest(encoded(audit)), outlines_sha256=digest(encoded(outlines)),
        latest_review_scope="All original photos and current fronts inspected. Neighboring contours support local surface ownership; no coordinates are copied blindly. Assistant proposals, not human sign-off.")
    out["policy_notes"] = ["Historical R46: " + s if s.startswith("Current R46 complete-image") else s
                           for s in out["policy_notes"]]
    out["policy_notes"].append("R47: a right-hand cassette's left bound can move right to exclude the intervening rack. Neighbors clarify physical shape but do not supply hidden target coordinates. Include the owned shaded front and molded rim; exclude neighbor-only and rack-only extrema.")
    complete, held = [], []
    for frame in out["frames"]:
        frame["unresolved_stable_ids"] = [r["stable_id"] for r in frame["rows"] + frame["additions"] if r["unresolved_fields"]]
        frame["annotation_complete"] = not frame["unresolved_stable_ids"]
        frame["eligible_for_candidate_export"] = frame["annotation_complete"]
        frame["disposition"] = "assistant_review_complete" if frame["annotation_complete"] else "held"
        frame["all_current_rows_inspected_round"] = 47
        frame["full_image_inspected"] = frame["all_baseline_rows_inspected"] = True
        frame["latest_reviewed_stable_ids"] = [r["stable_id"] for r in frame["rows"] + frame["additions"] if r["after_line"] is not None]
        (complete if frame["annotation_complete"] else held).append(frame["image_id"])
    out["annotation_complete_ids"], out["held_image_ids"] = complete, held
    out["policy_notes"].append("Current R47 complete-image candidates: " + (", ".join(complete) or "none") + ". This is an annotation gate; no training result or independent test assignment is implied.")
    r43.recount(out)
    r46.verify_outlines(outlines, out)
    for item in outlines["items"]:
        for anchor in item.get("context_anchors", []):
            sid, field = anchor["stable_id"], anchor["point"]
            need(sid in rows and rows[sid]["after_line"] is not None
                 and sid.split(":")[0] == item["stable_id"].split(":")[0]
                 and field in ("left_point", "right_point"), "context anchor identity")
            tokens = rows[sid]["after_line"].split()
            need(tokens[7 if field == "left_point" else 10] == "2", "context anchor is unavailable")
            point = r44.geometry(rows[sid]["after_line"])[field]
            need(math.dist(point, anchor["xy_px"]) < 0.01 and anchor["purpose"].strip(),
                 "context anchor must match the recorded locatable point")
    need(set(rows) == set(previous), "stable identities changed")
    need(all(rows[sid] == row for sid, row in previous.items() if row["after_line"] is None), "deleted history changed")
    return out


def verify_output(folder, check_receipt=True):
    folder = Path(folder)
    for name in SCRIPTS:
        need((folder / "scripts" / name).read_bytes() == Path(__file__).with_name(name).read_bytes(),
             "bundled source differs: " + name)
    old44 = read(folder / "history/r44-manifest.json")
    prior = r46.apply_audit(old44, read(folder / "history/r46-audit-ledger.json"),
                           read(folder / "history/r46-outlines.json"))
    need(encoded(prior) == (folder / "history/r46-manifest.json").read_bytes(), "R46 history replay")
    manifest = apply_audit(prior, read(folder / "audit-ledger.json"), read(folder / "outlines.json"))
    need(encoded(manifest) == (folder / "manifest.json").read_bytes(), "R47 manifest replay")
    ids = {f["image_id"] for f in manifest["frames"]}
    allowed = {"manifest.json", "audit-ledger.json", "outlines.json", "README.txt", "verification.json", "inventory.json"}
    allowed.update("history/" + name for name in ("r44-manifest.json", "r46-manifest.json", "r46-audit-ledger.json", "r46-outlines.json"))
    allowed.update("scripts/" + name for name in SCRIPTS)
    for prefix, suffix in (("originals", ".jpg"), ("source-labels", ".txt"),
                           ("baseline-labels", ".txt"), ("proposals", ".txt"), ("previous-labels", ".txt")):
        need({p.name for p in (folder / prefix).iterdir()} == {i + suffix for i in ids}, "unexpected files in " + prefix)
        allowed.update(prefix + "/" + i + suffix for i in ids)
    source = {}
    for frame in manifest["frames"]:
        fid = frame["image_id"]
        for dst, src in (("original.jpg", "originals/" + fid + ".jpg"),
                         ("source.txt", "source-labels/" + fid + ".txt"),
                         (fid + ".txt", "baseline-labels/" + fid + ".txt")):
            source["review/" + fid + "/" + dst] = (folder / src).read_bytes()
        need((folder / "proposals" / (fid + ".txt")).read_bytes() == legacy.candidate_bytes(frame), fid + ": proposal differs")
        old_frame = next(f for f in prior["frames"] if f["image_id"] == fid)
        need((folder / "previous-labels" / (fid + ".txt")).read_bytes() == legacy.candidate_bytes(old_frame), fid + ": prior labels differ")
    expected = {f["image_id"] + ".txt" for f in manifest["frames"] if f["eligible_for_candidate_export"]}
    need({p.name for p in (folder / "complete-candidates/labels").iterdir()} == expected, "complete-image export set")
    for name in expected:
        need((folder / "complete-candidates/labels" / name).read_bytes() == (folder / "proposals" / name).read_bytes(), "candidate bytes")
    allowed.update("complete-candidates/labels/" + name for name in expected)
    if (folder / "artifact-receipt.json").exists() or (folder / "r47-annotation-amendments.pdf").exists():
        receipt = read(folder / "artifact-receipt.json")
        need(receipt == pdf_receipt(folder), "annotation PDF is stale or belongs to different review/source")
        allowed.update(("artifact-receipt.json", "r47-annotation-amendments.pdf"))
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    need(not (actual - allowed), "unexpected review members: " + ", ".join(sorted(actual - allowed)))
    if (folder / "inventory.json").exists():
        inventory = read(folder / "inventory.json")
        need(set(inventory) == actual - {"inventory.json"}, "extracted package inventory membership")
        for name, item in inventory.items():
            raw = (folder / name).read_bytes()
            need(item == {"bytes": len(raw), "sha256": digest(raw)}, "extracted package inventory: " + name)
    report = legacy.verify(manifest, source)
    report.update(round=47, latest_counts=manifest["latest_counts"], previous_manifest_sha256=R46_MANIFEST_SHA,
                  counts_scope="counts is cumulative annotation history; latest_counts contains this R47 round only",
                  all_508_stable_identities_preserved=True, missing_slots_and_visibility_preserved=True,
                  human_review_performed=False, new_training=False, html_regenerated=False)
    if check_receipt:
        need((folder / "verification.json").read_bytes() == encoded(report), "verification receipt differs")
    return report


def pdf_receipt(folder):
    folder = Path(folder)
    path = folder / "r47-annotation-amendments.pdf"
    need(path.is_file(), "annotation PDF receipt has no PDF")
    return {"schema_version": 1, "manifest_sha256": digest((folder / "manifest.json").read_bytes()),
            "pdf_sha256": digest(path.read_bytes()), "pdf_bytes": path.stat().st_size,
            "render_sources": {name: digest(Path(__file__).with_name(name).read_bytes())
                               for name in ("render_front_review_r47.py", "render_front_review_r46.py")}}


def build(root, archive, audit_path, outlines_path, folder, pdf=None):
    old46, old44, payload, prior43, prior_audit, prior_outlines = predecessor(root, archive)
    audit, outlines = read(audit_path), read(outlines_path)
    manifest = apply_audit(old46, audit, outlines)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").exists():
        need((folder / "manifest.json").read_bytes() == encoded(manifest),
             "Existing review has different decisions; preserve it and choose a fresh output folder")
    for name, data in (("history/r44-manifest.json", encoded(old44)),
                       ("history/r46-manifest.json", encoded(old46)),
                       ("history/r46-audit-ledger.json", encoded(prior_audit)),
                       ("history/r46-outlines.json", encoded(prior_outlines)),
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
            write(folder / name, prior43[name])
        old_frame = next(f for f in old46["frames"] if f["image_id"] == fid)
        write(folder / "previous-labels" / (fid + ".txt"), legacy.candidate_bytes(old_frame))
        label = legacy.candidate_bytes(frame)
        write(folder / "proposals" / (fid + ".txt"), label)
        candidate = folder / "complete-candidates/labels" / (fid + ".txt")
        if frame["eligible_for_candidate_export"]:
            write(candidate, label)
        elif candidate.exists():
            candidate.unlink()
    write(folder / "README.txt", (
        "R47 cassette-front review, 27 November 2026 (Asia/Hong_Kong).\n"
        "All 12 original photos and 507 current fronts were reviewed.\n"
        "proposals includes held images. Only complete-candidates/labels passes the whole-image annotation gate.\n"
        "previous-labels is the exact R46 predecessor. Current stable IDs map to TXT rows in manifest.json.\n"
        "No original training/development split or evaluation label is overwritten.\n"
        "Neighbor contours support local interpretation, not blind copying or invented hidden points.\n"
        "No model is trained or promoted by this review. PR133 remains the latest measured pilot.\n"
        "Offline check: python scripts/build_front_review_r47.py --check-output .\n"
        "Quick preview: python scripts/preview_pose_labels.py --image originals/frame_0160.jpg --labels proposals/frame_0160.txt --output preview.png\n"
    ).encode())
    if pdf:
        inside = Path(pdf).resolve().is_relative_to(folder.resolve())
        if inside:
            need(Path(pdf).resolve() == (folder / "r47-annotation-amendments.pdf").resolve(),
                 "In-folder annotation PDF must use r47-annotation-amendments.pdf")
        from render_front_review_r47 import render_pdf
        render_pdf(folder, pdf)
        if inside:
            write(folder / "artifact-receipt.json", encoded(pdf_receipt(folder)))
    report = verify_output(folder, check_receipt=False)
    write(folder / "verification.json", encoded(report))
    return report


def package(folder, output):
    folder, output = Path(folder), Path(output)
    need(not output.resolve().is_relative_to(folder.resolve()), "ZIP must be outside its input directory")
    verify_output(folder)
    entries = {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*"))
               if p.is_file() and p.name != "inventory.json" and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    entries["inventory.json"] = encoded({n: {"bytes": len(b), "sha256": digest(b)} for n, b in entries.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(entries.items()):
            item = zipfile.ZipInfo(name, (2026, 11, 27, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            z.writestr(item, data)
    payload = legacy.archive_payload(output)
    need(set(payload) == set(entries), "package member set")
    return {"path": str(output), "bytes": output.stat().st_size,
            "sha256": digest(output.read_bytes()), "members": len(entries)}


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
        urllib.request.urlretrieve(r46.R44_URL, temp)
        need(temp.stat().st_size == r46.R44_BYTES and digest(temp.read_bytes()) == r46.R44_SHA, "R44 download identity")
        temp.replace(a.r44_archive)
    report = build(a.root, a.r44_archive, a.audit_ledger or a.root / REVIEW / "audit-ledger.json",
                   a.outlines or a.root / REVIEW / "outlines.json", a.output, a.pdf)
    if a.zip:
        report["package"] = package(a.output, a.zip)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
