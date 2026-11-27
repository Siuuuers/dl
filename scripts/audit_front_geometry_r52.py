#!/usr/bin/env python3
"""Screen all R48 supplementary labels for four-side review cues; never edit.

The R28 interior-corner and lower-gap rules are retained. R52 adds edge contacts
on all four sides and an explicit stable-identity map. These are review cues,
not physical laws: perspective and axis-aligned envelopes permit overlap.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = "data/reviews/additional-fronts-2026-11-27-r48"
AUDIT_SHA = "b3625992a9baae56a48689c89962f265c94821263853a13b40790795f0733ffc"
EDGES = ("xmin", "ymin", "xmax", "ymax")
TOLERANCE_PX = 2.0
REQUESTED = {f"frame_0160:source:{n:03d}" for n in (15, 16, 19, 20, 24, 25, 29, 30)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def geometry(line):
    values = list(map(float, line.split()))
    if len(values) != 11 or not all(map(math.isfinite, values)):
        raise ValueError("Expected finite two-corner YOLO pose row")
    cx, cy, width, height = values[1:5]
    box = [(cx-width/2)*1280, (cy-height/2)*720,
           (cx+width/2)*1280, (cy+height/2)*720]
    points = [{"side": side, "x": values[j]*1280, "y": values[j+1]*720}
              for side, j in (("left", 5), ("right", 8)) if values[j+2] > 0]
    return box, points


def load_rows(root):
    base = root / PARENT
    if sha(base / "audit-ledger.json") != AUDIT_SHA:
        raise ValueError("Immutable R48 audit differs")
    ledger = json.loads((base / "audit-ledger.json").read_bytes())
    release = json.loads((base / "release-status.json").read_bytes())
    rows = []
    for image_id in sorted({r["image_id"] for r in ledger["rows"]}):
        source = [r for r in ledger["rows"] if r["image_id"] == image_id]
        path = base / "proposals" / (image_id + ".txt")
        if sha(path) != release["files"]["proposals/"+image_id+".txt"]:
            raise ValueError("R48 proposal hash differs: " + image_id)
        lines = path.read_text().splitlines()
        # The ledger is in the same current-row order as the exported TXT.
        # A historical deletion therefore cannot silently shift source IDs.
        if lines != [r["after_line"] for r in source]:
            raise ValueError("Stable row mapping differs: " + image_id)
        for output_row, old in enumerate(source, 1):
            box, points = geometry(old["after_line"])
            rows.append({"stable_id": old["stable_id"], "image_id": image_id,
                         "source_row": old["source_row"], "output_row": output_row,
                         "line": old["after_line"], "source_image_sha256": old["source_image_sha256"],
                         "box_xyxy_px": box, "points": points,
                         "remaining_holds": old["remaining_holds"],
                         "annotation_complete": image_id in release["annotation_complete_ids"]})
    if len(rows) != 507 or len({r["stable_id"] for r in rows}) != 507:
        raise ValueError("Expected 507 unique current supplementary identities")
    return rows, release


def screen(rows):
    records = []
    counts = Counter()
    side_counts = Counter()
    for row in rows:
        x0, y0, x1, y1 = row["box_xyxy_px"]
        interior, contacts = [], []
        for other in rows:
            if other["image_id"] != row["image_id"] or other["stable_id"] == row["stable_id"]:
                continue
            for p in other["points"]:
                x, y = p["x"], p["y"]
                distances = {"xmin": x-x0, "ymin": y-y0, "xmax": x1-x, "ymax": y1-y}
                item = {"other_stable_id": other["stable_id"], "other_output_row": other["output_row"],
                        **p, "signed_inward_distance_px": distances}
                if min(distances.values()) > TOLERANCE_PX:
                    interior.append({**item, "nearest_edge": min(distances, key=distances.get)})
                edges = []
                for edge in ("xmin", "xmax"):
                    if y0-TOLERANCE_PX <= y <= y1+TOLERANCE_PX and abs(distances[edge]) <= TOLERANCE_PX:
                        edges.append(edge)
                for edge in ("ymin", "ymax"):
                    if x0-TOLERANCE_PX <= x <= x1+TOLERANCE_PX and abs(distances[edge]) <= TOLERANCE_PX:
                        edges.append(edge)
                if edges:
                    contacts.append({**item, "edges": edges})
                    side_counts.update(edges)
        ys = [p["y"] for p in row["points"]]
        lower_gap = y1-max(ys) if ys else None
        lower_threshold = max(5., .25*(y1-y0))
        flags = []
        if interior:
            flags.append("foreign_corner_strictly_inside")
        if contacts:
            flags.append("foreign_corner_within_2px_of_edge")
        if lower_gap is not None and lower_gap > lower_threshold:
            flags.append("large_lower_edge_gap")
        own_outside = [p for p in row["points"] if
                       min(p["x"]-x0, p["y"]-y0, x1-p["x"], y1-p["y"]) < -TOLERANCE_PX]
        if own_outside:
            flags.append("own_corner_over_2px_outside")
        item = {**row, "flags": flags, "requested": row["stable_id"] in REQUESTED,
                "foreign_interior_corners": interior, "foreign_edge_contacts": contacts,
                "own_corners_outside": own_outside, "lower_gap_px": lower_gap,
                "lower_gap_threshold_px": lower_threshold}
        records.append(item)
        counts["rows"] += 1
        counts["flagged_rows"] += bool(flags)
        counts["strict_interior_rows"] += bool(interior)
        counts["edge_contact_rows"] += bool(contacts)
        counts["strict_interior_corner_events"] += len(interior)
        counts["edge_contact_corner_events"] += len(contacts)
        counts["large_lower_gap_rows"] += "large_lower_edge_gap" in flags
        counts["own_corner_outside_rows"] += bool(own_outside)
        counts["locatable_corners"] += len(row["points"])
    return {"counts": dict(counts), "edge_contact_side_counts": dict(side_counts), "rows": records}


def make_report(root, rows):
    result = screen(rows)
    # Strict interior and containment violations are listed before ordinary
    # stack contacts; contact-heavy images must not dominate the first page.
    priority = {"own_corner_over_2px_outside": 0, "foreign_corner_strictly_inside": 1,
                "large_lower_edge_gap": 2, "foreign_corner_within_2px_of_edge": 3}
    queue = sorted((r for r in result["rows"] if r["flags"]),
                   key=lambda r: (min(priority[f] for f in r["flags"]),
                                  -len(r["foreign_interior_corners"]), r["stable_id"]))
    return {"schema_version": 1, "round": 52, "kind": "four_side_geometry_review_cues",
            "parent_audit": {"path": PARENT+"/audit-ledger.json", "sha256": AUDIT_SHA},
            "script_sha256": sha(Path(__file__)),
            "coordinates": "Original 1280x720 pixels; x increases right, y increases down.",
            "method": {"strict_interior": "Retain R28 criterion: foreign locatable point >2px inside all four edges.",
                       "edge_contact": "Foreign locatable point within +/-2px of an edge and within the other axis span expanded by 2px; record each qualifying edge.",
                       "lower_gap": "Retain R28 criterion: ymax-max(locatable lower-corner y) > max(5px,0.25*box height).",
                       "upper_gap": "No zero upper-gap constraint: lower-front landmarks naturally lie below the top boundary.",
                       "inputs": "All 507 current R48 supplementary TXT rows, with exact stable IDs; no predictions used.",
                       "action": "Review priorities only. Never move a box from geometry alone.",
                       "ranking": "Own-corner violation, then strict foreign interior, then large lower gap, then ordinary edge contact; retain separate counts.",
                       "fixed_tolerance_px": TOLERANCE_PX},
            "limits": ["An axis-aligned envelope can contain background or another object's projected point.",
                       "Perspective, adjoining surfaces, occlusion and 2px image blur can create legitimate contacts.",
                       "Unavailable 0 0 0 corners are excluded; no location is invented.",
                       "Raw counts are diagnostic workload, not annotation-error rates or model precision."],
            "priority_queue": [r["stable_id"] for r in queue], **result}


def render(root, rows, originals, out):
    from PIL import Image, ImageDraw, ImageFont
    out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    inventory = []
    for image_id in sorted({r["image_id"] for r in rows}):
        records = [r for r in rows if r["image_id"] == image_id]
        path = originals / (image_id+".jpg")
        if sha(path) != records[0]["source_image_sha256"]:
            raise ValueError("Original photo hash differs: "+image_id)
        original = Image.open(path).convert("RGB")
        if original.size != (1280, 720):
            raise ValueError("Original dimensions differ")
        context = original.copy()
        draw = ImageDraw.Draw(context)
        for r in records:
            draw.rectangle(r["box_xyxy_px"], outline=(255,210,20), width=1)
            draw.text(tuple(r["box_xyxy_px"][:2]), str(r["source_row"]), font=font,
                      fill=(255,70,50), stroke_width=1, stroke_fill="white")
        context.save(out/(image_id+"-full.png"))
        # Every row receives an original-pixel crop, paired with the current
        # exact box and corners. Nearest-neighbor enlargement adds no detail.
        panels = []
        for r in records:
            x0,y0,x1,y1 = r["box_xyxy_px"]
            crop_box = (max(0, math.floor(x0)-10), max(0, math.floor(y0)-8),
                        min(1280, math.ceil(x1)+10), min(720, math.ceil(y1)+8))
            raw = original.crop(crop_box)
            scale = min(4., 278/raw.width, 92/raw.height)
            size = (max(1,round(raw.width*scale)),max(1,round(raw.height*scale)))
            raw = raw.resize(size, Image.Resampling.NEAREST)
            overlay = raw.copy()
            d = ImageDraw.Draw(overlay)
            tr = lambda x,y: ((x-crop_box[0])*scale,(y-crop_box[1])*scale)
            d.rectangle((*tr(x0,y0),*tr(x1,y1)),outline=(255,210,20),width=1)
            for p in r["points"]:
                x,y=tr(p["x"],p["y"])
                d.ellipse((x-2,y-2,x+2,y+2),outline=(0,255,255),width=1)
            panel = Image.new("RGB",(288,224),"white")
            pd=ImageDraw.Draw(panel)
            pd.text((5,2),f"s{r['source_row']} / txt{r['output_row']}" + ("  FLAG" if r.get("flags") else ""),font=font,fill="black")
            panel.paste(raw,(5,22));panel.paste(overlay,(5,126))
            panels.append(panel)
        # Two modest-height pages per photo remain readable without a viewer
        # downsampling a single very tall sheet.
        for offset in range(0,len(panels),24):
            part=panels[offset:offset+24]
            sheet=Image.new("RGB",(4*288,30+math.ceil(len(part)/4)*224),(235,238,241))
            ImageDraw.Draw(sheet).text((8,6),image_id+"  raw above / R48 overlay below; 2px review tolerance",font=font,fill="black")
            for n,panel in enumerate(part):sheet.paste(panel,((n%4)*288,30+(n//4)*224))
            name=f"{image_id}-rows-{offset+1:03d}-{offset+len(part):03d}.png"
            sheet.save(out/name)
            inventory.append({"file": name,"image_id": image_id,
                              "stable_ids": [r["stable_id"] for r in records[offset:offset+len(part)]]})
    (out/"sheet-index.json").write_bytes(encoded(inventory))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--originals",type=Path)
    parser.add_argument("--sheets",type=Path)
    args=parser.parse_args()
    rows,_=load_rows(args.root)
    result=make_report(args.root,rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(encoded(result))
    if args.sheets:
        if not args.originals:parser.error("--sheets requires --originals")
        render(args.root,result["rows"],args.originals,args.sheets)
    print(json.dumps({"output":str(args.output),"counts":result["counts"],
                      "side_counts":result["edge_contact_side_counts"]},indent=2))
