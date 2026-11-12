#!/usr/bin/env python3
"""Render exact original-pixel evidence for the new_000010 corner review."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    review_path = args.review_root / "results/review/visible-corners-20261112/review.json"
    replay_path = args.review_root / "results/review/visible-corners-20261112/replay.json"
    review, replay = json.loads(review_path.read_bytes()), json.loads(replay_path.read_bytes())
    source = args.repo_root / "data/images/val/new_000010.png"
    if digest(source) != review["source_image"]["sha256"]:
        raise ValueError("Original image identity differs")
    original = Image.open(source).convert("RGB")
    if original.size != (1280, 720):
        raise ValueError("Expected original 1280x720 image")
    decisions = {d["row"]: d for d in review["accepted_refinements"]}
    model_rows = {d["row_one_based"]: d for d in replay["models"]["current"]["affected_rows"]}
    colors = {"bg": "#101821", "ink": "#f4f6f9", "muted": "#b7c4d2",
              "old": "#45dbad", "new": "#ffc065", "model": "#69baff"}
    font = lambda size: ImageFont.truetype("DejaVuSans.ttf", size)
    outputs = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save(canvas, name, metadata):
        path = args.output_dir / name
        if path.exists():
            raise FileExistsError(path)
        canvas.save(path)
        outputs.append({"path": name, "sha256": digest(path), "size": list(canvas.size), **metadata})

    # Four panels keep every marker out of the unmarked reference view.
    scale, patch_size, margin, gap = 8, 256, 24, 18
    width = margin * 2 + patch_size * 4 + gap * 3
    canvas = Image.new("RGB", (width, 902), colors["bg"])
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "Which point follows the visible turn?", fill=colors["ink"], font=font(27))
    draw.text((margin, 59), "new_000010 | right corners, rows 23 and 24 | original pixels enlarged 8x", fill=colors["muted"], font=font(17))
    crops = {23: [900, 439, 932, 471], 24: [893, 471, 925, 503]}
    panels = [("ORIGINAL", None, None), ("INHERITED LABEL", "old", "before_xy"),
              ("REVIEWED TURN", "new", "selected_xy"), ("PR35 OUTPUT", "model", None)]
    metadata = []
    for i, row in enumerate((23, 24)):
        y0 = 120 + i * 352
        decision, crop = decisions[row], crops[row]
        endpoints = model_rows[row]["endpoints"]
        prediction = next(e for e in endpoints if e["side"] == "right")
        draw.text((margin, y0 - 35), f"ROW {row}", fill=colors["ink"], font=font(19))
        patch = original.crop(crop).resize((patch_size, patch_size), Image.Resampling.NEAREST)
        for j, (title, color_key, field) in enumerate(panels):
            x0 = margin + j * (patch_size + gap)
            draw.text((x0, y0), title, fill=colors[color_key] if color_key else colors["muted"], font=font(16))
            canvas.paste(patch, (x0, y0 + 26))
            point = None if j == 0 else prediction["prediction_xy"] if j == 3 else decision[field]
            if point:
                x, y = x0 + (point[0] - crop[0]) * scale, y0 + 26 + (point[1] - crop[1]) * scale
                color = colors[color_key]
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)
                for xa, ya, xb, yb in ((x-12,y,x-8,y),(x+8,y,x+12,y),(x,y-12,x,y-8),(x,y+8,x,y+12)):
                    draw.line((xa, ya, xb, yb), fill=color, width=2)
                draw.text((x0, y0 + 290), f"({point[0]:.2f}, {point[1]:.2f})", fill=colors["muted"], font=font(16))
        metadata.append({"row": row, "crop_xyxy": crop, "scale": scale,
                         "inherited_xy": decision["before_xy"], "reviewed_xy": decision["selected_xy"],
                         "pr35_xy": prediction["prediction_xy"]})
    draw.text((margin, 831), "Visible rounded bends support a representative point, not subpixel certainty.", fill=colors["muted"], font=font(18))
    draw.text((margin, 860), "Reviewed points came from original-image judgments. No geometric estimates are shown.", fill=colors["muted"], font=font(17))
    save(canvas, "visible-corner-turns-20261112.png", {"cases": metadata, "geometry_shown": False})

    # Every accepted change gets an unmarked crop beside the before/after overlay.
    tile_w, tile_h = 368, 284
    rows = sorted(decisions)
    nrows = (len(rows) + 2) // 3
    sheet = Image.new("RGB", (24 + tile_w * 3, 108 + tile_h * nrows + 48), colors["bg"])
    d = ImageDraw.Draw(sheet)
    d.text((24, 18), "Accepted visible-corner refinements", fill=colors["ink"], font=font(26))
    d.text((24, 57), "Each pair: unmarked original | inherited green + reviewed amber. No PR35 points.", fill=colors["muted"], font=font(17))
    entries = []
    for i, row in enumerate(rows):
        decision = decisions[row]
        px, py = decision["selected_xy"]
        crop = [int(px)-18, int(py)-24, int(px)+18, int(py)+12]
        patch = original.crop(crop).resize((162, 162), Image.Resampling.NEAREST)
        x0, y0 = 24 + (i % 3) * tile_w, 108 + (i // 3) * tile_h
        d.text((x0, y0), f"Row {row} / {decision['side']}", fill=colors["ink"], font=font(18))
        for j in range(2):
            ox, oy = x0 + j * 174, y0 + 33
            sheet.paste(patch, (ox, oy))
            if j:
                for key, color in (("before_xy", colors["old"]), ("selected_xy", colors["new"])):
                    x, y = ox + (decision[key][0]-crop[0])*4.5, oy + (decision[key][1]-crop[1])*4.5
                    d.ellipse((x-4,y-4,x+4,y+4), outline=color, width=2)
        a,b = decision["before_xy"], decision["selected_xy"]
        d.text((x0,y0+208),f"Inherited ({a[0]:.2f}, {a[1]:.2f})",fill=colors["old"],font=font(15))
        d.text((x0,y0+232),f"Reviewed ({b[0]:.0f}, {b[1]:.0f})",fill=colors["new"],font=font(15))
        entries.append({"row":row,"side":decision["side"],"crop_xyxy":crop,"nearest_neighbor_scale":4.5})
    d.text((24,sheet.height-36),"Source pixels unchanged. Refinements remain approximate visual annotations; hidden slots stay missing.",fill=colors["muted"],font=font(16))
    save(sheet,"visible-corner-all-refinements-20261112.png",{"cases":entries,"model_points_shown":False})

    # A refinement can expose a worse PR35 error; show the competing saved model.
    crop, scale = [443, 615, 485, 672], 7
    patch = original.crop(crop).resize((294,399),Image.Resampling.NEAREST)
    counter = Image.new("RGB",(954,633),colors["bg"])
    d = ImageDraw.Draw(counter)
    d.text((24,18),"Row 31: a useful counterexample",fill=colors["ink"],font=font(26))
    d.text((24,57),"Left lower-front turn beside the hand | same revised label for both models",fill=colors["muted"],font=font(17))
    values=[]
    for j,(title,role,color) in enumerate((("ORIGINAL",None,None),("PR35", "current",colors["model"]),("PR52", "endpoint_reviewed","#e3a1f7"))):
        x0,y0=24+j*312,130
        d.text((x0,96),title,fill=color or colors["muted"],font=font(19))
        counter.paste(patch,(x0,y0))
        if role:
            record=next(r for r in replay["models"][role]["affected_rows"] if r["row_one_based"]==31)
            endpoint=next(e for e in record["endpoints"] if e["side"]=="left")
            for point,c in ((decisions[31]["selected_xy"],colors["new"]),(endpoint["prediction_xy"],color)):
                x=x0+(point[0]-crop[0])*scale;y=y0+(point[1]-crop[1])*scale
                d.ellipse((x-7,y-7,x+7,y+7),outline=c,width=2)
            d.text((x0,545),f"Error: {endpoint['amended_error_px']:.2f} px",fill=color,font=font(20))
            values.append({"role":role,"prediction_xy":endpoint["prediction_xy"],"error_px":endpoint["amended_error_px"]})
    d.text((24,585),"Amber = reviewed (457, 641). The right endpoint remains hidden and unscored.",fill=colors["muted"],font=font(17))
    save(counter,"visible-corner-row31-20261112.png",{"crop_xyxy":crop,"scale":scale,"reviewed_xy":decisions[31]["selected_xy"],"models":values})

    # An unresolved material-boundary question is shown without nominated points.
    crop, scale = [910, 395, 981, 500], 6
    patch = original.crop(crop).resize((426,630),Image.Resampling.NEAREST)
    question = Image.new("RGB",(894,780),colors["bg"])
    d = ImageDraw.Draw(question)
    d.text((24,18),"Open question: the shaded left boundary",fill=colors["ink"],font=font(25))
    d.text((24,57),"new_000010 / rightmost stack, rows 26-28 / existing labels retained",fill=colors["muted"],font=font(16))
    question.paste(patch,(24,100))
    text_lines=["Which material forms the target face?","","The outer shaded contour and the","brighter interior transition are different.","","Brightness alone cannot identify","the actual lower-front junction.","","No alternate point is accepted here.","A physical description of this cassette","would help resolve the boundary.","","Original crop [910,395,981,500]","Nearest-neighbor enlargement, 6x"]
    for i,line in enumerate(text_lines):
        d.text((474,119+i*32),line,fill=colors["ink"] if i==0 else colors["muted"],font=font(17))
    save(question,"visible-corner-shaded-boundary-20261112.png",{"crop_xyxy":crop,"scale":scale,"accepted_edits":False})

    index = {"source_image_sha256":digest(source),"review_sha256":digest(review_path),
             "replay_sha256":digest(replay_path),"renderer_sha256":digest(Path(__file__)),
             "pillow_version":Image.__version__,"processing":"Original RGB crops and nearest-neighbor sampling; no enhancement.","figures":outputs}
    index_path=args.output_dir/"visible-corner-figure-index-20261112.json"
    with index_path.open("x") as stream:
        stream.write(json.dumps(index,indent=2)+"\n")
    print(json.dumps({"figures":[x["path"] for x in outputs],"index":str(index_path)}))


if __name__ == "__main__":
    main()
