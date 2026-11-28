#!/usr/bin/env python3
"""Verify editable thesis equations against the manuscript and rendered PDF.

Run with CODEX_PRIMARY_RUNTIME_PYTHON. This checks the complete expression's
centering and visible bounds, not merely the existence of a center setting.
The PDF must come from the exact DOCX supplied. Inspect every final page PNG
as well: these measurements do not replace visual review in Microsoft Word.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import fitz
from lxml import etree

import build_thesis_docx_r56 as builder

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": builder.MATH_NS}


def attr(node, key="w:val"):
    prefix, name = key.split(":")
    return None if node is None else node.get(f"{{{NS[prefix]}}}{name}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def math_signature(nodes):
    return {
        "text": "".join(t.text or "" for node in nodes for t in node.iter(f"{{{NS['m']}}}t")),
        "structures": dict(Counter(etree.QName(item).localname for node in nodes for item in node.iter()
                                   if etree.QName(item).namespace == NS["m"]
                                   and etree.QName(item).localname not in ("r", "rPr", "t", "sty", "nor", "scr"))),
    }


def document_checks(source, docx):
    _, blocks, _ = builder.parse_markdown(source)
    formulas = [b for b in blocks if b.kind == "equation"]
    compiled = builder.compile_equations(formulas)
    with ZipFile(docx) as archive:
        root = etree.fromstring(archive.read("word/document.xml"))
    tables = root.findall(".//w:tbl", NS)
    equation_tables = [table for table in tables if table.find(".//m:oMathPara", NS) is not None]
    require(len(equation_tables) == len(formulas), "The display-equation count does not match the manuscript")
    require(len(tables) - len(equation_tables) == sum(b.kind == "table" for b in blocks),
            "Captioned data tables and equation layout rows were conflated")
    require(len(root.findall(".//m:oMathPara", NS)) == len(formulas), "Unexpected extra display math")
    section = root.find(".//w:sectPr", NS)
    size, margin = section.find("w:pgSz", NS), section.find("w:pgMar", NS)
    page_width = int(attr(size, "w:w"))
    left, right = int(attr(margin, "w:left")), int(attr(margin, "w:right"))
    result = []
    for formula, table in zip(formulas, equation_tables):
        label = f"Equation {formula.data}"
        rows = table.findall("w:tr", NS)
        require(len(rows) == 1, label + " has more than one layout row")
        row = rows[0]
        cells = row.findall("w:tc", NS)
        require(len(cells) == 3, label + " has no symmetric three-cell layout")
        widths = [int(attr(cell.find("w:tcPr/w:tcW", NS), "w:w")) for cell in cells]
        require(all(w > 0 for w in widths) and widths[0] == widths[2], label + " has unequal or invalid side widths")
        require(abs(sum(widths) - (page_width - left - right)) <= 2, label + " does not fill the text width")
        require(not row.findall("w:trPr/w:trHeight", NS), label + " has a fixed row height")
        require(row.find("w:trPr/w:cantSplit", NS) is not None, label + " may split across pages")
        borders = table.findall("w:tblPr/w:tblBorders/*", NS)
        require(len(borders) == 6 and all(attr(b) == "nil" for b in borders), label + " has visible layout borders")
        paragraph = cells[1].find("w:p", NS)
        require(attr(paragraph.find("w:pPr/w:jc", NS)) == "center", label + " paragraph is not centered")
        display = paragraph.find("m:oMathPara", NS)
        require(attr(display.find("m:oMathParaPr/m:jc", NS), "m:val") == "center", label + " display math is not centered")
        spacing = paragraph.find("w:pPr/w:spacing", NS)
        require(attr(spacing, "w:lineRule") == "auto", label + " does not have automatic line height")
        require(not table.findall(".//w:tab", NS), label + " still contains positioning tabs")
        require(not table.findall(".//w:fitText", NS), label + " compresses text to fit")
        number = "".join(cells[2].itertext()).strip()
        require(number == f"({formula.data})", label + " number does not match its source")
        require(attr(cells[2].find("w:p/w:pPr/w:jc", NS)) == "right", label + " number is not right-aligned")
        equation = display.find("m:oMath", NS)
        require(math_signature(list(equation)) == math_signature(compiled[formula.text]),
                label + " has lost or changed mathematical content")
        literal_words = [run.find("m:t", NS).text.strip() for run in equation.findall(".//m:r", NS)
                         if run.find("m:rPr/m:nor", NS) is not None
                         and attr(run.find("m:rPr/m:sty", NS), "m:val") == "p"
                         and (run.find("m:t", NS).text or "").strip().isalpha()]
        result.append({"number": formula.data, "cell_widths_twips": widths,
                       "paragraph_alignment": "center", "display_alignment": "center",
                       "line_height": "automatic", "row_height": "automatic",
                       "source_math_preserved": True, "upright_words": sorted(set(literal_words)),
                       "fraction_count": len(equation.findall(".//m:f", NS))})
    return {"page_width_twips": page_width, "left_margin_twips": left, "right_margin_twips": right,
            "data_tables": len(tables) - len(equation_tables), "equation_layout_rows": len(equation_tables),
            "equations": result}


def pdf_checks(pdf_path, structure):
    left = structure["left_margin_twips"] / 20
    right = (structure["page_width_twips"] - structure["right_margin_twips"]) / 20
    center = (left + right) / 2
    result = []
    with fitz.open(pdf_path) as pdf:
        pages = [[s for block in page.get_text("dict")["blocks"] if block.get("type") == 0
                  for line in block["lines"] for s in line["spans"]] for page in pdf]
        for equation in structure["equations"]:
            label = f"({equation['number']})"
            matches = [(i, span) for i, spans in enumerate(pages) for span in spans
                       if span["text"].strip() == label and span["bbox"][0] > center + 80]
            require(len(matches) == 1, "Rendered equation number missing or duplicated: " + label)
            page_index, number = matches[0]
            top, bottom = number["bbox"][1], number["bbox"][3]
            selected = []
            # Grow the vertical band through overlapping glyph bounds. This
            # includes stacked fractions and scripts, but stops at the spacing
            # separating the display equation from ordinary paragraphs.
            for _ in range(20):
                selected = [s for s in pages[page_index] if s is not number and s["text"].strip()
                            and s["bbox"][0] >= left - 1 and s["bbox"][2] <= right + 1
                            and s["bbox"][1] < bottom and s["bbox"][3] > top]
                new_top = min([top] + [s["bbox"][1] for s in selected])
                new_bottom = max([bottom] + [s["bbox"][3] for s in selected])
                if (new_top, new_bottom) == (top, bottom):
                    break
                top, bottom = new_top, new_bottom
            require(selected, "No formula glyphs near equation " + label)
            bounds = fitz.Rect(selected[0]["bbox"])
            for span in selected[1:]:
                bounds |= fitz.Rect(span["bbox"])
            # Fraction rules and radical strokes can protrude beyond a glyph's
            # text bounds. Include them in the width and height measurement.
            for drawing in pdf[page_index].get_drawings():
                rect = drawing["rect"]
                if rect.x0 >= left and rect.x1 <= number["bbox"][0] - 4 and rect.y0 <= bottom + 1 and rect.y1 >= top - 1:
                    bounds |= rect
            side_width = equation["cell_widths_twips"][0] / 20
            center_error = abs((bounds.x0 + bounds.x1) / 2 - center)
            horizontal_clearance = min(bounds.x0 - (left + side_width), (right - side_width) - bounds.x1)
            number_gap = number["bbox"][0] - bounds.x1
            require(bounds.width > 0 and bounds.height > 0, "Equation has non-positive visible bounds: " + label)
            require(center_error <= 2, f"Equation {label} is off-center by {center_error:.2f} pt")
            require(horizontal_clearance >= 1, "Equation exceeds its available width: " + label)
            require(number_gap >= 5, "Equation collides with its number: " + label)
            require(bounds.y0 >= 60 and bounds.y1 <= pdf[page_index].rect.height - 60,
                    "Equation is too close to the page edge: " + label)
            require(abs(number["bbox"][2] - right) < 2, "Equation number is not at the right margin: " + label)
            rendered_words = {}
            for word in equation["upright_words"]:
                matching = [s for s in selected if word in s["text"]]
                require(matching, f"Upright word {word!r} is missing from equation {label}")
                require(all(not (s["flags"] & 2) for s in matching), f"Word {word!r} rendered in italics in equation {label}")
                rendered_words[word] = "upright"
            require(not any("\ufffd" in s["text"] for s in selected), "Equation has a replacement glyph: " + label)
            fractions = []
            if equation["fraction_count"]:
                bars = [drawing["rect"] for drawing in pdf[page_index].get_drawings()
                        if drawing["rect"].width > 5 and drawing["rect"].height <= 1.5
                        and drawing["rect"].x0 >= bounds.x0 and drawing["rect"].x1 <= bounds.x1
                        and drawing["rect"].y0 >= bounds.y0 and drawing["rect"].y1 <= bounds.y1]
                require(len(bars) == equation["fraction_count"], "Fraction-rule count is wrong in equation " + label)
                characters = [c for block in pdf[page_index].get_text("rawdict")["blocks"] if block.get("type") == 0
                              for line in block["lines"] for span in line["spans"] for c in span["chars"]
                              if c["c"].strip()]
                for bar in bars:
                    bar_center = (bar.x0 + bar.x1) / 2
                    bar_y = (bar.y0 + bar.y1) / 2
                    errors = {}
                    for part in ("numerator", "denominator"):
                        chars = [c for c in characters
                                 if bar.x0 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= bar.x1
                                 and c["bbox"][1] >= bounds.y0 - 0.1 and c["bbox"][3] <= bounds.y1 + 0.1
                                 and (((c["bbox"][1] + c["bbox"][3]) / 2 < bar_y) == (part == "numerator"))]
                        require(chars, f"Missing fraction {part} in equation {label}")
                        part_left = min(c["bbox"][0] for c in chars)
                        part_right = max(c["bbox"][2] for c in chars)
                        error = abs((part_left + part_right) / 2 - bar_center)
                        require(error <= 1.5, f"Fraction {part} in equation {label} is off-center by {error:.2f} pt")
                        errors[part + "_center_error_pt"] = round(error, 3)
                    fractions.append(errors)
            result.append({"number": equation["number"], "page": page_index + 1,
                           "visible_bounds_pt": [round(x, 3) for x in bounds],
                           "width_pt": round(bounds.width, 3), "height_pt": round(bounds.height, 3),
                           "center_error_pt": round(center_error, 3),
                           "horizontal_clearance_pt": round(horizontal_clearance, 3),
                           "number_gap_pt": round(number_gap, 3), "rendered_words": rendered_words,
                           "fraction_alignment": fractions})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--docx", required=True, type=Path)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    structure = document_checks(args.source, args.docx)
    rendered = pdf_checks(args.pdf, structure)
    receipt = {"schema": "thesis-equation-layout-v1", "status": "passed",
               "source_sha256": sha256(args.source), "docx_sha256": sha256(args.docx),
               "rendered_pdf_sha256": sha256(args.pdf), "structure": structure, "rendered_equations": rendered,
               "scope": "Native XML structure and rendered glyph/stroke geometry. Separate visual review of every final page is required; this is not a Microsoft Word device test."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "equations": len(rendered), "receipt": str(args.output)}))


if __name__ == "__main__":
    main()
