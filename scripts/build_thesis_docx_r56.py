#!/usr/bin/env python3
"""Build the thesis DOCX from the repository's small Markdown report dialect.

Run with CODEX_PRIMARY_RUNTIME_PYTHON.  The first build writes a populated,
linked contents list without page numbers and a headings manifest.  Render it,
then rebuild with --page-map or --page-map-from-pdf; the latter also writes a
page map.  Render the final build again and check that its page map is stable.

Equations are editable OMML, translated from bundled KaTeX MathML.  Unsupported
MathML is an error, never silently flattened or left as raw LaTeX.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from dataclasses import dataclass

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Mm, Pt, RGBColor
from lxml import etree
from PIL import Image


FONT = "Times New Roman"
MATH_FONT = "Cambria Math"
PAGE_WIDTH_MM = 215.9
PAGE_HEIGHT_MM = 279.4
MARGIN_MM = 25.0
TEXT_WIDTH_MM = PAGE_WIDTH_MM - 2 * MARGIN_MM
FRESH_PAGE_SECTIONS = {"abstract", "table of contents", "introduction", "references", "appendices"}
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
HEADING = re.compile(r"^(#{2,6})\s*(.*?)\s*#*\s*$")
TYPED_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)(?:\.)?\s+(.+)$")
INLINE_MATH = re.compile(r"(?<!\\)\$([^$\n]+)(?<!\\)\$")
INLINE = re.compile(
    r"\*\*(.+?)\*\*|`([^`]+)`|\[([^\]]+)\]\((https?://[^\s)]+)\)"
    r"|(?<!\\)\$([^$\n]+)(?<!\\)\$|https?://[^\s<>]+"
)


@dataclass
class Block:
    kind: str
    text: str = ""
    data: object = None
    line: int = 0


def strip_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("**", "").replace("`", "").strip()


def split_cells(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [cell.strip().replace(r"\|", "|") for cell in cells]


def is_table(lines: list[str], i: int) -> bool:
    if i + 1 >= len(lines) or "|" not in lines[i]:
        return False
    cells = split_cells(lines[i + 1])
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c) for c in cells)


def parse_markdown(path: Path) -> tuple[str, list[Block], list[dict]]:
    source = path.read_text(encoding="utf-8")
    source = re.sub(
        r"<!--\s*DOCX_TOC_START\s*-->.*?<!--\s*DOCX_TOC_END\s*-->",
        "[CONTENTS]", source, flags=re.S,
    )
    source = re.sub(r"<!--.*?-->", "", source, flags=re.S)
    lines = source.splitlines()
    blocks, headings = [], []
    title = None
    counters = [0] * 5
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "[CONTENTS]":
            # The linked Word TOC is generated from actual heading paragraphs.
            i += 1
            continue
        line_no = i + 1
        if line.startswith("# "):
            if title is not None:
                raise ValueError(f"Line {line_no}: only one # cover title is supported")
            title = strip_inline(line[2:])
            blocks.append(Block("title", title, line=line_no))
            i += 1
            continue
        match = HEADING.match(line)
        if match:
            level = len(match[1]) - 1
            raw = match[2]
            explicit = TYPED_NUMBER.match(raw)
            name = strip_inline(explicit[2] if explicit else raw)
            if level > 1 and counters[level - 2] == 0:
                raise ValueError(f"Line {line_no}: heading hierarchy skips a level")
            counters[level - 1] += 1
            for k in range(level, len(counters)):
                counters[k] = 0
            number = ".".join(str(n) for n in counters[:level])
            if explicit and explicit[1] != number:
                raise ValueError(f"Line {line_no}: typed heading {explicit[1]} disagrees with Word sequence {number}")
            item = {"id": "hdg_" + number.replace(".", "_"), "number": number,
                    "level": level, "title": name, "source_line": line_no,
                    "search_text": number + " " + name}
            headings.append(item)
            blocks.append(Block("heading", name, item, line_no))
            i += 1
            continue
        if line.startswith("$$"):
            chunks = [line[2:]]
            while not chunks[-1].rstrip().endswith("$$"):
                i += 1
                if i >= len(lines):
                    raise ValueError(f"Line {line_no}: unclosed display equation")
                chunks.append(lines[i])
            chunks[-1] = chunks[-1].rstrip()[:-2]
            latex = "\n".join(chunks).strip()
            tags = re.findall(r"\\tag\{([^{}]+)\}", latex)
            if len(tags) != 1:
                raise ValueError(f"Line {line_no}: a major equation needs exactly one \\tag{{N}}")
            latex = re.sub(r"\\tag\{[^{}]+\}", "", latex).strip()
            blocks.append(Block("equation", latex, tags[0], line_no))
            i += 1
            continue
        image = re.fullmatch(r"!\[([^\]]*)\]\((.+?)\)", line)
        if image:
            blocks.append(Block("image", image[1], image[2].strip("<>"), line_no))
            i += 1
            continue
        if is_table(lines, i):
            header = split_cells(lines[i])
            align = split_cells(lines[i + 1])
            rows = [header]
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                row = split_cells(lines[i])
                if len(row) != len(header):
                    raise ValueError(f"Line {i + 1}: expected {len(header)} table columns, found {len(row)}")
                rows.append(row)
                i += 1
            blocks.append(Block("table", data={"rows": rows, "align": align}, line=line_no))
            continue
        if line.startswith("```"):
            chunks = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                chunks.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError(f"Line {line_no}: unclosed code block")
            blocks.append(Block("code", "\n".join(chunks), line=line_no))
            i += 1
            continue
        list_match = re.match(r"^(?:[-*+]\s+|(\d+)\.\s+)(.+)", line)
        if list_match:
            blocks.append(Block("list", list_match[2], bool(list_match[1]), line_no))
            i += 1
            continue
        chunks = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            nxt = lines[i].strip()
            if (HEADING.match(nxt) or nxt.startswith(("# ", "$$", "![", "```"))
                    or re.match(r"^(?:[-*+]\s+|\d+\.\s+)", nxt) or is_table(lines, i)):
                break
            chunks.append(nxt)
            i += 1
        text = " ".join(chunks)
        kind = "caption" if re.match(r"^(?:Table|Figure)\s+\d+\.", strip_inline(text)) else "paragraph"
        blocks.append(Block(kind, text, line=line_no))
    if title is None:
        raise ValueError("The report needs a # cover title")
    if sum(h["title"].casefold() == "table of contents" for h in headings) != 1:
        raise ValueError("Exactly one numbered Table of Contents heading is required")
    tags = [b.data for b in blocks if b.kind == "equation"]
    if len(tags) != len(set(tags)):
        raise ValueError("Equation tags must be unique")
    return title, blocks, headings


def math_element(tag: str, **attrs):
    node = etree.Element(f"{{{MATH_NS}}}{tag}")
    for key, value in attrs.items():
        node.set(f"{{{MATH_NS}}}{key}", str(value))
    return node


def math_run(text: str, variant: str = "normal", *, literal: bool = False):
    run = math_element("r")
    props = math_element("rPr")
    style = {"normal": "p", "italic": "i", "bold": "b", "bold-italic": "bi"}.get(variant, "p")
    # Explicit Latin bold symbols need their mathematical alphabet codepoints
    # as well as OMML styling: some Word-compatible importers otherwise drop
    # m:sty=b.  These remain editable characters inside the native equation.
    if variant == "bold" and text.isascii() and text.isalpha():
        text = "".join(chr(ord(ch) - ord("A") + 0x1D400) if ch.isupper()
                       else chr(ord(ch) - ord("a") + 0x1D41A) for ch in text)
        literal = True
    # Upright mathematics is still math.  Marking every number/operator as
    # literal text makes some Word-compatible renderers left-align fraction
    # arguments.  Reserve normal-text mode for actual words, explicit upright
    # alphabetic labels, styled Unicode letters and explicit spaces.
    if literal:
        props.append(math_element("nor", val="1"))
    if variant in ("double-struck", "script", "fraktur", "sans-serif", "monospace"):
        props.append(math_element("scr", val=variant))
    props.append(math_element("sty", val=style))
    run.append(props)
    wprops = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    for key in ("ascii", "hAnsi", "cs"):
        fonts.set(qn("w:" + key), MATH_FONT)
    wprops.append(fonts)
    for key, enabled in (("b", style in ("b", "bi")), ("i", style in ("i", "bi"))):
        flag = OxmlElement("w:" + key)
        flag.set(qn("w:val"), "1" if enabled else "0")
        wprops.append(flag)
    run.append(wprops)
    value = math_element("t")
    value.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    value.text = text
    run.append(value)
    return run


def merge_literal_math_runs(nodes: list) -> list:
    """Keep upright labels such as TP and reflected together in Word math."""
    result = []
    for node in nodes:
        literal = node.tag == f"{{{MATH_NS}}}r" and node.find(f"{{{MATH_NS}}}rPr/{{{MATH_NS}}}nor") is not None
        previous = result[-1] if result else None
        if literal and previous is not None and previous.tag == node.tag:
            same = all(etree.tostring(previous.find(name)) == etree.tostring(node.find(name))
                       for name in (f"{{{MATH_NS}}}rPr", qn("w:rPr")))
            if same:
                old_text, new_text = previous.find(f"{{{MATH_NS}}}t"), node.find(f"{{{MATH_NS}}}t")
                old_text.text = (old_text.text or "") + (new_text.text or "")
                continue
        result.append(node)
    return result


def math_sequence(children, variant=None):
    """Bind each n-ary operator to its following term; avoid empty operands."""
    result = []
    i = 0
    while i < len(children):
        converted = mathml_to_omml(children[i], variant)
        if len(converted) == 1 and etree.QName(converted[0]).localname == "nary":
            j, depth = i + 1, 0
            while j < len(children):
                child = children[j]
                base = child
                if etree.QName(child).localname in ("msub", "msup", "msubsup"):
                    base = child[0]
                text = "".join(base.itertext()) if etree.QName(base).localname == "mo" else ""
                if j > i + 1 and depth == 0 and text in ("+", "−", "=", "≤", "≥", "≈", ",", ";"):
                    break
                if text in ("(", "[", "{"):
                    depth += 1
                elif text in (")", "]", "}"):
                    if depth == 0:
                        break
                    depth -= 1
                j += 1
            if j == i + 1:
                raise ValueError("A summation or other n-ary operator has no following operand")
            converted[0].find(f"{{{MATH_NS}}}e").extend(math_sequence(children[i + 1:j], variant))
            i = j
        else:
            i += 1
        result.extend(converted)
    return merge_literal_math_runs(result)


def mathml_to_omml(node, inherited: str | None = None) -> list:
    """Strict structural mapping of common presentation MathML to Office Math."""
    tag = etree.QName(node).localname
    # KaTeX may emit an invisible function-application marker as an extra
    # sibling inside scripts such as x_{\min}; it has no Word math content.
    kids = [child for child in node
            if not (etree.QName(child).localname == "mo"
                    and "".join(child.itertext()) in ("\u2061", "\u2062", "\u2063"))]
    variant = node.get("mathvariant", inherited)
    if tag in ("annotation", "annotation-xml"):
        return []
    if tag in ("math", "semantics", "mrow", "mstyle", "mpadded"):
        return math_sequence(kids, variant)
    if tag in ("mi", "mn", "mo", "mtext", "ms"):
        text = "".join(node.itertext()).replace("\u2061", "").replace("\u2062", "").replace("\u2063", "")
        if not text:
            return []
        resolved_variant = variant or ("italic" if tag == "mi" and len(text) == 1 else "normal")
        return [math_run(text, resolved_variant,
                         literal=tag in ("mtext", "ms") or (resolved_variant == "normal" and text.isalpha()))]
    if tag == "mspace":
        match = re.fullmatch(r"(-?[\d.]+)em", node.get("width", ""))
        width = float(match[1]) if match else 1 / 6
        if width <= 0:
            return []
        if width >= 1:
            spacing = "\u2003" * round(width)
        elif width >= 0.4:
            spacing = "\u2002"
        else:
            spacing = "\u2009"
        return [math_run(spacing, literal=True)]
    spec = {
        "mfrac": ("f", ("num", "den")),
        "msub": ("sSub", ("e", "sub")),
        "msup": ("sSup", ("e", "sup")),
        "msubsup": ("sSubSup", ("e", "sub", "sup")),
        "mroot": ("rad", ("e", "deg")),
    }
    if tag in spec:
        out_tag, fields = spec[tag]
        if len(kids) != len(fields):
            raise ValueError(f"Malformed MathML {tag}")
        result = math_element(out_tag)
        children = dict(zip(fields, kids))
        if tag == "mroot":
            fields = ("deg", "e")
        for field in fields:
            part = math_element(field)
            part.extend(mathml_to_omml(children[field], variant))
            result.append(part)
        return [result]
    if tag == "msqrt":
        result = math_element("rad")
        props = math_element("radPr")
        props.append(math_element("degHide", val="1"))
        result.append(props)
        result.append(math_element("deg"))
        body = math_element("e")
        body.extend(out for child in kids for out in mathml_to_omml(child, variant))
        result.append(body)
        return [result]
    if tag in ("mover", "munder", "munderover"):
        base_text = "".join(kids[0].itertext())
        if base_text in ("∑", "∏", "∫", "⋃", "⋂", "∐"):
            result = math_element("nary")
            props = math_element("naryPr")
            props.append(math_element("chr", val=base_text))
            props.append(math_element("limLoc", val="undOvr"))
            props.append(math_element("subHide", val="1" if tag == "mover" else "0"))
            props.append(math_element("supHide", val="1" if tag == "munder" else "0"))
            result.append(props)
            sub, sup = math_element("sub"), math_element("sup")
            if tag in ("munder", "munderover"):
                sub.extend(mathml_to_omml(kids[1], variant))
            if tag in ("mover", "munderover"):
                sup.extend(mathml_to_omml(kids[-1], variant))
            result.extend((sub, sup, math_element("e")))
            return [result]
        if tag == "mover" and node.get("accent") == "true":
            accent = "".join(kids[1].itertext())
            is_bar = accent in ("¯", "‾", "ˉ", "̄")
            result = math_element("bar" if is_bar else "acc")
            props = math_element("barPr" if is_bar else "accPr")
            props.append(math_element("pos", val="top") if is_bar else math_element("chr", val=accent))
            result.append(props)
            body = math_element("e")
            body.extend(mathml_to_omml(kids[0], variant))
            result.append(body)
            return [result]
        base = mathml_to_omml(kids[0], variant)
        limits = [("limLow", kids[1])] if tag == "munder" else [("limUpp", kids[1])]
        if tag == "munderover":
            limits = [("limLow", kids[1]), ("limUpp", kids[2])]
        for out_tag, limit in limits:
            result = math_element(out_tag)
            body, lim = math_element("e"), math_element("lim")
            body.extend(base)
            lim.extend(mathml_to_omml(limit, variant))
            result.extend((body, lim))
            base = [result]
        return base
    if tag == "mtable":
        result = math_element("m")
        for row in kids:
            if etree.QName(row).localname != "mtr":
                raise ValueError("Only ordinary MathML matrix rows are supported")
            out_row = math_element("mr")
            for cell in row:
                out_cell = math_element("e")
                out_cell.extend(out for child in cell for out in mathml_to_omml(child, variant))
                out_row.append(out_cell)
            result.append(out_row)
        return [result]
    if tag == "mfenced":
        result, props = math_element("d"), math_element("dPr")
        props.extend((math_element("begChr", val=node.get("open", "(")),
                      math_element("endChr", val=node.get("close", ")"))))
        result.append(props)
        for child in kids:
            part = math_element("e")
            part.extend(mathml_to_omml(child, variant))
            result.append(part)
        return [result]
    raise ValueError(f"Unsupported MathML element {tag}; simplify the equation or extend the tested mapper")


def compile_equations(blocks: list[Block]) -> dict[str, list]:
    latex = set()
    for block in blocks:
        if block.kind == "equation":
            latex.add(block.text)
        elif block.kind == "table":
            for row in block.data["rows"]:
                for cell in row:
                    latex.update(INLINE_MATH.findall(cell))
        else:
            latex.update(INLINE_MATH.findall(block.text))
    if not latex:
        return {}
    runtime = os.environ.get("CODEX_PRIMARY_RUNTIME_NODE")
    modules = os.environ.get("CODEX_PRIMARY_RUNTIME_NODE_MODULES")
    if not runtime or not modules:
        raise RuntimeError("Equation conversion requires CODEX_PRIMARY_RUNTIME_NODE and CODEX_PRIMARY_RUNTIME_NODE_MODULES")
    matches = sorted((Path(modules) / "@oai/artifact-tool/dist/chunks").glob("katex-*.mjs"))
    if not matches:
        raise RuntimeError("Bundled KaTeX was not found; do not substitute an unpinned global installation")
    ordered = sorted(latex)
    js = """import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const katex = (await import(pathToFileURL(process.argv[1]).href)).default;
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(input.map(text => katex.renderToString(text,
 {displayMode:true, output:'mathml', throwOnError:true, strict:'error', trust:false}))));
"""
    completed = subprocess.run([runtime, "--input-type=module", "-e", js, str(matches[0])],
                               input=json.dumps(ordered), text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError("KaTeX equation conversion failed:\n" + completed.stderr)
    markup = json.loads(completed.stdout)
    result = {}
    for source, value in zip(ordered, markup):
        root = etree.fromstring(value.encode())
        math = root if etree.QName(root).localname == "math" else root.find(".//{http://www.w3.org/1998/Math/MathML}math")
        if math is None:
            raise ValueError(f"No MathML returned for {source!r}")
        result[source] = mathml_to_omml(math)
    return result


def set_font(style, size: float, *, bold=False):
    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    props = style.element.get_or_add_rPr()
    fonts = props.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        props.append(fonts)
    for attr in ("ascii", "hAnsi", "cs", "eastAsia"):
        fonts.set(qn("w:" + attr), FONT)
    for attr in ("asciiTheme", "hAnsiTheme", "csTheme", "eastAsiaTheme"):
        fonts.attrib.pop(qn("w:" + attr), None)


def setup_styles(doc):
    normal = doc.styles["Normal"]
    set_font(normal, 12)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.widow_control = True
    title = doc.styles["Title"]
    set_font(title, 20, bold=True)
    title.paragraph_format.space_before = Mm(62)
    title.paragraph_format.space_after = Pt(18)
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.keep_with_next = True
    for level in range(1, 6):
        style = doc.styles[f"Heading {level}"]
        set_font(style, 15 if level == 1 else 12, bold=True)
        style.paragraph_format.space_before = Pt(0 if level == 1 else 13)
        style.paragraph_format.space_after = Pt(9 if level == 1 else 6)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True
        style.paragraph_format.page_break_before = False
        style.paragraph_format.line_spacing = 1.08
    for name, size in (("Thesis TOC 1", 11), ("Thesis TOC 2", 11), ("Thesis TOC 3", 11),
                       ("Thesis TOC 4", 11), ("Thesis TOC 5", 11), ("Bibliography", 10.5),
                       ("Equation", 12), ("Code Block", 10.5)):
        if name not in doc.styles:
            doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style = doc.styles[name]
        style.base_style = normal
        set_font(style, size)
        style.paragraph_format.space_after = Pt(1.8 if "TOC" in name else 6)
        style.paragraph_format.line_spacing = 1.05 if "TOC" in name else 1.1
        if "TOC" in name:
            level = int(name[-1])
            style.paragraph_format.left_indent = Mm((level - 1) * 6)
            style.paragraph_format.keep_with_next = False
            style.paragraph_format.tab_stops.add_tab_stop(Mm(TEXT_WIDTH_MM), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        if name == "Bibliography":
            style.paragraph_format.left_indent = Mm(8)
            style.paragraph_format.first_line_indent = Mm(-8)
            style.paragraph_format.space_after = Pt(3)
            style.paragraph_format.keep_together = True
        if name == "Code Block":
            style.font.name = "Courier New"
            style.paragraph_format.line_spacing = 1
    caption = doc.styles["Caption"]
    set_font(caption, 10.5)
    caption.font.italic = False
    caption.paragraph_format.space_before = Pt(5)
    caption.paragraph_format.space_after = Pt(8)
    caption.paragraph_format.line_spacing = 1.05
    caption.paragraph_format.keep_together = True
    for name in ("List Bullet", "List Number", "Footer"):
        set_font(doc.styles[name], 10.5 if name == "Footer" else 12)
        if name != "Footer":
            doc.styles[name].paragraph_format.left_indent = Mm(6)
            doc.styles[name].paragraph_format.first_line_indent = Mm(-6)
            doc.styles[name].paragraph_format.tab_stops.add_tab_stop(Mm(6))
    for style in doc.styles:
        if style.type == WD_STYLE_TYPE.PARAGRAPH:
            props = style.element.find(qn("w:pPr"))
            if props is not None:
                for border in list(props.findall(qn("w:pBdr"))):
                    props.remove(border)


def create_heading_numbering(doc) -> int:
    root = doc.part.numbering_part.element
    abstract_id = max([int(n.get(qn("w:abstractNumId"))) for n in root.findall(qn("w:abstractNum"))] + [-1]) + 1
    num_id = max([int(n.get(qn("w:numId"))) for n in root.findall(qn("w:num"))] + [0]) + 1
    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "multilevel")
    abstract.append(multi)
    for level in range(5):
        lvl = OxmlElement("w:lvl")
        lvl.set(qn("w:ilvl"), str(level))
        for name, value in (("start", "1"), ("numFmt", "decimal"),
                            ("pStyle", f"Heading{level + 1}"),
                            ("lvlText", ".".join(f"%{j + 1}" for j in range(level + 1))),
                            ("lvlJc", "left"), ("suff", "space")):
            child = OxmlElement("w:" + name)
            child.set(qn("w:val"), value)
            lvl.append(child)
        if level:
            restart = OxmlElement("w:lvlRestart")
            restart.set(qn("w:val"), str(level))
            lvl.append(restart)
        props = OxmlElement("w:pPr")
        indent = OxmlElement("w:ind")
        indent.set(qn("w:left"), "0")
        indent.set(qn("w:hanging"), "0")
        props.append(indent)
        lvl.append(props)
        abstract.append(lvl)
    root.append(abstract)
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    pointer = OxmlElement("w:abstractNumId")
    pointer.set(qn("w:val"), str(abstract_id))
    num.append(pointer)
    root.append(num)
    for level in range(1, 6):
        props = doc.styles[f"Heading {level}"].element.get_or_add_pPr()
        num_props = OxmlElement("w:numPr")
        for name, value in (("ilvl", level - 1), ("numId", num_id)):
            child = OxmlElement("w:" + name)
            child.set(qn("w:val"), str(value))
            num_props.append(child)
        props.append(num_props)
    return num_id


def bookmark(paragraph, name: str, index: int):
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(index))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(index))
    paragraph._p.insert(1 if paragraph._p.find(qn("w:pPr")) is not None else 0, start)
    paragraph._p.append(end)


def breakable_url(value: str) -> str:
    return re.sub(r"([/._?=&%#-])", lambda match: match[0] + "\u200b", value)


def hyperlink(paragraph, text: str, *, url=None, anchor=None, bold=False):
    link = OxmlElement("w:hyperlink")
    if url:
        link.set(qn("r:id"), paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True))
    if anchor:
        link.set(qn("w:anchor"), anchor)
    link.set(qn("w:history"), "1")
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "000000")
    props.append(color)
    if bold:
        props.append(OxmlElement("w:b"))
    run.append(props)
    value = OxmlElement("w:t")
    value.set(qn("xml:space"), "preserve")
    value.text = text
    run.append(value)
    link.append(run)
    paragraph._p.append(link)


def add_math(paragraph, source: str, equations: dict):
    result = math_element("oMath")
    result.extend(etree.fromstring(etree.tostring(item)) for item in equations[source])
    paragraph._p.append(result)


def add_numbered_equation(doc, source: str, number: str, equations: dict):
    """Use genuine centered display math with a separate right-hand number.

    Equal empty/number side cells leave the formula centered on the complete
    text area.  The borderless layout row has no fixed height, so fractions,
    roots and scripts can expand naturally without clipping.  This is a layout
    container, not a numbered data table.
    """
    section = doc.sections[-1]
    usable_width = section.page_width - section.left_margin - section.right_margin
    side_width = Mm(10)
    table = doc.add_table(rows=1, cols=3)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_props = table._tbl.tblPr
    width = table_props.find(qn("w:tblW"))
    width.set(qn("w:type"), "dxa")
    width.set(qn("w:w"), str(round(usable_width / 635)))
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "bottom", "left", "right", "insideH", "insideV"):
        edge = OxmlElement("w:" + side)
        edge.set(qn("w:val"), "nil")
        borders.append(edge)
    table_props.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for side in ("top", "bottom", "left", "right"):
        margin = OxmlElement("w:" + side)
        margin.set(qn("w:w"), "0")
        margin.set(qn("w:type"), "dxa")
        margins.append(margin)
    table_props.append(margins)
    caption = OxmlElement("w:tblCaption")
    caption.set(qn("w:val"), f"Equation {number} layout")
    table_props.append(caption)
    for column, cell, cell_width in zip(table.columns, table.rows[0].cells,
                                        (side_width, usable_width - 2 * side_width, side_width)):
        column.width = cell_width
        cell.width = cell_width
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.paragraphs[0].paragraph_format.space_before = Pt(0)
        cell.paragraphs[0].paragraph_format.space_after = Pt(0)
        cell.paragraphs[0].paragraph_format.line_spacing = 1
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    paragraph = table.cell(0, 1).paragraphs[0]
    paragraph.style = "Equation"
    formatting = paragraph.paragraph_format
    formatting.alignment = WD_ALIGN_PARAGRAPH.CENTER
    formatting.left_indent = formatting.right_indent = formatting.first_line_indent = Pt(0)
    formatting.keep_together = True
    formatting.line_spacing = 1  # automatic height; never an exact point value
    formatting.space_before = Pt(8)
    formatting.space_after = Pt(12)
    formatting.tab_stops.clear_all()
    display = math_element("oMathPara")
    properties = math_element("oMathParaPr")
    properties.append(math_element("jc", val="center"))
    display.append(properties)
    formula = math_element("oMath")
    formula.extend(etree.fromstring(etree.tostring(item)) for item in equations[source])
    display.append(formula)
    paragraph._p.append(display)
    label = table.cell(0, 2).paragraphs[0]
    label.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    label.add_run(f"({number})")
    return paragraph


def inline(paragraph, text: str, equations: dict, bold=False):
    cursor = 0
    for match in INLINE.finditer(text):
        if match.start() > cursor:
            run = paragraph.add_run(text[cursor:match.start()].replace(r"\$", "$"))
            run.bold = bold
        if match[1] is not None:
            inline(paragraph, match[1], equations, bold=True)
        elif match[2] is not None:
            run = paragraph.add_run(match[2])
            run.font.name = "Courier New"
            run.font.size = Pt(10.5)
            run.bold = bold
        elif match[3] is not None:
            hyperlink(paragraph, breakable_url(match[3]) if match[3].startswith("http") else match[3], url=match[4], bold=bold)
        elif match[5] is not None:
            add_math(paragraph, match[5], equations)
        else:
            url = match[0].rstrip(".,;:")
            hyperlink(paragraph, breakable_url(url), url=url, bold=bold)
            paragraph.add_run(match[0][len(url):])
        cursor = match.end()
    if cursor < len(text):
        run = paragraph.add_run(text[cursor:].replace(r"\$", "$"))
        run.bold = bold


def table_column_widths(rows: list[list[str]]) -> list[float]:
    """Allocate the fixed page width while preserving unbreakable words."""
    import fitz
    columns = list(zip(*rows))
    weights, minimums = [], []
    fonts = (fitz.Font("tibo"), fitz.Font("tiro"))
    # 100 twips of padding per side, plus 0.8 mm for metric/font rounding.
    horizontal_padding_mm = 200 / 1440 * 25.4 + 0.8
    for col in columns:
        lengths = sorted(len(strip_inline(cell)) for cell in col)
        weights.append(max(8.0, min(42.0, lengths[len(lengths) // 2] * 0.7 + max(lengths) * 0.3)))
        word_width = 0.0
        for row_index, text in enumerate(col):
            font = fonts[0 if row_index == 0 else 1]
            words = re.split(r"[\s/\\\u200b–—-]+", strip_inline(text))
            for word in words:
                word_width = max(word_width, font.text_length(word, fontsize=10.5) / 72 * 25.4)
        minimums.append(max(12.0, word_width + horizontal_padding_mm))
    if sum(minimums) > TEXT_WIDTH_MM:
        raise ValueError("A table's unbreakable words exceed the text width; revise its columns or insert explicit break opportunities")
    widths = [0.0] * len(columns)
    remaining = set(range(len(columns)))
    available = TEXT_WIDTH_MM
    while remaining:
        weight_total = sum(weights[i] for i in remaining)
        proposed = {i: available * weights[i] / weight_total for i in remaining}
        constrained = [i for i in remaining if proposed[i] < minimums[i]]
        if not constrained:
            for i in remaining:
                widths[i] = proposed[i]
            break
        for i in constrained:
            widths[i] = minimums[i]
            available -= widths[i]
            remaining.remove(i)
    return widths


def add_table(doc, data: dict, equations: dict):
    rows, alignment = data["rows"], data["align"]
    columns = list(zip(*rows))
    widths = table_column_widths(rows)
    table = doc.add_table(rows=1, cols=len(columns))
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement("w:" + side)
        for attr, val in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            border.set(qn("w:" + attr), val)
        borders.append(border)
    table_pr.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for side, value in (("top", "85"), ("bottom", "85"), ("left", "100"), ("right", "100")):
        node = OxmlElement("w:" + side)
        node.set(qn("w:w"), value)
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    table_pr.append(margins)
    for j, width in enumerate(widths):
        table.columns[j].width = Mm(width)
    for row_index, values in enumerate(rows):
        row = table.rows[0] if row_index == 0 else table.add_row()
        props = row._tr.get_or_add_trPr()
        props.append(OxmlElement("w:cantSplit"))
        if row_index == 0:
            props.append(OxmlElement("w:tblHeader"))
        for j, (cell, text) in enumerate(zip(row.cells, values)):
            cell.width = Mm(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(1)
            paragraph.paragraph_format.space_before = Pt(1)
            paragraph.paragraph_format.line_spacing = 1.05
            # Keep compact result tables on one page. Longer tables retain a
            # repeated header and do not leave a lone final data row.
            paragraph.paragraph_format.keep_with_next = (
                row_index < len(rows) - 1 if len(rows) <= 8
                else row_index == 0 or row_index == len(rows) - 2
            )
            align = alignment[j]
            numeric = bool(re.fullmatch(r"[\d\s.,+−–/%()→-]+", strip_inline(text)))
            if (align.startswith(":") and align.endswith(":")) or numeric:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif align.endswith(":"):
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            inline(paragraph, text, equations, bold=(row_index == 0))
            for run in paragraph.runs:
                run.font.size = Pt(10.5)
            if row_index == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "DCEAF5")
                cell._tc.get_or_add_tcPr().append(shade)
    return table


def add_image(doc, source: Path, alt: str):
    if not source.is_file():
        raise FileNotFoundError(f"Figure not found: {source}")
    with Image.open(source) as img:
        width, height = img.size
    physical_width = min(TEXT_WIDTH_MM, 150.0 * width / height)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.keep_together = True
    picture = paragraph.add_run().add_picture(str(source), width=Mm(physical_width))
    picture._inline.docPr.set("descr", alt or source.stem)


def load_page_map(path: Path, headings: list[dict]) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "pages" in data:
        data = data["pages"]
    if isinstance(data, dict) and "headings" in data:
        data = {item["id"]: item["page"] for item in data["headings"]}
    result = {}
    for heading in headings:
        page = next((data[key] for key in (heading["id"], heading["number"], heading["search_text"], heading["title"]) if key in data), None)
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise ValueError(f"Page map lacks a positive integer page for {heading['search_text']}")
        result[heading["id"]] = page
    return result


def pages_from_pdf(path: Path, headings: list[dict]) -> dict[str, int]:
    """Find real bold heading text, excluding the smaller regular-font TOC."""
    import fitz
    def normalize(value):
        return re.sub(r"[^\w]+", "", value, flags=re.UNICODE).casefold()
    wanted = {normalize(h["search_text"]): h for h in headings}
    titles = {normalize(h["title"]): h for h in headings}
    candidates = {h["id"]: [] for h in headings}
    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                spans = [s for line in block["lines"] for s in line["spans"]]
                visible = [s for s in spans if s["text"].strip()]
                if not visible or max(s["size"] for s in visible) < 11.8:
                    continue
                if not all((s["flags"] & 16) or "bold" in s["font"].lower() for s in visible):
                    continue
                text = normalize(" ".join(s["text"] for s in visible))
                heading = wanted.get(text) or titles.get(text)
                if heading:
                    candidates[heading["id"]].append(page_index + 1)
    bad = {key: values for key, values in candidates.items() if len(values) != 1}
    if bad:
        raise ValueError("PDF heading detection was incomplete or ambiguous; supply --page-map manually: " + json.dumps(bad))
    return {key: values[0] for key, values in candidates.items()}


def build(args):
    source, target = args.input.resolve(), args.output.resolve()
    title, blocks, headings = parse_markdown(source)
    page_map = None
    if args.page_map:
        page_map = load_page_map(args.page_map, headings)
    elif args.page_map_from_pdf:
        page_map = pages_from_pdf(args.page_map_from_pdf, headings)
        page_map_path = args.page_map_output or target.with_suffix(".pages.json")
        page_map_path.parent.mkdir(parents=True, exist_ok=True)
        page_map_path.write_text(json.dumps({"pdf": str(args.page_map_from_pdf.resolve()), "pages": page_map}, indent=2) + "\n", encoding="utf-8")
    elif args.require_page_map:
        raise ValueError("A final build requires --page-map or --page-map-from-pdf")
    equations = compile_equations(blocks)
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Mm(PAGE_WIDTH_MM), Mm(PAGE_HEIGHT_MM)
    section.top_margin = section.bottom_margin = section.left_margin = section.right_margin = Mm(MARGIN_MM)
    section.header_distance = section.footer_distance = Mm(12)
    section.different_first_page_header_footer = True
    setup_styles(doc)
    create_heading_numbering(doc)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    doc.core_properties.title = title
    doc.core_properties.subject = "Cassette front recognition and two-corner localization"
    doc.core_properties.author = ""
    doc.core_properties.keywords = "cassette recognition; pose estimation; thesis"
    doc.core_properties.comments = ""
    bibliography, cover, previous_kind = False, True, None
    appendices, appendix_started = False, False
    for block in blocks:
        if block.kind == "title":
            doc.add_paragraph(block.text, style="Title")
        elif block.kind == "heading":
            cover = False
            h = block.data
            if h["level"] == 1:
                bibliography = h["title"].casefold() == "references"
                appendices = h["title"].casefold() == "appendices"
                appendix_started = False
            p = doc.add_paragraph(h["title"], style=f"Heading {h['level']}")
            if h["level"] == 1:
                # Keep front/back matter distinct while the concise main body
                # flows naturally instead of leaving short continuation pages.
                p.paragraph_format.page_break_before = h["title"].casefold() in FRESH_PAGE_SECTIONS
                if not p.paragraph_format.page_break_before:
                    p.paragraph_format.space_before = Pt(18)
            if appendices and h["level"] == 2:
                # Give each later appendix its own page instead of an isolated tail.
                if appendix_started:
                    p.paragraph_format.page_break_before = True
                appendix_started = True
            bookmark(p, h["id"], headings.index(h) + 1)
            if h["title"].casefold() == "table of contents":
                for item in headings:
                    entry = doc.add_paragraph(style=f"Thesis TOC {item['level']}")
                    hyperlink(entry, item["search_text"], anchor=item["id"])
                    entry.add_run("\t" + (str(page_map[item["id"]]) if page_map else ""))
        elif block.kind in ("paragraph", "caption", "list", "code"):
            style = "Bibliography" if bibliography else "Normal"
            if block.kind == "caption":
                style = "Caption"
            elif block.kind == "list":
                style = "List Number" if block.data else "List Bullet"
            elif block.kind == "code":
                style = "Code Block"
            paragraph = doc.add_paragraph(style=style)
            if previous_kind == "heading":
                paragraph.paragraph_format.keep_together = True
            if previous_kind == "table":
                paragraph.paragraph_format.space_before = Pt(6)
            if cover:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if block.kind == "caption":
                paragraph.paragraph_format.keep_with_next = strip_inline(block.text).startswith("Table ")
            if block.kind == "code":
                paragraph.add_run(block.text)
            else:
                inline(paragraph, block.text, equations)
        elif block.kind == "table":
            add_table(doc, block.data, equations)
        elif block.kind == "image":
            add_image(doc, source.parent / str(block.data), block.text)
        elif block.kind == "equation":
            add_numbered_equation(doc, block.text, block.data, equations)
        previous_kind = block.kind
    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(target)
    manifest_path = args.headings_manifest or target.with_suffix(".headings.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "docx": str(target), "title": title, "toc_has_page_numbers": page_map is not None,
                "page_numbering": "Physical pages counted from 1; cover footer hidden; subsequent page numbers equal PDF page indices.",
                "section_layout": "Flowing main body; Abstract, Table of Contents, Introduction, References and Appendices start fresh pages.",
                "equation_representation": "Editable native Office Math (OMML), translated from bundled KaTeX MathML",
                "equation_alignment": "Native centered display OMML inside borderless automatic-height rows with equal 10 mm side cells; the whole formula is centered on the text area and its number is separately right-aligned. No equation tabs or fixed row heights.",
                "headings": [dict(h, **({"page": page_map[h['id']]} if page_map else {})) for h in headings]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"docx": str(target), "headings_manifest": str(manifest_path),
                      "headings": len(headings), "equations": sum(b.kind == "equation" for b in blocks),
                      "toc_has_page_numbers": page_map is not None}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Markdown manuscript")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output DOCX")
    parser.add_argument("--headings-manifest", type=Path)
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--page-map", type=Path, help="JSON mapping heading ID (or number/title) to physical page")
    maps.add_argument("--page-map-from-pdf", type=Path, help="Read bold heading locations from a previous render")
    parser.add_argument("--page-map-output", type=Path, help="Where to save an extracted page map")
    parser.add_argument("--require-page-map", action="store_true", help="Refuse an unpaginated draft TOC")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
