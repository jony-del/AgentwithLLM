from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


SOURCE = Path("deliverables/LangChain多Agent协作平台_0-1建设方案计划.md")
OUTPUT = Path("deliverables/LangChain多Agent协作平台_0-1建设方案计划.docx")

PAGE_WIDTH_DXA = 12240
CONTENT_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120

FONT_CN = "Microsoft YaHei"
FONT_LATIN = "Aptos"
FONT_MONO = "Consolas"

NAVY = RGBColor(11, 37, 69)
BLUE = RGBColor(46, 116, 181)
DARK_BLUE = RGBColor(31, 77, 120)
INK = RGBColor(35, 39, 44)
MUTED = RGBColor(96, 104, 116)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(31, 111, 83)
GOLD = RGBColor(122, 90, 0)

FILL_LIGHT = "F2F4F7"
FILL_BLUE = "E8EEF5"
FILL_NAVY = "0B2545"
FILL_CALLOUT = "F4F6F9"
FILL_GREEN = "E9F4EF"
FILL_GOLD = "FFF7E0"
BORDER = "C7CED8"


def set_run_font(
    run,
    *,
    name: str = FONT_CN,
    size: float | None = None,
    color: RGBColor | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
) -> None:
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), FONT_LATIN if name == FONT_CN else name)
    rfonts.set(qn("w:hAnsi"), FONT_LATIN if name == FONT_CN else name)
    rfonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = color
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_paragraph_keep(paragraph, *, together: bool = False, with_next: bool = False) -> None:
    paragraph.paragraph_format.keep_together = together
    paragraph.paragraph_format.keep_with_next = with_next


def shade_cell(cell, fill: str) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    shd = tcpr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcpr.append(shd)
    shd.set(qn("w:fill"), fill)


def shade_paragraph(paragraph, fill: str) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    shd = ppr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        ppr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top: int = 80, start: int = 120, bottom: int = 80, end: int = 120) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    mar = tcpr.first_child_found_in("w:tcMar")
    if mar is None:
        mar = OxmlElement("w:tcMar")
        tcpr.append(mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    trpr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    trpr.append(header)


def set_cant_split(row) -> None:
    trpr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    trpr.append(cant)


def set_table_geometry(table, widths_dxa: list[int], indent_dxa: int = TABLE_INDENT_DXA) -> None:
    total = sum(widths_dxa)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tblpr = table._tbl.tblPr
    tblw = tblpr.first_child_found_in("w:tblW")
    if tblw is None:
        tblw = OxmlElement("w:tblW")
        tblpr.append(tblw)
    tblw.set(qn("w:w"), str(total))
    tblw.set(qn("w:type"), "dxa")

    tblind = tblpr.first_child_found_in("w:tblInd")
    if tblind is None:
        tblind = OxmlElement("w:tblInd")
        tblpr.append(tblind)
    tblind.set(qn("w:w"), str(indent_dxa))
    tblind.set(qn("w:type"), "dxa")

    layout = tblpr.first_child_found_in("w:tblLayout")
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tblpr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            width = widths_dxa[min(index, len(widths_dxa) - 1)]
            tcpr = cell._tc.get_or_add_tcPr()
            tcw = tcpr.first_child_found_in("w:tcW")
            if tcw is None:
                tcw = OxmlElement("w:tcW")
                tcpr.append(tcw)
            tcw.set(qn("w:w"), str(width))
            tcw.set(qn("w:type"), "dxa")
            cell.width = Inches(width / 1440)


def set_table_borders(table, color: str = BORDER, size: str = "6") -> None:
    tblpr = table._tbl.tblPr
    borders = tblpr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tblpr.append(borders)
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = borders.find(qn(f"w:{name}"))
        if edge is None:
            edge = OxmlElement(f"w:{name}")
            borders.append(edge)
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), size)
        edge.set(qn("w:color"), color)


def set_paragraph_border(paragraph, *, side: str = "left", color: str = "2E74B5", size: int = 16) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    pbdr = ppr.find(qn("w:pBdr"))
    if pbdr is None:
        pbdr = OxmlElement("w:pBdr")
        ppr.append(pbdr)
    edge = OxmlElement(f"w:{side}")
    edge.set(qn("w:val"), "single")
    edge.set(qn("w:sz"), str(size))
    edge.set(qn("w:space"), "8")
    edge.set(qn("w:color"), color)
    pbdr.append(edge)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("第 ")
    set_run_font(run, size=8.5, color=MUTED)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)
    run = paragraph.add_run(" 页")
    set_run_font(run, size=8.5, color=MUTED)


def add_hyperlink(paragraph, text: str, url: str) -> None:
    part = paragraph.part
    rel_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "2E74B5")
    rpr.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.append(underline)
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), FONT_LATIN)
    fonts.set(qn("w:hAnsi"), FONT_LATIN)
    fonts.set(qn("w:eastAsia"), FONT_CN)
    rpr.append(fonts)
    run.append(rpr)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.append(text_node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


INLINE_RE = re.compile(r"(\*\*.+?\*\*|`.+?`|\[[^\]]+\]\(https?://[^)]+\))")


def add_inline(paragraph, text: str, *, base_size: float = 10.5, base_color: RGBColor = INK) -> None:
    cursor = 0
    for match in INLINE_RE.finditer(text):
        if match.start() > cursor:
            run = paragraph.add_run(text[cursor:match.start()])
            set_run_font(run, size=base_size, color=base_color)
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            set_run_font(run, size=base_size, color=base_color, bold=True)
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, name=FONT_MONO, size=max(8.5, base_size - 1), color=DARK_BLUE)
            shade_run(run, "EDF2F7")
        else:
            link = re.match(r"\[([^\]]+)\]\((https?://[^)]+)\)", token)
            assert link is not None
            add_hyperlink(paragraph, link.group(1), link.group(2))
        cursor = match.end()
    if cursor < len(text):
        run = paragraph.add_run(text[cursor:])
        set_run_font(run, size=base_size, color=base_color)


def shade_run(run, fill: str) -> None:
    rpr = run._element.get_or_add_rPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    rpr.append(shd)


def build_numbering(doc: Document) -> tuple[int, int]:
    numbering = doc.part.numbering_part.element

    def next_id(tag: str, attr: str) -> int:
        values = [int(el.get(qn(attr))) for el in numbering.findall(qn(tag)) if el.get(qn(attr))]
        return max(values, default=0) + 1

    def make(kind: str) -> int:
        abstract_id = next_id("w:abstractNum", "w:abstractNumId")
        num_id = next_id("w:num", "w:numId")
        abstract = OxmlElement("w:abstractNum")
        abstract.set(qn("w:abstractNumId"), str(abstract_id))
        multi = OxmlElement("w:multiLevelType")
        multi.set(qn("w:val"), "multilevel")
        abstract.append(multi)
        for level in range(3):
            lvl = OxmlElement("w:lvl")
            lvl.set(qn("w:ilvl"), str(level))
            start = OxmlElement("w:start")
            start.set(qn("w:val"), "1")
            lvl.append(start)
            numfmt = OxmlElement("w:numFmt")
            numfmt.set(qn("w:val"), "bullet" if kind == "bullet" else "decimal")
            lvl.append(numfmt)
            lvltext = OxmlElement("w:lvlText")
            lvltext.set(qn("w:val"), ("•" if level == 0 else "–") if kind == "bullet" else f"%{level + 1}.")
            lvl.append(lvltext)
            jc = OxmlElement("w:lvlJc")
            jc.set(qn("w:val"), "left")
            lvl.append(jc)
            ppr = OxmlElement("w:pPr")
            tabs = OxmlElement("w:tabs")
            tab = OxmlElement("w:tab")
            tab.set(qn("w:val"), "num")
            tab.set(qn("w:pos"), str(720 + level * 360))
            tabs.append(tab)
            ppr.append(tabs)
            ind = OxmlElement("w:ind")
            ind.set(qn("w:left"), str(720 + level * 360))
            ind.set(qn("w:hanging"), "360")
            ppr.append(ind)
            lvl.append(ppr)
            rpr = OxmlElement("w:rPr")
            fonts = OxmlElement("w:rFonts")
            fonts.set(qn("w:ascii"), FONT_LATIN)
            fonts.set(qn("w:hAnsi"), FONT_LATIN)
            fonts.set(qn("w:eastAsia"), FONT_CN)
            rpr.append(fonts)
            lvl.append(rpr)
            abstract.append(lvl)
        numbering.append(abstract)
        num = OxmlElement("w:num")
        num.set(qn("w:numId"), str(num_id))
        abstract_ref = OxmlElement("w:abstractNumId")
        abstract_ref.set(qn("w:val"), str(abstract_id))
        num.append(abstract_ref)
        numbering.append(num)
        return num_id

    return make("bullet"), make("decimal")


def apply_numbering(paragraph, num_id: int, level: int = 0) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    numpr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), str(level))
    numpr.append(ilvl)
    numid = OxmlElement("w:numId")
    numid.set(qn("w:val"), str(num_id))
    numpr.append(numid)
    ppr.append(numpr)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.line_spacing = 1.167


def add_callout(doc: Document, text: str, *, label: str | None = None, fill: str = FILL_CALLOUT) -> None:
    table = doc.add_table(rows=1, cols=1)
    # A one-row callout is technically a Word table. Mark its only row so
    # accessibility tooling can navigate it consistently instead of reporting an
    # unlabelled table object.
    set_repeat_table_header(table.rows[0])
    set_table_geometry(table, [CONTENT_WIDTH_DXA], indent_dxa=180)
    set_table_borders(table, color="D5DBE3", size="6")
    cell = table.cell(0, 0)
    shade_cell(cell, fill)
    set_cell_margins(cell, top=150, bottom=150, start=180, end=180)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.12
    if label:
        run = p.add_run(label)
        set_run_font(run, size=10.5, color=NAVY, bold=True)
    add_inline(p, text, base_size=10.5)
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(1)
    set_run_font(after.add_run(""), size=2)


def architecture_diagram(doc: Document) -> None:
    caption = doc.add_paragraph()
    caption.paragraph_format.space_before = Pt(2)
    caption.paragraph_format.space_after = Pt(5)
    run = caption.add_run("图 1  平台逻辑架构（Word 原生示意）")
    set_run_font(run, size=9, color=MUTED, italic=True)
    set_paragraph_keep(caption, with_next=True)

    def band(text: str, fill: str, color: RGBColor = INK, subtitle: str | None = None) -> None:
        table = doc.add_table(rows=1, cols=1)
        set_repeat_table_header(table.rows[0])
        set_table_geometry(table, [CONTENT_WIDTH_DXA], indent_dxa=160)
        set_table_borders(table, color=BORDER)
        cell = table.cell(0, 0)
        shade_cell(cell, fill)
        set_cell_margins(cell, top=120, bottom=120, start=160, end=160)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        set_run_font(r, size=10.5, color=color, bold=True)
        if subtitle:
            r = p.add_run(f"\n{subtitle}")
            set_run_font(r, size=8.8, color=color)

    def arrow() -> None:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run("↓")
        set_run_font(r, size=10, color=MUTED, bold=True)

    band("调用方与接入层", FILL_NAVY, WHITE, "Web / API · FastAPI · 鉴权 · 限流 · SSE")
    arrow()
    band("LangGraph 编排层", FILL_BLUE, NAVY, "规范化 → 复杂度分类 → Supervisor 规划 → DAG 校验 → 并行执行 → 汇总 → Reviewer → 输出")
    arrow()

    table = doc.add_table(rows=1, cols=3)
    set_repeat_table_header(table.rows[0])
    set_table_geometry(table, [3120, 3120, 3120], indent_dxa=140)
    set_table_borders(table, color=BORDER)
    for cell, title, body in zip(
        table.rows[0].cells,
        ("Research Agent", "Repository Agent", "Solution Agent"),
        ("公开资料 / 内部知识\n证据链", "代码结构 / 调用链\n工程约束", "架构 / 计划 / 制品\n草稿生成"),
    ):
        shade_cell(cell, "F8FAFC")
        set_cell_margins(cell, top=130, bottom=130, start=140, end=140)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(title)
        set_run_font(r, size=9.6, color=DARK_BLUE, bold=True)
        r = p.add_run(f"\n{body}")
        set_run_font(r, size=8.7, color=INK)
    arrow()
    band("共享平台能力", FILL_GREEN, GREEN, "工具网关 · PostgreSQL Checkpoint · Redis · 知识库 / pgvector · 制品存储 · LangSmith / OpenTelemetry")

    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(5)
    note.paragraph_format.space_after = Pt(5)
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_inline(note, "控制原则：状态外置、上下文最小化、写冲突串行、失败局部化。", base_size=8.8, base_color=MUTED)


def display_width(text: str) -> int:
    value = re.sub(r"\*\*|`|\[[^\]]+\]\([^)]+\)", "", text)
    return sum(2 if ord(char) > 127 else 1 for char in value)


def choose_widths(rows: list[list[str]]) -> list[int]:
    columns = len(rows[0])
    if columns == 1:
        return [CONTENT_WIDTH_DXA]
    weights = []
    for index in range(columns):
        lengths = [display_width(row[index]) for row in rows if index < len(row)]
        weights.append(max(8, min(36, max(lengths, default=8))))
    if columns == 2:
        weights[0] = min(weights[0], 20)
    total_weight = sum(weights)
    minimum = 900 if columns >= 4 else 1250
    widths = [max(minimum, round(CONTENT_WIDTH_DXA * weight / total_weight)) for weight in weights]
    delta = CONTENT_WIDTH_DXA - sum(widths)
    target = max(range(columns), key=lambda i: weights[i])
    widths[target] += delta
    if widths[target] < minimum:
        deficit = minimum - widths[target]
        widths[target] = minimum
        for index in sorted(range(columns), key=lambda i: widths[i], reverse=True):
            if index != target and deficit:
                take = min(deficit, max(0, widths[index] - minimum))
                widths[index] -= take
                deficit -= take
    return widths


def add_markdown_table(doc: Document, rows: list[list[str]]) -> None:
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    widths = choose_widths(rows)
    set_table_geometry(table, widths)
    set_table_borders(table)
    set_repeat_table_header(table.rows[0])
    for row_index, (word_row, data_row) in enumerate(zip(table.rows, rows)):
        set_cant_split(word_row)
        for col_index, (cell, value) in enumerate(zip(word_row.cells, data_row)):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell, top=100, bottom=100, start=120, end=120)
            if row_index == 0:
                shade_cell(cell, FILL_LIGHT)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            if col_index == 0 and len(rows[0]) > 2 and display_width(value) < 16:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(
                p,
                value.replace("<br>", "\n"),
                base_size=8.6 if len(rows[0]) >= 4 else 9.0,
                base_color=NAVY if row_index == 0 else INK,
            )
            if row_index == 0:
                for run in p.runs:
                    run.bold = True
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(2)
    set_run_font(after.add_run(""), size=2)


def configure_styles(doc: Document) -> tuple[int, int]:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    normal = doc.styles["Normal"]
    normal.font.name = FONT_CN
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT_LATIN)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_LATIN)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = INK
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10
    normal.paragraph_format.widow_control = True

    heading_tokens = {
        "Heading 1": (16, BLUE, 16, 8),
        "Heading 2": (13, BLUE, 12, 6),
        "Heading 3": (11.5, DARK_BLUE, 8, 4),
    }
    for name, (size, color, before, after) in heading_tokens.items():
        style = doc.styles[name]
        style.font.name = FONT_CN
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT_LATIN)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_LATIN)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True
        style.paragraph_format.widow_control = True

    code = doc.styles.add_style("CodeBlock", WD_STYLE_TYPE.PARAGRAPH)
    code.base_style = normal
    code.font.name = FONT_MONO
    code._element.rPr.rFonts.set(qn("w:ascii"), FONT_MONO)
    code._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_MONO)
    code._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    code.font.size = Pt(7.8)
    code.font.color.rgb = INK
    code.paragraph_format.left_indent = Inches(0.12)
    code.paragraph_format.right_indent = Inches(0.08)
    code.paragraph_format.space_before = Pt(0)
    code.paragraph_format.space_after = Pt(0)
    code.paragraph_format.line_spacing = 1.0
    code.paragraph_format.keep_together = True
    code.paragraph_format.widow_control = False

    caption = doc.styles["Caption"]
    caption.font.name = FONT_CN
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    caption.font.size = Pt(9)
    caption.font.color.rgb = MUTED
    caption.font.italic = True

    return build_numbering(doc)


def set_running_furniture(doc: Document) -> None:
    section = doc.sections[0]
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.paragraph_format.space_after = Pt(0)
    run = header.add_run("NEXUS AGENTS  |  0→1 建设方案计划  |  V1.0")
    set_run_font(run, size=8, color=MUTED, bold=True)

    footer = section.footer.paragraphs[0]
    add_page_number(footer)


def add_front_matter(doc: Document) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run("技术方案 · 立项评审")
    set_run_font(run, size=10, color=BLUE, bold=True)

    title = doc.add_paragraph()
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(5)
    run = title.add_run("基于 LangChain 的\n多 Agent 协作平台")
    set_run_font(run, size=26, color=NAVY, bold=True)

    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(16)
    run = subtitle.add_run("0→1 建设方案计划（评审稿）")
    set_run_font(run, size=15, color=DARK_BLUE, bold=True)

    metadata = [
        ("项目代号", "Nexus Agents"),
        ("文档版本", "V1.0"),
        ("编制日期", "2026-08-12"),
        ("建议周期", "8 周"),
        ("建议团队", "4～5 人"),
        ("项目性质", "独立 greenfield 项目；非现有项目延续"),
    ]
    for label, value in metadata:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run(f"{label}：")
        set_run_font(run, size=10, color=MUTED, bold=True)
        run = p.add_run(value)
        set_run_font(run, size=10, color=INK)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(5)
    run = p.add_run("评审结论建议")
    set_run_font(run, size=11, color=NAVY, bold=True)

    add_callout(
        doc,
        "批准以“技术调研与研发方案生成”为首个示范场景，采用 **LangChain 1.x + LangGraph 1.x** 从零建设独立项目。MVP 采用“确定性工作流骨架 + Supervisor 动态拆解 + 专家 Agent 并行执行 + Reviewer 质量门禁”的混合架构。当前 AgentwithLLM 项目仅作为工程经验参考，不复用代码、不承担兼容迁移；沙箱机制不纳入本期范围。",
        fill=FILL_BLUE,
    )

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run("本次评审需确认：")
    set_run_font(run, size=10, color=NAVY, bold=True)
    run = p.add_run("示范场景、模型与预算、观测部署形态、数据接入范围、人员投入。")
    set_run_font(run, size=10, color=INK)

    doc.add_page_break()

    heading = doc.add_paragraph(style="Heading 1")
    heading.paragraph_format.space_before = Pt(0)
    heading.add_run("目录与阅读指引")
    sections = [
        "1. 项目概述", "2. 建设目标与原则", "3. 项目边界", "4. 总体技术方案",
        "5. 多 Agent 协作设计", "6. 核心模块与工程结构", "7. 数据与接口设计",
        "8. 可观测性与评估体系", "9. 实施计划", "10. 团队与职责",
        "11. 测试与验收", "12. 风险与应对", "13. 需确认事项",
        "14. 后续演进路线", "15. 参考依据",
    ]
    table = doc.add_table(rows=5, cols=3)
    set_repeat_table_header(table.rows[0])
    set_table_geometry(table, [3120, 3120, 3120], indent_dxa=140)
    set_table_borders(table, color="D8DEE7")
    for index, (cell, text) in enumerate(zip([c for row in table.rows for c in row.cells], sections)):
        shade_cell(cell, "F8FAFC" if index % 2 == 0 else "FFFFFF")
        set_cell_margins(cell, top=120, bottom=120, start=140, end=140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        add_inline(p, text, base_size=9.2, base_color=DARK_BLUE)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    add_inline(p, "**快速评审路径：**先阅读第 3 节范围、第 4 节架构、第 8.3 节验收指标、第 9 节计划与第 13 节决策事项。", base_size=10.5)
    add_callout(
        doc,
        "**范围提醒：**这是一个从 0 到 1 的独立项目方案。当前代码库仅用于提炼成熟工程经验；新项目不继承其实现、接口、配置或历史包袱。",
        fill=FILL_GOLD,
    )
    doc.add_page_break()


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].lstrip().startswith("|"):
        parts = [part.strip() for part in lines[index].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", part) for part in parts):
            rows.append(parts)
        index += 1
    return rows, index


def add_code_block(doc: Document, code_lines: list[str]) -> None:
    for line in code_lines or [""]:
        p = doc.add_paragraph(style="CodeBlock")
        shade_paragraph(p, "F6F8FA")
        p.paragraph_format.left_indent = Inches(0.12)
        p.paragraph_format.right_indent = Inches(0.08)
        r = p.add_run(line or " ")
        set_run_font(r, name=FONT_MONO, size=7.8, color=INK)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def build_body(doc: Document, bullet_id: int, decimal_id: int) -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    # Body starts after the source document's horizontal rule following the executive callout.
    start = next(i for i, line in enumerate(lines) if line.strip() == "---") + 1
    index = start
    in_code = False
    code_lang = ""
    code_lines: list[str] = []
    architecture_inserted = False

    while index < len(lines):
        raw = lines[index]
        line = raw.rstrip()

        if in_code:
            if line.strip().startswith("```"):
                if code_lang == "mermaid":
                    architecture_diagram(doc)
                    architecture_inserted = True
                else:
                    add_code_block(doc, code_lines)
                in_code = False
                code_lang = ""
                code_lines = []
            else:
                code_lines.append(line)
            index += 1
            continue

        if line.strip().startswith("```"):
            in_code = True
            code_lang = line.strip()[3:].strip().lower()
            index += 1
            continue

        if not line.strip() or line.strip() == "---":
            index += 1
            continue

        if line.startswith("# ") or line.startswith("## 0→1"):
            index += 1
            continue

        if line.startswith("## "):
            text = line[3:].strip()
            # Keep major sections visually separated without forcing every one to a new page.
            if text.startswith(("4.", "8.", "9.", "15.")):
                doc.add_page_break()
            p = doc.add_paragraph(style="Heading 1")
            p.add_run(text)
            index += 1
            continue
        if line.startswith("### "):
            p = doc.add_paragraph(style="Heading 2")
            p.add_run(line[4:].strip())
            index += 1
            continue
        if line.startswith("#### "):
            p = doc.add_paragraph(style="Heading 3")
            p.add_run(line[5:].strip())
            index += 1
            continue

        if line.lstrip().startswith("|") and index + 1 < len(lines) and lines[index + 1].lstrip().startswith("|"):
            rows, index = parse_table(lines, index)
            if rows:
                add_markdown_table(doc, rows)
            continue

        if line.startswith("> "):
            text = line[2:].strip()
            if text.startswith("**无沙箱条件下的运行约束**"):
                add_callout(doc, text, fill=FILL_GOLD)
            else:
                add_callout(doc, text, fill=FILL_BLUE)
            index += 1
            continue

        bullet = re.match(r"^(\s*)-\s+(.+)$", line)
        if bullet:
            level = min(2, len(bullet.group(1)) // 2)
            p = doc.add_paragraph()
            apply_numbering(p, bullet_id, level)
            add_inline(p, bullet.group(2), base_size=10.5)
            index += 1
            continue

        numbered = re.match(r"^(\s*)\d+\.\s+(.+)$", line)
        if numbered:
            level = min(2, len(numbered.group(1)) // 2)
            p = doc.add_paragraph()
            apply_numbering(p, decimal_id, level)
            add_inline(p, numbered.group(2), base_size=10.5)
            index += 1
            continue

        # Source metadata was already rendered in the designed opening block.
        if line.startswith("**") and any(
            line.startswith(f"**{label}")
            for label in ("项目代号", "文档版本", "编制日期", "建议周期", "建议团队", "文档性质")
        ):
            index += 1
            continue

        p = doc.add_paragraph()
        add_inline(p, line, base_size=10.5)
        index += 1

    if not architecture_inserted:
        raise RuntimeError("architecture diagram was not inserted")


def add_document_metadata(doc: Document) -> None:
    doc.core_properties.title = "基于 LangChain 的多 Agent 协作平台：0→1 建设方案计划"
    doc.core_properties.subject = "独立 greenfield 多 Agent 项目技术方案与实施计划"
    doc.core_properties.author = "Nexus Agents 项目组"
    doc.core_properties.keywords = "LangChain, LangGraph, Multi-Agent, 多智能体, 项目方案"
    doc.core_properties.comments = "本方案不包含沙箱机制；当前项目仅作为工程经验参考。"


def audit_document(doc: Document) -> None:
    assert len(doc.sections) == 1
    section = doc.sections[0]
    assert section.page_width == Inches(8.5)
    assert section.page_height == Inches(11)
    assert section.left_margin == Inches(1.0)
    assert section.right_margin == Inches(1.0)
    assert len(doc.tables) >= 10
    heading_count = sum(p.style.name.startswith("Heading") for p in doc.paragraphs)
    assert heading_count >= 25
    for table in doc.tables:
        tblpr = table._tbl.tblPr
        tblw = tblpr.first_child_found_in("w:tblW")
        grid = table._tbl.tblGrid
        assert tblw is not None and tblw.get(qn("w:type")) == "dxa"
        assert grid is not None and len(grid) >= 1
        for row in table.rows:
            for cell in row.cells:
                tcw = cell._tc.get_or_add_tcPr().first_child_found_in("w:tcW")
                assert tcw is not None and tcw.get(qn("w:type")) == "dxa"


def build() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    bullet_id, decimal_id = configure_styles(doc)
    set_running_furniture(doc)
    add_front_matter(doc)
    build_body(doc, bullet_id, decimal_id)
    add_document_metadata(doc)
    audit_document(doc)
    doc.save(OUTPUT)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    build()
