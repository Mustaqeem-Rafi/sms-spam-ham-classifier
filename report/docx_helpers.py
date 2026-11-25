"""python-docx helpers for the department report format (HANDOFF §11)."""
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)
INLINE = re.compile(r"(\*\*.+?\*\*|\[FILL:[^\]]*\]|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*])|`[^`]+`|_\{[^{}]*\}|\^\{[^{}]*\})")


def _rfonts(el, name=FONT):
    rpr = el.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), name)


def style_font(style, size, bold=False, italic=False, name=FONT):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = italic
    style.font.color.rgb = BLACK
    _rfonts(style.element, name)


def set_outline_level(style, level):
    ppr = style.element.get_or_add_pPr()
    ol = OxmlElement("w:outlineLvl")
    ol.set(qn("w:val"), str(level))
    ppr.append(ol)


def add_field(paragraph, instr, placeholder="", dirty=False):
    def run_with(child):
        r = paragraph.add_run()
        r._r.append(child)
        return r
    b = OxmlElement("w:fldChar")
    b.set(qn("w:fldCharType"), "begin")
    if dirty:
        b.set(qn("w:dirty"), "true")
    run_with(b)
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    run_with(it)
    s = OxmlElement("w:fldChar")
    s.set(qn("w:fldCharType"), "separate")
    run_with(s)
    res = paragraph.add_run(placeholder)
    e = OxmlElement("w:fldChar")
    e.set(qn("w:fldCharType"), "end")
    run_with(e)
    return res


def shade_cell(cell, fill="D9D9D9"):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def repeat_header(row):
    trpr = row._tr.get_or_add_trPr()
    h = OxmlElement("w:tblHeader")
    h.set(qn("w:val"), "true")
    trpr.append(h)


def cant_split(row):
    trpr = row._tr.get_or_add_trPr()
    c = OxmlElement("w:cantSplit")
    c.set(qn("w:val"), "true")
    trpr.append(c)


def set_cell_margins(table, top=40, bottom=40, left=80, right=80):
    tblpr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(v))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblpr.append(mar)


def set_pgnum(section, fmt):
    sectpr = section._sectPr
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), "1")
    cols = sectpr.find(qn("w:cols"))
    if cols is not None:
        cols.addprevious(pg)
    else:
        sectpr.append(pg)


class ReportDoc:
    """Thin wrapper around python-docx Document with the department formatting."""

    def __init__(self):
        self.doc = Document()
        self._setup_page(self.doc.sections[0])
        self._setup_styles()
        self._settings_update_fields()

    # ------------------------------------------------------------- setup
    @staticmethod
    def _setup_page(sec):
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = sec.top_margin = sec.bottom_margin = Inches(0.8)
        sec.header_distance = sec.footer_distance = Inches(0.4)

    def _setup_styles(self):
        st = self.doc.styles
        normal = st["Normal"]
        style_font(normal, 12)
        pf = normal.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(6)
        pf.space_before = Pt(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.widow_control = True

        for name, size, before, after, align in (("Heading 1", 16, 0, 18, WD_ALIGN_PARAGRAPH.CENTER),
                                                 ("Heading 2", 14, 14, 6, WD_ALIGN_PARAGRAPH.LEFT),
                                                 ("Heading 3", 12, 10, 4, WD_ALIGN_PARAGRAPH.LEFT)):
            s = st[name]
            style_font(s, size, bold=True)
            p = s.paragraph_format
            p.space_before, p.space_after = Pt(before), Pt(after)
            p.keep_with_next = True
            p.line_spacing = 1.15
            p.alignment = align
        st["Heading 1"].paragraph_format.page_break_before = True

        from docx.enum.style import WD_STYLE_TYPE
        def para_style(name, size=12, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=6,
                       spacing=1.15, outline=None, base="Normal"):
            s = st.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            s.base_style = st[base]
            style_font(s, size, bold=bold)
            p = s.paragraph_format
            p.alignment, p.space_before, p.space_after, p.line_spacing = align, Pt(before), Pt(after), spacing
            if outline is not None:
                set_outline_level(s, outline)
            return s

        ph = para_style("Prelim Heading", 16, True, before=0, after=18, outline=0)
        ph.paragraph_format.page_break_before = True
        ph.paragraph_format.keep_with_next = True
        ph2 = para_style("Prelim Heading NoTOC", 16, True, before=0, after=18)
        ph2.paragraph_format.page_break_before = True
        para_style("Figure Caption", 11, False, before=4, after=12)
        tc = para_style("Table Caption", 11, False, before=10, after=4)
        tc.paragraph_format.keep_with_next = True
        para_style("Table Text", 10, False, align=WD_ALIGN_PARAGRAPH.LEFT, after=0, spacing=1.0)
        para_style("Cover", 12, False, after=0, spacing=1.15)
        para_style("Equation", 12, False, before=4, after=8, spacing=1.15)
        cs = para_style("Code", 9, False, align=WD_ALIGN_PARAGRAPH.LEFT, after=0, spacing=1.0)
        style_font(cs, 9, name="Courier New")
        para_style("Reference", 11, False, align=WD_ALIGN_PARAGRAPH.LEFT, after=6, spacing=1.15)
        para_style("Bullet", 12, False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=3, spacing=1.5)
        for name in ("TOC 1", "TOC 2", "TOC 3", "Table of Figures"):
            try:
                s = st[name]
            except KeyError:
                s = st.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            style_font(s, 12, bold=(name == "TOC 1"))
            s.paragraph_format.space_after = Pt(3)
            s.paragraph_format.line_spacing = 1.15

    def _settings_update_fields(self):
        settings = self.doc.settings.element
        uf = OxmlElement("w:updateFields")
        uf.set(qn("w:val"), "true")
        settings.append(uf)

    # ------------------------------------------------------------- text
    def runs(self, p, text, size=None, bold=None, italic=None):
        """Add text with inline markup: **bold**, *italic*, `code`, [FILL: ...] (yellow)."""
        for part in INLINE.split(text):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                r = p.add_run(part[2:-2]); r.bold = True
            elif part.startswith("[FILL:"):
                r = p.add_run(part); r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            elif part.startswith("`") and part.endswith("`"):
                r = p.add_run(part[1:-1]); r.font.name = "Courier New"; _rfonts(r._r, "Courier New")
                r.font.size = Pt((size or 12) - 1)
            elif part.startswith("_{") and part.endswith("}"):
                r = p.add_run(part[2:-1]); r.font.subscript = True
            elif part.startswith("^{") and part.endswith("}"):
                r = p.add_run(part[2:-1]); r.font.superscript = True
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                r = p.add_run(part[1:-1]); r.italic = True
            else:
                r = p.add_run(part)
            if size:
                r.font.size = Pt(size)
            if bold is not None and not part.startswith("**"):
                r.bold = bold if not r.bold else True
            if italic:
                r.italic = True
        return p

    def para(self, text="", style=None, align=None, size=None, bold=None, italic=None,
             space_after=None, keep_next=False):
        p = self.doc.add_paragraph(style=style)
        if text:
            self.runs(p, text, size=size, bold=bold, italic=italic)
        if align is not None:
            p.alignment = align
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        if keep_next:
            p.paragraph_format.keep_with_next = True
        return p

    def paras(self, *texts):
        for t in texts:
            self.para(t)

    def bullets(self, items, numbered=False):
        for i, it in enumerate(items, 1):
            p = self.doc.add_paragraph(style="Bullet")
            pf = p.paragraph_format
            pf.left_indent, pf.first_line_indent = Inches(0.35), Inches(-0.25)
            p.add_run(f"{i}.\t" if numbered else "•\t")
            pf.tab_stops.add_tab_stop(Inches(0.35))
            self.runs(p, it)
        return p

    def h1(self, text):
        return self.doc.add_paragraph(text.upper(), style="Heading 1")

    def h2(self, text):
        return self.doc.add_paragraph(text, style="Heading 2")

    def h3(self, text):
        return self.doc.add_paragraph(text, style="Heading 3")

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def equation(self, expr, number):
        p = self.doc.add_paragraph(style="Equation")
        p.paragraph_format.tab_stops.add_tab_stop(Inches(3.1), alignment=1)  # center
        p.paragraph_format.tab_stops.add_tab_stop(Inches(6.2), alignment=2)  # right
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.add_run("\t")
        self.runs(p, expr)
        p.add_run(f"\t({number})")
        return p

    def code(self, text):
        lines = text.strip("\n").split("\n")
        for i, line in enumerate(lines):
            p = self.doc.add_paragraph(style="Code")
            p.paragraph_format.left_indent = Inches(0.2)
            ppr = p._p.get_or_add_pPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F2F2F2")
            ppr.append(shd)
            p.paragraph_format.keep_with_next = i < len(lines) - 1
            r = p.add_run(line if line else " ")
            r.font.name = "Courier New"; _rfonts(r._r, "Courier New")
        self.doc.add_paragraph(style="Code").paragraph_format.space_after = Pt(6)

    # ------------------------------------------------------------- figures & tables
    def figure(self, path, label, caption, width=6.0):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.add_run().add_picture(str(path), width=Inches(width))
        c = self.doc.add_paragraph(style="Figure Caption")
        r = c.add_run(f"Fig. {label}: ")
        r.bold = True
        self.runs(c, caption)
        return c

    def figure_pair(self, left, right, width=3.1):
        """Two figures side by side, each with its own numbered caption. left/right = (path, label, caption)."""
        t = self.doc.add_table(rows=1, cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        self.set_widths(t, [width + 0.1, width + 0.1])
        for cell, (path, label, caption) in zip(t.rows[0].cells, (left, right)):
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(path), width=Inches(width))
            c = cell.add_paragraph(style="Figure Caption")
            r = c.add_run(f"Fig. {label}: ")
            r.bold = True
            self.runs(c, caption)
        cant_split(t.rows[0])
        return t

    def table(self, header, rows, label, caption, widths=None, size=10, align=None, bold_cells=None,
              header_size=None):
        """header: list[str]; rows: list[list]; widths in inches; align: list of 'l'/'c'/'r'."""
        c = self.doc.add_paragraph(style="Table Caption")
        r = c.add_run(f"Table {label}: ")
        r.bold = True
        self.runs(c, caption)
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_cell_margins(t)
        align = align or ["l"] + ["c"] * (len(header) - 1)
        amap = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}
        bold_cells = bold_cells or set()
        for j, h in enumerate(header):
            cell = t.rows[0].cells[j]
            cell.paragraphs[0].style = "Table Text"
            self.runs(cell.paragraphs[0], str(h), size=header_size or size, bold=True)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            shade_cell(cell)
        repeat_header(t.rows[0])
        for i, row in enumerate(rows, 1):
            cant_split(t.rows[i])
            for j, v in enumerate(row):
                cell = t.rows[i].cells[j]
                cell.paragraphs[0].style = "Table Text"
                self.runs(cell.paragraphs[0], str(v), size=size, bold=True if (i - 1, j) in bold_cells else None)
                cell.paragraphs[0].alignment = amap[align[j]]
        if widths:
            self.set_widths(t, widths)
        spacer = self.doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(4)
        spacer.paragraph_format.line_spacing = 1.0
        return t

    @staticmethod
    def set_widths(t, widths):
        """Fixed column widths that Word *and* LibreOffice honour (tblW, tblLayout, gridCol, tcW)."""
        t.autofit = False
        dxa = [int(w * 1440) for w in widths]
        tblpr = t._tbl.tblPr
        tblw = tblpr.find(qn("w:tblW"))
        if tblw is None:
            tblw = OxmlElement("w:tblW")
            tblpr.append(tblw)
        tblw.set(qn("w:w"), str(sum(dxa)))
        tblw.set(qn("w:type"), "dxa")
        layout = tblpr.find(qn("w:tblLayout"))
        if layout is None:
            layout = OxmlElement("w:tblLayout")
            tblpr.append(layout)
        layout.set(qn("w:type"), "fixed")
        grid = t._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), dxa):
            gc.set(qn("w:w"), str(w))
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Inches(w)

    # ------------------------------------------------------------- sections
    def new_section(self, pgfmt=None):
        sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        self._setup_page(sec)
        sec.footer.is_linked_to_previous = False
        sec.header.is_linked_to_previous = False
        if pgfmt:
            set_pgnum(sec, pgfmt)
            p = sec.footer.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_field(p, "PAGE", "1")
            for r in p.runs:
                r.font.name = FONT; r.font.size = Pt(11)
        return sec

    def toc(self, instr, placeholder="Right-click and choose Update Field (or press F9) to build this list."):
        p = self.doc.add_paragraph()
        add_field(p, instr, placeholder, dirty=True)
        return p

    def save(self, path):
        self.doc.save(str(path))
