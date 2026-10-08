"""Build a clean general-academic Word manuscript (author version) from the LaTeX source.

Improves on make_docx.py, whose pandoc defaults produced blue Calibri headings, low-resolution (100 dpi)
figures, inline math typeset as Word equations in a different font (with a corrupted rendering of the
"+" sign), unnumbered display equations and loosely formatted tables.

What this script does
- Reuses make_docx.py to flatten the LaTeX, expand the number macros and resolve every cross-reference
  from main_author.aux, so all text, numbers, tables, captions and references match the compiled PDF.
- Rewrites simple inline math as ordinary text (italic variables, Unicode symbols, real sub/superscripts);
  the two display equations stay editable Word equations and get right-aligned numbers (1), (2).
- Rasterises every figure from its vector PDF at 300 dpi; wide figures span the text width, the two
  single-column figures are 3.4 in wide.
- Applies one typography system (Times New Roman; 11 pt body, 16/13/11 pt headings, 10 pt captions,
  9 pt tables, 9 pt code), black headings, three-rule tables, A4 with 2.5 cm margins and page numbers.

    python -m src.paper.make_docx_manuscript            # -> paper/export/rag_embed_transfer_manuscript.docx
"""
from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path

from src.paper import make_docx as md

LATEX, OUT, FIGS = md.LATEX, md.OUT, Path("paper/figures")
NARROW = {"F8_topk_ablation", "F9_hit_vs_em"}
FONT = "Times New Roman"

# ---------------------------------------------------------------- inline math -> text
SYM = {r"\times": "×", r"\leq": "≤", r"\geq": "≥", r"\in": "∈", r"\subset": "⊂", r"\mid": "|", r"\sim": "∼",
       r"\Delta": "Δ", r"\kappa": "κ", r"\rho": "ρ", r"\alpha": "α", r"\cdot": "·", r"\{": "\\{", r"\}": "\\}"}


def _group(s: str, i: int) -> tuple[str, int]:
    """s[i] starts an argument: {...} or a single char. Returns (content, next index)."""
    if s[i] == "{":
        c, j = md.balanced(s, i)
        return c, j
    if s[i] == "\\":
        mm = re.match(r"\\([A-Za-z]+|.)", s[i:])
        j = i + len(mm.group(0))
        if mm.group(1) in ("mathrm", "text", "mathit") and j < len(s) and s[j] == "{":
            _, j = md.balanced(s, j)
        return s[i:j], j
    return s[i], i + 1


def math_to_text(m: str, top: bool = True) -> str:
    """Translate one inline LaTeX math expression to pandoc-readable LaTeX text."""
    out, i = [], 0
    while i < len(m):
        ch = m[i]
        if ch == "\\":
            mm = re.match(r"\\([A-Za-z]+|.)", m[i:])
            if mm is None:
                out.append("\\"); i += 1
                continue
            name = "\\" + mm.group(1)
            i += len(mm.group(0))
            if name in SYM:
                out.append(SYM[name])
            elif name in (r"\mathrm", r"\text", r"\ensuremath", r"\mathit"):
                c, i = _group(m, i)
                out.append(c if name != r"\ensuremath" else math_to_text(c))
            elif name in (r"\,", r"\ ", r"\;"):
                out.append(" ")
            else:
                out.append(name)
            continue
        if ch in "^_":
            c, i = _group(m, i + 1)
            tag = "textsuperscript" if ch == "^" else "textsubscript"
            out.append(f"\\{tag}{{{math_to_text(c)}}}")
            continue
        if ch == "{":
            c, i = md.balanced(m, i)
            out.append(math_to_text(c, top=False))
            continue
        if ch == "-":
            out.append("−")
        elif ch == "=":
            out.append(" = ")
        elif ch in "<>":
            out.append(f" {ch} ")
        elif ch.isalpha():
            # a maximal run of letters: a single letter is a variable (italic); words stay roman
            mm = re.match(r"[A-Za-z]+", m[i:])
            w = mm.group(0)
            out.append(f"\\emph{{{w}}}" if len(w) == 1 else w)
            i += len(w)
            continue
        else:
            out.append(ch)
        i += 1
    s = "".join(out)
    s = re.sub(r" {2,}", " ", s)
    s = s.replace("( ", "(").replace(" )", ")")
    return s.strip() if top else s


def convert_inline_math(body: str) -> str:
    # protect display equations
    eqs = []

    def keep(mo):
        eqs.append(mo.group(0))
        return f"@@EQ{len(eqs) - 1}@@"
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", keep, body, flags=re.S)
    body = re.sub(r"(?<!\\)\$([^$]+?)(?<!\\)\$", lambda mo: math_to_text(mo.group(1)), body)
    # \ensuremath used outside math (e.g. signed differences)
    body = md.expand_macro(body, "ensuremath", 1, lambda c: math_to_text(c))
    for n, e in enumerate(eqs):
        body = body.replace(f"@@EQ{n}@@", e)
    return body


# ---------------------------------------------------------------- figures
def rasterise_figures(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(FIGS.glob("F*.pdf")):
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(pdf), str(dst / pdf.stem)], check=True)


# ---------------------------------------------------------------- build
def build_body() -> tuple[str, str]:
    nums = json.load(open(LATEX / "numbers_index.json"))
    lab = md.labels()
    src = (LATEX / "main.tex").read_text()
    body = src[src.index("\\begin{document}") + len("\\begin{document}"):src.index("\\end{document}")]
    body = md.read_inputs(body)
    for m in re.finditer(r"\\newcommand\{\\(fig[a-z]+)\}\{(\\fig\{.*)\}\n", src):
        body = body.replace("\\" + m.group(1) + "\n", m.group(2) + "\n")
    body = body.replace("\\ifdefined\\authorversion", "").replace("\\fi\n", "\n")
    body = re.sub(r"\\bibliography\{[^}]*\}", "", body)
    body = re.sub(r"\\bibliographystyle\{[^}]*\}", "", body)
    body = body.replace("\\maketitle", "").replace("\\appendix", "")
    body = re.sub(r"\\(setcounter|renewcommand)\{[^}]*\}\{[^}]*\}", "", body)
    body = body.replace("\\FloatBarrier", "")
    body = md.expand_macro(body, "NUM", 1, lambda k: nums["NUM"][k])
    body = md.expand_macro(body, "CI", 1, lambda k: nums["CI"][k])

    rast = OUT / "figs300"
    rasterise_figures(rast)

    def fig(env, file, cap, label):
        n = lab.get(label, "?")
        width = "3.4in" if file in NARROW else "6.3in"
        return (f"\n\\begin{{figure}}\n\\centering\n\\includegraphics[width={width}]{{{(rast / file).resolve()}.png}}\n"
                f"\\caption{{Figure {n}: {cap}}}\n\\end{{figure}}\n")
    body = md.expand_macro(body, "fig", 4, fig)

    def tabcap(m):
        cap, label = m.group(1), m.group(2)
        return f"\\caption{{Table {lab.get(label, '?')}: {cap}}}\\label{{{label}}}"
    body = re.sub(r"\\caption\{((?:(?!\\caption\{).)*?)\}\n\\label\{(tab:[^}]*)\}", tabcap, body, flags=re.S)
    body = re.sub(r"\\begin\{adjustbox\}\{[^}]*\}|\\end\{adjustbox\}", "", body)
    body = re.sub(r"\\(scriptsize|footnotesize|small)\b", "", body)
    body = re.sub(r"\{\s*(\[[^\]]*\])\}", r"\1", body)
    body = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", body)
    body = body.replace("table*", "table").replace("figure*", "figure")
    body = re.sub(r"\\S~?\\ref\{([^}]*)\}", lambda m: "§" + lab.get(m.group(1), "?"), body)
    body = re.sub(r"\\ref\{([^}]*)\}", lambda m: lab.get(m.group(1), "?"), body)
    body = re.sub(r"\\hyperref\[[^\]]*\]\{([^}]*)\}", r"\1", body)
    body = body.replace("\\citealp{", "\\citet{")
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = body.replace("\\begin{abstract}", "\\section*{Abstract}").replace("\\end{abstract}", "")
    body = convert_inline_math(body)
    body = body.replace("\\mathrm{Hit@5}", "\\mathrm{Hit}\\text{@}5")
    # keep confidence intervals on one line: [a, b] -> [a,~b]
    body = re.sub(r"\[(−?[\d.]+), (−?[\d.]+)\]", r"[\1,~\2]", body)

    sec = sub = 0
    app = False
    lines = []
    for line in body.splitlines():
        if line.startswith("\\section{") and "Reader Prompts" in line:
            app, sec = True, 0
        m = re.match(r"\\section\{(.*)\}", line)
        if m:
            sec, sub = sec + 1, 0
            line = f"\\section*{{{chr(64 + sec) if app else sec} {m.group(1)}}}"
        m = re.match(r"\\subsection\{(.*)\}", line)
        if m:
            sub += 1
            line = f"\\subsection*{{{sec}.{sub} {m.group(1)}}}"
        lines.append(line)
    body = "\n".join(lines)
    title = re.search(r"\\title\{(.*?)\}\n", src, re.S).group(1).replace("\\\\", " ").replace("\n", " ")
    return title, body


def make_reference_docx(path: Path) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor
    ref = subprocess.run(["pandoc", "-o", str(path), "--print-default-data-file", "reference.docx"], check=True)
    d = Document(str(path))
    black = RGBColor(0, 0, 0)

    def font(st, size=None, bold=None, italic=None, name=FONT):
        st.font.name = name
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts")
        if rf is None:
            from docx.oxml import OxmlElement
            rf = OxmlElement("w:rFonts")
            rpr.insert(0, rf)
        from docx.oxml.ns import qn
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), name)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]
        if size:
            st.font.size = Pt(size)
        if bold is not None:
            st.font.bold = bold
        if italic is not None:
            st.font.italic = italic
        st.font.color.rgb = black

    styles = {s.name: s for s in d.styles}
    for s in d.styles:
        if s.type in (1, 2):
            try:
                font(s)
            except Exception:
                pass
    for n in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if n in styles:
            font(styles[n], 11)
            pf = styles[n].paragraph_format
            pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(6), 1.15
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf.widow_control = True
    if "Compact" in styles:
        styles["Compact"].paragraph_format.space_after = Pt(2)
    for n, size, b, it, before, after in (("Title", 16, True, False, 0, 8), ("Heading 1", 13, True, False, 14, 6),
                                          ("Heading 2", 11, True, False, 10, 4), ("Heading 3", 11, True, True, 8, 3),
                                          ("Heading 4", 11, True, False, 8, 3)):
        if n in styles:
            font(styles[n], size, b, it)
            pf = styles[n].paragraph_format
            pf.space_before, pf.space_after, pf.keep_with_next = Pt(before), Pt(after), True
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER if n == "Title" else WD_ALIGN_PARAGRAPH.LEFT
    for n in ("Author", "Date", "Subtitle"):
        if n in styles:
            font(styles[n], 11, False, False)
            styles[n].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            styles[n].paragraph_format.space_after = Pt(0)
    for n in ("Caption", "Image Caption", "Table Caption"):
        if n in styles:
            font(styles[n], 10, False, False)
            pf = styles[n].paragraph_format
            pf.space_before, pf.space_after = Pt(4), Pt(10)
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if "Table Caption" in styles:
        styles["Table Caption"].paragraph_format.keep_with_next = True
        styles["Table Caption"].paragraph_format.space_before = Pt(10)
        styles["Table Caption"].paragraph_format.space_after = Pt(4)
    if "Captioned Figure" in styles:
        styles["Captioned Figure"].paragraph_format.keep_with_next = True
    for n in ("Source Code", "Verbatim Char"):
        if n in styles:
            font(styles[n], 9, name="Courier New")
    if "Source Code" in styles:
        pf = styles["Source Code"].paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.space_before, pf.space_after, pf.line_spacing = Pt(4), Pt(8), 1.0
        pf.left_indent = Cm(0.4)
    if "Bibliography" in styles:
        font(styles["Bibliography"], 10)
        pf = styles["Bibliography"].paragraph_format
        pf.left_indent, pf.first_line_indent, pf.space_after = Cm(0.6), Cm(-0.6), Pt(4)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if "Hyperlink" in styles:
        font(styles["Hyperlink"])
        styles["Hyperlink"].font.underline = False
    if "Abstract" in styles:
        font(styles["Abstract"], 10)
    d.save(str(path))


def post_process(src: Path, dst: Path) -> None:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    d = Document(str(src))

    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.5)
    p = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.font.name, run.font.size = FONT, Pt(10)
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), kind); run._r.append(el)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text; run._r.append(el)

    # display equations: centred, with a right-aligned number
    eq_no = 0
    for par in d.paragraphs:
        omp = par._p.find(M + "oMathPara")
        if omp is None:
            continue
        eq_no += 1
        omath = omp.find(M + "oMath")
        par._p.remove(omp)
        tabs = par.paragraph_format.tab_stops
        tabs.add_tab_stop(Cm(8.0), WD_TAB_ALIGNMENT.CENTER)
        tabs.add_tab_stop(Cm(16.0), WD_TAB_ALIGNMENT.RIGHT)
        r1 = par.add_run("\t")._r
        r1.addnext(omath)
        par.add_run(f"\t({eq_no})")
        par.paragraph_format.space_before = Pt(6)
        par.paragraph_format.space_after = Pt(6)
        par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    # math runs: merge split upright letters (E,M -> EM), then set 11 pt in schema order (m:rPr, w:rPr, m:t)
    def sty_p(r):
        mr = r.find(M + "rPr")
        return mr is not None and mr.find(M + "sty") is not None and mr.find(M + "sty").get(M + "val") == "p"
    for om in list(d.element.body.iter(M + "oMath")):
        for parent in [om] + list(om.iter(M + "e")) + list(om.iter(M + "num")) + list(om.iter(M + "den")):
            kids = list(parent)
            for a, b in zip(kids, kids[1:]):
                if a.tag == b.tag == M + "r" and sty_p(a) and sty_p(b) and a.getparent() is parent and b.getparent() is parent:
                    ta, tb = a.find(M + "t"), b.find(M + "t")
                    if ta is not None and tb is not None and (ta.text or "").isalpha() and (tb.text or "").isalpha():
                        tb.text = ta.text + tb.text
                        parent.remove(a)
    for r in d.element.body.iter(M + "r"):
        for old in r.findall(W + "rPr"):
            r.remove(old)
        rpr = OxmlElement("w:rPr")
        for tag in ("w:sz", "w:szCs"):
            el = OxmlElement(tag); el.set(qn("w:val"), "22"); rpr.append(el)
        mr = r.find(M + "rPr")
        if mr is not None:
            mr.addnext(rpr)
        else:
            r.insert(0, rpr)

    # tables: three rules, 9 pt, centred, compact, rows not split across pages
    for t in d.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = t._tbl.tblPr
        for old in tblPr.findall(qn("w:tblBorders")):
            tblPr.remove(old)
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{edge}")
            if edge in ("top", "bottom"):
                e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "8"); e.set(qn("w:color"), "000000")
            else:
                e.set(qn("w:val"), "nil")
            borders.append(e)
        tblPr.append(borders)
        for old in tblPr.findall(qn("w:tblW")):
            tblPr.remove(old)
        tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), "9072"); tw.set(qn("w:type"), "dxa"); tblPr.append(tw)
        mar = OxmlElement("w:tblCellMar")
        for edge, v in (("top", "15"), ("left", "60"), ("bottom", "15"), ("right", "60")):
            e = OxmlElement(f"w:{edge}"); e.set(qn("w:w"), v); e.set(qn("w:type"), "dxa"); mar.append(e)
        for old in tblPr.findall(qn("w:tblCellMar")):
            tblPr.remove(old)
        tblPr.append(mar)
        ncols = len(t.columns)
        size = Pt(8) if ncols >= 7 else Pt(9)
        # column widths proportional to the longest line in each column (text width 16 cm = 9072 dxa)
        # widths: each column gets at least its longest word, the rest is shared by content length
        cells = [[t.cell(r, c).text for r in range(len(t.rows))] for c in range(ncols)]
        need = [max(max((len(w) for w in x.split()), default=1) for x in col) + 1.5 for col in cells]
        want = [max(len(x) for x in col) + 1.5 for col in cells]
        total_chars = 9072 / (90 if size == Pt(8) else 100)   # approx. chars that fit across the text width
        if sum(need) >= total_chars:
            widths = [int(9072 * n / sum(need)) for n in need]
        else:
            extra = [max(w - n, 0) for w, n in zip(want, need)]
            room = total_chars - sum(need)
            add = [room * e / sum(extra) if sum(extra) else room / ncols for e in extra]
            add = [min(a, e) for a, e in zip(add, extra)] if sum(want) <= total_chars else add
            ch = [n + a for n, a in zip(need, add)]
            widths = [int(9072 * c / sum(ch)) for c in ch]
        grid = t._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(w))
        for row in t.rows:
            for ci, cell in enumerate(row.cells[:ncols]):
                tcPr = cell._tc.get_or_add_tcPr()
                for old in tcPr.findall(qn("w:tcW")):
                    tcPr.remove(old)
                tcw = OxmlElement("w:tcW"); tcw.set(qn("w:w"), str(widths[ci])); tcw.set(qn("w:type"), "dxa")
                tcPr.insert(0, tcw)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed")
        for old in tblPr.findall(qn("w:tblLayout")):
            tblPr.remove(old)
        tblPr.append(lay)
        for ri, row in enumerate(t.rows):
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); trPr.append(cs)
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.space_after = Pt(0)
                    par.paragraph_format.space_before = Pt(0)
                    par.paragraph_format.line_spacing = 1.0
                    for r in par.runs:
                        r.font.size = size
                        if ri == 0:
                            r.font.bold = True
                if ri == 0:
                    tcPr = cell._tc.get_or_add_tcPr()
                    b = OxmlElement("w:tcBorders")
                    e = OxmlElement("w:bottom"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000")
                    b.append(e); tcPr.append(b)
        # keep header row on the first line of a table page break
        hdr = t.rows[0]._tr.get_or_add_trPr()
        th = OxmlElement("w:tblHeader"); hdr.append(th)
        for row in t.rows[:-1]:
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.keep_with_next = True

    # schema order of table and cell properties
    TBL = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize", "tblW", "jc",
           "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"]
    TC = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar", "textDirection",
          "tcFitText", "vAlign", "hideMark"]
    def reorder(el, order):
        kids = list(el)
        for k in kids:
            el.remove(k)
        kids.sort(key=lambda k: order.index(k.tag.split("}")[1]) if k.tag.split("}")[1] in order else len(order))
        for k in kids:
            el.append(k)
    for t in d.tables:
        reorder(t._tbl.tblPr, TBL)
        for tc in t._tbl.iter(W + "tcPr"):
            reorder(tc, TC)
    for rp in d.element.body.iter(M + "rPr"):
        if rp.find(M + "nor") is not None:          # nor (normal text) and sty are mutually exclusive
            for st in rp.findall(M + "sty"):
                rp.remove(st)
        reorder(rp, ["lit", "nor", "scr", "sty", "brk", "aln"])
    for dp in d.element.body.iter(M + "dPr"):
        reorder(dp, ["begChr", "sepChr", "endChr", "grow", "shp", "ctrlPr"])
    # pandoc's template parts: put style children in schema order, fix nsid length, drop misplaced zoom
    STYLE = ["name", "aliases", "basedOn", "next", "link", "autoRedefine", "hidden", "uiPriority", "semiHidden",
             "unhideWhenUsed", "qFormat", "locked", "personal", "personalCompose", "personalReply", "rsid", "pPr", "rPr",
             "tblPr", "trPr", "tcPr", "tblStylePr"]
    PPR = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
           "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct",
           "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind",
           "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc", "textDirection", "textAlignment",
           "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange"]
    st_el = d.styles.element
    for st in st_el.iter(W + "style"):
        reorder(st, STYLE)
    for pp in st_el.iter(W + "pPr"):
        reorder(pp, PPR)
    for tc in st_el.iter(W + "tcPr"):
        reorder(tc, TC)
    for z in d.settings.element.findall(W + "zoom"):
        d.settings.element.remove(z)
    SETTINGS = """writeProtection view zoom removePersonalInformation removeDateAndTime doNotDisplayPageBoundaries
    displayBackgroundShape printPostScriptOverText printFractionalCharacterWidth printFormsData embedTrueTypeFonts
    embedSystemFonts saveSubsetFonts saveFormsData mirrorMargins alignBordersAndEdges bordersDoNotSurroundHeader
    bordersDoNotSurroundFooter gutterAtTop hideSpellingErrors hideGrammaticalErrors activeWritingStyle proofState
    formsDesign attachedTemplate linkStyles stylePaneFormatFilter stylePaneSortMethod documentType mailMerge revisionView
    trackRevisions doNotTrackMoves doNotTrackFormatting documentProtection autoFormatOverride styleLockTheme
    styleLockQFSet defaultTabStop autoHyphenation consecutiveHyphenLimit hyphenationZone doNotHyphenateCaps showEnvelope
    summaryLength clickAndTypeStyle defaultTableStyle evenAndOddHeaders bookFoldRevPrinting bookFoldPrinting
    bookFoldPrintingSheets drawingGridHorizontalSpacing drawingGridVerticalSpacing displayHorizontalDrawingGridEvery
    displayVerticalDrawingGridEvery doNotUseMarginsForDrawingGridOrigin drawingGridHorizontalOrigin
    drawingGridVerticalOrigin doNotShadeFormData noPunctuationKerning characterSpacingControl printTwoOnOne
    strictFirstAndLastChars noLineBreaksAfter noLineBreaksBefore savePreviewPicture doNotValidateAgainstSchema
    saveInvalidXml ignoreMixedContent alwaysShowPlaceholderText doNotDemarcateInvalidXml saveXmlDataOnly
    useXSLTWhenSaving saveThroughXslt showXMLTags alwaysMergeEmptyNamespace updateFields hdrShapeDefaults footnotePr
    endnotePr compat docVars rsids mathPr attachedSchema themeFontLang clrSchemeMapping doNotIncludeSubdocsInStats
    doNotAutoCompressPictures forceUpgrade captions readModeInkLockDown smartTagType schemaLibrary shapeDefaults
    doNotEmbedSmartTags decimalSymbol listSeparator""".split()
    reorder(d.settings.element, SETTINGS)
    try:
        num_el = d.part.numbering_part.element
        for ns in num_el.iter(W + "nsid"):
            v = ns.get(W + "val")
            if v and len(v) < 8:
                ns.set(W + "val", v.zfill(8))
    except Exception:
        pass
    pgMar = sec._sectPr.find(W + "pgMar")
    for a, v in (("header", "709"), ("footer", "709"), ("gutter", "0")):
        if pgMar.get(W + a) is None:
            pgMar.set(W + a, v)

    # code blocks: left-aligned (never justified), and a little space after every table
    for par in d.paragraphs:
        if par.style.name == "Source Code":
            par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for t in d.tables:
        nxt = t._tbl.getnext()
        if nxt is not None and nxt.tag == W + "p":
            from docx.text.paragraph import Paragraph
            Paragraph(nxt, d._body).paragraph_format.space_before = Pt(10)
    # drop empty body paragraphs
    for par in list(d.paragraphs):
        if not par.text.strip() and not par._p.findall(".//" + W + "drawing") and par._p.find(".//" + M + "oMath") is None \
                and par.style.name not in ("Title",):
            par._p.getparent().remove(par._p)
    # appendix headings start on a new page? (no: keep flow; only references heading gets a little space)
    d.core_properties.title = "How Much Retrieval Quality Reaches the Answer?"
    d.core_properties.author = "Muhammad Adeel"
    d.save(str(dst))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    title, body = build_body()
    doc = ("\\documentclass{article}\n\\usepackage{graphicx}\n"
           f"\\title{{{title}}}\n"
           "\\author{Muhammad Adeel\\\\Zhengzhou University\\\\muhammadadeel167@stu.zzu.edu.cn\\\\adeelcheena791@gmail.com}\n"
           "\\date{}\n\\begin{document}\n\\maketitle\n" + body + "\n\\end{document}\n")
    flat = OUT / "manuscript_flat.tex"
    flat.write_text(doc)
    ref = OUT / "reference_manuscript.docx"
    make_reference_docx(ref)
    raw = OUT / "manuscript_raw.docx"
    subprocess.run(["pandoc", str(flat), "--citeproc", "--bibliography", str(LATEX / "references.bib"),
                    "--metadata", "reference-section-title=References", "--metadata", "link-citations=true",
                    "--reference-doc", str(ref), "--standalone", "-o", str(raw)], check=True)
    out = OUT / "rag_embed_transfer_manuscript.docx"
    post_process(raw, out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
