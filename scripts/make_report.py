"""把 report.md 排版成 report.pdf 与 report.docx。

PDF：markdown → 带 CSS 的 HTML → Edge 无头打印（正式报告样式，
非 Markdown 直接匯出）。DOCX：python-docx 逐行解析（宋体正文、
黑体标题、居中表格），与 PDF 同源同样式。

用法：
    python scripts/make_report.py          # 生成 PDF + DOCX
    python scripts/make_report.py pdf      # 只生成 PDF
    python scripts/make_report.py docx     # 只生成 DOCX
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------- PDF ----------------

PDF_CSS = """
@page { size: A4; margin: 2.2cm 2.4cm; }
body {
  font-family: "SimSun", "Microsoft YaHei", serif;
  font-size: 11.5pt; line-height: 1.75; color: #1a1a1a;
  max-width: 100%;
}
h1 { font-family: "SimHei", "Microsoft YaHei"; font-size: 19pt;
     text-align: center; margin: 0 0 0.6em; }
h2 { font-family: "SimHei", "Microsoft YaHei"; font-size: 14.5pt;
     border-bottom: 1.5pt solid #333; padding-bottom: 3pt;
     margin: 1.4em 0 0.6em; page-break-after: avoid; }
h3 { font-family: "SimHei", "Microsoft YaHei"; font-size: 12.5pt;
     margin: 1.1em 0 0.4em; page-break-after: avoid; }
p { margin: 0.45em 0; text-align: justify; }
table { border-collapse: collapse; margin: 0.7em auto; width: 92%;
        font-size: 10.5pt; page-break-inside: avoid; }
th, td { border: 0.6pt solid #888; padding: 4pt 8pt; text-align: center; }
th { background: #f0f0f0; font-family: "SimHei", "Microsoft YaHei"; }
code { font-family: Consolas, monospace; font-size: 10pt;
       background: #f5f5f5; padding: 1pt 3pt; }
strong { font-family: "SimHei", "Microsoft YaHei"; }
ol, ul { margin: 0.4em 0; padding-left: 2em; }
li { margin: 0.25em 0; text-align: justify; }
"""

def make_pdf() -> None:
    import markdown

    md_text = (ROOT / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists"])
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>作业 2 报告</title><style>{PDF_CSS}</style></head>
<body>{body}</body></html>"""
    html_path = ROOT / "report.html"
    html_path.write_text(html, encoding="utf-8")

    edge = None
    for cand in [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]:
        if Path(cand).exists():
            edge = cand
            break
    if edge is None:
        raise SystemExit("未找到 Edge，无法打印 PDF")

    pdf_path = ROOT / "report.pdf"
    subprocess.run(
        [edge, "--headless", "--disable-gpu",
         "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}",
         html_path.as_uri()],
        check=True, capture_output=True, timeout=120,
    )
    html_path.unlink()
    print(f"已生成 {pdf_path}（{pdf_path.stat().st_size} 字节）")

# ---------------- DOCX ----------------

def _set_font(run, name: str, size: float, bold: bool = False) -> None:
    from docx.oxml.ns import qn
    from docx.shared import Pt

    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)

def _add_runs(par, text: str, size: float, base_bold: bool = False) -> None:
    """把 **加粗** 标记转成 run。"""
    for i, seg in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not seg:
            continue
        run = par.add_run(seg)
        _set_font(run, "宋体", size, bold=base_bold or (i % 2 == 1))

def _add_table(doc, rows: list[str]) -> None:
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(set(c) <= set("-: ") for c in r)]
    t = doc.add_table(rows=len(cells), cols=len(cells[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            par = t.cell(i, j).paragraphs[0]
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_runs(par, cell, 10.5, base_bold=(i == 0))

def _merge_bold_lines(text: str) -> str:
    """把跨行的 **加粗** 合并到一行，便于逐行解析。"""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.count("**") % 2 == 1:  # 奇数个 **：加粗跨行
            merged = line
            j = i + 1
            while j < len(lines) and merged.count("**") % 2 == 1:
                nxt = lines[j]
                if nxt.strip() and not nxt.strip().startswith(("|", "#", "```")):
                    merged += nxt.strip()
                    j += 1
                else:
                    break
            out.append(merged)
            i = j
        else:
            out.append(line)
            i += 1
    return "\n".join(out)

def make_docx() -> None:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    md = _merge_bold_lines((ROOT / "report.md").read_text(encoding="utf-8"))
    doc = Document()
    for section in doc.sections:
        section.top_margin = section.bottom_margin = Cm(2.2)
        section.left_margin = section.right_margin = Cm(2.4)

    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            _add_table(doc, rows)
            continue
        if s.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                par = doc.add_paragraph()
                run = par.add_run(lines[i])
                _set_font(run, "Consolas", 9.5)
                i += 1
            i += 1
            continue
        if s.startswith("# "):
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_runs(par, s[2:], 19, base_bold=True)
        elif s.startswith("## "):
            par = doc.add_paragraph()
            _add_runs(par, s[3:], 14.5, base_bold=True)
            par.paragraph_format.space_before = Pt(14)
        elif s.startswith("### "):
            par = doc.add_paragraph()
            _add_runs(par, s[4:], 12.5, base_bold=True)
            par.paragraph_format.space_before = Pt(10)
        elif re.match(r"^\d+\.\s", s):
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Cm(0.6)
            _add_runs(par, s, 11.5)
        elif s:
            par = doc.add_paragraph()
            par.paragraph_format.first_line_indent = Cm(0)
            par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _add_runs(par, s, 11.5)
        i += 1

    out = ROOT / "report.docx"
    doc.save(str(out))
    print(f"已生成 {out}（{out.stat().st_size} 字节）")

# ---------------- main ----------------

def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in ("all", "pdf"):
        make_pdf()
    if target in ("all", "docx"):
        make_docx()

if __name__ == "__main__":
    main()
