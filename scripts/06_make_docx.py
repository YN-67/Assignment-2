"""把 report.md 转成排版好的 report.docx（Word 版，与 PDF 同源同样式）。

Moodle 要求同时上传 PDF 与 Word 格式；两者都从 report.md 生成，
保证内容一致。样式：宋体正文、黑体标题、居中表格。
"""

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent


def set_font(run, name: str, size: float, bold: bool = False) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def add_runs(par, text: str, size: float, base_bold: bool = False) -> None:
    """把 **加粗** 标记转成 run。"""
    for i, seg in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not seg:
            continue
        run = par.add_run(seg)
        set_font(run, "宋体", size, bold=base_bold or (i % 2 == 1))


def add_table(doc, rows: list[list[str]]) -> None:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    # 去掉分隔行 |---|---|
    cells = [r for r in cells if not all(set(c) <= set("-: ") for c in r)]
    t = doc.add_table(rows=len(cells), cols=len(cells[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            par = t.cell(i, j).paragraphs[0]
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(par, cell, 10.5, base_bold=(i == 0))


def merge_bold_lines(text: str) -> str:
    """把跨行的 **加粗** 合并到一行，便于逐行解析。

    Markdown 里加粗可以跨软换行（如两行合成一个加粗段），
    逐行解析会漏掉闭合的 **，这里先合并。
    """
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
                # 只吞并普通文本行，不吞表格/标题/空行
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


def main() -> None:
    md = merge_bold_lines((ROOT / "report.md").read_text(encoding="utf-8"))
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
            add_table(doc, rows)
            continue
        if s.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                par = doc.add_paragraph()
                run = par.add_run(lines[i])
                set_font(run, "Consolas", 9.5)
                i += 1
            i += 1
            continue
        if s.startswith("# "):
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(par, s[2:], 19, base_bold=True)
        elif s.startswith("## "):
            par = doc.add_paragraph()
            add_runs(par, s[3:], 14.5, base_bold=True)
            par.paragraph_format.space_before = Pt(14)
        elif s.startswith("### "):
            par = doc.add_paragraph()
            add_runs(par, s[4:], 12.5, base_bold=True)
            par.paragraph_format.space_before = Pt(10)
        elif re.match(r"^\d+\.\s", s):
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Cm(0.6)
            add_runs(par, s, 11.5)
        elif s:
            par = doc.add_paragraph()
            par.paragraph_format.first_line_indent = Cm(0)
            par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_runs(par, s, 11.5)
        i += 1

    out = ROOT / "report.docx"
    doc.save(str(out))
    print(f"已生成 {out}（{out.stat().st_size} 字节）")


if __name__ == "__main__":
    main()
