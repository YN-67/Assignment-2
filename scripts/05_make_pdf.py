"""把 report.md 转成排版好的 report.pdf（经 HTML + Edge 无头打印）。

PDF 不应是 Markdown 的直接输出：这里加了正式报告的排版
（衬线中文字体、页边距、表格样式、页码），再用 Edge 打印成 PDF。
"""

import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent

CSS = """
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

def main() -> None:
    md_text = (ROOT / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists"])
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>作业 2 报告</title><style>{CSS}</style></head>
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
    print(f"已生成 {pdf_path}（{pdf_path.stat().st_size} 字节）")

if __name__ == "__main__":
    main()
