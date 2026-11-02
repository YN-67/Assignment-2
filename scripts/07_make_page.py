"""生成交互式分析页面 index.html（GitHub Pages 用）。

把 output/ 下的评估结果与红楼梦逐回预测整合成一个可交互页面：
  - 指标卡片（准确率/精确率/召回率/F1）
  - 红楼梦情感曲线（可悬停查看回目与分数，可切换移动平均窗口）
  - 最正面/最负面词的权重条形图
  - 错分文本浏览器（可按方向筛选、展开原文）
  - 域错位诊断说明

纯静态、零依赖（Chart.js 走 CDN），数据内嵌为 JSON。
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

def clean_title(title: str) -> str:
    """回目行去掉正文开头，只留标题。"""
    # 标题 = 前两段（以全角空格分隔），正文从「话说/却说/且说」等开始
    for kw in ["话说", "却说", "且说", "话说宝玉", "诗云"]:
        idx = title.find(kw)
        if idx > 0:
            return title[:idx].strip()
    return title[:30].strip()

def main() -> None:
    metrics = {}
    for line in (OUTPUT / "metrics.txt").read_text(encoding="utf-8").splitlines():
        if " = " in line:
            k, v = line.split(" = ")
            metrics[k.strip()] = float(v)

    hlm = pd.read_csv(OUTPUT / "hlm_predictions.csv")
    hlm["title"] = hlm["title"].map(clean_title)
    hlm_records = hlm[["chapter", "title", "p_positive"]].to_dict("records")

    mis = pd.read_csv(OUTPUT / "misclassified.csv")
    mis_records = mis.to_dict("records")

    tw = pd.read_csv(OUTPUT / "top_words.csv")
    pos = tw[tw["kind"] == "最正面"].sort_values("weight", ascending=False)
    neg = tw[tw["kind"] == "最负面"].sort_values("weight")
    words_data = {
        "pos": pos[["word", "weight"]].to_dict("records"),
        "neg": neg[["word", "weight"]].to_dict("records"),
    }

    data = {
        "metrics": metrics,
        "hlm": hlm_records,
        "misclassified": mis_records,
        "words": words_data,
    }

    template_path = ROOT / "scripts" / "index_template.html"
    html = template_path.read_text(encoding="utf-8")
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False))

    out = ROOT / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"已生成 {out}（{out.stat().st_size} 字节）")

if __name__ == "__main__":
    main()
