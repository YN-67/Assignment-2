"""从 douban.json 抽取平衡的正/负样本，生成 data/labeled.csv。

标签定义：
  label 1 = 正面（豆瓣 5 星）
  label 0 = 负面（豆瓣 1 星）
"""

import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N_PER_CLASS = 1000          # 每类抽取条数（作业要求 >=200，这里取 1000 更稳）
RANDOM_STATE = 42

def main() -> None:
    random.seed(RANDOM_STATE)

    src = ROOT / "douban.json"
    with src.open(encoding="utf-8") as f:
        reviews = json.load(f)

    positive = [r["text"] for r in reviews if r["label"] == 5]
    negative = [r["text"] for r in reviews if r["label"] == 1]
    print(f"可用样本：5 星 {len(positive)} 条，1 星 {len(negative)} 条")

    pos_sample = random.sample(positive, N_PER_CLASS)
    neg_sample = random.sample(negative, N_PER_CLASS)

    rows = [(t, 1) for t in pos_sample] + [(t, 0) for t in neg_sample]
    random.shuffle(rows)

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "labeled.csv"
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(out, index=False, encoding="utf-8-sig")

    print(f"已写出 {out}：共 {len(rows)} 条（正面 {N_PER_CLASS}，负面 {N_PER_CLASS}）")

if __name__ == "__main__":
    main()
