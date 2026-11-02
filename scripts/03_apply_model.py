"""把训练好的情感分类器应用到《红楼梦》，逐回打分。

流程：
  1. 读 corpora/红楼梦.txt，按"回"切分（120 回）
  2. 每回 jieba 分词 → TF-IDF → 模型打分 P(positive)
  3. 输出 output/hlm_predictions.csv 与情感曲线 output/hlm_sentiment_arc.png
"""

import re
from pathlib import Path

import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import RANDOM_STATE, ROOT, load_data, tokenize

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

CHAPTER_RE = re.compile(r"第[一二三四五六七八九十百零]+回")

def split_chapters(text: str) -> list[tuple[str, str]]:
    """按「第X回」标题切分全书，返回 (回目行, 回正文) 列表。

    只把「第X回 + 空白 + 回目文字」的行首匹配当作回目；
    正文里提到的「第X回」（后面直接跟正文）不算。
    """
    matches = []
    for m in CHAPTER_RE.finditer(text):
        at_line_start = text.rfind("\n", 0, m.start()) + 1 == m.start()
        after = text[m.end():m.end() + 30]
        # 真回目：后面紧跟空白和另一段标题文字（如「甄士隐梦幻识通灵」）
        if at_line_start and re.match(r"\s+\S", after):
            matches.append(m)
    chapters = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        title = " ".join(lines[:2]) if lines else f"第{i+1}回"
        chapters.append((title, chunk))
    return chapters

def tokenize(text: str) -> str:
    """与训练脚本一致的分词方式（去单字与标点）。"""
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    words = jieba.lcut(text)
    return " ".join(w for w in words if len(w) > 1)

def main() -> None:
    raw = (ROOT / "corpora" / "红楼梦.txt").read_text(encoding="utf-8")
    chapters = split_chapters(raw)
    print(f"切分出 {len(chapters)} 回")

    # 重新训练（与 02 脚本完全相同的流程，保证可复现）
    df = load_data()
    X_train, _, y_train, _ = train_test_split(
        df["tokens"], df["label"], test_size=0.2, stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    vec = TfidfVectorizer()
    X_train_vec = vec.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train_vec, y_train)

    # 逐回打分
    rows = []
    for i, (title, chunk) in enumerate(chapters, start=1):
        tokens = tokenize(chunk)
        if not tokens:
            continue
        p_pos = clf.predict_proba(vec.transform([tokens]))[0, 1]
        rows.append({"chapter": i, "title": title, "p_positive": p_pos})

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT / "hlm_predictions.csv", index=False, encoding="utf-8-sig")

    # 情感曲线
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(out["chapter"], out["p_positive"], marker="o", markersize=3,
            linewidth=0.8, alpha=0.5, label="每回 P(positive)")
    rolling = out["p_positive"].rolling(10, center=True).mean()
    ax.plot(out["chapter"], rolling, linewidth=2.5, color="crimson",
            label="10 回移动平均")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("回数")
    ax.set_ylabel("P(positive)")
    ax.set_title("《红楼梦》逐回情感曲线（豆瓣书评模型）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "hlm_sentiment_arc.png", dpi=150)
    print(f"已写出 hlm_predictions.csv 与 hlm_sentiment_arc.png")
    print(f"最高 3 回：")
    print(out.nlargest(3, "p_positive")[["chapter", "p_positive"]].to_string(index=False))
    print(f"最低 3 回：")
    print(out.nsmallest(3, "p_positive")[["chapter", "p_positive"]].to_string(index=False))

if __name__ == "__main__":
    main()
