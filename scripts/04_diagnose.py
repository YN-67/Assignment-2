"""诊断脚本：把报告里的猜想变成可检验的证据。

检验四个猜想：
  A. 错分文本是否比正确文本更长？（长评论转折多 → 更难？）
  B. 错分文本里反讽/否定标记词是否更多？
  C. 模型是否在学"书评套路"？（对比：去掉评论套路词后准确率掉多少）
  D. 红楼梦曲线偏负面是域错位还是小说本身？
     （对比：红楼梦 vs 一部现代小说的 P(positive) 分布）
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from common import RANDOM_STATE, ROOT, load_data, tokenize

OUTPUT = ROOT / "output"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 反讽/否定/转折的表面标记词（词袋能看到的）
IRONIC_MARKERS = {
    "呵呵", "真是", "居然", "竟然", "不过", "但是", "可是", "然而",
    "可惜", "遗憾", "本以为", "以为", "结果", "没想到", "无语", "呵呵哒",
    "呵", "哈", "哈喽", "行吧", "好吧", "呵呵呵",
}

def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["tokens"], df["label"], test_size=0.2, stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    vec = TfidfVectorizer()
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    test_df = pd.DataFrame({
        "text": df.loc[X_test.index, "text"].values,
        "tokens": X_test.values,
        "true_label": y_test.values,
        "pred_label": y_pred,
    })
    test_df["correct"] = test_df["true_label"] == test_df["pred_label"]
    test_df["length"] = test_df["text"].str.len()

    print("=" * 60)
    print(f"基线准确率：{acc:.3f}")
    print("=" * 60)

    # ---- A. 长度与错误率 ----
    print("\n【A. 文本长度 vs 错误率】")
    bins = [0, 100, 300, 1000, 10000]
    labels = ["<100字", "100-300字", "300-1000字", ">1000字"]
    test_df["len_bin"] = pd.cut(test_df["length"], bins=bins, labels=labels)
    len_err = test_df.groupby("len_bin", observed=True)["correct"].agg(["mean", "count"])
    len_err["mean"] = 1 - len_err["mean"]  # 错误率
    print(len_err.rename(columns={"mean": "错误率", "count": "条数"}).to_string())

    # ---- B. 反讽标记词与错误率 ----
    print("\n【B. 含反讽/转折标记词的评论 vs 错误率】")
    test_df["has_marker"] = test_df["text"].str.contains(
        "|".join(IRONIC_MARKERS), regex=True
    )
    marker_err = test_df.groupby("has_marker")["correct"].agg(["mean", "count"])
    marker_err["mean"] = 1 - marker_err["mean"]
    marker_err.index = marker_err.index.map({True: "含标记词", False: "不含"})
    print(marker_err.rename(columns={"mean": "错误率", "count": "条数"}).to_string())

    # ---- C. 套路词实验：去掉最正/最负的 40 个词再训练 ----
    print("\n【C. 套路词实验】")
    feature_names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    words_df = pd.DataFrame({"word": feature_names, "weight": coef})
    top40 = set(words_df.nlargest(20, "weight")["word"]) | set(
        words_df.nsmallest(20, "weight")["word"]
    )
    print(f"去掉的 40 个最极端词：{'、'.join(sorted(top40))}")

    df_no40 = df.copy()
    df_no40["tokens"] = df_no40["tokens"].map(
        lambda t: " ".join(w for w in t.split() if w not in top40)
    )
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        df_no40["tokens"], df_no40["label"], test_size=0.2,
        stratify=df_no40["label"], random_state=RANDOM_STATE,
    )
    vec2 = TfidfVectorizer()
    clf2 = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf2.fit(vec2.fit_transform(Xtr2), ytr2)
    acc2 = accuracy_score(yte2, clf2.predict(vec2.transform(Xte2)))
    print(f"基线准确率        ：{acc:.3f}")
    print(f"去掉 40 个极端词后：{acc2:.3f}  （下降 {acc - acc2:+.3f}）")

    # ---- D. 红楼梦 vs 现代小说的 P(positive) 分布 ----
    print("\n【D. 域错位检验：红楼梦 vs 现代小说】")
    # 用同一模型给红楼梦和一部现代小说打分，比较分布
    hlm_text = (ROOT / "corpora" / "红楼梦.txt").read_text(encoding="utf-8")
    hlm_chapters = split_chapters_simple(hlm_text)
    hlm_scores = score_texts(clf, vec, [c[1] for c in hlm_chapters])

    # 现代小说：用训练集里没见过的当代长文本做对照——
    # 这里用 douban.json 里 3 星（中性）评论拼成"伪章节"做域内对照
    import json
    with (ROOT / "douban.json").open(encoding="utf-8") as f:
        all_reviews = json.load(f)
    neutral = [r["text"] for r in all_reviews if r["label"] == 3]
    # 每 20 条拼一个"伪章节"，共 60 个
    pseudo = ["\n".join(neutral[i:i + 20]) for i in range(0, 1200, 20)]
    pseudo_scores = score_texts(clf, vec, pseudo)

    print(f"红楼梦 120 回 P(positive)：均值 {hlm_scores.mean():.3f}，"
          f"标准差 {hlm_scores.std():.3f}")
    print(f"豆瓣 3 星伪章节 P(positive)：均值 {pseudo_scores.mean():.3f}，"
          f"标准差 {pseudo_scores.std():.3f}")
    print("→ 若红楼梦均值显著低于域内对照，说明偏负主要来自域错位，"
          "而非小说本身更悲伤。")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pseudo_scores, bins=20, alpha=0.6, label="豆瓣3星伪章节（域内）")
    ax.hist(hlm_scores, bins=20, alpha=0.6, label="红楼梦120回（域外）")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("P(positive)")
    ax.set_ylabel("章节数")
    ax.set_title("域内 vs 域外文本的 P(positive) 分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "diagnosis_domain.png", dpi=150)
    print("分布图已写入 output/diagnosis_domain.png")

def split_chapters_simple(text: str) -> list[tuple[str, str]]:
    matches = []
    for m in re.finditer(r"第[一二三四五六七八九十百零]+回", text):
        at_line_start = text.rfind("\n", 0, m.start()) + 1 == m.start()
        after = text[m.end():m.end() + 30]
        if at_line_start and re.match(r"\s+\S", after):
            matches.append(m)
    chapters = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapters.append((f"第{i+1}回", text[m.start():end]))
    return chapters

def score_texts(clf, vec, texts: list[str]) -> pd.Series:
    tokens = [tokenize(t) for t in texts]
    tokens = [t for t in tokens if t]
    return pd.Series(clf.predict_proba(vec.transform(tokens))[:, 1])

if __name__ == "__main__":
    main()
