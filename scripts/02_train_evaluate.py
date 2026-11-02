"""训练逻辑回归情感分类器并在测试集上评估。

流程：
  1. 读 data/labeled.csv（text, label）
  2. jieba 分词 + 去停用词
  3. TF-IDF（只在训练集上 fit）
  4. 分层切分 80/20，random_state=42
  5. 训练 LogisticRegression
  6. 输出：指标、混淆矩阵、逐条预测、错分文本、最正/最负的词
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from common import RANDOM_STATE, ROOT, load_data, train_model

OUTPUT = ROOT / "output"

# 让 matplotlib 正常显示中文（Windows 自带微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

def main() -> None:
    OUTPUT.mkdir(exist_ok=True)

    df = load_data()
    print(f"读入 {len(df)} 条标注数据")

    vec, clf, test_idx, y_test, y_pred = train_model(df)
    print(f"训练集 {int(len(df) * 0.8)} 条，测试集 {len(y_test)} 条")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"准确率 {acc:.3f}  精确率 {prec:.3f}  召回率 {rec:.3f}  F1 {f1:.3f}")
    print("混淆矩阵：")
    print(cm)

    # ---- 指标文件 ----
    with (OUTPUT / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"accuracy  = {acc:.4f}\n")
        f.write(f"precision = {prec:.4f}\n")
        f.write(f"recall    = {rec:.4f}\n")
        f.write(f"f1        = {f1:.4f}\n")
        f.write(f"train_size= {int(len(df) * 0.8)}\n")
        f.write(f"test_size = {len(y_test)}\n")

    # ---- 混淆矩阵 CSV + 图 ----
    cm_df = pd.DataFrame(
        cm,
        index=["实际负面", "实际正面"],
        columns=["预测负面", "预测正面"],
    )
    cm_df.to_csv(OUTPUT / "confusion_matrix.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["预测负面", "预测正面"])
    ax.set_yticks([0, 1], ["实际负面", "实际正面"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=20)
    ax.set_title(f"混淆矩阵（准确率 {acc:.1%}）")
    fig.tight_layout()
    fig.savefig(OUTPUT / "confusion_matrix.png", dpi=150)

    # ---- 测试集逐条预测 ----
    test_df = pd.DataFrame({
        "text": df.loc[test_idx, "text"].values,
        "true_label": y_test.values,
        "pred_label": y_pred,
        "p_positive": clf.predict_proba(vec.transform(df.loc[test_idx, "tokens"]))[:, 1],
    })
    test_df.to_csv(OUTPUT / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # ---- 被错分的文本 ----
    wrong = test_df[test_df["true_label"] != test_df["pred_label"]]
    wrong.to_csv(OUTPUT / "misclassified.csv", index=False, encoding="utf-8-sig")
    print(f"错分 {len(wrong)} 条，已写入 misclassified.csv")

    # ---- 最正/最负的 20 个词 ----
    feature_names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    words_df = pd.DataFrame({"word": feature_names, "weight": coef})
    top_pos = words_df.nlargest(20, "weight")
    top_neg = words_df.nsmallest(20, "weight")
    pd.concat(
        [top_pos.assign(kind="最正面"), top_neg.assign(kind="最负面")],
        ignore_index=True,
    ).to_csv(OUTPUT / "top_words.csv", index=False, encoding="utf-8-sig")

    print("最正面 20 词：", "、".join(top_pos["word"]))
    print("最负面 20 词：", "、".join(top_neg["word"]))
    print("全部输出已写入 output/")

if __name__ == "__main__":
    main()
