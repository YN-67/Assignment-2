# 作业 2：训练自己的文本分类器

用豆瓣书评训练一个逻辑回归（Logistic Regression）情感分类器（正面 / 负面），
再把它应用到《红楼梦》上，逐回打分，观察模型的表现与局限。

## 研究问题

豆瓣读书评论的情感（正面 / 负面）能否由文本自动识别？
这样一个在电影/书评上训练的模型，应用到古典小说《红楼梦》时会怎样表现？

## 标签定义

| 标签 | 含义 | 来源 |
|---|---|---|
| `1`（正面） | 豆瓣 **5 星** 评论 | 星级评分直接给出标签 |
| `0`（负面） | 豆瓣 **1 星** 评论 | 星级评分直接给出标签 |

- 文本单位：一条完整的豆瓣评论。
- 只取两端（5 星 / 1 星），中间星级（2–4 星）情感模糊，不使用。
- 标签来自用户自己打的星级，属于"弱标注"，可能含少量噪声（如反讽）。

## 数据来源

- 训练数据：`douban.json`（课程提供，100,000 条豆瓣评论，1–5 星）。
  从中每类随机抽取 1,000 条（5 星 → 正面，1 星 → 负面），共 2,000 条。
- 分析文本：《红楼梦》全文（`corpora/红楼梦.txt`，人民文学出版社通行本，
  课程语料库提供），共 120 回。

## 方法

1. **预处理**：jieba 分词，去除停用词与标点。
2. **特征**：TF-IDF（`sklearn.TfidfVectorizer`，只在训练集上 fit）。
3. **切分**：按标签分层抽样，80% 训练 / 20% 测试，`random_state=42`。
4. **模型**：`sklearn.linear_model.LogisticRegression`。
5. **评估**：测试集上的准确率、混淆矩阵、精确率、召回率、F1。
6. **应用**：模型给《红楼梦》每一回打分，输出每回的 P(positive)。

## 如何运行

```bash
# 1. 准备数据（从 douban.json 抽样生成 data/labeled.csv）
python scripts/01_prepare_data.py

# 2. 训练 + 评估（生成 output/ 下所有评估结果）
python scripts/02_train_evaluate.py

# 3. 应用到红楼梦（生成 output/hlm_predictions.csv 与情感曲线图）
python scripts/03_apply_model.py

# 4. 诊断实验（生成 output/diagnosis_domain.png）
python scripts/04_diagnose.py

# 5. 把 report.md 排版成 report.pdf 与 report.docx
python scripts/make_report.py
```

依赖：`jieba`、`scikit-learn`、`pandas`、`matplotlib`、`markdown`。

## 仓库结构

```
├── README.md
├── data/
│   └── labeled.csv            # 标注好的训练数据（text, label）
├── corpora/
│   └── 红楼梦.txt              # 分析文本
├── scripts/
│   ├── 01_prepare_data.py     # 数据准备
│   ├── 02_train_evaluate.py   # 训练与评估
│   ├── 03_apply_model.py      # 应用到红楼梦
│   ├── 04_diagnose.py         # 诊断实验（域错位、长度、反讽、极端词）
│   ├── make_report.py         # report.md → report.pdf / report.docx
│   └── common.py              # 共用工具（分词、停用词、绘图）
├── output/
│   ├── metrics.txt            # 准确率/精确率/召回率/F1
│   ├── confusion_matrix.png   # 混淆矩阵图
│   ├── confusion_matrix.csv   # 混淆矩阵数值
│   ├── test_predictions.csv   # 测试集逐条预测
│   ├── misclassified.csv      # 被错分的测试文本
│   ├── top_words.csv          # 最正面/最负面的词与权重
│   ├── diagnosis_domain.png   # 域错位诊断图
│   ├── hlm_predictions.csv    # 红楼梦逐回预测
│   └── hlm_sentiment_arc.png  # 红楼梦情感曲线
└── report.md                  # 报告（2,000–2,500 字）
```

## 限制与说明

- 标签来自星级评分，是代理标签（proxy labels），不是人工细读的判断。
- 模型是词袋模型：不保留词序，无法理解否定与反讽。
- 训练语料是现代书评，应用到 18 世纪小说属于跨域使用，预期会有系统偏差。
