"""共享工具：分词、停用词、模型训练。

01/02/03/04 号脚本共用，保证分词方式与模型完全一致。
"""

import re
from pathlib import Path

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42

# 常见中文停用词（覆盖最影响 TF-IDF 的高频虚词）
STOPWORDS = {
    "的", "了", "和", "是", "就", "都", "而", "及", "与", "著", "或",
    "一个", "没有", "我们", "你们", "他们", "她们", "自己", "这个", "那个",
    "这些", "那些", "什么", "怎么", "为什么", "这样", "那样", "还是", "但是",
    "不过", "然而", "因为", "所以", "如果", "虽然", "可以", "可能", "应该",
    "就是", "不是", "也是", "还有", "以及", "并且", "而且", "只是", "有些",
    "一点", "一些", "非常", "真的", "感觉", "觉得", "知道", "已经", "现在",
    "时候", "地方", "问题", "东西", "事情", "书", "本书", "一本", "这部",
    "这部书", "作者", "读者", "内容", "故事", "小说", "电影", "看", "读",
    "看完", "读过", "阅读", "这本书", "那本书",
}

PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")

def tokenize(text: str) -> str:
    """jieba 分词，去停用词与标点，返回空格分隔的词串。"""
    text = PUNCT_RE.sub(" ", text)
    words = jieba.lcut(text)
    return " ".join(w for w in words if w not in STOPWORDS and len(w) > 1)

def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "labeled.csv")
    df["tokens"] = df["text"].astype(str).map(tokenize)
    return df

def train_model(df: pd.DataFrame):
    """返回 (vectorizer, classifier, X_test_index, y_test, y_pred)。"""
    X = df["tokens"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    vec = TfidfVectorizer()
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    return vec, clf, X_test.index, y_test, y_pred
