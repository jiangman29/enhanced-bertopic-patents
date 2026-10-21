from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from keyphrase_vectorizers import KeyphraseCountVectorizer
from config import *
import pandas as pd
import numpy as np
from config import MAIN_MODEL_PARAMS


def has_prediction_data(topic_model):
    """
    检查 BERTopic 模型是否启用了 prediction_data，
    用于判断是否能执行相关的预测与离群分析。
    """
    try:
        return hasattr(topic_model.hdbscan_model_, "_prediction_data") and topic_model.hdbscan_model_._prediction_data is not None
    except Exception:
        return False


def build_umap(config_dict):
    return UMAP(**config_dict)

def build_hdbscan(config_dict):
    return HDBSCAN(**config_dict)

def build_vectorizer(stopwords):
    vectorizer_model = KeyphraseCountVectorizer(
        spacy_pipeline="en_core_web_sm",
        stop_words=stopwords,
        pos_pattern="<N.*>*<ADJ>*",
        min_df=3,
        max_df=600,
        decay=1.0,
        delete_min_df=1.0
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    return vectorizer_model, ctfidf_model


def build_topic_model(embedding_model, umap_model, hdbscan_model, vectorizer_model, ctfidf_model=None):
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=True,
        verbose=True
    )

# 训练主模型
def train_topic_model(texts, embeddings, vectorizer_model, ctfidf_model, model_config, embedding_model):
    print("⚙️ Training BERTopic main model...")
    umap_model = build_umap(model_config["umap_config"])
    hdbscan_model = build_hdbscan(model_config["hdbscan_config"])
    topic_model = build_topic_model(
        embedding_model=embedding_model,  # ✅ 直接传入模型对象
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model
    )
    topics, _ = topic_model.fit_transform(texts, embeddings)
    return topic_model, topics


# 训练子模型
def train_sub_topic_model(df, texts, embeddings, vectorizer_model, ctfidf_model, model_config, embedding_model):
    outlier_mask = df["topic"] == -1
    texts_outliers = df.loc[outlier_mask, "clean_text"].tolist()
    embeddings_outliers = embeddings[outlier_mask]

    if len(texts_outliers) == 0:
        df["topic_sub"] = -1
        df["topic_combined"] = df["topic"]
        return None, None, df

    umap_model = build_umap(model_config["umap_config"])              # ✅ 修改变量名
    hdbscan_model = build_hdbscan(model_config["hdbscan_config"])    # ✅ 修改变量名

    topic_model_sub = build_topic_model(
        embedding_model=embedding_model,             
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model
    )
    topics_sub, _ = topic_model_sub.fit_transform(texts_outliers, embeddings_outliers)

    df.loc[outlier_mask, "topic_sub"] = topics_sub
    df["topic_combined"] = df.apply(
        lambda x: f"sub_{int(x['topic_sub'])}" if x["topic"] == -1 and x["topic_sub"] != -1 else x["topic"],
        axis=1
    )

    return topic_model_sub, topics_sub, df
