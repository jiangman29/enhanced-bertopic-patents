from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics import silhouette_score
import numpy as np
from topic_modeling import has_prediction_data

def get_outlier_scores(topic_model):
    if not has_prediction_data(topic_model):
        print("⚠️ 当前模型未启用 prediction_data，跳过离群分数")
        return None
    try:
        return topic_model.get_outlier_scores()
    except Exception as e:
        print(f"⚠️ 获取离群分数失败: {e}")
        return None


def compute_coherence(topic_model, texts, labels):
    # 仅保留非离群点的文本和对应标签
    filtered = [(text.split(), label) for text, label in zip(texts, labels) if label != -1]
    tokenized_texts, valid_labels = zip(*filtered)

    # 获取所有主题词（排除 -1）
    topic_words_dict = topic_model.get_topics()
    topic_words = [
        [word for word, _ in topic_words_dict[tid]]
        for tid in topic_words_dict if tid != -1
    ]

    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    return coherence_model.get_coherence()


def compute_silhouette(embeddings, labels):
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    mask = labels != -1
    return silhouette_score(embeddings[mask], labels[mask], metric="cosine")

    

def evaluate_topic_model(topic_model, df, embeddings, stage, model_type="main"):
    print(f"\n📊 Evaluating {model_type} model...")

    texts = df["clean_text"].tolist()
    labels = df["topic"] if model_type == "main" else df["topic_sub"]
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    # ✅ 一致性得分（默认关键词）
    coherence_default = compute_coherence(topic_model, texts, labels)
    print(f"🎯Coherence Score: {coherence_default:.4f}")

    # ✅ Silhouette 分数（主题嵌入结构清晰度）
    silhouette = compute_silhouette(embeddings, labels)
    print(f"📐 Silhouette Score: {silhouette:.4f}")

