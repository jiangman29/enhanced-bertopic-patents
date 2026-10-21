# visualization.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from config import SAVE_DIR
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_topic_model_with_custom_keywords_and_labels_v2(model_path, keyword_csv_path, summary_csv_path, model_type="main"):
    import pickle
    import pandas as pd

    # 1. 加载模型
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. 注入关键词（用于 barchart，避免 visualize_documents 报错）
    df_keywords = pd.read_csv(keyword_csv_path)
    topic_keywords_dict = {
        int(topic): [(kw, float(sim)) for kw, sim in zip(group["keyword"], group["similarity"])]
        for topic, group in df_keywords.groupby("topic")
    }

    # ✅ 补齐缺失主题
    all_topic_ids = [tid for tid in model.get_topics().keys() if tid != -1]
    for tid in all_topic_ids:
        if tid not in topic_keywords_dict:
            topic_keywords_dict[tid] = [("placeholder", 0.0)]

    model.topic_representations_ = topic_keywords_dict

    # 3. 注入 AI 命名（来自 final_combined_topic_summary）
    df_summary = pd.read_csv(summary_csv_path)
    topic_labels = {}
    for _, row in df_summary.iterrows():
        tid_full = str(row["topic_combined"])
        if not tid_full.startswith(model_type):
            continue
        try:
            tid = int(tid_full.replace(f"{model_type}_", ""))
            topic_labels[tid] = row["Name"]
        except Exception as e:
            print(f"⚠️ 无效命名跳过：{tid_full} - {e}")

    model.set_topic_labels(topic_labels)
    model.custom_labels_ = topic_labels  # 供条形图使用

    return model




def visualize_barchart_grid(topic_model, top_n_topics=8, cols=2, top_n_words=5, save_path=None):
    import math
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    topic_words = topic_model.get_topics()
    topic_ids = [tid for tid in topic_words.keys() if tid != -1][:top_n_topics]

    rows = math.ceil(len(topic_ids) / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Topic {tid}" for tid in topic_ids],
        horizontal_spacing=0.2  # ✅ 增大水平间距，默认是 0.1
    )

    for idx, tid in enumerate(topic_ids):
        row = idx // cols + 1
        col = idx % cols + 1
        words = topic_words[tid][:top_n_words]
        words_list = [w for w, _ in words][::-1]
        scores = [s for _, s in words][::-1]

        fig.add_trace(
            go.Bar(
                x=scores,
                y=words_list,
                orientation='h',
                name=f"Topic {tid}",
                showlegend=False
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        height=300 * rows,
        width=400 * cols,  # 可略增宽
        title_text="Top Keywords per Topic",
        margin=dict(t=80),  # ⬆️ 增加顶部空间
        font=dict(size=12)  # ✅ 整体字体稍调小，避免文字挤
    )

    if save_path:
        pio.write_html(fig, save_path)
        print(f"✅ Saved topic bar chart grid to {save_path}")
    
    return fig




def save_visualizations(topic_model, texts, embeddings, stage, model_type="main", save_dir="/content/drive/My Drive/Colab Notebooks/Pictures"):
    import os
    import plotly.io as pio
    from visualization import visualize_barchart_grid

    os.makedirs(save_dir, exist_ok=True)

    # ✅ 设置主题标签（使用 AI 命名）
    if hasattr(topic_model, "custom_labels_"):
        try:
            topic_model.set_topic_labels(topic_model.custom_labels_)
        except Exception as e:
            print(f"⚠️ set_topic_labels 设置失败: {e}")

    # ✅ 补齐必要字段
    if not hasattr(topic_model, "docs_") or not isinstance(topic_model.docs_, list):
        topic_model.docs_ = texts
    if not hasattr(topic_model, "topics_") or not isinstance(topic_model.topics_, list):
        try:
            topic_model.topics_ = topic_model.transform(texts)[0]
        except Exception as e:
            print(f"⚠️ Failed to infer topics_: {e}")
            topic_model.topics_ = [-1] * len(texts)

    # ✅ 条形图：关键词展示
    try:
        barchart_path = f"{save_dir}/stage{stage}_topic_barchart_{model_type}.html"
        visualize_barchart_grid(topic_model, save_path=barchart_path, top_n_words=10)
    except Exception as e:
        print(f"❌ Failed to generate barchart for {model_type} model: {e}")

    # ✅ 文档嵌入图（UMAP）+ 使用 AI 命名
    try:
        doc_scatter = topic_model.visualize_documents(texts, embeddings=embeddings)
        path_doc = f"{save_dir}/stage{stage}_topic_documents_{model_type}.html"
        pio.write_html(doc_scatter, path_doc)
        print(f"✅ Saved document UMAP for {model_type} to {path_doc}")
    except Exception as e:
        print(f"❌ Failed to generate document scatter for {model_type} model: {e}")

    # ✅ 气泡图：LDA 风格
    try:
        bubble_chart = topic_model.visualize_topics()
        path_bubble = f"{save_dir}/stage{stage}_topic_bubbles_{model_type}.html"
        pio.write_html(bubble_chart, path_bubble)
        print(f"✅ Saved bubble chart for {model_type} to {path_bubble}")
    except Exception as e:
        print(f"❌ Failed to generate bubble chart for {model_type} model: {e}")

    print(f"📊 Visualizations saved for {model_type} model (Stage {stage})!")











def plot_topic_similarity_heatmap(topic_model, stage, model_type="main", save_dir=SAVE_DIR):
    """
    可视化主题嵌入的余弦相似度热力图。
    如果 topic_model.topic_embeddings_ 存在，则绘图；否则跳过。

    参数：
        topic_model: BERTopic 模型对象
        stage: int，当前阶段编号
        model_type: str，模型类型（main / sub / combined）
    """
    os.makedirs(f"{save_dir}/Pictures", exist_ok=True)  # 🔧 确保目录存在
    
    if hasattr(topic_model, "topic_embeddings_") and topic_model.topic_embeddings_ is not None:
        topic_vectors = topic_model.topic_embeddings_
        similarity_matrix = cosine_similarity(topic_vectors)

        topic_labels = [f"{model_type}_{i}" for i in range(len(topic_vectors))]
        similarity_df = pd.DataFrame(similarity_matrix, index=topic_labels, columns=topic_labels)

        plt.figure(figsize=(14, 12))
        sns.heatmap(similarity_df, annot=False, cmap="YlGnBu", square=True, cbar_kws={"shrink": .6})
        plt.title(f"Topic Similarity Heatmap ({model_type.capitalize()}, Stage {stage})", fontsize=18)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/Pictures/stage{stage}_topic_similarity_heatmap_{model_type}.png")
        plt.close()
        print(f"✅ Saved heatmap for {model_type} model at stage {stage}")
    else:
        print(f"⚠️ {model_type} model has no topic_embeddings_. Skipped heatmap.")





def load_model_and_prepare_for_visualization(stage, model_type="main"):
    """
    加载 BERTopic 模型，并手动补充 visualize_documents 所需的依赖属性：docs_ 和 topics_。

    参数：
    - stage: int, 阶段编号
    - model_type: str, "main" 或 "sub"

    返回：
    - topic_model: BERTopic 对象，含补全属性
    - texts: 文本列表
    - embeddings: 嵌入向量（np.ndarray）
    """
    # 🔹 加载专利数据
    df, _, _ = load_and_clean_data(stage)
    df["patent_id"] = df.index

    # 🔹 主模型 or 子模型
    if model_type == "main":
        topic_path = f"{SAVE_DIR}/stage{stage}_document_topics_main.npy"
        embedding_path = f"{SAVE_DIR}/stage{stage}_embeddings.npy"
        model_path = f"{SAVE_DIR}/stage{stage}_topic_model.pkl"
        mask = df["topic"] != -1
    elif model_type == "sub":
        # 先加载主模型主题编号，构造 mask
        df["topic"] = np.load(f"{SAVE_DIR}/stage{stage}_document_topics_main.npy")
        mask = df["topic"] == -1

        topic_path = f"{SAVE_DIR}/stage{stage}_document_topics_sub.npy"
        embedding_path = f"{SAVE_DIR}/stage{stage}_embeddings.npy"
        model_path = f"{SAVE_DIR}/stage{stage}_topic_model_sub.pkl"

    else:
        raise ValueError("model_type 必须是 'main' 或 'sub'")

    # 🔹 加载模型与数据
    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)

    topics = np.load(topic_path)
    embeddings = np.load(embedding_path)
    texts = df.loc[mask, "clean_text"].tolist()
    embeddings = embeddings[mask.to_numpy()]

    # 🔹 手动补充可视化依赖
    topic_model.docs_ = texts
    topic_model.topics_ = topics.tolist()
    topic_model._embedding_model = None

    return topic_model, texts, embeddings



def load_topic_model_with_custom_keywords(model_path, keyword_csv_path):
    """
    加载 BERTopic 模型并注入自定义关键词（用于可视化）
    """
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df_keywords = pd.read_csv(keyword_csv_path)
    topic_keywords_dict = {
        int(topic): [(kw, 1.0) for kw in group["keyword"].tolist()[:10]]
        for topic, group in df_keywords.groupby("topic")
    }
    model.topic_representations_ = topic_keywords_dict
    return model

