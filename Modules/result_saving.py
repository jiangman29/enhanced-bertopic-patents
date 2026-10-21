from config import SAVE_DIR
import re
import os
import pickle
import numpy as np
import pandas as pd
from topic_modeling import has_prediction_data  

def save_topic_summary(topic_model, theme_keyword_dict, topic_labels, df, stage, model_type="main"):
    topic_info = topic_model.get_topic_info().copy()
    topic_info["Representation"] = topic_info["Topic"].map(lambda tid: [kw for kw, _ in theme_keyword_dict.get(tid, [])])
    topic_info["Representative_Docs"] = topic_info["Topic"].map(lambda tid: topic_model.get_representative_docs().get(tid, [])[:3])
    topic_info["Name"] = topic_info["Topic"].map(
        lambda tid: f"{tid}_{topic_labels.get(tid, f'{model_type}_{tid}')}" if tid != -1 else "-1_unknown"
    )

    patent_ids_by_topic = df.groupby("topic")["patent_id"].apply(list).reset_index()
    patent_ids_by_topic.columns = ["Topic", "Patent_IDs"]
    topic_info = topic_info.merge(patent_ids_by_topic, on="Topic", how="left")

    topic_info.to_csv(
        os.path.join(SAVE_DIR, f"topic_summary_{model_type}_stage{stage}.csv"),
        index=False
    )

def save_topic_documents_and_vectors(df, topics, embeddings, stage, model_type="main"):
    df["patent_id"] = df.index
    df[["patent_id", "TI", "AB", "topic"]].to_csv(
        os.path.join(SAVE_DIR, f"stage{stage}_patents_with_topics_{model_type}.csv"),
        index=False
    )
    np.save(os.path.join(SAVE_DIR, f"stage{stage}_document_topics_{model_type}.npy"), topics)
    np.save(os.path.join(SAVE_DIR, f"stage{stage}_document_embeddings_{model_type}.npy"), embeddings)

def save_topic_results(
    topic_model, df, theme_keyword_dict, topic_labels, topic_embeddings,
    topics, embeddings, stage, model_type="main"
):  
    save_topic_summary(topic_model, theme_keyword_dict, topic_labels, df, stage, model_type)
    save_topic_documents_and_vectors(df, topics, embeddings, stage, model_type)

    theme_keywords_df = pd.DataFrame([
        {"topic": tid, "keyword": kw, "similarity": sim}
        for tid, kws in theme_keyword_dict.items()
        for kw, sim in kws
    ])
    theme_keywords_df.to_csv(
        os.path.join(SAVE_DIR, f"stage{stage}_topic_keywords_{model_type}.csv"),
        index=False
    )

    np.save(
        os.path.join(SAVE_DIR, f"stage{stage}_topic_embeddings_{model_type}.npy"),
        topic_embeddings
    )

def save_combined_topic_summary(df, topic_labels, theme_keyword_enhanced, representative_docs_dict, stage, save_dir):
    topic_info = df.groupby("topic_combined").agg({
        "patent_id": list
    }).reset_index()

    topic_info["Count"] = topic_info["patent_id"].map(len)
    topic_info["Name"] = topic_info["topic_combined"].map(lambda tid: topic_labels.get(tid, f"topic_{tid}"))
    topic_info["Representation_Words"] = topic_info["topic_combined"].map(
        lambda tid: [kw for kw, _ in theme_keyword_enhanced.get(tid, [])]
    )
    topic_info["Representation_Documents"] = topic_info["topic_combined"].map(
        lambda tid: representative_docs_dict.get(tid, [])[:5]
    )
    topic_info.rename(columns={"patent_id": "Patent_IDs"}, inplace=True)

    topic_info.to_csv(
        os.path.join(save_dir, f"stage{stage}_final_combined_topic_summary.csv"),
        index=False
    )

    def extract_sort_key(x):
        match = re.match(r"(main|sub)_(\\d+)", str(x))
        if match:
            prefix, num = match.groups()
            return (0 if prefix == "main" else 1, int(num))
        else:
            return (2, 9999)

    topic_info["sort_key"] = topic_info["topic_combined"].map(extract_sort_key)
    topic_info_sorted = topic_info.sort_values("sort_key").drop(columns="sort_key")
    topic_info_sorted.to_csv(
        os.path.join(save_dir, f"stage{stage}_final_combined_topic_summary.csv"),
        index=False
    )
    return topic_info

def save_combined_patent_table(df, stage, save_dir):
    df_patent = df[["patent_id", "TI", "AB", "topic_combined"]].copy()
    df_patent.to_csv(os.path.join(save_dir, f"stage{stage}_patents_with_topics_combined.csv"), index=False)
    return df_patent

def save_combined_topic_embeddings(topic_embeddings_combined, stage, save_dir):
    np.save(os.path.join(save_dir, f"stage{stage}_topic_embeddings_combined.npy"), topic_embeddings_combined)

def build_combined_topic_embeddings(stage, save_dir):
    def load_embeddings_with_fallback(path):
        arr = np.load(path, allow_pickle=True)
        if arr.shape == ():
            return arr.item()
        return arr

    main_embeds = load_embeddings_with_fallback(os.path.join(save_dir, f"stage{stage}_topic_embeddings_main.npy"))
    sub_embeds = load_embeddings_with_fallback(os.path.join(save_dir, f"stage{stage}_topic_embeddings_sub.npy"))

    topic_embeddings_combined = {}

    if isinstance(main_embeds, dict):
        for tid, vec in main_embeds.items():
            topic_embeddings_combined[f"main_{tid}"] = vec
    else:
        for i, vec in enumerate(main_embeds):
            topic_embeddings_combined[f"main_{i}"] = vec

    if isinstance(sub_embeds, dict):
        for tid, vec in sub_embeds.items():
            topic_embeddings_combined[f"sub_{tid}"] = vec
    else:
        for i, vec in enumerate(sub_embeds):
            topic_embeddings_combined[f"sub_{i}"] = vec

    return topic_embeddings_combined

def save_topic_probabilities(topic_model, stage, model_type="main", save_dir=SAVE_DIR):
    if not hasattr(topic_model, "probabilities_") or topic_model.probabilities_ is None:
        print(f"⚠️ {model_type} 模型未包含 probabilities_，可能未启用 calculate_probabilities=True 或未生成。跳过保存。")
        return

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"stage{stage}_topic_probabilities_{model_type}.npy")
    np.save(save_path, topic_model.probabilities_)
    print(f"✅ Saved topic probabilities for {model_type} model at stage {stage} to {save_path}")