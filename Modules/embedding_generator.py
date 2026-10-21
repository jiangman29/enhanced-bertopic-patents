import numpy as np
from sentence_transformers import SentenceTransformer
from config import embedding_model_name, SAVE_DIR

# 每次使用时都加载模型（可控、清晰）
def generate_embeddings(texts, embedding_model, stage):
    embeddings = embedding_model.encode(texts, batch_size=64, show_progress_bar=True)
    np.save(f"{SAVE_DIR}/stage{stage}_embeddings.npy", embeddings)
    return embeddings


def load_embeddings(stage):
    return np.load(f"{SAVE_DIR}/stage{stage}_embeddings.npy")
