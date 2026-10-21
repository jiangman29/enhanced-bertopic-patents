from sentence_transformers import SentenceTransformer
import sys
import os


# 阶段设置
stage = "all" # 可在主控中改为 2, 3, 4

# 阶段对应年份
stage_year_mapping = {
    1: (2015, 2017),
    2: (2018, 2020),
    3: (2021, 2022),
    4: (2023, 2024),
    "all": (2015, 2024)
}

# 嵌入模型选择
embedding_model_name = "mpi-inno-comp/paecter"

# ========== 主模型参数 ==========
umap_main_config = {
    "n_neighbors": 50,
    "n_components": 10,
    "min_dist": 0.07,
    "metric": "cosine",
    "random_state": 42
}

hdbscan_main_config = {
    "min_cluster_size": 40,
    "min_samples": 15,
    "cluster_selection_epsilon": 0.1,
    "cluster_selection_method": "leaf",
    "metric": "euclidean",
    "prediction_data": True
}

# ========== 子模型参数 ==========
umap_sub_config = {
    "n_neighbors": 30,
    "n_components": 10,
    "min_dist": 0.07,
    "metric": "cosine",
    "random_state": 42
}

hdbscan_sub_config = {
    "min_cluster_size": 20,
    "min_samples": 10,
    "cluster_selection_epsilon": 0.1,
    "cluster_selection_method": "leaf",
    "metric": "euclidean",
    "prediction_data": True  # ✅ 已更新
}

# ========== 主子模型聚合 ==========
MAIN_MODEL_PARAMS = {
    "embedding_model_name": embedding_model_name,
    "umap_config": umap_main_config,
    "hdbscan_config": hdbscan_main_config,
}

SUB_MODEL_PARAMS = {
    "embedding_model_name": embedding_model_name,
    "umap_config": umap_sub_config,
    "hdbscan_config": hdbscan_sub_config,
}

# 判断是否在 Google Colab 环境
if "google.colab" in sys.modules:
    DRIVE_PREFIX = "/content/drive/My Drive/Colab Notebooks/Bertopic"
else:
    # ✅ 本地路径（虚拟机磁盘）
    DRIVE_PREFIX = os.path.expanduser("~/local_results/Bertopic")

# ========== 文件路径配置 ==========
DATA_PATH = f"{DRIVE_PREFIX}/Data/all_patents.xlsx"
SAVE_DIR = f"{DRIVE_PREFIX}/Result"
STOPWORDS_PATH = f"{DRIVE_PREFIX}/Data/US_stopwords.txt"