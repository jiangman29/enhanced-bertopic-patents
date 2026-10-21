import pandas as pd
import re
import spacy
from config import DATA_PATH, STOPWORDS_PATH, stage, stage_year_mapping

def load_data():
    df_all = pd.read_excel(DATA_PATH)[["AD", "TI", "AB"]].dropna()
    year_range = stage_year_mapping[stage]
    df = df_all[(df_all["AD"] >= year_range[0]) & (df_all["AD"] <= year_range[1])].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess_text(title, abstract):
    text = str(title) + " " + str(abstract)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    text = re.sub(r"(novelty|use|advantage|description of drawing\(s\)|detailed description|independent claims|an independent claim|drawing show|summary of invention|background of invention|field of the invention)\s*[-:]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\d+\)", "", text)
    text = re.sub(r"\b(?!(?:3d|4d|3g|4g|5g)\b)(?<!\d)(?<!\d-)\d+(?![a-z])\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\b[a-z]\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text_column(df):
    df["clean_text"] = df.apply(lambda x: preprocess_text(x["TI"], x["AB"]), axis=1)
    return df

def load_stopwords():
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        file_stopwords = [line.strip() for line in f if line.strip()]
    
    custom_stopwords = [
    # 专利通用结构词
    "method", "system", "device", "apparatus", "means", "unit", "module",
    "structure", "mechanism", "element", "circuit", "process", "program",
    # 医疗通用词（你领域的“背景词”）
    "medical",
    # 专利语言常用词
    "includes", "comprises", "provides", "configured", "according", "plurality", "information","s't", "comprises",
    # 通用泛词
    "first", "second", "based", "including", "used", "described", "prior", "known", "one", "another", "other", "said", "may", "can", "will", "having"
    ]
    stopwords = custom_stopwords + file_stopwords
    return stopwords


def load_and_clean_data(stage):
    df = load_data()
    df = clean_text_column(df)
    stopwords = load_stopwords()
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        file_stopwords = [line.strip() for line in f if line.strip()]
    texts = df["clean_text"].tolist()
    return df, texts, stopwords
