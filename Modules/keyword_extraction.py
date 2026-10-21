import os
import pandas as pd
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from config import SAVE_DIR

def extract_keywords_and_embeddings(
    topic_model, kw_model, embedding_model, df, stage, stopwords,
    model_type="main", top_k_terms=100, top_k_docs=5, top_n_keywords=15
):
    abbreviation_whitelist = {
        "ecg", "his", "emr", "ehr", "ct", "mri", "pacs", "ris",
        "hrm", "cdss", "fhir", "hl7", "dicom", "lis", "eicu",
        "teleradiology", "cpoe", "ai", "iot"
    }

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    def clean_and_lemmatize(term):
        doc = nlp(term.lower())
        lemmas = [token.lemma_ for token in doc if token.lower_ not in {"the", "an", "a"}]
        return " ".join(lemmas)

    def strip_edge_stopwords(phrase, stopword_set):
        words = phrase.split()
        while words and words[0] in stopword_set and words[0] not in abbreviation_whitelist:
            words.pop(0)
        while words and words[-1] in stopword_set and words[-1] not in abbreviation_whitelist:
            words.pop()
        return " ".join(words)

    def normalize_phrase_order(phrase):
        words = phrase.lower().split()
        return " ".join(sorted(words))

    def deduplicate_phrases(phrases):
        seen = set()
        unique = []
        for phrase in phrases:
            norm = normalize_phrase_order(phrase)
            if norm not in seen:
                seen.add(norm)
                unique.append(phrase)
        return unique

    def get_ctfidf_candidates(topic_id):
        topic_words = topic_model.get_topic(topic_id)
        candidates = []
        for word, _ in topic_words[:top_k_terms]:
            if 1 <= len(word.split()) <= 3 or word.lower() in abbreviation_whitelist:
                lemma = clean_and_lemmatize(word)
                if lemma:
                    candidates.append(lemma)
        return candidates

    def get_keybert_noun_chunk_terms(text):
        keybert_terms = set()
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words=stopwords,
            top_n=top_k_terms
        )
        for kw, _ in keywords:
            phrase = clean_and_lemmatize(kw.lower().strip())
            if phrase and (1 <= len(phrase.split()) <= 3 or phrase in abbreviation_whitelist):
                keybert_terms.add(phrase)

        doc = nlp(text.lower())
        noun_phrases = set()
        for chunk in doc.noun_chunks:
            phrase = clean_and_lemmatize(chunk.text.lower().strip())
            if phrase and (1 <= len(phrase.split()) <= 3 or phrase in abbreviation_whitelist):
                noun_phrases.add(phrase)

        return keybert_terms.union(noun_phrases)

    def get_semantic_top_terms(topic_id):
        rep_docs = topic_model.get_representative_docs().get(topic_id, [])[:top_k_docs]
        if not rep_docs:
            return []

        text_block = " ".join(rep_docs)
        candidates_ctfidf = get_ctfidf_candidates(topic_id)
        candidates_keybert = get_keybert_noun_chunk_terms(text_block)
        candidates = list(set(candidates_ctfidf).union(candidates_keybert).union(abbreviation_whitelist))

        candidates = [strip_edge_stopwords(term, set(stopwords)) for term in candidates]
        candidates = [term for term in candidates if 1 <= len(term.split()) <= 3]
        candidates = deduplicate_phrases(candidates)

        if not candidates:
            return []

        doc_embeds = embedding_model.encode(rep_docs)
        term_embeds = embedding_model.encode(candidates)
        topic_vector = np.mean(doc_embeds, axis=0).reshape(1, -1)
        sims = cosine_similarity(term_embeds, topic_vector).flatten()
        all_terms_sorted = sorted(zip(candidates, sims), key=lambda x: -x[1])

        return all_terms_sorted[:top_n_keywords]

    print(f"\U0001F4DA Extracting keywords for {model_type} model...")
    theme_keyword = {}
    for tid in topic_model.get_topics().keys():
        if tid == -1:
            continue
        print(f"\U0001F9E9 Topic {tid}...", end=" ")
        top_keywords = get_semantic_top_terms(tid)
        theme_keyword[tid] = top_keywords
        print(f"✅ {len(top_keywords)} terms")

    theme_keywords_df = pd.DataFrame([
        {"topic": tid, "keyword": kw, "similarity": sim}
        for tid, kws in theme_keyword.items()
        for kw, sim in kws
    ])
    theme_keywords_df.to_csv(os.path.join(SAVE_DIR, f"stage{stage}_topic_keywords_{model_type}.csv"), index=False)

    topic_embeddings = {
        tid: np.mean(embedding_model.encode([" ".join([kw for kw, _ in theme_keyword[tid]])]), axis=0)
        for tid in theme_keyword
    }
    np.save(os.path.join(SAVE_DIR, f"stage{stage}_topic_embeddings_{model_type}.npy"), topic_embeddings)

    return theme_keyword, topic_embeddings

extract_keywords = extract_keywords_and_embeddings
