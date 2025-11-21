from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from without_api.agent.utils import split_sentences, clean_whitespace


def extractive_summary(text: str, n_sentences: int = 6) -> List[str]:
    """Simple extractive summarizer using TF-IDF + sentence similarity ranking.

    Returns the top `n_sentences` in their original order.
    """
    text = clean_whitespace(text)
    sents = split_sentences(text)
    if not sents:
        return []
    if len(sents) <= n_sentences:
        return sents

    # Vectorize sentences
    vect = TfidfVectorizer(stop_words="english")
    tfidf = vect.fit_transform(sents)

    # Similarity matrix and sentence scores (sum of similarities)
    sim = cosine_similarity(tfidf)
    scores = sim.sum(axis=1)

    # Pick top sentences
    top_idx = np.argsort(scores)[-n_sentences:]
    top_idx_sorted = sorted(top_idx)
    return [sents[i] for i in top_idx_sorted]



def keyword_summary(text: str, n_keywords: int = 10) -> List[str]:
    # Lightweight TF-IDF keywords via vectorizer feature names
    text = clean_whitespace(text)
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=2000)
    tfidf = vect.fit_transform([text])
    feature_array = np.array(vect.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:n_keywords]
    return top_n.tolist()
