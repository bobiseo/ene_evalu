import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_embd(df) -> np.ndarray:
    """
    Returns an n×D TF-IDF embedding matrix for df['text'].
    """
    vect = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1,2),
        stop_words="english"
    )
    X = vect.fit_transform(df["text"]).toarray()
    return X

