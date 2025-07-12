import numpy as np
import pandas as pd
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embd(df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(
        max_features=2000, min_df=4, max_df=0.90, sublinear_tf=True
    )
    # pre-merged column
    data = df["text"]

    # Optional: append upstream predictions if they exist
    if "predicted_intent" in df.columns:
        data = df["predicted_intent"].astype(str) + " " + data
    if "predicted_tone" in df.columns:
        data = df["predicted_tone"].astype(str) + " " + data

    X = tfidfconverter.fit_transform(data).toarray()
    return X


def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)

