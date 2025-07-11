import numpy as np
import pandas as pd
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embd(df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)

    # basic text input
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]

    # Reflect predicted intent/tone in features
    if "predicted_intent" in df.columns:
        data = df["predicted_intent"].astype(str) + ' ' + data

    if "predicted_tone" in df.columns:
        data = df["predicted_tone"].astype(str) + ' ' + data

    X = tfidfconverter.fit_transform(data).toarray()
    return X


# def get_tfidf_embd(df:pd.DataFrame):
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
#     data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
#     X = tfidfconverter.fit_transform(data).toarray()
#     return X

def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)

