import numpy as np
import pandas as pd
from Config import *
import random
from sklearn.feature_extraction.text import TfidfVectorizer

seed = 0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embd(df:pd.DataFrame):
    min_df_value = min(4, len(df))
    
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=min_df_value, max_df=0.90)
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    
    try:
        X = tfidfconverter.fit_transform(data).toarray()
    except ValueError:
        print("Warning: Could not build a vocabulary for this subset. Returning zero vectors.")
        X = np.zeros((len(data), tfidfconverter.max_features))
        
    return X, tfidfconverter

def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)