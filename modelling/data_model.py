import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data:
    """
    • Holds one canonical feature matrix X (n × D)
    • Splits into train / test once (stratified on Intent = y2)
    • Exposes helpers the chain & single-stage models both expect
    """

    def __init__(self, X: np.ndarray, df: pd.DataFrame, min_rows_per_class: int = 3):

        # ── keep only rows whose global y (df["y"]) appears ≥ min_rows_per_class
        keep_mask = df["y"].isin(
            df["y"].value_counts()[lambda s: s >= min_rows_per_class].index
        )
        X = X[keep_mask]
        df = df.loc[keep_mask].reset_index(drop=True)

        # AFTER   (use the already-clean columns you made in preprocess_data)
        # self.y_df = df[["intent", "tone", "resolution"]].reset_index(drop=True)

        self.y_df = df[["intent", "combo_23", "combo_234"]].reset_index(drop=True)

        # single label (Intent) for legacy models
        self.y = df["intent"].to_numpy()

        # ── stratified split once on Intent
        idx_all = np.arange(len(X))
        idx_tr, idx_te = train_test_split(
            idx_all,
            test_size=0.2,
            stratify=self.y_df["intent"],
            random_state=42,
        )
        self.X_train_idx, self.X_test_idx = idx_tr, idx_te
        self.X_train, self.X_test = X[idx_tr], X[idx_te]
        self.y_train, self.y_test = self.y[idx_tr], self.y[idx_te]

        # keep embeddings for wrappers that look at the full set
        self.embeddings = X

    # ── getters used by ChainModel
    def y_train_df(self):
        return self.y_df.iloc[self.X_train_idx]

    def y_test_df(self):
        return self.y_df.iloc[self.X_test_idx]

    # ── legacy getters (still used by single-stage code paths)
    def get_embeddings(self):       return self.embeddings
    def get_X_train(self):          return self.X_train
    def get_X_test(self):           return self.X_test
    def get_type_y_train(self):     return self.y_train
    def get_type_y_test(self):      return self.y_test
    def get_type(self):             return self.y



# class Data:
#     def __init__(self, X: np.ndarray, df: pd.DataFrame):

#         # ── keep only rows whose y-value occurs ≥3 times ───────────────
#         y_all = df["y"].to_numpy()
#         keep_mask = pd.Series(y_all).isin(
#             pd.Series(y_all).value_counts()[lambda s: s >= 3].index)

#         X_good      = X[keep_mask]
#         y_good      = y_all[keep_mask]ß
#         self.y_df   = df.loc[keep_mask, ["y2", "y3", "y4"]].reset_index(drop=True)

#         # ── one stratified split on Intent (y2) ────────────────────────
#         idx_all = np.arange(len(X_good))
#         idx_tr, idx_te = train_test_split(
#             idx_all,
#             test_size=0.2,
#             stratify=self.y_df["y2"],
#             random_state=42,
#         )

#         # save indices so ChainModel can look them up  ← NEW
#         self.X_train_idx, self.X_test_idx = idx_tr, idx_te

#         # matrices for any single-stage model
#         self.X_train, self.X_test = X_good[idx_tr], X_good[idx_te]
#         self.y_train, self.y_test = y_good[idx_tr], y_good[idx_te]

#         # keep full embeddings as well (some wrappers use it)
#         self.embeddings = X_good


#     def get_type(self):
#         return  self.y
#     def get_X_train(self):
#         return  self.X_train
#     def get_X_test(self):
#         return  self.X_test
#     def get_type_y_train(self):
#         return  self.y_train
#     def get_type_y_test(self):
#         return  self.y_test
#     def get_train_df(self):
#         return  self.train_df
#     def get_embeddings(self):
#         return  self.embeddings
#     def get_type_test_df(self):
#         return  self.test_df
#     def get_X_DL_test(self):
#         return self.X_DL_test
#     def get_X_DL_train(self):
#         return self.X_DL_train


#     def get_y_df_train(self):
#         return self.y_df.iloc[self.X_train_idx]

#     def get_y_df_test(self):
#         return self.y_df.iloc[self.X_test_idx]