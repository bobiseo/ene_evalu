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
    • Splits once into train / test (stratified on `intent`)
    • Exposes helpers that the **CombinedLabelModel** expects
    """

    def __init__(self, X: np.ndarray, df: pd.DataFrame, min_rows_per_class: int = 3):

        # ── keep only rows whose INTENT appears ≥ min_rows_per_class
        keep_mask = df["intent"].isin(
            df["intent"].value_counts()[lambda s: s >= min_rows_per_class].index
        )
        X = X[keep_mask]
        df = df.loc[keep_mask].reset_index(drop=True)
 
        # three target views used by CombinedLabelModel
        self.y_df = df[["intent", "combo_23", "combo_234"]].reset_index(drop=True)
 
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

        # keep embeddings for wrappers that look at the full set
        self.embeddings = X
       
    # ── getters used by combined
    def y_train_df(self):
        return self.y_df.iloc[self.X_train_idx]

    def y_test_df(self):
        return self.y_df.iloc[self.X_test_idx]



# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from Config import *
# import random
# seed =0
# random.seed(seed)
# np.random.seed(seed)

# class Data:
#     """
#     • Holds one canonical feature matrix X (n × D)
#     • Splits into train / test once (stratified on Intent = y2)
#     • Exposes helpers the chain & single-stage models both expect
#     """

#     def __init__(self, X: np.ndarray, df: pd.DataFrame, min_rows_per_class: int = 3):

#         # ── keep only rows whose global y (df["y"]) appears ≥ min_rows_per_class
#         keep_mask = df["y"].isin(
#             df["y"].value_counts()[lambda s: s >= min_rows_per_class].index
#         )
#         X = X[keep_mask]
#         df = df.loc[keep_mask].reset_index(drop=True)

#         # AFTER   (use the already-clean columns you made in preprocess_data)
#         # self.y_df = df[["intent", "tone", "resolution"]].reset_index(drop=True)

#         self.y_df = df[["intent", "combo_23", "combo_234"]].reset_index(drop=True)

#         # ── stratified split once on Intent
#         idx_all = np.arange(len(X))
#         idx_tr, idx_te = train_test_split(
#             idx_all,
#             test_size=0.2,
#             stratify=self.y_df["intent"],
#             random_state=42,
#         )
#         self.X_train_idx, self.X_test_idx = idx_tr, idx_te
#         self.X_train, self.X_test = X[idx_tr], X[idx_te]
#         self.y_train, self.y_test = self.y[idx_tr], self.y[idx_te]

#         # keep embeddings for wrappers that look at the full set
#         self.embeddings = X

#     # ── getters used by ChainModel
#     def y_train_df(self):
#         return self.y_df.iloc[self.X_train_idx]

#     def y_test_df(self):
#         return self.y_df.iloc[self.X_test_idx]
