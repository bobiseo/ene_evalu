import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config

class Data:
    """
    Encapsulates feature matrix X and label DataFrame df,
    applies a single stratified train/test split on `intent`,
    and provides attribute-based accessors compatible with existing pipeline.
    """
    def __init__(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 0,
        min_rows_per_class: int = 3,
    ):
        # 1. filter out rare intents
        intent_counts = df['intent'].value_counts()
        keep_intents = intent_counts[intent_counts >= min_rows_per_class].index
        mask = df['intent'].isin(keep_intents)

        # store filtered DataFrame and features
        self.df = df.loc[mask].reset_index(drop=True)
        self.X  = X[mask]

        # 2. stratified train/test split
        indices = np.arange(self.X.shape[0])
        idx_tr, idx_te = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['intent'],
        )
        # attribute-based splits for compatibility
        self.X_train_idx = idx_tr
        self.X_test_idx  = idx_te
        self.X_train     = self.X[idx_tr]
        self.X_test      = self.X[idx_te]

    def y_train_df(self) -> pd.DataFrame:
        """Returns training labels for intent, combo_23, and combo_234."""
        return self.df.iloc[self.X_train_idx][['intent', 'combo_23', 'combo_234']]

    def y_test_df(self) -> pd.DataFrame:
        """Returns test labels for intent, combo_23, and combo_234."""
        return self.df.iloc[self.X_test_idx][['intent', 'combo_23', 'combo_234']]


