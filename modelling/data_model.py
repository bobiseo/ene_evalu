import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
from collections import Counter

from imblearn.over_sampling import SMOTE

seed = 0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self, X: np.ndarray, df: pd.DataFrame, label_column: str):
        self.label_column = label_column
        y_series = df[self.label_column]
        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index

        if len(good_y_value) < 2:
            print(f"Not enough classes in {self.label_column} to apply SMOTE. Skipping this model.")
            self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
            self.classes = []
            return

        y_good_mask = y_series.isin(good_y_value)
        self.df = df[y_good_mask]
        X_good = X[y_good_mask.to_numpy()]
        y_good = y_series[y_good_mask]
        
        self.embeddings = X_good
        self.y = y_good.to_numpy()
        self.classes = good_y_value

        if X_good.shape[0] > 0:
            new_test_size = min(0.2, 1.0 * X.shape[0] * 0.2 / X_good.shape[0]) if X_good.shape[0] > 5 else 0
            
            if new_test_size > 0:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X_good, y_good.to_numpy(), test_size=new_test_size, random_state=seed, stratify=y_good.to_numpy()
                )
                
                print("Original training set distribution: %s" % Counter(self.y_train))
                
                min_class_size = min(Counter(self.y_train).values())
                
                k_neighbors_value = min(5, min_class_size - 1) if min_class_size > 1 else 1

                if k_neighbors_value > 0:
                    sm = SMOTE(random_state=seed, k_neighbors=k_neighbors_value)
                    self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)
                    print("Resampled training set distribution: %s" % Counter(self.y_train))
                else:
                    print("Skipping SMOTE: Not enough samples in the minority class.")
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = X_good, np.array([]), y_good.to_numpy(), np.array([])
        else:
             self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
             self.classes = []

    def get_type(self):
        return self.y
    def get_X_train(self):
        return self.X_train
    
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df
    def get_X_DL_test(self):
        return self.X_DL_test
    def get_X_DL_train(self):
        return self.X_DL_train

