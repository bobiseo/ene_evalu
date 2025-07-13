import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random

num_folds = 0
seed =0

np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None
        self.probabilities = None

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        if X_test.shape[0] > 0:
            self.predictions = self.mdl.predict(X_test)
        else:
            self.predictions = np.array([])

    def predict_proba(self, X_test: pd.Series):
        if X_test.shape[0] > 0:
            self.probabilities = self.mdl.predict_proba(X_test)
        else:
            self.probabilities = np.array([])

    def print_results(self, data):
        if self.predictions is not None and data.y_test is not None and len(data.y_test) > 0:
            print(f"Results for {self.model_name}:")
            print(classification_report(data.y_test, self.predictions, labels=self.mdl.classes_, zero_division=0))
            
            print("Confusion Matrix:")
            print(confusion_matrix(data.y_test, self.predictions, labels=self.mdl.classes_))

