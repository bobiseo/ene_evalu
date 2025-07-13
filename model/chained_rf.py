# model/chained_rf.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.base      import clone
from sklearn.metrics   import classification_report
from model.base        import BaseModel

class ChainedLabelModel(BaseModel):
    """
    Trains one RandomForest per chained target:
      1) intent
      2) combo_23   (intent + tone)
      3) combo_234  (intent + tone + resolution)
    """
    name = "chained_rf"

    def __init__(self, **rf_params):
        super().__init__()
        # default to balanced RF; override via rf_params
        self.rf_template = RandomForestClassifier(
            class_weight="balanced",
            **rf_params
        )
        self.models = {}

    def train(self, data) -> None:
        """
        Fits three RandomForest models on the training portion of `data`.
        """
        # extract training features and labels
        X_tr = data.X_train      # assumes attribute-based API
        y_tr_df = data.y_train_df()

        for tgt in ["intent", "combo_23", "combo_234"]:
            m = clone(self.rf_template)
            m.fit(X_tr, y_tr_df[tgt])
            self.models[tgt] = m

    def predict(self, X):
        """
        Predict on feature matrix X.
        Returns a dict mapping each target to its predictions.
        """
        return {t: m.predict(X) for t, m in self.models.items()}

    def print_results(self, data) -> None:
        """
        Prints classification reports for each chained target using the test split in `data`.
        """
        X_te = data.X_test
        y_te_df = data.y_test_df()

        print("\n--- Intent ---")
        print(classification_report(y_te_df["intent"],
                                    self.models["intent"].predict(X_te)))

        print("\n--- Intent + Tone (combo_23) ---")
        print(classification_report(y_te_df["combo_23"],
                                    self.models["combo_23"].predict(X_te)))

        print("\n--- Intent + Tone + Resolution (combo_234) ---")
        print(classification_report(y_te_df["combo_234"],
                                    self.models["combo_234"].predict(X_te)))


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.base      import clone
# from model.base        import BaseModel
# from sklearn.metrics   import classification_report


# class ChainedLabelModel(BaseModel):
#     """
#     ONE RandomForest template
#     fitted three times on progressively aggregated targets:
#         ① intent
#         ② combo_23   (intent + tone)
#         ③ combo_234  (intent + tone + resolution)
#     """
#     name = "chained_rf"

#     def __init__(self, **rf_params):
#         super().__init__()
#         self.rf_template = RandomForestClassifier(
#             class_weight="balanced",  # handles imbalance
#             **rf_params
#         )
#         self.models = {}  # dict: target_name -> fitted RF

#     def train(self, data):
#         """
#         Fit three copies of the RF template on
#           - intent
#           - combo_23
#           - combo_234
#         """
#         X_tr, _ = self.data_transform(data)
#         y_tr_df = data.y_train_df()        # method already exists in Data

#         for tgt in ["intent", "combo_23", "combo_234"]:
#             m = clone(self.rf_template)
#             m.fit(X_tr, y_tr_df[tgt])
#             self.models[tgt] = m
#         return self


#     def predict(self, X):
#         return {t: m.predict(X) for t, m in self.models.items()}


#     def print_results(self, data):
#         _, X_te = self.data_transform(data)
#         y_te_df = data.y_test_df()

#         print("\n--- Intent ---")
#         print(classification_report(
#             y_te_df["intent"],
#             self.models["intent"].predict(X_te)
#         ))

#         print("\n--- Intent + Tone (combo_23) ---")
#         print(classification_report(
#             y_te_df["combo_23"],
#             self.models["combo_23"].predict(X_te)
#         ))

#         print("\n--- Intent + Tone + Resolution (combo_234) ---")
#         print(classification_report(
#             y_te_df["combo_234"],
#             self.models["combo_234"].predict(X_te)
#         ))