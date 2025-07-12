from sklearn.ensemble import RandomForestClassifier
from sklearn.base      import clone
from model.base        import BaseModel
from sklearn.metrics   import classification_report


class CombinedLabelModel(BaseModel):
    """
    ONE RandomForest template
    fitted three times on progressively aggregated targets:
        ① intent
        ② combo_23   (intent + tone)
        ③ combo_234  (intent + tone + resolution)
    """
    name = "combined"

    def __init__(self, **rf_params):
        super().__init__()
        self.rf_template = RandomForestClassifier(
            class_weight="balanced",  # handles imbalance
            **rf_params
        )
        self.models = {}  # dict: target_name -> fitted RF

    # --------- abstract hook 1 ----------
    def data_transform(self, data):
        """No feature engineering needed; just return X unchanged."""
        return data.X_train, data.X_test

    # --------- abstract hook 2 ----------
    def train(self, data):
        """
        Fit three copies of the RF template on
          - intent
          - combo_23
          - combo_234
        """
        X_tr, _ = self.data_transform(data)
        y_tr_df = data.y_train_df()        # method already exists in Data

        for tgt in ["intent", "combo_23", "combo_234"]:
            m = clone(self.rf_template)
            m.fit(X_tr, y_tr_df[tgt])
            self.models[tgt] = m
        return self

    # ---------- inference ---------------
    def predict(self, X):
        return {t: m.predict(X) for t, m in self.models.items()}

    # ---------- pretty report -----------
    def print_results(self, data):
        _, X_te = self.data_transform(data)
        y_te_df = data.y_test_df()

        print("\n--- Intent ---")
        print(classification_report(
            y_te_df["intent"],
            self.models["intent"].predict(X_te)
        ))

        print("\n--- Intent + Tone (combo_23) ---")
        print(classification_report(
            y_te_df["combo_23"],
            self.models["combo_23"].predict(X_te)
        ))

        print("\n--- Intent + Tone + Resolution (combo_234) ---")
        print(classification_report(
            y_te_df["combo_234"],
            self.models["combo_234"].predict(X_te)
        ))
