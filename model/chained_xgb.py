# chained_xgboost_model.py
import xgboost as xgb
from sklearn.base      import clone
from model.base        import BaseModel
from sklearn.metrics import classification_report


# from sklearn.metrics   as skm

class ChainedLabelXGBModel(BaseModel):
    """
    ONE XGBoost template fitted three times:
        ① intent
        ② combo_23   (intent | tone)
        ③ combo_234  (intent | tone | resolution)
    """
    name = "chained_xgboost"

    def __init__(self, **xgb_params):
        super().__init__()
        # sensible defaults for sparse TF-IDF features
        self.xgb_template = xgb.XGBClassifier(
            objective="multi:softprob",
            tree_method="hist",
            max_depth=6,
            n_estimators=600,
            learning_rate=0.15,
            colsample_bytree=0.6,
            subsample=0.8,
            eval_metric="mlogloss",
            n_jobs=-1,
            **xgb_params
        )
        self.models = {}          # target_name → fitted XGB

    # ---------- abstract hook 1 ----------
    def data_transform(self, data):
        """Return sparse TF-IDF matrix unchanged."""
        return data.X_train, data.X_test

    # ---------- abstract hook 2 ----------
    def train(self, data):
        X_tr, _  = self.data_transform(data)
        y_tr_df  = data.y_train_df()

        for tgt in ["intent", "combo_23", "combo_234"]:
            m = clone(self.xgb_template)
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
