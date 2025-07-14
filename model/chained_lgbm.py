# model/chained_lgbm.py
from lightgbm import LGBMClassifier
from model.base  import BaseModel
from sklearn.base import clone
from sklearn.metrics import classification_report

class ChainedLGCBMModel(BaseModel):
    """
    Trains one LightGBM per chained target:
      1) intent
      2) combo_23
      3) combo_234
    """
    def __init__(self, **lgbm_params):
        super().__init__()
        # defaults: you can override via lgbm_params
        self.lgbm_template = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=0,
            verbose=-1,
            **lgbm_params
        )
        self.models = {}

    def train(self, data) -> None:
        X_tr = data.X_train      # attribute style
        y_df = data.y_train_df()

        for tgt in ["intent", "combo_23", "combo_234"]:
            m = clone(self.lgbm_template)
            m.fit(X_tr, y_df[tgt])
            self.models[tgt] = m

    def predict(self, X):
        return {t: m.predict(X) for t, m in self.models.items()}

    def print_results(self, data) -> None:
        X_te, y_te = data.X_test, data.y_test_df()
        print("\n--- Intent (LGBM) ---")
        print(classification_report(y_te["intent"],
                                    self.models["intent"].predict(X_te)))
        print("\n--- Intent + Tone (LGBM)---")
        print(classification_report(y_te["combo_23"],
                                    self.models["combo_23"].predict(X_te)))
        print("\n--- Intent + Tone + Resolution (LGBM)---")
        print(classification_report(y_te["combo_234"],
                                    self.models["combo_234"].predict(X_te)))
