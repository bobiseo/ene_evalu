# model/chained_lr.py
from sklearn.linear_model     import LogisticRegression
from sklearn.base              import clone
from sklearn.metrics           import classification_report
from model.base                import BaseModel

class ChainedLRModel(BaseModel):
    """
    Chain three LogisticRegression classifiers:
      1) intent
      2) combo_23   (intent + tone)
      3) combo_234  (intent + tone + resolution)
    """
    def __init__(self, **lr_params):
        super().__init__()
        # elastic-net or l2 by default—tweak penalty/C as needed
        self.lr_template = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            **lr_params
        )
        self.models = {}

    def train(self, data):
        X_tr = data.X_train       # attribute
        y_df = data.y_train_df()
        for tgt in ["intent", "combo_23", "combo_234"]:
            m = clone(self.lr_template)
            m.fit(X_tr, y_df[tgt])
            self.models[tgt] = m

    def predict(self, X):
        return {t: m.predict(X) for t, m in self.models.items()}

    def print_results(self, data):
        X_te, y_te = data.X_test, data.y_test_df()
        print("\n--- Intent (LR) ---")
        print(classification_report(y_te["intent"],
                                    self.models["intent"].predict(X_te)))
        print("\n--- Intent + Tone (LR) ---")
        print(classification_report(y_te["combo_23"],
                                    self.models["combo_23"].predict(X_te)))
        print("\n--- Intent + Tone + Res (LR) ---")
        print(classification_report(y_te["combo_234"],
                                    self.models["combo_234"].predict(X_te)))
