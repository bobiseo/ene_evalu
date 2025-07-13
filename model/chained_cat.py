# model/chained_cat.py
from catboost import CatBoostClassifier
from sklearn.base    import clone
from sklearn.metrics import classification_report
from model.base      import BaseModel

class ChainedCatBoostModel(BaseModel):
    """
    Chain three CatBoost classifiers on TF-IDF features:
      1) intent
      2) combo_23   (intent + tone)
      3) combo_234  (intent + tone + resolution)
    """

    def __init__(self, **cb_params):
        super().__init__()
        # default params; you can override via cb_params
        self.cb_template = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            verbose=False,
            **cb_params
        )
        self.models = {}

    def train(self, data) -> None:
        X_tr = data.X_train       # NumPy array of TF-IDF
        y_df = data.y_train_df()
        for tgt in ["intent", "combo_23", "combo_234"]:
            m = clone(self.cb_template)
            m.fit(X_tr, y_df[tgt])
            self.models[tgt] = m

    # def predict(self, X):
    #     # CatBoost returns numpy arrays for .predict
    #     return {t: m.predict(X) for t, m in self.models.items()}
    #     # return {t: m.predict(X).astype(str).tolist() for t, m in self.models.items()}


    def predict(self, X):
        out = {}
        for tgt, m in self.models.items():
            arr = m.predict(X)         # numpy array of labels
            # flatten & ensure plain str
            out[tgt] = [str(v) for v in arr.tolist()]
        return out

    def print_results(self, data) -> None:
        X_te = data.X_test
        y_te = data.y_test_df()
        print("\n--- Intent (CatBoost) ---")
        print(classification_report(y_te["intent"],
                                    self.models["intent"].predict(X_te)))
        print("\n--- Intent + Tone (CatBoost) ---")
        print(classification_report(y_te["combo_23"],
                                    self.models["combo_23"].predict(X_te)))
        print("\n--- Intent + Tone + Res (CatBoost) ---")
        print(classification_report(y_te["combo_234"],
                                    self.models["combo_234"].predict(X_te)))
