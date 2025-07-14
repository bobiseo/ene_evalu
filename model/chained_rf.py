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

        print("\n--- Intent (RF) ---")
        print(classification_report(y_te_df["intent"],
                                    self.models["intent"].predict(X_te)))

        print("\n--- Intent + Tone (RF) ---")
        print(classification_report(y_te_df["combo_23"],
                                    self.models["combo_23"].predict(X_te)))

        print("\n--- Intent + Tone + Resolution (RF) ---")
        print(classification_report(y_te_df["combo_234"],
                                    self.models["combo_234"].predict(X_te)))
