# chain.py  ─ replace the entire class body with this tidy version
from model.randomforest import RandomForest
from model.base         import BaseModel
import numpy as np

class ChainModel(BaseModel):
    def __init__(self, learner_cls=RandomForest, use_soft=True):
        super().__init__()
        self.use_soft = use_soft
        self.m_intent = learner_cls("Intent", None, None)
        self.m_tone   = learner_cls("Tone",   None, None)
        self.m_res    = learner_cls("Resol",  None, None)

    # ────────────────────────────────────────
    def train(self, data):
        X_tr = data.X_train
        y_df = data.y_train_df()
        y_intent = y_df["intent"].values
        y_tone   = y_df["tone"].values
        y_res    = y_df["resolution"].values

        # stage-1  (fit underlying estimator directly)
        self.m_intent.mdl.fit(X_tr, y_intent)
        z2 = self._enc(self.m_intent.mdl, X_tr)

        # stage-2
        X2 = np.hstack([X_tr, z2])
        self.m_tone.mdl.fit(X2, y_tone)
        z3 = self._enc(self.m_tone.mdl, X2)

        # stage-3
        X3 = np.hstack([X2, z3])
        self.m_res.mdl.fit(X3, y_res)

    # ────────────────────────────────────────
    def predict(self, X):
        z2 = self._enc(self.m_intent.mdl, X)
        X2 = np.hstack([X, z2])

        z3 = self._enc(self.m_tone.mdl, X2)
        X3 = np.hstack([X2, z3])

        return {
            "intent":      self.m_intent.mdl.predict(X),
            "tone":        self.m_tone.mdl.predict(X2),
            "resolution":  self.m_res.mdl.predict(X3),
        }

    # helper: soft probs if available else one-hot
    def _enc(self, est, X):
        if self.use_soft and hasattr(est, "predict_proba"):
            return est.predict_proba(X)
        hard = est.predict(X)
        onehot = np.zeros((len(hard), len(est.classes_)), float)
        idx = {c: i for i, c in enumerate(est.classes_)}
        rows = np.arange(len(hard))
        onehot[rows, [idx[c] for c in hard]] = 1.0
        return onehot

    # satisfy abstract interface
    def data_transform(self, X): return X
