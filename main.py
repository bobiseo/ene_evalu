# main.py  ──────────────────────────────────────────────────────
from preprocess            import *      
from embeddings           import get_tfidf_embd
from modelling.data_model import Data
from modelling.modelling  import model_predict
from model                import MODEL_REGISTRY
import random, numpy as np, pandas as pd

# ───────── reproducibility ─────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# ───────── choose which model to run ─────────
# "chained_rf", "chained_lr", "chained_svm", ...
MODEL_KEY = "chained_rf"  

# ───────── main entry ─────────
if __name__ == "__main__":

    # 1. Load raw data & full preprocessing
    df = preprocess_data(get_input_data())
    df.to_csv("out/cleaned_data_preview.csv", index=False)
    print("✅ out/cleaned_data_preview.csv saved")

    # 2. Build feature embeddings
    X = get_tfidf_embd(df)

    # 3. Wrap into Data object (handles train/test split)
    data = Data(X, df)             

    # 4. Train & predict using the chosen model
    preds = model_predict(data, model_key=MODEL_KEY, name=f"run_{MODEL_KEY}")

    # 5. Assemble & save test-set predictions with chain scoring
    out_fname = f"out/predictions_{MODEL_KEY}.csv"

    # def save_chained(preds, data, df, out_fname):
    y_test_df = data.y_test_df().reset_index(drop=True)
    texts     = df.reset_index(drop=True).iloc[data.X_test_idx]["text"].tolist()

    df_out = pd.DataFrame({
        "text"           : texts,
        "intent_true"    : y_test_df["intent"],
        "tone_true"      : [s.split(" | ")[1] for s in y_test_df["combo_23"]],
        "resolution_true": [s.split(" | ")[2] for s in y_test_df["combo_234"]],
        "intent_pred"    : preds["intent"],
        "tone_pred"      : [s.split(" | ")[1] for s in preds["combo_23"]],
        "resolution_pred": [s.split(" | ")[2] for s in preds["combo_234"]],
    })

    # compute the chain score inline
    def chain_score(row) -> float:
        """
        0.00  if intent wrong
        0.33  if only intent right
        0.66  if intent + tone right
        1.00  if all three right
        """
        if row["intent_pred"] != row["intent_true"]:
            return 0.00
        score = 1.0 / 3                           # intent correct
        if row["tone_pred"] == row["tone_true"]:
            score += 1.0 / 3                      # tone correct
            if row["resolution_pred"] == row["resolution_true"]:
                score += 1.0 / 3                  # resolution correct
        return round(score, 2)

    df_out["score"] = df_out.apply(chain_score, axis=1)
    df_out.to_csv(out_fname, index=False)
    print(f"📄 {out_fname} saved")
    
    print("🎉 Done.")