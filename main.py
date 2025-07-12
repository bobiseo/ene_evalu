# main.py  ──────────────────────────────────────────────────────
from preprocess            import *          # cleaning utilities
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
# "rf", "lr", "svm", "chain", "chain_lr", "chain_svm", …
MODEL_KEY = "combined"            # ← flip it here only

# ───────── pipeline helpers ─────────
def load_data() -> pd.DataFrame:
    return get_input_data()                    # from preprocess.py

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = de_duplication(df)
    df = coarse_clean(df)

    # translate only after coarse_clean
    df[Config.INTERACTION_CONTENT] = translate_to_en(df[Config.INTERACTION_CONTENT].tolist())
    df[Config.TICKET_SUMMARY]      = translate_to_en(df[Config.TICKET_SUMMARY].tolist())

    df = noise_remover(df)
    df = merge_text_columns(df)                # makes df["text"]

    # label hygiene
    df["intent"]     = df["y2"].fillna("none").replace('', "none")
    df["tone"]       = df["y3"].fillna("none").replace('', "none")
    df["resolution"] = df["y4"].fillna("none").replace('', "none")

    # ── NEW combined targets for Design-1 ──────────────────────
    df["combo_23"]  = df["intent"] + " | " + df["tone"]
    df["combo_234"] = (
        df["intent"] + " | " + df["tone"] + " | " + df["resolution"]
    )
    return df

# ───────── feature builder ─────────
def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    return get_tfidf_embd(df)                  # returns X ∈ℝ^{n×D}

def get_data_object(X: np.ndarray, df: pd.DataFrame) -> Data:
    return Data(X, df)                         # Data now holds intent/tone/res

# ───────── per-row chain score ─────────
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

# ───────── main entry ─────────
if __name__ == "__main__":

    # 1. load & clean
    df = preprocess_data(load_data())
    df.to_csv("cleaned_data_preview.csv", index=False)
    print("✅ cleaned_data_preview.csv saved")

    # 2. quick label sanity
    for col in ["intent", "tone", "resolution"]:
        print(f"✅ {col} distribution:\n", df[col].value_counts(dropna=False), "\n")

    # 3. embed + wrap into Data object
    X = build_embeddings(df)
    data = get_data_object(X, df)              # splits train/test inside

    # 4. run chosen model
    if MODEL_KEY not in MODEL_REGISTRY:
        raise ValueError(f"Unknown MODEL_KEY: {MODEL_KEY}")

    preds = model_predict(data, model_key=MODEL_KEY, name=f"run_{MODEL_KEY}")

    # ────────────────────────────────────────────────────────────────
    # 5.  SAVE THE TEST-SET PREDICTIONS SO YOU CAN INSPECT THEM
    if MODEL_KEY.startswith("chain"):
        y_test_df = data.y_test_df().reset_index(drop=True)
        out = pd.DataFrame({
            "text": df.reset_index(drop=True).iloc[data.X_test_idx]["text"].reset_index(drop=True),
            "intent_true"     : y_test_df["intent"],
            "tone_true"       : y_test_df["tone"],
            "resolution_true" : y_test_df["resolution"],
            "intent_pred"     : preds["intent"],
            "tone_pred"       : preds["tone"],
            "resolution_pred" : preds["resolution"],
        })
        out["score"] = out.apply(chain_score, axis=1)   # add chain-score
        out.to_csv("predictions_chain.csv", index=False)
        print("📄 predictions_chain.csv saved")

    elif MODEL_KEY == "combined":
        # ── export three separate columns extracted from the combined strings
        y_test_df = data.y_test_df().reset_index(drop=True)

        texts = (
            df.reset_index(drop=True)          # ensure positions match 0..n-1
            .iloc[data.X_test_idx]["text"]   # positional slice
            .tolist()
        )
        intent_true       = y_test_df["intent"].tolist()
        tone_true         = [s.split(" | ")[1] for s in y_test_df["combo_23"]]
        resolution_true   = [s.split(" | ")[2] for s in y_test_df["combo_234"]]
        intent_pred       = preds["intent"].tolist()
        tone_pred         = [s.split(" | ")[1] for s in preds["combo_23"]]
        resolution_pred   = [s.split(" | ")[2] for s in preds["combo_234"]]

        out = pd.DataFrame({
            "text"            : texts,
            "intent_true"     : intent_true,
            "tone_true"       : tone_true,
            "resolution_true" : resolution_true,
            "intent_pred"     : intent_pred,
            "tone_pred"       : tone_pred,
            "resolution_pred" : resolution_pred,
        })
        out["score"] = out.apply(chain_score, axis=1)   # add chain-score
        out.to_csv("predictions_combined.csv", index=False)
        print("📄 predictions_combined.csv saved")

    else:  # other single-stage models
        out = pd.DataFrame({
            "text"       : df.reset_index(drop=True).iloc[data.X_test_idx]["text"].reset_index(drop=True),
            "label_true" : data.y_test,
            "label_pred" : preds,
        })
        out.to_csv(f"predictions_{MODEL_KEY}.csv", index=False)
        print(f"📄 predictions_{MODEL_KEY}.csv saved")
    # ────────────────────────────────────────────────────────────────

    print("🎉 Done.")
