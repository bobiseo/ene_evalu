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
MODEL_KEY = "chain"            # ← flip it here only

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
    return df

def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    return get_tfidf_embd(df)                  # returns X ∈ℝ^{n×D}

def get_data_object(X: np.ndarray, df: pd.DataFrame) -> Data:
    return Data(X, df)                         # Data now holds y2,y3,y4

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

    _ = model_predict(data, model_key=MODEL_KEY, name=f"run_{MODEL_KEY}")

    print("🎉 Done.")
