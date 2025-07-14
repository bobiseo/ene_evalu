# main.py  ──────────────────────────────────────────────────────
from preprocess            import *      
from embeddings           import get_tfidf_embd
from modelling.data_model import Data
from modelling.modelling  import model_predict
from model                import MODEL_REGISTRY
from utils                 import save_chained_results
import random, numpy as np, pandas as pd

# ───────── reproducibility ─────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# ───────── choose which model to run ─────────
# "chained_rf", "chained_lr", "chained_lgbm", "chained_cat",...
MODEL_KEY = "chained_cat"  

# ───────── main entry ─────────
if __name__ == "__main__":

    # 1. Load raw data & full preprocessing
    df = preprocess_data(get_input_data())
    df.to_csv("out/cleaned_data_preview.csv", index=False)

    # 2. Build embeddings
    X = get_tfidf_embd(df)

    # 3. Wrap into Data object (handles train/test split)
    data = Data(X, df)             

    # 4. Train & predict using the chosen model
    preds = model_predict(data, model_key=MODEL_KEY, name=f"run_{MODEL_KEY}")

    # 5. assemble & save predictions + chain‐score
    save_chained_results(preds, data, df, out_dir="out", model_key=MODEL_KEY)
    
    print("🎉 Done.")