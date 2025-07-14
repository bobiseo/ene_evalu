# main.py (Final Hybrid)
from preprocess            import *      
from embeddings           import get_tfidf_embd
from modelling.data_model import Data
from modelling.modelling  import model_predict
from model                import MODEL_REGISTRY
import Config
# The save_chained_results utility is removed as we are evaluating groups separately
import random, numpy as np, pandas as pd

# ───────── reproducibility ─────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# ───────── choose which model type to use for all specialists ─────────
MODEL_KEY = "chained_lr"  

# ───────── main entry ─────────
if __name__ == "__main__":

    # 1. Load and preprocess all data at once
    df = preprocess_data(get_input_data())
    print("✅ Data preprocessing complete.")

    # 2. Hierarchical Split: Group data by the top-level category 'y1'
    grouped = df.groupby(Config.Config.GROUPED)
    print(f"Found {len(grouped)} groups to process: {list(grouped.groups.keys())}\n")

    # 3. Loop through each group and train a dedicated specialist model
    for group_name, group_df in grouped:
        print(f"--- Training Specialist Model for Group: [{group_name}] ---")

        # 3a. Build embeddings SPECIFIC to this group's vocabulary
        X_group = get_tfidf_embd(group_df)
        print(f"Embeddings created for '{group_name}' with shape: {X_group.shape}")

        # 3b. Wrap group data (this handles the train/test split for the group)
        data_group = Data(X_group, group_df.reset_index(drop=True))
        
        # 3c. Train & Evaluate the Chained Model on this group's data
        # The model_predict function will print the detailed classification reports
        model_predict(data_group, model_key=MODEL_KEY, name=f"run_{MODEL_KEY}_{group_name}")
        
        print(f"--- Finished Processing Group: [{group_name}] ---\n")
    
    print("🎉 All specialist models trained and evaluated.")