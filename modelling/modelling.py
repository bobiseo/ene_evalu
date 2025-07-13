from model.randomforest import RandomForest
from modelling.data_model import Data
import numpy as np
import pandas as pd
from embeddings import get_tfidf_embd
from Config import *

def perform_modelling(X, df, hierarchy):
    if len(df) < Config.PRUNING_THRESHOLD:
        print(f"\nPruning Branch: Data size ({len(df)}) is below threshold ({Config.PRUNING_THRESHOLD})")
        return
    
    if not hierarchy:
        return

    current_label = hierarchy[0]
    remaining_hierarchy = hierarchy[1:]

    print(f"\nTraining model for label: {current_label}")
    data = Data(X, df, current_label)

    if data.X_train is None or len(data.X_train) == 0:
        print(f"Not enough data to train a model for {current_label}. Stopping this branch.")
        return

    model = RandomForest(f"RandomForest_{current_label}", data.get_embeddings(), data.get_type())
    model.train(data)
    
    if data.X_test is not None and len(data.X_test) > 0:
        model.predict(data.X_test)
        model.print_results(data)

    model.predict(X)
    model.predict_proba(X)
    
    df_with_preds = df.copy()
    df_with_preds['predicted_class'] = model.predictions
    df_with_preds['confidence'] = np.max(model.probabilities, axis=1)

    for cls in data.classes:
        print(f"\nBranching for class: {cls}")
        
        high_confidence_mask = (df_with_preds['predicted_class'] == cls) & (df_with_preds['confidence'] >= Config.CONFIDENCE_THRESHOLD)
        filtered_df = df_with_preds[high_confidence_mask]
        
        if not filtered_df.empty:
            original_indices = df.index.get_indexer(filtered_df.index)
            filtered_X = X[original_indices]
            
            perform_modelling(filtered_X, filtered_df, remaining_hierarchy)
        else:
            print(f"No high-confidence samples found for class '{cls}'. Stopping this branch.")