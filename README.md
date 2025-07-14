# Customer Interaction Tagging System

** Mirae (24132799) & Aakash (23412101)

## Overview
A chained multi-output classifier over two CSV datasets.  
Swap in different models (LR, RF, LGBM, CatBoost) by changing one `MODEL_KEY` in **main.py**.


## Structure

- **main.py**  
  Implements the five-step pipeline: load → preprocess → embed → train/predict → save  
- **preprocess.py**  
  Data loading, cleaning, translation, merging, and label-combining  
- **embeddings.py**  
  TF-IDF feature builder  
- **utils.py**  
  Scoring and CSV-export helper  
- **modelling/**  
  - **data_model.py** splits data and exposes `.X_train`, `.X_test`, `.y_train_df()`, `.y_test_df()`  
  - **modelling.py** looks up `MODEL_KEY` in `model/MODEL_REGISTRY`, then trains & predicts  
- **model/**  
  Contains chained-model wrappers: `chained_lr.py`, `chained_rf.py`, `chained_lgbm.py`, `chained_catboost.py`  
- **data/**  
  Raw CSV datasets
- **out/**  
  CSV outputs


## Requirements  
```bash
pip install -r requirements.txt
# + plus, for LightGBM & CatBoost:
pip install lightgbm catboost
```

## how to run 
Edit `main.py`s `MODEL_KEY` to one of:
'chained_lr, chained_rf, chained_lgbm, chained_catboost'

```bash
python main.py
```

## Inspect outputs
out/:
 `cleaned_data_preview.csv` 
 `predictions_<model_key>.csv` 