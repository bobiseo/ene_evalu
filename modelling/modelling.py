from model import MODEL_REGISTRY                      
from sklearn.metrics import classification_report
from warnings import filterwarnings
from sklearn.exceptions import UndefinedMetricWarning

filterwarnings("ignore", category=UndefinedMetricWarning)

def model_predict(data, model_key, name=None):
    """
    Controller helpers that:
    • receive a Data object (already split into train / test)
    • pick a model from MODEL_REGISTRY
    • train, predict, and print the right evaluation reports
    """
    print(f"🔍 Running model key: {model_key}")
    # 1. Instantiate the wrapper class from registry
    ModelCls = MODEL_REGISTRY.get(model_key)
    if ModelCls is None:
        raise ValueError(f"No model named {model_key}")
    model = ModelCls()
    # 2. Train
    model.train(data)
    # 3. Predict on the held-out set
    preds = model.predict(data.X_test)
    # # 4. Evaluation
    model.print_results(data)
    return preds
