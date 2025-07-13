"""
modelling.py
Controller helpers that:
  • receive a Data object (already split into train / test)
  • pick a model from MODEL_REGISTRY  ("rf", "lr", "chain", …)
  • train, predict, and print the right evaluation reports
"""

from model import MODEL_REGISTRY                       # <-- new
from sklearn.metrics import classification_report

from warnings import filterwarnings
from sklearn.exceptions import UndefinedMetricWarning

filterwarnings("ignore", category=UndefinedMetricWarning)


def model_predict(data, model_key: str = "rf", name: str = "Model"):
    """
    Train the requested model on data.X_train / y_train
    and evaluate on data.X_test.

    • For single-stage models ("rf", "lr", …) we call the wrapper’s
      own .print_results().ß
    • For classifier-chains ("chain", "chain_lr", …) we print three
      tier-level reports (Intent / Tone / Resolution).
    """

    print(f"🔍 Running model key: {model_key}")

    # 1. Instantiate the wrapper class from registry
    ModelCls = MODEL_REGISTRY[model_key]
    model = ModelCls()                    # ChainModel or single-stage wrapper

    # 2. Train
    model.train(data)

    # 3. Predict on the held-out set
    preds = model.predict(data.X_test)

    # # 4. Evaluation
    model.print_results(data)

    # 5. Hand back predictions to caller (dict or 1-D ndarray)
    return preds


# Optional convenience wrapper
def model_evaluate(model, data):
    """Re-print results on demand."""
    model.print_results(data)