from .chained_rf import ChainedLabelModel
from .chained_xgb import ChainedLabelXGBModel

MODEL_REGISTRY = {
    "chained_rf": ChainedLabelModel,
    "chained_xgboost": ChainedLabelXGBModel
}