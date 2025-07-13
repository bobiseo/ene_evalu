from .chained_rf import ChainedLabelModel
from .chained_lr import ChainedLRModel
from .chained_lgbm import ChainedLGCBMModel
from .chained_cat  import ChainedCatBoostModel

MODEL_REGISTRY = {
    "chained_rf"    : ChainedLabelModel,
    "chained_lr"    : ChainedLRModel,
    "chained_lgbm"  : ChainedLGCBMModel,
    "chained_cat": ChainedCatBoostModel,
}