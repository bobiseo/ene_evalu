from model.chain import ChainModel
from model.randomforest import RandomForest
from .combined_model import CombinedLabelModel

MODEL_REGISTRY = {
    "rf":    RandomForest,
    "chain": ChainModel,
    "combined": CombinedLabelModel,
}