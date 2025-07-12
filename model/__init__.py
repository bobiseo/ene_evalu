from model.chain import ChainModel
from model.randomforest import RandomForest

MODEL_REGISTRY = {
    "rf":    RandomForest,
    "chain": ChainModel,
}