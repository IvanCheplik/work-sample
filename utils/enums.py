from enum import Enum


class AlgorithmType(Enum):
    tensorflow_nn = 0
    xgboost = 1


class SupervisedTask(Enum):
    regression = 0
    classification = 1
