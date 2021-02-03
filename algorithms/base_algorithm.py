from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    def __init__(self, algorithm_settings):
        self.model = None
        self.algorithm_settings = algorithm_settings

    @abstractmethod
    def build(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def train(self, train_x, train_y, settings):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, test_x):
        raise NotImplementedError()

    @abstractmethod
    def load(self, model_path):
        raise NotImplementedError()

    @abstractmethod
    def save(self, settings):
        raise NotImplementedError()
