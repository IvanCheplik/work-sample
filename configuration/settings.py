import os

from configuration.algorithms_settings import XGBoostRegressionSettings, XGBoostClassificationSettings, \
    TensorFlowNNRegressionSettings, TensorFlowNNClassificationSettings
from utils.enums import AlgorithmType, SupervisedTask


class Settings:
    base_path = os.getcwd()
    input_data_path = os.path.join(base_path, 'data', 'input')
    output_path = os.path.join(base_path, 'data', 'output')
    models_path = os.path.join(base_path, 'models')

    def __init__(self):
        self.train_set = os.path.join(self.input_data_path, 'train_set.csv')
        self.test_set = os.path.join(self.input_data_path, 'test_set.csv')

        # two algorithms available - AlgorithmType.xgboost and AlgorithmType.tensorflow_nn
        self.algorithm_type = AlgorithmType.tensorflow_nn

        # two problem_types available - SupervisedTask.regression and SupervisedTask.classification
        self.problem_type = SupervisedTask.classification
        self.use_pretrained_model = False
        self.algorithm_settings = None
        self.initialize_algorithm_settings()

    def initialize_algorithm_settings(self):
        if self.algorithm_type == AlgorithmType.xgboost and self.problem_type == SupervisedTask.regression:
            self.algorithm_settings = XGBoostRegressionSettings()

        elif self.algorithm_type == AlgorithmType.xgboost and self.problem_type == SupervisedTask.classification:
            self.algorithm_settings = XGBoostClassificationSettings()

        elif self.algorithm_type == AlgorithmType.tensorflow_nn and self.problem_type == SupervisedTask.regression:
            self.algorithm_settings = TensorFlowNNRegressionSettings()

        elif self.algorithm_type == AlgorithmType.tensorflow_nn and self.problem_type == SupervisedTask.classification:
            self.algorithm_settings = TensorFlowNNClassificationSettings()

        else:
            raise Exception('Unexpected algorithm type: ' + str(self.algorithm_type.name))
