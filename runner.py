import os
import numpy as np
import pandas as pd
from sklearn import metrics

from algorithms.tensorflow_nn import TensorFlowNN
from algorithms.xgboost_algorithm import XGBoost
from data_processing.data_processor import DataProcessor
from utils.enums import AlgorithmType, SupervisedTask


class Runner:

    def __init__(self, settings):
        self.settings = settings

    def run(self):
        data_processor = DataProcessor(self.settings)
        train_x, train_y, train_ids, test_x, test_y, test_ids = data_processor.prepare_data()
        prediction = self.run_algorithm(train_x, train_y, test_x)
        self.save_prediction(test_ids, test_y, prediction)
        self.evaluate_metrics(test_y, prediction)

    def run_algorithm(self, train_x, train_y, test_x):
        if self.settings.algorithm_type == AlgorithmType.xgboost:
            model = self.prepare_xgboost_model()

        elif self.settings.algorithm_type == AlgorithmType.tensorflow_nn:
            model = self.prepare_tensorflow_model()

        prediction = self.get_prediction(model, test_x, train_x, train_y)

        return prediction

    def prepare_xgboost_model(self):
        model = XGBoost(self.settings.algorithm_settings, self.settings.problem_type)
        model.build()

        if self.settings.use_pretrained_model:
            model_folder = os.path.join(self.settings.models_path, 'xgboost_models')
            model_name = model.get_model_name(self.settings)
            model_path = os.path.join(model_folder, model_name)

            if os.path.isfile(model_path):
                model.load(model_path)

            else:
                raise FileNotFoundError(f'{model_name} not found. Please set settings.use_pretrained_model to False')

        return model

    def prepare_tensorflow_model(self):
        model = TensorFlowNN(self.settings.algorithm_settings, self.settings.problem_type)

        if not self.settings.use_pretrained_model:
            model.build()

        else:
            model_folder = os.path.join(self.settings.models_path, 'tensorflow_models')
            model_name = model.get_model_name(self.settings)
            model_path = os.path.join(model_folder, model_name)

            if os.path.isfile(model_path):
                model.load(model_path)

            else:
                raise FileNotFoundError(f'{model_name} not found. Please set settings.use_pretrained_model to False')

        return model

    def get_prediction(self, model, test_x, train_x, train_y):
        if self.settings.use_pretrained_model:
            prediction = model.evaluate(test_x)

        else:
            model.train(train_x, train_y, self.settings)
            prediction = model.evaluate(test_x)

        return prediction

    def save_prediction(self, test_ids, test_y, prediction):
        result = np.hstack([test_ids, test_y, prediction])
        result_df = pd.DataFrame(result, columns=['id', 'real_y', 'predicted_y'])
        file_name = f'{self.settings.algorithm_type.name}_{self.settings.problem_type.name}_prediction.csv'
        save_path = os.path.join(self.settings.output_path, file_name)
        result_df.to_csv(save_path, index=False)
        print(f'\nPrediction saved to {save_path}')

    def evaluate_metrics(self, test_y, prediction):
        if self.settings.problem_type == SupervisedTask.regression:
            mae = metrics.mean_absolute_error(test_y, prediction)
            mse = metrics.mean_squared_error(test_y, prediction)
            print(f'\nPrediction mean absolute error: {mae} \nPrediction mean squared error: {mse}')

        else:
            accuracy = metrics.accuracy_score(test_y, prediction)
            print(f'\nPrediction accuracy: {accuracy}')
