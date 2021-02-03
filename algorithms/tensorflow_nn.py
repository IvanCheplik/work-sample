import os
import numpy as np

from algorithms.base_algorithm import BaseAlgorithm
from utils.enums import SupervisedTask


class TensorFlowNN(BaseAlgorithm):
    def __init__(self, algorithm_settings, problem_type):
        super().__init__(algorithm_settings)
        self.problem_type = problem_type

    def build(self):
        if self.problem_type == SupervisedTask.regression:
            self.build_regression_model()

        elif self.problem_type == SupervisedTask.classification:
            self.build_classification_model()

        else:
            raise TypeError('Unknown problem_type')

    def build_regression_model(self):
        from tensorflow.keras import layers, regularizers, optimizers, models

        model = models.Sequential([
            layers.Dense(self.algorithm_settings.hidden_units_layer_1, activation='relu',
                         kernel_regularizer=regularizers.l2(self.algorithm_settings.l2)),
            layers.Dense(self.algorithm_settings.hidden_units_layer_2, activation='relu',
                         kernel_regularizer=regularizers.l2(self.algorithm_settings.l2)),
            layers.Dense(self.algorithm_settings.hidden_units_layer_3, activation='relu',
                         kernel_regularizer=regularizers.l2(self.algorithm_settings.l2)),
            layers.Dense(1)
        ])

        model.compile(loss="mse", optimizer=optimizers.Adam(lr=self.algorithm_settings.learning_rate))
        self.model = model

    def build_classification_model(self):
        from tensorflow.keras import layers, regularizers, optimizers, models

        model = models.Sequential([
            layers.Dense(self.algorithm_settings.hidden_units_layer_1, activation='relu',
                         kernel_regularizer=regularizers.l2(self.algorithm_settings.l2)),
            layers.Dense(self.algorithm_settings.hidden_units_layer_2, activation='relu',
                         kernel_regularizer=regularizers.l2(self.algorithm_settings.l2)),
            layers.Dense(self.algorithm_settings.hidden_units_layer_3, activation='softmax',
                         kernel_regularizer=regularizers.l2(self.algorithm_settings.l2)),
            layers.Dense(1)
        ])

        model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                      optimizer=optimizers.Adam(lr=self.algorithm_settings.learning_rate))
        self.model = model

    def train(self, train_x, train_y, settings):
        self.model.fit(train_x, train_y, epochs=self.algorithm_settings.max_epochs)
        self.save(settings)

    def evaluate(self, test_x):
        prediction = self.model.predict(test_x)

        if self.problem_type == SupervisedTask.classification:
            prediction = np.where(prediction < 0.35, 0, 1)

        return prediction

    def load(self, model_path):
        from tensorflow.keras import models, optimizers
        self.model = models.load_model(model_path, compile=False)

        if self.problem_type == SupervisedTask.regression:
            self.model.compile(loss="mse", optimizer=optimizers.Adam(lr=self.algorithm_settings.learning_rate))

        else:
            self.model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                               optimizer=optimizers.Adam(lr=self.algorithm_settings.learning_rate))

    def save(self, settings):
        model_save_dir = os.path.join(settings.models_path, 'tensorflow_models')
        os.makedirs(model_save_dir, exist_ok=True)
        model_name = self.get_model_name(settings)
        save_path = os.path.join(model_save_dir, model_name)
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")

    def get_model_name(self, settings):
        if settings.problem_type == SupervisedTask.regression:
            return "regression_model.h5"

        else:
            return 'classification_model.h5'
