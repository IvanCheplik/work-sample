import os

from algorithms.base_algorithm import BaseAlgorithm
from utils.enums import SupervisedTask


class XGBoost(BaseAlgorithm):
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
        from xgboost import XGBRegressor
        self.model = XGBRegressor(max_depth=self.algorithm_settings.max_depth,
                                  learning_rate=self.algorithm_settings.learning_rate,
                                  n_estimators=self.algorithm_settings.n_estimators,
                                  objective=self.algorithm_settings.objective,
                                  booster=self.algorithm_settings.booster,
                                  n_jobs=self.algorithm_settings.n_jobs,
                                  gamma=self.algorithm_settings.gamma,
                                  min_child_weight=self.algorithm_settings.min_child_weight,
                                  max_delta_step=self.algorithm_settings.max_delta_step,
                                  subsample=self.algorithm_settings.subsample,
                                  reg_alpha=self.algorithm_settings.reg_alpha,
                                  reg_lambda=self.algorithm_settings.reg_lambda,
                                  random_state=self.algorithm_settings.random_state
                                  )

    def build_classification_model(self):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(max_depth=self.algorithm_settings.max_depth,
                                   learning_rate=self.algorithm_settings.learning_rate,
                                   n_estimators=self.algorithm_settings.n_estimators,
                                   objective=self.algorithm_settings.objective,
                                   booster=self.algorithm_settings.booster,
                                   n_jobs=self.algorithm_settings.n_jobs,
                                   gamma=self.algorithm_settings.gamma,
                                   min_child_weight=self.algorithm_settings.min_child_weight,
                                   max_delta_step=self.algorithm_settings.max_delta_step,
                                   subsample=self.algorithm_settings.subsample,
                                   reg_alpha=self.algorithm_settings.reg_alpha,
                                   reg_lambda=self.algorithm_settings.reg_lambda,
                                   random_state=self.algorithm_settings.random_state
                                   )

    def train(self, train_x, train_y, settings):
        self.model.fit(train_x, train_y, eval_metric=self.algorithm_settings.eval_metric)
        self.save(settings)

    def evaluate(self, test_x):
        prediction = self.model.predict(test_x)
        prediction = prediction.reshape(-1, 1)
        return prediction

    def load(self, model_path):
        self.model.load_model(fname=model_path)

    def save(self, settings):
        model_save_dir = os.path.join(settings.models_path, 'xgboost_models')
        os.makedirs(model_save_dir, exist_ok=True)
        model_name = self.get_model_name(settings)
        save_path = os.path.join(model_save_dir, model_name)
        self.model.save_model(fname=save_path)
        print(f"Model saved to: {save_path}")

    def get_model_name(self, settings):
        if settings.problem_type == SupervisedTask.regression:
            return 'regression_model.xgb'

        else:
            return 'classification_model.xgb'
