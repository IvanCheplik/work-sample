

class XGBoostRegressionSettings:
    def __init__(self):
        self.max_depth = 4
        self.learning_rate = 0.1
        self.n_estimators = 330
        self.objective = "reg:squarederror"
        self.booster = 'gbtree'
        self.n_jobs = -1
        self.gamma = 0.05
        self.min_child_weight = 1.0
        self.max_delta_step = 0.0
        self.subsample = 1.0
        self.reg_alpha = 0.0
        self.reg_lambda = 1.0
        self.random_state = 0
        self.early_stopping_rounds = 10
        self.eval_metric = 'mae'


class XGBoostClassificationSettings:
    def __init__(self):
        self.max_depth = 4
        self.learning_rate = 0.1
        self.n_estimators = 330
        self.objective = "binary:logistic"
        self.booster = 'gbtree'
        self.n_jobs = -1
        self.gamma = 0.05
        self.min_child_weight = 1.0
        self.max_delta_step = 0.0
        self.subsample = 1.0
        self.reg_alpha = 0.0
        self.reg_lambda = 1.0
        self.random_state = 0
        self.early_stopping_rounds = 10
        self.eval_metric = 'error'


class TensorFlowNNRegressionSettings:
    def __init__(self):
        self.max_epochs = 330
        self.l2 = 0.005
        self.learning_rate = 0.01

        self.hidden_units_layer_1 = 5
        self.hidden_units_layer_2 = 25
        self.hidden_units_layer_3 = 5


class TensorFlowNNClassificationSettings:
    def __init__(self):
        self.max_epochs = 200
        self.l2 = 0.000005
        self.learning_rate = 0.00001

        self.hidden_units_layer_1 = 16
        self.hidden_units_layer_2 = 32
        self.hidden_units_layer_3 = 16
