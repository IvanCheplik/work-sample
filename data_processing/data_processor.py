import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from copy import deepcopy

from utils.enums import SupervisedTask


class DataProcessor:
    def __init__(self, settings):
        self.data_preprocessors = {}
        self.classification_y_name = 'y_classification'
        self.regression_y_name = 'y_regression'
        self.id_column_name = 'id'
        self.continuous_column_names = ['x1', 'x2', 'x4']
        self.discrete_column_names = ['x3', 'x5']
        self.settings = settings

    def prepare_data(self):
        train_matrix, train_y, train_ids, columns_names = self.prepare_train_data()
        test_matrix, test_y, test_ids = self.prepare_test_data()
        self.save_preprocessed_data(train_matrix, train_y, train_ids, test_matrix, test_y, test_ids, columns_names)

        return train_matrix, train_y, train_ids, test_matrix, test_y, test_ids

    def prepare_train_data(self):
        train_x, train_y = self.get_x_and_y_data(self.settings.train_set)
        train_matrix, columns_names = self.preprocess_train_data(train_x)
        train_ids = np.array(train_x[self.id_column_name]).reshape(-1, 1)

        return train_matrix, train_y, train_ids, columns_names

    def get_x_and_y_data(self, data_path):
        dataframe = pd.read_csv(data_path, delimiter=';')
        y = self.get_y_data(dataframe)
        x = dataframe.drop(columns=[self.regression_y_name, self.classification_y_name])

        return x, y

    def get_y_data(self, x):
        if self.settings.problem_type == SupervisedTask.regression:
            y = np.array(x[self.regression_y_name]).reshape(-1, 1)

        elif self.settings.problem_type == SupervisedTask.classification:
            y = np.array(x[self.classification_y_name]).reshape(-1, 1)

        else:
            raise TypeError('Unknown problem_type. Please check problem_type at configuration/settings.py')

        return y

    def preprocess_train_data(self, train_df):
        processed_data = []
        columns_names = [self.id_column_name]

        for column_name in train_df.columns:

            if column_name != self.id_column_name:
                array = np.array(train_df[column_name]).reshape(-1, 1)
                processed_column_data = self.preprocess_train_column_data(array, column_name)
                columns_names = self.collect_column_names(columns_names, column_name)
                processed_data.append(processed_column_data)

        matrix = np.hstack(processed_data)

        return matrix, columns_names

    def preprocess_train_column_data(self, array, column_name):
        if self.is_continuous(column_name):
            processed_column_data = self.column_data_scaling(array, column_name)

        elif self.is_discrete(column_name):
            processed_column_data = self.one_hot_encoding(array, column_name)

        else:
            raise TypeError('Unknown column_name. Please add column_name to continuous or discrete column_names '
                            'in data_processor')

        return processed_column_data

    def column_data_scaling(self, array, column_name):
        float_array = array.astype(float)
        scaler = preprocessing.StandardScaler().fit(float_array)
        processed_column_data = scaler.transform(float_array)
        self.data_preprocessors[column_name] = deepcopy(scaler)

        return processed_column_data

    def one_hot_encoding(self, array, column_name):
        encoder = preprocessing.OneHotEncoder(categories='auto')
        array_as_string = array.astype(str)
        encoder = encoder.fit(array_as_string)
        encoded_data = self.get_encoded_array(array_as_string, encoder)
        self.data_preprocessors[column_name] = deepcopy(encoder)

        return encoded_data

    def collect_column_names(self, columns_names, column_name):
        if self.is_continuous(column_name):
            feature_columns_names = [column_name]
        else:
            feature_columns_names = []

            for discrete_value in self.data_preprocessors[column_name].categories_[0].tolist():
                new_column_name = f'{column_name}[{discrete_value}]'
                feature_columns_names.append(new_column_name)

        columns_names.extend(feature_columns_names)

        return columns_names

    def prepare_test_data(self):
        test_x, test_y = self.get_x_and_y_data(self.settings.test_set)
        test_matrix = self.preprocess_test_data(test_x)
        test_ids = np.array(test_x[self.id_column_name]).reshape(-1, 1)

        return test_matrix, test_y, test_ids

    def preprocess_test_data(self, test_df):
        processed_data = []

        for column_name in test_df.columns:

            if column_name != self.id_column_name:
                processed_column_data = self.get_processed_test_column_data(column_name, test_df)
                processed_data.append(processed_column_data)

        matrix = np.hstack(processed_data)

        return matrix

    def get_processed_test_column_data(self, column_name, test_df):
        array = np.array(test_df[column_name]).reshape(-1, 1)

        if self.is_continuous(column_name):
            scaler = self.data_preprocessors[column_name]
            processed_column_data = scaler.transform(array.astype(float))

        elif self.is_discrete(column_name):
            processed_column_data = self.process_test_discrete_values(array, column_name)

        else:
            raise TypeError('Unknown column_name. Please add column_name to continuous or discrete column_names '
                            'in data_processor')

        return processed_column_data

    def process_test_discrete_values(self, array, column_name):
        encoder = self.data_preprocessors[column_name]
        array_as_string = array.astype(str)
        processed_column_data = self.get_encoded_array(array_as_string, encoder)

        return processed_column_data

    def save_preprocessed_data(self, train_matrix, train_y, train_ids, test_matrix, test_y, test_ids, columns_names):
        train_data = np.hstack([train_ids, train_matrix, train_y])
        test_data = np.hstack([test_ids, test_matrix, test_y])
        columns_names = columns_names + ['y']

        train_df = pd.DataFrame(train_data, columns=columns_names)
        test_df = pd.DataFrame(test_data, columns=columns_names)

        train_data_save_path = os.path.join(self.settings.output_path, 'preprocessed_train_data.csv')
        test_data_save_path = os.path.join(self.settings.output_path, 'preprocessed_test_data.csv')

        os.makedirs(self.settings.output_path, exist_ok=True)
        train_df.to_csv(train_data_save_path, index=False)
        test_df.to_csv(test_data_save_path, index=False)

        print(f'\nData preprocessing was finished. Preprocessed data saved to {self.settings.output_path}')

    def get_encoded_array(self, array_as_string, encoder):
        return np.array(encoder.transform(array_as_string).todense())

    def is_continuous(self, column_name):
        return column_name in self.continuous_column_names

    def is_discrete(self, column_name):
        return column_name in self.discrete_column_names
