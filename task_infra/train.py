from __future__ import annotations
import pandas as pd

from abc import ABC, abstractmethod
from typing import Type
from sklearn.pipeline import Pipeline

import xgboost as xgb

from task_infra.task import Task
from task_infra.sampling import Sampler
from task_infra.consts import DATE_COL


class TrainModel(Task):

    def run(self):
        train_test_split = TrainTestSplitter(self.params['train_test_split_params'], self.input_df)
        self.subtasks.append(('TrainTestSplit', train_test_split))
        trainset_sample = Sampler.get_sampler(self.params['trainset_sampler_params'], train_test_split.outputs[train_test_split.train_df_key])
        self.subtasks.append(('TrainsetSampler', trainset_sample))
        model_features_targets_train = PrepareTrainFeaturesTargets(
            self.params['features_targets_params'],
            trainset_sample.outputs[trainset_sample.output_df_key]
        )
        self.subtasks.append(('FeaturesTargetsTrain', model_features_targets_train))
        classifier = Classifier.get_classifier(self.params['classifier_params'])
        # No time, but here would be another splitter to allow K-fold and early stopping.
        classifier.fit(
            features=model_features_targets_train.outputs[model_features_targets_train.features_key],
            target=model_features_targets_train.outputs[model_features_targets_train.target_key]
        )
        self.subtasks.append(('Classifier', classifier))

    def get_prediction_steps(self):
        return self.get_sub_tasks_predicion_steps()


class Classifier(Task):
    model_key = 'model'
    trainset_features_key = 'trainset_features'
    trainset_target_key = 'trainset_target_key'

    @staticmethod
    def get_classifier(classifier_params: dict) -> Classifier:
        classifier = {
            XgbClassifier.classifier_type: XgbClassifier,
        }.get(classifier_params['classifier_type'], None)
        if classifier is None:
            raise ValueError(f"Classifier of type {classifier_params['classifier_type']} not found.")
        else:
            data_sampler = classifier(classifier_params['additional_model_params'])
        return data_sampler

    def fit(self, features: pd.DataFrame, target: pd.Series, **kwargs) -> None:
        self.outputs[self.trainset_features_key] = features
        self.outputs[self.trainset_target_key] = target
        clf = self.outputs[self.model_key]
        clf.fit(features, target, **self.params['model_fit_params'], **kwargs)

    def get_prediction_steps(self) -> Pipeline:
        return Pipeline(steps=['Classifier', self.outputs[self.model_key]])


class XgbClassifier(Classifier):
    classifier_type = "xgb"

    def run(self):
        self.outputs[self.model_key] = xgb.XGBClassifier(**self.params['model_initialize_params'])


class PrepareTrainFeaturesTargets(Task):
    """
    This class will prepare engineered features, OneHotEncoding etc, on train set, and will generate fitted
    transformer to apply onto test set.
    """
    features_key = 'features_df'
    target_key = 'target'

    def run(self) -> None:
        drop_columns = DropColumns(self.params["drop_columns"], self.input_df)
        self.subtasks.append(('DropColumn', drop_columns))
        self.outputs[self.features_key] = drop_columns.outputs[drop_columns.df_after_drop_key]
        target = self.input_df[self.params['target_col']].map(self.params["target_mapping"]) #TODO: This should actually be a task.
        self.outputs[self.target_key] = target

    def get_prediction_steps(self) -> Pipeline:
        return self.get_sub_tasks_predicion_steps()


class DropColumns(Task):
    """
    Currently dropping columns is done by manually adding columns to drop, including features before preprocessing
    (e.g., age before clipping). This should be automated, with each transformer announcing it's output cols (or cols
    to drop since he made redundant).
    """
    df_after_drop_key = 'df_after_drop'

    def run(self) -> None:
        self.outputs[self.df_after_drop_key] = self.transform(self.input_df)

    def transform(self, df: pd.DataFrame):
        return df.drop(columns=self.params['columns_names'])

    def get_prediction_steps(self) -> Pipeline:
        return Pipeline(steps=['DropColumns', self])


class TrainTestSplitter(Task):
    """
    #TODO: Add functionality to split by dataset size proportion (i.e. find the relevant date according to ratio),
    # minimal number of positive labels, etc.)
    """
    train_df_key = 'train_df'
    test_df_key = 'test_df'

    def run(self) -> None:
        self.outputs[self.train_df_key] = self.input_df[
            (self.input_df[DATE_COL] >= self.params['train_start_date']) &
            (self.input_df[DATE_COL] < self.params['test_start_date'])
        ]
        self.outputs[self.test_df_key] = self.input_df[
            (self.input_df[DATE_COL] >= self.params['test_start_date']) &
            (self.input_df[DATE_COL] < self.params['test_end_date'])
            ]
        self.run_asserts()

    def get_prediction_steps(self) -> Pipeline:
        return self.get_sub_tasks_predicion_steps()

    def run_asserts(self) -> None:
        """
        Ideally this gets a list of which assert objects to run, indicated in params json, but I had no time to implement.
        Currently asserts dfs are not empty, can be configured to allow this.
        """
        assert len(self.outputs['train_df']) > 0, f"Training data df is empty! When splitting with params {self.params}"
        assert len(self.outputs['test_df']) > 0, f"Test data df is empty! When splitting with params {self.params}"

