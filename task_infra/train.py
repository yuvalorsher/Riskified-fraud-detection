from __future__ import annotations
import pandas as pd

from abc import ABC, abstractmethod
from typing import Type
from sklearn.pipeline import Pipeline

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
        # classifier = Classifier

    def get_prediction_steps(self):
        return self.get_sub_tasks_predicion_steps()


class Classifier(Task):
    @staticmethod
    def get_classifier(classifier_params: dict, input_df) -> Classifier:
        data_sampler = {
            RandomSampler.sampler_type: RandomSampler,
            OverSampler.sampler_type: OverSampler,
        }.get(sampler_params['sampler_type'], None)
        if data_sampler is None:
            raise ValueError(f"Data sampler of type {sampler_params['sampler_type']} not found.")
        else:
            data_sampler = data_sampler(sampler_params['additional_sampler_params'], input_df)
        return data_sampler


class PrepareTrainFeaturesTargets(Task):
    """
    This class will prepare engineered features, OneHotEncoding etc, on train set, and will generate fitted
    transformer to apply onto test set.
    """
    features_key = 'features_df'
    target_key = 'target_df'

    def run(self) -> None:
        drop_columns = DropColumns(self.params["drop_columns"], self.input_df)
        self.subtasks.append(('DropColumn', drop_columns))
        self.outputs[self.features_key] = drop_columns.outputs[drop_columns.df_after_drop_key]
        self.outputs[self.target_key] = self.input_df[self.params['target_col']]

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

