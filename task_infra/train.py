import pandas as pd

from abc import ABC, abstractmethod
from task_infra.task import Task
from typing import Type
from sklearn.pipeline import Pipeline

from task_infra.consts import DATE_COL

class TrainModel(Task):

    def run(self):
        train_test_split = TrainTestSplitter(self.params['train_test_split_params'], self.input_df)

    def get_prediction_steps(self):
        return self.get_sub_tasks_predicion_steps()

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
        self.outputs[selftest_df_key] = self.input_df[
            (self.input_df[DATE_COL] >= self.params['test_start_date']) &
            (self.input_df[DATE_COL] < self.params['test_end_date'])
            ]
        self.run_asserts()

    def get_prediction_steps(self) -> Pipeline:
        return self.get_sub_tasks_predicion_steps()

    def run_asserts(self) -> None:
        """
        Ideally this gets a list of which assert objects to run, indicated in params json, but I had no time to implement.
        """
        assert len(self.outputs['train_df']) > 0, f"Training data df is empty! When splitting with params {self.params}"
        assert len(self.outputs['test_df']) > 0, f"Test data df is empty! When splitting with params {self.params}"

