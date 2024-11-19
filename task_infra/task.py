from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator


class Task(ABC, BaseEstimator):
    """
    A Task object that deals with part of, or the whole, model training.
    #TODO: Add function to get expected parametrs for the task, and for child task
    #TODO: Add function to get expected output columns for the task, and for child task
    #TODO: Set random seeds.
    """

    def __init__(self, params: dict, input_df: pd.DataFrame | None = None):
        self.params = params
        self.input_df = input_df
        self.outputs = dict()
        self.subtasks: list[tuple[str, Task]] = []

        self.run_or_load_from_cache()

    @abstractmethod
    def run(self):
        """
        Code to perform the task at hand. Save results to self.outputs.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_prediction_steps(self) -> list[(str, BaseEstimator)]:
        """
        Extract relevant steps from dataprep and the train preparation, and the model from train
        """
        raise NotImplementedError()

    @staticmethod
    def get_param_hash(s):
        pass

    def get_sub_tasks_predicion_steps(self):
        steps_per_subtask = [named_subtask[1].get_prediction_steps() for named_subtask in self.subtasks]
        flattened_steps = [step for steplist in steps_per_subtask for step in steplist]
        return flattened_steps

    def check_if_cached(self):
        # param_hash = get_param_hash(self.params)
        return False

    def load_if_cached(self):
        pass

    def run_or_load_from_cache(self):
        if self.check_if_cached() is True:
            print("Experiment Found! Loading cached results.")
            self.load_if_cached()
        else:
            self.run()

    def get_subtask(self, subtask_name: str) -> Task | None:
        """
        #TODO: Perform recursively
        """
        for subtask in self.subtasks:
            if subtask[0] == subtask_name:
                return subtask[1]
        else:
            return
