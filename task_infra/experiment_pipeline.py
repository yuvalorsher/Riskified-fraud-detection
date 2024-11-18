import pandas as pd
import json
from sklearn.pipeline import Pipeline

from task_infra.task import Task
from task_infra.data_preparation import DataPrep


class Experiment(Task):
    def __init__(self,
                 data_prep_params: dict,
                 train_params: dict, #including model type, HP tuning
                 model_performance_params: dict,
                 ):
        self.data_prep_params = data_prep_params
        self.train_params: pd.DataFrame | None = None
        self.cleaned_data: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        self.run_or_load_from_hash()

    def run(self):
        self.data_prep = DataPrep(self.data_prep_params)
        self.train_model = TrainModel(
            train_df=self.data_prep.outputs['train_df'],
            train_params=self.train_params,
        )

    def get_model_pipline(self) -> Pipeline:
        """
        Extract relevant steps from dataprep and the train preparation, and the model from train
        :return:
        """
        pass
