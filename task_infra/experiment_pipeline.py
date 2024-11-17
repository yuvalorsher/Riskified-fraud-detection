import pandas as pd
import json
from sklearn.pipeline import Pipeline

from task_infra.task import Task
from task_infra.data_preparation import DataPrep


# class ExperimentParams:
#     def __init__(self, path_to_parameters_file: str):
#         self.path_to_parameters_file = path_to_parameters_file
#         self.params_dict: dict | None = None
#
#     def load_parameters(self, override: bool = False):
#         if self.params_dict is not None and override is True:
#             raise ValueError("Parameters already loaded. To overide, call with 'override=True'")
#         self.params_dict = self.load_params_from_file(path_to_parameters_file)
#
#     @staticmethod
#     def load_params_from_json_txt(json_txt: str):
#         return json.loads(json_txt)
#
#     @staticmethod
#     def load_params_from_file(params_file_path: str) -> dict:
#         with open(params_file_path) as f:
#             parameters_dict = json.load(f)
#         return parameters_dict


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
