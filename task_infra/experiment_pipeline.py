import pandas as pd
import json
from sklearn.pipeline import Pipeline

from task_infra.task import Task
from task_infra.data_preparation import DataPrep


class Experiment(Task):

    def run(self):
        data_prep = DataPrep(self.params['data_prep_params'])
        self.subtasks.append(('DataPrep', data_prep))
        train_model = TrainModel(
            train_df=data_prep.outputs['train_df'],
            train_params=self.params['train_params'],
        )
        self.subtasks.append(("TrainModel", train_model))

    def get_model_pipline(self) -> Pipeline:
        """
        Extract relevant steps from dataprep and the train preparation, and the model from train
        :return:
        """
        return self.get_sub_tasks_predicion_steps()
