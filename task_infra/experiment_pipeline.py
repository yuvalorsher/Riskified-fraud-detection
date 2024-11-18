import pandas as pd
import json
from sklearn.pipeline import Pipeline

from task_infra.task import Task
from task_infra.data_preparation import DataPrep
from task_infra.train import TrainModel
from task_infra.evaluations import Evaluator


class Experiment(Task):
    def run(self):
        data_prep = DataPrep(self.params['data_prep_params'])
        self.subtasks.append(('DataPrep', data_prep))
        trained_model = TrainModel(
            self.params['train_params'],
            data_set=data_prep.outputs[data_prep.output_df_key],
            dropped_label_dataset=data_prep.get_declined_samples(),
        )
        self.subtasks.append(("TrainedModel", trained_model))
        evaluations = Evaluator(
            params=self.params['evaluation_params'],
            train_set=trained_model.outputs[trained_model.train_set_key],
            train_target=trained_model.outputs[trained_model.train_target_key],
            test_set=trained_model.outputs[trained_model.test_set_key],
            test_target=trained_model.outputs[trained_model.test_target_key],
            declined_test_set=trained_model.outputs[trained_model.declined_test_set_key],
            predictions=trained_model.outputs[trained_model.predictions_key]
        )
        self.subtasks.append(("Evaluations", evaluations))

    def get_prediction_steps(self) -> Pipeline:
        """
        Extract relevant steps from dataprep and the train preparation, and the model from train
        ##TODO: The returned trained model currently does not predict we need to make it work
        :return:
        """
        subtasks_pipeline = self.get_sub_tasks_predicion_steps()
        trained_model = subtasks_pipeline.named_steps
        return trained_model
