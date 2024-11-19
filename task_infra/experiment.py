from sklearn.pipeline import Pipeline

from task_infra.task import Task
from task_infra.data_preparation import DataPrep
from task_infra.train import TrainModel
from task_infra.evaluations import Evaluator

import pickle


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
        evaluation = Evaluator(
            params=self.params['evaluation_params'],
            train_set=trained_model.outputs[trained_model.train_set_key],
            train_target=trained_model.outputs[trained_model.train_target_key],
            test_set=trained_model.outputs[trained_model.test_set_key],
            test_target=trained_model.outputs[trained_model.test_target_key],
            declined_test_set=trained_model.outputs[trained_model.declined_test_set_key],
            predictions=trained_model.outputs[trained_model.predictions_key]
        )
        self.subtasks.append(("Evaluation", evaluation))

    def get_prediction_steps(self):
        subtasks_named_steps = self.get_sub_tasks_predicion_steps()
        return subtasks_named_steps

    def get_trained_model(self) -> Pipeline:
        """
        Collects all steps from subtasks and wraps in fitted Pipeline.
        """
        return Pipeline(self.get_prediction_steps())

    def save_trained_model(self, save_path: str) -> None:
        trained_model: Pipeline = self.get_trained_model()
        with open(save_path, 'wb') as handle:
            pickle.dump(trained_model, handle)

    def save_experiment(self, save_path: str) -> None:
        with open(save_path, 'wb') as handle:
            pickle.dump(self, handle)
