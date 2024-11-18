import pandas as pd

from abc import ABC, abstractmethod
from task_infra.task import Task
from typing import Type
from sklearn.pipeline import Pipeline


class Evaluator(Task):

    def run(self):
        train_test_splitter = TrainTestSplit(['params'])
        data_loader = DataLoader.get_loader(self.params['data_loader_params'])
        self.subtasks.append(('DataLoader', data_loader))
        value_clipper = ValueClipper(self.params['clipper_params'], data_loader.outputs['raw_data'])
        self.subtasks.append(('ValueClipper', value_clipper))

    def get_prediction_steps(self):
        return self.get_sub_tasks_predicion_steps()

