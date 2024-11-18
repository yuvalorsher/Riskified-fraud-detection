import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from task_infra.task import Task
from typing import Callable
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix


class Evaluator(Task):
    classifier_metrics_key = 'classifier_metrics'
    break_even_fee_key = 'break_even_fee'

    def __init__(
            self,
            params: dict,
            train_set: pd.DataFrame,
            train_target: pd.DataFrame,
            test_set: pd.DataFrame,
            test_target: pd.DataFrame,
            declined_test_set: pd.DataFrame,
            predictions: dict,
    ):
        self.train_set = train_set
        self.train_target = train_target
        self.test_set = test_set
        self.test_target = test_target
        self.declined_test_set = declined_test_set
        self.predictions = predictions
        super().__init__(params)

    def run(self):
        self.outputs[self.classifier_metrics_key] = self.get_classifier_metrics()
        self.outputs[self.break_even_fee_key] = self.get_break_even_fee()

    def get_classifier_metrics(self):
        return {
            metric_name: self._get_train_test_metric_dict(
                train_y_true=self.train_target,
                train_y_pred=self.predictions['train'],
                test_y_true=self.test_target,
                test_y_pred=self.predictions['test'],
                metric=metric
            ) for metric_name, metric in [
                ('accuracy', accuracy_score),
                ('precision', precision_score),
                ('recall', recall_score),
                ('roc_auc_score', roc_auc_score),
                ('f1_score', f1_score),
                ('confusion_matrix', confusion_matrix),
            ]
        }

    def get_prediction_steps(self):
        return Pipeline(steps=[])

    def get_break_even_fee(self):
        pass

    @staticmethod
    def _get_train_test_metric_dict(
            train_y_true: pd.Series,
            train_y_pred: pd.Series,
            test_y_true: pd.Series,
            test_y_pred: pd.Series,
            metric: Callable[[pd.Series, pd.Series], int | float | np.ndarray],
    ):
        return dict(
                train=metric(train_y_true, train_y_pred),
                test=metric(test_y_true, test_y_pred),
        )

