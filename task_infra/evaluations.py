import numpy as np
import pandas as pd

from task_infra.task import Task
from task_infra.consts import EXCHANGE_RATES, DEDAULT_EXCHANGE_RATE

from typing import Callable
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score


class Evaluator(Task):
    classification_metrics_key = 'classification_metrics'
    required_fee_key = 'required_fee'

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
        print(f"Start Evaluation step.")
        self.outputs[self.classification_metrics_key] = self.get_classification_metrics()
        self.outputs[self.required_fee_key] = self.get_min_fee()
        print(f"Finished Evaluation step.")

    def get_classification_metrics(self):
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
                # ('predicted_true_rate_to_actual_true_rate', lambda y_true, y_pred: y_pred.sum() / y_true.sum()), # Not so relevant when we over-sample train
                # ('confusion_matrix', confusion_matrix),
            ]
        }

    def get_prediction_steps(self):
        return self.get_sub_tasks_predicion_steps()

    def get_min_fee(self) -> float:
        """
        Calculate the fee to meet the reuired ratio of CB to revenue.
        Assumes all (or most) of merchant's declines will also be declined by us.
        """
        approved_transactions = self.test_set.loc[~self.predictions['test']]
        transaction_values = self.convert_currency(
            values=approved_transactions['total_spent'],
            currencies=approved_transactions['currency_code'],
        )
        cb_mask = self.test_target.loc[approved_transactions.index]
        cost_of_cb = transaction_values.loc[cb_mask].sum()
        non_cb_transaction_value = transaction_values.loc[~cb_mask].sum()
        fee = cost_of_cb / (self.params['cost_of_cb_to_revenue_ratio'] * non_cb_transaction_value)
        return fee

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

    @staticmethod
    def convert_currency(
            values: pd.Series,
            currencies: pd.Series,
    ) -> pd.Series:
        exchange_rates = currencies.map(lambda x: EXCHANGE_RATES.get(x, DEDAULT_EXCHANGE_RATE))
        return values*exchange_rates
