from __future__ import annotations
import pandas as pd

from abc import ABC, abstractmethod
from task_infra.task import Task
from typing import Type
from sklearn.pipeline import Pipeline


class DataPrep(Task):

    def run(self):
        data_loader = DataLoader.get_loader(self.params['data_loader_params'])
        self.subtasks.append(('DataLoader', data_loader))
        value_clipper = ValueClipper(self.params['clipper_params'], data_loader.outputs['raw_data'])
        self.subtasks.append('ValueClipper', value_clipper)

    def get_prediction_steps(self):
        return self.get_sub_tasks_predicion_steps()

class ValueClipper(Task):
    clipped_suffix = '_clipped'

    def transform(self, df: pd.DataFrame):
        df_transformed = df.assign(
            **{
                f'{col}{self.clipped_suffix}': df[col].clip(bounds['lower'], bounds['upper'])
                for col, bounds in self.params.items()
            }
        )
        return df_transformed

    def fit(self, X, y):
        return self

    def run(self):
        self.outputs['clipped_df'] = self.transform(self.input_df)

    def get_prediction_steps(self):
        return Pipeline(steps=['ValueClipper', self])



class DataLoader(Task):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        raise NotImplementedError()

    @staticmethod
    def get_loader(data_loader_params: dict) -> Type[DataLoader]:
        data_loader = {
            CsvDataLoader.data_type: CsvDataLoader(data_loader_params),
        }.get(data_loader_params['data_type'], None)
        if data_loader is None:
            raise ValueError(f"Data loader for data type {data_loader_params['data_type']} not found.")
        return data_loader

    def get_prediction_steps(self):
        return Pipeline([])


class CsvDataLoader(DataLoader):
    data_type = 'csv'

    def __init__(self, params):
        self.data_path = params['data_path']
        self.additional_load_params = params['additional_load_params']
        super().__init__(params)

    def run(self) -> None:
        self.outputs['raw_data'] = self.load_data()

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path, **self.additional_load_params)
