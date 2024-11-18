from __future__ import annotations
import pandas as pd

from abc import ABC, abstractmethod
from task_infra.task import Task
from typing import Type
from sklearn.pipeline import Pipeline

class DataPrep(Task):
    def __init__(self, params: dict):
        self.params = params
        self.data_loader = None
        self.data_preprocessor = None

    def run(self):
        data_loader = DataLoader.get_loader(self.params['data_loader_params'])
        self.data_loader = data_loader


class DataLoader(Task):
    @abstractmethod
    def __init__(self, params: dict):
        raise NotImplementedError()

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
        return Pipeline()


class CsvDataLoader(DataLoader):
    data_type = 'csv'

    def __init__(self, params):
        self.data_path = params['data_path']
        self.additional_load_params = params['additional_load_params']
        self.run_or_load_from_hash()

    def run(self) -> None:
        self.outputs['raw_data'] = self.load_data()

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path, **self.additional_load_params)
