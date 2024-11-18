from __future__ import annotations
import pandas as pd

from abc import ABC, abstractmethod
from typing import Type
from sklearn.pipeline import Pipeline

from task_infra.task import Task
from task_infra.consts import LABEL_COL
from task_infra.sampling import Sampler


class DataPrep(Task):
    output_df_key = 'clean_df'

    def run(self):
        data_loader = DataLoader.get_loader(self.params['data_loader_params'])
        self.subtasks.append(('DataLoader', data_loader))
        data_sampler = Sampler.get_sampler(self.params['dataset_sampler_params'])
        self.subtasks.append(('SampleData', data_sampler))
        label_clearer = ClearLabel(self.params['clear_label_params'], data_sampler.outputs[data_sampler.output_df_key])
        self.subtasks.append(('ClearLabel', label_clearer))
        value_clipper = ValueClipper(self.params['clipper_params'], label_clearer.outputs[label_clearer.cleared_df_key])
        self.subtasks.append(('ValueClipper', value_clipper))
        self.outputs[self.output_df_key] = value_clipper.outputs[value_clipper.output_df_key]

    def get_prediction_steps(self):
        return self.get_sub_tasks_predicion_steps()


class ClearLabel(Task):
    cleared_df_key = 'cleared_df'
    dropped_df_key = 'dropped_df'

    def __init__(self, params: dict, input_df: pd.DataFrame):
        self.labels_to_clear = params['labels_to_clear']
        super().__init__(params, input_df)

    def run(self):
        print(f"Dropping labels: {self.labels_to_clear}")
        mask = self.get_mask(self.input_df)
        self.outputs[self.cleared_df_key] = self.input_df.loc[mask]
        self.outputs[self.dropped_df_key] = self.input_df.loc[~mask]
        print(f"Cleared {len(self.outputs[self.dropped_df_key])} labels, out of total {len(self.input_df)}.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = self.get_mask(df)
        return df[mask]

    def get_mask(self, df: pd.DataFrame) -> pd.Series:
        """Returns bool Series, True indicates labels to keep."""
        return ~df[LABEL_COL].isin(self.labels_to_clear)

    def get_prediction_steps(self):
        return Pipeline(steps=['ClearLabel', self])


class ValueClipper(Task):
    clipped_suffix = '_clipped'
    output_df_key = 'clipped_df'

    def transform(self, df: pd.DataFrame):
        df_transformed = df.assign(
            **{
                f'{col}{self.clipped_suffix}': df[col].clip(bounds['lower'], bounds['upper'])
                for col, bounds in self.params.items()
            }
        )
        return df_transformed

    def fit(self, x, y):
        return self

    def run(self):
        self.outputs[self.output_df_key] = self.transform(self.input_df)

    def get_prediction_steps(self):
        return Pipeline(steps=['ValueClipper', self])


class DataLoader(Task):
    output_df_key = 'raw_data'

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        raise NotImplementedError()

    @staticmethod
    def get_loader(data_loader_params: dict) -> DataLoader:
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
        self.outputs[self.output_df_key] = self.load_data()

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path, **self.additional_load_params)
