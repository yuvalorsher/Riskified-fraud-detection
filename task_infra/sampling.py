from __future__ import annotations
import pandas as pd

from abc import ABC, abstractmethod
from task_infra.task import Task
from typing import Type
from sklearn.pipeline import Pipeline

from task_infra.consts import DATE_COL


class Sampler(Task):
    output_df_key = 'sampled_data'

    @abstractmethod
    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @staticmethod
    def get_sampler(sampler_params: dict, input_df) -> Sampler:
        data_sampler = {
            RandomSampler.sampler_type: RandomSampler,
            OverSampler.sampler_type: OverSampler,
        }.get(sampler_params['sampler_type'], None)
        if data_sampler is None:
            raise ValueError(f"Data sampler of type {sampler_params['sampler_type']} not found.")
        else:
            data_sampler = data_sampler(sampler_params['additional_sampler_params'], input_df)
        return data_sampler

    def run(self) -> None:
        print(f"Sampling dataset.")
        self.outputs[self.output_df_key] = self.sample_data(self.input_df)

    def get_prediction_steps(self):
        return []


class RandomSampler(Sampler):
    sampler_type = "random_sampler"

    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(n=self.params['n'], frac=self.params['frac'])


class OverSampler(Sampler):
    sampler_type = "over_sampler"

    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        positive_mask = df[self.params['label_col']] == self.params['positive_label']
        positive_examples = df.loc[positive_mask]
        negative_examples = df.loc[~positive_mask]
        n_positives_samples = round(len(negative_examples) * self.params['positive_to_negative_ratio'])
        oversampled_positives = positive_examples.sample(n=n_positives_samples, replace=True)
        return pd.concat([oversampled_positives, negative_examples])
