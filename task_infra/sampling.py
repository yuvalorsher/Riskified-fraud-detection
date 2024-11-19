from __future__ import annotations
import pandas as pd

from abc import abstractmethod

# from imblearn.over_sampling import SMOTE

from task_infra.task import Task
# from task_infra.consts import RANDOM_STATE, LABEL_COL, SMOTE_FILLNA_VALUE


class Sampler(Task):
    output_df_key = 'sampled_data'

    @abstractmethod
    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @staticmethod
    def get_sampler(sampler_params: dict, input_df) -> Sampler:
        data_sampler = {
            RandomSampler.sampler_type: RandomSampler,
            RandomOverSampler.sampler_type: RandomOverSampler,
            # SmoteOverSampler.sampler_type: SmoteOverSampler,
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


class RandomOverSampler(Sampler):
    sampler_type = "random_over_sampler"

    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        positive_mask = df[self.params['label_col']] == self.params['positive_label']
        positive_examples = df.loc[positive_mask]
        negative_examples = df.loc[~positive_mask]
        n_positives_samples = round(len(negative_examples) * self.params['positive_to_negative_ratio'])
        oversampled_positives = positive_examples.sample(n=n_positives_samples, replace=True)
        return pd.concat([oversampled_positives, negative_examples])


# class SmoteOverSampler(Sampler):
#     """
#     I did not have time to make this work, mainly due to lack of proper dtyping.
#     """
#     sampler_type = "smote_over_sampler"
#
#     def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         sm = SMOTE(random_state=RANDOM_STATE, sampling_strategy=self.params['positive_to_negative_ratio'])
#         X_res, _ = sm.fit_resample(df[self.params['distribution_cols']].fillna(SMOTE_FILLNA_VALUE), df[LABEL_COL] == self.params['positive_label'])
#         return df.loc[X_res.index]
