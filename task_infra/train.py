from __future__ import annotations

import numpy as np
import pandas as pd

from abc import abstractmethod
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

from task_infra.task import Task
from task_infra.sampling import Sampler
from task_infra.consts import DATE_COL


class TrainModel(Task):
    predictions_key = 'predictions'
    train_set_key = 'train_set'
    train_target_key = 'train_target'
    test_set_key = 'test_set'
    test_target_key = 'test_target'
    declined_test_set_key = 'desclined_set'

    def __init__(self, params: dict, data_set: pd.DataFrame, dropped_label_dataset):
        """
        droppe_label_dataset is the transactions that were dropped due to having 'declined' label.
        """
        self.dropped_label_dataset = dropped_label_dataset
        super().__init__(params, input_df=data_set)

    def run(self):
        print(f"Start training step.")
        train_test_split = TrainTestSplitter(self.params['train_test_split_params'], self.input_df)
        self.subtasks.append(('TrainTestSplit', train_test_split))
        trainset_sample = Sampler.get_sampler(
            self.params['trainset_sampler_params'],
            train_test_split.outputs[train_test_split.train_df_key]
        )
        self.subtasks.append(('TrainsetSampler', trainset_sample))
        self.outputs[self.train_set_key] = trainset_sample.outputs[trainset_sample.output_df_key]
        self.outputs[self.test_set_key] = train_test_split.outputs[train_test_split.test_df_key]
        model_features_train = PrepareTrainFeatures(
            self.params['features_params'],
            trainset_sample.outputs[trainset_sample.output_df_key]
        )
        self.subtasks.append(('PrepairFeaturesTrain', model_features_train))
        model_target_train = PrepareTrainTarget(
            self.params['target_params'],
            trainset_sample.outputs[trainset_sample.output_df_key]
        )
        self.subtasks.append(('PrepairTargetTrain', model_target_train))
        self.outputs[self.train_target_key] = model_target_train.outputs[model_target_train.target_key]
        test_features = model_features_train.transform(train_test_split.outputs[train_test_split.test_df_key])
        test_target = model_target_train.transform(train_test_split.outputs[train_test_split.test_df_key])
        self.outputs[self.test_target_key] = test_target
        declined_testset = train_test_split.get_test_set(self.dropped_label_dataset)
        self.outputs[self.declined_test_set_key] = declined_testset
        declined_features = model_features_train.transform(declined_testset)
        classifier = Classifier.get_classifier(
            classifier_params=self.params['classifier_params'],
            train_set=self._get_dataset_dict(
                features=model_features_train.outputs[model_features_train.features_key],
                target=model_target_train.outputs[model_target_train.target_key]
            ),
            test_set=self._get_dataset_dict(
                features=test_features,
                target=test_target
            ),
            declined_features=declined_features,
        )
        # No time, but here would be another splitter to allow K-fold and early stopping.
        classifier.fit()
        self.subtasks.append(('Classifier', classifier))
        self.outputs[self.predictions_key] = dict(
            train=classifier.predict(model_features_train.outputs[model_features_train.features_key]),
            test=classifier.predict(test_features),
            declined=classifier.predict(declined_features)
        )

    def get_prediction_steps(self):
        return self.get_sub_tasks_predicion_steps()

    @staticmethod
    def _get_dataset_dict(features: pd.DataFrame, target: pd.Series) -> dict:
        return dict(
            features=features,
            target=target
        )


class Classifier(Task):
    model_key = 'model'
    threshold_key = 'threshold'

    def __init__(self, params: dict, train_set: dict, test_set: dict, declined_features: pd.DataFrame):
        self.train_set = train_set
        self.test_set = test_set
        self.declined_features = declined_features
        super().__init__(params)

    @abstractmethod
    def _fit_estimator(self, **kwargs) -> None:
        raise NotImplementedError

    def fit(self, **kwargs) -> None:
        """
        fit estimator + calculate required threshold to achieve required_approval_rate
        """
        print(f"Fitting estimator.")
        self._fit_estimator(**kwargs)
        test_period_flow = pd.concat([self.declined_features, self.test_set['features']])
        probabilities = self.outputs[self.model_key].predict_proba(test_period_flow)
        self.outputs[self.threshold_key] = self.get_required_threshold(
            approved_probabilities=probabilities[:, 0],
            required_approval_rate=self.params['required_approval_rate'],
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Use predict_proba and self.threshold to predict
        """
        charge_back_probabilities = self.outputs[self.model_key].predict_proba(features)
        return pd.Series(
            data=charge_back_probabilities[:,0]<self.outputs[self.threshold_key],
            index=features.index
        )

    @staticmethod
    def get_classifier(classifier_params: dict, train_set: dict, test_set: dict, declined_features: pd.DataFrame) -> Classifier:
        classifier = {
            XgbClassifier.classifier_type: XgbClassifier,
        }.get(classifier_params['classifier_type'], None)
        if classifier is None:
            raise ValueError(f"Classifier of type {classifier_params['classifier_type']} not found.")
        else:
            classifier = classifier(
                params=classifier_params['additional_model_params'],
                train_set=train_set,
                test_set=test_set,
                declined_features=declined_features
            )
        return classifier

    @staticmethod
    def get_required_threshold(approved_probabilities: np.ndarray, required_approval_rate: float) -> float:
        """
        approved_probabilities: if probability > threshold then they are approved.
        i.e., these are the probabilities for label 0.
        """
        descending_probabilities = np.sort(approved_probabilities)[::-1]
        threshold_ind = int(np.ceil(len(descending_probabilities) * required_approval_rate) - 1)
        return descending_probabilities[threshold_ind]

    def get_prediction_steps(self):
        return [('Classifier', self.outputs[self.model_key])]


class XgbClassifier(Classifier):
    classifier_type = "xgb"

    def run(self):
        self.outputs[self.model_key] = xgb.XGBClassifier(**self.params['model_initialize_params'])

    def _fit_estimator(self, **kwargs) -> None:
        clf = self.outputs[self.model_key]
        clf.fit(
            self.train_set['features'],
            self.train_set['target'],
            eval_set=[(self.test_set['features'], self.test_set['target'])],
            **self.params['model_fit_params'],
            **kwargs,
        )


class PrepareTrainTarget(Task):
    """
    This class will perform target transformations.
    """
    target_key = 'target'

    def run(self) -> None:
        print("Preparing target.")
        map_target = MapTarget(self.params["target_mapping"], self.input_df)
        self.subtasks.append(('MapTarget', map_target))
        self.outputs[self.target_key] = map_target.outputs[map_target.target_after_mapping_key]

    def get_prediction_steps(self):
        return []

    def transform(self, df: pd.DataFrame) -> pd.Series:
        return Pipeline(steps=self.subtasks).transform(df)


class PrepareTrainFeatures(Task):
    features_key = 'features_df'

    def run(self) -> None:
        print("Starting to prepare train features.")
        drop_columns = DropColumns(self.params["drop_columns"], self.input_df)
        self.subtasks.append(('DropColumn', drop_columns))
        one_hot = OneHot(self.params['one_hot_params'], drop_columns.outputs[drop_columns.df_after_drop_key])
        self.subtasks.append(('OneHot', one_hot))
        self.outputs[self.features_key] = one_hot.outputs[one_hot.df_after_onehot_key]
        print("Finished preparing train features.")

    def get_prediction_steps(self):
        return self.get_sub_tasks_predicion_steps()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return Pipeline(steps=self.subtasks).transform(df)


class MapTarget(Task):
    target_after_mapping_key = 'target_after_mapping'

    def run(self) -> None:
        print("Mapping target.")
        self.outputs[self.target_after_mapping_key] = self.transform(self.input_df)

    def transform(self, df: pd.DataFrame) -> pd.Series:
        return df[self.params['target_col']].map(self.params['target_mapping_dict']).astype(bool)

    def get_prediction_steps(self) -> Pipeline:
        return Pipeline(steps=[])


class OneHot(Task):
    df_after_onehot_key = 'df_after_onehot'
    encoder_key = 'encoder'

    def run(self) -> None:
        print(f"One Hot Encoding: {self.params['columns_to_onehot']}")
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        enc.fit(self.input_df[self.params['columns_to_onehot']])
        self.outputs[self.encoder_key] = enc
        encoded_df = self.transform(self.input_df)
        self.outputs[self.df_after_onehot_key] = encoded_df

    def transform(self, df: pd.DataFrame):
        one_hot_encoded = self.outputs[self.encoder_key].transform(df[self.params['columns_to_onehot']])
        one_hot_col_names = self.outputs[self.encoder_key].get_feature_names_out(self.params['columns_to_onehot'])
        encoded_df = pd.DataFrame(index=df.index, data=one_hot_encoded, columns=one_hot_col_names)
        final_df = pd.concat([df.drop(columns=self.params['columns_to_onehot']), encoded_df], axis=1)
        return final_df

    def get_prediction_steps(self):
        return [('OneHot', self)]


class DropColumns(Task):
    df_after_drop_key = 'df_after_drop'

    def run(self) -> None:
        print(f"Dropping columns: {self.params['columns_names']}.")
        self.outputs[self.df_after_drop_key] = self.transform(self.input_df)

    def transform(self, df: pd.DataFrame):
        return df.drop(columns=self.params['columns_names'])

    def get_prediction_steps(self):
        return [('DropColumns', self)]


class TrainTestSplitter(Task):
    """
    #TODO: Add functionality to split by dataset size proportion (i.e. find the relevant date according to ratio),
    minimal number of positive labels, etc.)
    """
    train_df_key = 'train_df'
    test_df_key = 'test_df'

    def run(self) -> None:
        print("Splitting train-test sets.")
        self.outputs[self.train_df_key] = self.get_train_set(self.input_df)
        self.outputs[self.test_df_key] = self.get_test_set(self.input_df)
        self.run_asserts()

    def get_train_set(self, data_set: pd.DataFrame) -> pd.DataFrame:
        return data_set[
            data_set[DATE_COL].between(
                self.params['train_start_date'],
                self.params['test_start_date'],
                inclusive='left'
            )
        ]

    def get_test_set(self, data_set: pd.DataFrame) -> pd.DataFrame:
        return data_set[
            data_set[DATE_COL].between(
                self.params['test_start_date'],
                self.params['test_end_date'],
                inclusive='left'
            )
        ]

    def get_prediction_steps(self):
        return []

    def run_asserts(self) -> None:
        """
        Ideally this gets a list of which assert objects to run, indicated in params json, but I had no time to implement.
        Currently asserts dfs are not empty, can be configured to allow this.
        """
        assert len(self.outputs['train_df']) > 0, f"Training data df is empty! When splitting with params {self.params}"
        assert len(self.outputs['test_df']) > 0, f"Test data df is empty! When splitting with params {self.params}"

