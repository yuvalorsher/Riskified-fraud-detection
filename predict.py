import numpy as np
import pandas as pd
import pickle

path_to_data = './Data/transactions.csv'
path_to_model = 'yuval_trained_model.pkl'
predictions_csv_savepath = 'predictions.csv'
predict_samples = None  # To make quicker, set to None if you want to predict on all transactions


def predictions_to_csv(predictions: np.ndarray, pred_index: pd.Index, save_path: str) -> None:
    pd.Series(data=predictions, index=pred_index, name='preds').to_csv(save_path)


if __name__ == '__main__':
    transactions = pd.read_csv(path_to_data)
    with open(path_to_model, 'rb') as handle:
        trained_model = pickle.load(handle)

    if predict_samples is not None:
        transactions = transactions.sample(predict_samples, random_state=42)

    preds = trained_model.predict(transactions)
    predictions_to_csv(preds, transactions.index, predictions_csv_savepath)
