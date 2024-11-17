import numpy as np
import pandas as pd
from scipy.stats import zscore


def _assert_numerical(df: pd.DataFrame) -> None:
    """
    Assert that all columns in df are numerical
    """
    if df.select_dtypes(exclude='number').size != 0:
        raise (ValueError("df must contain numerical values only"))


def get_max_zscore(df: pd.DataFrame) -> pd.Series:
    """
    Get maximal absolute zscore per column in df.
    """
    _assert_numerical(df)
    zscored = df.apply(lambda s: zscore(s.dropna()))
    return zscored.abs().max()


def sort_cols_by_max_zscore(df):
    max_abs_zscore = get_max_zscore(df)
    sorted_cols = np.argsort(-max_abs_zscore)
    return df.iloc[:, sorted_cols]


def get_cols_w_outliers(df, outlier_zscore: float = 4, exclude: list[str] = None):
    max_abs_zscore = get_max_zscore(df.drop(columns=exclude))
    outlier_cols = max_abs_zscore[lambda s: s > outlier_zscore].index
    return df[outlier_cols]