import pandas as pd


def copy_if_not_inplace(df: pd.DataFrame, inplace: bool) -> pd.DataFrame:
    df = df if inplace else df.copy()
    return df


def make_list_if_single_item(x) -> list:
    x = x if isinstance(x, list) else [x]
    return x
