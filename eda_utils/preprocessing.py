import pandas as pd
from eda_utils.consts import VALUES_COL_KEY, GROUPBY_COL_KEY, LABEL_ENCODED_POSTFIX, TARGET_ENCODED_POSTFIX
from eda_utils.general import copy_if_not_inplace
from sklearn.preprocessing import LabelEncoder


# Missing values imputation
def impute_cols(
    df: pd.DataFrame,
    impute_col: str,
    values_col: str,
    groupby_cols: list[str] | pd.Index | None = None,
) -> pd.Series:
    """
    Impute missing values in impute_col by the mean of values_col. If groupby_cols is not empty, returns the mean per group.
    """
    if groupby_cols is not None and len(groupby_cols)>0:
        return df[impute_col].fillna(df.groupby(groupby_cols)[values_col].transform('mean'))
    else:
        return df[impute_col].fillna(df[values_col].mean())


def impute_by_group_mean(
        df: pd.DataFrame,
        impute_col_by_groupby: dict,
        inplace: bool = False,
) -> pd.DataFrame:
    """
    Imputes columns defined in impute_col_by_groupby's keys, according to the dictionary defined in their values.
    #TODO: Lose redundant functionality allowing calculating the mean from a different col from the one that's being imputed. i.e. dict should only be {'col1': ['col2', 'col3']}
    Input:
     - impute_col_by_groupby: dict who's keys are columns to impute and values are dictionaries with the target column from which to calc the mean, and columns to group by when calculating mean.
    Example:
        impute_col_by_groupby = {
        'col1': {
            VALUES_COL_KEY: 'col1',
            GROUPBY_COL_KEY: ['col2', 'col3'],
            },
        }
        then col1 will be imputed by the mean of df.groupby([col2,col3])[col1]
    """
    if not inplace:
        df = df.copy()
    for impute_col, impute_by in impute_col_by_groupby.items():
        df[impute_col] = impute_cols(df, impute_col, impute_by[VALUES_COL_KEY], impute_by[GROUPBY_COL_KEY])
    if not inplace:
        return df


# Encodings
def one_hot_encode(
        df: pd.DataFrame,
        col_encoding: dict,
        original_cols_to_keep: list[str] | pd.Index | None = None,
) -> pd.DataFrame:
    """
    Encodes the columns in col_encoding's keys, and passes the kwargs in their values to pd.get_dummies()
    If original_cols_to_keep is None, keeps all original encoded cols.
    Returns the whole df, with the encoded columns
    #TODO: More detailed docstring
    """
    cols_to_encode = list(col_encoding.keys())
    original_cols_to_keep = original_cols_to_keep if original_cols_to_keep is not None else cols_to_encode
    bad_cols_to_keep = set(original_cols_to_keep) - set(cols_to_encode)
    assert len(
        bad_cols_to_keep) == 0, f"original_cols_to_keep must be a subset of col_encoding's keys. Additional cols are: {bad_cols_to_keep}"
    none_encoded_cols = df.columns.difference(col_encoding.keys())
    encodings = pd.concat([pd.get_dummies(df[column], **args) for column, args in col_encoding.items()], axis=1)
    return pd.concat([df[none_encoded_cols], df[original_cols_to_keep], encodings], axis=1)


def label_encode(
        df: pd.DataFrame,
        col_encoding: dict,
        encoded_postfix: str = LABEL_ENCODED_POSTFIX,
        inplace: bool = False,
):
    """
    Ignores missing values.
    """
    df = copy_if_not_inplace(df, inplace)
    label_encoder = LabelEncoder()
    for column, categories in col_encoding.items():
        label_encoder.fit(categories)
        missing = df[column].isna()
        if sum(missing) > 0:
            print(f'{missing.sum()} missing values in {column} are not label encoded.')
        df.loc[~missing, f'{column}{encoded_postfix}'] = label_encoder.transform(df.loc[~missing, column])
    return df


def target_endode(
        df: pd.DataFrame,
        col_encodings: dict,
        encoded_postfix: str = TARGET_ENCODED_POSTFIX,
        inplace: bool = False,
) -> pd.DataFrame:
    """
    Encodes the categories in col_encodings according to a predefined mapping.
    Each column in col_encodings keys is encoded using the mappings defined in its values.
    E.g. col_encodings = dict(Embarked={'C': 0, 'Q': 1, 'S': 1}). Mappings should be valid input to pd.Series.map()
    """
    df = copy_if_not_inplace(df, inplace)
    for cat_col, encoding in col_encodings.items():
        df[f'{cat_col}{encoded_postfix}'] = df[cat_col].map(encoding)
    return df


def target_encode_train(
        df: pd.DataFrame,
        target: str | pd.Series,
        col_encoding_aggs: dict,
        encoded_postfix: str = TARGET_ENCODED_POSTFIX,
        inplace: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Encodes the categories in columns defined in col_encoding_aggs according to aggregates calculated from target.
    Also returns the mappings to be used on test set where target is not accessible for prediction.
    Target is either column name in df or a series (in case x,y are already seperated).
    col_encoding_aggs is a dict with column names to encode as keys and the type of aggregation to apply to target_col as values,
    e.g.
    col_encoding_aggs = dict(Embarked=pd.Series.mode)
    """
    df = copy_if_not_inplace(df, inplace)
    col_encodings = dict()
    if type(target) is str:
        target = df[target]
    for cat_col, agg in col_encoding_aggs.items():
        encoding = target.groupby(df[cat_col]).agg(agg)
        col_encodings[cat_col] = encoding

    df = target_endode(df, col_encodings, encoded_postfix, inplace)

    return df, col_encodings
