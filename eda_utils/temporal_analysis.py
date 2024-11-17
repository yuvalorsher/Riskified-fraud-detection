import pandas as pd
from typing import Callable

from matplotlib import pyplot as plt

from eda_utils.general import make_list_if_single_item

AGG_COL = 'price'
DATE_COL = 'datesold'
DEFAULT_FREQ = 'M'


def _convert_to_period_if_needed(
        df: pd.DataFrame,
        date_col: str,
        freq: str | pd.PeriodDtype | None = None
) -> tuple[pd.DataFrame, str]:
    """
    Check that either date_col's dtype is period, or freq is not None (but not both).
    Returns:
        - df with the date_col as period.
        - freq - input freq if not None, otherwise date_col's period frequency.
    """
    assert isinstance(df[date_col].dtype, pd.PeriodDtype) != (
                freq is not None), "Either values of date_col must be period, xor freq must be given."  # xor
    if freq is not None:
        df = df.copy()
        df[date_col] = df[date_col].dt.to_period(freq)
    else:
        freq = df[date_col].iloc[0].freqstr  # Surely there is a nicer way, haven't fount it though
    return df, freq


def add_missing_periods(
        period_indexed_df: pd.DataFrame,
        freq: str,
):
    """
    Given a df with period index, makes sure it has all the periodes from the earliest to the latest, given frequency.
    """
    period_indexed_df = period_indexed_df.sort_index()
    all_periods = pd.period_range(period_indexed_df.index[0], period_indexed_df.index[-1], freq=freq)
    return pd.DataFrame(index=all_periods).merge(period_indexed_df, left_index=True, right_index=True, how='left')


def agg_over_time(
        df: pd.DataFrame,
        agg_col: str,
        date_col: str,
        freq: str | None = None,
        agg_func: str | Callable[[pd.Series], int] = 'mean',
        is_add_missing_periods: bool = True,
) -> pd.Series:
    """
    Aggregates df[agg_col] using agg_func per date_col. date_col's dtype must be period, or freq is not None (but not both).
    If freq is not None, first converts date_col to period.
    If is_add_missing_periods, adds missing time periods with NaNs according to freq (which when None is calculated from date_col).
    """
    df, freq = _convert_to_period_if_needed(df=df, date_col=date_col, freq=freq)
    agged = df.groupby(date_col)[agg_col].agg(agg_func)
    if is_add_missing_periods:
        agged = add_missing_periods(agged, freq)
        # all_periods = pd.period_range(agged.index[0], agged.index[-1], freq=freq)
        # agged = pd.DataFrame(index = all_periods).merge(agged, left_index=True, right_index=True, how='left')
    return agged.iloc[:, 0]


def agg_multiple_cols_over_time(
        df: pd.DataFrame,
        date_col: str,
        agg_cols: list[str],
        freq: str | None = None,
        agg_func: str | Callable[[pd.Series], int] = 'mean',
) -> pd.DataFrame:
    """
    Applies agg_over_time on multiple columns and concatenates into a DF.
    """
    aggs = pd.concat([
        agg_over_time(df, agg_col=agg_col, date_col=date_col, freq=freq, agg_func=agg_func)
        for agg_col in agg_cols
    ], axis=1)
    return aggs


def get_nans_per_period(
        df: pd.DataFrame,
        date_col: str,
        columns: list[str] | pd.Index | None = None,
        freq: str | None = None,
) -> pd.DataFrame:
    """
    Calculates the NaN rate per period defined by date_col. date_col's dtype must be period, or freq is not None (but not both).
    If freq is not None, first converts date_col to period.
    if Columns is None, calculates for all columns but date_col.
    """
    df, freq = _convert_to_period_if_needed(df=df, date_col=date_col, freq=freq)
    columns = columns if columns is not None else df.columns.difference([date_col])
    return agg_multiple_cols_over_time(
        df=df,
        date_col=date_col,
        freq=None,  # date_col converter outside of loop so freq = None
        agg_cols=columns,
        agg_func=lambda s: s.isna().mean(),
    )


def agg_over_time_per_group(
        df: pd.DataFrame,
        date_col: str,
        agg_col: str,
        groupby_cols: str | list[str],
        freq: str | None = None,
        agg_func: str | Callable[[pd.Series], int] = 'mean',
) -> pd.DataFrame:
    """
    Aggregates agg_col over time periods defined by date_col and per groups in groupby_col, and returns the time series per period.
    date_col's dtype must be period, or freq is not None (but not both).
    If freq is not None, first converts date_col to period.
    """
    df, freq = _convert_to_period_if_needed(df=df, date_col=date_col, freq=freq)
    aggregations = [
        agg_over_time(
            df=group_df,
            agg_col=agg_col,
            date_col=date_col,
            freq=None,  # date_col converter outside of loop so freq = None
            agg_func=agg_func,
        ).rename(name) for name, group_df in df.groupby(groupby_cols)
    ]
    return pd.concat(aggregations, axis=1)


def disp_nans_over_time(
        df: pd.DataFrame,
        date_col: str,
        columns: list[str] | pd.Index | None = None,
        freq: str | None = None,
        axis: int | None = 0,
):
    """
    Aggregates the NaN rate per period per column in columns.
    Return a Styler object in which columns are sorted by max NaN rate.
    """
    agged = get_nans_per_period(df, date_col=date_col, columns=columns, freq=freq)
    sorted_cols = agged.max().sort_values(ascending=False).index
    return agged[sorted_cols].style.background_gradient(axis=axis)


def plot_trends(
        df: pd.DataFrame,
        date_col: str,
        trend_cols: str | list[str],
        freq: str | None = None,
        agg_func: str | Callable[[pd.Series], int] = 'mean',
        ax: plt.Axes | None = None,
        figsize: tuple[int, int] | None = None,
) -> plt.Axes:
    """
    Plots the trends of trend_cols over time, aggregated per freq using agg_func
    """
    _, ax = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    trend_cols = make_list_if_single_item(trend_cols)
    trends = agg_multiple_cols_over_time(
        df=df,
        date_col=date_col,
        agg_cols=trend_cols,
        freq=freq,
        agg_func=agg_func,
    )

    ax = trends.plot(ax=ax)
    return ax
