import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import PairGrid
from eda_utils.consts import FREQ_SHORT_TO_FULL


def calc_nrows_from_ncols(nplots: int, ncols: int = 3, nrows: int | None = None) -> int:
    """
    Calculate how many rows are needed in a subplot, given nplots and desired ncols.
    If nrows is given, function just returns nrows.
    """
    assert nplots > 0 and ncols > 0, f"nplots and ncols must be a positive integers. Values are nplots={nplots}, ncols={ncols}"
    nrows = int(np.ceil(nplots / ncols)) if nrows is None else nrows
    return nrows


def create_subplots(
        nplots: int,
        nrows: int,
        ncols: int,
        figsize: tuple[int, int] | None = None
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Some boiler-plate code for plotting multiple EDA plots
    """
    assert nplots > 0 and ncols > 0, f"nplots and ncols must be a positive integers. Values are nplots={nplots}, ncols={ncols}"
    nrows = calc_nrows_from_ncols(nplots=nplots, ncols=ncols, nrows=nrows)
    assert nrows > 0, f"nrows must all be a positive integers. nrows={nrows}"
    figsize = (3 * ncols, 3 * nrows) if figsize is None else figsize
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs_1d = axs.reshape(-1)  # easier view
    for i in range(nplots, ncols * nrows):
        axs_1d[i].set_axis_off()

    return fig, axs


def plot_cat_counts(
        df: pd.DataFrame,
        cat_cols: list[str],
        ncols: int = 3,
        nrows: int | None = None,
        figsize: tuple[int, int] | None = None
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot the value-counts of categorical columns in df using a bar plot.
    """
    nrows = calc_nrows_from_ncols(nplots=len(cat_cols), ncols=ncols, nrows=nrows)
    figsize = (3 * ncols, 3 * nrows) if figsize is None else figsize
    fig, axs = create_subplots(nplots=len(cat_cols), nrows=nrows, ncols=ncols, figsize=figsize)
    axs_1d = axs.reshape(-1)  # easier view
    for i, cat_col in enumerate(cat_cols):
        df[cat_col].value_counts().sort_index().plot.bar(ax=axs_1d.reshape(-1)[i])
        axs_1d[i].set_title(cat_col)
        axs_1d[i].set_xlabel('')
    plt.tight_layout()
    return fig, axs


def plot_boxplots(
        df: pd.DataFrame,
        boxplot_cols: list[str],
        ncols: int = 3,
        nrows: int | None = None,
        figsize: tuple[int, int] | None = None
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot the boxplots for boxplot_cols
    #TODO: Very similar code to plot_cat_counts, share code.
    """
    fig, axs = create_subplots(nplots=len(boxplot_cols), nrows=nrows, ncols=ncols, figsize=figsize)
    axs_1d = axs.reshape(-1)  # easier view
    for i, boxplot_col in enumerate(boxplot_cols):
        df[boxplot_col].plot(ax=axs_1d.reshape(-1)[i], kind='box')
        axs_1d[i].set_title(boxplot_col)
        axs_1d[i].set_xlabel('')
    plt.tight_layout()
    return fig, axs


def plot_temporal_distribution(dates: pd.Series, freq: str) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the number of observations distirbution per frequency (month, quarter, year, etc.).
    freq may be any of the frequencies allowed by pd's to_period(freq).
    """
    fig, ax = plt.subplots()
    dates.dt.to_period(freq=freq).dt.start_time.dt.date.value_counts().sort_index().plot.bar(ax=ax)
    ax.set_xlabel(FREQ_SHORT_TO_FULL.get(freq, freq))
    return fig, ax


# def _get_period_start_times_from_series(
#     s: pd.Series,
# ) -> pd.Series:
#     return s.dt.start_time.dt.date

# def _get_period_start_times_from_period_index(
#     ind: pd.PeriodIndex,
# ) -> pd.Series:
#     return _get_period_start_times_from_series(ind.to_series())

# # def plot_temporal_distribution(dates: pd.Series, freq: str) -> tuple[plt.Figure, plt.Axes]:
# #     """
# #     Plots the number of observations distirbution per frequency (month, quarter, year, etc.).
# #     freq may be any of the frequencies allowed by pd's to_period(freq).
# #     """
# #     fig, ax = plt.subplots()
# #     dates.dt.to_period(freq=freq).dt.start_time.dt.date.value_counts().sort_index().plot.bar(ax=ax)
# #     ax.set_xlabel(FREQ_SHORT_TO_FULL.get(freq, freq))
# #     return fig, ax

# def plot_temporal_distribution(dates: pd.Series, freq: str) -> tuple[plt.Figure, plt.Axes]:
#     """
#     Plots the number of observations distirbution per frequency (month, quarter, year, etc.).
#     freq may be any of the frequencies allowed by pd's to_period(freq).
#     """
#     fig, ax = plt.subplots()
#     counts_per_period = dates.dt.to_period(freq=freq).value_counts().sort_index()
#     counts_per_period_full = add_missing_periods(counts_per_period, freq).fillna(0)
#     x = _get_period_start_times_from_period_index(counts_per_period_full.index).values
#     y = counts_per_period_full.values.flatten()
#     print(y)
#     ax.bar(x, y)
#     ax.set_xlabel(FREQ_SHORT_TO_FULL.get(freq, freq))
#     return fig, ax

def plot_scatter_matrix(
        df: pd.DataFrame,
        include_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
        # figsize: tuple[int, int] | None = None,
        alpha: float = 0.1,
) -> PairGrid:
    include_cols = df.columns if include_cols is None else include_cols
    exclude_cols = [] if exclude_cols is None else exclude_cols
    # ncols = len(include_cols) - len(exclude_cols)
    # figsize = (ncols, ncols) if figsize is None else figsize
    # axs = pd.plotting.scatter_matrix(df[include_cols].drop(columns=exclude_cols), figsize=figsize, alpha=alpha)
    axs = sns.pairplot(df[include_cols].drop(columns=exclude_cols), plot_kws={'alpha': alpha})
    plt.tight_layout()
    return axs