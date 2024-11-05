"""
# Submodule with pandas-wrapper functions

This submodule contains some helper functions for handling pandas DataFrames. It
is partly already denoted in the "language" of the questionnaire data (e. g.
`append_missing_text_columns`) and could be furhter abstracted. It mainly
provides these helpers to the other modules and should mostly be not used
standalone, which is why there will be no examples here.
"""

import numpy as np
import pandas as pd


def dataframes_equal(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    check_names: bool=False,
    **kwargs,
) -> bool:
    """Checks wether to pandas dataframes are equal by wrapping `pandas.testing.assert_frame_equal`
    in a try-except block returning true or false

    Parameters
    ----------
    df1 : pd.DataFrame
    df2 : pd.DataFrame
    check_names : bool
        Wether the column names should be identical as well.

    Returns
    -------
    bool
    """
    try:
        pd.testing.assert_frame_equal(df1, df2, kwargs, check_names=check_names)
        return True
    except AssertionError:
        return False


def reorder_column(df: pd.DataFrame, column: str, before: str) -> pd.DataFrame:
    """Wrapper to reorder a specific column in of a pandas dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the columns to reorder.
    column : str
        The column to reorder.
    before : str
        The column, that the other column should be inserted before.

    Returns
    -------
    pd.DataFrame
        The dataframe with reordered columns.
    """
    columns = df.columns.to_list()
    columns.remove(column)
    idx = columns.index(before)
    columns.insert(idx, column)
    df = df[columns]
    return df


def merge_columns(
    df: pd.DataFrame,
    columns: list[str],
    new_colname: str=None,
    verbose: bool=False,
) -> pd.DataFrame:
    """Wrapper to sum columns together (primary for MC-columns). The crux is
    that in case all of the columns entries are `np.nan` the resulting
    merged column keeps this `np.nan`.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the columns to sum.
    columns : list[str]
        Columns to sum together.
    new_colname : str
        New columns name.
    verbose : bool
        Wether information on the merged columns should be printed.

    Returns
    -------
    pd.DataFrame
        The dataframe with merged columns.
    """
    df = df.copy()

    if new_colname is None:
        new_colname = "".join(columns)
    if verbose:
        print(f"Merging {columns} to '{new_colname}'.")

    # Extracting rows where all columns are NaN
    all_na = (df.loc[:, columns].isna().mean(axis=1).to_numpy()) == 1.0
    df[new_colname] = df.loc[:, columns].sum(axis=1, skipna=True)

    df = reorder_column(df, new_colname, columns[0])
    df = df.drop(columns=columns)

    # Replace rows where all columns were NaN
    df.loc[all_na, new_colname] = np.nan
    return df


def merge_and_clip(
    df: pd.DataFrame,
    columns: list[str],
    new_colname: str=None,
    a_min: float=0,
    a_max: float=1,
) -> pd.DataFrame:
    """Wrapper to sum columns together and clip the resulting values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the columns to sum.
    columns : list[str]
        Columns to sum together.
    new_colname : str
        New columns name. If `None` the old column names get joined.
    a_min : float
        The min value for the clip.
    a_max : float
        The max value for the clip

    Returns
    -------
    pd.DataFrame
        The dataframe with merged and clipped columns.
    """
    if new_colname is None:
        new_colname = "".join(columns)
    try:
        df = merge_columns(df, columns, new_colname)
        df[new_colname] = np.clip(df[new_colname].values, a_min=a_min, a_max=a_max)
    except KeyError:
        print(f"Warning: not all of {columns} are available in dataframe.")
    return df


def unite_str_columns(
    df: pd.DataFrame,
    cols: list[str]=None,
    new_name: str="united",
    drop: bool=False,
    sep: str=" "
) -> pd.DataFrame:
    """Wrapper function to unite string type columns to a single column like the
    R-dplyr "unite" function. Replacing NAs with "" beforehand is strongly
    recommended.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to unite.
    cols : list[str]
        List with the column names of the columns to unite. If `None` all are
        united.
    new_name : str
        Name of the united column.
    drop : bool
        Whether to drop the single columns.
    sep : str
        Separation characters.

    Returns
    -------
    pd.DataFrame
    """
    if cols is None:
        cols = df.columns.to_list()

    df_new = df.copy()
    df_new[new_name] = ""

    for col in cols:
        s = df_new[col].astype(str).replace("nan", "")
        if col == cols[0]:
            df_new[new_name] = df_new[new_name] + s
        else:
            df_new[new_name] = df_new[new_name] + sep + s

    df_new[new_name] = df_new[new_name].str.strip()
    df_new.loc[df_new[new_name]=="", new_name] = np.nan

    df_new = reorder_column(df_new, new_name, cols[0])

    if drop:
        df_new = df_new.drop(columns=cols)

    return df_new


def pivot_to_wide(
    df: pd.DataFrame,
    value_cols: str="predicted_scores",
    index_cols: list[str]=["ID"],
    column_names: str="item",
) -> pd.DataFrame:
    """Pivots a DataFrame with responses in rows to wide format with a row
    for each questionnaire edit.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with item-responses in rows.
    value_cols : str
        Columns containing the values that should be inserted in the
        pivoted dataframe.
    index_cols : list[str]
        Columns that should be used as index for pivoting. Defaults to the
        ID-column.
    column_names : str
        Column name of the column whichs entries should be used as the columns
        of the pivoted DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    df = df.pivot(index=index_cols, columns=column_names, values=value_cols)
    try:
        df.columns = df.columns.get_level_values(1)
    except IndexError:
        pass
    df = df.reset_index()
    return df


def append_missing_text_cols(
    df: pd.DataFrame,
    text_task_names: list[str],
    index_cols: list[str]
) -> pd.DataFrame:
    """Appends missing text columns to the dataframe and fills them with zeros.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to which the columns should be appended.
    text_task_names : list[str]
        Names of the all text columns.
    index_cols : list[str]
        Columns that should be kept in the resulting DataFrame.
    """
    missing_text_cols = np.setdiff1d(text_task_names, df.columns.to_list())
    df[missing_text_cols] = np.nan
    df = df[index_cols + text_task_names]
    return df


def append_mc_columns(
    df: pd.DataFrame,
    df_scores: pd.DataFrame,
    mc_cols: list[str],
    all_task_names: list[str],
    merge_on: str="ID",
    **kwargs,
) -> pd.DataFrame:
    """Appends additional score-columns to DataFrame `df`. Used for the multiple
    choice (mc) columns. Keeps only the id-, and scores columns in the result.
    The scores table is optional if there are no multiple choice tasks in the
    qudata.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to which the columns should be appended.
    df_scores : pd.DataFrame
        Dataframe containing the additional columns.
    mc_cols : list[str]
        Names of the multiple choice columns.
    all_task_names : list[str]
        Names of all task-columns. Used for reshaping.
    merge_on : str
        Column for the merge.

    Returns
    -------
    pd.DataFrame
    """
    if len(mc_cols) != 0:

        df = pd.merge(
            left=df,
            right=df_scores[[merge_on] + mc_cols],
            on=merge_on,
            validate=kwargs.get("validate", None),
        )

    df = df[[merge_on] + all_task_names]
    return df
