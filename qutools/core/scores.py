"""
# Submodule with wrapper functions to work with questionnaire scores data

This submodule contains wrapper functions to work with questionnaire scores data.
It is already written in the "language" of these score-dataframes. The functions
are used primarily in the [qutools.data.data.QuData][qutools.data.data.QuData]
class and should not be used standalone, which is why there will be no examples here.
"""

import numpy as np
import pandas as pd

from .pandas import merge_columns

from .config import MCTaskConfig


def score_mc_items(
    df_scr: pd.DataFrame,
    mc_tasks: list[MCTaskConfig],
    verbose: bool=False,
) -> pd.DataFrame:
    """Function to score multiple-choice (mc) items via the correct value
    defined as the value of the `MCTaskConfig` item dictlist.

    Parameters
    ----------
    df_scr : pd.DataFrame
        DataFrame with MC items to score.
    mc_tasks : list[MCTaskConfig]
        Dictionary of tasks with task-names as keys and `MCTaskConfig`s as
        values.
    verbose : bool
        Wether information on the scored tasks should be printed.

    Returns
    -------
    pd.DataFrame
        DataFrame with MC items scored accoring to the Kprim-rubrics.
    """
    if verbose:
        print("Scoring of multiple-choice items.")

    for task in mc_tasks:
        for item in task.get_items():
            correct_val = item.correct_response
            nan_idx = df_scr[item.name].isna().values
            df_scr[item.name] = (df_scr[item.name] == correct_val).astype(int)
            df_scr.loc[nan_idx, item.name] = np.nan
            if sum(df_scr[item.name])==0 and verbose:
                print(
                    f"Warning: None of the responses of MC-item {item.name} is \n" +
                    f"considered as correct using the comparison value {correct_val}.\n" +
                    "There might be an error in the configuration or the data."
                )

    return df_scr


def construct_kprim_threshold(n_items: int) -> list[int]:
    """ Wrapper to define threshold for grading a multiple choice task using
    the K' (Kprim) procedure.

    For more information on K' refer to:
        Krebs, R. (1997). The Swiss way to score multiple true-false items:
            Theoretical and empirical evidence. In A. J. J. A. Scherpbier,
            C. P. M. van der Vleuten, J. J. Rethans, & A. F. W. van der Steeg
            (Eds.), Advances in medical education (pp. 158-161). Springer
            Netherlands. https://doi.org/10.1007/978-94-011-4886-3_46

    Parameters
    ----------
    n_items : int
        Number of single choice items in the multiple choice task.

    Returns
    -------
    thresholds : list
        The thresholds for 1 and 2 points in a list.
    """
    if n_items==3:
        th = [2, 3]
    elif n_items==4:
        th = [3, 4]
    elif n_items==5:
        th = [3, 4]
    elif n_items==6:
        th = [4, 5]
    elif n_items==7:
        th = [4, 6]
    elif n_items==8:
        th = [5, 7]
    elif n_items==9:
        th = [5, 8]
    elif n_items==10:
        th = [6, 8]
    elif n_items==11:
        th = [6, 9]
    elif n_items==12:
        th = [7, 10]
    else:
        raise NotImplementedError("There is no implementation for Kprim scoring of multiple-choice tasks with less than 3 or more than 12 single items.")
    return th


def apply_kprim_threshold(
    df: pd.DataFrame,
    taskname: str,
    thresholds: list[float]
) -> pd.DataFrame:
    """Function to apply kprim-`thresholds` for task `taskname` to dataframe
    `df`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to apply the thresholds.
    taskname: str
        Name of the task. Must be a column of `df`.
    thresholds : list[float]
        Thresholds for the task as returned by `construct_kprim_threshold`.

    Returns
    -------
    pd.DataFrame
        Dataframe with applied thresholds at the columns specified in the
        thresholds_dict.
    """
    df_ret = df.copy() # Copying df because updates are later performed inplace!
    thresholds = [-np.inf] + thresholds + [np.inf]
    for k in range(len(thresholds)-1):
        row_idx = (df[taskname] >= thresholds[k]) & (df[taskname] < thresholds[k+1])
        df_ret.loc[row_idx, taskname] = k
    return df_ret


def score_mc_tasks(
    df_scr: pd.DataFrame,
    mc_tasks: list[MCTaskConfig],
    verbose: bool=False,
) -> pd.DataFrame:
    """ Function to score multiple-choice (mc) items. The kprim thresholds get
    inferred automatically based on the number of columns per mc item.

    Parameters
    ----------
    df_scr : pd.DataFrame
        DataFrame with MC items to score.
    mc_tasks : list[MCTaskConfig]
        Dictionary of tasks with task-names as keys and `MCTaskConfig`s as
        values.
    verbose : bool
        Wether information on the scored tasks should be printed.

    Returns
    -------
    pd.DataFrame
        DataFrame with MC items scored accoring to the Kprim-rubrics.
    """
    if verbose:
        print("Scoring of multiple-choice tasks.")

    for task in mc_tasks:
        task_item_names = task.get_item_names()
        scoring = task.scoring
        n_items = len(task_item_names)
        df_scr = merge_columns(df_scr, task_item_names, task.name, verbose)

        if scoring == "thresholds":
            if verbose:
                print(f"Threshold-scoring of task {task.name}, items: {task_item_names}")
            mc_thresholds = construct_kprim_threshold(n_items)
            df_scr = apply_kprim_threshold(df_scr, task.name, mc_thresholds)

        if scoring == "summation":
            if verbose:
                print(f"Summation-scoring of task {task.name}, items: {task_item_names}")
            max_score = task.max_score
            df_scr[task.name] = df_scr[task.name].astype(float) / float(n_items) * max_score

    return df_scr


def drop_missing_threshold(
    df_scr: pd.DataFrame,
    spare_columns: list[str],
    threshold: float=0.5,
    verbose: bool=True,
) -> pd.DataFrame:
    """Drops rows from dataframe if more than 100*threshold`% of the fields
    are missing. `spare_columns` are not included in the caluclation.

    Parameters
    ----------
    df_scr : pd.DataFrame
        DataFrame with missings.
    spare_columns : list[str]
        Columns to ignore in the calculation of the missing percentage.
    threshold : float
        Missing part threshold. If the missing amount in a row exeeds this value
        the row will be dropped.
    verbose : bool
        Whether information about the dropped instances should be printed.

    Returns
    -------
    pd.DataFrame
        Dataframe with high-missing rows removed.
    """
    missing_table = df_scr.drop(columns=spare_columns).isna()
    missing_percentage_per_row = missing_table.mean(axis=1).values
    keep_mask = missing_percentage_per_row <= threshold
    if verbose:
        print(f"Dropping {sum(keep_mask==False)} instances due to incompletness.")  # noqa: E712, E501
    return df_scr.loc[keep_mask, :]


def drop_earlystopping_threshold(
    df_scr: pd.DataFrame,
    spare_columns: list[str],
    threshold: float=0.25,
    verbose: bool=True,
) -> pd.DataFrame:
    """Drops rows from dataframe if more than 100*threshold`% of consecutive
    fields at the end (right side) are missing.

    Parameters
    ----------
    df_scr : pd.DataFrame
        DataFrame with missings.
    spare_columns : list[str]
        Columns to ignore in the calculation of the missing percentage.
    threshold : float
        Missing part threshold. If the amount of consecutive missings (counted
        from the right side) in a row exeeds this value the row will be dropped.
    verbose : bool
        Whether information about the dropped instances should be printed.

    Returns
    -------
    pd.DataFrame
        Dataframe with earlystop-missing rows removed.
    """
    missing_table = df_scr.drop(columns=spare_columns).isna()
    missing_table_reversed = missing_table.loc[:, ::-1].astype(float).values

    first_non_missing = np.argmax(missing_table_reversed==False, axis=1)    # noqa: E712, E501
    consecutive_missings_perc = first_non_missing / missing_table_reversed.shape[1]

    keep_mask = consecutive_missings_perc < threshold
    if verbose:
        print(f"Dropping {sum(keep_mask==False)} instances due to earlystopping.")  # noqa: E712, E501
    return df_scr.loc[keep_mask, :]
