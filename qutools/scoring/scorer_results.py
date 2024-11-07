"""
# Scorer Results

The `QuScorerResults` class is a container for the results of a `QuScorer` fit or
evaluation. It provides methods for the analysis, visualization, and comparison
of such results. The results are stored in a long-format dataframe, which can be
converted to a wide-format dataframe for further analysis.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from json import load as json_load
from json import dump as json_dump

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..data.data import QuData
from ..data.id_splits import IDsSplit, IDsKFoldSplit

from ..core.io import read_data, empty_or_create_dir, write_data, path_suffix_warning
from ..core.trainulation import evaluate_predictions, print_scores
from ..core.pandas import append_mc_columns, append_missing_text_cols, pivot_to_wide



@dataclass
class QuScorerResults:
    qudata: QuData
    df_preds: pd.DataFrame=None
    df_fit_histories: pd.DataFrame=None
    id_split: IDsSplit=None
    unit_col: str="task"
    target_col: str="score"
    prediction_col: str="score_pred"
    validate_scores: Literal["none", "warn", "error"]="warn"


    def __init__(
        self,
        qudata: QuData,
        df_preds: pd.DataFrame=None,
        df_fit_histories: pd.DataFrame=None,
        id_split: IDsSplit=None,
        unit_col: str="task",
        target_col: str="score",
        prediction_col: str="score_pred",
        validate_scores: Literal["none", "warn", "error"]="warn",
        **kwargs,
    ) -> None:
        """Initializes a `QuScorerResults` object.

        Parameters
        ----------
        qudata : QuData
            The QuData that was used for the scorer's training. This is needed
            to return wide-format data including multiple choice columns.
        df_preds : pd.DataFrame
            The training predictions
        df_fit_histories : pd.DataFrame
            The fit history data as a dataframe.
        id_split : IDsSplit
            The id-split used in the evaluation.
        unit_col : str
            The name of the column denoting the scorable units.
        target_col : str
            The name of the target column.
        prediction_col : str
            The name that should be used for the target column
        validate_scores : Literal["none", "warn", "error"]
            Wether the scores should be validated using the `quconfig`.
        """
        self.qudata = qudata
        if "quconfig" in kwargs:
            print("Overwriting `qudata`'s argument `quconfig`.")
            self.quconfig = kwargs.pop("quconfig")
        else:
            self.quconfig = qudata.quconfig

        self.id_col =  self.quconfig .id_col
        self.unit_col = unit_col
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.info_cols = ["split", "mode"]
        self.id_split = id_split

        self.validate_scores = validate_scores

        if df_preds is not None:
            self.__validate_columns(df_preds)
            if not kwargs.get("supress_quconfig_validation", False):
                df_val = df_preds.drop(columns=self.info_cols)
                df_val = df_val[[self.id_col, self.unit_col, self.prediction_col]]
                self.__validate_long_df_preds(df_val)

        self.df_preds = df_preds
        self.df_fit_histories = df_fit_histories


    ## Validation
    def __validate_long_df_preds(self, df_val: pd.DataFrame) -> None:
        self.quconfig.validate_long_df_scr(
            df_scr=df_val,
            value_col=self.prediction_col,
            name_col=self.unit_col,
            units="no_mc",
            validate_scores=self.validate_scores,
            all_units=False,
        )

    def __validate_columns(self, df_preds: pd.DataFrame):
        cols = {
            "Id-column": self.id_col,
            "Unit-column": self.unit_col,
            "Target-column": self.target_col,
            "Prediction-column": self.prediction_col,
            "Info-column (1)": "split",
            "Info-column (2)": "mode",
        }
        for k, v in cols.items():
            if v not in df_preds.columns:
                raise KeyError(
                    f"{k} \"{v}\" is not present in the passed dataframe `df_preds`. \n" +
                    f"Present Columns: {df_preds.columns.to_list()}"
                )


    ## Information
    def get_split_nos(self) -> list[int]:
        """
        Returns
        -------
        list[int]
            The CV-split numbers in the instance.
        """
        if self.df_preds is None:
            return None
        ls = list(self.df_preds["split"].unique())
        ls.sort()
        return ls

    def get_n_splits(self) -> int:
        """
        Returns
        -------
        int
            Number of CV-splits in the instance.
        """
        if self.df_preds is None:
            return 0
        n = len(self.df_preds["split"].unique())
        return n

    def get_test_n(self) -> int:
        """
        Returns
        -------
        int
            Number of test/validation-samples in the instance.
        """
        if self.df_preds is None:
            return 0
        n = np.sum(self.df_preds["mode"] == "test")
        return n

    def _has_test_data(self) -> bool:
        return self.get_test_n() > 0


    ## Appending and removing split-predictions
    def append(
        self,
        df_trn: pd.DataFrame=None,
        df_tst: pd.DataFrame=None,
        fit_history: dict=None,
        idx_split: int|None=None,
        **kwargs,
    ):
        """Appends a training, test/validation, and fit history dataframe to
        the existing data in the instance.

        Parameters
        ----------
        df_trn : pd.DataFrame
            Dataframe with the training data and predictions.
        df_tst : pd.DataFrame
            Dataframe with the test/validation data and predictions.
        fit_history : dict
            The train history dataframe.
        idx_split : int
            A fixed index to label the split with
        """
        supress_validation = kwargs.get("supress_validation", False)

        if idx_split is None:
            idx_split = self.get_n_splits()

        if df_trn is not None:
            self._check_for_split_idx(idx_split=idx_split)
            if not supress_validation:
                self.__validate_long_df_preds(df_trn)
            self._append_df_pred(df_trn, n_split=idx_split, mode="train")

        if df_tst is not None:
            if not supress_validation:
                self.__validate_long_df_preds(df_tst)
            self._append_df_pred(df_tst, n_split=idx_split, mode="test")

        if fit_history is not None:
            self._append_fit_history(fit_history=fit_history, n_split=idx_split)

    def _check_for_split_idx(self, idx_split: int) -> None:
        if self.df_preds is None:
            return
        if (self.df_preds["split"] == idx_split).any():
            print(
                f"Warning: This QuScorerResults object already contains a split labelled {idx_split}.\n" +
                "  You might want to alter the split label / number you have set."
            )

    def _append_df_pred(
        self,
        df_pred: pd.DataFrame,
        n_split: int,
        mode: Literal["train", "test"],
    ) -> None:
        df_pred["split"] = n_split
        df_pred["mode"] = mode
        self.__validate_columns(df_pred)

        if self.df_preds is None:
            self.df_preds = df_pred
        else:
            df_preds = self.df_preds
            df_preds = pd.concat([df_preds, df_pred])
            df_preds = df_preds.reset_index(drop=True)
            self.df_preds = df_preds

    def _append_fit_history(
        self,
        fit_history: dict|pd.DataFrame,
        n_split: int,
    ) -> None:
        if isinstance(fit_history, dict):
            fit_history = pd.DataFrame(fit_history)
        df_fh = fit_history
        df_fh["split"] = n_split
        if self.df_fit_histories is None:
            self.df_fit_histories = df_fh
        else:
            df_fh = pd.concat([self.df_fit_histories, df_fh])
            self.df_fit_histories = df_fh

    def remove(self, idx: int|str):
        """Removes a split from the instance.

        Parameters
        ----------
        idx : int | str
            The index / label of the split to be removed.
        """
        df_preds = self.df_preds
        if df_preds is not None:
            # if (df_preds["split"]==idx).sum() == 0:
            #     print(f"Split-idx {idx} not present in QuScorerResults, nothing to drop.")
            df_preds = df_preds[df_preds["split"]!=idx]
            self.df_preds = df_preds

        df_fh = self.df_fit_histories
        if df_fh is not None:
            df_fh = df_fh[df_fh["split"]!=idx]
            self.df_fit_histories = df_fh




    ## Retrieving wide-format
    def get_wide_txt_only_predictions(self) -> pd.DataFrame:
        """Returns a "wide"-dataframe, containing the predictions of the
        text-items.
        Wide-format retrievals always return "test"-predictions.

        Returns
        -------
        pd.DataFrame
            A wide-format dataframe containing the test-edit-wise / person-wise
            predictions. A `"split"`-column is denoting the CV-split number and
            a `"split"`-column is denoting the predicition(-row) belonging to the
            train or the test/validation data.
        """
        return self._get_wide_df(omit_mc=True, append_mc_only=False)

    def get_wide_txt_only_targets(self) -> pd.DataFrame:
        """Returns a "wide"-dataframe, containing the targets of the
        text-items.
        Wide-format retrievals always return "test"-targets/predictions.

        Returns
        -------
        pd.DataFrame
            A wide-format dataframe containing the test-edit-wise / person-wise
            targets. A `"split"`-column is denoting the CV-split number and
            a `"split"`-column is denoting the predicition(-row) belonging to the
            train or the test/validation data.
        """
        return self._get_wide_df(omit_mc=True, mode="target", append_mc_only=False)

    def get_wide_predictions(
        self,
        mc_units: Literal["mc_items", "mc_tasks"]="mc_items",
        append_mc_only: bool=True,
    ) -> pd.DataFrame:
        """Returns a "wide"-dataframe, containing the predictions of the
        text-items, combined with predictions from the MC-items (as contained)
        in `df_scores` in a "test-edit-wise" format. The scores table is
        optional, if there are no multiple choice tasks in the qudata.
        Wide-format retrievals always return "test"-predictions.

        Parameters
        ----------
        mc_units : Literal["mc_items", "mc_tasks"]
        append_mc_only : bool

        Returns
        -------
        pd.DataFrame
            A wide-format dataframe containing the test-edit-wise / person-wise
            predictions. A `"split"`-column is denoting the CV-split number and
            a `"split"`-column is denoting the predicition(-row) belonging to the
            train or the test/validation data.
        """
        return self._get_wide_df(
            mode="prediction",
            mc_units=mc_units,
            append_mc_only=append_mc_only,
        )

    def get_wide_targets(
        self,
        mc_units: Literal["mc_items", "mc_tasks"]="mc_items",
        append_mc_only: bool=True,
    ) -> pd.DataFrame:
        """Returns a "wide"-dataframe, containing the targets of the
        text-items, combined with predictions from the MC-items (as contained)
        in `df_scores` in a "test-edit-wise" format. This is mainly a helper
        function, to enable the evaluation without passing QuestionnaireData.
        Wide-format retrievals always return "test"-predictions.
        Wide-format retrievals always return "test"-targets/predictions.

        Parameters
        ----------
        mc_units : Literal["mc_items", "mc_tasks"]
        append_mc_only : bool

        Returns
        -------
        pd.DataFrame
            A wide-format dataframe containing the test-edit-wise / person-wise
            targets. A `"split"`-column is denoting the CV-split number and
            a `"split"`-column is denoting the predicition(-row) belonging to the
            train or the test/validation data.
        """
        return self._get_wide_df(
            mode="target",
            mc_units=mc_units,
            append_mc_only=append_mc_only,
        )

    def __append_no_txt_instances(self, df: pd.DataFrame, mc_units: Literal["mc_items", "mc_tasks"]) -> pd.DataFrame:
        split_ids = self.df_preds["split"].unique()

        text_cols = self.quconfig.get_text_task_names()

        mc_scored = mc_units == "mc_tasks"
        df_scr = self.qudata.get_scr(mc_scored=mc_scored, verbose=False)
        df_miss_tst = df_scr[~df_scr["ID"].isin(df["ID"])]
        only_mc_mask = df_miss_tst[text_cols].isna().mean(axis=1).values == 1.
        df_miss_tst: pd.DataFrame = df_miss_tst[only_mc_mask]

        n_miss = df_miss_tst.shape[0]

        if n_miss > 0:
            if self.id_split is not None:
                df_ids = self.id_split.df_ids.copy()
                df_miss_tst = df_miss_tst.merge(df_ids[["ID", "split"]])
            else:
                df_miss_tst["split"] = np.random.choice(split_ids, n_miss)

            df_miss_trn = []
            for _, row in df_miss_tst.iterrows():
                split_id = row["split"]
                other_splits = [s for s in split_ids if s != split_id]
                df_miss_trn_ = pd.DataFrame(len(other_splits) * [row]).copy()
                df_miss_trn_["split"] = other_splits
                df_miss_trn.append(df_miss_trn_)

            df_miss_trn = pd.concat(df_miss_trn)
            df_miss_trn["mode"] = "train"
            df_miss_tst["mode"] = "test"

            df_miss = pd.concat([df_miss_trn, df_miss_tst])
            df_miss = df_miss[df.columns]
            df = pd.concat([df, df_miss]).reset_index(drop=True)

        return df


    def _get_wide_df(
        self,
        omit_mc: bool=False,
        mode: Literal["target", "prediction"]="prediction",
        mc_units: Literal["mc_items", "mc_tasks"]="mc_items",
        append_mc_only: bool=True,
    ) -> pd.DataFrame:
        mc_itemnames = self.quconfig.get_mc_item_names()
        text_task_cols = self.quconfig.get_text_columns("tasks")
        if not omit_mc:
            if len(mc_itemnames) > 0:
                if not self.qudata.mc_item_scores_available() and mc_units == "mc_items":
                    raise ValueError(
                        "If you want to append the multiple choice item-scores to the predictions " +
                        "you need to pass QuestionnaireData, that can obtain single item-scores, i.e., " +
                        "it must be initialized with an non-mc-scored table."
                    )
                mc_scored = mc_units == "mc_tasks"
                df_scr = self.qudata.get_scr(mc_scored=mc_scored, verbose=False)
                df_scr = df_scr.copy().fillna(0)

                if mc_units == "mc_items":
                    mc_cols: list[str]=self.quconfig.get_mc_item_names()
                    all_task_cols: list[str]=self.quconfig.get_scores_columns("mc_items")
                elif mc_units == "mc_tasks":
                    mc_cols: list[str]=self.quconfig.get_mc_task_names()
                    all_task_cols: list[str]=self.quconfig.get_scores_columns("mc_tasks")

        if mode == "target":
            value_cols = self.target_col
        elif mode == "prediction":
            value_cols = self.prediction_col

        df_ret = self.df_preds
        index_cols = [self.quconfig.id_col, "mode", "split"]
        df_ret = pivot_to_wide(
            df=df_ret,
            value_cols=value_cols,
            index_cols=index_cols,
            column_names=self.unit_col,
        )
        df_ret.index.name = None
        df_ret.columns.name = None


        df_ret = append_missing_text_cols(
            df=df_ret,
            text_task_names=text_task_cols,
            index_cols=index_cols,
        )

        if len(mc_itemnames) != 0 and not omit_mc:
            df_ret_mc = append_mc_columns(
                df=df_ret,
                df_scores=df_scr,
                mc_cols=mc_cols,
                all_task_names=all_task_cols,
            )
            df_ret_mc[index_cols] = df_ret[index_cols]
            df_ret = df_ret_mc
            df_ret = df_ret[index_cols + all_task_cols]


        if append_mc_only:
            df_ret = self.__append_no_txt_instances(df_ret, mc_units=mc_units)


        df_ret = df_ret.fillna(0)
        df_ret = df_ret.sort_values(by=self.id_col)
        df_ret = df_ret.reset_index(drop=True)
        df_ret["split"] = df_ret["split"].astype(int)

        return df_ret


    ## Retrieving long-format
    def get_long_df(self) -> pd.DataFrame:
        """Returns a long-version of the contained data, i.e., alls CV-splits
        concatenated with a `"split"`-column denoting the CV-split number.

        Returns
        -------
        pd.DataFrame
        """
        return self.df_preds.copy()


    ## Retrieve Fit-Histories
    def get_fit_histories_dataframe(self) -> pd.DataFrame:
        """Returns the fit-histories as a dataframe. Generates the necessairy
        dataframe, if the fit histories previously have been stored in dict-form.

        Returns
        -------
        pd.DataFrame
        """
        if self.df_fit_histories is None:
            return None
        self.df_fit_histories = self.df_fit_histories.reset_index(drop=True)
        return self.df_fit_histories.copy()


    ## Evaluation
    #   Wrappers
    def evaluation_text_tasks(
        self,
        splitwise: bool=False,
        mode: Literal["train", "test"]="test",
        evaluate_na_filled: bool=True,
    ) -> None:
        """Evaluation of the text-tasks only.

        Parameters
        ----------
        splitwise : bool
            Wether a CV-splitwise evluation or a "concatenated" overall evaluation
            should be presentet
        mode : Literal["train", "test"]
            Wether to evaluate train or test predictions
        evaluate_na_filled : bool
            Wether to evaluate the predictions with missings filled with 0s.

        Raises
        ------
        ValueError
        """
        mode = self.__validate_eval_mode(mode=mode)

        if not splitwise:
            self._evaluate_text_tasks(mode=mode)
            print("")
            self._evaluate_impossible_preds(mode=mode)
            if evaluate_na_filled:
                print("")
                self._evaluate_text_tasks_na_filled(mode=mode)
        else:
            self._evaluate_splitwise(mode=mode)

    def evaluation_with_mc(
        self,
        mode: Literal["train", "test"]="test",
        include_mc_only: bool=False,
    ) -> None:
        """Evaluation combining questionnaire text-task predictions and
        multiple choice-tasks.

        Parameters
        ----------
        mode : Literal["train", "test"]
            Wether to evaluate train or test predictions
        """
        mode = self.__validate_eval_mode(mode=mode)

        mc_itemnames = self.quconfig.get_mc_item_names()
        if len(mc_itemnames) == 0:
            return

        # Missings filled, with MC tasks
        self._evaluate_with_mc_tasks(
            mode=mode,
            include_mc_only=include_mc_only,
        )

        # Missings filled, with MC items
        if self.qudata.mc_item_scores_available():
            print("")
            self._evaluate_with_mc_items(
                mode=mode,
                include_mc_only=include_mc_only,
            )
        else:
            print(
                "Warning: The QuestionnaireData passed does not carry mc-item wise information. " +
                "There will be no information on evaluation including the mc-tasks on item-level."
            )

    def evaluation(
        self,
        mode: Literal["train", "test"]="test",
        include_mc_only: bool=False,
        evaluate_na_filled: bool=True,
    ) -> None:
        """Prints a verbose evaluation of the test/validation predictions in 3
        forms:
        1. Basic evaluation of the text-columns.
        2. Evaluation together with the MC-columns as single items.
        3. Evaluation together with the MC-tasks with threshold-scoring.
        The scores table is optional, if there are no multiple choice tasks in
        the contained qudata.

        Parameters
        ----------
        mode : Literal["train", "test"]
            Wether to evaluate train or test predictions
        include_mc_only : bool
            Wether to include only the multiple choice columns in the evaluation.
        evaluate_na_filled : bool
            Wether to evaluate the predictions with missings filled with 0s.

        """
        self.evaluation_text_tasks(
            splitwise=False,
            mode=mode,
            evaluate_na_filled=evaluate_na_filled,
        )
        print("")
        self.evaluation_with_mc(
            mode=mode,
            include_mc_only=include_mc_only,
        )

    def get_confusion_matrix(self, mode: Literal["train", "test"]="test") -> np.ndarray:
        """Get the confusion matrix of the text-column scoring.

        Parameters
        ----------
        mode : Literal["train", "test"]
            Wether to evaluate train or test predictions

        Returns
        -------
        np.ndarray
        """
        mode = self.__validate_eval_mode(mode=mode)

        df = self.get_long_df()
        df = df[df["mode"] == mode]

        y1 = df[self.target_col].astype(int).values
        y2 = df[self.prediction_col].astype(int).values

        mat = confusion_matrix(y1, y2)
        return mat

    def taskwise_evaluation(self, only_avg: bool=True) -> pd.DataFrame:
        """Taskwise evaluation of the text-tasks.

        Parameters
        ----------
        only_avg : bool
            Wether to return only the average values or the individual task
            evaluations as well.
        """
        quconfig = self.quconfig
        text_tasks = quconfig.get_text_task_names()
        df = self.get_long_df()
        df = df[df["mode"]=="test"]
        df = df.drop(columns=["mode", "split"])

        df_ret = pd.DataFrame()
        for tt in text_tasks:
            df_ = df[df["task"]==tt]
            n_r = df_.shape[0]

            y_true = df_["score"].values
            y_pred = df_["score_pred"].values

            acc = accuracy_score(y_true, y_pred)
            f1_w = f1_score(y_true, y_pred, average='weighted')
            f1_m = f1_score(y_true, y_pred, average='macro')
            kappa = cohen_kappa_score(y_true, y_pred)
            qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

            df_ret[tt] = [n_r, acc, f1_w, f1_m, kappa, qwk]

        df_ret.index = ["N Responses", "Accuracy", "F1 (weighted)", "F1 (macro)", "Cohens Kappa", "Quadratic Weighted Kappa"]
        df_ret = df_ret.T
        df_ret["N Responses"] = df_ret["N Responses"].astype(int)

        df_avg = pd.DataFrame(df_ret.mean(axis=0)).T
        df_avg.index = ["Average"]

        df_ret = pd.concat([df_avg, df_ret])

        if only_avg:
            return df_avg
        return df_ret


    #   Helpers
    def __validate_eval_mode(
        self,
        mode: Literal["train", "test"]
    ) -> Literal["train", "test"]:
        if mode not in ["train", "test"]:
            raise ValueError(f"Argument `mode` must be of `[\"train\", \"test\"]`. Got \"{mode}\".")

        if not self._has_test_data() and mode=="test":
            print("No test-data available, switching to `mode=\"train\"`.")
            mode = "train"

        return mode

    def __get_true_pred_cols(
        self,
        units: Literal["mc_tasks", "mc_items", "no_mc"],
    ) -> tuple[list[str], list[str]]:
        cols = self.quconfig.get_scores_columns(units)
        cols_true = [it + "_true" for it in cols]
        cols_pred = [it + "_pred" for it in cols]
        return cols_true, cols_pred

    def __print_overall_evaluation_metrics(
        self,
        df: pd.DataFrame,
        cols_pred: list[str],
        cols_true: list[str],
    ) -> None:
        y_true = df[cols_true].values.reshape((-1,))
        y_pred = df[cols_pred].values.reshape((-1,))
        print(f" - N-Persons = {df.shape[0]}")
        print(f" - N-Units = {len(cols_true)}")
        print(f" - N-Responses = {len(y_true)}")
        print(f" - Acc = {accuracy_score(y_true, y_pred):.3f}")
        print(f" - F1 (weighted) = {f1_score(y_true, y_pred, average='weighted'):.3f}")
        print(f" - F1 (macro) = {f1_score(y_true, y_pred, average='macro'):.3f}")
        print(f" - Cohens Kappa = {cohen_kappa_score(y_true, y_pred):.3f}")
        print(f" - Quadratic Weighted Kappa = {cohen_kappa_score(y_true, y_pred, weights='quadratic'):.3f}")

    #   Single Elements
    def _evaluate_splitwise(
            self,
            mode: Literal["train", "test"]="test",
            n_labels: int=None,
        ):
        """Evaluation of the predictions.

        Parameters
        ----------
        mode : Literal["train", "test"]
            Wether to evaluate train or test predictions
        n_labels : int
            To overwrite the number of labels contained in `y_true`. Necessary
            depending on the metrics used.
        """
        scores = {}
        idxs = self.df_preds["split"].unique()
        for idx in idxs:
            df = self.df_preds
            df = df[df["mode"]==mode]
            df = df[df["split"]==idx]
            scores = evaluate_predictions(
                df[self.target_col],
                df[self.prediction_col],
                mode=mode,
                n_labels=n_labels,
                scores_dict=scores,
            )
        print_scores(scores)

    def _get_text_task_evaluation(self, mode: Literal["train", "test"]="test") -> dict[str, float]:
        df_long = self.get_long_df()
        df_long = df_long[df_long["mode"]==mode]
        n_tasks = len(df_long[self.unit_col].unique())
        y_true = df_long[self.target_col].values
        y_pred = df_long[self.prediction_col].values
        return {
            "n_pers": len(df_long[self.id_col].unique()),
            "n_tasks": n_tasks,
            "n_responses": len(y_true),
            "acc": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
            "f1_macro": f1_score(y_true, y_pred, average='macro'),
            "cohens_kappa": cohen_kappa_score(y_true, y_pred),
            "quadratic_weighted_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        }


    def _evaluate_text_tasks(self, mode: Literal["train", "test"]="test") -> None:
        print(f"Valid Text Responses Only ({mode})")
        dct = self._get_text_task_evaluation(mode=mode)
        print(f" - N-Persons = {dct['n_pers']}")
        print(f" - N-Text-tasks = {dct['n_tasks']}")
        print(f" - N-Responses = {dct['n_responses']}")
        print(f" - Acc = {dct['acc']:.3f}")
        print(f" - F1 (weighted) = {dct['f1_weighted']:.3f}")
        print(f" - F1 (macro) = {dct['f1_macro']:.3f}")
        print(f" - Cohens Kappa = {dct['cohens_kappa']:.3f}")
        print(f" - Quadratic Weighted Kappa = {dct['quadratic_weighted_kappa']:.3f}")

    def _evaluate_text_tasks_na_filled(
        self,
        mode: Literal["train", "test"]="test",
    ) -> None:
        df_pred = self.get_wide_txt_only_predictions()
        df_pred = df_pred.drop(columns="split").fillna(0)
        df_pred = df_pred[df_pred["mode"]==mode].reset_index(drop=True)
        df_true = self.get_wide_txt_only_targets()
        df_true = df_true.drop(columns="split").fillna(0)
        df_true = df_true[df_true["mode"]==mode].reset_index(drop=True)
        df = pd.merge(
            left=df_pred,
            right=df_true,
            how="left",
            on=self.id_col,
            suffixes=["_pred", "_true"],
        )
        cols_true, cols_pred = self.__get_true_pred_cols("no_mc")
        print(f"Evaluation without MC Columns, with Missings Filled ({mode})")
        self.__print_overall_evaluation_metrics(
            df=df, cols_pred=cols_pred, cols_true=cols_true,
        )

    def _evaluate_impossible_preds(
        self,
        mode: Literal["train", "test"]="test",
    ) -> None:
        text_task_cols = self.quconfig.get_text_columns("tasks")
        max_scores = self.quconfig.get_max_scores()
        txt_max_scores = np.array(
            [[val for key, val in max_scores.items() if key in text_task_cols]]
        )

        df = self.get_wide_txt_only_predictions()
        df = df[df["mode"]==mode].reset_index(drop=True)
        df_pred = df[text_task_cols].copy()

        too_large_pctg = 100. * np.mean(df_pred.values > txt_max_scores)
        print(f"Scores Larger than possible ({mode})")
        print(f" - {too_large_pctg:.2f} %")

    def _evaluate_with_mc_tasks(
        self,
        mode: Literal["train", "test"]="test",
        include_mc_only: bool=False,
    ) -> None:
        df_pred: pd.DataFrame = self.get_wide_predictions(
            mc_units="mc_tasks",
            append_mc_only=include_mc_only
        )
        df_pred = df_pred.drop(columns="split").fillna(0)
        df_pred = df_pred[df_pred["mode"]==mode].reset_index(drop=True)
        df_true = self.qudata.get_scr(mc_scored=True, verbose=False)
        df_true = df_true.fillna(0)
        df_true = df_true.reset_index(drop=True)
        df = pd.merge(
            left=df_pred,
            right=df_true,
            how="left",
            on=self.id_col,
            suffixes=["_pred", "_true"],
        )
        cols_true, cols_pred = self.__get_true_pred_cols("mc_tasks")
        print(f"Multiple-Choice Tasks Scored, Missings Filled ({mode})")
        self.__print_overall_evaluation_metrics(
            df=df, cols_pred=cols_pred, cols_true=cols_true,
        )

    def _evaluate_with_mc_items(
        self,
        mode: Literal["train", "test"]="test",
        include_mc_only: bool=False,
    ) -> None:
        df_pred: pd.DataFrame = self.get_wide_predictions(
            mc_units="mc_items",
            append_mc_only=include_mc_only
        )
        df_pred = df_pred.drop(columns="split").fillna(0)
        df_pred = df_pred[df_pred["mode"]==mode].reset_index(drop=True)
        df_true = self.qudata.get_scr(mc_scored=False, verbose=False)
        df_true = df_true.fillna(0)
        df_true = df_true.reset_index(drop=True)
        df = pd.merge(
            left=df_pred,
            right=df_true,
            how="left",
            on=self.id_col,
            suffixes=["_pred", "_true"],
        )
        cols_true, cols_pred = self.__get_true_pred_cols("mc_items")
        print(f"Multiple Choice as Single Items, Missings Filled ({mode})")
        self.__print_overall_evaluation_metrics(
            df=df, cols_pred=cols_pred, cols_true=cols_true,
        )


    #   Visualizations
    def plot_confusion_matrix(
        self,
        mode: Literal["train", "test"]="test",
        save_path: str=None,
        **kwargs,
    ) -> Figure:
        """Plot the confusion matrix of the text-column scoring as a heatmap.

        Parameters
        ----------
        mode : Literal["train", "test"]
            Wether to evaluate train or test predictions
        save_path : str
            A path to save the plot to. If `None` the plot will not be saved.

        Returns
        -------
        Figure
            The plot as a `matplotlib.figure.Figure`
        """
        mat = self.get_confusion_matrix(mode=mode)
        mat_rnormed = 100 * mat / np.sum(mat, axis=1, keepdims=True)
        pctgs = np.round(mat_rnormed, 0).astype(int)
        labels = np.array([
            [f"{p} %" for p in p_row]
            for p_row in pctgs
        ])

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 5)))
        sns.heatmap(mat_rnormed, cmap="mako", annot=labels, fmt="", ax=ax) # annot=True, fmt=".0f"
        _ = ax.set_xlabel(kwargs.get("xlabel", 'Vorhergesagte Scores'))
        _ = ax.set_ylabel(kwargs.get("ylabel", 'Wahre Scores'))

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return fig

    def plot_fit_history(
            self,
            trn_bins: int=50,
            tst_bins: int=30,
            save_path: str=None,
            **kwargs,
        ) -> Figure:
        """Plots the fit history "averaged" over the different CV-splits.
        The splits might yield nuanced differences in the `"epoch"`s-column
        values because not all splits might be equally long. Therefore the
        epochs-column gets binned / discretized to better visualize the overall
        trends.

        Parameters
        ----------
        trn_bins : int
            Bin-count for the train-metrics (loss). If $\\leq 0$ no binning is
            performed.
        tst_bins : int
            Bin-count for the test/validation-metrics (loss). If $\\leq 0$ no
            binning is performed.
        save_path : str
            A path to save the plot to. If `None` the plot will not be saved.

        Returns
        -------
        Figure
            The plot as a `matplotlib.figure.Figure`.
        """

        df_fh = self.get_fit_histories_dataframe()
        if df_fh is None:
            print("No fit-history data available - nothing to plot.")
            return

        test_history_available = "eval_loss" in df_fh.columns

        if test_history_available:
            df_fh_trn = df_fh[["epoch", "loss"]].copy().dropna()
            df_fh_tst = df_fh[["epoch", "eval_loss", "eval_accuracy"]].copy().dropna()

        else:
            df_fh_trn = df_fh[["epoch", "train_loss"]].copy().dropna()
            df_fh_trn = df_fh_trn.rename(columns={"train_loss": "loss"})

        if self.id_split.get_n_splits() > 1:
            n_epochs = df_fh_trn["epoch"].max()
            trn_bins = np.linspace(0, n_epochs, trn_bins)
            trn_labels = trn_bins[1:]

            if test_history_available:
                tst_bins = np.linspace(0, n_epochs, tst_bins)
                tst_labels = tst_bins[1:]

            df_fh_trn["epoch"] = pd.cut(
                df_fh_trn["epoch"],
                bins=trn_bins,
                labels=trn_labels,
            ).astype(float)

            if test_history_available:
                df_fh_tst["epoch"] = pd.cut(
                    df_fh_tst["epoch"],
                    bins=tst_bins,
                    labels=tst_labels,
                ).astype(float)

        fig: Figure
        ax1: Axes
        ax2: Axes
        plt.rcParams.update({'font.size': 12})

        fig, ax1 = plt.subplots(figsize=(8, 6))
        if test_history_available:
            sns.lineplot(df_fh_tst, x="epoch", y="eval_loss", errorbar="sd", color="royalblue", ax=ax1)
        sns.lineplot(df_fh_trn, x="epoch", y="loss", errorbar="sd", color="black", linestyle="--", ax=ax1)

        if test_history_available:
            ax2 = plt.twinx()
            sns.lineplot(df_fh_tst, x="epoch", y="eval_accuracy", errorbar="sd", color="orangered", ax=ax2)
            _ = ax1.set_xlabel(kwargs.get("loss_xlabel", 'Epoch'))
            _ = ax1.set_ylabel(kwargs.get("loss_ylabel", 'Loss'))
            _ = ax2.set_ylabel(kwargs.get("acc_ylabel", "Accuracy (Prozentuale Übereinstimmung)"))

        if test_history_available:
            h1 = Line2D([0], [0], label=kwargs.get("evl_loss_legend", "Loss (Val.)"), color='royalblue')
            h2 = Line2D([0], [0], label=kwargs.get("trn_loss_legend", "Loss (Train.)"), color='black', linestyle="--")
            h3 = Line2D([0], [0], label=kwargs.get("evl_acc_legend", "Accuracy (Val.)"), color='orangered')
            hs = [h1, h2, h3]
            _ = ax1.legend(handles=hs, loc='center left', bbox_to_anchor=(1.15, 0.5))

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return fig


    ## Comparison
    def compare(
        self,
        other: "QuScorerResults",
        labels: list[str]=None,
        **kwargs
    ) -> Figure:
        """Comparison of the evaluation metrics of two QuScorerResults instances.

        Parameters
        ----------
        other : QuScorerResults
            The other QuScorerResults instance to compare with.
        labels : list[str]
            Labels for the two scorers. If `None` the default labels `["Scorer 1", "Scorer 2"]` are used.

        Returns
        -------
        Figure
            The plot as a `matplotlib.figure.Figure`.
        """
        if labels is None:
            labels = ["Scorer 1", "Scorer 2"]
        if self._has_test_data() and other._has_test_data():
            mode = "test"
        else:
            mode = "train"
        print(f"Setting evaluation mode to \"{mode}\" because of availability.")
        self_dct = self._get_text_task_evaluation(mode=mode)
        othr_dct = other._get_text_task_evaluation(mode=mode)

        dct = {}
        diff_dct = {}
        for key in self_dct.keys():
            dct[key] = [self_dct[key], othr_dct[key]]
            diff_dct[key] = self_dct[key] - othr_dct[key]

        print(f"Comparisons {labels[0]} - {labels[1]}:")
        print(f"  Valid Text Responses Only ({mode})")
        print(f" - N-Persons    = {self_dct['n_pers']} vs. {othr_dct['n_pers']}")
        print(f" - N-Text-tasks = {self_dct['n_tasks']} vs. {othr_dct['n_tasks']}")
        print(f" - N-Responses  = {self_dct['n_responses']} vs. {othr_dct['n_responses']}")
        print(f" - Acc           = {diff_dct['acc']:.3f}  ({self_dct['acc']:.3f} vs. {othr_dct['acc']:.3f})")
        print(f" - F1 (weighted) = {diff_dct['f1_weighted']:.3f}  ({self_dct['f1_weighted']:.3f} vs. {othr_dct['f1_weighted']:.3f})")
        print(f" - F1 (macro)    = {diff_dct['f1_macro']:.3f}  ({self_dct['f1_macro']:.3f} vs. {othr_dct['f1_macro']:.3f})")
        print(f" - Cohens Kappa  = {diff_dct['cohens_kappa']:.3f}  ({self_dct['cohens_kappa']:.3f} vs. {othr_dct['cohens_kappa']:.3f})")
        print(f" - Quadratic Weighted Kappa = {diff_dct['quadratic_weighted_kappa']:.3f}  ({self_dct['quadratic_weighted_kappa']:.3f} vs. {othr_dct['quadratic_weighted_kappa']:.3f})")

        df = pd.DataFrame(dct, index=labels).reset_index(names="labels")
        df = df[["labels", "acc", "f1_weighted", "f1_macro", "cohens_kappa", "quadratic_weighted_kappa"]]
        df.columns = ["Models", "Accuracy", "F1 (weighted)", "F1 (macro)", "Cohens Kappa", "Quadratic Weighted Kappa"]
        df = df.melt(id_vars="Models", var_name="metric", value_name="value")

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(df, x="metric", y="value", hue="Models", ax=ax)
        ax.set_title(kwargs.get("title", f"Performance Comparison {labels[0]} vs. {labels[1]}"))
        ax.set_xlabel(kwargs.get("xlabel", "Metric"))
        ax.set_ylabel(kwargs.get("ylabel", "Metric Value"))
        ax.set_ylim(bottom=kwargs.get("ylim_bottom", 0), top=kwargs.get("ylim_top", 1))
        _ = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

        return fig



    ## IO
    def __settings_dict(self) -> dict:
        return {
            "unit_col": self.unit_col,
            "target_col": self.target_col,
            "prediction_col": self.prediction_col,
            "validate_scores": self.validate_scores,
        }

    def to_dir(self, path: str, allow_empty_preds: bool=False) -> None:
        """Saves the data to a directory. Overwrites possible existing files
        automatically.

        Parameters
        ----------
        path : str
            The directory to save the data to.
        allow_empty_preds : bool
            Wether to allow saving the data even if there are no predictions
            available. If `False` an exception is raised if there are no
            predictions available.
        """
        dir_ = Path(path)
        path_suffix_warning(dir_.suffix)
        empty_or_create_dir(dir_)

        df_preds = self.df_preds
        df_fh = self.get_fit_histories_dataframe()

        if self.df_preds is None:
            if not allow_empty_preds:
                raise AttributeError(
                    "This QuScorerResults instance has not prediction data yet. Nothing to be saved."
                )
        else:
            df_preds = df_preds.reset_index(drop=True)
            write_data(df=df_preds, path=dir_ / "predictions.gzip")

        if df_fh is not None:
            df_fh = df_fh.reset_index(drop=True)
            write_data(df=df_fh, path=dir_ / "fit_history.gzip")

        if self.id_split is not None:
            self.id_split.to_dir(dir_ / "id_split")

        self.qudata.to_dir(dir_ / "qudata")

        with open(dir_ / "settings.json", "w") as f:
            json_dump(self.__settings_dict(), f, indent=2)


    @staticmethod
    def from_dir(path: str, verbose: bool=False) -> "QuScorerResults":
        """Loads a `CVPredictions`-object from directory.

        Parameters
        ----------
        path : str
            The filepath containing the prediction tables for the different
            splits.

        Returns
        -------
        CVPredictions
        """
        p = Path(path)
        path_suffix_warning(p.suffix)
        qudata = QuData.from_dir(path=p / "qudata", verbose=verbose)
        try:
            df_preds = read_data(path=p / "predictions.gzip")
            if df_preds.size == 0:
                df_preds = None
        except FileNotFoundError:
            df_preds = None
        df_fh = read_data(path=p / "fit_history.gzip", return_none=True)
        try:
            id_split = IDsKFoldSplit.from_dir(path=p / "id_split")
        except FileNotFoundError:
            id_split = None

        with open(p / "settings.json", "r") as f:
            settings_dict: dict = json_load(f)
        validate_scores = settings_dict.pop("validate_scores", "none")

        qusr = QuScorerResults(
            qudata=qudata,
            df_preds=df_preds,
            df_fit_histories=df_fh,
            id_split=id_split,
            **settings_dict,
            validate_scores="none",
            supress_quconfig_validation=True,
        )

        if df_preds is not None:
            qusr.validate_scores = validate_scores
            for n in range(1, qusr.get_n_splits() + 1):
                df = df_preds[df_preds["split"]==n]
                df_trn = df[df["mode"]=="trn"].reset_index(drop=True)
                df_tst = df[df["mode"]=="test"].reset_index(drop=True)
                qusr.__validate_long_df_preds(df_trn)
                qusr.__validate_long_df_preds(df_tst)

        return qusr
