"""
# Interrater Data

A submodule for including the analysis of interrater data in the QuTools package.
Interrater-Analyses are used as a baseline for predictive models and to evaluate
the quality and generalizability of the prediction. Often the human-human interrater-
agreement is viewed as a ceiling for the predictive models.

The `QuInterraterAnalysis` class is the main component of this submodule. It makes
use of existing code and logic, especially the `QuData`, `IDsSplit`,
`QuScorerResults` and `QuConfig` classes, to provide an analyses of itnerrater-consistency
that resembles the analyses of predictive models. It therefore uses one of the
passed score-tables as the ground truth and the other as the prediction.
"""


import pandas as pd

from matplotlib.figure import Figure

from typing import Literal

from ..core import read_data, score_mc_items, score_mc_tasks

from ..data import QuData
from ..id_splits.id_split_k_fold import IDsKFoldSplit
from ..id_splits.id_split_train_test import IDsTrainTestSplit
from ..data.config import QuConfig

from ..clustering.clusters import QuScoreClusters
from .scorer_results import QuScorerResults



class QuInterraterAnalysis:
    unit_col: str = "task"
    target_col: str = "score"
    prediction_col: str = "score_pred"
    quconfig: QuConfig

    def __init__(self,
        quconfig: QuConfig,
        df_rater1: pd.DataFrame|str,
        df_rater2: pd.DataFrame|str,
        mc_units: Literal["mc_items", "mc_tasks", "no_mc"]="mc_items",
    ) -> None:
        """Class for the analysis of interrater data.

        Parameters
        ----------
        quconfig : QuConfig
            The QuConfig object that holds the configuration of the questionnaire.
        df_rater1 : pd.DataFrame|str
            The score-table of the first rater. Can be a path to a tabular-file or a pandas DataFrame.
        df_rater2 : pd.DataFrame|str
            The score-table of the second rater. Can be a path to a tabular-file or a pandas DataFrame.
        mc_units : Literal["mc_items", "mc_tasks", "no_mc"]
            The type of multiple-choice units in the score-tables.
        """
        self.qudata = QuData(
            quconfig=quconfig,
            df_scr=df_rater1,
            mc_score_col_type=mc_units,
        )
        self.quconfig = quconfig.copy()
        self.mc_units = mc_units
        self.id_col = quconfig.id_col
        self.df1 = self.__get_single_data(df_rater1, mc_units=mc_units)
        self.df2 = self.__get_single_data(df_rater2, mc_units=mc_units)
        df_wide = self.__get_wide_df()
        print(f"Found {df_wide.shape[0]} ID-matches in the two ratings passed.")

    def __get_single_data(
            self,
            df: pd.DataFrame|str,
            mc_units: Literal["mc_items", "mc_tasks", "no_mc"]="mc_items",
        ) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            df = read_data(df)
        self.quconfig.validate_wide_df_scr(
            df_scr=df,
            units=mc_units,
            all_units=True,
            validate_scores="warn",
        )
        df = df.sort_values(by=self.id_col).reset_index(drop=True)
        return df

    def __get_wide_df(self) -> pd.DataFrame:
        df = pd.merge(self.df1, self.df2, how="inner", on=self.id_col)
        return df

    def get_long_df(self, text_data_only: bool=False, mc_scored: bool=True) -> pd.DataFrame:
        """Get the long-format DataFrame for the interrater data.

        Parameters
        ----------
        text_data_only : bool
            If True, only the text-data columns are included in the DataFrame.
        mc_scored : bool
            If True, the multiple-choice items are scored, if false, the single items
            are included in the DataFrame.
        """
        df1 = self.df1.copy()
        df2 = self.df2.copy()

        if text_data_only:
            df1 = df1[[self.id_col] + self.quconfig.get_text_columns("tasks")]
            df2 = df2[[self.id_col] + self.quconfig.get_text_columns("tasks")]
        else:
            if mc_scored and self.mc_units == "mc_items":
                mc_tasks = self.quconfig.get_mc_tasks()
                df1 = score_mc_items(df1, mc_tasks=mc_tasks)
                df2 = score_mc_items(df2, mc_tasks=mc_tasks)
                df1 = score_mc_tasks(df1, mc_tasks=mc_tasks)
                df2 = score_mc_tasks(df2, mc_tasks=mc_tasks)

        df1 = df1.melt(id_vars=self.id_col, var_name="task", value_name="score")
        df2 = df2.melt(id_vars=self.id_col, var_name="task", value_name="score_pred")
        df = pd.merge(df1, df2, "inner", on=[self.id_col, "task"])
        df["split"] = 1
        df["mode"] = "test"
        df = df.dropna().reset_index(drop=True)
        return df

    def _has_test_data(self):
        return True

    def _as_scorer_results(self,
        split: Literal["all_train", "random_cv", "random_train_test"]=None,
        quclst: QuScoreClusters=None,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
        stratify: bool=False,
        **kwargs,
    ) -> QuScorerResults:
        df_preds = self.get_long_df(text_data_only=True)
        if split is None:
            id_split = None
        elif split == "all_train":
            id_split = None
            df_preds["mode"] = "train"
        elif split == "random_cv":
            id_split = IDsKFoldSplit.from_qudata(
                qudata=self.qudata,
                quclst=quclst,
                df_strat=df_strat,
                strat_col=strat_col,
                stratify=stratify,
                n_splits=kwargs.get("n_splits", 10),
                verbose=kwargs.get("id_split_verbose", True),
            )
        elif split == "random_train_test":
            id_split = IDsTrainTestSplit.from_qudata(
                qudata=self.qudata,
                quclst=quclst,
                df_strat=df_strat,
                strat_col=strat_col,
                stratify=stratify,
                test_size=kwargs.get("test_size", 0.2),
                verbose=kwargs.get("id_split_verbose", True),
            )

        if id_split is not None:
            qusr = QuScorerResults(qudata=self.qudata, id_split=id_split)
            df0 = df_preds.drop(columns=["split"]).copy()
            for idx, id_list in enumerate(id_split.get_tst_ids_lists()):
                df_ = df0.copy()
                tst_msk = df_[self.id_col].isin(id_list).values
                df_["mode"] = "train"
                df_.loc[tst_msk, "mode"] = "test"
                df_["split"] = idx + 1

                df_trn = df_[df_["mode"]=="train"].drop(columns=["split", "mode"])
                df_tst = df_[df_["mode"]=="test"].drop(columns=["split", "mode"])

                qusr.append(df_trn=df_trn, df_tst=df_tst)

        else:
            qusr = QuScorerResults(
                qudata=self.qudata,
                df_preds=df_preds,
                id_split=id_split,
            )
        return qusr

    def evaluate(self) -> None:
        """Evaluate the interrater data. Anaogous to the `evaluate` method of the `QuScorerResults` class."""
        qusr = self._as_scorer_results()
        qusr.evaluation()

    def plot_confusion_matrix(self, save_path: str=None) -> Figure:
        """Plot the confusion matrix of the interrater data. Anaogous to the `plot_confusion_matrix` method of the `QuScorerResults` class.

        Parameters
        ----------
        save_path : str
            The path to save the figure to.
        """
        qusr = self._as_scorer_results()
        fig = qusr.plot_confusion_matrix(save_path=save_path)
        return fig
