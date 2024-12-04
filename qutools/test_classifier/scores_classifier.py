"""
# Scores Classifier

This module contains the class for direct classification of the complete edits
to the questionnaire based on the "true" scores.
"""


import pandas as pd

from sklearn.linear_model import LogisticRegression

from tqdm import tqdm

from ..data.config import QuConfig
from ..data.data import QuData
from ..id_splits import IDsSplit, IDsTrainTestSplit, IDsKFoldSplit

from ..clustering.clusters import QuScoreClusters

from ..core.classifier import Classifier, ScikitClassifier
from ..core.trainulation import Split, get_random_oversampler, training_core

from .classifier_results import QuClassifierResults, QuClassifierResults



class QuScoresClassifierError(Exception):
    """An Exception class for the QuScoresClassifier"""


class QuScoresClassifier:
    def __init__(
        self,
        model: Classifier=None,
        target_name: str="cluster",
        omit_classes: list=None,
    ) -> None:
        """Initialize the QuScoresClassifier.

        Parameters
        ----------
        model : Classifier
            The classifier model to use.
        target_name : str
            The name of the target column.
        omit_classes : list
            A list of classes to omit from the target."""
        self.target = target_name
        self.pred = target_name + "_pred"

        self.omit_classes = omit_classes
        self.class_order = None

        if model is None:
            model = ScikitClassifier(LogisticRegression(C=0.5, max_iter=500))
        self.clf_model = model


    ## Setup
    def __set_class_label_dicts(self, df: pd.DataFrame):
        unique_targets = df[self.target].unique()
        self.target_to_int = dict(zip(unique_targets, range(len(unique_targets))))
        self.int_to_target = {v: k for k, v in self.target_to_int.items()}

    def __targets_to_int(self, df: pd.DataFrame, no_pred: bool=False) -> pd.DataFrame:
        with pd.option_context("future.no_silent_downcasting", True):
            df[self.target] = df[self.target].replace(self.target_to_int).infer_objects(copy=False)
            if self.pred in df.columns and not no_pred:
                df[self.pred] = df[self.pred].replace(self.target_to_int).infer_objects(copy=False)
        return df

    def __targets_to_labels(self, df: pd.DataFrame, no_pred: bool=False) -> pd.DataFrame:
        df[self.target] = df[self.target].replace(self.int_to_target)
        if self.pred in df.columns and not no_pred:
            df[self.pred] = df[self.pred].replace(self.int_to_target)
        return df



    def __get_df_target(
        self,
        qudata: QuData,
        quclst: QuScoreClusters=None,
        df_target: pd.DataFrame=None,
    ) -> pd.DataFrame:
        self.target_is_quclst = False
        if quclst is not None and df_target is None:
            print("Using cluster data as targets.")
            self.target_is_quclst = True
            df_target = quclst.clusters_most(qudata=qudata)
            df_target = df_target[[quclst.quconfig.id_col, "cluster"]]
        elif quclst is None and df_target is not None:
            pass
        else:
            raise QuScoresClassifierError(
                "Either pass the `quclst` or `df_target` argument."
            )
        return df_target

    def __get_df_strat(
        self,
        df_target: pd.DataFrame,
        df_strat: pd.DataFrame,
        startify: bool,
        strat_col: str,
    ) -> pd.DataFrame:
        self.strat_col = strat_col
        self.stratify_by_target = False
        if not startify:
            if strat_col is not None or df_strat is not None:
                print("Warning: `stratify` is False, `strat_col` and `df_strat` will be omitted.")
            return None
        if df_strat is None:
            print(f"Stratifying by target (\"{self.target}\").")
            self.strat_col = self.target
            self.stratify_by_target = True
            return df_target.copy()
        else:
            if strat_col not in df_strat.columns:
                raise QuScoresClassifierError(
                    f"The `strat_col` \"{strat_col}\" is not part of `df_strat`."
                )
            return df_strat

    def __get_train_df(
        self,
        qudata: QuData,
        df_target: pd.DataFrame,
        df_strat: pd.DataFrame=None
    ) -> pd.DataFrame:
        id_col = qudata.quconfig.id_col
        if id_col not in df_target.columns:
            raise QuScoresClassifierError(
                f"The id_col \"{id_col}\" of the passed `qudata` is not present in the passed `df_target`."
            )
        if self.target not in df_target.columns:
            raise QuScoresClassifierError(
                f"The specified target-column {self.target} is not present in the passed `df_target`."
            )
        df_features = qudata.get_scr(mc_scored=True, verbose=False).fillna(0)
        df = pd.merge(df_target, df_features, on=id_col, how="inner")
        if df_strat is None:
            print(f"Found {df.shape[0]} common IDs in the passed `qudata` and target data..")

        if df_strat is not None and not self.stratify_by_target:
            df = pd.merge(df, df_strat, on=id_col, how="inner")
            print(f"Found {df.shape[0]} IDs common in the passed `qudata`, target-data, and stratification-data.")

        df = df[[self.id_col] + self.features + [self.target]]
        return df


    def __get_strat_label_dict(self, quclst: QuScoreClusters=None) -> dict:
        if hasattr(quclst, "_cluster_label_dict"):
            return quclst._cluster_label_dict
        else:
            return None



    ## Eval
    def fixed_eval(
        self,
        id_split: IDsSplit,
        qudata: QuData,
        quclst: QuScoreClusters,
        df_target: pd.DataFrame,
        oversample: bool=False,
    ) -> QuClassifierResults:
        """Evaluate the classifier with a fixed train-test- or CV-split.

        Parameters
        ----------
        id_split : IDsSplit
            The split of the data.
        qudata : QuData
            The data to use.
        quclst : QuScoreClusters
            The cluster object, that can be used to provide the target data.
            Only either `quclst` or `df_target` should be passed.
        df_target : pd.DataFrame
            The target data. Must be matchable via the ID-col.
        oversample : bool
            Whether to use oversampling.

        Returns
        -------
        QuClassifierResults
            The results of the evaluation.
        """
        full_train_fit = id_split.get_n_test() == 0
        self.features = qudata.quconfig.get_task_names()
        self.id_col = qudata.id_col

        df_target = self.__get_df_target(
            qudata=qudata,
            quclst=quclst,
            df_target=df_target,
        )
        self.__set_class_label_dicts(df=df_target)
        df_target = self.__targets_to_int(df=df_target)

        self.clf_model.reset()
        self.id_split = id_split
        quconfig = qudata.quconfig.copy()
        quclfr = QuClassifierResults(
            quconfig=quconfig,
            id_split=id_split,
            target_col=self.target,
            prediction_col=self.pred,
        )
        id_col = quconfig.id_col

        df_trn = self.__get_train_df(
            qudata=qudata,
            df_target=df_target,
            df_strat=None,
        )

        if oversample:
            print("Using oversampling.")

        test_ids_list = id_split.get_tst_ids_lists()

        for tst_ids in tqdm(test_ids_list):
            split = Split.from_tst_ids(df_trn, tst_ids, id_col)
            if oversample:
                oversampler = get_random_oversampler()
            else:
                oversampler = None
            df_pred_trn, df_pred_tst, fit_history, model_copy = training_core(
                model=self.clf_model,
                split=split,
                feature_cols=self.features,
                target_col=self.target,
                prediction_col=self.pred,
                oversampler=oversampler,
            )

            df_pred_trn = self.__targets_to_labels(df_pred_trn)
            if not full_train_fit:
                df_pred_tst = self.__targets_to_labels(df_pred_tst)

            quclfr.append(df_trn=df_pred_trn, df_tst=df_pred_tst, fit_history=fit_history)

        return quclfr

    def random_cross_validate(
        self,
        qudata: QuData,
        quclst: QuScoreClusters=None,
        df_target: pd.DataFrame=None,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
        stratify: bool=True,
        oversample: bool=False,
        n_splits: int=10,
        random_state: int=42,
        verbose_split: bool=True,
    ) -> QuClassifierResults:
        """Cross-validate the classifier with a random CV-split.

        Parameters
        ----------
        qudata : QuData
            The data to use.
        quclst : QuScoreClusters
            The cluster object, that can be used to provide the target data.
            Only either `quclst` or `df_target` should be passed.
        df_target : pd.DataFrame
            The target data. Must be matchable via the ID-col.
        df_strat : pd.DataFrame
            The stratification data. Must be matchable via the ID-col.
        strat_col : str
            The column to stratify by.
        stratify : bool
            Whether to stratify.
        oversample : bool
            Whether to use oversampling.
        n_splits : int
            The number of splits.
        random_state : int
            The random state for the CV-split.
        verbose_split : bool
            Whether to print the split information.
        """
        df_target = self.__get_df_target(
            qudata=qudata,
            quclst=quclst,
            df_target=df_target,
        )
        df_strat = self.__get_df_strat(
            df_target=df_target,
            df_strat=df_strat,
            startify=stratify,
            strat_col=strat_col,
        )
        id_split = IDsKFoldSplit.from_qudata(
            qudata=qudata,
            df_strat=df_strat,
            strat_col=self.strat_col,
            stratify=stratify,
            n_splits=n_splits,
            random_state=random_state,
            strat_label_dict=self.__get_strat_label_dict(quclst=quclst),
            verbose=verbose_split,
        )
        quclfr = self.fixed_eval(
            id_split=id_split,
            qudata=qudata,
            quclst=None,
            df_target=df_target,
            oversample=oversample,
        )
        return quclfr


    def random_train_testing(
        self,
        qudata: QuData,
        quclst: QuScoreClusters=None,
        df_target: pd.DataFrame=None,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
        stratify: bool=True,
        oversample: bool=False,
        test_size: float=0.2,
        random_state: int=42,
        verbose_split: bool=True,
    ) -> QuClassifierResults:
        """Train-test the classifier with a random train-test-split (non-CV).

        Parameters
        ----------
        qudata : QuData
            The data to use.
        quclst : QuScoreClusters
            The cluster object, that can be used to provide the target data.
            Only either `quclst` or `df_target` should be passed.
        df_target : pd.DataFrame
            The target data. Must be matchable via the ID-col.
        df_strat : pd.DataFrame
            The stratification data. Must be matchable via the ID-col.
        strat_col : str
            The column to stratify by.
        stratify : bool
            Whether to stratify.
        oversample : bool
            Whether to use oversampling.
        test_size : float
            The size of the test set.
        random_state : int
            The random state for the split.
        verbose_split : bool
            Whether to print the split information.
        """
        df_target = self.__get_df_target(
            qudata=qudata,
            quclst=quclst,
            df_target=df_target,
        )
        df_strat = self.__get_df_strat(
            df_target=df_target,
            df_strat=df_strat,
            startify=stratify,
            strat_col=strat_col,
        )
        id_split = IDsTrainTestSplit.from_qudata(
            qudata=qudata,
            df_strat=df_strat,
            strat_col=self.strat_col,
            stratify=stratify,
            test_size=test_size,
            random_state=random_state,
            strat_label_dict=self.__get_strat_label_dict(quclst=quclst),
            verbose=verbose_split,
        )
        quclfr = self.fixed_eval(
            id_split=id_split,
            qudata=qudata,
            quclst=None,
            df_target=df_target,
            oversample=oversample,
        )
        return quclfr
