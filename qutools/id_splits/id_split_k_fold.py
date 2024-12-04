"""
# $k$-Fold-Split

This module provides a class to create a $k$-fold split of the data.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold

from ..data.config import QuConfig
from ..data.data import QuData
from ..clustering.clusters import QuScoreClusters

from .id_split_base import IDsSplit



class IDsKFoldSplit(IDsSplit):
    def __init__(
        self,
        quconfig: QuConfig,
        df_ids: pd.DataFrame,
        strat_col: str=None,
        stratify: bool=False,
        n_splits: int=10,
        random_state: int=5555,
        strat_label_dict: dict=None,
    ):
        """Create a k-fold split of the data.

        Parameters
        ----------
        quconfig : QuConfig
            The configuration object of the questionnaire.
        df_ids : pd.DataFrame
            A DataFrame with the IDs of the instances.
        strat_col : str
            The column name of the stratification labels.
        stratify : bool
            Whether to stratify the splits by the labels.
        n_splits : int
            The number of splits.
        random_state : int
            The random state for the split.
        strat_label_dict : dict
            A dictionary with the labels for the stratification. This overwrites
            the labels from the `strat_col` for visualizations and alike if provided.
        """
        super().__init__(
            quconfig=quconfig,
            df_ids=df_ids,
            strat_col=strat_col,
            stratify=stratify,
            random_state=random_state,
            strat_label_dict=strat_label_dict,
        )
        self.n_splits = n_splits
        self.__kfold_split()


    ## Setup
    def __kfold_split(self) -> None:
        if "split" in self.df_ids.columns:
            return
        if self.n_splits == 1:
            self.df_ids["split"] = 0
            return
        if not self.stratify:
            kfold = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = kfold.split(self.df_ids)

        else:
            kfold = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = kfold.split(self.df_ids, self.df_ids[self.strat_col])

        df = pd.DataFrame()
        for idx, (trn_idx, tst_idx) in enumerate(splits):
            df_ = self.df_ids.iloc[tst_idx, :].copy()
            df_["split"] = idx
            df = pd.concat([df, df_])

        self.df_ids = df


    ## Generation
    @staticmethod
    def from_qudata(
        qudata: QuData,
        quclst: QuScoreClusters=None,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
        stratify: bool=False,
        n_splits: int=10,
        random_state: int=5555,
        verbose: bool=True,
        strat_label_dict: dict=None,
    ) -> "IDsKFoldSplit":
        """Create a train-test split directly from a `QuData`-object.

        Parameters
        ----------
        qudata : QuData
            The QuData object to "split".
        quclst : QuScoreClusters
            A cluster-model of the `QuData` that can be used for stratification.
        df_strat : pd.DataFrame
            A DataFrame with stratification labels instead of the `quclst` argument.
            Can only be used by also specifying the `strat_col` argument.
        strat_col : str
            The column name of the stratification labels.
        stratify : bool
            Whether to stratify the splits by the labels.
        n_splits : int
            The number of splits ($k$).
        random_state : int
            The random state for the split.
        verbose : bool
            Whether to print the split information.
        strat_label_dict : dict
            A dictionary with the labels for the stratification. This overwrites
            the labels from the `strat_col` for visualizations and alike if provided.
        """
        kwargs = IDsSplit._kwargs_from_qudata(
            qudata=qudata,
            quclst=quclst,
            df_strat=df_strat,
            strat_col=strat_col,
            stratify=stratify,
            strat_label_dict=strat_label_dict,
        )
        id_split = IDsKFoldSplit(
            **kwargs,
            n_splits=n_splits,
            random_state=random_state,
        )
        if verbose:
            id_split.full_info()
        return id_split

    ## IO
    @staticmethod
    def from_dir(path: str) -> "IDsKFoldSplit":
        """Load a k-fold split from a directory.

        Parameters
        ----------
        path : str
            A path to a directory.
        """
        kwargs = IDsSplit._kwargs_from_path(path=path)
        df_ids: pd.DataFrame = kwargs["df_ids"]
        n_splits = len(df_ids["split"].unique())
        return IDsKFoldSplit(
            **kwargs,
            n_splits=n_splits,
            random_state=None,
        )


    ## Info
    def get_n_splits(self):
        """Get the number of splits."""
        return self.n_splits

    def get_n_test(self) -> int:
        """Get the number of test-instances."""
        return len(self.df_ids["split"].unique())

    def get_tst_ids_lists(self) -> list[list]:
        """Get the IDs of the test-instances as a list of lists - one for each split."""
        df = self.df_ids
        test_ids_list = []
        for idx in range(self.n_splits):
            test_ids = df[self.id_col][df["split"]==idx].to_list()
            test_ids_list.append(test_ids)
        return test_ids_list

    def get_tst_ids_dict(self) -> dict[int, list]:
        """Get the IDs of the test-instances as a dictionary - a number-list pair for each split."""
        test_ids_list = self.get_tst_ids_lists()
        n_splits = len(test_ids_list)
        return dict(zip(np.arange(1, n_splits+1), test_ids_list))
