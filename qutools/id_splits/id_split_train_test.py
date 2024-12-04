"""
# Train-Test-Split

This module provides a class to create a "plain" train-test split of the data.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from ..data.config import QuConfig
from ..data.data import QuData
from ..clustering.clusters import QuScoreClusters

from .id_split_base import IDsSplit



class IDsTrainTestSplit(IDsSplit):
    def __init__(
        self,
        quconfig: QuConfig,
        df_ids: pd.DataFrame,
        strat_col: str,
        stratify: bool,
        test_size: float=0.2,
        random_state: int=5555,
        strat_label_dict: dict=None,
    ):
        """Create a train-test split of the data.

        Parameters
        ----------
        quconfig : QuConfig
            The configuration object of the questionnaire.
        df_ids : pd.DataFrame
            A DataFrame with the IDs of the instances.
        strat_col : str
            The column name of the stratification labels.
        stratify : bool
            Whether to stratify the splits by these labels.
        test_size : float
            The proportion of the test-instances.
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
        self.test_size = test_size
        self.__train_test_split()


    ## Setup
    def __train_test_split(self) -> None:
        if "split" in self.df_ids.columns:
            return
        stratify_arr = None
        if self.stratify:
            stratify_arr = self.df_ids[self.strat_col]

        df_trn, df_tst = train_test_split(
            self.df_ids[self.cols],
            test_size=0.2,
            random_state=5555,
            shuffle=True,
            stratify=stratify_arr,
        )
        df_trn["split"] = "train"
        df_tst["split"] = "test"
        df = pd.concat([df_trn, df_tst], axis=0)
        df = df.reset_index(drop=True)
        self.df_ids = df


    ## Generation
    @staticmethod
    def from_qudata(
        qudata: QuData,
        quclst: QuScoreClusters=None,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
        stratify: bool=False,
        test_size: float=0.2,
        random_state: int=5555,
        verbose: bool=True,
        strat_label_dict: dict=None
    ) -> "IDsTrainTestSplit":
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
        test_size : float
            The proportion of the test-instances.
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
        id_split = IDsTrainTestSplit(
            **kwargs,
            test_size=test_size,
            random_state=random_state,
        )
        if verbose:
            id_split.full_info()
        return id_split


    ## IO
    @staticmethod
    def from_dir(path: str) -> "IDsTrainTestSplit":
        """Load a train-test split from a directory.

        Parameters
        ----------
        path : str
            A path to a directory.
        """
        kwargs = IDsSplit._kwargs_from_path(path=path)
        df_ids: pd.DataFrame = kwargs["df_ids"]
        n_trn = (df_ids["split"]=="train").sum()
        n_tst = (df_ids["split"]=="test").sum()
        return IDsTrainTestSplit(
            **kwargs,
            test_size=n_tst/n_trn,
            random_state=None,
        )


    ## Info
    def get_n_splits(self):
        """The number of splits is always 1 for a single Train-Test-Split."""
        return 1

    def get_n_test(self) -> int:
        """Get the number of test-instances."""
        return (self.df_ids["split"]=="test").sum()

    def get_tst_ids_list(self) -> list:
        """Get the IDs of the test-instances as a list."""
        df = self.df_ids
        ids = df[self.id_col][df["split"]=="test"].to_list()
        return ids

    def get_tst_ids_lists(self) -> list[list]:
        """Get the IDs of the test-instances as a list of lists. This is needed
        for interoperability with the $k$-fold splits."""
        return [self.get_tst_ids_list()]

    def get_tst_ids_dict(self) -> dict[int, list]:
        """Get the IDs of the test-instances as a dictionary. This is needed
        for interoperability with the $k$-fold splits."""
        return {1: self.get_tst_ids_list()}
