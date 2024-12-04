"""
# IDs-Split Base

This module provides a base class for ID-wise splits of the data.
"""

import numpy as np
import pandas as pd

from matplotlib.pyplot import close as pltclose
from matplotlib.pyplot import show as pltshow

import seaborn as sns

from abc import ABC, abstractmethod
from pathlib import Path
from json import dump, load

from ..core.io import (
    read_data,
    empty_or_create_dir,
    write_data,
    path_suffix_warning,
)

from ..data.config import QuConfig
from ..data.data import QuData
from ..clustering.clusters import QuScoreClusters




class IDsSplit(ABC):

    def __init__(
        self,
        quconfig: QuConfig,
        df_ids: pd.DataFrame,
        strat_col: str,
        stratify: bool,
        random_state: int=5555,
        strat_label_dict: dict=None,
    ):
        """Base class for ID-wise splits of the data.

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
        random_state : int
            The random state for the split.
        strat_label_dict : dict
            A dictionary with the labels for the stratification. This overwrites
            the labels from the `strat_col` for visualizations and alike if provided.
        """
        self.id_col = quconfig.id_col
        self.quconfig = quconfig
        self.df_ids = df_ids
        self.cols = df_ids.columns.to_list()
        self.random_state = random_state

        self.strat_col = strat_col
        self.stratify = stratify
        self.strat_label_dict = strat_label_dict

    def __eq__(self, other: "IDsSplit") -> bool:
        if self.cols != other.cols:
            return False
        msk = self.df_ids == other.df_ids
        msk = msk.values
        return np.all(msk)


    ## Setup
    @staticmethod
    def _get_df_strat(
        qudata: QuData,
        quclst: QuScoreClusters=None,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
    ) -> tuple[pd.DataFrame, str]:
        if quclst is not None and (df_strat is None and strat_col is None):
            return quclst.clusters_most(qudata=qudata), "cluster"
        elif quclst is None and (df_strat is not None and strat_col is not None):
            return df_strat, strat_col
        elif quclst is None and df_strat is None and strat_col is None:
            return None, None
        else:
            raise ValueError(
                "IDsSplit: Pass either the `quclst` argument or the `df_strat` and `strat_col` arguments."
            )

    @staticmethod
    def _get_ids_from_data(
        qudata: QuData,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
        stratify: bool=False,
    ) -> pd.DataFrame:
        id_col = qudata.id_col
        cols = [id_col, "total_score"]
        if stratify:
            cols.append(strat_col)

        df = qudata.get_total_score()
        if stratify:
            if df_strat is None or strat_col is None:
                raise ValueError(
                    "If you want to stratify by labels, you have to additionally \n" +
                    "pass the `quclst` argument or the `df_strat` and `strat_col` arguments."
                )

            else:
                if df_strat[id_col].duplicated().sum() > 0:
                    raise ValueError("The IDs in the stratification DataFrame are not unique.")
                df_strat = df_strat[[id_col, strat_col]]
                df = pd.merge(df, df_strat, on=id_col, how="left")
                drop_ids = df[id_col][df[strat_col].isna()].to_list()
                if len(drop_ids) > 0:
                    print(f"Dropping {len(drop_ids)} IDs because of missing stratification information: \n{drop_ids}")
                    df = df.dropna()

        df = df[cols].reset_index(drop=True)

        return df


    ## Generation
    def _kwargs_from_qudata(
        qudata: QuData,
        quclst: QuScoreClusters=None,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
        stratify: bool=False,
        strat_label_dict: dict=None,
    ) -> dict:
        quconfig = qudata.quconfig
        id_col = quconfig.id_col

        if hasattr(quclst, "_cluster_label_dict"):
            if strat_label_dict is not None:
                print("Warning: `strat_label_dict` but passed `quclst` has a `_cluster_label_dict`-attribute. Using the latter.")
            strat_label_dict = quclst._cluster_label_dict
        else:
            strat_label_dict = strat_label_dict

        cols = [id_col, "total_score"]
        if stratify:
            cols.append(strat_col)

        df_strat, strat_col = IDsSplit._get_df_strat(
            qudata=qudata,
            quclst=quclst,
            df_strat=df_strat,
            strat_col=strat_col,
        )

        df_ids = IDsSplit._get_ids_from_data(
            qudata=qudata,
            df_strat=df_strat,
            strat_col=strat_col,
            stratify=stratify,
        )
        return {
            "quconfig": quconfig,
            "df_ids": df_ids,
            "stratify": stratify,
            "strat_col": strat_col,
            "strat_label_dict": strat_label_dict,
        }

    @staticmethod
    @abstractmethod
    def from_qudata():
        """Create an IDSplit-instance of the class from a `QuData` object."""
        pass


    ## Info
    @abstractmethod
    def get_n_splits(self) -> int:
        """Get the number of splits."""
        pass

    @abstractmethod
    def get_n_test(self) -> int:
        """Get the number of test-instances."""
        pass

    @abstractmethod
    def get_tst_ids_lists(self) -> list[list]:
        """Get the IDs of the test-instances as a list of lists - one for each split."""
        pass

    @abstractmethod
    def get_tst_ids_dict(self) -> dict[int, list]:
        """Get the IDs of the test-instances as a dictionary - a number-list pair for each split."""
        pass

    def score_dstributions_plot(self):
        """Plot the total-score distributions of the splits."""
        if self.get_n_splits() < 4:
            fig = sns.histplot(
                self.df_ids,
                x="total_score",
                hue="split",
                multiple="dodge",
                stat = 'density',
                common_norm=False,
            )
        fig = sns.kdeplot(
            self.df_ids,
            x="total_score",
            hue="split",
            common_norm=False,
        )
        sns.move_legend(fig, loc='center left', bbox_to_anchor=(1, 0.5))
        return fig

    def stratification_plot(self, normed: bool=True):
        """Plot the stratification of the splits, i.e., the distribution of the
        stratification-labels in each split.

        Parameters
        ----------
        normed : bool
            Whether to plot the proportions instead of the counts.
        """
        if not self.stratify:
            print("Warning: No stratification, nothing to plot.")
            return

        df = self.df_ids.copy()
        if self.strat_label_dict is not None:
            def get_clst_no(k: str):
                if k == "noise":
                    return -1
                else:
                    return int(k)
            order_dict = {v: get_clst_no(k) for k, v in self.strat_label_dict.items()}
            order_list = list(order_dict.keys())
            order_list.sort(key=lambda k: order_dict[k])
        else:
            order_list = df[self.strat_col].to_list().sort()
        df[self.strat_col] = pd.Categorical(df[self.strat_col], order_list)

        stat = "count"
        common_norm = None
        if normed:
            stat = "proportion"
            common_norm = False

        fig = sns.histplot(
            df,
            x=self.strat_col,
            hue="split",
            multiple="dodge",
            stat=stat,
            common_norm=common_norm,
        )
        if not normed:
            for container in fig.containers:
                fig.bar_label(container)
        sns.move_legend(fig, loc='center left', bbox_to_anchor=(1, 0.5))
        return fig

    def info(self):
        """Print the number of splits and the number of test-instances."""
        df = self.df_ids.copy()
        df = df.drop(columns=self.id_col)
        if self.strat_col in df.columns:
            df = df.drop(columns=self.strat_col)

        df_aggs = df.groupby("split").aggregate("mean")
        ds_counts = df.groupby("split").size()
        df_ret = pd.concat([ds_counts, df_aggs], axis=1)
        df_ret.columns = ["count", "average total score"]
        return df_ret

    def full_info(self):
        """Print the number of splits and the number of test-instances, and plot the
        stratification and score distributions.
        """
        _ = self.stratification_plot()
        pltshow()
        pltclose()
        _ = self.score_dstributions_plot()
        pltshow()
        pltclose()
        print(
            f"Generatred id-split (n_splits={self.get_n_splits()}, random_state={self.random_state}, N-test={self.get_n_test()}) \n"
        )

    ## IO
    def to_dir(self, path: str) -> None:
        """Save the split information to a directory. Using a directory is
        necessary because the cluster-labels are stored alongside the split data.

        Parameters
        ----------
        path : str
            A path to a directory.
        """
        p = Path(path)
        path_suffix_warning(p.suffix)
        empty_or_create_dir(p)

        write_data(self.df_ids, path=p / "split_data.gzip")
        self.quconfig.to_yaml(path=p / "quconfig.yaml")

        settings_dict = {
            "strat_col": self.strat_col,
            "stratify": self.stratify,
            "strat_label_dict": self.strat_label_dict,
        }
        with open(p / "settings_dict.json", "w") as f:
            dump(settings_dict, f, indent=2)

    @staticmethod
    def _kwargs_from_path(path: str) -> dict:
        p = Path(path)
        path_suffix_warning(p.suffix)

        quconfig = QuConfig.from_yaml(p / "quconfig.yaml")

        df_ids = read_data(path=p / "split_data.gzip")
        with open(p / "settings_dict.json", "r") as f:
            settings_dict = load(f)

        return {
            "quconfig": quconfig,
            "df_ids": df_ids,
            "strat_col": settings_dict["strat_col"],
            "stratify": settings_dict["stratify"],
            "strat_label_dict": settings_dict["strat_label_dict"],
        }
