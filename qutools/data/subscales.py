"""
# Questionnaire Subscales

The `QuestionnaireSubscales` class is used to represent the allocation of tasks
to subscales. It can be used to aggregate score data from tasks to subscales.
It is can also be used as a preparation step for cluster analyses, where the
subscales are used as features as well as predictive models, where the subscales
are used as the target variable.

Usage
-----
An example usage of the `QuestionnaireSubscales` class is shown in the
[qutools.clustering.clusters][qutools.clustering.clusters] documentation.
"""


import numpy as np
import pandas as pd

from pathlib import Path
from typing import Self
from copy import deepcopy
from json import dump, load

from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import cohen_kappa_score

from ..core.io import path_suffix_warning, empty_or_create_dir, read_data, write_data
from ..core.pandas import merge_and_clip, dataframes_equal
from .config import QuConfig
from .data import QuData




class QuSubscalesError(Exception):
    """Error class for `QuestionnaireSubscales` errors."""
    pass


class QuSubscales:
    allowed_values = [0, 1]

    def __init__(
        self,
        quconfig: QuConfig,
        df_cat: pd.DataFrame|str,
        override_names: list=None,
    ):
        """`QuestionnaireSubscales`s object, built from a
        `QuestionnaireConfig` object and a categorization dataframe (`df_cat`)
        allocaing the tasks to subscales.

        Parameters
        ----------
        quconfig : QuestionnaireConfig
            The questionnaire config of the respective questionnaire.
        df_cat : pd.DataFrame
            The categorization dataframe. Must have a "Task" or "task" column
            containing the task-names, which must match the task names defined
            in `quconfig`. The other columns represent the subscales. The
            tasks are matched to the subscales by inserting a `1` in the
            respective cell. Can be a dataframe or a path.
        """
        if not isinstance(df_cat, pd.DataFrame):
            df_cat = read_data(df_cat)

        df_cat = df_cat.replace(quconfig._na_aliases, 0).copy()
        self.quconfig = deepcopy(quconfig)

        self.__set_task_column(df_cat)

        if hasattr(quconfig, "task_omit_list"):
            print(
                f"Omitting tasks {quconfig.task_omit_list}," +
                "\nthat were previously dropped from the quconfig." +
                "\nNote that this can only happen before column-validation and is therefore error prone." +
                "\nConsider dropping the respective items manually by loading and manipulating the tables with " +
                "\npandas."
            )
            df_cat = df_cat[df_cat[self.task_col].isin(quconfig.task_omit_list)==False]  # noqa: E712

        self.__validate_df_cat(df_cat)
        self.__set_df_cat(df_cat)

        self.override_names = override_names
        self.__override_names(override_names)


    ## IO
    def to_dir(self, path: str) -> None:
        """Saves the subscales to a directory.

        Parameters
        ----------
        path : str
            The path to the directory.
        """
        dir_ = Path(path)
        path_suffix_warning(dir_.suffix, obj="QuSubscales")
        empty_or_create_dir(dir_)

        self.quconfig.to_yaml(dir_ / "quconfig.yaml")
        write_data(self.df_cat, dir_ / "df_cat.gzip")
        with open(dir_ / "settings.json", "w") as f:
            settings_dict = {
                "override_names": self.override_names,
            }
            dump(settings_dict, f, indent=2)

    @staticmethod
    def from_dir(path) -> "QuSubscales":
        """Loads the subscales from a directory.

        Parameters
        ----------
        path : str
            The path to the directory.
        """
        dir_ = Path(path)
        path_suffix_warning(dir_.suffix)

        quconfig = QuConfig.from_yaml(dir_ / "quconfig.yaml")
        df_cat = read_data(dir_ / "df_cat.gzip")
        with open(dir_ / "settings.json", "r") as f:
            settings_dict = load(f)

        qusub = QuSubscales(
            quconfig=quconfig,
            df_cat=df_cat,
            **settings_dict,
        )
        print(f"Found QuSubscales containing the subscales:\n{qusub.get_subscales()}")
        return qusub

    def copy(self) -> "QuSubscales":
        """Creates a deep copy of the `QuestionnaireSubscales` object.

        Returns
        -------
        QuSubscales
        """
        override_names = self.override_names
        if override_names is not None:
            override_names = override_names.copy()
        return QuSubscales(
            quconfig=self.quconfig.copy(),
            df_cat=self.df_cat.copy(),
            override_names=self.override_names
        )


    ## Dunders
    def __str__(self):
        s = f"QuestionnaireSubscales(subscales: {self.get_subscales()})"
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: "QuSubscales") -> bool:
        return (
            dataframes_equal(self.df_cat, other.df_cat, check_names=True) and
            self.quconfig == other.quconfig and
            self.override_names == other.override_names
        )


    ## Internals
    def _get_X(self) -> np.ndarray:
        """Returns only the numeric data contained in the categorization
        dataframe ommitting the task-column.

        Returns
        -------
        np.ndarray
        """
        return self.df_cat.drop(columns=self.task_col).values

    def __validate_df_cat(self, df_cat: pd.DataFrame) -> None:
        allowed_msk = df_cat.drop(columns="Task").isin(
            self.allowed_values +
            self.quconfig._na_aliases
        )
        forbidden_msk = allowed_msk == False  # noqa: E712
        forbidden_cnt = forbidden_msk.sum().sum()
        if forbidden_cnt != 0:
            raise QuSubscalesError(
                f"There are {forbidden_cnt} non-allowed values in the categorization dataframe `df_cat`."
            )

        tasks = df_cat[self.task_col].to_list()
        tasks_config = self.quconfig.get_task_names()

        mismatch = np.setdiff1d(tasks, tasks_config)
        if len(mismatch) > 0:
            raise QuSubscalesError(
                "There are surplus tasks in the categorization dataframe `df_cat`,\n" +
                f"that are not present in the `QuestionnaireConfig`: \n{mismatch}."
            )

        mismatch = np.setdiff1d(tasks_config, tasks)
        if len(mismatch) > 0:
            raise QuSubscalesError(
                "There are tasks missing in the categorization dataframe `df_cat`,\n" +
                f"that are present in the `QuestionnaireConfig`: \n{mismatch}."
            )

    def __set_task_column(self, df_cat: pd.DataFrame) -> None:
        tcol_mask = df_cat.columns.str.lower() == "task"
        if np.sum(tcol_mask) != 1:
            raise QuSubscalesError(
                "There is not task-column in the categorization dataframe `df_cat`.\n" +
                "The task-column must be named \"Task\" (capitalization does not matter)."
            )
        self.task_col = df_cat.columns[tcol_mask][0]

    def __set_df_cat(self, df_cat: pd.DataFrame) -> None:
        tasks_order = self.quconfig.get_task_names()
        df_cat.index = df_cat[self.task_col]
        df_cat = df_cat.reindex(tasks_order)
        df_cat = df_cat.reset_index(drop=True)
        self.df_cat = df_cat

    def __override_names(self, override_names: list) -> None:
        if override_names is None:
            return
        old_names = self.get_subscales()
        if len(old_names) != len(override_names):
            raise QuSubscalesError(
                f"You are trying to overwrite the subscale-labels \n{old_names}" +
                f"with the new labels \n{override_names}." +
                "The length of these lists is not equal, i.e., this is invalid."
            )
        self.df_cat.columns = [[self.task_col] + override_names]


    ## Getting info
    def get_subscales(self) -> list[str]:
        """Returns the names of the questionnaire subscales.

        Returns
        -------
        list[str]
            Questionnaire subscale names.
        """
        return self.df_cat.drop(columns=self.task_col).columns.to_list()

    def get_taskcounts(self, as_dataframe: bool=False) -> dict[str, int]|pd.DataFrame:
        """Getting the taskcount per subscale of the contained subscales.

        Parameters
        ----------
        as_dataframe : bool
            Whether the taskcount sould be rerturned as a dataframe

        Returns
        -------
        dict[str, int]|pd.DataFrame
            Taskcounts in a dict with the subscale names as keys and the count as
            values / Taskcounts in a pandas dataframe
        """
        dct = self.df_cat.drop(columns=self.task_col).sum().to_dict()
        if not as_dataframe:
            return dct
        return pd.DataFrame(dct, index=["taskcount"])

    def get_itemcounts(self, as_dataframe: bool=False) -> dict[str, int]|pd.DataFrame:
        """Getting the itemcount per subscale of the contained subscales.

        Parameters
        ----------
        as_dataframe : bool
            Whether the itemcount sould be rerturned as a dataframe

        Returns
        -------
        dict[str, int]|pd.DataFrame
            Itemcounts in a dict with the subscale names as keys and the count as
            values / Itemcounts in a pandas dataframe
        """
        items_per_task = self.quconfig.get_item_counts_per_task()
        df = self.df_cat
        dct = {}
        for sscl in self.get_subscales():
            items = 0
            for task in self.quconfig.get_task_names():
                row = (df["Task"] == task).argmax()
                col = (df.columns == sscl).argmax()
                if df.iloc[row, col] > 0:
                    items += items_per_task[task]
            dct[sscl] = items
        if not as_dataframe:
            return dct
        return pd.DataFrame(dct, index=["itemcount"])


    def get_max_scores(self, as_dataframe: bool=False) -> dict[str, int]|pd.DataFrame:
        """Getting the maximum per subscale of the contained subscales.

        Parameters
        ----------
        as_dataframe : bool
            Whether the max scores sould be rerturned as a dataframe

        Returns
        -------
        dict[str, int]|pd.DataFrame
            Maxmimum scores in a dict with the subscale names as keys and the
            maximum score as values / Maxmimum scores in a pandas dataframe
        """
        X_cat = self._get_X()
        task_maxscores = list(self.quconfig.get_max_scores().values())
        task_maxscores = np.array(task_maxscores).reshape((1, -1))
        sscl_maxscores = task_maxscores @ X_cat

        dct = dict(zip(
            self.get_subscales(),
            np.squeeze(sscl_maxscores, axis=0)
        ))
        if not as_dataframe:
            return dct
        return pd.DataFrame(dct, index=["max_score"])


    ## Altering
    def drop_subscales(self, drop_scales: list[str], inplace: bool=True) -> Self:
        """Dropping subscales of the `QuestionnaireSubscales`.

        Parameters
        ----------
        drop_scales : list[str]
            Scales to drop.
        inplace : bool
            Whether the operation should be performed inplace.

        Returns
        -------
        Self
            For convenience.

        Raises
        ------
        QuestionnaireSubscalesError
        """
        for col in drop_scales:
            if col not in self.get_subscales():
                raise QuSubscalesError(
                    f"The (drop) subscale {col} is not part of the available subscales."
                )
        df_cat = self.df_cat.copy().drop(columns=drop_scales)
        if inplace:
            self.df_cat = df_cat
            ret = self
        else:
            ret  = QuSubscales(deepcopy(self.quconfig), df_cat)
        return ret

    def combine_subscales(self, old_scales: list[str], new_name: str, inplace: bool=True) -> Self:
        """Combining subscales by merging the respective columns of the
        categorization dataframe together. Afterwards, the values are clipped
        to 1.

        Parameters
        ----------
        old_scales : list[str]
            Scales to combine.
        new_name : str
            Name of the new, combined scale.
        inplace : bool
            Wether the operation should be performed inplace.

        Returns
        -------
        Self
            For convenience.

        Raises
        ------
        QuestionnaireSubscalesError
        """
        for col in old_scales:
            if col not in self.get_subscales():
                raise QuSubscalesError(
                    f"The (old) subscale {col} is not part of the available subscales."
                )
        df_cat = merge_and_clip(self.df_cat.copy(), old_scales, new_name)
        if inplace:
            self.df_cat = df_cat
            ret = self
        else:
            ret  = QuSubscales(deepcopy(self.quconfig), df_cat)
        return ret

    def rename_subscale(self, subscale: str, new_name: str, inplace: bool=True) -> Self:
        """Renaming a subscale.

        Parameters
        ----------
        subscale : str
            Scale to rename.
        new_name : str
            New name of the scale.
        inplace : bool
            Wether the operation should be performed inplace.

        Returns
        -------
        Self
            For convenience.

        Raises
        ------
        QuestionnaireSubscalesError
        """
        if subscale not in self.get_subscales():
            raise QuSubscalesError(
                f"The subscale \"{subscale}\" is not part of the available subscales." +
                "\nPerhaps you already renamed the subscale?"
            )
        df_cat = self.df_cat.rename(columns={subscale: new_name})
        if inplace:
            self.df_cat = df_cat
            ret = self
        else:
            ret  = QuSubscales(deepcopy(self.quconfig), df_cat)
        return ret


    ## Application
    def apply_to_dataframe(self, df_scr: pd.DataFrame, normed: bool=False) -> pd.DataFrame:
        """Applys the subscaling to passed (task-)scores.

        Parameters
        ----------
        df_scr : pd.DataFrame
            Dataframe with test edits in rows and tasks in columns.

        Returns
        -------
        pd.DataFrame
        """
        ids = None
        id_col = self.quconfig.id_col
        if id_col in df_scr.columns:
            ids = df_scr[id_col].values
            df_scr = df_scr.drop(columns=id_col)
        X_scr = df_scr.fillna(0).values

        X_cat = self._get_X()
        qusubscales = self.get_subscales()
        try:
            X_sscl = X_scr @ X_cat
        except ValueError as e:
            raise ValueError(
                "The shape of the score-data is not compatible with the subscales-allocation dataframe. \n" +
                "Note, that the subscales are meant to act on tasks (i. e. also mc-items must be scored). \n" +
                f"Original Error: {e}"
            )
        df_sscl = pd.DataFrame(X_sscl, columns=qusubscales)
        if ids is not None:
            df_sscl[id_col] = ids
            df_sscl = df_sscl[[id_col] + qusubscales]

        if normed:
            max_scores = self.get_max_scores()
            for k, v in max_scores.items():
                df_sscl[k] = df_sscl[k].astype(float) / float(v)

        return df_sscl

    def apply(self, qudata: QuData, normed: bool=False) -> pd.DataFrame:
        """Applys the subscales to a `QuestionnaireData` objects internal data.

        Parameters
        ----------
        qudata : QuestionnaireData

        Returns
        -------
        pd.DataFrame
        """
        return self.apply_to_dataframe(qudata.get_scr(True, False), normed=normed)


    def compare_allocations(self, other: "QuSubscales") -> pd.DataFrame:
        """Compares the `QuestionnaireSubscales` to another `QuestionnaireSubscales`
        with the goal to assess interrater reliability of the task-subscale allocation.

        Parameters
        ----------
        other : QuestionnaireSubscales
            The other `QuestionnaireSubscales` object.

        Returns
        -------
        pd.DataFrame
            A pandas Dataframe with Cohens Kappa accordance values for each subscale.
        """
        return compare_subscale_allocations(self, other)


    @staticmethod
    def compare_n_subscale_allocations(qusubs: dict[str, "QuSubscales"]) -> pd.DataFrame:
        """Compares multiple `QuestionnaireSubscales` objects with the goal to assess
        interrater reliability of the task-subscale allocation.

        Parameters
        ----------
        qusubs : dict[str, QuestionnaireSubscales]
            A dictionary with the `QuestionnaireSubscales` objects to compare. The keys
            are used to label the columns of the resulting dataframe, to denote, which
            2 `QuestionnaireSubscales` objects were compared.

        Returns
        -------
        pd.DataFrame
            A pandas Dataframe with Cohens Kappa accordance values for each subscale.
        """
        return compare_n_subscale_allocations(qusubs)




class SubscalesDimReducer(FunctionTransformer):
    X_cat: np.ndarray

    def __init__(self, qusubscales: QuSubscales):
        """A wrapper to scikit-learns `FunctionTransformer` to provide easier setup
        of the dimensionality reduction introduced by `QuestionnaireSubscales`.
        Can be used to map scores from a per-task dataframe to a per-subscale dataframe.
        Provides easily human readable formatting methods.

        Parameters
        ----------
        qusubscales : QuestionnaireSubscales
            The questionnaire subscales to use.
        """
        self.X_cat = qusubscales._get_X()
        qusubscales = qusubscales.get_subscales()
        super().__init__(self._matmul)
        self.qusubscales = qusubscales
        self.__n_in = self.X_cat.shape[0]
        self.__n_out = self.X_cat.shape[1]

    def _matmul(self, X: np.ndarray):
        return X @ self.X_cat

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.qusubscales

    def __str__(self):
        s = f"SubscalesDimReducer({self.__n_in} -> {self.__n_out})"
        return s

    def __repr__(self) -> str:
        return self.__str__()





def compare_subscale_allocations(qusub1: QuSubscales, qusub2: QuSubscales) -> pd.DataFrame:
    """Compares two `QuestionnaireSubscales` objects with the goal to assess
    interrater reliability of the task-subscale allocation.

    Parameters
    ----------
    qusub1 : QuestionnaireSubscales
        The first `QuestionnaireSubscales` object.
    qusub2 : QuestionnaireSubscales
        The second `QuestionnaireSubscales` object.

    Returns
    -------
    pd.DataFrame
        A pandas Dataframe with Cohens Kappa accordance values for each subscale.
    """
    subscales = qusub1.get_subscales()
    df1 = qusub1.df_cat[subscales]
    df2 = qusub2.df_cat[subscales]
    kappas = []
    for col in subscales:
        kappa = cohen_kappa_score(df1[col], df2[col])
        kappas.append(kappa)
    return pd.DataFrame(kappas, index=subscales, columns=["Cohen's Kappa"])



def compare_n_subscale_allocations(qusubs: dict[str, QuSubscales]) -> pd.DataFrame:
    """Compares multiple `QuestionnaireSubscales` objects with the goal to assess
    interrater reliability of the task-subscale allocation.

    Parameters
    ----------
    qusubs : dict[str, QuestionnaireSubscales]
        A dictionary with the `QuestionnaireSubscales` objects to compare. The keys
        are used to label the columns of the resulting dataframe, to denote, which
        2 `QuestionnaireSubscales` objects were compared.

    Returns
    -------
    pd.DataFrame
        A pandas Dataframe with Cohens Kappa accordance values for each subscale.
    """
    dfs = []
    existing = []
    for key1, qusub in qusubs.items():
        for key2, other in qusubs.items():
            if qusub != other and {key1, key2} not in existing:
                df = qusub.compare_allocations(other)
                df.columns = [f"{key1} vs {key2}"]
                dfs.append(df)
                existing.append({key1, key2})
    return pd.concat(dfs, axis=1)
