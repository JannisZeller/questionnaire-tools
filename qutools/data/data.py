"""
# Handling Questionnaire Data

Thisa submodule is designed to manage and manipulate questionnaire data within the qutools package.
It provides classes and functions for loading, processing, and analyzing questionnaire data.
It is built to work seamlessly with the configurations defined in qutools.data.config and the evaluation mechanisms
in qutools.core.trainulation, facilitating a comprehensive approach to questionnaire data management.
This submodule is essential for researchers and developers working with complex questionnaire datasets,
offering tools for data validation, transformation, and summarization.

Usage
-----
To use the qutools.data.data submodule, you first need to have your questionnaire data in a structured format,
such as a CSV file or a Pandas DataFrame. The submodule's functions and classes can then be used to validate the data against configurations
from qutools.data.config, prepare the data for analysis or machine learning models, and perform various data manipulation tasks.

Key steps in using the submodule include:

- **Loading Data**: Use the provided functions to load your questionnaire data into a suitable format for processing.
The data must obey the structure defined in the questionnaire's configuration ([QuConfig][qutools.data.config.QuConfig]) file, i. e., there must
be an ID-column, and a column at least for each task in the text- and score-data. It is strongly recommended to always
use the most fine-grained data available, i.e., the single items for text-data and the single multiplce-choice items for score-data.
Note that in the workflow of this package, using text-items in the score data is not senseful, because single text-items are not scored.
In such cases, view your text-"items" as tasks.

- **Validating against Configurations**: Utilize configurations defined in qutools.data.config to structure your data according to the questionnaire's layout and scoring logic.

- **Data Processing**: Employ the submodule's tools to clean, transform, and summarize your data, making it ready for analysis or model training.

Example
-------
```python
from qutools.data.config import QuConfig
from qutools.data.data import QuData

quconfig = QuConfig.from_yaml("./config/q-metadata-views.yaml")
qudata = QuData(
    quconfig=quconfig,
    df_txt="anonymous_text_data.gzip", # .gzip, .csv, and .xlsx are supported
    df_scr="anonymous_score_data.csv",
    text_col_type="items",
    mc_score_col_type="mc_items",
)
```
```
> Checked ID-matches. ✓
> Validated ID-columns. ✓
> Validated text-columns. ✓
> Cleaning text-data whitespaces. ✓
> All scores in correct ranges. ✓
> Validated score-columns. ✓
```
```python
df_scr = qudata.get_scr(mc_scored=True, verbose=True)
```
```
> Scoring of multiple-choice items.
> Scoring of multiple-choice tasks.
> Merging ['A5a.', 'A5b.', 'A5c.', 'A5d.', 'A5e.', 'A5f.'] to 'A5.'.
> Threshold-scoring of task A5., items: ['A5a.', 'A5b.', 'A5c.', 'A5d.', 'A5e.', 'A5f.']
```
```python
print(df_scr.head(2))
```
```
>     ID   A1a.  A1b.  A2.  A3.  A4.  A5
> 0  AAA   0.0   0.0  0.0  0.0  NaN  0.0
> 1  BBB   0.0   1.0  0.0  2.0  2.0  1.0
```
"""


import numpy as np
import pandas as pd

from json import load, dump
from pathlib import Path
from typing import Literal
from copy import deepcopy

from .config import QuConfig

from ..core.io import read_data, empty_or_create_dir, write_data, path_suffix_warning
from ..core.pandas import unite_str_columns, dataframes_equal
from ..core import (
    check_uniqueness,
    score_mc_items,
    score_mc_tasks,
    drop_missing_threshold,
    drop_earlystopping_threshold,
    cleanup_whitespaces,
    replace_abbreviations,
)



class QuDataError(Exception):
    """An Exception-class for errors in the context of `QuestionnaireData`
    objects.
    """
    pass


class QuData:
    _txt_init_cols: Literal["items", "tasks"]
    _scr_init_cols: Literal["mc_tasks", "mc_items", "no_mc"]

    def __init__(
        self,
        quconfig: QuConfig,
        df_txt: pd.DataFrame|str=None,
        df_scr: pd.DataFrame|str=None,
        mc_scores_only: bool=False,
        text_col_type: Literal["items", "tasks"]="items",
        mc_score_col_type: Literal["mc_items", "mc_tasks", "no_mc"]="mc_items",
        clip_sparingly_occurring_scores: bool=False,
        abbreviation_replacement_path: str=None,
        omit_tasks: list|Literal["text", "multiple_choice"]=None,
        verbose: bool=True,
        **kwargs,
    ) -> None:
        """Generates a `QuestionnaireData` object encapsulating all information
        and responses to the questionnaire. The scores table `df_scr` can (in
        the logic of the `QuestionnaireConfig`) only contain scores for text-
        tasks but in some cases might contain the multiple-choice items already
        scored (as "tasks") or as single "items". This is handeled
        automatically.
        Analoguously the text-data `df_txt` can be either passed with responses
        for the single items or already task-wise concatenated.

        Both data-tables must be in wide-format with a unique person-identifier
        in the `qconfid.id_col`-column.

        Parameters
        ----------
        quconfig : QuestionnaireConfig
            The questionnaire config, already loaded from a yaml file.
        df_txt : pd.DataFrame|str
            The textual responses to the single text items or tasks. Can be a
            pandas dataframe or a path.
        df_scr : pd.DataFrame|str
            The scores for the text-tasks and multiple-choice tasks or items
            (see above). Can be a pandas dataframe or a path.
        mc_scores_only : bool
            Wether only multiple choice scores are available. This is used if
            some "unscored" QuData is setup, i.e., with new assessments.
        text_col_type : Literal["items", "tasks"]
            The type of columns for the text-tasks/items passed in the
            text data. Passing them as items is preferred.
        mc_score_col_type: Literal["mc_items", "mc_tasks", "no_mc"]
            The type of columns for the multiple choice tasks/items passed in the
            score data. Passing them as items is preferred.
        abbreviation_replacement_path : str
            A path to an abbreviations replacement table as described in the core
            functions.
        omit_tasks  : list|Literal["text", "multiple_choice"]
            Some tasks can be omitted. They will be dropped from the config and
            all data tables before any additional validation is carried out.
            Note that only tasks, not items can be omitted.
        verbose : bool
            Whether verbose information should be printed during setup and later
            on.
        """
        quconfig = deepcopy(quconfig)
        self.verbose = verbose
        self.id_col = quconfig.id_col
        self.clip_sos = clip_sparingly_occurring_scores

        self._scr_init_cols = mc_score_col_type
        self._txt_init_cols = text_col_type

        df_txt = self.__load_data(df_txt)
        df_scr = self.__load_data(df_scr)

        self._omit_tasks = omit_tasks
        if omit_tasks is not None:
            _, tol, iol = quconfig.omit_tasks(omit_tasks)
            df_txt = QuData.__omit_tasks(tol, iol, df_txt)
            df_scr = QuData.__omit_tasks(tol, iol, df_scr)

        if hasattr(quconfig, "task_omit_list"):
            df_txt = self.__omit_tasks_qconfig(quconfig=quconfig, df=df_txt)
            df_scr = self.__omit_tasks_qconfig(quconfig=quconfig, df=df_scr)

        self.quconfig = quconfig

        self.__validate_id_cols(df_txt, df_scr, kwargs.get("check_id_matches", True))

        self.__validate_txt_cols(df_txt)
        df_txt = self.__cleanup_df_txt_wsp(df_txt)
        df_txt = self.__fill_na_aliases(df_txt)
        if abbreviation_replacement_path is not None:
            self._df_abbr = read_data(abbreviation_replacement_path)
            cols = self.quconfig.get_text_columns(self._txt_init_cols)
            df_txt = replace_abbreviations(
                path=abbreviation_replacement_path,
                df=df_txt,
                cols=cols,
            )
            df_txt = self.__fill_na_aliases(df_txt)
        else:
            self._df_abbr = None

        self._abbreviations_removed = kwargs.get("abbreviations_removed", False)
        if self._df_abbr is not None or self._abbreviations_removed:
            self._abbreviations_removed = True
            print("Replacing text-data abbreviations. ✓")
        self.df_txt = df_txt

        self.__validate_scr_cols(df_scr=df_scr, mc_only=mc_scores_only)
        df_scr = self.__fill_na_aliases(df_scr)
        self.df_scr = df_scr

        self._sos_info_str = kwargs.get("sos_info_str", None)
        if self.clip_sos:
            if self._sos_info_str is None:
                self.__sos_clip()
            else:
                if self.verbose:
                    print(self._sos_info_str)


    ## IO and Comparison
    def _get_settings_dict(self) -> dict:
        return {
            "text_col_type": self._txt_init_cols,
            "mc_score_col_type": self._scr_init_cols,
            "abbreviations_removed": self._abbreviations_removed,
            "sos_info_str": self._sos_info_str,
            "clip_sparingly_occurring_scores": self.clip_sos,
            "omit_tasks": self._omit_tasks,
            "verbose": self.verbose,
        }

    def __eq__(self, other: "QuData") -> bool:
        return (
            self.quconfig == other.quconfig and
            dataframes_equal(self.df_scr, other.df_scr) and
            dataframes_equal(self.df_txt, other.df_txt) and
            self._get_settings_dict() == other._get_settings_dict()
        )


    def filter_ids(self, contains: str, exclude: bool=False, regex: bool=False):
        """Filter the data by the IDs containing a certain string. Regex can be
        enabled

        Parameters
        ----------
        contains : str
            The string that the IDs must contain.
        exclude : bool
            Wether the IDs should be excluded that contain the `contains`-string
        regex : bool
            Whether the `contains`-string should be interpreted as a regex.
        """
        def _filter_df(df: pd.DataFrame):
            msk = df[self.id_col].str.contains(pat=contains, regex=regex)
            msk = msk if not exclude else ~msk
            return df[msk].reset_index(drop=True)
        self.df_txt = _filter_df(self.df_txt)
        self.df_scr = _filter_df(self.df_scr)


    def to_dir(self, path: str):
        dir_ = Path(path)
        path_suffix_warning(dir_.suffix, obj="QuData")
        empty_or_create_dir(dir_)

        self.quconfig.to_yaml(dir_ / "quconfig.yaml")

        if self.df_txt is not None:
            write_data(self.df_txt, dir_ / "df_txt.gzip")
        if self.df_scr is not None:
            write_data(self.df_scr, dir_ / "df_scr.gzip")

        settings_dict = self._get_settings_dict()

        with open(dir_ / "settings.json", "w") as f:
            dump(settings_dict, f, indent=2)

    @staticmethod
    def from_dir(path: str, verbose: bool=True) -> "QuData":
        dir_ = Path(path)
        quconfig = QuConfig.from_yaml(dir_ / "quconfig.yaml")

        df_txt = read_data(dir_ / "df_txt.gzip", return_none=True)
        na_msk = df_txt.isnull().values
        df_txt[na_msk] = np.nan

        df_scr = read_data(dir_ / "df_scr.gzip", return_none=True)

        with open(dir_ / "settings.json", "r") as f:
            settings_dict = load(f)
        settings_dict["verbose"] = verbose

        return QuData(
            quconfig=quconfig,
            df_txt=df_txt,
            df_scr=df_scr,
            **settings_dict,
        )



    ## Info
    def mc_item_scores_available(self) -> bool:
        if len(self.quconfig.get_mc_item_names()) == 0:
            return True
        return self._scr_init_cols == "mc_items"

    def __str__(self) -> str:
        s = "QuData("
        s += f"\n  - Config: {str(self.quconfig)}"
        s += f"\n  - Data: {self.short_str()}"
        s += "\n)"

        return s

    def short_str(self) -> str:
        s = ""
        if self.df_txt is not None:
            s += f"text-data: {self.df_txt.shape} ({self._txt_init_cols}-cols)"
        else:
            s += "text-data: None"

        s += " | "

        if self.df_scr is not None:
            s += f"score-data: {self.df_scr.shape} ({self._scr_init_cols}-cols)"
        else:
            s += "score-data: None"
        return s


    ## Retrieval and Transformations
    def __sos_clip(self):
        df = self.get_scr(mc_scored=True, verbose=False).drop(columns=self.id_col)
        values = df.fillna(0).values.flatten()
        v_counts = dict(zip(*np.unique(values, return_counts=True)))
        v_total = np.sum(list(v_counts.values()))
        v_rel = {v: 100 * k/v_total for v, k in v_counts.items()}
        v_clip = [k for k, v in v_rel.items() if v < 1]

        values = np.array(list(v_counts.keys()))
        clip_dict = {}
        for v in v_clip:
            diffs = np.abs(values - v)
            diffs[diffs==0] = np.inf
            nearest = np.argmin(diffs)
            clip_dict[v] = nearest

        print_dict = {k: (str(round(v, 2)) + ' %') for k, v in v_rel.items()}
        self._sos_info_str = (
            "Found the following score proportions:" +
            f"\n\t {print_dict}" +
            f"\n\t-> Replacing according to {clip_dict}"
        )
        if self.verbose:
            print(self._sos_info_str)

        df = self.df_scr.drop(columns=self.id_col)
        df = df.replace(clip_dict).reset_index(drop=True)
        df_ids = self.df_scr[self.id_col].copy().reset_index(drop=True)
        df = pd.concat([df_ids, df], axis=1)
        self.df_scr = df
        self._sos_clip_dict = clip_dict

    def get_scr(self,
        mc_scored: bool=True,
        verbose: bool=True,
        drop_incomplete: bool=False,
        drop_earlystoppers: bool=False,
        **kwargs,
    ):
        """Returns a unified and structured dataset of the available score-data.
        Keyword-arguments ca contain additional parameters passed to internal
        functions. Refer to the implementation for details.

        Parameters
        ----------
        mc_scored : bool
            Whether the multiple choice tasks should get scored.
        verbose : bool
            Whether information on the scoring procedure should be printed.
        drop_incomplete : bool
            Whether incomplete test-edits should be dropped.
        drop_earlystoppers : bool
            Whether earlystopped test-edits should be dropped.

        Raises
        ------
        QuestionnaireDataError
        """
        self.__df_scr_exists()
        if mc_scored:
            if self._scr_init_cols == "mc_items":
                df_ret = self.__score_mc_items(verbose)
            else:
                df_ret = self.df_scr.copy()
        else:
            if self._scr_init_cols == "mc_tasks":
                raise QuDataError(
                    "\nThis `QuestionnaireData` has been generated using a score-dataset with\n" +
                    "the multiple-choice tasks already scored - at least from what the\n" +
                    "QuestionnaireConfig passed is denoting. Therefore the unscored\n" +
                    "multiple-choice scores can not be returned, i.e., the `mc_scored`\n" +
                    "argument can not be `False`."
                )
            else:
                df_ret = self.df_scr

        if drop_incomplete:
            df_ret = drop_missing_threshold(
                df_scr=df_ret,
                spare_columns=[self.id_col],
                threshold=kwargs.get("incomplete_threshold", 0.5),
                verbose=verbose,
            )

        if drop_earlystoppers:
            df_ret = drop_earlystopping_threshold(
                df_scr=df_ret,
                spare_columns=[self.id_col],
                threshold=kwargs.get("earlystopping_threshold", 0.25),
                verbose=verbose,
            )

        return df_ret

    def get_n_classes(self) -> int:
        self.__df_scr_exists()
        text_cols = self.quconfig.get_text_columns("tasks")
        df_scr = self.get_scr(mc_scored=True, verbose=False)
        X_scr = df_scr[text_cols].to_numpy().flatten()
        return int(max(X_scr) + 1)

    def get_scr_dropout(
        self,
        mc_scored: bool=True,
        verbose: bool=True,
        incomplete_threshold: float=0.5,
        earlystopping_threshold: float=0.25,
    ) -> pd.DataFrame:
        """Returns a unified and structured dataset of the available score-data.
        This is a wrapper for the `.get_scr`-method with the drop-arguments set
        to true. The `incomplete_threshold` and `earlystopping_threshold` get
        passed on as keyword arguments to `.get_scr`.

        Parameters
        ----------
        mc_scored : bool
            Whether the multiple choice tasks should get scored.
        verbose : bool
            Whether information on the scoring procedure should be printed.
        incomplete_threshold : bool
            Test edits with a relative amount of missings exceeding this
            threshold will be dropped (rowwise).
        earlystopping_threshold : bool
            Test edits with a relative amount of consecutive missings exceeding
            at the end of the test-edit exceeding this threshold will be dropped
            (rowwise).

        Raises
        ------
        QuestionnaireDataError
        """
        return self.get_scr(
            mc_scored=mc_scored,
            verbose=verbose,
            drop_incomplete=True,
            drop_earlystoppers=True,
            incomplete_threshold=incomplete_threshold,
            earlystopping_threshold=earlystopping_threshold
        )

    def get_txt(
        self,
        units: Literal["tasks", "items", "persons"],
        table: Literal["wide", "long"]=None,
        with_scores: bool=False,
        sep: str=" ",
        verbose: bool=True,
    ) -> pd.DataFrame:
        """Returns the text-responses in different formats.

        Parameters
        ----------
        units : Literal["tasks", "items", "persons"]
            Whether the single item-texts should be concatenated item-wise (no
            concatenation), task-wise or person-wise.
        table : Literal["wide", "long"]
            Wether the table should be in wide or long format. If `units` is
            "tasks" or "items", this must be set. If `units` is "persons", this
            has no effect and can be set to `None`.
        sep : str
            Separator-string if concatenations (task- or person-wise) are
            required.
        with_scores : bool
            Wether the scores should be appended in an additional column.
        verbose : bool
            Whether information on the scoring procedure should be printed.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
        """
        self.__df_txt_exists()
        df_ret = self.df_txt

        if units == "tasks":
            if self._txt_init_cols == "items":
                df_ret = self.__concat_txt_tasks()
        elif units == "items":
            if self._txt_init_cols == "tasks":
                raise QuDataError(
                    "The text-data was initially passed as tasks-wise concatenated and you requested them item-wise.\n" +
                    "This can not be reconstructed."
                )
        elif units == "persons":
            if table is not None and self.verbose:
                print("Info: If `units` is \"persons\", the `tables` parameter has no effect and can be ommited.")
            text_cols = df_ret.drop(columns=self.id_col).columns
            df_ret = unite_str_columns(
                df=df_ret,
                cols=text_cols,
                new_name="text",
                drop=True,
                sep=sep,
            )
            df_ret = df_ret.dropna()
        else:
            raise ValueError("The `units` parameter must be \"tasks\", \"items\" or \"persons\".")

        if table is None:
            if units != "persons":
                raise ValueError("If the `units` parameter is not \"persons\", the `table` parameter can not be `None`.")
        elif table == "wide":
            pass
        elif table == "long":
            if units != "persons":
                df_ret = df_ret.melt(id_vars=self.id_col, var_name=units[:-1], value_name="text")
                df_ret = df_ret.dropna()
        else:
            raise ValueError("The `table` parameter must be \"wide\" or \"long\".")

        df_ret = df_ret.reset_index(drop=True)

        if self.df_scr is not None:
            txt_ids = list(df_ret[self.id_col].unique())
            scr_ids = list(self.df_scr[self.id_col])
            mismatch = np.setdiff1d(scr_ids, txt_ids)
            if len(mismatch) > 0 and verbose:
                print(
                    "Info: There are persons/test edits without any valid text-responses, which will not \n" +
                    "\tbe included in the `.get_txt(...)`-method's result:\n" +
                    f"\t{mismatch}"
                )

        if with_scores:
            if self.df_scr is None:
                print("Warning: No score-data available. Can not append scores.")
                return df_ret

            if units == "persons":
                df_scr = self.get_total_score()
                df_ret = pd.merge(df_ret, df_scr, how="inner", on=[self.id_col])
                return df_ret

            if table == "wide":
                print("Warning: For `table==\"wide\"` and `units!=\"persons\"` the scores can not be appended.")
                return df_ret

            if units == "items":
                print("Warning: For `units!=\"items\"` the scores can not be appended.")
                return df_ret

            df_scr = self.get_scr(mc_scored=True, verbose=False)
            df_scr = df_scr.fillna(0)
            df_scr = df_scr.melt(id_vars=self.id_col, var_name="task", value_name="score")
            df_ret = pd.merge(df_ret, df_scr, how="inner", on=[self.id_col, "task"])

        return df_ret

    def get_total_score(self, normed: bool=False) -> pd.DataFrame:
        self.__df_scr_exists()
        df_ret = self.get_scr(verbose=False).copy()
        df_ret["total_score"] = df_ret.drop(columns=self.id_col).sum(axis=1)
        df_ret = df_ret[[self.id_col, "total_score"]].copy()
        if normed:
            df_ret["total_score"] = df_ret["total_score"] / self.quconfig.get_max_score()
        return df_ret

    def omit_tasks(self, omit_tasks: list|Literal["text", "multiple_choice"]) -> "QuData":
        quconfig = self.quconfig
        df_scr = self.df_scr
        df_txt = self.df_txt
        self = QuData(
            quconfig=quconfig,
            df_txt=df_txt,
            df_scr=df_scr,
            omit_tasks=omit_tasks,
        )
        return self

    def drop_ids(self, ids: list) -> "QuData":
        """Unility to drop IDs. This might be necessairy if subsets of the data
        or subsets of the tasks are analyzed.

        Parameters
        ----------
        ids : list
            List of IDs to drop.

        Returns
        -------
        QuestionnaireData
            For convenience.

        Raises
        ------
        QuestionnaireDataError
        """
        df_txt = self.df_txt
        df_scr = self.df_scr
        self.df_txt = df_txt[df_txt[self.id_col].isin(ids)==False]
        self.df_scr = df_scr[df_scr[self.id_col].isin(ids)==False]


    ## Internals
    @staticmethod
    def __omit_tasks(
        task_omit_list: list[str],
        item_omit_list: list[str],
        df: pd.DataFrame,
    ) -> tuple[QuConfig, pd.DataFrame, pd.DataFrame]:
        if df is None:
            return None
        df = df.drop(columns=task_omit_list, errors="ignore")
        df = df.drop(columns=item_omit_list, errors="ignore")
        return df

    def __load_data(self, df: pd.DataFrame|str) -> pd.DataFrame:
        if df is None:
            return None
        if not isinstance(df, pd.DataFrame):
            df = read_data(df)
        df = df.sort_values(by=self.id_col).reset_index(drop=True)
        return df

    def __omit_tasks_qconfig(
        self,
        quconfig: QuConfig,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df is None:
            return df
        print(
            f"Omitting tasks {quconfig.task_omit_list}," +
            "\nthat were previously dropped from the quconfig." +
            "\nNote that this can only happen before column-validation and is therefore error prone." +
            "\nConsider dropping the respective items manually by loading and manipulating the tables with " +
            "\npandas, or in with some external tooling."
        )
        df = df.drop(columns=quconfig.item_omit_list, errors="ignore")
        df = df.drop(columns=quconfig.task_omit_list, errors="ignore")
        return df

    def __score_mc_items(self, verbose: bool=True) -> pd.DataFrame:
        self.__df_scr_exists()
        mc_tasks = self.quconfig.get_mc_tasks()
        df_scr = self.df_scr.copy()
        if len(mc_tasks) > 0:
            df_scr = score_mc_items(df_scr, mc_tasks=mc_tasks, verbose=verbose)
            df_scr = score_mc_tasks(df_scr, mc_tasks=mc_tasks, verbose=verbose)
        return df_scr

    def __concat_txt_tasks(self, sep: str=" ") -> pd.DataFrame:
        self.__df_txt_exists()
        df = self.df_txt
        for task in self.quconfig.get_text_tasks():
            items = task.get_item_names()
            if len(items) > 1:
                df = unite_str_columns(
                    df=df,
                    cols=items,
                    new_name=task.name,
                    drop=True,
                    sep=sep,
                )
            else:
                df = df.rename(columns={items[0]: task.name})
        return df

    def __fill_na_aliases(self, df: pd.DataFrame, fill_val=np.nan) -> pd.DataFrame:
        if df is None:
            return None
        for col in df.columns:
            if df[col].dtype==object or df[col].dtype==str:
                df[col] = df[col].str.strip()
        with pd.option_context('future.no_silent_downcasting', True):
            df = df.replace(self.quconfig._na_aliases, fill_val)
        return df

    def __cleanup_df_txt_wsp(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return None
        cols = self.quconfig.get_text_columns(self._txt_init_cols)
        df = cleanup_whitespaces(df, cols)
        if self.verbose:
            print("Cleaning text-data whitespaces. ✓")
        return df

    def __df_txt_exists(self) -> None:
        if self.df_txt is None:
            raise QuDataError(
                "This QuestionnaireData does not contain text-data."
            )

    def __df_scr_exists(self) -> None:
        if self.df_scr is None:
            raise QuDataError(
                "This QuestionnaireData does not contain score-data."
            )


    ## Validators
    def __validate_id_col(self, df: pd.DataFrame, data_name: Literal["data", "text-data", "score-data"]="data"):
        if df is None:
            return
        if self.id_col not in df.columns:
            raise QuDataError(
                f"\nThe id column `{self.id_col}` specified in the `QuestionnaireConfig`\n" +
                f"is not present in the {data_name}."
            )
        ids = df[self.id_col].to_list()
        check_uniqueness(
            arr=ids,
            Ex=QuDataError,
            ex_str=f"The IDs of the passed {data_name} are non unique.",
        )

    def __validate_id_cols(self, df_txt: pd.DataFrame, df_scr: pd.DataFrame, check_id_matches: bool=True):
        self.__validate_id_col(df_txt, "text-data")
        self.__validate_id_col(df_scr, "score-data")
        if (df_txt is not None) and (df_scr is not None):
            if check_id_matches:
                self.quconfig._check_id_matches(df1=df_scr, df2=df_txt)
                if self.verbose:
                    print("Checked ID-matches. ✓")
            if self.verbose:
                print("Validated ID-columns. ✓")

    def __validate_txt_cols(self, df_txt: pd.DataFrame) -> None:
        if df_txt is None:
            return
        self.quconfig.validate_wide_df_txt(
            df_txt=df_txt,
            units=self._txt_init_cols,
            all_units=True,
        )
        if self.verbose:
            print("Validated text-columns. ✓")

    def __validate_scr_cols(self, df_scr: pd.DataFrame, mc_only: bool=False) -> None:
        if df_scr is None:
            return
        self.quconfig.validate_wide_df_scr(
            df_scr=df_scr,
            units=self._scr_init_cols,
            all_units=True,
            mc_only=mc_only,
            validate_scores="error",
        )
        if self.verbose:
            print("All scores in correct ranges. ✓")
            print("Validated score-columns. ✓")
