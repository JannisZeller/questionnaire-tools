"""
# Questionnaire Configurations

The `config.py` module provides classes and utilities for configuring and
validating questionnaire metadata. It defines structures for handling configurations related to questionnaire
views, tasks, and data validation. Questionnaires in this package are structured in
three layers:

1. **Items**: Items are the single response elements, e.g., the text in
a single text field of the questionnaire, or the true or false value
of a single multiple choice option.

2. **Tasks**: Tasks are the elements actually used for scoring. I. e.,
"A1a." and "A1b." might get are scored separately. But "A1b." might
consists of two text-fields though but these are scored together.
Similarly, multiple choice tasks might consist of several single
choices that might be considered as items.

3. **Views**: Grouped tasks that are intendet to be presented to the
participants together.

In general tasks are used by human or machine scorers as the units that
get scored. Therefore, typically also analyses of the scores use this
layer. Items might be needed to separate inputs. When configuring the
analysis by setting the yaml-file such that the wanted structure is
achieved. For instance one might want to consider all elements of the
questionnaire as tasks, if every element should be scored separately.

The configurations for single views, tasks and items are loaded from
[qutools.core.config][qutools.core.config]. This module is essential for ensuring that questionnaire data is correctly
structured and processed according to predefined schemas and rules.

Usage
-----
The primary class in this module, `QuConfig`, is used to wrap and validate questionnaire metadata. This includes
defining views (how questionnaire data is presented or grouped), specifying the identifier column for
corresponding data, and setting aliases for representing missing data. The module also defines custom exceptions
for handling errors specific to questionnaire configuration and data validation.

Example
-------
Below is an example of how to use the `QuConfig` class to define a questionnaire configuration, including
specifying views and handling missing data aliases.

```python
from qutools.data.config import MCItemConfig, TextItemConfig
from qutools.data.config import MCTaskConfig, TextTaskConfig
from qutools.data.config import ViewConfig
from qutools.data.config import QuConfig

na_aliases = ['NA', 'n/a', 999]

quconfig = QuConfig(
    id_col='ID',
    na_aliases=na_aliases,
    views=[
        ViewConfig(
            view_name='A1.',
            tasks=[
                TextTaskConfig(
                    task_name="A1a.",
                    max_score=2,
                    items=[
                        TextItemConfig(name='A1a.'),
                    ]
                ),
                TextTaskConfig(
                    task_name="A1b.",
                    max_score=2,
                    items=[
                        TextItemConfig(name='A1b.1'),
                        TextItemConfig(name='A1b.2'),
                    ]
                )
            ]
        ),
        ViewConfig(
            view_name='A2.',
            tasks=[
                TextTaskConfig(
                    task_name="A2.",
                    max_score=2,
                    items=[
                        TextItemConfig(name='A2.1'),
                        TextItemConfig(name='A2.2')
                    ]
                )
            ]
        ),
        ViewConfig(
            view_name='A3',
            tasks=[
                MCTaskConfig(
                    task_name="Task 1.1.",
                    max_score=2,
                    scoring="summation",
                    items=[
                        MCItemConfig(name='A3a.', correct_response=1),
                        MCItemConfig(name='A3b.', correct_response=1),
                        MCItemConfig(name='A3c.', correct_response=0),
                        MCItemConfig(name='A3d.', correct_response=1),
                    ]
                )
            ]
        ),
    ],
)

# for instance:
quconfig.get_max_scores()
# {'A1a.': 2, 'A1b.': 2, 'A2.': 2, 'Task 1.1.': 2}
```

This is equivalent to using the following yaml-configuration file:
```yaml
# yaml
metadata:
  id_column: "ID"

views:
  A1.:
    viewinfo: "This is some information on view A1."
    tasks:
      A1a.:
        task_type: "text"
        max_score: 1
        items:
          - A1a.
      A1b.:
        task_type: "text"
        max_score: 2
        items:
          - A1b.1
          - A1b.2
  A2.:
    tasks:
      A2.:
        task_type: "text"
        max_score: 2
        items:
          - A2.1
          - A2.2
  A3.:
    tasks:
      A3.:
        task_type: "multiple_choice"
        scoring: "thresholds"
        max_score: 2
        items:
          A3a.:
            correct_response: 1
          A3b.:
            correct_response: 1
          A3c.:
            correct_response: 0
          A3d.:
            correct_response: 1
```

... and loading it via:
```python
quconfig = QuConfig.from_yaml("path/to/config.yaml")
```

Tip
---
When working with a dataset where the multiple-choice items/tasks are already
scored, just set the `correct_response` fields of all those items to `1` (formatted as an integer).

"""



import numpy as np
import pandas as pd

from pathlib import Path
from typing import Literal, Self
from yaml import dump as ymldump

from copy import deepcopy
from dataclasses import dataclass

from ..core.config import ItemConfig
from ..core.config import TaskConfig, MCTaskConfig, TextTaskConfig
from ..core.config import ViewConfig

from ..core.yaml_loader import UniqueKeyError
from ..core.yaml_loader import load_unique_key_yaml
from ..core.validation import check_type, check_options
from ..core.pandas import pivot_to_wide



class QuConfigError(Exception):
    """An Exception-class for errors in the context of `QuestionnaireConfig`
    objects.
    """
    pass

class QuDataInvalidError(Exception):
    """An Exception-class for data-validation.
    """
    pass



@dataclass
class QuConfig:
    """Class for wrapping the questionnaire meta-data.
    """
    views: list[ViewConfig]
    id_col: str
    _na_aliases: list[str|int|float]

    def __init__(
        self,
        id_col: str,
        views: list[ViewConfig],
        na_aliases: list[str|int|float],
    ) -> None:
        """Initializes a `QuestionnaireConfig`-instance. Validates the structure
        and types of the task-dict passed.

        Parameters
        ----------
        id_col : str
            The column-name of the id-column in the data.
        views : list[ViewConfig]
            A list of `ViewConfig`-instances.
        na_aliases : list[str|int|float]
            A list of aliases for missing values in the data.
        """
        self.id_col = id_col
        self.views = views
        self._na_aliases = na_aliases



    ## IO
    def copy(self) -> "QuConfig":
        """
        Returns a deep copy of the `QuestionnaireConfig`-instance.

        Returns
        -------
        QuConfig
        """
        return deepcopy(self)

    @staticmethod
    def from_yaml(path: str) -> "QuConfig":
        """Initializes a questionnaire config from a yaml-file.

        Parameters
        ----------
        path : str
            Path to a yaml-file containing the questionnaire metadata in a
            fixed structure.
        """
        try:
            config_dict = load_unique_key_yaml(path)
        except UniqueKeyError as e:
            raise QuConfigError(f"Error when loading the metadata from the yaml file. There are duplicate keys. Probabliy a duplicate task name or a duplicate key in one of the tasks. Orignal error: {e}")

        try:
            metadata = config_dict["metadata"]
        except KeyError:
            metadata = {}
            print(
                "Warning: You did not specify any metadata, defaults are used.\n" +
                "Refer to the documentation for the default values."
            )
        id_col: str = QuConfig.__get_id_col(metadata)
        na_aliases: list = QuConfig.__get_na_aliases(metadata)

        toplevel_tasks = "tasks" in config_dict
        toplevel_views = "views" in config_dict

        if (not toplevel_tasks and not toplevel_views) or (toplevel_tasks and toplevel_views):
            raise QuConfigError(
                "The top-level key besides `\"metadata\"` must either be `\"tasks\"` or `\"views\"`."
            )
        elif toplevel_tasks:
            tasks = QuConfig.__construct_tasks(config_dict["tasks"])
            views = [ViewConfig(view_name="Q", tasks=tasks)]
        elif toplevel_views:
            views = QuConfig.__construct_views(config_dict["views"])

        return QuConfig(id_col=id_col, na_aliases=na_aliases, views=views)

    def to_dict(self) -> dict:
        """
        Returns the questionnaire configuration as a dictionary.

        Returns
        -------
        dict
        """
        return {
            "metadata": {
                "id_column": self.id_col,
                "na_aliases": self._na_aliases,
            },
            "views": {view.name: view.to_dict() for view in self.views}
        }

    def to_yaml(self, path: str) -> None:
        """
        Saves the questionnaire configuration to a yaml-file.

        Parameters
        ----------
        path : str
            The path to the yaml-file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            ymldump(self.to_dict(), f, sort_keys=False)


    ## Internals
    def __str__(self) -> str:
        """
        Returns a representation of the `QuestionnaireConfig`-instance as a string.

        Returns
        -------
        str
        """
        tic = self.get_text_task_count()
        mic = self.get_mc_task_count()
        return f"QuestionnaireConfig({tic} text, {mic} multiple_choice)"

    def __eq__(self, other: Self) -> bool:
        """
        Compares two `QuestionnaireConfig`-instances for equality.

        Parameters
        ----------
        other : QuConfig
            The other `QuestionnaireConfig`-instance to compare to.

        Returns
        -------
        bool
        """
        if self.views != other.views:
            return False
        if self.id_col != other.id_col:
            return False
        if len(np.setdiff1d(self._na_aliases, other._na_aliases)) != 0:
            return False
        if len(np.setdiff1d(other._na_aliases, self._na_aliases)) != 0:
            return False
        return True

    @staticmethod
    def __get_id_col(meta_dict: dict) -> str:
        if "id_column" not in meta_dict:
            print("Warning: The `id_column` is not specified in the config-yaml under the \"metadata\" primary key. Defaulting to \"ID\"")
            return "ID"

        id_col = meta_dict["id_column"]
        check_type(arg=id_col, type_=str, Ex=QuConfigError, arg_name="id_column")
        return id_col

    @staticmethod
    def __get_na_aliases(meta_dict: dict) -> None:
        if "na_aliases" not in meta_dict:
            defaults = [99, "99", "", "-", "--", "NA", "Na", "na", "Null", "null", " ", "nan", "NaN", "Nan", "NAN", np.nan]
            print(f"Warning: The `na_aliases` are not specified in the config-yaml under the \"metadata\" primary key. Defaulting to \n{defaults}.")
            return defaults

        na_aliases = meta_dict["na_aliases"]
        na_aliases += [np.nan]
        check_type(
            arg=na_aliases,
            type_=list,
            Ex=QuConfigError,
            arg_name="na_aliases",
        )
        return na_aliases

    @staticmethod
    def __construct_tasks(tasks_dict: dict[str, dict]) -> list[TaskConfig]:
        ret = []
        for task_name, task_cfg in tasks_dict.items():
            task = TaskConfig.from_dict(task_name, task_cfg)
            ret.append(task)
        return ret

    @staticmethod
    def __construct_views(views_dict: dict[str, dict]) -> list[TaskConfig]:
        ret = []
        for view_name, view_cfg in views_dict.items():
            view = ViewConfig.from_dict(view_name, view_cfg)
            ret.append(view)
        return ret


    ## External
    def get_text_columns(self, option: Literal["items", "tasks"]) -> list[str]:
        """Returns a list of the (expected) text-columns of a suitable data-
        file containing text-responses to the questionnaire.

        Parameters
        ----------
        option : Literal["items", "tasks"]
            Whether the columns should countain the text as single items
            (one column per item), or as tasks (one column per task).

        Returns
        -------
        list[str]
        """
        text_tasks = self.get_text_tasks()
        if option == "tasks":
            return [task.name for task in text_tasks]
        elif option == "items":
            cols = []
            for task in text_tasks:
                cols += task.get_item_names()
            return cols
        else:
            raise QuConfigError("The `option` must be in `[\"items\", \"tasks\"]`.")

    def get_scores_columns(
        self,
        option: Literal["mc_items", "mc_tasks", "no_mc"],
        mc_only: bool=False,
    ) -> list[str]:
        """Returns a list of the (expected) score-columns of a suitable data-
        file containing scores to the questionnaire.

        Parameters
        ----------
        option : Literal["mc_items", "mc_tasks", "no_mc"]
            Whether the columns should countain the multiple-choice tasks as
            single items (one column per item), as tasks (one column per task)
            or if the multiple-choice tasks should be ommitted.
        mc_only : bool
            Wether only the multiple choice columns should be returned.

        Returns
        -------
        list[str]
        """
        if option == "mc_tasks":
            if mc_only:
                return self.get_mc_task_names()
            return [task.name for task in self.get_tasks()]

        elif option == "no_mc":
            if mc_only:
                return []
            return [task.name for task in self.get_text_tasks()]

        elif option == "mc_items":
            if mc_only:
                return self.get_mc_item_names()
            cols = []
            for task in self.get_tasks():
                if task.type == "multiple_choice":
                    cols += task.get_item_names()
                else:
                    cols.append(task.name)
            return cols

        else:
            raise QuConfigError("The `option` must be in `[\"mc_items\", \"mc_tasks\", \"no_mc\"]`.")

    #   Views
    def get_views(self) -> list[ViewConfig]:
        """Returns the internal views-dictionary containing all information
        on the views.

        Returns
        -------
        list[ViewConfig]
        """
        return self.views

    #   Tasks
    def get_tasks(self) -> list[TaskConfig]:
        """Returns the internal tasks-dictionary containing all information
        on the tasks.

        Returns
        -------
        list[TaskConfig]
        """
        tasks = []
        for view in self.views:
            tasks += view.get_tasks()
        return tasks

    def get_task_count(self) -> int:
        """Returns the number of tasks in the test-instrument, i.e. the number
        of columns in a `.get_scr`-resulting dataframe

        Returns
        -------
        int
        """
        return len(self.get_tasks())

    def get_task_names(self) -> list[str]:
        """Returns the name of the tasks as a list of strings.

        Returns
        -------
        list[str]
        """
        return [task.name for task in self.get_tasks()]

    def get_text_task_count(self) -> int:
        """Returns the count of text-tasks in the questionnaire.

        Returns
        -------
        int
        """
        return len(self.get_text_tasks())

    def get_text_task_names(self) -> list[str]:
        """Returns the name of the tasks as a list of strings.

        Returns
        -------
        list[str]
        """
        return [t_task.name for t_task in self.get_text_tasks()]

    def get_mc_task_count(self) -> int:
        """Returns the count of multuple-choice-tasks in the questionnaire.

        Returns
        -------
        int
        """
        return len(self.get_mc_tasks())

    def get_mc_tasks(self) -> list[MCTaskConfig]:
        """Returns the task-configuration information on the multiple-choice
        tasks as a dict.

        Returns
        -------
        list[MCTaskConfig]
        """
        mc_tasks = [
            task for task in self.get_tasks()
            if task.type=="multiple_choice"
        ]
        return mc_tasks

    def get_mc_task_names(self) -> list[str]:
        """Returns the task names of the multiple-choice tasks.

        Returns
        -------
        list[str]
        """
        return [mc_task.name for mc_task in self.get_mc_tasks()]

    def get_text_tasks(self) -> list[TextTaskConfig]:
        """Returns the task-configuration information on the text
        tasks as a dict.

        Returns
        -------
        list[TextTaskConfig]
        """
        text_tasks = [
            text_task for text_task in self.get_tasks()
            if text_task.type=="text"
        ]
        return text_tasks

    def get_max_scores(self, mc_itemcount: bool=False) -> dict[int|float]:
        """Returns the maximum score that can be achieved in a task as a dict.

        Parameters
        ----------
        mc_itemcount : bool
            Whether for the multiple-choice tasks, the number of items per task
            should be returned as the maximum score.

        Returns
        -------
        dict[int|float]
        """
        max_scores = {}
        for task in self.get_tasks():
            if task.type=="multiple_choice":
                if mc_itemcount:
                    max_scores[task.name] = len(task.items)
                    continue
            max_scores[task.name] = task.max_score
        return max_scores

    def get_max_score(self, mc_itemcount: bool=False) -> float:
        """Returns the maximum score of the complete test instrument.

        Parameters
        ----------
        mc_itemcount : bool
            Whether for the multiple-choice tasks, the number of items per task
            should be counted as the maximum score.

        Returns
        -------
        float
        """
        max_scores = self.get_max_scores(mc_itemcount=mc_itemcount)
        max_scores = np.sum(list(max_scores.values()))
        return max_scores


    #   Items
    def get_items(self) -> list[ItemConfig]:
        """Returns the internal items-dictionary containing all information
        on the items.

        Returns
        -------
        list[ItemConfig]
        """
        items = []
        for task in self.get_tasks():
            items += task.items
        return items

    def get_item_names(self) -> list[str]:
        """Returns the name of the items as a list of strings.

        Returns
        -------
        list[str]
        """
        return [item.name for item in self.get_items()]

    def get_text_items(self) -> list[str]:
        """Returns a list of all text items in the questionnaire.

        Returns
        -------
        list[str]
        """
        text_items = [
            item for task in self.get_text_tasks()
            for item in task.items
        ]
        return text_items

    def get_mc_item_names(self) -> list[str]:
        """Returns a list of all multiple-choice items in the questionnaire.

        Returns
        -------
        list[str]
        """
        mc_items = [
            item_name for task in self.get_mc_tasks()
            for item_name in task.get_item_names()
        ]
        return mc_items

    def get_item_counts_per_task(self) -> dict[str, int]:
        """Returns the number of items per task as a dictionary.

        Returns
        -------
        dict[str, int]
        """
        return {task.name: len(task.get_items()) for task in self.get_tasks()}


    ## Dataframe-Validation
    def _validate_id_col(self, df: pd.DataFrame, err_str: str=None):
        if err_str is None:
            err_str = "The passed tabular data does not contain the set id-column."
        if self.id_col not in df.columns:
            raise QuDataInvalidError(err_str)

    def _check_id_matches(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        names: list[str]=["score", "text"],
        mode: Literal["info", "warn", "error"]="info",
    ):
        check_options(
            arg=mode,
            valid_opts=["info", "warn", "error"],
            Ex=ValueError,
            arg_name="mode",
        )
        ids_scr = df1[self.id_col].to_list()
        ids_txt = df2[self.id_col].to_list()
        names_rev = names[::-1]

        def check_mismatch(ids1: list[str], ids2: list[str], names: list[str]) -> None:
            mismatch = np.setdiff1d(ids1, ids2)
            if len(mismatch) > 0:
                s = (
                    f"There are {len(mismatch)} IDs present in the {names[0]}-data,\n" +
                    f"that are not contained in the {names[1]}-data:\n{list(mismatch)}"
                )
                if mode == "error":
                    raise QuDataInvalidError(s)
                elif mode == "info":
                    print("Info: " + s)
                elif mode == "warn":
                    print("Warning: " + s)

        check_mismatch(ids1=ids_scr, ids2=ids_txt, names=names)
        check_mismatch(ids1=ids_txt, ids2=ids_scr, names=names_rev)

    def validate_wide_df_scr(
        self,
        df_scr: pd.DataFrame,
        units: Literal["mc_items", "mc_tasks", "no_mc"],
        all_units: bool,
        mc_only: bool=False,
        validate_scores: Literal["none", "warn", "error"]="none",
        id_err_str: str=None,
    ) -> None:
        """Validates wide-format (id-column + unit-columns) dataframes that
        (should) contain valid score-data to the questionnaire.

        Parameters
        ----------
        df_scr : pd.DataFrame
            The dataframe containing the correct id column and a column for each
            unit (task or item).
        units : Literal["items", "tasks"]
            The units that should be considered
        all_units : bool
            Whether all unit-columns must occur in `df`.
        mc_only : bool
            Whether only the multiple choice tasks should be considered. This
            is typically used, if some unscored QuData is set up, i.e., data
            with text tasks without scores.
        validate_scores : Literal["none", "warn", "error"]
            Validation of actual scores.
        id_err_str : str
            The string to be printed if the id column is missing.
        """
        self._validate_id_col(df=df_scr, err_str=id_err_str)

        if all_units:
            scr_units_config = self.get_scores_columns(units, mc_only=mc_only)
            scr_units_df = df_scr.drop(columns=self.id_col).columns.to_list()
            mismatch = np.setdiff1d(scr_units_df, scr_units_config)

            if len(mismatch) > 0:
                raise QuDataInvalidError(
                    "\nThere are additional units/columns (apart from the id-column) in the scores-data,\n" +
                    f"that are not present in the `QuestionnaireConfig`: \n{mismatch}.\n" +
                    f"Did you set the `mc_score_col_type` correctly (currently \"{units}\")?)"
                )

            mismatch = np.setdiff1d(scr_units_config, scr_units_df)
            if len(mismatch) > 0:
                raise QuDataInvalidError(
                    "\nThere units/columns missing in the in the scores-data, \n" +
                    f"that are present in the `QuestionnaireConfig`:\n{mismatch}.\n" +
                    f"Did you set the `mc_score_col_type` correctly (currently \"{units}\")?)"
                )

        if validate_scores != "none":
            text_tasks = self.get_text_tasks()
            max_reached = df_scr.drop(columns=self.id_col).max(axis=0).to_dict()
            for task in text_tasks:
                if task.name not in max_reached:
                    continue
                if max_reached[task.name] > task.max_score:
                    if validate_scores == "error":
                        raise QuDataInvalidError(
                            f"\nSome entries exceed the maximum possible score: Task {task.name}"
                        )
                    if validate_scores == "warn":
                        print (
                            f"Warning: \nSome entries exceed the maximum possible score: Task {task.name}"
                        )

    def validate_long_df_scr(
        self,
        df_scr: pd.DataFrame,
        value_col: str,
        name_col: str,
        units: Literal["mc_items", "mc_tasks", "no_mc"],
        all_units: bool,
        validate_scores: Literal["none", "warn", "error"]="none",
    ) -> None:
        """Validates wide-format (id-column + unit-columns) dataframes that
        (should) contain valid score-data to the questionnaire.

        Parameters
        ----------
        df_scr : pd.DataFrame
            The dataframe containing the correct id column and a column for each
            unit (task or item).
        value_col : str
            Name of the value column
        name_col : str
            Name of the unit-names column
        units : Literal["items", "tasks"]
            The units that should be considered
        all_units : bool
            Whether all unit-columns must occur in `df`.
        validate_scores : Literal["none", "warn", "error"]
            How the scores should be validated.
        """
        self._validate_id_col(df_scr)
        df_scr = pivot_to_wide(
            df=df_scr,
            value_cols=value_col,
            index_cols=self.id_col,
            column_names=name_col,
        )
        self.validate_wide_df_scr(
            df_scr=df_scr,
            units=units,
            all_units=all_units,
            validate_scores=validate_scores,
        )


    def validate_wide_df_txt(
        self,
        df_txt: pd.DataFrame,
        units: Literal["items", "tasks"],
        all_units: bool,
    ) -> None:
        if self.id_col not in df_txt.columns:
            raise QuDataInvalidError(
                "The passed text-data does not contain the set id-column."
            )

        text_cols_config = self.get_text_columns(units)
        text_cols_df = df_txt.drop(columns=self.id_col).columns.to_list()

        mismatch = np.setdiff1d(text_cols_df, text_cols_config)
        if all_units and len(mismatch) > 0:
            raise QuDataInvalidError(
                "\nThere are additional columns (apart from the id-column) in the text-data,\n" +
                f"that are not present in the `QuestionnaireConfig`: \n{mismatch}.\n" +
                f"Did you set the `text_col_type` correctly (currently \"{units}\")?)"
            )

        mismatch = np.setdiff1d(text_cols_config, text_cols_df)
        if len(mismatch) > 0:
            raise QuDataInvalidError(
                "\nThere are additional columns (apart from the id-column) in the\n" +
                f"`QuestionnaireConfig`, that are missing in the text-data: \n{mismatch}.\n" +
                f"Did you set the `text_col_type` correctly (currently \"{units}\")?)"
            )

    def validate_long_df_txt(
        self,
        df_txt: pd.DataFrame,
        value_col: str,
        name_col: str,
        units: Literal["items", "tasks"],
        all_units: bool,
    ) -> None:
        self._validate_id_col(df_txt)
        df_txt = pivot_to_wide(
            df=df_txt,
            value_cols=value_col,
            index_cols=self.id_col,
            column_names=name_col,
        )
        self.validate_wide_df_txt(
            df_scr=df_txt,
            units=units,
            all_units=all_units,
        )


    ## Altering
    def omit_tasks(self, omit_tasks: list|Literal["text", "multiple_choice"]) -> tuple[list, list, list]:
        """Drops some tasks from the config.

        Parameters
        ----------
        omit_tasks : list|Literal["text", "multiple_choice"]
        """
        if not isinstance(omit_tasks, str):
            surplus_tasks = [t for t in omit_tasks if t not in self.get_task_names()]
            if len(surplus_tasks) > 0:
                raise QuConfigError(
                    f"There are tasks to be dropped, that are not available in the config:\n{surplus_tasks}"
                )
        task_omit_list = []
        view_omit_list = []
        item_omit_list = []
        for view in self.views:
            iol_, tol_ = view._omit_tasks(omit_tasks)
            task_omit_list += tol_
            item_omit_list += iol_
            if len(view.tasks) == 0:
                view_omit_list.append(view.name)

        new_viewlist = []
        for view in self.views:
            if view.name not in view_omit_list:
                new_viewlist.append(view)
        self.views = new_viewlist

        print(
            f"Info: Dropped the following tasks from the config/data:\n{task_omit_list}" +
            f"\nwhich resulted in the following empty views being dropped:\n{view_omit_list}"
        )
        self.task_omit_list = task_omit_list
        self.item_omit_list = item_omit_list

        return view_omit_list, task_omit_list, item_omit_list
