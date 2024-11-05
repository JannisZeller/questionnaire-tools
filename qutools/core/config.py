"""
# Submodule for the test-instrument specification logic

This submodule provides the building blocks for the core of the test-instrument specification logic.
It is not designed to be used standalone but rather as a helper for the other submodules,
which is why there will be no examples here.
It provides the building blocks are used and explained in more detail in [qutools.data.config][qutools.data.config]-submodule.
"""

from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import Literal

from .validation import check_key, check_options, check_type, check_uniqueness



## Item Configurations
class ItemConfigError(Exception):
    """An exception for Item-Configs."""


@dataclass
class ItemConfig(ABC):
    """A class for item configuration as the are expected in yaml config files.

    Parameters
    ----------
    name : str
        The name of the item.
    """
    name: str


@dataclass
class TextItemConfig(ItemConfig):
    """A class for text-item configuration as the are expected in yaml config files.

    Parameters
    ----------
    name : str
        The name of the item.
    """
    @staticmethod
    def from_name(name: str) -> "TextItemConfig":
        """
        Generates a text-item configuration from the item name. Basically just
        a wrapper for the constructor meant to provide an analogous interface
        as the `.from_dict` method of the `MCItemConfig`.

        Parameters
        ----------
        name : str
            The items name.

        Returns
        -------
        TextItemConfig
            The text-item configuration.

        """
        instance = TextItemConfig(name=name)
        return instance



@dataclass
class MCItemConfig(ItemConfig):
    """A class for text-item configuration as the are expected in yaml config files.
    An mc-item is a single response to one question of a set of mc-questions.
    The set is typically a task (see below).

    Parameters
    ----------
    name : str
        The name of the item.
    correct_response: int
        The correct response, i. e. 1 (true / option 1) or 0 (false / option 2).

    Returns
    -------
    MCItemConfig
    """
    correct_response: int

    @staticmethod
    def from_dict(name: str, dct: dict) -> "MCItemConfig":
        """Constructing the mc-item config from a dict as recieved from config yamls.
        Validates the existance of a the "name" and "correct_response" keys and
        asserts that the correct response is 0 or 1.

        Parameters
        ----------
        name : str
            The name of the item.
        dct: dict
            Dictionary containing the "correct_response" key.

        Returns
        -------
        MCItemConfig
        """
        check_key(
            key="correct_response",
            dct=dct,
            Ex=ItemConfigError,
            ex_str=f"The (for mc-items) mandatory `\"correct_response\"` key is missing in the passed config-dict: \n{dct}",
        )
        item_name = name
        correct_response = dct["correct_response"]
        if correct_response not in {0, 1}:
            raise ItemConfigError(
                f"The `correct_response` of item {item_name} must be in" + "{0, 1}."
            )
        instance = MCItemConfig(name=item_name, correct_response=correct_response)
        return instance

    def to_dict(self) -> dict:
        """Exports the mc-item config to a dictionary.

        Returns
        -------
        dict
            The dictionary `{"correct_response": self.correct_response}`.
        """
        return {"correct_response": self.correct_response}




## Task Configurations
class TaskConfigError(Exception):
    """An exception for Task-Configs."""


@dataclass
class TaskConfig(ABC):
    """Class for questionnaire-tasks. A task is a single scoreable unit of the
    questionnaire. A scoreable unit might consist of multiple items, that are
    only scored together. There is built in capability of handling multiple-choice
    items as tasks together. If this is not wanted, configure the questionnaire
    such that each mc-item is a single task containing just one item.
    Currently only threshold-scoring and summation-scoring is implemented.
    """
    name: str
    type: Literal["text", "multiple_choice"]
    max_score: int|float
    items: list[ItemConfig]


    def __init__(
        self,
        task_name: str,
        task_type: str,
        max_score: int|float,
        items: list[ItemConfig],
    ):
        """Sets up the task and validates the arguments.

        Parameters
        ----------
        task_name : str
            Name of the task
        task_type : Literal["text", "multiple_choice"]
            Type of the task
        max_score : int | float
            The maximum score that can be achieved in the task. In case of mc-tasks
            threshold or summation scoring will yield maximum scores of this value.
        items : list[ItemConfig]
            The items contained in the task.
        """
        check_options(
            arg=task_type,
            valid_opts=["text", "multiple_choice"],
            Ex=TaskConfigError,
            arg_name="task_type",
        )
        check_type(
            arg=max_score,
            type_=int|float,
            Ex=TaskConfigError,
            arg_name="max_score",
        )
        if max_score < 0:
            raise TaskConfigError(f"For task {task_name}: A `max_score` can not be negative. Got {max_score}")

        self.name = task_name
        self.type = task_type
        self.max_score = max_score
        self.items = items

        check_uniqueness(
            arr=self.get_item_names(),
            Ex=TaskConfigError,
            ex_str=f"Some items of task {task_name} are not unique.",
        )


    @staticmethod
    def from_dict(name: str, dct: dict) -> "TaskConfig":
        """Instantiates a task config from a dictionary. Inferes the task-
        and item-type from the parameters and instantiates the correct
        subclass accordingly.

        Parameters
        ----------
        name : str
            Name of the task.
        dct : dict
            Task information (see constructor) in a dict.

        Returns
        -------
        Self
        """
        def key_missing_str(key: str) -> str:
            return f"The mandatory \"{key}\" key is missing in the passed config-dict. Got as config-dict: \n{dct}"
        check_key(
            key="task_type",
            dct=dct,
            Ex=TaskConfigError,
            ex_str=key_missing_str("task_type"),
        )
        task_type = dct["task_type"]
        check_options(
            arg=task_type,
            valid_opts=["text", "multiple_choice"],
            Ex=TaskConfigError,
            arg_name="task_type",
        )

        check_key(
            key="max_score",
            dct=dct,
            Ex=TaskConfigError,
            ex_str=key_missing_str("max_score"),
        )
        max_score = dct["max_score"]

        check_key(
            key="items",
            dct=dct,
            Ex=TaskConfigError,
            ex_str=key_missing_str("items"),
        )
        items_dct: dict = dct["items"]

        items = []

        if task_type == "text":
            check_type(
                arg=items_dct,
                type_=list,
                Ex=TaskConfigError,
                ex_str=f"The data structure under the `\"items\"` key must be a list if the task ({name}) is a text-task.",
            )
            for item_name in items_dct:
                check_type(
                    arg=item_name,
                    type_=str,
                    Ex=TaskConfigError,
                    ex_str=f"The item-config for an item {item_name} of a text-task must be a single string. Got:\n {item_name}",
                )
                item = TextItemConfig.from_name(item_name)
                items.append(item)
        elif task_type == "multiple_choice":
            check_type(
                arg=items_dct,
                type_=dict,
                Ex=TaskConfigError,
                ex_str=f"The data structure under the `\"items\"` key must be a dict if the task ({name}) is an mc-task."
            )
            for item_name, item_cfg in items_dct.items():
                check_type(
                    arg=item_cfg,
                    type_=dict,
                    Ex=TaskConfigError,
                    ex_str=f"The item-config for an item {item_name} of a mc-task must be a dictionary. Got:\n {item_cfg}",
                )
                item = MCItemConfig.from_dict(item_name, item_cfg)
                items.append(item)

        if task_type == "text":
            instance = TextTaskConfig(
                task_name=name,
                max_score=max_score,
                items=items,
            )
        elif task_type == "multiple_choice":
            check_key(
                key="scoring",
                dct=dct,
                Ex=None,
                ex_str=f"Without `scoring` set for taks {name} in the task config, it defaults to `\"summation\"`.",
            )
            scoring = dct.get("scoring", "summation")

            instance = MCTaskConfig(
                task_name=name,
                max_score=max_score,
                items=items,
                scoring=scoring,
            )

        return instance

    def get_item_names(self) -> list[str]:
        """Returns the itemnames in the task as a list.

        Returns
        -------
        list[str]
        """
        return [item.name for item in self.items]

    @abstractmethod
    def get_items(self) -> dict:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


@dataclass
class TextTaskConfig(TaskConfig):
    """A class for text-task configuration as they are expected in the yaml files.
    """
    items: list[TextItemConfig]

    def __init__(
        self,
        task_name: str,
        max_score: int|float,
        items: list[TextItemConfig],
    ):
        """Constructs a text-task config. See `TaskConfig` constructor for references.
        """
        super().__init__(
            task_name=task_name,
            task_type="text",
            max_score=max_score,
            items=items,
        )

    def get_items(self) -> list[TextItemConfig]:
        """Returns the dictionary of items in the task.

        Returns
        -------
        list[TextItemConfig]
        """
        return self.items

    def to_dict(self) -> dict:
        """Returns the information contained as a dict.

        Returns
        -------
        dict
            ```python
            { "task_type": self.type,
              "max_score": self.max_score,
              "items": [it.name for it in self.items], }
            ```
        """
        return {
            "task_type": self.type,
            "max_score": self.max_score,
            "items": [it.name for it in self.items],
        }


@dataclass
class MCTaskConfig(TaskConfig):
    """A class for multiple-choice task configuration as they are expected in the yaml files.
    """
    scoring: Literal["thresholds", "summation"]
    items: list[MCItemConfig]

    def __init__(
        self,
        task_name: str,
        max_score: int|float,
        items: list[MCItemConfig],
        scoring: Literal["thresholds", "summation"],
    ):
        """Constructs an mc-task config.
        """
        super().__init__(
            task_name=task_name,
            task_type="multiple_choice",
            max_score=max_score,
            items=items,
        )
        check_options(
            arg=scoring,
            valid_opts=["summation", "thresholds"],
            Ex=TaskConfigError,
            arg_name="scoring",
        )
        self.scoring = scoring

    def get_items(self) -> list[MCItemConfig]:
        """Returns the dictionary of items in the task.

        Returns
        -------
        list[MCItemConfig]
        """
        return self.items

    def to_dict(self) -> dict:
        """Returns the contained intormation as a dict.

        Returns
        -------
        dict
            ```python
            { "task_type": self.type,
              "max_score": self.max_score,
              "scoring": self.scoring,
              "items": {item.name: item.to_dict() for item in self.items}, }
            ```
        """
        return {
            "task_type": self.type,
            "max_score": self.max_score,
            "scoring": self.scoring,
            "items": {item.name: item.to_dict() for item in self.items},
        }



## View Configurations
class ViewConfigError(Exception):
    """An exception class for ViewConfig-Objects."""

@dataclass
class ViewConfig:
    """Class for questionnaire-views. A view is a single "page" of the
    questionnaire, i.e., a set of tasks, to be presented as one page in a
    digital presentation or a set of task with a shared part, like an image-
    or text-element. A view often (but not always) consists of just a single
    task. The usage of views is optional."""
    name: str
    tasks: list[TaskConfig]

    def __init__(self, view_name: str, tasks: list[TaskConfig]) -> None:
        """Initializes a view-config.

        Parameters
        ----------
        view_name : str
            Name of the view.
        tasks : list[TaskConfig]
            Tasks in the view.
        """
        self.name = view_name
        self.tasks = tasks
        check_uniqueness(
            arr=self.get_item_names(),
            Ex=ViewConfigError,
            ex_str=f"Some items of view {view_name} are not unique.",
        )
        check_uniqueness(
            arr=self.get_task_names(),
            Ex=ViewConfigError,
            ex_str=f"Some tasks of view {view_name} are not unique.",
        )


    @staticmethod
    def from_dict(name: str, dct: dict) -> "ViewConfig":
        """Constructs a view from a dict. Validates `dct`'s structure and
        value options.

        Parameters
        ----------
        name : str
            Name of the view
        dct : dict
            Dictionary containing task configs.

        Returns
        -------
        Self
        """
        check_type(
            arg=dct,
            type_=dict,
            Ex=ViewConfigError,
            ex_str=f"Passed `dct` must be a dictionary.",
        )
        check_key(
            key="tasks",
            dct=dct,
            Ex=ViewConfigError,
            ex_str=f"The mandatory \"tasks\" key is missing in the passed task-config-dict: \n{dct}",
        )

        tasks_dct: dict = dct["tasks"]
        check_type(
            arg=tasks_dct,
            type_=dict,
            Ex=ViewConfigError,
            ex_str=f"The data structure under the \"tasks\" key must be a dictionary again.",
        )

        tasks = []
        for task_name, task_cfg in tasks_dct.items():
            task = TaskConfig.from_dict(task_name, task_cfg)
            tasks.append(task)

        return ViewConfig(view_name=name, tasks=tasks)

    def to_dict(self) -> dict:
        """Returns the contained information as a dict.

        Returns
        -------
        dict
            ```python
            {"tasks": {
                task.name: task.to_dict() for task in self.tasks
            }}
            ```
        """
        return {"tasks": {
            task.name: task.to_dict() for task in self.tasks
        }}


    def get_tasks(self) -> list[TaskConfig]:
        """Returns the contained tasks as a dict.

        Returns
        -------
        list[TaskConfig]
        """
        return self.tasks

    def get_task_names(self):
        """Returns the tasknames in the view as a list.

        Returns
        -------
        list[str]
        """
        return [task.name for task in self.tasks]

    def get_items(self) -> dict[str, ItemConfig]:
        """Returns the dictionary all items in the view.

        Returns
        -------
        dict[str, ItemConfig]
        """
        items = {}
        for task in self.tasks:
            for item_name, item_cfg in task.get_items():
                items[item_name] = item_cfg
        return items

    def get_item_names(self) -> list[str]:
        """Returns the names of all items in the view.

        Returns
        -------
        list[str]
        """
        item_names = []
        for task in self.tasks:
            item_names += task.get_item_names()
        return item_names

    def _omit_tasks(self, omit_tasks: list|Literal["text", "multiple_choice"]) -> tuple[list, list]:
        """Drops some tasks from the config. Either the task names has to be
        explicitly passed as a list or alls text- / alls mc-tasks can be
        dropped by specifying the corresponding literal option.

        Parameters
        ----------
        omit_tasks : list|Literal["text", "multiple_choice"]

        Returns
        -------
        item_omit_list, task_omit_list : tuple[list, list]
            Returnes the omitted items and omitted tasks as a list, such that
            they can be stored and further used.
        """
        if isinstance(omit_tasks, str):
            if omit_tasks not in ["text", "multiple_choice"]:
                omit_tasks = [omit_tasks]

        task_omit_list = []
        item_omit_list = []
        for task in self.tasks:
            if isinstance(omit_tasks, str):
                if task.type == omit_tasks:
                    task_omit_list.append(task.name)
                    item_omit_list += task.get_item_names()
            else:
                if task.name in omit_tasks:
                    task_omit_list.append(task.name)
                    item_omit_list += task.get_item_names()

        new_tasklist = []
        for task in self.tasks:
            if task.name not in task_omit_list:
                new_tasklist.append(task)
        self.tasks = new_tasklist

        return item_omit_list, task_omit_list
