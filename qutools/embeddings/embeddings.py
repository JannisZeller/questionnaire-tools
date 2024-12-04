"""
# Embeddings

The `QuEmbeddings` class is a wrapper around `QuestionnaireData` objects. It
loads or generates embeddings for the observed responses (text-data). It makes
use of a passed `EmbeddingModel` generate the embeddings.
"""


import numpy as np
import pandas as pd

from json import dump, load
from typing import Literal
from pathlib import Path
from shutil import rmtree

from ..core.io import read_data, write_data

from ..data.data import QuData
from ..clustering.clusters import QuScoreClusters
from ..core.io import path_suffix_warning

from .embedding_models import (
    EmbeddingModel,
    OpenAIEmbdModel,
    SentenceTransformersEmbdModel,
)


class QuEmbeddingsError(Exception):
    """An Exception-class for errors in the context of `QuestionnaireEmbeddings`
    objects.
    """
    pass


class QuEmbeddings:
    def __init__(
        self,
        qudata: QuData,
        embedding_model: EmbeddingModel=None,
        embedding_data: str|pd.DataFrame=None,
        save_dir: str=None,
        quclst: QuScoreClusters=None,
        units: Literal["items", "tasks", "persons"]="tasks",
        item_task_prefix: bool=True,
        it_repl_dict: dict[str, str]={"A": "Aufgabe "},
        verbose: bool=True,
        **kwargs
    ) -> None:
        """Wraps `QuestionnaireData` in an embeddings-layer. This layer
        loads or generates embeddings for task-wise concatenated responses
        (text-data).

        Parameters
        ----------
        qudata : QuestionnaireData
            The questionnaire data.
        embedding_model : EmbeddingModel
            The embedding model. If only a string is passed, `embedding_data` must be a path
            to a suiable previously saved embedding-data table.
        embedding_data : str|pd.DataFrame
            Finished embedding-data as a dataframe or path to a tabular datafile.
            The IDs and unit-columns are validated against `qudata`.
        save_dir : str
            A path to save the embedding-data. If `None`, the data does not
            get saved and must be perhaps recomputed later on. It is therefore
            highly suggested to store newly created embedding-data tables.
        quclst : QuScoreClusters
            The cluster object to be used and potentially stored along with the
            embeddings. If not passed, the embeddings will be created without
            information on clusters. The clusters can also be added later on.
            They are typically used, if the goal is to predict cluster-assignments
            based on the embeddings.
        units : Literal["items", "tasks", "persons"]
            The units to embed by. See `QuestionnaireData.get_txt` for details.
        item_task_prefix : bool
            Wether the single items should be prefixed with a certrain string
            (using the `it_repl_dict`).
        it_repl_dict : dict[str, str]
            The dictionary to replace patterns of the task name in the item or
            task prefixing.
        verbose : bool
            Verbosity of the processing.

        Raises
        ------
        QuestionnaireEmbeddingsError
        """
        self.qudata = qudata
        self.id_col = qudata.id_col
        self.units = units
        self.info_cols = [qudata.id_col]
        self.embd_model_name = self._set_ebd_model_name(embedding_model=embedding_model)
        self.item_task_prefix = item_task_prefix
        self.it_repl_dict = it_repl_dict
        self.units = units

        if units != "persons":
            self.unit_col = units[:-1]
            self.info_cols.append(self.unit_col)

        if quclst is not None:
            if not quclst.quconfig == qudata.quconfig:
                raise QuEmbeddingsError(
                    "The configurations of the passed data and scores object are not identical."
                )
        self.quclst = quclst

        self.edb_model = embedding_model

        if embedding_data is not None and embedding_model is None:
            if isinstance(embedding_data, str) or isinstance(embedding_data, Path):
                if not Path(embedding_data).exists():
                    raise QuEmbeddingsError(
                        "If `embedding_model` is a only string, `embedding_data` must be a path to a table with the\n " +
                        "task-wise embedded data with an id-column with 1:1 matched to the ids in the\n" +
                        "`qudata`. The passed path does not exists."
                    )
                df_txt = self.qudata.get_txt(units, "long")
                df_ebd = read_data(embedding_data)
                self.__check_id_matches(df_ebd, df_txt)
                self.df_ebd = df_ebd
            elif isinstance(embedding_data, pd.DataFrame):
                df_txt = self.qudata.get_txt(units, "long")
                df_ebd = embedding_data
                self.__check_id_matches(df_ebd, df_txt)
                self.df_ebd = df_ebd

        elif embedding_data is None and embedding_model is not None:
            self.df_ebd = embedding_model.get_embeddings(
                qudata=qudata,
                units=units,
                item_task_prefix=item_task_prefix,
                it_repl_dict=it_repl_dict,
                verbose=verbose,
                **kwargs,
            )
        else:
            raise QuEmbeddingsError(
                "You need to set either the `embedding_data` or the `embedding_model` argument."
            )

        if save_dir is not None:
            self.to_dir(save_dir=save_dir)


    def _set_ebd_model_name(self, embedding_model: EmbeddingModel=None) -> None:
        if embedding_model is None:
            self.ebd_model_name = "None"
        elif isinstance(embedding_model, OpenAIEmbdModel):
            self.ebd_model_name = "openai"
        elif isinstance(embedding_model, SentenceTransformersEmbdModel):
            self.ebd_model_name = "sentence-transformers"
        else:
            self.ebd_model_name = "custom"


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
        self.qudata.filter_ids(contains=contains, exclude=exclude, regex=regex)
        msk = self.df_ebd[self.id_col].str.contains(contains, regex=regex)
        msk = msk if not exclude else ~msk
        self.df_ebd = self.df_ebd[msk].reset_index(drop=True)


    def get_ebds(
        self,
        with_scores: bool=True,
        fillna_scores: bool=True,
        with_clusters: bool=False,
    ) -> pd.DataFrame:
        """Returns the embedding-data.

        Parameters
        ----------
        with_scores : bool
            Wether a "score" or "total_score" column based on the contained
            QuestionnaireData object should be appended.
        fillna_scores : bool
            Wether the scores should be na-filled with 0s. This should be true,
            if missing scores can be assumed to be probably missing 0s.

        Returns
        -------
        pd.DataFrame
        """
        df_ret = self.df_ebd.copy()

        if not with_scores:
            return df_ret

        id_col = self.qudata.id_col
        unit_col = self.unit_col
        df_scr = self.qudata.get_scr(verbose=False)

        if self.units == "persons":
            df_scr["total_score"] = df_scr.drop(columns=id_col).fillna(0).sum(axis=0)
            df_scr = df_scr[[id_col, "total_score"]]
            df_ret = pd.merge(df_ret, df_scr, on=id_col)

        else:
            df_scr = pd.melt(df_scr, id_vars=id_col, value_name="score", var_name=unit_col)
            df_ret = pd.merge(df_ret, df_scr, on=[id_col, unit_col])

            if fillna_scores:
                df_ret["score"] = df_ret["score"].fillna(0)

        if with_clusters:
            df_ret = self._append_clusters(df_ret)

        return df_ret

    def get_embedding_dimension(self):
        """Calculates the dimension of the embeddings. This is the number of columns
        in the embeddings-data minus the number of info-columns (ID and optionally unit).

        Returns
        -------
        int
            The dimension of the embeddings.
        """
        n_info = 1
        if self.units != "persons":
            n_info += 1
        return self.df_ebd.shape[1] - n_info

    def get_n_classes(self) -> int:
        """Returns the number of classes in the QuestionnaireData object.

        Returns
        -------
        int
            The number of classes. Mostly, this is the number of distinct scores
            (e.g., 0, 1, 2) the questionnaire-tasks can have.
        """
        return self.qudata.get_n_classes()


    def _internal_cls_model_available(self) -> bool:
        return self.quclst is not None


    def _append_clusters(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        id_col = self.qudata.id_col
        if self.quclst is None:
            raise QuEmbeddingsError(
                "Can not return embeddings with clusters, if no QuestionnaireScoreCluster \n"+
                "instance was passed at initialization."
            )
        try:
            df_cls = self.quclst.clusters_most(self.qudata)
        except:
            print(
                "Can not return a cluster for every instance because of the chosen \n" +
                "cluster-algorithm and dropout settings. Returning a cluster only for \n" +
                "the instances used at cluster generation."
            )
            df_cls = self.quclst.clusters()

        df_cls = df_cls[[id_col, "cluster"]]
        df = pd.merge(df, df_cls, on=id_col, how="left")
        return df


    def __check_id_matches(self, df_ebd: pd.DataFrame, df_txt: pd.DataFrame):
        if self.id_col not in df_ebd.columns:
            raise QuEmbeddingsError(
                f"The id column `\"{self.id_col}\"` specified in the passed questionnaire data\n" +
                "must be contained in the passed embeddings data."
            )
        ids_txt = df_txt[self.id_col].to_list()
        ids_ebd = df_ebd[self.id_col].to_list()

        if hasattr(self, "unit_col"):
            if self.unit_col not in df_ebd.columns:
                raise QuEmbeddingsError(
                    f"The unit-column `\"{self.unit_col}\"` must be contained in the passed embeddings data."
                )

            tasklist_txt = df_txt[self.unit_col].to_list()
            x_txt = list(zip(ids_txt, tasklist_txt))

            tasklist_ebd = df_ebd[self.unit_col].to_list()
            x_ebd = list(zip(ids_ebd, tasklist_ebd))

            mismatch = np.setdiff1d(x_txt, x_ebd)
            if len(mismatch) > 0:
                raise QuEmbeddingsError(
                    f"Info: There are {len(mismatch)} ID-Task combinations present in the text-data,\n" +
                    f"that are not contained in the embeddings-data:\n{list(mismatch)}"
                )
            mismatch = np.setdiff1d(x_ebd, x_txt)
            if len(mismatch) > 0:
                raise QuEmbeddingsError(
                    f"Info: There are {len(mismatch)} ID-Task combinations present in the embeddings-data,\n" +
                    f"that are not contained in the text-data. {list(mismatch)}"
                )

        else:
            if "task" in df_ebd:
                raise QuEmbeddingsError(
                    "You initizalized a QuestionnaireEmbeddings object with the unit-type \"persons\"\n" +
                    "but the embeddings data table contains a \"task\"-column."
                )
            if "item" in df_ebd:
                raise QuEmbeddingsError(
                    "You initizalized a QuestionnaireEmbeddings object with the unit-type \"persons\"\n" +
                    "but the embeddings data table contains a \"item\"-column."
                )
            mismatch = np.setdiff1d(ids_txt, ids_ebd)
            if len(mismatch) > 0:
                raise QuEmbeddingsError(
                    f"Info: There are {len(mismatch)} IDs present in the text-data,\n" +
                    f"that are not contained in the embeddings-data:\n{list(mismatch)}"
                )
            mismatch = np.setdiff1d(ids_ebd, ids_txt)
            if len(mismatch) > 0:
                raise QuEmbeddingsError(
                    f"Info: There are {len(mismatch)} IDs present in the embeddings-data,\n" +
                    f"that are not contained in the text-data. {list(mismatch)}"
                )


    ## IO
    def _save_settings_dict(self, path: str) -> None:
        settings_dict = {
            "ebd_model_name": self.ebd_model_name,
            "item_task_prefix": self.item_task_prefix,
            "it_repl_dict": self.it_repl_dict,
            "units": self.units
        }
        with open(path, "w") as f:
            dump(settings_dict, f, indent=2)

    def to_dir(self, save_dir: str) -> None:
        """Exports the QuestionnaireEmbeddings object to a directory. The
        QuestionnaireData object is also saved in the directory. The embeddings
        are saved as a gzip-compressed table. The settings-dictionary is saved
        as a json-file.

        Parameters
        ----------
        save_dir : str
            The path to the directory to save the data to.
        """
        dir_ = Path(save_dir)
        path_suffix_warning(suffix=dir_.suffix)
        try:
            dir_.mkdir(parents=True)
        except FileExistsError:
            rmtree(dir_)
            dir_.mkdir(parents=True)
        self.qudata.to_dir(dir_ / "qudata")
        write_data(self.df_ebd, dir_ / "embeddings.gzip")
        self._save_settings_dict(dir_ / "settings_dict.json")


    @staticmethod
    def _load_settings_dict(path: str) -> dict:
        with open(path, "r") as f:
            settings_dict: dict = load(f)
        s = f"Found the following embedding-data-settings: "
        for k, v in settings_dict.items():
            s += f"\n - ({k}: {v}) "
        print(s)
        return settings_dict

    @staticmethod
    def from_dir(load_dir: str, quclst: QuScoreClusters=None) -> "QuEmbeddings":
        """Loads a QuestionnaireEmbeddings object from a directory.

        Parameters
        ----------
        load_dir : str
            The path to the directory to load the data from.
        quclst : QuScoreClusters
            The clusters to load. If not passed, the QuestionnaireEmbeddings object
            will be created without clusters.
        """
        dir_ = Path(load_dir)
        qudata = QuData.from_dir(dir_ / "qudata")
        settings_dict = QuEmbeddings._load_settings_dict(dir_ / "settings_dict.json")

        quebds = QuEmbeddings(
            qudata=qudata,
            embedding_data=dir_ / "embeddings.gzip",
            quclst=quclst,
            **settings_dict,
        )

        return quebds
