"""
# Embedding Models

This module contains classes for embedding models. The classes are used to
generate embeddings for the questionnaire data. The embeddings could be used for
various purposes, e.g. for clustering or classification tasks. Mainly two types
of embedding models are implemented:

1. SentenceTransformersEmbdModel: This class uses the sentence-transformers
    package to generate embeddings. The sentence-transformers package provides
    a wide range of open-source pre-trained models that can be used to generate embeddings
    for text data.

2. OpenAIEmbdModel: This class uses the OpenAI API to generate embeddings. The
    OpenAI API provides models that can be used to generate embeddings for text data.
    It is especially useful for generating embeddings for longer texts, that might
    not be handeled properly by the sentence-transformers models. To use this class,
    you need to have an OpenAI API account and access key.

"""

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from typing import Literal
from abc import ABC, abstractmethod

from ..data.data import QuData
from ..core import batched
from ..core.text import set_item_task_prefix, check_seq_lens




def _get_df_txt(
    qudata: QuData,
    max_len: int,
    units: Literal["items", "tasks", "persons"]="tasks",
    item_task_prefix: bool=True,
    it_repl_dict: dict[str, str]=None,
    sep: str=" [SEP] ",
    verbose: bool=True,
) -> tuple[pd.DataFrame, list[str]]:
    info_cols = [qudata.id_col]
    if units != "persons":
        info_cols.append(units[:-1])

    df_txt = qudata.get_txt(
        units=units,
        table="long",
    )

    if item_task_prefix and units != "persons":
        df_txt = set_item_task_prefix(
            df_txt=df_txt,
            unit_col=units[:-1],
            it_repl_dict=it_repl_dict,
            sep=sep,
            verbose=verbose,
        )

    check_seq_lens(df_txt=df_txt, max_len=max_len, verbose=verbose)

    return df_txt, info_cols


class EmbeddingModel(ABC):
    """
    An abstract class for embedding models. The class should be subclassed
    and the `get_embeddings`-method should be implemented.
    """
    @abstractmethod
    def get_embeddings(
        self,
        qudata: QuData,
        units: Literal['items', 'tasks', 'persons'] = "tasks",
        item_task_prefix: bool=True,
        it_repl_dict: dict[str, str]=None,
        sep: str=" [SEP] ",
        verbose: bool=True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        This method should return a pandas DataFrame with the embeddings as
        columns and the IDs and possibly the unit-columns as index-columns.

        Parameters
        ----------
        qudata : QuData
            The questionnaire data.
        units : Literal['items', 'tasks', 'persons']
            The units to embed by. See [QuestionnaireData][qutools.data.data]'s
            `.get_txt`-method for details.
            for details.
        item_task_prefix : bool
            Wether the response texts should be prefixed with a certrain string
            representing or containing some information on the respective item/task.
            (using the `it_repl_dict`).
        it_repl_dict : dict[str, str]
            The dictionary to replace patterns of the item/task name for prefixing.
        sep : str
            The separator string to use for the item/task prefixing.
        verbose : bool
            Verbosity of the processing.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with the embeddings as columns and the IDs and
            possibly the unit-columns as index-columns.
        """
        pass



class SentenceTransformersEmbdModel(EmbeddingModel):
    def __init__(self, model_name: str=None) -> None:
        """Initializes a [sentence-transformers](https://sbert.net/) embedding
        model. Available model names can be found in their documentation. The
        maximum sequence length is set to 512, which [should be suitable](https://sbert.net/examples/applications/computing-embeddings/README.html?highlight=sequence%20length#input-sequence-length)
        for most sentence-transformers models. It can be easily overwritten using
        the `.set_model_max_seq_length`-method, if your own research suggests a
        different length.

        Parameters
        ----------
        model_name : str
            Model name from the sentence_transformers documentation. If a model has
            previously been saved to disk, can also be a filepath.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except:
            raise ModuleNotFoundError(
                "To use the SentenceTransformersEmbdModel, please install the sentence_transformers-package."
            )
        if model_name is None:
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(device)
        print(f"Using {device} device for the sentence-transformers embedding model:\n{model_name}")
        self.model.max_seq_length = 512
        self.model_name = model_name


    def set_model_max_seq_length(self, length: int):
        """Sets the maximum sequence length for the sentence-transformers model.
        Please note that the maximum sequence length should be set to a value that
        is suitable for the model you are using. The default value is 512, which
        [should be suitable](https://sbert.net/examples/applications/computing-embeddings/README.html?highlight=sequence%20length#input-sequence-length)
        for most sentence-transformers models.

        Parameters
        ----------
        length : int
            The maximum sequence length.
        """
        self.model.max_seq_length = length


    def get_embeddings(
        self,
        qudata: QuData,
        units: Literal['items', 'tasks', 'persons'] = "tasks",
        item_task_prefix: bool=True,
        it_repl_dict: dict[str, str]=None,
        sep: str=" [SEP] ",
        verbose: bool=True,
        **kwargs,
    ) -> pd.DataFrame:
        """Returns the embeddings for the questionnaire data. The embeddings are
        generated by the sentence-transformers model.

        Parameters
        ----------
        qudata : QuData
            The questionnaire data.
        units : Literal['items', 'tasks', 'persons']
            The units to embed by. See [QuestionnaireData][qutools.data.data]'s
            `.get_txt`-method for details.
            for details.
        item_task_prefix : bool
            Wether the response texts should be prefixed with a certrain string
            representing or containing some information on the respective item/task.
            (using the `it_repl_dict`).
        it_repl_dict : dict[str, str]
            The dictionary to replace patterns of the item/task name for prefixing.
        sep : str
            The separator string to use for the item/task prefixing.
        verbose : bool
            Verbosity of the processing.
        **kwargs
            Keyword argument: `batch_size` (`int`) for the embedding calculation.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with the embeddings as columns and the IDs and
            possibly the unit-columns as index-columns.

        """
        df_txt, info_cols = _get_df_txt(
            qudata=qudata,
            max_len=self.model.get_max_seq_length(),
            units=units,
            item_task_prefix=item_task_prefix,
            it_repl_dict=it_repl_dict,
            sep=sep,
            verbose=verbose,
        )
        print("Generating Embeddings")
        docs_iter = batched(df_txt, batch_size=kwargs.get("batch_size", 32))
        df_ebd = pd.DataFrame()


        if verbose:
            iterator_ = tqdm(docs_iter, total=len(docs_iter))
        else:
            iterator_ = docs_iter

        for docs_batch in iterator_:
            docs = docs_batch["text"].to_list()
            embeddings = self.model.encode(docs)

            df_ = pd.DataFrame(embeddings)
            df_ = df_.rename(columns=lambda x: "dim-" + str(int(x)+1))

            df_ = pd.concat(
                [docs_batch[info_cols].reset_index(drop=True), df_],
                axis=1
            )

            df_ebd = pd.concat([df_ebd, df_], axis=0)
        return df_ebd

    def save_model(self, path: str) -> None:
        """Saves the used sentence-transformers model to disk.

        Parameters
        ----------
        path : str
            The path to save the model to.
        """
        self.model.save(path=path)



class OpenAIEmbdModel(EmbeddingModel):
    def __init__(
        self,
        open_ai_api_key: str,
        model_name: str="text-embedding-3-small",
    ):
        """Initializes an OpenAI embedding model. The model is used to generate embeddings
        for the questionnaire data. Available models and specs can be found in
        the [OpenAI API documentation](https://platform.openai.com/docs/guides/embeddings).
        To use this class, the `openai`-package and the `backoff`-package must be installed.
        The latter is needed to handle rate limits and similar issues.

        Parameters
        ----------
        open_ai_api_key : str
            The OpenAI API key for your account.
        model_name : str
            The model name from the OpenAI API documentation.
        """
        try:
            from openai import OpenAI, AsyncOpenAI, RateLimitError
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "To use the OpenAIEmbdModel, please install the openai-package (tested with v1.21.2)."
            )
        try:
            import backoff
        except:
            raise ModuleNotFoundError(
                "To use the OpenAIEmbdModel, please install the backoff-package (tested with v2.2.1)."
            )

        import asyncio

        self.async_client = AsyncOpenAI(api_key=open_ai_api_key)
        self.client = OpenAI(api_key=open_ai_api_key)

        def _get_embedding_numbers(embedding) -> np.ndarray: # embedding : openai.types.CreateEmbeddingResponse
            embd_nums = np.array(embedding.data[0].embedding)
            return embd_nums

        @backoff.on_exception(backoff.expo, RateLimitError) # The async option is currently not in use.
        async def _async_embedding_request(
            data: pd.Series,
            id_col: str,
            units: Literal["items", "tasks", "persons"],
        ) -> dict[str, str|list]:
            embedding = await self.async_client.embeddings.create(
                input=data["text"],
                model=model_name,
            )
            embd_nums = _get_embedding_numbers(embedding)
            ret_dct = {id_col: data[id_col], 'embedding': embd_nums}
            if units != "persons":
                ret_dct[units[:-1]] = data[units[:-1]]
            return ret_dct

        async def _gather_batch(
            df_batch: pd.DataFrame,
            id_col: str,
            units: Literal["items", "tasks", "persons"],
        ):
            batch_async_res = await asyncio.gather(*[
                _async_embedding_request(data, id_col, units)
                for _, data in df_batch.iterrows()
            ])
            return batch_async_res

        self._gather_batch = _gather_batch

        @backoff.on_exception(backoff.expo, RateLimitError)
        def _sync_embedding_request(
            data: pd.DataFrame,
            id_col: str,
            units: Literal["items", "tasks", "persons"],
        ) -> dict[str, str|list]:
            embedding = self.client.embeddings.create(
                input=data["text"],
                model=model_name,
            )
            embd_nums = [ebd.embedding for ebd in embedding.data]
            ret_dct = {id_col: data[id_col].to_list(), 'embedding': embd_nums}
            if units != "persons":
                ret_dct[units[:-1]] = data[units[:-1]].to_list()
            return ret_dct
        self._sync_embedding_request = _sync_embedding_request


    def get_embeddings(
        self,
        qudata: QuData,
        units: Literal['items', 'tasks', 'persons'] = "tasks",
        item_task_prefix: bool=True,
        it_repl_dict: dict[str, str]=None,
        sep: str=" [SEP] ",
        verbose: bool=True,
        **kwargs,
    ) -> pd.DataFrame:
        """Returns the embeddings for the questionnaire data. The embeddings are
        generated by the Open AI model.

        Parameters
        ----------
        qudata : QuData
            The questionnaire data.
        units : Literal['items', 'tasks', 'persons']
            The units to embed by. See [QuestionnaireData][qutools.data.data]'s
            `.get_txt`-method for details.
            for details.
        item_task_prefix : bool
            Wether the response texts should be prefixed with a certrain string
            representing or containing some information on the respective item/task.
            (using the `it_repl_dict`).
        it_repl_dict : dict[str, str]
            The dictionary to replace patterns of the item/task name for prefixing.
        sep : str
            The separator string to use for the item/task prefixing.
        verbose : bool
            Verbosity of the processing.
        **kwargs
            Keyword argument: `batch_size` (`int`) for the embedding calculation.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with the embeddings as columns and the IDs and
            possibly the unit-columns as index-columns.

        """

        df_txt, info_cols = _get_df_txt(
            qudata=qudata,
            max_len=8191,
            units=units,
            item_task_prefix=item_task_prefix,
            it_repl_dict=it_repl_dict,
            sep=sep,
            verbose=verbose,
        )

        docs_iter = batched(df_txt, batch_size=kwargs.get("batch_size", 32))

        embeddings = []
        print("Requesting Embeddings")
        for docs_batch in tqdm(docs_iter, total=len(docs_iter)):
            embeddings_batch = self._sync_embedding_request(docs_batch, qudata.id_col, units)
            embeddings.append(embeddings_batch)

        df_ebd = pd.DataFrame()
        embd_dims = np.array(embeddings[0]["embedding"]).shape[1]
        embd_cols = [f"dim-{idx+1}" for idx in range(embd_dims)]

        X_info = np.concatenate([
            [embd[info_col] for info_col in info_cols]
            for embd in embeddings
        ], axis=1).T
        X_embd = np.concatenate([embd["embedding"] for embd in embeddings], axis=0)

        df_info = pd.DataFrame(X_info, columns=info_cols)
        df_ebd = pd.DataFrame(X_embd, columns=embd_cols)

        df_ebd = pd.concat([df_info, df_ebd], axis=1)

        return df_ebd
