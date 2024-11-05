"""
# Submodule with wrappers for text-data handling

This submodule provides functions for cleaning and preprocessing text data. It
is primarily used for the text-data processing of [qutools.data.data.QuData][qutools.data.data.QuData]-objects
and not meant to be used standalone, which is why there are only few examples here.
For an example especially on how to provide abbreviation-replacement data, refer to
[qutools.data.data][qutools.data.data].
"""

import re
import numpy as np
import pandas as pd

from .io import read_data


def cleanup_whitespace(s: str) -> str:
    """Reduces multiple consecutive whitespaces to a single one

    Parameters
    ----------
    s : str

    Returns
    -------
    str
    """
    s = s.strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


def cleanup_whitespaces(
    df: pd.DataFrame,
    cols: list[str]=["text"],
) -> pd.DataFrame:
    """Cleans whitespaces in the specified columns

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str]
        Columns to be whitespace-cleaned.

    Returns
    -------
    pd.DataFrame
    """
    whitespace_remover = np.vectorize(cleanup_whitespace)
    for col in cols:
        df[col] = whitespace_remover(df[col].astype(str).values)
    return df


def load_abbreviation_regex(path: str) -> list[tuple[str, str]]:
    """Loads a replacement-csv file. This file must contain a "pattern" column
    with regex-patterns to be replaced by the entries of the "replacement"-column.
    Automatically appends prefix and suffix for word beginning and ending.
    Two additional columns "supress_prefix" and "supress_suffix" denoting wether
    the start of word-prefix and the "end of word or period"-suffix should be
    added, which can be denoted by entering a "1" in the respective field.

    Parameters
    ----------
    path : str
        Path to the file

    Returns
    -------
    list[tuple[str, str]]
        List of tuples where the first entry of each tuple is the regex to
        be replaced and the second entry is the replacement.
    """
    # prefix for word beginning
    prefix = r"\b"
    # suffix for end of word with or without period
    suffix = r"\.?(?=\s|\:|$)"

    df = read_data(path)
    for col in ["pattern", "replacement", "supress_prefix", "supress_suffix"]:
        if col not in df.columns:
            raise KeyError(f"The \"{col}\" column is missing in the abbreviation data.")

    repl_tpls = []
    for _, row in df.iterrows():
        pat = row["pattern"]
        if not row["supress_prefix"] == 1:
            pat = prefix + pat
        if not row["supress_suffix"] == 1:
            pat = pat + suffix

        repl_tpls.append((pat, row["replacement"]))

    return repl_tpls


def replace_abbreviation(s: str, repl_tpls: list[tuple[str, str]]) -> str:
    """Replaces typically found abbreviations in text strings.

    Parameters
    ----------
    s : str
    repl_tpls : list[tuple[str, str]]
        The replacements as list of tuples. If `None` it is loaded from the
        `data.text` submodule.

    Returns
    -------
    str
    """
    for abbr, expr in repl_tpls:
        s = re.sub(abbr, expr, s, flags=re.IGNORECASE)
    return s


def replace_abbreviations(
    path : str,
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Replaces abbreviations in the specified columns

    Parameters
    ----------
    path : str
        Path to the abbreviation replacement table.
    df : pd.DataFrame
    cols : list[str]
        Columns to be abbreviations-cleaned.

    Returns
    -------
    pd.DataFrame
    """
    repl_tpls = load_abbreviation_regex(path)
    abbreviations_replacer = np.vectorize(lambda x: replace_abbreviation(x, repl_tpls))
    for col in cols:
        df[col] = abbreviations_replacer(df[col].astype(str).values)
    return df


def set_item_task_prefix(
    df_txt: pd.DataFrame,
    unit_col: str,
    it_repl_dict: dict[str, str]=None,
    sep: str=" [SEP] ",
    verbose: bool=True,
) -> pd.DataFrame:
    """Prepends a prefix derived from one column to the text in another column of a pandas DataFrame.

    Parameters
    ----------
    df_txt : pandas.DataFrame
        The DataFrame containing the text to which prefixes will be added.
    unit_col : str
        The name of the column from which the prefix will be derived.
    it_repl_dict : dict
        A dictionary specifying replacements to be made in the `unit_col` values before
        using them as prefixes. The keys in the dictionary are the strings to be replaced,
        and the values are the strings to replace them with. If `None`, no replacements are made.
    sep : str
        The separator to be used between the task-prefix and the text.
    verbose : bool
        If `True`, prints progress and actions to the console.

    Returns
    -------
    pandas.DataFrame
        The modified DataFrame with prefixes added to the specified text column.

    Notes
    -----
    The function uses regular expressions for replacements specified in `it_repl_dict` for
    flexibility in matching patterns. It defines a local helper function `__add_prefix` to
    concatenate the prefix and the text string with ": " as a separator. This helper function
    is applied efficiently across arrays using `numpy.vectorize`.

    Example
    -------
    ```python
    import pandas as pd
    df = pd.DataFrame({'task': ['A', 'B'], 'text': ['test1', 'test2']})
    print(set_item_task_prefix(
        df_txt=df,
        unit_col='task',
        it_repl_dict={'A': 'Alpha', 'B': 'Beta'},
        verbose=False
    ))
    ```
    ```
    >   task               text
    > 0    A  Alpha [SEP] test1
    > 1    B   Beta [SEP] test2
    ```
    ```python
    print(set_item_task_prefix(
        df_txt=df,
        unit_col='task',
        it_repl_dict={'A': 'Alpha', 'B': 'Beta'},
        sep=': ',
        verbose=True,
    ))
    ```
    ```
    > Prepending task names.
    > -> Replacing {'A': 'Alpha', 'B': 'Beta'}
    >   task                      text
    > 0    A  Alpha: Alpha [SEP] test1
    > 1    B    Beta: Beta [SEP] test2
    ```
    """
    if verbose:
        print(f"Prepending {unit_col} names.")
    item_names = df_txt[unit_col].to_list()

    if it_repl_dict is not None:
        if verbose:
            print(f" -> Replacing {it_repl_dict}")
        for old, new in it_repl_dict.items():
            item_names = np.array([re.sub(old, new, itn) for itn in item_names])

    def __add_prefix(prefix: str, text: str) -> str:
        return prefix + sep + text

    concatter = np.vectorize(__add_prefix)
    texts = df_txt["text"].values
    texts = concatter(item_names, texts)
    df_txt["text"] = texts

    return df_txt


def check_seq_lens(df_txt: pd.DataFrame, max_len: int, verbose: bool=True) -> None:
    """
    Checks the lengths of the sequences in a text DataFrame against a maximum length.

    Parameters
    ----------
    df_txt : pandas.DataFrame
        The DataFrame containing the text sequences.
    max_len : int
        The maximum expected length of the sequences.
    verbose : bool
        If `True`, prints the number and percentage of documents exceeding the maximum length.
        This argument is only set to `False` for reloading data, that has already been checked.

    Example
    -------
    ```python
    import pandas as pd
    df = pd.DataFrame({'text': ['short text', 'a very long piece of text that exceeds the max length']})
    check_seq_lens(df, max_len=10, verbose=True)
    ```
    ```
    > Documents exeeding language model's maximum sequence length (10):
    >  - Absolute: ~ 1
    >  - Relative: ~ 50.00 %
    ```
    """
    seq_lengths = 1.5 * df_txt["text"].str.split(" ").map(len).values
    exeeding_seqs = seq_lengths >= max_len

    if verbose:
        if any(exeeding_seqs):
            print(
                f"Documents exeeding language model's maximum sequence length ({max_len}):" +
                f"\n - Absolute: ~ {np.sum(exeeding_seqs)}" +
                f"\n - Relative: ~ {100 * np.mean(exeeding_seqs):.2f} %"
            )
