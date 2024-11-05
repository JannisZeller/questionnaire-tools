"""
# Submodule for in-out-stream helpers

This submodule contains some shorthand functions for reading and writing data
using pandas, pathlib und shutil. It also contains some error handling for
differentiation between file- and directory-paths. It is not designed to be
used standalone but rather as a helper for the other submodules, which is why there
will be no examples here.
"""

import pandas as pd

from pathlib import Path
from shutil import rmtree


class FiletypeError(Exception):
    """An exception to denote wrong filetypes as denoted by suffixes."""
    pass


def empty_or_create_dir(path: Path) -> None:
    """Empties or creates the passed path.

    Parameters
    ----------
    path : Path
        The path to create or empty.
    """
    dir_ = Path(path)
    if dir_.exists():
        rmtree(dir_)
    dir_.mkdir(parents=True)


def filetype_error(suffix: str) -> None:
    """Asserts only suitable filetypes are used.

    Parameters
    ----------
    suffix : str
        Throws a `FiletypeError` with an informative string.
    """
    raise FiletypeError(
        "Only filetypes .xlsx, .xls, .gzip (parquet) and .csv can be automatically read and saved.\n" +
        f"Got: {suffix}"
    )

def path_suffix_warning(suffix: str, obj: str="this") -> None:
    """Warns if the passed path has a suffix. This indicates, that the passed
    string might be a filepath not a directory.

    Parameters
    ----------
    suffix : str
        The suffix of the current path.
    obj : str
        The object name that is likely to be saved.
    """
    if suffix != "":
        print(
            "Warning: You most likely entered a file-name not a directory. To save\n" +
            f"a {obj} object, a directory is needed. Your entered string will be interpreted\n" +
            "as a directory."
        )


def read_data(path: str, return_none: bool=False, **kwargs) -> pd.DataFrame:
    """Wrapper for loading data using pandas by automatically inferring the
    suffix.

    Parameters
    ----------
    path : str
        Path to the data-file to load. Must be a excel (.xls, .xlsx), csv (.csv)
        or parquet (.gzip) file.
    return_none : bool
        Wether `None` should be returned if the file is not found.

    Returns
    -------
    pd.DataFrame
    """
    suffix = Path(path).suffix
    try:
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path, sheet_name=kwargs.get("sheet_name", 0))
        elif suffix in [".gzip"]:
            return pd.read_parquet(path)
        elif suffix in [".csv"]:
            return pd.read_csv(path)
        else:
            filetype_error(suffix=suffix)
    except FileNotFoundError as e:
        if return_none:
            return None
        else:
            raise e

def write_data(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Wrapper for saving data using pandas by automatically inferring the
    suffix.

    Parameters
    ----------
    df : pd.DataFrame
        The data to save.
    path : str
        Path to the data-file to save to. Must be a excel (.xls, .xlsx), csv (.csv)
        or parquet (.gzip) file.

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)

    if p.parent != Path(".") and not p.parent.exists():
        p.parent.mkdir(parents=True)

    suffix = p.suffix
    if suffix in [".xlsx", ".xls"]:
        df.to_excel(p, index=False)
    if suffix in [".gzip"]:
        df.to_parquet(p, index=False)
    if suffix in [".csv"]:
        df.to_csv(p, index=False)
