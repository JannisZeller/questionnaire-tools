"""
# Some helper functions for argument and dict validation
"""

import numpy as np


def check_key(
    key: str,
    dct: dict,
    Ex: Exception,
    dct_name: str=None,
    ex_str: str=None,
) -> None:
    """Checks the presence of the specified key in the passed dictionary. If
    the exception class to raise is set to `None` the `ex_str` is printed as a
    warning.

    Parameters
    ----------
    key : str
        Key to validate.
    dct : dict
        Dictionary supposed to contain the key.
    Ex : Exception
        Exception to raise if the key is not present
    dct_name : str
        The name of the dictionary argument for more informative error.
    ex_str : str
        A completely custom error string.

    Raises
    ------
    Ex
        An exeption of the classed passed when calling.
    """
    if key not in dct:
        if ex_str is None:
            s = f"The key \"{key}\" is not contained in the passed {dct_name}-dictionary."
        else:
            s = ex_str

        if Ex is not None:
            raise Ex(s)
        else:
            print("Warning:" + ex_str)


def check_options(
    arg,
    valid_opts: list,
    Ex: Exception,
    arg_name: str,
    ex_str: str=None,
) -> None:
    """Checks wether a (literal) argument is one of some valid options

    Parameters
    ----------
    arg : Any
        The argument
    valid_opts : list
        The valid options
    Ex : Exception
        The exception to raise if the arguments value is not contained in the
        valid options.
    arg_name : str
        The name of the argument for more informative error.
    ex_str : str
        A completely custom error string.

    Raises
    ------
    Ex
        An exeption of the classed passed when calling.
    """
    if arg not in valid_opts:
        if ex_str is None:
            s = f"The `{arg_name}` must be in `{valid_opts}`. Got:\n{arg}"
        else:
            s = ex_str
        raise Ex(s)


def check_type(
    arg,
    type_,
    Ex: Exception,
    arg_name: str=None,
    ex_str: str=None,
)-> None:
    """Checks the type of an argument passed.

    Parameters
    ----------
    arg : Any
        The argument
    type_ : _type_
        The expected type
    Ex : Exception
        The exception to be raised if the arguments type does not match the
        expected type.
    arg_name : str
        The name of the argument for more informative error.
    ex_str : str
        A completely custom error string.

    Raises
    ------
    Ex
        An exeption of the classed passed when calling.
    """
    if not isinstance(arg, type_):
        if ex_str is None:
            s = f"The `{arg_name}` must be of type `{type_}`. Got:\n{type(arg)}"
        else:
            s = ex_str
        raise Ex(s)


def check_uniqueness(
    arr: np.ndarray | list,
    Ex: Exception,
    arg_name: str=None,
    ex_str: str=None,
):
    """Checks wether the elements of an iterable are unique.

    Parameters
    ----------
    arr : np.ndarray | list
        An iterable
    Ex : Exception
        The exception to be raised if the elements of `arr` are non-unique.
    arg_name : str
        The name of the argument for more informative error.
    ex_str : str
        A completely custom error string.

    Raises
    ------
    Ex
        An exeption of the classed passed when calling.
    """
    vals, val_counts = np.unique(arr, return_counts=True)
    non_unique_vals = vals[val_counts > 1]

    if not len(non_unique_vals) == 0:
        if ex_str is None:
            s = f"Some Elements of `{arg_name}` are not unique:\n{non_unique_vals}"
        else:
            s = f"{ex_str}\nNon unique values: {non_unique_vals}"
        raise Ex(s)
