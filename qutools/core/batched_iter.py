"""
# Submodule for batch-iterators

This submodule provides a generic batch iterator, `BatchIter`, designed to work with various types of iterables
such as lists, NumPy arrays, and pandas DataFrames. It allows for the efficient processing of large datasets
by dividing them into smaller, more manageable batches. This is particularly useful in scenarios where operating
on the entire dataset at once would be computationally expensive or infeasible.

The `BatchIter` class is a workaround for compatibility issues with PyTorch and Python versions above 3.11,
offering functionality similar to the `itertools.batched` available in Python versions 3.12 and above.
It is designed to be flexible and can handle any slicable iterable, providing a convenient way to iterate
over data in fixed-size chunks.

Usage
-----
The `BatchIter` class can be instantiated with any slicable iterable (e.g., list, NumPy array, pandas DataFrame)
and a batch size. Once instantiated, it can be used in a for-loop to process each batch sequentially.

Example
-------
```python
import numpy as np
import pandas as pd
from qutools.core.batched_iter import BatchIter
```
Example with a list:
```python
data_list = [i for i in range(100)]
batch_iter_list = BatchIter(data_list, batch_size=10)
for batch in batch_iter_list:
    print(batch)
```
```
> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
> [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
> ...
> [100, 101, 102]
```
Example with a numpy array:
```python
data_array = np.arange(100)
batch_iter_array = BatchIter(data_array, batch_size=10)
for batch in batch_iter_array:
    print(batch)
```
```
> [0 1 2 3 4 5 6 7 8 9]
> [10 11 12 13 14 15 16 17 18 19]
> ...
> [100 101 102]
```
Example with a pandas DataFrame:
```python
data_df = pd.DataFrame({'numbers': range(100)})
batch_iter_df = BatchIter(data_df, batch_size=10)
for batch in batch_iter_df:
    print(batch)
```
```
> [0 1 2 3 4 5 6 7 8 9]
> [10 11 12 13 14 15 16 17 18 19]
> ...
> [100 101 102]
```
"""

import numpy as np
import pandas as pd
from math import ceil

from typing import TypeVar, Generic, Self



T = TypeVar('T', list, np.ndarray, pd.DataFrame)
class BatchIter(Generic[T]):
    """Class for a batched iterator. Similar to `itertools.batched` in Python
    \\> 3.12 but PyTorch is incompatible with Python > 3.11 atm (2024-01-23).
    Works for anything that is slicable like `ls[lower : upper]`.
    `T = TypeVar('T', list, np.ndarray, pd.DataFrame)`
    """

    def __init__(self, iterable: T, batch_size: int) -> Self:
        """Constructs a BatchIter from the passed iterable.

        Parameters
        ----------
        iterable : Generic[T]
            An iterable with the generic type `T`.
        batch_size: int
            The batch size to be used.

        Returns
        -------
        BatchIter(Generic[T])
            A batched iterator corresponding to the input iteratble type.
        """
        self.current_idx = 0
        self.ls = iterable
        self.lslen = len(iterable)
        self.len = ceil(len(iterable) / batch_size)
        self.batch_size = batch_size

    def __getslice(self, lower: int, upper: int) -> T:
        if upper <= self.lslen:
            return self.ls[lower : upper]
        if lower <= self.lslen and upper > self.lslen:
            return self.ls[lower : ]

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> Self:
        self.current_idx = 0
        return self

    def __next__(self) -> T:
        if self.current_idx >= len(self.ls):
            raise StopIteration
        lower = self.current_idx
        upper = lower + self.batch_size
        self.current_idx = upper
        return self.__getslice(lower, upper)

    def __getitem__(self, idx: int) -> T:
        if idx >= self.len:
            raise IndexError("Index out of Bounds")
        lower = idx * self.batch_size
        upper = lower + self.batch_size

        return self.__getslice(lower, upper)

    def reset(self):
        self.current_idx = 0


def batched(iterable: T, batch_size: int):
    """A util function that returns a batched iterator (`BatchIter`)
    `T = TypeVar('T', list, np.ndarray, pd.DataFrame)`

    Parameters
    ----------
    iterable : Generic[T]
        An iterable with the generic type `T`.
    batch_size: int

    Returns
    -------
    BatchIter(Generic[T])
        A batched iterator corresponding to the input iteratble type.
    """
    return BatchIter(iterable, batch_size)
