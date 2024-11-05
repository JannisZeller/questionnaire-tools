"""
The "core"-library containing wrapper and helper functions. It is intended to
outsource some of the logic of the main library to make the used interface
classes at least a little bit more readable and smaller. They get large enough
anyway, though...
"""

from .pandas import *
from .scores import *
from .batched_iter import *
from .validation import *
from .classifier import *
from .text import *
from .ols import LinearModel

from .io import read_data, write_data
