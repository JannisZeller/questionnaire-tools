"""
This module is used for setting up train-eval / train-test-splits for the
questionnaire data. The Splits are done "booklet-wise" (i.e., the data is split
such that single responses from the same questionnaire booklet are not present
in both the train and test set).
"""

from .id_split_base import IDsSplit
from .id_split_k_fold import IDsKFoldSplit
from .id_split_train_test import IDsTrainTestSplit
