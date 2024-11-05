import numpy as np
import pandas as pd
from typing import Literal

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


from ..scorer_results_classifier.sr_classifier_results import QuSRClassifierResults

from ..data.config import QuConfig
from ..data.data import QuData
from ..data.id_splits import IDsSplit, IDsKFoldSplit
from ..core.trainulation import evaluate_predictions, print_scores


class QuClassifierResults(QuSRClassifierResults):

    def get_df(self) -> pd.DataFrame:
        return self.df_preds
