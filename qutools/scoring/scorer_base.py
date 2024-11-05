"""
# Base-Scorer

Base class for the scorer-models that can be trained to predict the scores
for the tasks of a questionnaire. The scores are predicted based on the
textual responses by the individual participants.

Results of evaluation workflows
etc. are always provided as [`QuScorerResults`][qutools.scoring.scorer_results] objects.
These provide methods for the evaluation and visualization of the results
themselves and can also be used to easily save the results to disk.
"""


import numpy as np
import pandas as pd

from matplotlib.pyplot import close as pltclose
from matplotlib.pyplot import show as pltshow

from abc import ABC, abstractmethod

from ..data.data import QuData

from .scorer_results import QuScorerResults




class QuScorerError(Exception):
    """An exeption class for scorer-models"""


class QuScorer(ABC):
    """Base class for the different scoring models that can be trained to
    predict the scores for the tasks of a questionnaire based on the textual
    responses of the participants. The arguments of the methods might vary
    depending on the specific model."""

    @abstractmethod
    def predict(self, *args, **kwargs) -> pd.DataFrame:
        """Predict the scores for the tasks of the questionnaire based on the
        textual responses of the participants (might already be embeddings
        in case of the `QuEmbeddingsScorer`).
        """
        pass


    @abstractmethod
    def fixed_eval(self, *args, **kwargs) -> pd.DataFrame:
        """Evaluate the model on a fixed evaluation-split of the data. This can
        be used in a cross-validation or basic train-test-split scenario.
        """
        pass


    @abstractmethod
    def random_train_testing(self, *args, **kwargs) -> QuScorerResults:
        """Train the model on a random train-test-split of the data. The model
        is trained on the training-part and evaluated on the test-part."""
        pass


    @abstractmethod
    def random_cross_validate(self, *args, **kwargs) -> QuScorerResults:
        """Perform a random $k$-fold cross-validation on the data. The data
        is split randomly into $k$ parts and the model is trained and evaluated
        $k$ times using each part as the test-data (and excluding it from the
        training-data) once."""
        pass


    @abstractmethod
    def complete_data_fit(self, *args, **kwargs,) -> QuScorerResults:
        """Train the model on the complete data and evaluate it on the complete
        data. This is useful for the final model that should be used to predict
        the scores for new data in a "production" setting."""
        pass


    @staticmethod
    def _append_mc_only(
        df: pd.DataFrame,
        qudata_mc: QuData,
        df_ebd: pd.DataFrame=None,
    ) -> pd.DataFrame:
        id_col = qudata_mc.quconfig.id_col
        df_scr = qudata_mc.get_scr(mc_scored=True, verbose=False)
        missing_ids = list(np.setdiff1d(df_scr[id_col], df[id_col]))
        if df_ebd is not None:
            if any([id_ in df_ebd[id_col] for id_ in missing_ids]):
                raise QuScorerError(
                    "The missing IDs (append_mc_only) are present in the ebmbeddings data."
                )
        df_miss = df_scr[df_scr[id_col].isin(missing_ids)].reset_index(drop=True)
        df = pd.concat([df, df_miss])
        df = df.sort_values(by=id_col).reset_index(drop=True)
        return df
