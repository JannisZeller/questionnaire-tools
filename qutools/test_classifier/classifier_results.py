"""
# Questionnaire-Classifier-Results

This class is a wrapper around the classification results. It simply borrows
all its functionality from the [QuSRClassifierResults][qutools.scorer_results_classifier.QuSRClassifierResults]
class, because the prediction output is similar.
"""

import pandas as pd

from ..scorer_results_classifier.sr_classifier_results import QuSRClassifierResults


class QuClassifierResults(QuSRClassifierResults):

    def get_df(self) -> pd.DataFrame:
        """Get the DataFrame of the classification results.
        """
        return self.df_preds
