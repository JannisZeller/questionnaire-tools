"""
# Scorer-Results-Regression

Like the [scorer_results_classifier][qutools.scorer_results_classifier]-module,
this module is a wrapper around the scoring-results for data-leakage free evaluation
of multi-layer models based on scoring results. However, this can be used for
regression models, for the prediction of continuous values, especially subscale
scores.
"""

from .sr_regressor import QuScorerResultsRegressor
from .sr_regressor_results import QuRegressionResults
