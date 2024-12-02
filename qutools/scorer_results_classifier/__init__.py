"""
# Scorer-Results-Classification

This module provides a class to classify the results of a QuScorer object. It can
be used for data-leakage-free evaluation of a multi-layer classification pipeline,
that is based on a task-wise scoring of the questionnaire.
"""

from .sr_classifier import QuScorerResultsClassifier
from .sr_classifier_results import QuSRClassifierResults
