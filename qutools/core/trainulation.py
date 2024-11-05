"""
# Submodule with ML-classification-model training- and evaluation-helpers

This submodule provides abstractions for train-test data splits and wrappers
for evaluations of model predictions. The core of the training loops is the
`training_core` function, which trains a model on one train-test-split and returnes
the predictions and the trained model.

Usage
-----
```python
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from qutools.core.classifier import ScikitClassifier
from qutools.core.trainulation import (
    Split,
    get_random_oversampler,
    training_core,
    evaluate_predictions,
    print_scores,
)

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=5555,
)
features = [f"X{i+1}" for i in range(X.shape[1])]
target = "target"

df = pd.DataFrame(np.c_[X, y], columns=features + [target])
df_trn, df_tst = train_test_split(df, test_size=0.2)
split = Split(df_trn, df_tst)

clf_kernel = RandomForestClassifier(n_estimators=100, random_state=5555)
clf = ScikitClassifier(model=clf_kernel)

oversampler = get_random_oversampler()

df_trn_pred, df_tst_pred, *_ = training_core(
    model=clf,
    split=split,
    feature_cols=features,
    target_col=target,
    prediction_col="prediction",
    oversampler=oversampler,
)

trn_eval = evaluate_predictions(
    y_true=df_trn_pred["target"],
    y_pred=df_trn_pred["prediction"],
    mode="train",
)
tst_eval = evaluate_predictions(
    y_true=df_tst_pred["target"],
    y_pred=df_tst_pred["prediction"],
    mode="test",
)

print_scores(trn_eval)
```
```
>   Acc_train:                1.0
>
>   F1 (weighted)_train:      1.0
>
>   Cohens Kappa_train:       1.0
>
>   Confusion matrix_train:
>         Pred 0  Pred 1
> True 0   412.0     0.0
> True 1     0.0   388.0
```
```python
print_scores(tst_eval)
```
```
>   Acc_train:                0.975
>
>   F1 (weighted)_train:      0.975
>
>   Cohens Kappa_train:       0.949
>
>   Confusion matrix_train:
>         Pred 0  Pred 1
> True 0    85.0     4.0
> True 1     1.0   110.0
```
"""

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

from scipy import stats

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import matplotlib.pyplot as plt

from typing import Literal
from dataclasses import dataclass
from copy import deepcopy


from .classifier import Classifier, PyTorchClassifier, ScikitClassifier



@dataclass
class Split:
    """Dataclass for train and test datasets
    """
    df_trn: pd.DataFrame
    df_tst: pd.DataFrame=None

    def __init__(self, df_trn: pd.DataFrame, df_tst: pd.DataFrame=None) -> None:
        """Initilizes a Split by storing a train- and a test-dataset.

        Parameters
        ----------
        df_trn : pd.DataFrame
            Training dataset
        df_tst : pd.DataFrame
            Test dataset

        Returns
        -------
        Split
        """
        self.df_trn = df_trn.sample(frac=1)
        if df_tst is None:
            self.df_tst = df_tst
        else:
            self.df_tst = df_tst.sample(frac=1)

    def has_test_set(self):
        return self.df_tst is not None

    @staticmethod
    def from_mode_col(
        df: pd.DataFrame,
        mode_col: str="mode",
        assert_test_data: bool=True,
    ) -> "Split":
        """Initializes a split from a full dataframe and a column denoting the
        split-mode (i. e. "train" or "test").

        Parameters
        ----------
        df : pd.DataFrame
            A pandas dataframe containing the mode column whichs entries should
            be "train" or "test".
        mode_col : str
            The name of the mode column.
        assert_test_data : bool
            Whether the existance of test data should be asserted.

        Returns
        -------
        Split
        """
        df_trn = df[df[mode_col] == "train"].copy().reset_index(drop=True)
        df_tst = df[df[mode_col] == "test"].copy().reset_index(drop=True)
        if df_trn.shape[0] == 0:
            raise ValueError(
                f"The passed data has no instance, where the mode-col \"{mode_col}\" is equal to \"train\"."
            )
        if df_tst.shape[0] == 0:
            if assert_test_data:
                raise ValueError(
                    f"The passed data has no instance, where the mode-col \"{mode_col}\" is equal to \"test\"."
                )
            else:
                df_tst = None
        return Split(df_trn=df_trn, df_tst=df_tst)

    @staticmethod
    def from_tst_ids(
        df: pd.DataFrame,
        tst_ids: list[str]=None,
        id_col: str="ID",
    ) -> "Split":
        """Creates a split dataclass given a test-id list and an id-column.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the features, targets and an ID-column to
            identify test-edits.
        tst_ids : list[str]
            The ids, that should be used as test set.
        id_col : str
            The name of the ID-column as a string

        Returns
        -------
        Split
            A [Split][qutools.core.trainulation.Split] dataclass instance
        """
        """
        """
        if tst_ids is not None:
            df_trn = df[df[id_col].isin(tst_ids)==False].copy().reset_index(drop=True)  # noqa: E712
            df_tst = df[df[id_col].isin(tst_ids)].copy().reset_index(drop=True)
            assert (df_trn.shape[0] + df_tst.shape[0]) == df.shape[0], "Error in splitting!"
        else:
            df_trn = df
            df_tst = None
        return Split(df_trn=df_trn, df_tst=df_tst)



@dataclass
class SplitData:
    """Dataclass for train-test data. Other than a [Split][qutools.core.trainulation.Split]
    this only contains features and targets and already separates them in
    differents attributes.
    """
    X_trn: np.ndarray|list
    y_trn: np.ndarray|list
    X_tst: np.ndarray|list=None
    y_tst: np.ndarray|list=None

    def has_test_set(self):
        """If the SplitData contains test data."""
        return self.y_tst is not None

    def get_data(self, mode: Literal["train", "test"]="train") -> tuple[np.ndarray, np.ndarray]:
        """Returns train- or test-dataset in array form.

        Parameters
        ----------
        mode : Literal["train", "test"]
            Wether training or test data should be returned.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The features and labels (in that order).
        """
        if mode == "train":
            return self.X_trn, self.y_trn
        if mode == "test":
            return self.X_tst, self.y_tst

    @staticmethod
    def get_split_data(
        split: Split,
        feature_cols: list[str],
        target_col: str,
    ) -> "SplitData":
        """Extracts relevant data for training from a [Split][qutools.core.trainulation.Split]
        and wraps it in a [SplitData][qutools.core.trainulation.SplitData] container.

        Parameters
        ----------
        split : Split
            A [Split][qutools.core.trainulation.Split] containing `pd.DataFrame`s
            with `feature_cols` and a `target_col`.
        feature_cols : list[str]
            Column names of the features.
        target_col : str
            Column name of the target.

        Returns
        -------
        SplitData
            A [SplitData][qutools.core.trainulation.SplitData] object containing
            the separated feature and target data.
        """
        X_trn = split.df_trn[feature_cols].values
        y_trn = split.df_trn[target_col].values

        if split.has_test_set():
            X_tst = split.df_tst[feature_cols].values
            y_tst = split.df_tst[target_col].values
        else:
            X_tst = None
            y_tst = None

        return SplitData(
            X_trn=X_trn,
            y_trn=y_trn,
            X_tst=X_tst,
            y_tst=y_tst,
        )




def get_random_oversampler():
    """Tries to initialize an [imbalanced-learn](https://imbalanced-learn.org/stable/install.html#getting-started)
    `RandomOverSampler`.

    Returns
    -------
    RandomOverSampler

    Raises
    ------
    ModuleNotFoundError
        If the imblearn package is not installed.
    """
    try:
        from imblearn.over_sampling import RandomOverSampler
        oversampler = RandomOverSampler()
        return oversampler
    except ModuleNotFoundError as e:
        estr = (
            "To use oversampling the imbalanced-learn package " +
            "(https://imbalanced-learn.org/stable/install.html#getting-started) " +
            "is needed.\n" +
            f"Original Error: {e}"
        )
        raise ModuleNotFoundError(estr)



def training_core(
    model: Classifier,
    split: Split,
    feature_cols: list[str],
    target_col: str,
    prediction_col: str,
    oversampler=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict|pd.DataFrame|None, Classifier]:
    """Core of a cross validation step.

    Parameters
    ----------
    model : Classifier
        A [Classifier][qutools.core.classifier.Classifier] object to train and
        evaluate.
    split : Split
        A data split.
    feature_cols : list[str]
        Feature columns used for prediction.
    target_col : str
        Target column.
    prediction_col: str
        Name of the column to store the predictions in.
    oversampler : None
        Oversampler to be applied to the training data (not for prediction).
        Typically used with the `imbalanced-learn` packages `RandomOverSampler`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict|pd.DataFrame|None, Classifier]
        In order:

        1. The dataframe containing the training data, with predictions
        2. The dataframe containing the test data, with predictions
        3. The training history as a dict or dataframe.
        4. The `Classifier`-model trained on the training data.
    """
    model.reset()

    split_data = SplitData.get_split_data(split, feature_cols, target_col)
    X_trn, y_trn = split_data.get_data("train")
    X_tst, y_tst = split_data.get_data("test")

    if oversampler is not None:
        X_trn, y_trn = oversampler.fit_resample(X_trn, y_trn)

    if isinstance(model, PyTorchClassifier):
        fit_history = model.fit(X_trn, y_trn, X_tst, y_tst)
    elif isinstance(model, ScikitClassifier):
        fit_history = model.fit(X_trn, y_trn)
    else:
        raise ValueError(
            "The passed model instance must be an instance of either `PyTorchClassifier` or `ScikitClassifier`."
        )

    y_pred_trn = model.predict(split_data.X_trn)
    df_pred_trn = split.df_trn.copy().drop(columns=feature_cols)
    df_pred_trn[prediction_col] = y_pred_trn

    if split.has_test_set():
        y_pred_tst = model.predict(split_data.X_tst)
        df_pred_tst = split.df_tst.copy().drop(columns=feature_cols)
        df_pred_tst[prediction_col] = y_pred_tst
    else:
        df_pred_tst = None

    model_copy = deepcopy(model)

    return df_pred_trn, df_pred_tst, fit_history, model_copy



def _pad_square_matrix(arr: np.ndarray, n: int, pad_value: float=0) -> np.ndarray:
    """Pads a array if arbitrary dimensionality to "square" format, i.e., all
    dimensions have the same length `n`.

    Parameters
    ----------
    arr : np.ndarray
        Array to be padded.
    n : int
        Dimension length to pad to.
    pad_value : float
        Value to pad with.

    Returns
    -------
    np.ndarray
        Padded array of shape $n \\times n \\times n \\times ...$ with the
        dimensionality of `m`
    """
    shape = arr.shape
    pad_vals = []
    for nic in shape:
        ni = n - nic
        if ni < 0:
            raise ValueError(
                "Only can pad to square shape, i.e., all dimensions must be " +
                "smaller than `n`."
            )
        pad_vals.append(ni)
    pad_tuples = tuple([(0, ni) for ni in pad_vals])

    m_ret = np.pad(arr, pad_tuples, "constant", constant_values=pad_value)
    return m_ret



def aggregate_score(metric_name: str, score_values: list) -> np.ndarray|float:
    """Aggregates a metric-scores-list.

    Parameters
    ----------
    metric_name : str
        The name of the metric.
    score_values : list
        List of scores in the corresponding metric.

    Returns
    -------
    np.ndarray | float
    """
    if "matrix" in metric_name:
        n = np.max([m.shape[0] for m in score_values])
        matrix_list = [_pad_square_matrix(m, n) for m in score_values]
        matrix_list = [np.expand_dims(m, axis=0) for m in matrix_list]
        stacked_matrix = np.concatenate(matrix_list, axis=0)
        mean_score = np.mean(stacked_matrix, axis=0)
    else:
        mean_score = np.mean(score_values)
    return mean_score



def print_scores(scores: dict[str, float|np.ndarray]) -> None:
    """Prints summaries for a scores-dict.

    Parameters
    ----------
    scores: dict[str, float|np.ndarray]
        Metric-scores dict to be summarized
    """
    for score_name, score_values in scores.items():
        mean_score = aggregate_score(score_name, score_values)
        if "matrix" in score_name:
            print(f"\n  {(score_name+':')}")
            con_tab = pd.DataFrame(np.round(mean_score, 1))
            con_tab.index = [f"True {i}" for i in range(con_tab.shape[0])]
            con_tab.columns = [f"Pred {k}" for k in range(con_tab.shape[1])]
            print(con_tab)
        else:
            print_s = f"\n  {(score_name+':'):<25} {np.round(mean_score, 3)}"
            if len(score_values) > 1:
                sd_score = np.std(score_values, ddof=1)
                df = len(score_values)
                print_s += f" ({np.round(sd_score, 3)})"
                if df >= 20:
                    conf_int = stats.t.interval(0.95, loc=mean_score, scale=sd_score, df=df)
                    print_s += f"  [{np.round(conf_int[0], 3)}, {np.round(conf_int[1], 3)}] (CI)"
                else:
                    print_s += f"  [{np.min(score_values):.3f}, {np.max(score_values):.3f}] (min,max)"
            print(print_s)



def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mode: Literal["train", "eval", "test"],
    metrics_dict: dict=None,
    n_labels: int=None,
    scores_dict: dict[str, list]=None,
) -> dict[str, list]:
    """A wrapper to evaluate a model (or its outputs) accoring to some metrics
    given in the `metrics_dict`. Can construct a new dict with scores or
    append to a new one.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Model predictions.
    mode : Literal["train", "eval", "test"]
        Mode in use. Does not have an impact on the actual scoring but the
        names in the returned `scores_dict`.
    metrics_dict : dict
        Metrics to be used. If none is provided a default one containing
        accuracy, f1, & Cohen's Kappa is used.
    n_labels : int
        To overwrite the number of labels contained in `y_true`.
    scores_dict : dict[str, list]
        The scores dict to be appended to. Will then be added to and returned.

    Returns
    -------
    dict[str, float | np.ndarray]
        The newly constructed or extended `scores_dict`. The keys have the
        structure `f"{metric_name}_{mode}"`. The values are the lists containing
        the scores.
    """
    if n_labels is None:
        distinct_labels = np.unique(y_true)
    else:
        distinct_labels = np.arange(0, n_labels)

    if metrics_dict is None:
        metrics_dict = {
            "Acc": accuracy_score,
            "F1 (weighted)": lambda y_true, y_pred: f1_score(
                y_true, y_pred, labels=distinct_labels,
                average="weighted",
                zero_division=0,
            ),
            "Cohens Kappa": lambda y_true, y_pred: cohen_kappa_score(
                y_true, y_pred, labels=distinct_labels
            ),
            "Confusion matrix": lambda y_true, y_pred: confusion_matrix(
                y_true, y_pred, labels=distinct_labels,
            ),
        }

    if scores_dict is None:
        scores_dict: dict[str, list] = {}
    for metric_name, score_fun in metrics_dict.items():
        key = f"{metric_name}_{mode}"
        if key not in scores_dict:
            scores_dict[key] = [score_fun(y_true, y_pred)]
        else:
            scores_dict[key].append(score_fun(y_true, y_pred))

    return scores_dict
