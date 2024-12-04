"""
# Scorer-Results-Classifier Results

This is the last layer of the 2-step classification pipeline. It is used to
store the predictions and fit-histories of the the [QuScorerResultsClassifier][qutools.scorer_results_classifier.sr_classifier.QuScorerResultsClassifier]
objects for evaluation.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from shutil import rmtree

from ..data.config import QuConfig
from ..id_splits.id_split_k_fold import IDsKFoldSplit

from ..core.io import read_data, empty_or_create_dir, write_data, path_suffix_warning
from ..core.trainulation import evaluate_predictions, print_scores



@dataclass
class QuSRClassifierResults:
    quconfig: QuConfig
    df_preds: pd.DataFrame=None
    df_fit_histories: pd.DataFrame=None
    id_split: IDsKFoldSplit=None
    target_col: str="cluster"
    prediction_col: str="cluster_pred"
    class_order: list[str]=None


    def __init__(self,
        quconfig: QuConfig,
        df_preds: pd.DataFrame=None,
        df_fit_histories: pd.DataFrame=None,
        id_split: IDsKFoldSplit=None,
        target_col: str="cluster",
        prediction_col: str="cluster_pred",
        class_order: list[str]=None,
    ) -> None:
        """The `QuSRClassifierResults`-class is used to store the predictions and
        fit-histories of the the [QuScorerResultsClassifier][qutools.scorer_results_classifier.sr_classifier.QuScorerResultsClassifier]
        objects for evaluation. It is produced by the fit-methods of the classifiers.

        Parameters
        ----------
        quconfig : QuConfig
            The configuration of the Qu-Classifier.
        df_preds : pd.DataFrame
            The predictions of the classifier.
        df_fit_histories : pd.DataFrame
            The fit-histories of the classifier.
        id_split : IDsKFoldSplit
            The ID-split used for the cross-validation.
        target_col : str
            The name of the target-column.
        prediction_col : str
            The name of the prediction-column.
        class_order : list[str]
            The order of the classes in the confusion matrix.
        """
        self.quconfig = quconfig

        self.id_col = quconfig.id_col
        self.target_col = target_col
        self.info_cols = ["split", "mode"]
        self.prediction_col = prediction_col
        self.class_order = class_order
        self.id_split = id_split

        if df_preds is not None:
            self.__validate_columns(df_preds)
        self.df_preds = df_preds

        self.df_fit_histories = df_fit_histories


    ## Validation
    def __validate_columns(self, df_preds: pd.DataFrame):
        cols = {
            "Id-column": self.id_col,
            "Target-column": self.target_col,
            "Prediction-column": self.prediction_col,
            "Info-column (1)": "split",
            "Info-column (2)": "mode",
        }
        for k, v in cols.items():
            if v not in df_preds.columns:
                raise KeyError(
                    f"{k} \"{v}\" is not present in the passed dataframe `df_preds`. \n" +
                    f"Present Columns: {df_preds.columns.to_list()}"
                )


    ## Information
    def get_n_splits(self) -> int:
        """
        Returns
        -------
        int
            Number of CV-splits in the instance.
        """
        if self.df_preds is None:
            return 0
        n = len(self.df_preds["split"].unique())
        return n

    def get_test_n(self) -> int:
        """
        Returns
        -------
        int
            Number of test/validation-samples in the instance.
        """
        if self.df_preds is None:
            return 0
        n = np.sum(self.df_preds["mode"] == "test")
        return n

    def _has_test_data(self) -> bool:
        return self.get_test_n() > 0


    ## Appending split-predictions
    def append(
        self,
        df_trn: pd.DataFrame=None,
        df_tst: pd.DataFrame=None,
        fit_history: dict=None,
    ):
        """Appends new predictions and fit-histories to the instance. Typically
        used in the cross-validation loop.

        Parameters
        ----------
        df_trn : pd.DataFrame
            The training predictions to append.
        df_tst : pd.DataFrame
            The test predictions to append.
        fit_history : dict
            The fit-history to append.
        """
        n_split = self.get_n_splits() + 1

        if df_trn is not None:
            self._append_df_pred(df_trn, n_split=n_split, mode="train")
        if df_tst is not None:
            self._append_df_pred(df_tst, n_split=n_split, mode="test")

        if fit_history is not None:
            self._append_fit_history(fit_history=fit_history, n_split=n_split)

    def _append_df_pred(
        self,
        df_pred: pd.DataFrame,
        n_split: int,
        mode: Literal["train", "test"],
    ) -> None:
        df_pred["split"] = n_split
        df_pred["mode"] = mode
        self.__validate_columns(df_pred)

        if self.df_preds is None:
            self.df_preds = df_pred
        else:
            df_preds = self.df_preds
            df_preds = pd.concat([df_preds, df_pred])
            df_preds = df_preds.reset_index(drop=True)
            self.df_preds = df_preds

    def _append_fit_history(
        self,
        fit_history: dict|pd.DataFrame,
        n_split: int,
    ) -> None:
        if isinstance(fit_history, dict):
            fit_history = pd.DataFrame(fit_history)
        df_fh = fit_history
        df_fh["split"] = n_split
        if self.df_fit_histories is None:
            self.df_fit_histories = df_fh
        else:
            df_fh = pd.concat([self.df_fit_histories, df_fh])
            self.df_fit_histories = df_fh


    ## Retrieving Data
    def _class_name_dict(self) -> dict:
        df = self.get_df()

        if df[self.target_col].dtype in ["int", "float"]:
            labels = list(df[self.target_col].unique())
            return dict(zip(labels, range(len(labels))))

        trg_labels = list(df[self.target_col].astype(str).unique())
        pred_labels = list(df[self.prediction_col].astype(str).unique())
        labels = list(set(trg_labels + pred_labels))
        n_labels = len(labels)
        label_dict = dict(zip(labels, range(n_labels)))

        return label_dict


    def _class_names_list(self, class_order: list|None=None):
        if self.class_order is not None:
            internal_co = self.class_order
        else:
            internal_co = list(self._class_name_dict().keys())

        if class_order is None:
            return internal_co

        mismatch = np.setdiff1d(internal_co, class_order)
        if len(mismatch) > 0:
            raise ValueError(
                f"There are class-names missing in the passed `class_order`: {mismatch}"
            )
        mismatch = np.setdiff1d(class_order, internal_co)
        if len(mismatch) > 0:
            raise ValueError(
                f"There are surplus class-names in the passed `class_order`: {mismatch}"
            )
        return class_order

    def get_df(self) -> pd.DataFrame:
        """Returns the predictions as a dataframe as a deepcopy.

        Returns
        -------
        pd.DataFrame
        """
        if self.df_preds is None:
            raise AttributeError("No prediction data available.")
        return self.df_preds.copy()

    def __get_vals(
        self,
        opt: Literal["targets", "predictions"],
        mode: Literal["train", "test"]="test",
    ) -> np.ndarray:
        if opt == "targets":
            col = self.target_col
        elif opt == "predictions":
            col = self.prediction_col
        else:
            raise ValueError("Invalid choice of `opt`")
        df = self.get_df()
        df = df[df["mode"] == mode]
        y = df[col].values
        # with pd.option_context("future.no_silent_downcasting", True):
        #     y = (df[col]
        #         .astype(str)
        #         .replace(self._clf_name_dict())
        #         .infer_objects(copy=False)
        #         .values)
        return y

    def _get_trgts(self, mode: Literal["train", "test"]="test") -> np.ndarray:
        return self.__get_vals(opt="targets", mode=mode)

    def _get_preds(self, mode: Literal["train", "test"]="test") -> np.ndarray:
        return self.__get_vals(opt="predictions", mode=mode)


    ## Retrieve Fit-Histories
    def get_fit_histories_dataframe(self, **kwargs) -> pd.DataFrame:
        """Returns the fit-histories as a dataframe. Generates the necessairy
        dataframe, if the fit histories previously have been stored in dict-form.

        Returns
        -------
        pd.DataFrame
        """
        if self.df_fit_histories is None:
            if not kwargs.get("supress_no_fh_print", False):
                print("No fit history data available.")
            return None
        self.df_fit_histories = self.df_fit_histories.reset_index(drop=True)
        return self.df_fit_histories.copy()


    ## Evaluation
    #   Wrappers
    def evaluation(
        self,
        splitwise: bool=False,
        mode: Literal["train", "test"]="test",
    ) -> None:
        """Evaluation of the predictions, i.e., prints performance metrics.

        Parameters
        ----------
        splitwise : bool
            Wether to evaluate the predictions splitwise.
        mode : Literal["train", "test"]
            Wether to evaluate train or test predictions.
        """
        mode = self.__validate_eval_mode(mode=mode)

        if not splitwise:
            self._evaluate(mode=mode)
        else:
            self._evaluate_splitwise(mode=mode)

    def _evaluate(
        self,
        mode: Literal["train", "test"]="test",
    ) -> None:
        df = self.get_df()
        df = df[df["mode"]==mode]
        y_true = df[self.target_col].values
        y_pred = df[self.prediction_col].values
        print(f" - N-Persons = {df.shape[0]}")
        print(f" - Acc = {accuracy_score(y_true, y_pred):.3f}")
        print(f" - F1 (weighted) = {f1_score(y_true, y_pred, average='weighted'):.3f}")
        print(f" - F1 (macro) = {f1_score(y_true, y_pred, average='macro'):.3f}")
        print(f" - Cohens Kappa = {cohen_kappa_score(y_true, y_pred):.3f}")

    #   Single Elements
    def _evaluate_splitwise(
            self,
            mode: Literal["train", "test"]="test",
            n_labels: int=None,
        ):
        """Evaluation of the predictions.

        Parameters
        ----------
        mode : Literal["train", "test"]
            Wether to evaluate train or test predictions
        n_labels : int
            To overwrite the number of labels contained in `y_true`. Necessary
            depending on the metrics used.
        """
        scores = {}
        idxs = self.df_preds["split"].unique()
        for idx in idxs:
            df = self.df_preds
            df = df[df["mode"]==mode]
            df = df[df["split"]==idx]
            scores = evaluate_predictions(
                df[self.target_col],
                df[self.prediction_col],
                mode=mode,
                n_labels=n_labels,
                scores_dict=scores,
            )
        print_scores(scores)

    def get_confusion_matrix(
        self,
        mode: Literal["train", "test"]="test",
        class_order: list=None,
        label_true_pred: bool=True,
    ) -> pd.DataFrame:
        """Get the confusion matrix of the text-column scoring.

        Parameters
        ----------
        mode : Literal["train", "test"]
            Wether to evaluate train or test predictions
        class_order : list
            The order of the classes in the confusion matrix. If `None` the
            order of the classes is determined by the order of the unique values
            in the target-column.
        label_true_pred : bool
            Wether to label the columns and rows with "_true" and "_pred".

        Returns
        -------
        pd.DataFrame
        """
        mode = self.__validate_eval_mode(mode=mode)
        cls_names = self._class_names_list(class_order=class_order)

        y1 = self._get_trgts(mode=mode)
        y2 = self._get_preds(mode=mode)

        mat = pd.DataFrame(confusion_matrix(y1, y2, labels=cls_names))
        if label_true_pred:
            mat.columns = [str(x) + "_pred" for x in cls_names]
            mat.index = [str(x) + "_true" for x in cls_names]
        else:
            mat.columns = cls_names
            mat.index = cls_names
        return mat

    #   Helpers
    def __validate_eval_mode(
        self,
        mode: Literal["train", "test"]
    ) -> Literal["train", "test"]:
        if mode not in ["train", "test"]:
            raise ValueError(f"Argument `mode` must be of `[\"train\", \"test\"]`. Got \"{mode}\".")

        if not self._has_test_data() and mode=="test":
            print("No test-data available, switching to `mode=\"train\"`.")
            mode = "train"

        return mode

    #   Visualizations
    def plot_confusion_matrix(
        self,
        mode: Literal["train", "test"]="test",
        class_order: list=None,
        save_path: str=None,
        **kwargs,
    ) -> Figure:
        """Plots the confusion matrix of the classification.

        Parameters
        ----------
        mode : Literal["train", "test"]
            Wether to evaluate train or test predictions
        class_order : list
            The order of the classes in the confusion matrix. If `None` the
            order of the classes is determined by the order of the unique values
            in the target-column.
        save_path : str
            A path to save the plot to. If `None` the plot will not be saved.

        Returns
        -------
        Figure
            The plot as a `matplotlib.figure.Figure`.
        """
        mat = self.get_confusion_matrix(
            mode=mode,
            class_order=class_order,
            label_true_pred=False,
        )
        mat_rnormed = 100 * mat / np.sum(mat.values, axis=1, keepdims=True)
        pctgs = np.round(mat_rnormed.values, 0).astype(int)
        labels = np.array([
            [f"{p} %" for p in p_row]
            for p_row in pctgs
        ])

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 5)))
        sns.heatmap(mat_rnormed, cmap=None, annot=labels, fmt="", ax=ax) # annot=True, fmt=".0f"
        _ = ax.set_xlabel(kwargs.get("xlabel", 'Vorhergesagte Cluster'))
        _ = ax.set_ylabel(kwargs.get("ylabel", 'Wahre Cluster'))
        _ = ax.tick_params(axis='x', rotation=kwargs.get("x_tick_rotation", 30))
        _ = ax.set_title(kwargs.get("title", None))

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return fig

    def plot_averaged_fit_history(
            self,
            trn_bins: int=50,
            tst_bins: int=30,
            save_path: str=None,
            **kwargs,
        ) -> Figure:
        """Plots the fit history "averaged" over the different CV-splits.
        The splits might yield nuanced differences in the `"epoch"`s-column
        values because not all splits might be equally long. Therefore the
        epochs-column gets binned / discretized to better visualize the overall
        trends.

        Parameters
        ----------
        trn_bins : int
            Bin-count for the train-metrics (loss). If $\\leq 0$ no binning is
            performed.
        tst_bins : int
            Bin-count for the test/validation-metrics (loss). If $\\leq 0$ no
            binning is performed.
        save_path : str
            A path to save the plot to. If `None` the plot will not be saved.

        Returns
        -------
        Figure
            The plot as a `matplotlib.figure.Figure`.
        """

        df_fh = self.get_fit_histories_dataframe()
        if df_fh is None:
            print("No fit-history data available - nothing to plot.")
            return

        df_fh_trn = df_fh[["epoch", "loss", "train_accuracy", "train_f1"]].copy().dropna()
        test_history_available = "eval_loss" in df_fh.columns

        if test_history_available:
            df_fh_tst = df_fh[["epoch", "eval_loss", "eval_accuracy"]].copy().dropna()

        if self.get_n_splits() > 1:
            n_epochs = df_fh_trn["epoch"].max()
            trn_bins = np.linspace(0, n_epochs, trn_bins)
            trn_labels = trn_bins[1:]

            if test_history_available:
                tst_bins = np.linspace(0, n_epochs, tst_bins)
                tst_labels = tst_bins[1:]

            df_fh_trn["epoch"] = pd.cut(
                df_fh_trn["epoch"],
                bins=trn_bins,
                labels=trn_labels,
            ).astype(float)

            if test_history_available:
                df_fh_tst["epoch"] = pd.cut(
                    df_fh_tst["epoch"],
                    bins=tst_bins,
                    labels=tst_labels,
                ).astype(float)

        fig: Figure
        ax1: Axes
        ax2: Axes
        plt.rcParams.update({'font.size': 12})

        fig, ax1 = plt.subplots(figsize=(8, 6))
        if test_history_available:
            sns.lineplot(df_fh_tst, x="epoch", y="eval_loss", errorbar="sd", color="royalblue", ax=ax1)
        sns.lineplot(df_fh_trn, x="epoch", y="loss", errorbar="sd", color="black", linestyle="--", ax=ax1)
        # sns.lineplot(df_fh_trn, x="epoch", y="train_f1", errorbar="sd", color="royalblue", linestyle="--", ax=ax1)

        ax2 = plt.twinx()
        sns.lineplot(df_fh_trn, x="epoch", y="train_accuracy", errorbar="sd", color="orangered", linestyle="--", ax=ax2)
        _ = ax1.set_xlabel(kwargs.get("loss_xlabel", 'Epoch'))
        _ = ax1.set_ylabel(kwargs.get("loss_ylabel", 'Loss'))
        _ = ax2.set_ylabel(kwargs.get("acc_ylabel", "Accuracy (Prozentuale Übereinstimmung)"))
        if test_history_available:
            sns.lineplot(df_fh_tst, x="epoch", y="eval_accuracy", errorbar="sd", color="orangered", ax=ax2)

        hs = []
        hs.append(Line2D([0], [0], label=kwargs.get("trn_loss_legend", "Loss (Train.)"), color='black', linestyle="--"))
        hs.append(Line2D([0], [0], label=kwargs.get("trn_acc_legend", "Accuracy (Train.)"), color='orangered', linestyle="--"))
        if test_history_available:
            hs.append(Line2D([0], [0], label=kwargs.get("evl_loss_legend", "Loss (Val.)"), color='royalblue'))
            hs.append(Line2D([0], [0], label=kwargs.get("evl_acc_legend", "Accuracy (Val.)"), color='orangered'))
        _ = ax1.legend(handles=hs, loc='center left', bbox_to_anchor=(1.15, 0.5))

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return fig


    ## Save-Load Logic
    def to_dir(self, path: str) -> None:
        """Saves the data to a directory. Overwrites possible existing files
        automatically.

        Parameters
        ----------
        path : str
        """
        dir_ = Path(path)
        path_suffix_warning(dir_.suffix)
        empty_or_create_dir(dir_)

        df_preds = self.df_preds
        df_fh = self.get_fit_histories_dataframe()

        if self.df_preds is None:
            raise AttributeError(
                "This QuScorerResults instance has not prediction data yet. Nothing to be saved."
            )

        self.quconfig.to_yaml(path=dir_ / "quconfig.yaml")

        df_preds = df_preds.reset_index(drop=True)
        write_data(df=df_preds, path=dir_ / "predictions.gzip")

        if df_fh is not None:
            df_fh = df_fh.reset_index(drop=True)
            write_data(df=df_fh, path=dir_ / "fit_history.gzip")

        if self.id_split is not None:
            self.id_split.to_dir(dir_ / "id_split")

    @staticmethod
    def from_dir(
        path: str,
        target_col: str="cluster",
        prediction_col: str="cluster_pred",
    ) -> "QuSRClassifierResults":
        """Loads a `CVPredictions`-object from directory.

        Parameters
        ----------
        path : str
            The filepath containing the prediction tables for the different
            splits.
        target_col : str
            The name of the target column.
        prediction_col : str
            The name of the prediction column.

        Returns
        -------
        CVPredictions
        """
        p = Path(path)
        path_suffix_warning(p.suffix)
        quconfig = QuConfig.from_yaml(path=p / "quconfig.yaml")
        df_preds = read_data(path=p / "predictions.gzip")
        df_fh = read_data(path=p / "fit_history.gzip", return_none=True)
        try:
            id_split = IDsKFoldSplit.from_dir(path=p / "id_split")
        except FileNotFoundError:
            id_split = None

        qcv = QuSRClassifierResults(
            quconfig=quconfig,
            df_preds=df_preds,
            df_fit_histories=df_fh,
            id_split=id_split,
            target_col=target_col,
            prediction_col=prediction_col,
        )

        return qcv


    @staticmethod
    def qusr_clf_performance_table(
        qusr_clfs: dict[str, "QuSRClassifierResults"],
        mode: str = "test",
        labels: list[str]=None,
    ) -> pd.DataFrame:
        """Generates a table to compare the performance of different classifiers.

        Parameters
        ----------
        qusr_clfs : dict[str, QuSRClassifierResults]
            The classifiers to compare.
        mode : str
            Wether to evaluate train or test predictions.
        labels : list[str]
            The labels to use for the models.

        Returns
        -------
        pd.DataFrame
            The table as a `pandas.DataFrame`.
        """
        return _qusr_clf_performance_table(qusr_clfs, mode=mode, labels=labels)

    @staticmethod
    def qusr_clf_performance_plot(
        qusr_clfs: dict[str, "QuSRClassifierResults"],
        mode: str = "test",
        labels: list[str]=None,
        **kwargs,
    ) -> Figure:
        """Generates a plot to compare the performance of different classifiers.

        Parameters
        ----------
        qusr_clfs : dict[str, QuSRClassifierResults]
            The classifiers to compare.
        mode : str
            Wether to evaluate train or test predictions.
        labels : list[str]
            The labels to use for the models.

        Returns
        -------
        Figure
            The plot as a `matplotlib.figure.Figure`.
        """
        return _qusr_clf_performance_plot(qusr_clfs, mode=mode, labels=labels, **kwargs)




def _qusr_clf_performance_table(
        qusr_clfs: dict[str, QuSRClassifierResults],
        mode: str = "test",
        labels: list[str]=None,
    ) -> pd.DataFrame:

    dct = {}
    for key in qusr_clfs.keys():
        y_true = qusr_clfs[key]._get_trgts(mode=mode)
        y_pred = qusr_clfs[key]._get_preds(mode=mode)
        dct[key] = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1 (weighted)": f1_score(y_true, y_pred, average="weighted"),
            "F1 (macro)": f1_score(y_true, y_pred, average="macro"),
            "Cohens Kappa": cohen_kappa_score(y_true, y_pred),
            "Quadratic Weighted Kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        }

    df = pd.DataFrame(dct, columns=qusr_clfs.keys()).T
    df["Model"] = df.index
    df = df.reset_index(drop=True)

    if labels:
        df["Model"] = df["Model"].replace(qusr_clfs.keys(), labels)

    df = df[["Model", "Accuracy", "F1 (weighted)", "F1 (macro)", "Cohens Kappa", "Quadratic Weighted Kappa"]]

    return df


def _qusr_clf_performance_plot(
    qusr_clfs: dict[str, QuSRClassifierResults],
    mode: str = "test",
    labels: list[str]=None,
    **kwargs,
) -> Figure:
    n_comparisons = len(qusr_clfs)
    df = _qusr_clf_performance_table(qusr_clfs, mode=mode, labels=labels)
    df = df.drop(columns=["Quadratic Weighted Kappa", "F1 (macro)"])
    df = pd.melt(df, id_vars="Model", var_name="metric", value_name="value")

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (3 * n_comparisons, 6)))
    sns.barplot(
        data=df,
        x="metric",
        y="value",
        hue="Model",
        palette=kwargs.get("palette", None),
        ax=ax,
    )
    ax.set_title(kwargs.get("title", f"Performance Comparison Models"))
    ax.set_xlabel(kwargs.get("xlabel", "Metric"))
    ax.set_ylabel(kwargs.get("ylabel", "Metric Value"))
    ax.set_ylim(bottom=kwargs.get("ylim_bottom", 0), top=kwargs.get("ylim_top", 1))

    if kwargs.get("outline", True):
        for bar in ax.patches:
            bar.set_edgecolor("black")
            bar.set_linewidth(1.5)

    if kwargs.get("annotate", True):
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f")

    if "legend_loc" in kwargs:
        _ = ax.legend(
            loc=kwargs.get("legend_loc", 'center left'),
            bbox_to_anchor=kwargs.get("legend_anchor", None),
        )

    else:
        _ = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    return fig
