"""
# Taskwise Analyses

This submodule provides the `QuTextTaskAnalysis` class which can be used to
analyze the performance of a scorer-model w. r. t. single tasks and to compare
the perfomance with several task-properties such as the number of responses or
the skewness of the score-labels based on the true labels. This class
mainly wrappes the `QuScorerResults` and `QuInterraterAnalysis` classes.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from typing import Literal

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from scipy.stats import skew, spearmanr, pearsonr

from ..data.data import QuData
from .scorer_results import QuScorerResults
from .interrater_analysis import QuInterraterAnalysis



def corr_analysis(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    mode: Literal["spearman", "pearson"]="spearman",
) -> None:
    if mode == "spearman":
        r, p = spearmanr(x, y)
    elif mode == "pearson":
        r, p = pearsonr(x, y)
    else:
        raise ValueError(f"`mode` must be of `[\"spearman\", \"pearson\"]`. Got: \"{mode}\".")
    print(f"Correlation ({mode.capitalize()}): {name}")
    print(f" - r = {r:.3f}")
    if p > 0.001:
        print(f" - p = {p:.3f}")
    else:
        print(f" - p < 0.001")


class QuTextTaskAnalysis:
    df_tt: pd.DataFrame

    def __init__(
        self,
        qusr: QuScorerResults|QuInterraterAnalysis,
        mode: Literal["test", "train"]="test",
        verbose: bool=False,
    ):
        """Analyzes the performance of a scorer-model or interrater-data on a task-basis.

        Parameters
        ----------
        qusr : QuScorerResults | QuInterraterAnalysis
            The scorer-results or interrater-data to be analyzed.
        mode : Literal["test", "train"]
            The mode of the data to be analyzed.
        verbose : bool
            Whether to print additional information.
        """
        self.quconfig = qusr.quconfig.copy()
        text_tasks = self.quconfig.get_text_tasks()
        self.df_tt = pd.DataFrame(columns=["task", "N Responses", "Score-Label-Skewness", "0-Fraction", "Accuracy", "F1", "Cohens Kappa"])

        if isinstance(qusr, QuScorerResults):
            df = qusr.get_long_df()
        if isinstance(qusr, QuInterraterAnalysis):
            df = qusr.get_long_df(text_data_only=True)
        if not qusr._has_test_data():
            if mode == "test":
                print("No test-data available, switching to `mode=\"train\".`")
                mode = "train"
        df = df[df["mode"]==mode].reset_index(drop=True)

        for idx, task in enumerate(text_tasks):
            df_ = df[df[qusr.unit_col]==task.name].copy()
            y_true = df_[qusr.target_col].values
            y_pred = df_[qusr.prediction_col].values

            n = df_.shape[0]
            skew_ = skew(y_true)
            zerofrac = np.sum(y_true == 0) / n
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            k = cohen_kappa_score(y_true, y_pred)

            self.df_tt.loc[idx] = [task.name, n, skew_, zerofrac, acc, f1, k]

        if verbose:
            ns = self.df_tt["N Responses"].values
            skews = self.df_tt["Score-Label-Skewness"].values
            zerofracs = self.df_tt["0-Fraction"].values
            ks = self.df_tt["Cohens Kappa"].values
            accs = self.df_tt["Accuracy"].values
            corr_analysis(ns, ks, "N Responses ~ Cohens Kappa")
            corr_analysis(ns, accs, "N Responses ~ Accuracy")
            corr_analysis(skews, ks, "Score-Label-Skewness ~ Cohens Kappa")
            corr_analysis(skews, accs, "Score-Label-Skewness ~ Accuracy")
            corr_analysis(skews, zerofracs, "Score-Label-Skewness ~ 0-Fraction")


    def get_taskwise_performace(
        self,
        qudata: QuData=None,
        quird: QuInterraterAnalysis=None,
    ) -> pd.DataFrame:
        """Get the taskwise performance of the scorer-model or interrater-data.

        Parameters
        ----------
        qudata : QuData
            The data to be used for additional analyses.
        quird : QuInterraterAnalysis
            The interrater-data to be used for additional analyses.

        Returns
        -------
        pd.DataFrame
            The taskwise performance of the scorer-model or interrater-data.
        """
        df_tt = self.df_tt.copy()
        tasks = df_tt["task"].to_list()

        if qudata is not None:
            df_txt = qudata.get_txt(units="tasks", table="long", )
            df_txt["Avg. Response Length"] = df_txt["text"].str.split().apply(len)
            df_counts = df_txt.groupby("task").agg({"Avg. Response Length": "mean"})
            df_counts = df_counts.reset_index()
            df_tt = df_tt.merge(df_counts[["task", "Avg. Response Length"]])

        if quird is not None:
            quta_ir = QuTextTaskAnalysis(quird)
            df_ir = quta_ir.df_tt.copy()
            df_ir = df_ir.rename(columns={
                "Accuracy": "IR-Accuracy",
                "F1": "IR-F1",
                "Cohens Kappa": "IR-Cohens Kappa",
            })
            df_ir = df_ir.drop(columns=["N Responses", "Score-Label-Skewness", "0-Fraction"])
            df_tt = df_tt.merge(df_ir)

        assert tasks == df_tt["task"].to_list(), "Tasks do not match after merges."

        return df_tt


    def plot_performance_dependencies(
        self,
        performance_measure: Literal["acc", "f1", "kappa"]="kappa",
        save_path: str=None,
        **kwargs
    ) -> plt.Figure:
        """Plot the performance of the scorer-model w. r. t. the number of responses and the skewness of the score-labels.

        Parameters
        ----------
        performance_measure : Literal["acc", "f1", "kappa"]
            The performance measure to be plotted.
        save_path : str
            The path to save the figure to.
        """
        ns = self.df_tt["N Responses"].values
        skews = self.df_tt["Score-Label-Skewness"].values
        if performance_measure == "acc":
            perf = self.df_tt["Accuracy"].values
            ylabel = "Accuracy"
        elif performance_measure == "f1":
            perf = self.df_tt["F1"].values
            ylabel = "F1-Score"
        elif performance_measure == "kappa":
            perf = self.df_tt["Cohens Kappa"].values
            ylabel = "Cohens $\\kappa$"

        ax1: plt.Axes
        ax2: plt.Axes

        fig, ax1 = plt.subplots(figsize=(8, 5))

        ylabel = kwargs.get("ylabel", ylabel)
        ax1.set_ylabel(ylabel, size=12)
        xlabel_top = kwargs.get("xlabel_top", '$N$ Responses')
        ax1.set_xlabel(xlabel_top, color="black", size=12)
        ax1.scatter(ns, perf, color="black")
        sns.lineplot(x=ns, y=perf, c="black")
        ax1.tick_params(axis='y', labelcolor="black")
        ax1.tick_params(axis="both", labelsize=11)

        ax2 = ax1.twiny()
        xlabel_bottom = kwargs.get("xlabel_bottom", "Skewness")
        ax2.set_xlabel(xlabel_bottom, color="tab:blue", size=12)
        ax2.scatter(skews, perf, color="tab:blue")
        sns.lineplot(x=skews, y=perf, c="tab:blue", ax=ax2)
        ax2.tick_params(axis='x', labelcolor="tab:blue")

        fig.tight_layout()
        title = kwargs.get("title", "Task Analysis")
        fig.suptitle(title, y=1.05)
        ax2.tick_params(axis="both", labelsize=11)

        legend_n_label = kwargs.get("legend_n_label", f"$N$-Responses vs. {ylabel}")
        h1 = mpatches.Patch(color='black', label=legend_n_label)
        legend_skew_label = kwargs.get("legend_skew_label", f"Score-Label-Skewness vs. {ylabel}")
        h2 = mpatches.Patch(color='tab:blue', label=legend_skew_label)
        legend_position = kwargs.get("legend_position", (0.075, -0.15))
        fig.legend(handles=[h1, h2], fontsize=12, bbox_to_anchor=legend_position, loc='lower left')

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        plt.show()
        return fig


    def task_wise_performance_correlation(
        self,
        other: "QuTextTaskAnalysis",
        fig_save_path: str=None,
        mode: Literal["spearman", "pearson"]="spearman",
        **kwargs,
    ) -> plt.Figure:
        """Plot the correlation between the performance of two scorer-models w. r. t. the tasks.

        Parameters
        ----------
        other : QuTextTaskAnalysis
            The other `QuTextTaskAnalysis` object to be compared.
        fig_save_path : str
            The path to save the figure to.
        mode : Literal["spearman", "pearson"]
            The correlation mode to be used.

        Returns
        -------
        plt.Figure
            The figure of the correlation analysis.
        """
        its = self.quconfig.get_text_columns("tasks")

        df1 = self.df_tt.copy()
        df2 = other.df_tt.copy()

        accs1 = df1["Accuracy"].copy()
        accs2 = df2["Accuracy"].copy()

        ks1 = df1["Cohens Kappa"].copy()
        ks2 = df2["Cohens Kappa"].copy()

        corr_analysis(accs1, accs2, "Accuracy", mode)
        corr_analysis(ks1, ks2, "Cohens Kappa",mode)


        ax1: plt.Axes
        ax2: plt.Axes

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        xlabel = kwargs.get("xlabel", "Human-Human")
        ylabel = kwargs.get("ylabel", "Human-Machine")

        ax1.set_title("Accuracy", fontsize=14)
        # ax1.set_xlim(0.4, 1)
        ax1.scatter(accs2, accs1, color="black")
        sns.regplot(x=accs2, y=accs1, color="black", ax=ax1)
        for it, x, y in zip(its, accs2, accs1):
            ax1.annotate(it, (x+0.005, y+0.005))
        ax1.set_xlabel(xlabel, size=12)
        ax1.set_ylabel(ylabel, color="black", size=12)
        ax1.grid()
        ax1.tick_params(axis="both", labelsize=11)

        ax2.set_title("Cohens $\\kappa$", fontsize=14)
        # ax2.set_xlim(0, 1)
        ax2.scatter(ks2, ks1, color="tab:blue")
        ax2.tick_params(axis="both", labelsize=11)
        sns.regplot(x=ks2, y=ks1, color="tab:blue", ax=ax2)
        # ax3.set_ylim(0, 1)
        ax2.set_xlabel(xlabel, size=12)
        ax2.set_ylabel(ylabel, size=12)
        for it, x, y in zip(its, ks2, ks1):
            ax2.annotate(it, (x+0.007, y+0.007))
        ax2.grid()
        ax2.tick_params(axis="both", labelsize=11)

        fig.tight_layout()
        title = kwargs.get("title", "Task-Analysis: Human-Human vs. Human-Machine")
        fig.suptitle(title, fontsize=16, y=1.05)

        if fig_save_path is not None:
            plt.savefig(fig_save_path, bbox_inches="tight", dpi=300)

        plt.show()
        return fig
