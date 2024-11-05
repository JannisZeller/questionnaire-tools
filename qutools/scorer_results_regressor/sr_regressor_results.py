import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from typing import Literal


from ..data.config import QuConfig
from ..data.subscales import QuSubscales
from ..core import check_options, LinearModel



def jitter(x: np.ndarray, n_sds: float=0.05) -> np.ndarray:
    sd = np.std(x)
    noise_sd = n_sds * sd
    # noise = np.random.normal(loc=0, scale=noise_sd, size=x.shape)
    # noise = np.clip(noise, -noise_sd, +noise_sd)
    noise = np.random.uniform(low=-noise_sd, high=noise_sd, size=x.shape)
    return x + noise

def jitter_cols(
    df: pd.DataFrame,
    cols: list[str]|None=None,
    n_sds: float=0.05,
) -> pd.DataFrame:
    if cols is None:
        cols = df.drop(columns="ID").columns.to_list()
    for col in cols:
        df[col + "_jit"] = jitter(df[col].values, n_sds=n_sds)
    return df


class QuRegressionResults:

    def __init__(
        self,
        quconfig: QuConfig,
        qusubs: list[QuSubscales],
        df_targets: pd.DataFrame=None,
        df_preds: pd.DataFrame=None,
    ) -> None:
        self.id_col = quconfig.id_col
        self.quconfig = quconfig

        self.df_preds = df_preds
        self.df_targets = df_targets

        self.qusubs = qusubs
        self.subscale_cols = []
        for qs in self.qusubs:
            self.subscale_cols += qs.get_subscales()


    def get_long_df(
            self,
            mode: Literal["train", "test", "all"]="test",
            add_jitter: bool=False,
        ) -> pd.DataFrame:
        check_options(mode, ["train", "test", "all"], ValueError, "mode")

        df_p = self.df_preds.copy()
        if mode != "all":
            df_p = df_p[df_p["mode"]==mode]
        df_p = df_p.drop(columns=["mode", "split"])
        df_p = df_p.melt(id_vars=self.id_col, var_name="dim", value_name="score_pred")

        df_t = self.df_targets.copy()
        df_t = df_t.melt(id_vars=self.id_col, var_name="dim", value_name="score_true")

        df = pd.merge(df_p, df_t, on=[self.id_col, "dim"], how="left", validate="m:1")

        if add_jitter:
            df = jitter_cols(df=df, cols=["score_pred", "score_true"], n_sds=0.1)

        return df


    def _get_hue_order(self) -> dict:
        n_dims = len(self.subscale_cols)
        if n_dims > 10:
            hue_order = dict(zip(
                self.subscale_cols,
                sns.color_palette("rocket", n_colors=n_dims)
            ))
        else:
            hue_order = dict(zip(
                self.subscale_cols,
                sns.color_palette(n_colors=n_dims)
            ))
        return hue_order


    def get_qusubs_max_scores(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for qs in self.qusubs:
            ms = qs.get_max_scores()
            for key, val in ms.items():
                df[key] = [val]
        return df


    def correlation_analysis(
            self,
            mode: Literal["train", "test", "all"]="test",
        ) -> pd.DataFrame:
        df = self.get_long_df(mode=mode, add_jitter=False)

        ssc: str
        df_show = pd.DataFrame(index=["Pearson r", "p-value"])
        for ssc in self.subscale_cols:
            df_ = df[df["dim"]==ssc].copy()
            x = df_["score_pred"].values
            y = df_["score_true"].values
            r, p = stats.pearsonr(x, y)
            df_show[ssc] = [r, p]

        print(f"N-{mode.capitalize()} = {len(x)}")

        return df_show


    def regression_analysis(
            self,
            mode: Literal["train", "test", "all"]="test",
            verbose: bool=False,
        ) -> tuple[pd.DataFrame, list[LinearModel]]:
        df = self.get_long_df(mode=mode, add_jitter=False)

        ssc: str
        df_show = pd.DataFrame(index=["Rsq", "slope"])

        lms = []
        for ssc in self.subscale_cols:
            df_ = df[df["dim"]==ssc].copy()
            lm = LinearModel()
            lm.fit(df_, features=["score_pred"], target="score_true", verbose=verbose)
            df_show[ssc] = [lm.stats["rsq"], lm.coef[1]]
            lms.append(lm)

        print(f"N-{mode.capitalize()} = {df_.shape[0]}")

        return df_show, lms


    def full_analysis(
        self,
        mode: Literal["train", "test", "all"]="test",
        verbose: bool=False,
    ) -> pd.DataFrame:
        df_cor = self.correlation_analysis(mode=mode)
        df_reg, _ = self.regression_analysis(mode=mode, verbose=verbose)

        return pd.concat([df_cor, df_reg], axis=0)


    def regplots(
            self,
            mode: Literal["train", "test", "all"]="test",
            add_jitter: bool=True,
            save_path: str=None,
            **kwargs,
        ) -> sns.FacetGrid:
        df_plot = self.get_long_df(mode=mode, add_jitter=add_jitter)
        hue_order = self._get_hue_order()

        name_true = kwargs.get("ylabel", "Score Target")
        name_pred = kwargs.get("xlabel", "Score Prediction")
        df_plot = df_plot.rename(columns={
            'score_true': name_true,
            'score_pred': name_pred,
        })

        g = sns.FacetGrid(
            df_plot,
            col="dim", col_wrap=kwargs.get("col_wrap", 5), col_order=self.subscale_cols,
            hue="dim", palette=hue_order,
            sharex=False, sharey=False,
        )
        _ = g.map(sns.regplot, name_pred, name_true, marker="", color="k")
        if add_jitter:
            _ = g.map(sns.scatterplot,"score_pred_jit", "score_true_jit", alpha=0.1)
        _ = g.map(sns.kdeplot, name_pred, name_true, alpha=0.4, bw_adjust=1.5, fill=True)
        g.set_titles(col_template="{col_name}")

        if save_path is not None:
            g.savefig(save_path, bbox_inches="tight", dpi=150)

        return g


    def violinplots(
            self,
            mode: Literal["train", "test", "all"]="test",
            save_path: str=None,
            **kwargs,
        ) -> plt.Figure:
        df_plot = self.get_long_df(mode=mode, add_jitter=False)
        hue_order = self._get_hue_order()
        sccs = self.subscale_cols

        self.qusubs[0].get_max_scores()

        name_true = 'score_true'
        name_pred = 'score_pred'

        df_diffs = df_plot.drop(columns=[name_true, name_pred]).copy()
        df_diffs["Score Delta"] = df_plot[name_pred].values - df_plot[name_true].values
        df_diffs = df_diffs.pivot(index="ID", columns="dim", values="Score Delta")
        df_diffs = df_diffs.reset_index()
        df_diffs = df_diffs[["ID"] + sccs]

        df_ms = self.get_qusubs_max_scores()
        df_ms = df_ms[sccs]
        df_diffs[sccs] = df_diffs[sccs] / df_ms.values

        var_name = kwargs.get("xticks", "Dimension")
        value_name = kwargs.get("ylabel", "Devience relative to maximum Score")
        df_diffs = df_diffs.melt(id_vars=self.id_col, var_name=var_name, value_name=value_name)

        fig = plt.figure(figsize=(14, 7))
        _ = sns.violinplot(
            df_diffs, x=var_name, y=value_name,
            hue=var_name, palette=hue_order,
        )
        _ = plt.legend([],[], frameon=False)
        _ = plt.xticks(rotation=15)
        _ = plt.xlabel("")

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)

        return fig
