import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from typing import Literal
from abc import ABC, abstractmethod


from ..data.subscales import QuSubscales
from ..data.data import QuData
from ..scoring.scorer_results import QuScorerResults
from ..core import check_options

from .sr_regressor_results import QuRegressionResults



class QuRegressorBase(ABC):
    qusubs: list[QuSubscales]
    subscale_cols: list[str]

    def __init__(
        self,
        qusubscales: list[QuSubscales],
        append_total_score: bool=True,
        total_score_name: str="Total Score",
    ) -> None:
        self.quconfig = qusubscales[0].quconfig.copy()
        self.id_col = self.quconfig.id_col
        self.task_cols = self.quconfig.get_task_names()
        for qusub in qusubscales:
            if qusub.quconfig != self.quconfig:
                raise ValueError(
                    "The `quconfig` of the different `QuSubscales` objects is not equal."
                )

        if append_total_score:
            qc = self.quconfig.copy()
            n_tasks = qc.get_task_count()
            ones = np.ones(shape=(n_tasks, 1))
            df_ts = pd.DataFrame(qc.get_task_names(), columns=["Task"])
            df_ts[total_score_name] = pd.DataFrame(ones, dtype=int)
            qusub_ts = QuSubscales(quconfig=qc, df_cat=df_ts)
            qusubscales = [qusub_ts] + qusubscales

        self.qusubs = qusubscales

        self.subscale_cols = []
        for qs in self.qusubs:
            self.subscale_cols += qs.get_subscales()

    def summation_df_scr(self, df_scr: pd.DataFrame) -> pd.DataFrame:
        df_ret = df_scr[[self.quconfig.id_col]].copy()
        for qusub in self.qusubs:
            df_sub = qusub.apply_to_dataframe(df_scr=df_scr)
            df_ret = pd.merge(df_ret, df_sub, on="ID")
        return df_ret

    def predict_qudata(self, qudata: QuData) -> pd.DataFrame:
        df_scr = qudata.get_scr(mc_scored=True, verbose=False)
        return self.predict_df_scr(df_scr=df_scr)

    @abstractmethod
    def fit(self, df_trg: pd.DataFrame, df_prd: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def predict_df_scr(self, df_scr: pd.DataFrame) -> pd.DataFrame:
        pass



class QuRegressorSummation(QuRegressorBase):
    def fit(self, df_trg: pd.DataFrame, df_prd: pd.DataFrame) -> None:
        pass

    def reset(self) -> None:
        pass

    def predict_df_scr(self, df_scr: pd.DataFrame) -> pd.DataFrame:
        return self.summation_df_scr(df_scr=df_scr)



class QuRegressorLR(QuRegressorBase):
    kernels: list[LinearRegression] = []

    def fit(self, df_trg: pd.DataFrame, df_prd: pd.DataFrame) -> None:
        if not (df_prd[self.id_col] == df_trg[self.id_col]).all():
            raise ValueError(
                "The IDs of the passed score (~target)- and prediction-data are not equal."
            )
        df_trg = self.summation_df_scr(df_scr=df_trg)

        for col in self.subscale_cols:
            y = df_trg[col].values
            X = df_prd[self.task_cols].values
            lm = LinearRegression()
            lm.fit(X=X, y=y)
            self.kernels.append(lm)

    def reset(self) -> None:
        self.kernels: list[LinearRegression] = []

    def predict_df_scr(self, df_scr: pd.DataFrame) -> pd.DataFrame:
        df_ret = df_scr.copy()
        df_ret = df_ret[[self.id_col]]
        for idx, col in enumerate(self.subscale_cols):
            X = df_scr[self.task_cols].values
            lm = self.kernels[idx]
            y = lm.predict(X=X)
            df_ret[col] = y
        return df_ret







class QuScorerResultsRegressor:

    def __init__(
        self,
        qusubs: QuSubscales|list[QuSubscales],
        model: Literal["summation", "linear_regression"]|QuRegressorBase="summation",
        append_total_score: bool=True,
        total_score_name: str="Total Score",
    ) -> None:
        if isinstance(qusubs, QuSubscales):
            self.qusubs = [qusubs]
        elif isinstance(qusubs, list):
            if any([isinstance(qs, QuSubscales)==False for qs in qusubs]):
                raise TypeError(
                    "Some of the elements of the `qusubscales`-list is not of type QuSubscales."
                )
            self.qusubs = qusubs
        else:
            raise TypeError(
                "The passed `qusubscales` are not of the correct type `QuSubscales` or `list[QuSubscales]. \n" +
                f"Got `{type(qusubs)}`"
            )

        if isinstance(model, QuRegressorBase):
            self.model = model
        else:
            if model == "summation":
                self.model = QuRegressorSummation(
                    qusubscales=self.qusubs,
                    append_total_score=append_total_score,
                    total_score_name=total_score_name,
                )
            elif model == "linear_regression":
                self.model = QuRegressorLR(
                    qusubscales=self.qusubs,
                    append_total_score=append_total_score,
                    total_score_name=total_score_name,
                )
            check_options(
                arg=model,
                valid_opts=["summation", "linear_regression"],
                Ex=ValueError,
                arg_name=model,
            )
        self.qusubs = self.model.qusubs

    @staticmethod
    def __get_df_split(
        df: pd.DataFrame,
        split: int,
        mode: Literal["train", "test"],
    ) -> pd.DataFrame:
        df = df.copy()
        df = df[df["split"] == split]
        df = df[df["mode"] == mode]
        df = df.drop(columns=["split", "mode"])
        df = df.reset_index(drop=True)
        return df


    def __append_split_info(
        self,
        df: pd.DataFrame,
        split: int,
        mode: Literal["train", "test"],
        id_col: str="ID",
    ) -> pd.DataFrame:
        cols = self.model.subscale_cols
        df["mode"] = mode
        df["split"] = split
        df = df[[id_col, "split", "mode"] + cols]
        return df


    def fit(self, qusr: QuScorerResults) -> QuRegressionResults:
        df_prd = qusr.get_wide_predictions(mc_units="mc_tasks")
        df_trg = qusr.get_wide_targets(mc_units="mc_tasks")

        split_nos = qusr.get_split_nos()
        tst_data_av = qusr._has_test_data()

        df_tst_pred_list = []
        df_trn_pred_list = []

        for idx in split_nos:
            self.model.reset()

            df_prd_idx = self.__get_df_split(df=df_prd, split=idx, mode="train")
            df_trg_idx = self.__get_df_split(df=df_trg, split=idx, mode="train")

            self.model.fit(df_trg=df_prd_idx, df_prd=df_trg_idx)

            df_trn_pred = self.model.predict_df_scr(df_scr=df_prd_idx)
            df_trn_pred = self.__append_split_info(
                df=df_trn_pred,
                split=idx,
                mode="train",
                id_col=qusr.id_col,
            )
            df_trn_pred_list.append(df_trn_pred)

            if tst_data_av:
                df_tst = self.__get_df_split(df=df_prd, split=idx, mode="test")
                df_tst_pred = self.model.predict_df_scr(df_scr=df_tst)
                df_tst_pred = self.__append_split_info(
                    df=df_tst_pred,
                    split=idx,
                    mode="test",
                    id_col=qusr.id_col,
                )
                df_tst_pred_list.append(df_tst_pred)

        df_preds = pd.concat(
            df_trn_pred_list + df_tst_pred_list,
            axis=0,
        )
        df_preds = df_preds.sort_values(by=[qusr.id_col, "split", "mode"])
        df_preds = df_preds.reset_index(drop=True)

        df_targets = df_trg[df_trg["split"]==idx].copy()
        df_targets = df_targets.drop(columns=["split", "mode"])
        df_targets = df_targets.sort_values(by=[qusr.id_col])
        df_targets = df_targets.reset_index(drop=True)
        df_targets = self.model.summation_df_scr(df_scr=df_targets)

        return QuRegressionResults(
            quconfig=qusr.quconfig,
            df_preds=df_preds,
            df_targets=df_targets,
            qusubs=self.qusubs
        )
