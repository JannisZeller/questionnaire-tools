import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from typing import Literal
from tqdm import tqdm

from ..data.data import QuData
# from ..data.interrater_data import QuInterraterData

from ..clustering.clusters import QuScoreClusters

from ..scoring.scorer_results import QuScorerResults
from .sr_classifier_results import QuSRClassifierResults

from ..core.classifier import Classifier, ScikitClassifier
from ..core.trainulation import Split
from ..core.trainulation import get_random_oversampler, training_core




class QuSRClassifierError(Exception):
    """An Exception class for the QuScoreClassifier"""


class QuScorerResultsClassifier:
    def __init__(
        self,
        model: Classifier=None,
        target_name: str="cluster",
        feature_names: list=None,
        omit_classes: list=None,
        **kwargs,
    ) -> None:
        self.target = target_name
        self.pred = target_name + "_pred"
        self.features = feature_names

        self.omit_classes = omit_classes
        self.mc_units = kwargs.get("mc_units", "mc_tasks")

        self.class_order = None

        if model is None:
            model = ScikitClassifier(LogisticRegression(C=0.5, max_iter=500))
        self.clf_model = model


    ## Validation
    def __validate_quconfigs(
        self,
        qu1: QuScorerResults|QuData|QuScoreClusters,
        qu2: QuScorerResults|QuData|QuScoreClusters,
    ) -> None:
        if qu1.quconfig != qu2.quconfig:
            raise QuSRClassifierError(
                f"The QuConfig of the passed {type(qu1)} and the QuConfig of the passed {type(qu2)} are not identical."
            )

    def __get_df_scr_val(
        self,
        qusr: QuScorerResults,
        full_train_fit: bool=False
    ) -> pd.DataFrame:
        df_scr_val = qusr.get_wide_predictions(mc_units=self.mc_units)
        if full_train_fit:
            df_scr_val = df_scr_val[df_scr_val["mode"]=="train"]
        else:
            df_scr_val = df_scr_val[df_scr_val["mode"]=="test"]
        df_scr_val = df_scr_val.drop(columns=["mode", "split"])
        df_scr_val = df_scr_val.reset_index(drop=True)
        return df_scr_val

    def __set_columns(
        self,
        df_scr: pd.DataFrame,
        df_trg: pd.DataFrame,
        target_name: str,
        feature_names: list,
    ) -> None:
        if feature_names is None:
            feature_names = df_scr.drop(columns=self.id_col).columns.to_list()
            print(f"No `feature_names` passed. Defaulting to ALL columns of `df_scr` apart from the ID-column \"{self.id_col}\".")

        if not target_name in df_trg.columns:
            raise ValueError(
                f"The passed `target_name` \"{target_name}\" is not present as a column in `df_trg`."
            )
        for fn in feature_names:
            if not fn in df_scr.columns:
                raise ValueError(
                    f"The feature name \"{fn}\" in the `feature_names` is not present as a column in `df_scr`."
                )
        self.target = target_name
        self.features = feature_names

    def __validate_data(
        self,
        df_scr: pd.DataFrame,
        df_trg: pd.DataFrame,
        id_match_mode: Literal["info", "warn", "error"]
    ) -> None:
        self.quconfig._validate_id_col(
            df=df_trg,
            err_str=(
                f"The passed target-data does not contain the id column \"{self.id_col}\"\n" +
                "that is specified in the passed `QuConfig`."
            ),
        )
        self.quconfig.validate_wide_df_scr(
            df_scr=df_scr,
            units=self.mc_units,
            all_units=False,
            id_err_str=(
                f"The passed score-data does not contain the id column \"{self.id_col}\"\n" +
                "that is specified in the passed `QuConfig`."
            ),
        )
        self.quconfig._check_id_matches(
            df1=df_scr,
            df2=df_trg,
            names=["score", "target"],
            mode=id_match_mode,
        )

    def __omit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        n_splits = len(df["split"].unique())
        n = df.shape[0]
        df = df.dropna()
        n = n - df.shape[0]
        if n > 0:
            print(f"Info: Dropping {n} rows / {int(n/n_splits)} edits due to missing target.")

        if self.omit_classes is not None:
            n = df.shape[0]
            df = df[df[self.target].isin(self.omit_classes)==False]
            n = n - df.shape[0]
            if n > 0:
                print(f"Info: Dropping {n} rows / {int(n/n_splits)} edits due to omit-classes.")

        df = df.reset_index(drop=True)
        return df

    def __teacher_forcing(
        self,
        df: pd.DataFrame,
        df_trg: pd.DataFrame,
        qusr: QuScorerResults,
        teacher_force_prop: float=0,
    ) -> pd.DataFrame:
        if teacher_force_prop > 0:
            if teacher_force_prop > 1:
                raise ValueError("Using more than a factor of 1 for teacher forcing is not suggested.")
            df_scr = qusr.get_wide_targets(mc_units=self.mc_units)
            df_scr = df_scr[df_scr["mode"]=="train"]
            df_scr = df_scr.sample(frac=teacher_force_prop, replace=True)
            df_scr = df_scr.reset_index(drop=True)
            df_scr = pd.merge(df_scr, df_trg[[self.id_col, self.target]], how="left", validate="m:1")
            df_scr = df_scr.dropna()
            if self.omit_classes is not None:
                df_scr = df_scr[df_scr[self.target].isin(self.omit_classes)==False]
            df = pd.concat([df, df_scr], axis=0)
        return df

    def __set_class_label_dicts(self, df: pd.DataFrame):
        unique_targets = df[self.target].unique()
        self.target_to_int = dict(zip(unique_targets, range(len(unique_targets))))
        self.int_to_target = {v: k for k, v in self.target_to_int.items()}

    def __targets_to_int(self, df: pd.DataFrame, no_pred: bool=False) -> pd.DataFrame:
        try:
            with pd.option_context("future.no_silent_downcasting", True):
                df[self.target] = df[self.target].replace(self.target_to_int).infer_objects(copy=False)
                if self.pred in df.columns and not no_pred:
                    df[self.pred] = df[self.pred].replace(self.target_to_int).infer_objects(copy=False)
        except:
            df[self.target] = df[self.target].replace(self.target_to_int)
            if self.pred in df.columns and not no_pred:
                df[self.pred] = df[self.pred].replace(self.target_to_int)
        return df

    def __targets_to_labels(self, df: pd.DataFrame, no_pred: bool=False) -> pd.DataFrame:
        df[self.target] = df[self.target].replace(self.int_to_target)
        if self.pred in df.columns and not no_pred:
            df[self.pred] = df[self.pred].replace(self.int_to_target)
        return df


    def _get_data(
        self,
        qusr: QuScorerResults,
        df_trg: pd.DataFrame,
        teacher_force_prop: float=0,
        include_mc_only: bool=True,
    ) -> pd.DataFrame:
        id_col = qusr.quconfig.id_col
        df_scr = qusr.get_wide_predictions(
            mc_units=self.mc_units,
            append_mc_only=include_mc_only,
        )
        df_scr = df_scr[[id_col, "mode", "split"] + self.features]
        df_scr = df_scr.fillna(0)
        df_trg = df_trg[[id_col, self.target]]
        df = pd.merge(df_scr, df_trg, how="left", validate="m:1")
        df = self.__omit_data(df=df)
        df = self.__teacher_forcing(
            df=df,
            df_trg=df_trg,
            qusr=qusr,
            teacher_force_prop=teacher_force_prop,
        )

        self.__set_class_label_dicts(df=df)
        df = self.__targets_to_int(df=df)

        return df

    def _validate_and_get_data(
        self,
        qusr: QuScorerResults,
        targets: pd.DataFrame|QuScoreClusters,
        id_match_mode: Literal["error", "warn", "info"],
        teacher_force_prop: float=0,
        include_mc_only: bool=True,
        full_train_fit: bool=False,
        cluster_mode: Literal["clusters", "clusters_all", "clusters_most"]="clusters_most",
    ) -> pd.DataFrame:
        self.quconfig = qusr.quconfig
        self.id_col = qusr.quconfig.id_col

        if isinstance(targets, QuScoreClusters):
            self.class_order = targets.get_cluster_order()
            if self.omit_classes is not None:
                self.class_order = [x for x in self.class_order if x not in self.omit_classes]
            if cluster_mode == "clusters":
                df_trg = targets.clusters()
            elif cluster_mode == "clusters_all":
                df_trg = targets.clusters_all()
            else:
                df_trg = targets.clusters_most(qudata=qusr.qudata)
        else:
            df_trg = targets

        df_scr_val = self.__get_df_scr_val(
            qusr=qusr,
            full_train_fit=full_train_fit
        )
        self.__set_columns(
            df_scr=df_scr_val,
            df_trg=df_trg,
            target_name=self.target,
            feature_names=self.features,
        )

        self.__validate_data(
            df_scr=df_scr_val,
            df_trg=df_trg,
            id_match_mode=id_match_mode,
        )

        df = self._get_data(
            qusr=qusr,
            df_trg=df_trg,
            teacher_force_prop=teacher_force_prop,
            include_mc_only=include_mc_only,
        )

        del self.quconfig
        del self.id_col

        return df

    @staticmethod
    def _get_train_only_mode(qusr: QuScorerResults) -> bool:
        if qusr.get_test_n() == 0:
            print("Got a `QuScorerFullFitResults` object to fit with. No test-evluation will be available.")
            return True
        return False


    ## Fit / Evaluate
    def fit(
        self,
        qusr: QuScorerResults,
        targets: pd.DataFrame|QuScoreClusters,
        teacher_force_prop: float=0, # TODO:
        oversample: bool=False,
        include_mc_only: bool=True,
        **kwargs,
    ) -> QuSRClassifierResults:
        full_train_fit = QuScorerResultsClassifier._get_train_only_mode(qusr=qusr)
        if oversample:
            print("Using oversampling.")

        if isinstance(targets, QuScoreClusters):
            self.__validate_quconfigs(qu1=qusr, qu2=targets)
        df = self._validate_and_get_data(
            qusr=qusr,
            targets=targets,
            id_match_mode=kwargs.get("id_match_mode", "info"),
            teacher_force_prop=teacher_force_prop,
            include_mc_only=include_mc_only,
            full_train_fit=full_train_fit,
            cluster_mode=kwargs.get("cluster_mode", "clusters_most"),
        )

        quconfig = qusr.quconfig

        quclfr = QuSRClassifierResults(
            quconfig=quconfig,
            target_col=self.target,
            prediction_col=self.pred,
            class_order=self.class_order,
        )
        self.cv_model_copys: list[Classifier] = []

        for _, df_ in tqdm(df.groupby("split")):
            if oversample:
                oversampler = get_random_oversampler()
            else:
                oversampler = None

            split = Split.from_mode_col(
                df=df_.copy(),
                assert_test_data=full_train_fit==False,
            )

            df_pred_trn, df_pred_tst, fit_history, model_copy = training_core(
                model=self.clf_model,
                split=split,
                feature_cols=self.features,
                target_col=self.target,
                prediction_col=self.pred,
                oversampler=oversampler
            )

            if full_train_fit:
                del fit_history["eval_loss"]
                del fit_history["eval_accuracy"]
                del fit_history["eval_f1"]

            df_pred_trn = self.__targets_to_labels(df_pred_trn)
            if not full_train_fit:
                df_pred_tst = self.__targets_to_labels(df_pred_tst)

            quclfr.append(
                df_trn=df_pred_trn,
                df_tst=df_pred_tst,
                fit_history=fit_history,
            )
            self.cv_model_copys.append(model_copy)

        return quclfr

    def evaluate_no_fit(
        self,
        models: list[Classifier]|Classifier,
        qusr: QuScorerResults,
        targets: pd.DataFrame|QuScoreClusters,
        include_mc_only: bool=True,
        **kwargs,
    ) -> QuSRClassifierResults:
        full_train_fit = QuScorerResultsClassifier._get_train_only_mode(qusr=qusr)

        if not isinstance(models, list):
            print("Warning: The CV might not be interpretable properly if only a single model is used. Are you sure that there is not data leakage?")
            models = qusr.get_n_splits() * [models]

        if isinstance(targets, QuScoreClusters):
            self.__validate_quconfigs(qu1=qusr, qu2=targets)
        df = self._validate_and_get_data(
            qusr=qusr,
            targets=targets,
            id_match_mode=kwargs.get("id_match_mode", "info"),
            include_mc_only=include_mc_only,
            full_train_fit=full_train_fit,
            cluster_mode=kwargs.get("cluster_mode", "clusters_most"),
        )
        quconfig = qusr.quconfig

        quclfr = QuSRClassifierResults(
            quconfig=quconfig,
            target_col=self.target,
            prediction_col=self.pred,
            class_order=self.class_order,
        )

        df["split"] = df["split"].astype(int)

        for idx, df_ in tqdm(df.groupby("split")):
            model: Classifier = models[idx - 1]
            df_trn = df_[df_["mode"]=="train"].copy()
            df_tst = df_[df_["mode"]=="test"].copy()

            if df_trn.shape[0] == 0 and df_tst.shape[0] == 0:
                raise QuSRClassifierError(f"Neither test- nor train-data available for the current split ({idx}).")

            if df_trn.shape[0] != 0:
                y_pred_trn = model.predict(df_trn[self.features].values)
                df_trn[self.pred] = y_pred_trn
                df_trn = self.__targets_to_labels(df=df_trn, no_pred=True)
            else:
                df_trn = None

            if not full_train_fit and df_tst.shape[0] != 0:
                y_pred_tst = model.predict(df_tst[self.features].values)
                df_tst[self.pred] = y_pred_tst
                df_tst = self.__targets_to_labels(df=df_tst, no_pred=True)
            else:
                df_tst = None

            quclfr.append(df_trn=df_trn, df_tst=df_tst, fit_history=None)

        return quclfr
