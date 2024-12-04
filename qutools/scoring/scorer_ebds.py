"""
# Embeddings-Scorer

This module provides the functionality to score the tasks a questionnaire using
only the precomputed embeddings of the responses.

Results of evaluation workflows
etc. are always provided as [`QuScorerResults`][qutools.scoring.scorer_results] objects.
These provide methods for the evaluation and visualization of the results
themselves and can also be used to easily save the results to disk.

Active Development!
-------------------
To ease the behavior and usage of the taskwise-finetuning, the `QuFinetuneScorer`-class
and the `QuEmbeddingScorer`-class will/should probably get a rework in the future. This
will contain a separate `TaskWiseScorer` class, that consumes and uses the
`QuScorer`-classes for easier management of taskwise scoring.

Usage
-----
The usage will be shown in a future example notebook.
"""


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from json import dump, load
from pathlib import Path
from tqdm import tqdm
from typing import Literal
from shutil import rmtree


from ..core.classifier import Classifier, ScikitClassifier, PyTorchClassifier
from ..core.trainulation import Split, get_random_oversampler, training_core
from ..core.io import path_suffix_warning

from ..data.config import QuConfig
from ..data.data import QuData
from ..id_splits import IDsSplit, IDsKFoldSplit, IDsTrainTestSplit
from ..embeddings.embeddings import QuEmbeddings, EmbeddingModel, SentenceTransformersEmbdModel

from ..clustering.clusters import QuScoreClusters

from .scorer_base import QuScorer, QuScorerError
from .scorer_results import QuScorerResults




class QuEmbeddingScorer(QuScorer):
    def __init__(
        self,
        scoring_model: Classifier=None,
        ebd_model: EmbeddingModel=None,
        item_task_prefix: bool=True,
        it_repl_dict: dict[str, str]={"A": "Aufgabe "},
        sep: str=" [SEP] ",
        verbose: bool=True,
        **kwargs,
    ) -> None:
        """Scorer for the tasks of a questionnaire based on the precomputed
        embeddings of the responses. The embeddings can be generated using
        the `EmbeddingModel`-objects.

        Parameters
        ----------
        scoring_model : Classifier
            The classifier used to predict the scores of the tasks. The default
            model is a scikit-learn logistic regression with `C=0.5` and
            `max_iter=1000`.
        ebd_model : EmbeddingModel
            The model used to generate the embeddings of the responses. The model
            is not needed to be passed to the `QuEmbeddingScorer`-object if the
            embeddings are already available and get passed to the fit-procedure-
            and prediction-methods as `QuEmbeddings`-objects. It is however
            needed to generate the embeddings from `QuData`-objects, i.e.,
            especially for new and unseen data.
        item_task_prefix : bool
            Wether the response texts should be prefixed with a certrain string
            representing or containing some information on the respective item/task.
            (using the `it_repl_dict`). Also refer to the [embedding_models][qutools.embeddings.embedding_models]
            submodule. This is only effective if `QuData` objects are used
            instead of `QuEmbeddings` objects.
        it_repl_dict : dict[str, str]
            The dictionary to replace patterns of the item/task name for prefixing.
        sep : str
            The separator string to use for the item/task prefixing.
        verbose : bool
            Wether to print information during the training and prediction process.
        **kwargs
            Additional keyword arguments. The following are available:
            - `is_trained` : bool
                Wether the model is already trained. Default: `False`.
            - `target` : str
                The column-name of the target variable. Default: `"score"`.
            These are mainly used for the internal handling when loading and saving
            the model.
        """
        if scoring_model is None:
            scoring_kernel = LogisticRegression(C=0.5, max_iter=1000)
            scoring_model = ScikitClassifier(scoring_kernel)

        self.scr_model = scoring_model
        self.ebd_model = ebd_model
        self.item_task_prefix = item_task_prefix
        self.it_repl_dict = it_repl_dict
        self.sep = sep
        self.verbose = verbose
        self.is_trained = kwargs.get("is_trained", False)
        self.cv_model_copys: list[Classifier] = []

        self.target = kwargs.get("target", "score")
        self.pred = self.target + "_pred"


    ## (Cross) Validation
    def fixed_eval(
        self,
        id_split: IDsSplit,
        quebds: QuEmbeddings=None,
        qudata: QuData=None,
        oversample: bool=False,
        validate_scores: Literal["none", "warn", "error"]="none",
        save_path: str=None,
        supress_not_saving_warning: bool=False,
    ) -> QuScorerResults:
        """Wrapper to evaluate a model based on the evaluation-`IDsSplit`-object
        passed.

        Parameters
        ----------
        id_split : IDsKFoldSplit
            The split-object to use for the evaluation denoting the train- and test-IDs.
        quebds : QuestionnaireEmbeddings
            The embeddings of the questionnaire-responses. If not passed, the embeddings
            are generated from the `qudata`-object, i.e., in this case the `qudata`-object
            must be passed.
        qudata : QuestionnaireData
            The data of the questionnaire-responses. Must be passed to generate the
            embeddings if no `quebds`-object is passed.
        oversample : bool
            Wether to use oversampling during the training.
        validate_scores : Literal["none", "warn", "error"]
            Wether to validate the scores of the passed `qudata`-object. If set to
            `"warn"` a warning is issued if the scores are not valid. If set to
            `"error"` an error is raised if the scores are not valid.
        save_path : str
            The path to save the results of the evaluation. If not passed, the results
            are not stored, which is not advised.
        supress_not_saving_warning : bool
            Wether to suppress the warning if no `save_path` is passed.

        Returns
        -------
        QuScorerResults
        """
        self.scr_model.reset()
        self.id_split = id_split
        quconfig = self._get_quconfig(quebds=quebds, qudata=qudata)
        id_col = quconfig.id_col

        df_ebd: pd.DataFrame
        df_ebd, _ = self._get_df_trn(qudata=qudata, quebds=quebds)
        feature_cols = df_ebd.columns.str.startswith("dim-")
        feature_cols = df_ebd.columns[feature_cols].to_list()

        quebds, qudata = QuEmbeddingScorer.__manage_quargs(
            qudata=qudata,
            quebds=quebds,
        )
        qusr = QuScorerResults(
            qudata=qudata,
            id_split=id_split,
            target_col=self.target,
            prediction_col=self.pred,
            validate_scores=validate_scores,
        )

        if oversample:
            print("Using oversampling.")

        test_ids_list = id_split.get_tst_ids_lists()

        for tst_ids in tqdm(test_ids_list):
            split = Split.from_tst_ids(df_ebd, tst_ids, id_col)
            if oversample:
                oversampler = get_random_oversampler()
            else:
                oversampler = None
            df_pred_trn, df_pred_tst, fit_history, model_copy = training_core(
                model=self.scr_model,
                split=split,
                feature_cols=feature_cols,
                target_col=self.target,
                prediction_col=self.pred,
                oversampler=oversampler,
            )
            qusr.append(df_trn=df_pred_trn, df_tst=df_pred_tst, fit_history=fit_history)
            self.cv_model_copys.append(model_copy)

        if save_path is None:
            if not supress_not_saving_warning:
                print(
                    "Warning: Not storing the scoring-cv results. It is suggested to set the `save_path`-argument. \n" +
                    "You can also call the `to_dir` method on the resulting `QuScorerCVResults`."
                )
        else:
            qusr.to_dir(path=save_path)

        return qusr

    def random_cross_validate(
        self,
        qudata: QuData=None,
        quclst: QuScoreClusters=None,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
        quebds: QuEmbeddings=None,
        oversample: bool=False,
        stratify: bool=True,
        validate_scores: Literal["none", "warn", "error"]="none",
        n_splits: int=10,
        random_state: int=42,
        save_path: str=None,
        supress_id_split_info: bool=False,
        supress_not_saving_warning: bool=False,
    ) -> QuScorerResults:
        """Wrapper to perform an automatic cross-validation on the data. The
        `IDsKFoldSplit`-object is generated from the passed `QuData`-object and
        the specified parameters.

        Parameters
        ----------
        qudata : QuestionnaireData
            The data of the questionnaire-responses. Must be passed to generate the
            embeddings if no `quebds`-object is passed.
        quclst : QuScoreClusters
            A cluster model to use for stratification when setting up the
            `IDsKFoldSplit`-object.
        df_strat : pd.DataFrame
            Alternatively to the `quclst`-object: A dataframe can be passed to
            to use for the stratification. The dataframe must contain the id-column
            and the column specified in `strat_col`.
        strat_col : str
            The column to use for the stratification when using the `df_strat`-argument.
        quebds : QuestionnaireEmbeddings
            The embeddings of the questionnaire-responses. If not passed, the embeddings
            are generated from the `qudata`-object, i.e., in this case the `qudata`-object
            must be passed.
        oversample : bool
            Wether to use oversampling during the training.
        stratify : bool
            Wether to stratify the splits based on the scores of the tasks.
        validate_scores : Literal["none", "warn", "error"]
            Wether to validate the scores of the passed `qudata`-object. If set to
            `"warn"` a warning is issued if the scores are not valid. If set to
            `"error"` an error is raised if the scores are not valid.
        n_splits : int
            The number of splits to use for the cross-validation.
        random_state : int
            The random state to use for the cross-validation splits.
        save_path : str
            The path to save the results of the evaluation. If not passed, the results
            are not stored, which is not advised.
        supress_id_split_info : bool
            Wether to suppress the information-output on the (newly generated) ID-split.
        supress_not_saving_warning : bool
            Wether to suppress the warning if no `save_path` is passed.

        Returns
        -------
        QuScorerResults
        """
        quebds, qudata = QuEmbeddingScorer.__manage_quargs(
            qudata=qudata,
            quclst=quclst,
            quebds=quebds,
        )
        id_split = IDsKFoldSplit.from_qudata(
            qudata=qudata,
            quclst=quclst,
            df_strat=df_strat,
            strat_col=strat_col,
            stratify=stratify,
            n_splits=n_splits,
            random_state=random_state,
            verbose=supress_id_split_info==False,
        )
        if quebds is not None:
            qudata = None
        qusr = self.fixed_eval(
            id_split=id_split,
            quebds=quebds,
            qudata=qudata,
            oversample=oversample,
            validate_scores=validate_scores,
            save_path=save_path,
            supress_not_saving_warning=supress_not_saving_warning,
        )
        return qusr

    def random_train_testing(
        self,
        qudata: QuData=None,
        quclst: QuScoreClusters=None,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
        quebds: QuEmbeddings=None,
        oversample: bool=False,
        stratify: bool=True,
        validate_scores: Literal["none", "warn", "error"]="none",
        test_size: float=0.2,
        random_state: int=42,
        save_path: str=None,
        supress_id_split_info: bool=False,
        supress_not_saving_warning: bool=False,
    ) -> QuScorerResults:
        """Wrapper to perform an automatic evaluation using a simple train-test split. The
        `IDsTrainTestSplit`-object is generated from the passed `QuData`-object and
        the specified parameters.

        Parameters
        ----------
        qudata : QuestionnaireData
            The data of the questionnaire-responses. Must be passed to generate the
            embeddings if no `quebds`-object is passed.
        quclst : QuScoreClusters
            A cluster model to use for stratification when setting up the
            `IDsTrainTestSplit`-object.
        df_strat : pd.DataFrame
            Alternatively to the `quclst`-object: A dataframe can be passed to
            to use for the stratification. The dataframe must contain the id-column
            and the column specified in `strat_col`.
        strat_col : str
            The column to use for the stratification when using the `df_strat`-argument.
        quebds : QuestionnaireEmbeddings
            The embeddings of the questionnaire-responses. If not passed, the embeddings
            are generated from the `qudata`-object, i.e., in this case the `qudata`-object
            must be passed.
        oversample : bool
            Wether to use oversampling during the training.
        stratify : bool
            Wether to stratify the splits based on the scores of the tasks.
        validate_scores : Literal["none", "warn", "error"]
            Wether to validate the scores of the passed `qudata`-object. If set to
            `"warn"` a warning is issued if the scores are not valid. If set to
            `"error"` an error is raised if the scores are not valid.
        test_size: float=0.2
            The (relative) size of the test-dataset to use for the train-test-split.
        random_state : int
            The random state to use for the cross-validation splits.
        save_path : str
            The path to save the results of the evaluation. If not passed, the results
            are not stored, which is not advised.
        supress_id_split_info : bool
            Wether to suppress the information-output on the (newly generated) ID-split.
        supress_not_saving_warning : bool
            Wether to suppress the warning if no `save_path` is passed.

        Returns
        -------
        QuScorerResults
        """
        quebds, qudata = QuEmbeddingScorer.__manage_quargs(
            qudata=qudata,
            quclst=quclst,
            quebds=quebds,
        )
        id_split = IDsTrainTestSplit.from_qudata(
            qudata=qudata,
            quclst=quclst,
            df_strat=df_strat,
            strat_col=strat_col,
            stratify=stratify,
            test_size=test_size,
            random_state=random_state,
            verbose=supress_id_split_info==False,
        )
        if quebds is not None:
            qudata = None
        qusr = self.fixed_eval(
            id_split=id_split,
            quebds=quebds,
            qudata=qudata,
            oversample=oversample,
            validate_scores=validate_scores,
            save_path=save_path,
            supress_not_saving_warning=supress_not_saving_warning,
        )
        return qusr


    def evaluate_new_test_data(
        self,
        quebds: QuEmbeddings=None,
        qudata: QuData=None,
        validate_scores: Literal["none", "warn", "error"]="none",
    ) -> QuScorerResults:
        """Evaluation for a "new" test-data. The data is matched to the previously
        generated CV-splits by the id-column. This enables the evaluation of the
        performance with data, that has undergone more or less preprocessing
        than the dataset used at CV-evaluation time.

        Parameters
        ----------
        qudata : QuestionnaireData
            The data of the questionnaire-responses. Must be passed to generate the
            embeddings if no `quebds`-object is passed.
        quebds : QuestionnaireEmbeddings
            The embeddings of the questionnaire-responses. If not passed, the embeddings
            are generated from the `qudata`-object, i.e., in this case the `qudata`-object
            must be passed.
        validate_scores : Literal["none", "warn", "error"]
            Wether to validate the scores of the passed `qudata`-object. If set to
            `"warn"` a warning is issued if the scores are not valid. If set to
            `"error"` an error is raised if the scores are not valid.

        Returns
        -------
        QuScorerResults
        """
        if not hasattr(self, "id_split"):
            raise AttributeError(
                "This QuEmbeddingScorer-instance has not yet been Train-Test-evaluated.\n"
                "For an additional evaluation you first need to perform a \"standard\" evaluation."
            )
        quconfig = self._get_quconfig(quebds=quebds, qudata=qudata)

        df_ebd, info_cols = self._get_df_trn(quebds=quebds, qudata=qudata)

        feature_cols = df_ebd.columns.str.startswith("dim-")
        feature_cols = df_ebd.columns[feature_cols].to_list()

        all_passed_ids = df_ebd[quconfig.id_col].to_list()
        quebds, qudata = self.__manage_quargs(
            qudata=qudata,
            quebds=quebds,
        )

        if isinstance(self.id_split, IDsTrainTestSplit):
            qures = QuScorerResults(
                qudata=qudata,
                target_col=self.target,
                prediction_col=self.pred,
                validate_scores=validate_scores,
            )
            df_splits = self.id_split.df_ids
            df_split = df_splits[df_splits["split"]=="test"]
            ids_split_ = df_ebd[quconfig.id_col].isin(df_split[quconfig.id_col]).values
            df_ebd_ = df_ebd[ids_split_].copy()

            model_ = self.cv_model_copys[0]
            y_pred = model_.predict(df_ebd_[feature_cols].values)
            df_ebd_[self.pred] = y_pred

            df_pred = df_ebd_.drop(columns=feature_cols)
            qures.append(df_trn=None, df_tst=df_pred)
            print(
                f"Info: {len(y_pred)} of the passed IDs found in the QuEmbeddingScorer's internal  train-test-split."
            )

        if isinstance(self.id_split, IDsKFoldSplit):
            available_ids = []

            qures = QuScorerResults(
                qudata=qudata,
                target_col=self.target,
                prediction_col=self.pred,
                validate_scores=validate_scores,
            )

            for k, (_, df_split) in enumerate(self.id_split.df_ids.groupby("split")):
                ids_split_k = df_ebd[quconfig.id_col].isin(df_split[quconfig.id_col]).values
                df_ebd_k = df_ebd[ids_split_k].copy()
                available_ids += df_ebd_k[quconfig.id_col].to_list()
                model_k = self.cv_model_copys[k]
                y_pred = model_k.predict(df_ebd_k[feature_cols].values)
                df_ebd_k[self.pred] = y_pred

                df_pred = df_ebd_k.drop(columns=feature_cols)
                qures.append(df_trn=None, df_tst=df_pred)

            missing_ids = np.setdiff1d(all_passed_ids, available_ids)
            if len(missing_ids) > 0:
                print(
                    f"Warning: {len(missing_ids)} not found in the QuEmbeddingScorer's internal CV-split. \n" +
                    f"Missing IDs: {list(missing_ids)}"
                )

        return qures


    ## Complete Fit and prediction
    def complete_data_fit(
        self,
        save_path: str,
        qudata: QuData=None,
        quebds: QuEmbeddings=None,
        oversample: bool=False,
    ) -> QuScorerResults:
        """Train the model on the complete data and evaluate it on the complete
        data. This is useful for the final model that should be used to predict
        the scores for new data in a "production" setting.

        Parameters
        ----------
        save_path : str
            The path to save the scoring model.
        qudata : QuestionnaireData
            The data of the questionnaire-responses. Must be passed to generate the
            embeddings if no `quebds`-object is passed.
        quebds : QuestionnaireEmbeddings
            The embeddings of the questionnaire-responses. If not passed, the embeddings
            are generated from the `qudata`-object, i.e., in this case the `qudata`-object
            must be passed.
        oversample : bool
            Wether to use oversampling during the training.

        Returns
        -------
        QuScorerResults
            This should be used for "evaluation" with caution because, there is
            no actual "evaluation"-dataset used in this method. It should be only
            compared to "train"-performance of previous evaluations.
        """
        quebds, qudata = QuEmbeddingScorer.__manage_quargs(
            qudata=qudata,
            quclst=None,
            quebds=quebds,
        )
        if quebds is not None:
            qudata = None

        df_ebd: pd.DataFrame
        df_ebd, _ = self._get_df_trn(qudata=qudata, quebds=quebds)
        feature_cols = df_ebd.columns.str.startswith("dim-")
        feature_cols = df_ebd.columns[feature_cols].to_list()
        if oversample:
            oversampler = get_random_oversampler()
        else:
            oversampler = None

        if quebds is not None:
            qudata = quebds.qudata

        id_split = IDsKFoldSplit.from_qudata(
            qudata=qudata,
            quclst=None,
            df_strat=None,
            strat_col=None,
            stratify=False,
            n_splits=1,
            verbose=False,
        )

        split = Split(df_trn=df_ebd)

        self.scr_model.reset()
        df_pred_trn, _, fit_history, _ = training_core(
            model=self.scr_model,
            split=split,
            feature_cols=feature_cols,
            target_col=self.target,
            prediction_col=self.pred,
            oversampler=oversampler,
        )
        if fit_history is not None:
            fit_history = {k: v for k, v in fit_history.items() if "eval" not in k}

        qusr = QuScorerResults(
            qudata=qudata,
            target_col=self.target,
            prediction_col=self.pred,
            id_split=id_split,
        )
        qusr.append(df_trn=df_pred_trn, fit_history=fit_history)

        if save_path is not None:
            self.save_scoring_model(save_path)
        return qusr


    def predict(
        self,
        qudata: QuData=None,
        quebds: QuEmbeddings=None,
    ) -> QuData:
        """Generate predictions for the tasks of the questionnaire based on the
        textual responses of the participants. The predictions are returned as
        a new `QuData`-object.

        Parameters
        ----------
        qudata : QuestionnaireData
            The data of the questionnaire-responses. Must be passed to generate the
            embeddings if no `quebds`-object is passed.
        quebds : QuestionnaireEmbeddings
            The embeddings of the questionnaire-responses. If not passed, the embeddings
            are generated from the `qudata`-object, i.e., in this case the `qudata`-object
            must be passed.

        Returns
        -------
        QuData
            The new `QuData`-object containing the predictions for the tasks of the
            questionnaire.
        """
        if not self.is_trained:
            print(
                "Warning: The scorer-model has not yet been trained, perhaps only evaluated. \n" +
                "cross-validation or train-test-validation is meant to only generate performance \n" +
                "estimations, not to set up a prediction pipeline for new data. You might want \n" +
                "to train the model on your complete dataset first."
            )

        quconfig = self._get_quconfig(quebds=quebds, qudata=qudata)
        df_ebd, info_cols = self._get_df_ebd(qudata=qudata, quebds=quebds)
        if qudata is None:
            qudata = quebds.qudata

        X_ebd = df_ebd.drop(columns=info_cols).values
        df_ebd[self.pred] = self.scr_model.predict(X=X_ebd)

        df_preds = df_ebd[[quconfig.id_col, "task", self.pred]].copy()
        df_preds[self.target] = df_preds[self.pred]
        df_preds["split"] = 0
        df_preds["mode"] = "test"

        qusr = QuScorerResults(
            qudata=qudata,
            df_preds=df_preds,
        )

        df_preds = qusr.get_wide_predictions(
            mc_units=qudata._scr_init_cols,
            append_mc_only=False,
        )
        df_preds = df_preds.drop(columns=["mode", "split"])

        qudata_ret = QuData(
            quconfig=quconfig,
            df_txt=qudata.get_txt(units=qudata._txt_init_cols, table="wide"),
            df_scr=df_preds,
            text_col_type=qudata._txt_init_cols,
            mc_score_col_type=qudata._scr_init_cols,
            clip_sparingly_occurring_scores=False,
            verbose=False,
        )
        return qudata_ret


    ## Helpers
    @staticmethod
    def __manage_quargs(
        qudata: QuData=None,
        quclst: QuScoreClusters=None,
        quebds: QuEmbeddings=None,
    ) -> tuple[QuEmbeddings, QuData]:
        if quebds is not None:
            if qudata is not None:
                print("Info: Passed `quebds`, omitting `qudata`")
            qudata = quebds.qudata
            if quebds._internal_cls_model_available():
                if quclst is not None:
                    print("Warning: Overwriting the available internal cluster model of `quebds`.")
                    quebds.quclst = quclst

        return quebds, qudata

    def _get_quconfig(
        self,
        quebds: QuEmbeddings=None,
        qudata: QuData=None,
    ) -> QuConfig:
        if quebds is not None and qudata is None:
            return quebds.qudata.quconfig
        elif quebds is None and qudata is not None:
            if self.ebd_model is None:
                raise QuScorerError(
                    "Cannot predict from QuestionaireData-instances if no embedding model\n" +
                    "(`self.ebd_model`) is set at initialization or during training."
                )
            return qudata.quconfig
        else:
            raise ValueError(
                "You have to pass either the `quebds` or the `qudata` argument."
            )

    def _get_df_ebd(
        self,
        quebds: QuEmbeddings=None,
        qudata: QuData=None,
    ) -> tuple[pd.DataFrame, list[str]]:

        if quebds is not None and qudata is None:
            pass
        elif quebds is None and qudata is not None:
            if self.ebd_model is None:
                raise QuScorerError(
                    "Cannot predict from QuestionaireData-instances if no embedding model\n" +
                    "(`self.ebd_model`) is set at initialization or during training."
                )
            df_ebd = self.ebd_model.get_embeddings(
                qudata=qudata,
                item_task_prefix=self.item_task_prefix,
                it_repl_dict=self.it_repl_dict,
                sep=self.sep,
                verbose=self.verbose,
            )

            quebds = QuEmbeddings(
                embedding_data=df_ebd,
                qudata=qudata,
                save_dir=None,
            )
        else:
            raise ValueError(
                "You have to pass either the `quebds` or the `qudata` argument."
            )

        info_cols = quebds.info_cols
        df_ebd = quebds.get_ebds(False, False, False)

        return df_ebd, info_cols

    def _get_df_trn(
        self,
        quebds: QuEmbeddings=None,
        qudata: QuData=None,
    ) -> tuple[pd.DataFrame, list[str]]:

        if quebds is not None and qudata is None:
            pass
        elif quebds is None and qudata is not None:
            passed_data = True
            if self.ebd_model is None:
                raise QuScorerError(
                    "Cannot train with QuestionaireData-instances if no embedding model\n" +
                    "(`self.ebd_model`) is set at initialization or during training."
                )
            df_ebd = self.ebd_model.get_embeddings(
                qudata=qudata,
                item_task_prefix=self.item_task_prefix,
                it_repl_dict=self.it_repl_dict,
                verbose=self.verbose,
            )
            quebds = QuEmbeddings(
                embedding_data=df_ebd,
                qudata=qudata,
                quclst=None,
                save_dir=None,
            )

        else:
            raise ValueError(
                "You have to pass either the `quebds` or the `qudata` argument."
            )

        info_cols = quebds.info_cols

        df_ebd = quebds.get_ebds(
            with_scores=True,
            fillna_scores=True,
            with_clusters=False,
        )
        df_ebd = df_ebd.dropna()

        if self.target not in df_ebd.columns:
            if passed_data:
                raise QuScorerError(
                    f"You specified `\"{self.target}\"` to be the target variable of the\n" +
                    "used `QuEmbeddingsScorer` but this column does not get generated using\n" +
                    f"the passed `EmbeddingModel`: {self.ebd_model}"
                )
            else:
                raise QuScorerError(
                    f"You specified `\"{self.target}\"` to be the target variable of the\n" +
                    "used `QuEmbeddingsScorer` but this column is not available in the passed\n" +
                    "embedding data, which has the structure:\n" +
                    f"{df_ebd.head(3)}\n" +
                    "with the columns:\n" +
                    f"{df_ebd.columns.to_list()}"
                )

        return df_ebd, info_cols


    ## IO
    def save_scoring_model(self, path: str) -> None:
        """Save the scoring model to disk.

        Parameters
        ----------
        path : str
            The path to save the scoring model.
        """
        self.scr_model.save(path)

    @staticmethod
    def _load_scoring_model(path: str, ModelClass=None, verbose: bool=False) -> Classifier:
        model_found = None
        if ModelClass is not None:
            scr_model = ModelClass.load(path=path)
            model_found = ModelClass
        else:
            if ".pt" in str(path):
                scr_model = PyTorchClassifier.load(path=path)
                model_found = "PyTorch"
            elif ".pkl" in str(path):
                scr_model = ScikitClassifier.load(path=path)
                model_found = "Scikit"
        if model_found is None:
            raise FileNotFoundError(
                "Found no file for the scoring model."
            )
        if verbose:
            print(f"Found {model_found} scoring model.")
        return scr_model

    def load_scoring_model(self, path: str, ModelClass=None) -> None:
        """Load a scoring model from disk.

        Parameters
        ----------
        path : str
            The path to load the scoring model from.
        ModelClass
            The class of the model to load. Can either be `PyTorchClassifier` or
            `ScikitClassifier`. If not passed, the model is loaded
            based on the file-extension.
        """
        self.scr_model = QuEmbeddingScorer._load_scoring_model(
            path=path,
            ModelClass=ModelClass,
            verbose=True,
        )

    def save_embedding_model(self, path: str) -> None:
        """Save the embedding model to disk.

        Parameters
        ----------
        path : str
            The path to save the embedding model.
        """
        if self.ebd_model is None:
            print(f"Info: There is currently no embedding model set.")
            return
        if hasattr(self.ebd_model, "save"):
            self.ebd_model.save_model(path)
        else:
            print(f"Info: The current embedding model {self.ebd_model} can not be saved to disk and must be re-initialized when loading the model.")

    def _save_settings_dict(self, path: str) -> None:
        settings_dict = {
            "item_task_prefix": self.item_task_prefix,
            "it_repl_dict": self.it_repl_dict,
            "verbose": self.verbose,
        }
        with open(path, "w") as f:
            dump(settings_dict, f, indent=2)

    @staticmethod
    def _load_settings_dict(path: str) -> dict:
        with open(path, "r") as f:
            settings_dict: dict = load(f)
        s = f"Found the following scorer-settings: "
        for k, v in settings_dict.items():
            s += f" ({k}: {v}) "
        print(s)
        return settings_dict

    def _save_cv_model_copys(self, path: str, model_filesuffix: str) -> None:
        p = Path(path)
        p.mkdir(parents=True)
        for idx, model_ in enumerate(self.cv_model_copys):
            model_.save(p / ("cv_model_" + str(idx) + model_filesuffix) )

    def _load_cv_model_copys(self, path: str, ModelClass=None) -> None:
        p = Path(path)
        for p_ in p.glob("*"):
            model_ = QuEmbeddingScorer._load_scoring_model(
                path=p_,
                ModelClass=ModelClass,
            )
            self.cv_model_copys.append(model_)
        print(f"Loaded {len(self.cv_model_copys)} cv-modelcopys.")


    def load_sentence_transformers_ebd_model(self, path: str) -> None:
        """Load a SentenceTransformers-embedding model from disk.

        Parameters
        ----------
        path : str
            The path to load the SentenceTransformers-embedding model from.
        """
        self.ebd_model = SentenceTransformersEmbdModel(model_name=path)
        print("Loaded SentenceTransformersEmbdModel as embedding model.")


    def save(self, save_dir: str, save_cv_model_copys: bool=False) -> None:
        """Save the whole `QuEmbeddingScorer`-object to disk.

        Parameters
        ----------
        save_dir : str
            The directory to save the object to.
        save_cv_model_copys : bool
            Wether to save the cross-validation model copies. This is needed if
            a later evaluation of altered data is planned.
        """
        dir_ = Path(save_dir)
        path_suffix_warning(suffix=dir_.suffix)
        try:
            dir_.mkdir(parents=True)
        except FileExistsError:
            rmtree(dir_)
            dir_.mkdir(parents=True)

        if isinstance(self.scr_model, PyTorchClassifier):
            model_filesuffix = ".pt"
        else:
            model_filesuffix = ".pkl"

        self.save_scoring_model(dir_ / ("main_model" + model_filesuffix))

        if save_cv_model_copys and len(self.cv_model_copys) > 0:
            self._save_cv_model_copys(
                path=dir_ / "cv_model_copys",
                model_filesuffix=model_filesuffix,
            )

        self.save_embedding_model(str(dir_ / "embedding_model"))

        self._save_settings_dict(dir_ / "settings_dict.json")


    @staticmethod
    def _find_scr_model(path: str):
        if Path(str(path) + ".pt").exists():
            return str(path) + ".pt"
        elif Path(str(path) + ".pkl").exists():
            return str(path) + ".pkl"


    @staticmethod
    def load(load_dir: str, ScrModelClass=None) -> "QuEmbeddingScorer":
        """Load a `QuEmbeddingScorer`-object from disk.

        Parameters
        ----------
        load_dir : str
            The directory to load the object from.
        ScrModelClass
            The class of the scoring model to load. Can either be `PyTorchClassifier` or
            `ScikitClassifier`. If not passed, the model is loaded
            based on the file-extension.

        Returns
        -------
        QuEmbeddingScorer
        """
        dir_ = Path(load_dir)

        settings_dict = QuEmbeddingScorer._load_settings_dict(dir_ / "settings_dict.json")
        scorer = QuEmbeddingScorer(**settings_dict)

        main_model_path = QuEmbeddingScorer._find_scr_model(dir_ / "main_model")
        scorer.load_scoring_model(main_model_path, ModelClass=ScrModelClass)

        embedding_model_path = dir_ / "embedding_model"
        if embedding_model_path.exists():
            scorer.load_sentence_transformers_ebd_model(str(embedding_model_path))

        cv_model_copies_path = dir_ / "cv_model_copys"
        if cv_model_copies_path.exists():
            scorer._load_cv_model_copys(path=cv_model_copies_path, ModelClass=ScrModelClass)

        scorer.is_trained = True

        return scorer
