"""
# Finetune-Scorer
The scorer class to be used for the finetuning of a huggingface model for the
prediction of the scores from the responses to the tasks of a questionnaire.

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

from imblearn.over_sampling import RandomOverSampler

from torch import no_grad
from torch.cuda import is_available as cuda_available
from torch.backends.mps import is_available as mps_available

from transformers import AutoTokenizer, BatchEncoding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EvalPrediction
from transformers import TrainerState, TrainerControl, TrainerCallback
from transformers.modeling_outputs import SequenceClassifierOutput

from datasets import Dataset
import evaluate

import seaborn as sns
from tqdm import tqdm
from json import dump as json_dump
from json import load as json_load

from typing import Literal
from pathlib import Path
from shutil import rmtree


from ..core.trainulation import Split, get_random_oversampler
from ..core.io import path_suffix_warning
from ..core.text import set_item_task_prefix
from ..core.batched_iter import batched
from ..core.classifier import print_torch_param_count

from ..data.config import QuConfig
from ..data.data import QuData
from ..data.id_splits import IDsSplit, IDsKFoldSplit, IDsTrainTestSplit

from ..clustering.clusters import QuScoreClusters

from .scorer_base import QuScorer
from .scorer_results import QuScorerResults




class Evaluator:
    def __init__(self) -> None:
        self.acc = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")

    def compute_metrics(self, evl_pred: EvalPrediction) -> dict[str, float]:
        logits, labels = evl_pred
        predictions = np.argmax(logits, axis=-1)
        ret = {
            "accuracy": self.acc.compute(
                predictions=predictions,
                references=labels)["accuracy"],
            "f1": self.f1.compute(
                predictions=predictions,
                references=labels,
                average="weighted")["f1"]
        }
        return ret


class EvalStuckCallback(TrainerCallback):
    def __init__(
        self,
        stuck_std_threshold_acc: float,
        stuck_std_threshold_loss: float,
        verbose: bool=False
    ) -> None:
        """A callback to determine wether the training is stuck (w. r. t. the
        evluation metrics). This is used to automatically restart the training
        of the model for a specific split if the training appears to be stuck.
        The number of attempts per split can be set in the training arguments
        of the `QuFinetuneScorer` using the `cv_training_stuck_attemts`-parameter.

        Parameters
        ----------
        stuck_std_threshold_acc : float
            The standard deviation threshold from which the accuracy is assumed
            to be stuck.
        stuck_std_threshold_loss : float
            The standard deviation threshold from which the eval loss is assumed
            to be stuck.
        verbose : bool
            Wether information should be printed if the training appears to be
            stuck.
        """
        self.stuck_std_threshold_acc = stuck_std_threshold_acc
        self.stuck_std_threshold_loss = stuck_std_threshold_loss
        self.verbose = verbose

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs
    ):
        df_fh = pd.DataFrame(state.log_history)[["eval_loss", "eval_accuracy"]]
        df_fh = df_fh.dropna()
        evl_loss = df_fh["eval_loss"].values
        evl_acc = df_fh["eval_accuracy"].values

        if len(evl_loss) < 5:
            return

        evl_loss_sd = np.std(evl_loss)
        evl_acc_sd = np.std(evl_acc)

        if (
            evl_loss_sd < self.stuck_std_threshold_loss and
            evl_acc_sd < self.stuck_std_threshold_acc
        ):
            if self.verbose:
                print("Training appears to be (eval-) stuck. Stopping training.")
            control.should_training_stop = True



class QuFinetuneScorer(QuScorer):
    def __init__(
        self,
        save_dir: str,
        num_labels: int=None,
        model_name: str="dbmdz/bert-base-german-uncased",
        local_model_path: str=None,
        local_tokenizer_path: str=None,
        item_task_prefix: bool=True,
        it_repl_dict: dict[str, str]={"A": "Aufgabe "},
        sep: str=" [SEP] ",
        verbose: bool=True,
        **kwargs,
    ) -> None:
        """A Questionnaire Scorer that is trained by finetuning a huggingface
        model.

        Parameters
        ----------
        save_dir : str
            The direction that should be used to store trained models and
            predictions for evaluation.
        num_labels : int
            The number of lables that is to be predicted. Typically the score
            range.
        model_name : str
            The name of the huggingface model to be used.
        local_model_path : str
            Can be used if the model has already be downloaded. Will be used to
            load the model instead of using the model name.
        local_tokenizer_path : str, optional
            Can be used if the model has already be downloaded. Will be used to
            load the tokenizer instead of using the model name.
        item_task_prefix : bool
            Wether the task responses should be prepended with the task names.
        it_repl_dict : dict[str, str]
            A dictionary for string replacements of the task names for prepending.
        sep : str
            The separator to be used between the task-prefix and the text,
            if / when prepending the task names.
        verbose : bool
            Sets the verbosity of the Scorer object.
        **kwargs
            Additional keyword arguments. The following are available:
            - `is_trained` : bool
                Wether the model is already trained. Default: `False`.
            - `target` : str
                The column-name of the target variable. Default: `"score"`.
            These are mainly used for the internal handling when loading and saving
            the model.
        """
        self.model_name = model_name
        self.init_model_path = local_model_path
        self.local_tokenizer_path = local_tokenizer_path
        self.num_labels = num_labels

        if self.local_tokenizer_path is not None:
            self.tokenizer = self._load_tokenizer(path=self.local_tokenizer_path)
        else:
            self.tokenizer = self._load_tokenizer(path=model_name)

        if self.init_model_path is not None:
            self.init_scr_model = self._load_language_model(
                path=self.init_model_path,
                num_labels=self.num_labels,
            )
        else:
            self.init_scr_model = self._load_language_model(
                path=self.model_name,
                num_labels=self.num_labels,
            )

        self.print_model_info()
        self.trnd_scr_model = None
        self._trnarg = None

        self.item_task_prefix = item_task_prefix
        self.it_repl_dict = it_repl_dict
        self.sep = sep
        self.verbose = verbose
        self.is_trained = kwargs.get("is_trained", False)

        self.target = kwargs.get("target", "score")
        self.pred = self.target + "_pred"
        self.training_stuck_counts = {}

        self.init_model_path = Path(save_dir) / "init_language_model"
        self.save_dir = save_dir
        self.save(save_dir=save_dir)


    ## Config
    def set_training_args(
        self,
        n_epochs: int=3,
        test_batch_size: int=16,
        eval_steps: int=200,
        logging_steps: int = 50,
        device: Literal["cuda", "cpu", "mps"]=None,
        cv_training_stuck_attemts: int=3,
        stuck_std_threshold_acc: float=1e-4,
        stuck_std_threshold_loss: float=1e-2,
    ) -> None:
        """Set the training arguments for the finetuning of the model.

        Parameters
        ----------
        n_epochs : int
            The number of epochs to train the model.
        test_batch_size : int
            The batch size to be used for the evaluation of the model.
        eval_steps : int
            The number of steps after which the model is evaluated.
        logging_steps : int
            The number of steps after which the training progress is logged.
        device : Literal["cuda", "cpu", "mps"]
            The device to be used for the training. If `None` the device is
            automatically set to `"cuda"` if available, otherwise to `"mps"`
            if available, otherwise to `"cpu"`.
        cv_training_stuck_attemts : int
            The number of attempts to restart the training of a split if it
            appears to be stuck.
        stuck_std_threshold_acc : float
            The standard deviation threshold from which the accuracy is assumed
            to be stuck.
        stuck_std_threshold_loss : float
            The standard deviation threshold from which the eval loss is assumed
            to be stuck.
        """
        if device is None:
            device = "cuda" if cuda_available() else "mps" if mps_available() else "cpu"
        self._trnarg = {
            "n_epochs": n_epochs,
            "test_batch_size": test_batch_size,
            "eval_steps": eval_steps,
            "logging_steps": logging_steps,
            "device": device,
            "cv_training_stuck_attemts": cv_training_stuck_attemts,
            "stuck_std_threshold_acc": stuck_std_threshold_acc,
            "stuck_std_threshold_loss": stuck_std_threshold_loss,
        }
        print("Using the following training args:")
        for k, v in self._trnarg.items():
            print(f" - {k} = {v}")


    def __set_taskwise_finetune_trainargs(self, quconfig: QuConfig) -> None:
        # For taskwise finetuning the eval_steps and logging_steps need to be
        # adjusted, such that the evaluation and logging is done for each task
        # and not for the whole training set.
        self._non_finetune_trnarg = self._trnarg.copy()
        n_tasks = quconfig.get_text_task_count()
        self._trnarg["eval_steps"] = int(self._trnarg["eval_steps"] / n_tasks)
        self._trnarg["logging_steps"] = int(self._trnarg["logging_steps"] / n_tasks)
        if self._trnarg["eval_steps"] == 0:
            self._trnarg["eval_steps"] = 1
        if self._trnarg["logging_steps"] == 0:
            self._trnarg["logging_steps"] = 1


    def __reset_trainargs(self) -> None:
        self._trnarg = self._non_finetune_trnarg.copy()
        del self._non_finetune_trnarg




    ## Info
    def print_model_info(self) -> None:
        """Prints the information of the model used for the scoring.
        """
        print(f"Scoring Model: {self.model_name}")
        try:
            print_torch_param_count(self.init_scr_model)
        except:
            print("Warning: Cannot print the param-count of the model type passed.")


    ## Helpers
    @staticmethod
    def _token_count_info(df: pd.DataFrame, tokenizer: AutoTokenizer) -> None:
        encoded_input = tokenizer(df["text"].to_list(), padding=True, truncation=True, return_tensors='np')
        seq_lengths = np.sum(encoded_input['input_ids'] != 0, axis=1)

        ax = sns.histplot(seq_lengths,  bins=60)
        ax.set(xlabel='Tokens per Document', ylabel='Count ($N_{ges}=15600$)')

        wrd_cnt = np.sum([len(x.split(" ")) for x in df["text"].to_list()])
        tkn_cnt = np.sum(seq_lengths)
        tkn_per_wrd = np.sum(tkn_cnt) / np.sum(wrd_cnt)

        print(f"- Word-Count (approx.): {wrd_cnt}")
        print(f"- Token-Count (exact.): {tkn_cnt}")
        print(f"- Token per Word (approx.): {tkn_per_wrd:.2f}")

        print("- Document length exeeding embedding model's maximum sequence length:")
        print(f"\t- Absolute: {np.sum(seq_lengths >= 512)}")
        print(f"\t- Relative: {100 * np.mean(seq_lengths >= 512):.2f} %")

    def _reset_init_scr_model(self) -> None:
        self.init_scr_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.init_model_path,
            num_labels=self.num_labels,
        )


    @staticmethod
    def _get_single_task_split(
        split: Split,
        task: str,
    ) -> Split:
        if "task" not in split.df_trn.columns:
            raise KeyError("The split passed for taskwise does not contain a 'task' column.")
        df_trn = split.df_trn[split.df_trn["task"]==task]
        if split.has_test_set():
            df_tst = split.df_tst[split.df_tst["task"]==task]
            if df_tst.shape[0] == 0:
                df_tst = None
        else:
            df_tst = None
        return Split(df_trn=df_trn, df_tst=df_tst)


    @staticmethod
    def _get_df_trn(
        qudata: QuData,
        verbose: bool=False,
        it_repl_dict: dict=None,
        sep: str=" [SEP] ",
        add_scores: bool=True,
    ) -> pd.DataFrame:
        df_txt = qudata.get_txt(units="tasks", table="long", with_scores=add_scores)
        df_trn = set_item_task_prefix(
            df_txt=df_txt,
            unit_col="task",
            it_repl_dict=it_repl_dict,
            sep=sep,
            verbose=verbose,
        )
        return df_trn.reset_index(drop=True)

    @staticmethod
    def _get_datasets(
        split: Split,
        tokenizer: AutoTokenizer,
        oversampler: RandomOverSampler=None,
        non_target_cols: list[str]=None,
    ) -> tuple[Dataset, Dataset]:
        def tokenize_fun(datum: dict) -> BatchEncoding:
            return tokenizer(datum["text"], padding="max_length", truncation=True)

        df_trn = split.df_trn
        if oversampler is not None:
            X_trn = df_trn[non_target_cols].values
            y_trn = df_trn["score"].values
            X_trn, y_trn = oversampler.fit_resample(X_trn, y_trn)
            df_trn = pd.DataFrame(X_trn, columns=non_target_cols)
            df_trn["score"] = y_trn

        df_trn = df_trn.rename(columns={'score': "label"})
        df_trn["label"] = df_trn["label"].astype(int)
        ds_trn = Dataset.from_pandas(df_trn)

        shuffle_rs_trn = np.random.randint(0, 1e9+1, 1)[0]
        ds_trn = ds_trn.map(tokenize_fun, batched=True).shuffle(shuffle_rs_trn)

        if split.has_test_set():
            df_tst = split.df_tst
            df_tst = df_tst.rename(columns={'score': "label"})
            df_tst["label"] = df_tst["label"].astype(int)
            ds_tst = Dataset.from_pandas(df_tst)

            shuffle_rs_tst = np.random.randint(0, 1e9+1, 1)[0]
            ds_tst = ds_tst.map(tokenize_fun, batched=True).shuffle(shuffle_rs_tst)
        else:
            ds_tst = None

        return ds_trn, ds_tst

    @staticmethod
    def _get_trainer(
        model: AutoModelForSequenceClassification,
        ds_trn: Dataset,
        ds_evl: Dataset,
        evaluator: Evaluator|None,
        n_epochs: int,
        save_path: str,
        eval_steps: int,
        logging_steps: int,
        stuck_std_threshold_acc: float=0.01,
        stuck_std_threshold_loss: float=0.1,
    ) -> Trainer:
        if evaluator is not None:
            evl_stuck_callback = EvalStuckCallback(
                stuck_std_threshold_acc=stuck_std_threshold_acc,
                stuck_std_threshold_loss=stuck_std_threshold_loss,
                verbose=False,
            )
            training_args = TrainingArguments(
                num_train_epochs=n_epochs,
                output_dir=save_path + "/checkpoints",
                eval_strategy="steps",
                eval_steps=eval_steps,
                logging_strategy="steps",
                logging_steps=logging_steps,
                metric_for_best_model='f1',
                save_total_limit=3,
                seed=int(np.random.randint(0, 1e6+1, 1)[0]),
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=ds_trn,
                eval_dataset=ds_evl,
                compute_metrics=evaluator.compute_metrics,
                callbacks=[evl_stuck_callback],
            )
        else:
            training_args = TrainingArguments(
                num_train_epochs=n_epochs,
                output_dir=save_path + "/checkpoints",
                logging_strategy="steps",
                logging_steps=logging_steps,
                save_total_limit=3,
                seed=int(np.random.randint(0, 1e6+1, 1)[0]),
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=ds_trn,
            )

        return trainer

    @staticmethod
    def _get_train_history_df(trainer: Trainer) -> pd.DataFrame:
        try:
            prog_keys = ["epoch", "step"]
            metric_keys = ["loss", "eval_loss", "eval_accuracy", "eval_f1"]
            th = pd.DataFrame(trainer.state.log_history)[prog_keys + metric_keys]
        except KeyError:
            prog_keys = ["epoch", "step"]
            metric_keys = ["loss"]
            th = pd.DataFrame(trainer.state.log_history)[prog_keys + metric_keys]
            # th = th.rename(columns={"train_loss": "loss"})

        drop_mask = th[metric_keys].isna().mean(axis=1) != 1
        th = th[drop_mask]
        return th

    @staticmethod
    def _predict(
        df: pd.DataFrame,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        device: Literal["cuda", "mps", "cpu"],
    	test_batch_size: int,
    ) -> pd.DataFrame:
        model = model.eval()
        model.to(device)
        texts = df["text"].to_list()
        y_pred = np.array([])

        for batch in tqdm(batched(texts, test_batch_size)):
            tokenized_batch: BatchEncoding = tokenizer(
                batch, padding="max_length", truncation=True, return_tensors='pt'
            )
            tokenized_batch = tokenized_batch.to(device)
            with no_grad():
                model_output: SequenceClassifierOutput = model(**tokenized_batch)
                logits = model_output.logits
                if logits.device.type == "cuda":
                    logits = logits.cpu()

            y_pred_batch = np.argmax(logits.numpy(), axis=1)
            y_pred = np.hstack([y_pred, y_pred_batch])

        df["score_pred"] = y_pred

        return df


    ## (Cross) Validation
    def _hf_training_core(
        self,
        model: AutoModelForSequenceClassification,
        split: Split,
        save_path: str,
        oversampler: RandomOverSampler=None,
        non_target_cols: list[str]=None,
    ) -> pd.DataFrame:
        # constructing datasets
        print("Encoding:")
        ds_trn, ds_tst = QuFinetuneScorer._get_datasets(
            split=split,
            tokenizer=self.tokenizer,
            oversampler=oversampler,
            non_target_cols=non_target_cols
        )

        # training
        if split.has_test_set():
            evaluator = Evaluator()
            eval_steps = self._trnarg["eval_steps"]
        else:
            ds_tst = None
            evaluator = None
            eval_steps = None

        model.train()
        trainer = QuFinetuneScorer._get_trainer(
            model=model,
            ds_trn=ds_trn,
            ds_evl=ds_tst,
            evaluator=evaluator,
            n_epochs=self._trnarg["n_epochs"],
            save_path=save_path,
            eval_steps=eval_steps,
            logging_steps=self._trnarg["logging_steps"],
            stuck_std_threshold_acc=self._trnarg["stuck_std_threshold_acc"],
            stuck_std_threshold_loss=self._trnarg["stuck_std_threshold_loss"],
        )

        rmtree(f"{save_path}/checkpoints")
        print("\nTraining:")
        trainer.train()

        # storing train history
        df_fh = QuFinetuneScorer._get_train_history_df(trainer=trainer)

        return df_fh


    def _hf_predictions(
        self,
        df: pd.DataFrame,
        model: AutoModelForSequenceClassification,
    ) -> pd.DataFrame:
        df_pred_trn = QuFinetuneScorer._predict(
            df=df.copy(),
            model=model,
            tokenizer=self.tokenizer,
            device=self._trnarg["device"],
            test_batch_size=self._trnarg["test_batch_size"],
        )
        df_pred = df_pred_trn.drop(columns="text")
        return df_pred


    def _hf_split_predicions(
        self,
        split: Split,
        model: AutoModelForSequenceClassification,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # storing predictions
        print("\nStoring Predictions:")
        if split.df_trn.shape[0] != 0:
            df_pred_trn = self._hf_predictions(df=split.df_trn, model=model)
        else:
            df_pred_trn = None

        if split.has_test_set():
            df_pred_tst = self._hf_predictions(df=split.df_tst, model=model)
        else:
            df_pred_tst = None

        return df_pred_trn, df_pred_tst



    def _hf_taskwise_traineval_core(
        self,
        quconfig: QuConfig,
        split: Split,
        save_path: str,
        oversampler: RandomOverSampler=None,
        non_target_cols: list[str]=None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Setting training args
        self.__set_taskwise_finetune_trainargs(quconfig=quconfig)

        # Store all-tasks-pretrained model
        self._save_init_lanugage_model(path=self.save_dir + "/pretrained_language_model")
        df_fh_list = []
        df_pred_trn_list = []
        df_pred_tst_list = []

        for task in quconfig.get_task_names():

            print(f"\nTaskwise-Finetuning: {task}")


            # Load all-tasks-pretrained model
            _model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.save_dir + "/pretrained_language_model",
            )
            _model.to(self._trnarg["device"])

            # extracting single task split
            _split = self._get_single_task_split(split=split, task=task)

            if _split.df_trn.shape[0] == 0:
                print(f"Info: No data for task {task} in the training set. Skipping training.")

            else:
                # training specific task
                _df_fh = self._hf_training_core(
                    model=_model,
                    split=_split,
                    save_path=save_path + "/taskwise_finetune_model",
                    oversampler=oversampler,
                    non_target_cols=non_target_cols,
                )
                df_fh_list.append(_df_fh)

                # predictions for specific task
                _df_pred_trn, _df_pred_tst = self._hf_split_predicions(
                    split=_split,
                    model=_model,
                )

                df_pred_trn_list.append(_df_pred_trn)
                df_pred_tst_list.append(_df_pred_tst)

        # Resetting training args
        self.__reset_trainargs()

        # Emptying pretrained model folder (next split is different model)
        rmtree(self.save_dir + "/pretrained_language_model")

        df_pred_trn = pd.concat(df_pred_trn_list)
        df_pred_tst = pd.concat(df_pred_tst_list)
        df_fh = pd.concat(df_fh_list)

        return df_pred_trn, df_pred_tst, df_fh

    def _get_qusr(
            self,
            id_split: IDsSplit,
            qudata: QuData,
            validate_scores: Literal["none", "warn", "error"],
            continue_qusr: bool|QuScorerResults,
            finetune_taskwise: bool=False,
        ):
        if finetune_taskwise:
            p = self.save_dir + "/qusr_tw"
        else:
            p = self.save_dir + "/qusr"

        if isinstance(continue_qusr, bool):
            if continue_qusr:
                try:
                    qusr = QuScorerResults.from_dir(p)
                    if finetune_taskwise:
                        print(f"Loaded taskwise-finetune QuScorerResults from \"{p}\".")
                    else:
                        print(f"Loaded QuScorerResults from \"{p}\".")
                except FileNotFoundError or OSError:
                    if finetune_taskwise:
                        print("Info: No taskwise-finetune QuScorerResults-data found in the `save_dir` of the QuFinetuneScorer. Initializing new training.")
                    else:
                        print("Info: QuScorerResults-data found in the `save_dir` of the QuFinetuneScorer. Initializing new training.")

                    qusr = QuScorerResults(
                        qudata=qudata,
                        id_split=id_split,
                        target_col=self.target,
                        prediction_col=self.pred,
                        validate_scores=validate_scores,
                    )
                return qusr
            else:
                qusr = QuScorerResults(
                    qudata=qudata,
                    id_split=id_split,
                    target_col=self.target,
                    prediction_col=self.pred,
                    validate_scores=validate_scores,
                )
                return qusr
        elif isinstance(continue_qusr, QuScorerResults):
            if id_split != continue_qusr.id_split:
                raise ValueError(
                    "The passed IDSplit is not identical to the passed `continue_qusr`'s (QuScorerResults object to train on with) IDSplit."
                )
            if qudata != continue_qusr.qudata:
                raise ValueError(
                    "The passed QuData is not identical to the passed `continue_qusr`'s (QuScorerResults object to train on with) QuData."
                )
            print("Info: Continuing the training on the QuScorerResults-data passed. Make sure, that this data stems from the correct model.")
            return continue_qusr
        else:
            raise TypeError(
                "The `continue_qusr`-argument must be either a boolean or a QuScorerResults-object."
            )

    def _split_idx_training_stuck(
        self,
        idx: int,
        df_fh: pd.DataFrame,
    ) -> bool:
        cv_training_stuck_attemts = self._trnarg["cv_training_stuck_attemts"]
        stuck_std_threshold_acc = self._trnarg["stuck_std_threshold_acc"]
        stuck_std_threshold_loss = self._trnarg["stuck_std_threshold_loss"]

        if df_fh is None:
            return False

        try:
            df_fh_evl = df_fh[["eval_loss", "eval_accuracy"]].dropna()
            evl_std_dct = df_fh_evl.agg("std").T.to_dict()

            acc_std = evl_std_dct["eval_accuracy"]
            loss_std = evl_std_dct["eval_loss"]

            is_stuck = acc_std < stuck_std_threshold_acc and loss_std < stuck_std_threshold_loss
            if not is_stuck:
                return False

            if idx in self.training_stuck_counts:
                self.training_stuck_counts[idx] += 1
                if self.training_stuck_counts[idx] < cv_training_stuck_attemts:
                    print(
                        f"Warning: The training for CV-Split \"{idx}\" appears to be stuck. \n" +
                        "Attempting another fit with new random seeds."
                    )
                    return True
                else:
                    print(
                        f"Warning: The training for CV-Split \"{idx}\" has been stuck for 3 times. \n" +
                        "There will be no further attempt in fitting a model with that split-data \n" +
                        "and it will be missing in the returned QuScorerResults."
                    )
                    return False
            else:
                self.training_stuck_counts[idx] = 1
                print(
                    f"Warning: The training for CV-Split \"{idx}\" appears to be stuck. \n" +
                    "Attempting another fit with new random seeds."
                )
                return True

        except KeyError:
            print(f"Info: No information on eval/test-predictions or metrcis. Can not determine whether the training of split {idx} is stuck.")
            return False

    def _split_idx_is_trained(self, idx: int, qusr: QuScorerResults) -> bool:
        df_fh = qusr.get_fit_histories_dataframe()
        if df_fh is None:
            return False

        idx_in_qusr: list = df_fh["split"].unique().tolist()
        if idx not in idx_in_qusr:
            return False

        df_fh = (df_fh[df_fh["split"]==idx]
            .drop(columns="split")
            .reset_index(drop=True)
        )

        idx_training_stuck = self._split_idx_training_stuck(idx=idx, df_fh=df_fh)
        return not idx_training_stuck


    def fixed_eval(
        self,
        id_split: IDsSplit,
        qudata: QuData,
        finetune_taskwise: bool=False,
        continue_qusr: bool|QuScorerResults=True,
        oversample: bool=False,
        validate_scores: Literal["none", "warn", "error"]="none",
    ) -> QuScorerResults:
        """Train the model on the training set and evaluate it on the test set.
        The training is done for each split of the `id_split`-object.

        Parameters
        ----------
        id_split : IDsSplit
            The IDSplit-object to be used for the training and evaluation.
        qudata : QuData
            The QuData-object containing the responses and scores.
        finetune_taskwise : bool
            ***Experimental***: Wether the model should be finetuned for each task separately.
            This is much more resource intensive depending on the number of tasks in
            the questionnaire. It not yet integrated fully with the saving
            and loading workflows of the `QuFinetuneScorer` objects. When used, you
            are advised to store the results of the training in a separate directory
            to avoid overwriting previous results.
        continue_qusr : bool | QuScorerResults
            Wether the training should be continued on a previous training. If
            `True` the training is continued on the QuScorerResults-object found
            in the `save_dir` of the QuFinetuneScorer. If a QuScorerResults-object
            is passed the training is continued on that object.
        oversample : bool
            Wether the training data should be oversampled.
        validate_scores : Literal["none", "warn", "error"]
            Wether the scores should be validated before training. If set to
            `"warn"` a warning is printed if scores are invalid. If set to
            `"error"` an error is raised if scores are invalid.
        """
        if self._trnarg is None:
            self.set_training_args()

        self.id_split = id_split
        quconfig = qudata.quconfig
        id_col = quconfig.id_col

        df_trn = QuFinetuneScorer._get_df_trn(
            qudata=qudata,
            verbose=self.verbose,
            it_repl_dict=self.it_repl_dict,
            sep=self.sep,
        )

        if self.verbose:
            QuFinetuneScorer._token_count_info(df=df_trn, tokenizer=self.tokenizer)


        if oversample:
            print("Using oversampling.")

        test_ids_dict = id_split.get_tst_ids_dict()
        qusr = self._get_qusr(
            id_split=id_split,
            qudata=qudata,
            validate_scores=validate_scores,
            continue_qusr=continue_qusr,
            finetune_taskwise=False,
        )

        # For taskwise finetuning
        if finetune_taskwise:
            qusr_tw = self._get_qusr(
                id_split=id_split,
                qudata=qudata,
                validate_scores=validate_scores,
                continue_qusr=continue_qusr,
                finetune_taskwise=True,
            )
            qusr_tw.to_dir(path=Path(self.save_dir) / "qusr_tw", allow_empty_preds=True)

        # Only train missing or stuck CV-splits
        qusr.to_dir(path=Path(self.save_dir) / "qusr", allow_empty_preds=True)
        continue_qusr = True

        for idx, tst_ids in test_ids_dict.items():

            # Reset initial model
            self._reset_init_scr_model()
            qusr = QuScorerResults.from_dir(self.save_dir + "/qusr")

            if not finetune_taskwise:
                if self._split_idx_is_trained(idx=idx, qusr=qusr):
                    continue
            else:
                if (
                    self._split_idx_is_trained(idx=idx, qusr=qusr) and
                    self._split_idx_is_trained(idx=idx, qusr=qusr_tw)
                ):
                    continue

            _train_count = 0

            # Model training with all tasks
            while True:

                _train_count += 1
                print(f"\nCV-Split {idx} of {id_split.get_n_splits()}:")
                if _train_count > 1:
                    print(f"  - Training attempt {_train_count}/{self._trnarg['cv_training_stuck_attemts']}")
                print("---------------------------------------------------------------------------------------------------")

                qusr.remove(idx)
                split = Split.from_tst_ids(df=df_trn, tst_ids=tst_ids, id_col=id_col)

                if oversample:
                    oversampler = get_random_oversampler()
                    non_target_cols = [quconfig.id_col, "task", "text"]
                else:
                    oversampler = None
                    non_target_cols = None

                df_fit_history = self._hf_training_core(
                    model=self.init_scr_model,
                    split=split,
                    save_path=self.save_dir + "/pretrained_language_model",
                    oversampler=oversampler,
                    non_target_cols=non_target_cols,
                )

                if not self._split_idx_training_stuck(idx=idx, df_fh=df_fit_history):
                    df_pred_trn, df_pred_tst = self._hf_split_predicions(
                        split=split,
                        model=self.init_scr_model,
                    )
                    qusr.append(
                        df_trn=df_pred_trn,
                        df_tst=df_pred_tst,
                        fit_history=df_fit_history,
                        idx_split=idx,
                    )
                    qusr.to_dir(path=Path(self.save_dir) / "qusr")
                    break

            # Taskwise finetuning
            if finetune_taskwise:

                print(f"\nTaskwise-Finetuning CV-Split {idx} of {id_split.get_n_splits()}:")
                print("- - - - - - - - - - - - - - - - - - - - - - - -")

                qusr_tw = QuScorerResults.from_dir(self.save_dir + "/qusr_tw")
                qusr_tw.remove(idx)

                if oversample:
                    oversampler = get_random_oversampler()
                    non_target_cols = [quconfig.id_col, "task", "text"]
                else:
                    oversampler = None
                    non_target_cols = None

                # amend
                df_pred_trn, df_pred_tst, df_fit_history = self._hf_taskwise_traineval_core(
                    quconfig=quconfig,
                    split=split,
                    save_path=self.save_dir,
                    oversampler=oversampler,
                    non_target_cols=non_target_cols
                )
                qusr_tw.append(
                    df_trn=df_pred_trn,
                    df_tst=df_pred_tst,
                    fit_history=df_fit_history,
                    idx_split=idx,
                )
                qusr_tw.to_dir(path=Path(self.save_dir) / "qusr_tw")

        if finetune_taskwise:
            print("\n\n\"Pre-Trained\"-Model Results:\n")
            qusr.evaluation()
            print(f"\nReturning Task-Wise-Finetuned Results. Load Pre-Trained-Model Results from {self.save_dir}/qusr for comparison.")
            return qusr_tw

        return qusr


    def continue_evaluation(
        self,
        oversample: bool=False,
        validate_scores: Literal["none", "warn", "error"]="none",
    ) -> QuScorerResults:
        """Continue the evaluation of the model based on the previous
        `QuScorerResults` saved alongside the object.

        Parameters
        ----------
        oversample : bool
            Wether the training data should be oversampled.
        validate_scores : Literal["none", "warn", "error"]
            Wether the scores should be validated before training. If set to
            `"warn"` a warning is printed if scores are invalid. If set to
            `"error"` an error is raised if scores are

        Returns
        -------
        QuScorerResults
            The QuScorerResults-object containing the results of the training and evaluation.
        """
        try:
            qusr = QuScorerResults.from_dir(self.save_dir + "/qusr")
        except FileNotFoundError:
            print("Warning: There is no QuScorerResults-object stored within the model's `save_dir`. Cannot continue training.")
        qusr = self.fixed_eval(
            id_split=qusr.id_split,
            qudata=qusr.qudata,
            continue_qusr=qusr,
            oversample=oversample,
            validate_scores=validate_scores,
        )
        return qusr


    def random_cross_validate(
        self,
        qudata: QuData,
        quclst: QuScoreClusters=None,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
        continue_qusr: bool|QuScorerResults=True,
        oversample: bool=False,
        stratify: bool=True,
        finetune_taskwise: bool=False,
        validate_scores: Literal["none", "warn", "error"]="none",
        n_splits: int=10,
        random_state: int=42,
        supress_id_split_info: bool=False,
    ) -> QuScorerResults:
        """Perform a random cross-validation on the data.

        Parameters
        ----------
        qudata : QuestionnaireData
            The data of the questionnaire-responses.
        quclst : QuScoreClusters
            A cluster model to use for stratification when setting up the
            `IDsKFoldSplit`-object.
        df_strat : pd.DataFrame
            Alternatively to the `quclst`-object: A dataframe can be passed to
            to use for the stratification. The dataframe must contain the id-column
            and the column specified in `strat_col`.
        strat_col : str
            The column to use for the stratification when using the `df_strat`-argument.
        continue_qusr: bool|QuScorerResults
            If `True` the training is continued on the QuScorerResults-object found
            in the `save_dir` of the QuFinetuneScorer. If a QuScorerResults-object
            is passed the training is continued on that object.
        oversample : bool
            Wether to use oversampling during the training.
        stratify : bool
            Wether to stratify the splits based on the scores of the tasks.
        finetune_taskwise : bool
            ***Experimental***: Wether the model should be finetuned for each task separately.
            This is much more resource intensive depending on the number of tasks in
            the questionnaire. It not yet integrated fully with the saving
            and loading workflows of the `QuFinetuneScorer` objects. When used, you
            are advised to store the results of the training in a separate directory
            to avoid overwriting previous results.
        validate_scores : Literal["none", "warn", "error"]
            Wether to validate the scores of the passed `qudata`-object. If set to
            `"warn"` a warning is issued if the scores are not valid. If set to
            `"error"` an error is raised if the scores are not valid.
        n_splits : int
            The number of splits to use for the cross-validation.
        random_state : int
            The random state to use for the cross-validation splits.
        supress_id_split_info : bool
            Wether to suppress the information-output on the (newly generated) ID-split.

        Returns
        -------
        QuScorerResults
        """
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
        qusr = self.fixed_eval(
            id_split=id_split,
            qudata=qudata,
            finetune_taskwise=finetune_taskwise,
            continue_qusr=continue_qusr,
            oversample=oversample,
            validate_scores=validate_scores,
        )
        return qusr


    def random_train_testing(
        self,
        qudata: QuData,
        quclst: QuScoreClusters=None,
        df_strat: pd.DataFrame=None,
        strat_col: str=None,
        continue_qusr: bool|QuScorerResults=True,
        oversample: bool=False,
        stratify: bool=True,
        finetune_taskwise: bool=False,
        validate_scores: Literal["none", "warn", "error"]="none",
        test_size: float=0.2,
        random_state: int=42,
        supress_id_split_info: bool=False,
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
        continue_qusr : bool|QuScorerResults
            If `True` the training is continued on the QuScorerResults-object found
            in the `save_dir` of the QuFinetuneScorer. If a QuScorerResults-object
            is passed the training is continued on that object.
        oversample : bool
            Wether to use oversampling during the training.
        stratify : bool
            Wether to stratify the splits based on the scores of the tasks.
        finetune_taskwise : bool
            ***Experimental***: Wether the model should be finetuned for each task separately.
            This is much more resource intensive depending on the number of tasks in
            the questionnaire. It not yet integrated fully with the saving
            and loading workflows of the `QuFinetuneScorer` objects. When used, you
            are advised to store the results of the training in a separate directory
            to avoid overwriting previous results.
        validate_scores : Literal["none", "warn", "error"]
            Wether to validate the scores of the passed `qudata`-object. If set to
            `"warn"` a warning is issued if the scores are not valid. If set to
            `"error"` an error is raised if the scores are not valid.
        test_size: float=0.2
            The (relative) size of the test-dataset to use for the train-test-split.
        random_state : int
            The random state to use for the cross-validation splits.
        supress_id_split_info : bool
            Wether to suppress the information-output on the (newly generated) ID-split.

        Returns
        -------
        QuScorerResults
        """
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
        qusr = self.fixed_eval(
            id_split=id_split,
            qudata=qudata,
            finetune_taskwise=finetune_taskwise,
            continue_qusr=continue_qusr,
            oversample=oversample,
            validate_scores=validate_scores,
        )
        return qusr


    ## Complete Fit and prediction
    def complete_data_fit(
        self,
        qudata: QuData=None,
        oversample: bool=False,
    ) -> QuScorerResults:
        """Train the model on the complete data and evaluate it on the complete
        data. This is useful for the final model that should be used to predict
        the scores for new data in a "production" setting.

        Parameters
        ----------
        qudata : QuestionnaireData
            The data of the questionnaire-responses.
        oversample : bool
            Wether to use oversampling during the training.

        Returns
        -------
        QuScorerResults
            This should be used for "evaluation" with caution because, there is
            no actual "evaluation"-dataset used in this method. It should be only
            compared to "train"-performance of previous evaluations.
        """
        quconfig = qudata.quconfig
        df_trn = QuFinetuneScorer._get_df_trn(
            qudata=qudata,
            verbose=self.verbose,
            it_repl_dict=self.it_repl_dict,
            sep=self.sep,
        )

        if oversample:
            oversampler = get_random_oversampler()
            non_target_cols = [quconfig.id_col, "task", "text"]
        else:
            oversampler = None
            non_target_cols = None

        id_split = IDsKFoldSplit.from_qudata(
            qudata=qudata,
            quclst=None,
            df_strat=None,
            strat_col=None,
            stratify=False,
            n_splits=1,
            verbose=False,
        )

        split = Split(df_trn=df_trn)

        self._reset_init_scr_model()
        df_fh = self._hf_training_core(
            model=self.init_scr_model,
            split=split,
            save_path=self.save_dir + "/pretrained_language_model",
            oversampler=oversampler,
            non_target_cols=non_target_cols,
        )
        df_pred_trn = self._hf_predictions(df=split.df_trn, model=self.init_scr_model)

        qusr = QuScorerResults(
            qudata=qudata,
            target_col=self.target,
            prediction_col=self.pred,
            id_split=id_split,
        )
        qusr.append(df_trn=df_pred_trn, fit_history=df_fh)

        self.trnd_scr_model = self.init_scr_model
        self.is_trained = True
        self._reset_init_scr_model()

        self.save(save_dir=self.save_dir)
        qusr.to_dir(path=Path(self.save_dir) / "qusr-full-fit")

        return qusr


    def predict(
        self,
        qudata: QuData,
    ) -> QuData:
        """Predict the scores for the passed `QuData`-object.

        Parameters
        ----------
        qudata : QuData
            The QuData-object containing the responses to predict the scores for.

        Returns
        -------
        QuData
            The QuData-object with the predicted scores.
        """
        if not self.is_trained:
            print(
                "Warning: The scorer-model has not yet been trained, perhaps only evaluated. \n" +
                "cross-validation or train-test-validation is meant to only generate performance \n" +
                "estimations, not to set up a prediction pipeline for new data. You might want \n" +
                "to train the model on your complete dataset first."
            )
            model = self.init_scr_model
        else:
            model = self.trnd_scr_model

        quconfig = qudata.quconfig
        df_txt = QuFinetuneScorer._get_df_trn(
            qudata=qudata,
            it_repl_dict=self.it_repl_dict,
            verbose=self.verbose,
            add_scores=False,
        )

        df_preds = QuFinetuneScorer._predict(
            df=df_txt,
            model=model,
            tokenizer=self.tokenizer,
            device=self._trnarg["device"],
            test_batch_size=self._trnarg["test_batch_size"],
        )
        df_preds = df_preds.drop(columns="text")
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


    ## IO
    def _save_init_lanugage_model(self, path: str) -> None:
        self.init_scr_model.save_pretrained(path)

    def _save_trained_language_model(self, path: str) -> None:
        if self.trnd_scr_model is not None:
            self.trnd_scr_model.save_pretrained(path)
        else:
            print("No trained model to be saved.")

    @staticmethod
    def _load_language_model(
        path: str,
        num_labels: int,
    ) -> AutoModelForSequenceClassification:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=path,
            num_labels=num_labels,
        )
        return model


    def _save_tokenizer(self, path: str) -> None:
        self.tokenizer.save_pretrained(path)

    @staticmethod
    def _load_tokenizer(path: str) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=path,
        )
        return tokenizer


    @staticmethod
    def _load_settings_dict(path: str) -> dict:
        with open(path, "r") as f:
            settings_dict: dict = json_load(f)
        s = f"Found the following scorer-settings: \n"
        for k, v in settings_dict.items():
            s += f" - {k}: {v}\n"
        print(s)
        return settings_dict

    def _save_settings_dict(self, path: str) -> None:
        settings_dict = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "item_task_prefix": self.item_task_prefix,
            "it_repl_dict": self.it_repl_dict,
            "verbose": self.verbose,
        }
        with open(path, "w") as f:
            json_dump(settings_dict, f, indent=2)


    def save(self, save_dir: str) -> None:
        """Save the QuFinetuneScorer-object to the specified directory.

        Parameters
        ----------
        save_dir : str
            The directory to save the QuFinetuneScorer-object to.
        """
        dir_ = Path(save_dir)
        qusr = None
        qusr_tw = None
        path_suffix_warning(suffix=dir_.suffix)
        try:
            dir_.mkdir(parents=True)
        except FileExistsError:
            # Saving a potential previous fitting / evaluation results
            if (dir_ / "qusr").exists():
                try:
                    qusr = QuScorerResults.from_dir(dir_ / "qusr")
                except FileNotFoundError:
                    qusr = None
                    rmtree(dir_ / "qusr")
            # Saving a potential previous fitting / evaluation results (taskwise-finetuning)
            if (dir_ / "qusr_tw").exists():
                try:
                    qusr_tw = QuScorerResults.from_dir(dir_ / "qusr_tw")
                except FileNotFoundError:
                    qusr = None
                    rmtree(dir_ / "qusr_tw")
            if (
                self.trnd_scr_model is None and
                (dir_ / "pretrained_language_model").exists()
            ):
                try:
                    self.trnd_scr_model = QuFinetuneScorer._load_language_model(
                        path=dir_ / "pretrained_language_model",
                        num_labels=self.num_labels
                    )
                except OSError:
                    pass
            rmtree(dir_)
            dir_.mkdir(parents=True)
            if qusr is not None:
                qusr.to_dir(dir_ / "qusr", allow_empty_preds=True)
            if qusr_tw is not None:
                qusr_tw.to_dir(dir_ / "qusr_tw", allow_empty_preds=True)

        self._save_tokenizer(dir_ / "tokenizer")
        self._save_init_lanugage_model(dir_ / "init_language_model")
        self._save_settings_dict(dir_ / "settings_dict.json")
        self._save_trained_language_model(dir_ / "pretrained_language_model")


    @staticmethod
    def load(load_dir: str) -> "QuFinetuneScorer":
        """Load a QuFinetuneScorer-object from the specified directory.

        Parameters
        ----------
        load_dir : str
            The directory to load the QuFinetuneScorer-object from.

        Returns
        -------
        QuFinetuneScorer
        """
        dir_ = Path(load_dir)

        settings_dict = QuFinetuneScorer._load_settings_dict(dir_ / "settings_dict.json")
        scorer = QuFinetuneScorer(
            local_tokenizer_path=dir_ / "tokenizer",
            local_model_path=dir_ / "init_language_model",
            save_dir=load_dir,
            **settings_dict,
        )
        if Path(dir_ / "pretrained_language_model").exists():
            print("Found trained language model for scoring.")
            trnd_scr_model = QuFinetuneScorer._load_language_model(
                path=dir_ / "pretrained_language_model",
                num_labels=settings_dict["num_labels"],
            )
            scorer.trnd_scr_model = trnd_scr_model
            scorer.is_trained = True

        return scorer
