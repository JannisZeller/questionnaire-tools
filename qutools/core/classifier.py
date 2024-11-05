"""
# Submodule for classification model wrappers

This submodule provides an abstract base and utility functions for creating and managing
classification models. It is designed to support both traditional machine learning classifiers and deep learning
models using PyTorch. The submodule includes a mix of utilities for model parameter inspection and an abstract
base class for implementing classifiers.

Usage
-----
This submodule is intended to be used as a foundation for implementing various types of classifiers,
both traditional machine learning models and deep learning models. By subclassing the `Classifier` ABC,
developers can create new classifiers that adhere to a standardized interface, simplifying the process
of model training, evaluation, and integration with other systems.

Example
-------
Below is an example of how to use the `ScikitClassifier` class to create a simple logistic regression model:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from qutools.core.classifier import ScikitClassifier

X, y = make_classification(n_samples=1000, n_classes=4, n_informative=10)
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2)

clf_core = RandomForestClassifier()
clf = ScikitClassifier(model=clf_core)
clf.fit(X=X_trn, y=y_trn)

y_tst_pred = clf.predict(X_tst)

print(f"F1-Score: {f1_score(y_tst, y_tst_pred, average='weighted'):.2f}")
```
```
> F1-Score: 0.81
```

Below is an example on how to use the `PyTorchClassifier` class together with the `DenseNN` class
to create a simple neural network model:
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from qutools.core.classifier import PyTorchClassifier, DenseNN

X, y = make_classification(n_samples=1000, n_classes=4, n_informative=10)
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2)

clf_core = DenseNN(in_features=X_trn.shape[1], n_classes=len(set(y_trn)))
clf = PyTorchClassifier(model=clf_core)
clf.fit(X_trn=X_trn, y_trn=y_trn)

y_tst_pred = clf.predict(X_tst)

print(f"F1-Score: {f1_score(y_tst, y_tst_pred, average='weighted'):.2f}")
```
```
> F1-Score: 0.87
```

This submodule is a key component for projects requiring flexible and standardized
classification model management, especially when working with a mix of model types
and training routines.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

from joblib import load as jl_load
from joblib import dump as jl_dump

from typing import Literal
from collections import OrderedDict
from abc import ABC, abstractmethod



def print_torch_param_count(model: nn.Module) -> None:
    """A utility function to print the total number of parameters and the number
    of trainiable parameters of a `torch.nn.Module`. Also works for huggingface
    torch-models.

    Parameters
    ----------
    model : nn.Module
    """
    trainable_ps = 0
    all_ps = 0
    for _, p in model.named_parameters():
        n_p = p.numel()
        all_ps += n_p
        if p.requires_grad:
            trainable_ps += n_p
    print(f"Trainable: {trainable_ps} / Total: {all_ps}")




class Classifier(ABC):
    """Abstract base class for classifiers and easy unified tranining and
    evaluation.
    """

    is_pretrained_model = False
    is_trained = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Abstract base method for model-fitting. States the default argument
        and return behavior. Also, this method fixes a check for already
        trained models, which is relevant in crossvalidation workflows.

        Parameters
        ----------
        X : np.ndarray
            The (typically numeric) training features as a $N_{\\mathrm{samples}} \\times N_{\\mathrm{features}}$
            `np.ndarray`.
        y : np.ndarray
            The labels as a len-$N_{\\mathrm{samples}}$ `np.ndarray`.
        """
        if self.is_trained:
            print("WARNING: The model already has been trained. If you are \"training on\", this is fine. Else, if you are cross-validating, this is data-leakage and you should reset the model!")
        self.is_trained = True

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Abstract base method for model-predictions. States the default argument
        and return behavior.

        Parameters
        ----------
        X : np.ndarray
            The (typically numeric) evaluation features as a $N_{\\mathrm{samples}} \\times N_{\\mathrm{features}}$
            `np.ndarray`.

        Returns
        -------
        np.ndarray
            The predictions as a len-$N_{\\mathrm{samples}}$ `np.ndarray`.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Abstract basemethod for resetting of the model. Needed in crossvalidation
        workflows.
        """
        if self.is_pretrained_model:
            print(
                "Warning: This classifier uses a model, that has beem load from disk,\n" +
                "i.e., it might be pre-trained. Resetting a pretrained model, is perhaps \n" +
                "not a true \"reset\"."
            )
        self.is_trained = False

    def train(self) -> None:
        """This method is needed for PyTorch-Classifiers to switch between the
        train- and eval- states of the model. The method is implemented in the
        base class for unification and convenience purposes.
        """
        pass

    def eval(self) -> None:
        """This method is needed for PyTorch-Classifiers to switch between the
        train- and eval- states of the model. The method is implemented in the
        base class for unification and convenience purposes.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save classifier to disc.

        Parameters
        ----------
        path : str
        """

    @staticmethod
    @abstractmethod
    def load(path: str) -> "Classifier":
        """Load the classifier from the passed filepath.

        Parameters
        ----------
        path : str
        """

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load the classifiers internal model from a passed filepath.

        Parameters
        ----------
        path : str
        """




class ScikitClassifier(Classifier):
    """A wrapper class for Scikit-Learn classifiers, to provide a unified
    interface of `fit`, `predict`, and `reset` methods. The `train` and `eval`
    methods of the baseclass are basically ignored, because they are not
    relevant for Scikit-Learn models.
    """
    def __init__(self, model: ClassifierMixin=None) -> None:
        """Initializes the model. Stores a clones version of the model internally
        for easy reset-capability.

        Parameters
        ----------
        model : ClassifierMixin
            A Scikit-Learn classification model, e. g. `LogisticRegression`
            instance.
        """
        if model is None:
            model = LogisticRegression(C=0.5, max_iter=1000)
        self.model = model
        self.reset_model = clone(model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """The fit method for the model. Calls the base-classes fit, that checks
        for previous training and then the internal scikit-models fit.

        Parameters
        ----------
        X : np.ndarray
            The (typically numeric) training features as a $N_{\\mathrm{samples}} \\times N_{\\mathrm{features}}$
            `np.ndarray`.
        y : np.ndarray
            The labels as a len-$N_{\\mathrm{samples}}$ `np.ndarray`.
        """
        super().fit(X, y)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> None:
        """Method for generating model predictions.

        Parameters
        ----------
        X : np.ndarray
            The (typically numeric) evaluation features as a $N_{\\mathrm{samples}} \\times N_{\\mathrm{features}}$
            `np.ndarray`.

        Returns
        -------
        np.ndarray
            The predictions as a len-$N_{\\mathrm{samples}}$ `np.ndarray`.
        """
        return self.model.predict(X)

    def reset(self) -> None:
        super().reset()
        self.model = self.reset_model

    def save(self, path: str) -> None:
        """Save classifier to disc. The joblib-library is used, i.e, a conventional
        file-suffix would be ".joblib".

        Parameters
        ----------
        path : str
        """
        jl_dump(self.model, path)

    def load_model(self, path: str) -> None:
        """Load the classifiers internal model from a passed filepath.

        Parameters
        ----------
        path : str
        """
        self.is_pretrained_model = True
        self.model = jl_load(path)

    @staticmethod
    def load(path: str) -> "ScikitClassifier":
        """Load the classifier from the passed filepath.

        Parameters
        ----------
        path : str
        """
        model = jl_load(path)
        clf = ScikitClassifier(model)
        clf.is_pretrained_model = True
        return clf




class PyTorchClassifier(Classifier):
    """A wrapper class for PyTorch classifiers, to provide a unified
    interface of `fit`, `predict`, `reset`, `train`, and `eval` methods.
    """
    def __init__(
        self,
        model: nn.Module,
        max_epochs: int=500,
        device: Literal["cuda", "cpu"]="cuda",
        verbose: bool=False,
        optimizer: optim.Optimizer=None,
        **kwargs,
    ) -> None:
        """Initializes a wrapper for a PyTorch classifier to provide a unified
        interface and a simple training setup.

        Parameters
        ----------
        model : nn.Module
            PyTorch model that is the kernel of the classification.
        max_epochs : int=500
            Maximum epochs for trianing
        device : Literal[\"cuda\", \"cpu\"]
            The device used for training. Has to be set to
            \"cpu\", if no cuda-GPU is available.
        verbose : bool
            Sets the behavior for later information outputs
        """
        self.model = model
        self.max_epochs = max_epochs
        self.lr = kwargs.get("lr", 0.01)
        self.weight_decay = kwargs.get("weight_decay", 0.001)
        self.verbose = verbose
        self.device = device
        self.model.to(self.device)
        if optimizer is None:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        self.optimizer = optimizer


    def fit(
        self,
        X_trn: np.ndarray,
        y_trn: np.ndarray,
        X_evl: np.ndarray=None,
        y_evl: np.ndarray=None,
        log_interval: int=10,
        **kwargs,
    ) -> dict[str, list]:
        """Fitting of the model. Optionally, evaluation data can be passed to
        generate data for plotting learning curves etc.

        Parameters
        ----------
        X_trn : np.ndarray
            The (typically numeric) training features as a $N_{\\mathrm{samples}} \\times N_{\\mathrm{features}}$
            `np.ndarray`.
        y_trn : np.ndarray
            The training labels as a len-$N_{\\mathrm{samples}}$ `np.ndarray`.
        X_evl : np.ndarray
            The (typically numeric) evaluation features as a $N_{\\mathrm{samples}} \\times N_{\\mathrm{features}}$
            `np.ndarray`
        y_evl : np.ndarray
            The evaluation labels as a len-$N_{\\mathrm{samples}}$ `np.ndarray`
        log_interval : int
            The (epoch-) interval used for logging

        Returns
        -------
        fit_history : dict[str, list]
            The fit history as a dictionary:
            ```python
            {
                'epoch': ...,
                'train_loss': ...,
                'eval_loss': ...,
                'train_acc': ...,
                'eval_acc': ...,
                'train_f1': ...,
                'eval_f1': ...,
            }
            ```
        """
        fit_history = {
            'epoch': [],
            'loss': [],
            'eval_loss': [],
            'train_accuracy': [],
            'eval_accuracy': [],
            'train_f1': [],
            'eval_f1': [],
        }
        super().fit(X_trn, y_trn)

        X_trn = torch.tensor(X_trn, dtype=torch.float32).to(self.device)
        y_trn_np = y_trn
        y_trn = torch.tensor(y_trn, dtype=torch.long).to(self.device)
        if X_evl is not None:
            X_evl = torch.tensor(X_evl, dtype=torch.float32).to(self.device)
            y_evl_np = y_evl
            y_evl = torch.tensor(y_evl, dtype=torch.long).to(self.device)

        optimizer = self.optimizer

        loss_functon = nn.CrossEntropyLoss().to(self.device)

        self.model.train()
        for i in range(self.max_epochs):
            optimizer.zero_grad()

            trn_logits = self.model.forward(X_trn)
            trn_loss: torch.Tensor = loss_functon(trn_logits, y_trn)
            trn_loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                fit_history["epoch"].append(i)

                with torch.no_grad():
                    self.model.eval()

                    trn_loss = trn_loss.detach().cpu().item()
                    y_trn_pred = torch.argmax(trn_logits, axis=1).cpu().numpy()
                    trn_acc = accuracy_score(y_trn_np, y_trn_pred)
                    trn_f1 = f1_score(y_trn_np, y_trn_pred, average="weighted")

                    fit_history["loss"].append(trn_loss)
                    fit_history["train_accuracy"].append(trn_acc)
                    fit_history["train_f1"].append(trn_f1)

                    if X_evl is not None:
                        evl_logits = self.model.forward(X_evl)
                        evl_loss: torch.Tensor = loss_functon(evl_logits, y_evl)
                        y_evl_pred = torch.argmax(evl_logits, axis=1).cpu().numpy()
                        evl_acc = accuracy_score(y_evl_np, y_evl_pred)
                        evl_f1 = f1_score(y_evl_np, y_evl_pred, average="weighted")

                        fit_history["eval_loss"].append(evl_loss.cpu().item())
                        fit_history["eval_accuracy"].append(evl_acc)
                        fit_history["eval_f1"].append(evl_f1)

                    self.model.train()

                if self.verbose == 2:
                    print(f"Epoch {i}:  Loss {trn_loss}")

        return fit_history

    def predict(self, X: np.ndarray, return_logits: bool=False) -> np.ndarray:
        """Predicting the labels for new data.

        Parameters
        ----------
        X : np.ndarray
            The (typically numeric) features as a $N_{\\mathrm{samples}} \\times N_{\\mathrm{features}}$
            `np.ndarray`.
        return_logits : bool
            Whether the models output-logits should be returned. Otherwise, the
            actual categoritcal predictions get returned (via argmax)

        Returns
        -------
        np.ndarray
            The label predictions.
        """
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits: torch.Tensor = self.model.forward(X)
            if return_logits:
                return logits
            pred: torch.Tensor = torch.argmax(logits, axis=1)
            return pred.cpu().numpy()

    def reset(self) -> None:
        """Resets the model to its "untrained" state.
        """
        super().reset()
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def train(self) -> None:
        """Sets the kernel model to its training-state.
        """
        self.model.train()

    def eval(self) -> None:
        """Sets the kernel model to its evaluation-state.
        """
        self.model.eval()

    def save(self, path: str, state_dict_only: bool=False) -> None:
        """Save classifier to disc. A conventional file-suffix for pytorch-files
        is ".pt".

        Parameters
        ----------
        path : str
        """
        if state_dict_only:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model, path)

    def load_model(self, path: str) -> None:
        """Load the classifiers internal model from a passed filepath. For this
        approach use a previously saved model's `.state_dict()`. If a you want
        to load a full model use the static load.

        Parameters
        ----------
        path : str
        """
        self.is_pretrained_model = True
        self.model.load_state_dict(torch.load(path))

    @staticmethod
    def load(
        path: str,
        model: nn.Module=None,
        max_epochs: int=500,
        device: Literal["cpu", "cuda"]="cuda",
    ) -> "PyTorchClassifier":
        """Load the classifier from the passed filepath.

        Parameters
        ----------
        path : str
        model : nn.Module
            A instance of the kernel networt with suitable hyperparameters set.
            Can be used if the state-dict approach is preferred, or if the model
            data is only available as a state dict.
        max_epochs : int
            Epochs setting for training (passed to constructor).
        device : Literal["cpu", "cuda"]
            Device to load the model to  (passed to constructor).
        """
        if model is not None:
            model.load_state_dict(torch.load(path))
        else:
            model = torch.load(path)
            if isinstance(model, OrderedDict):
                raise ValueError(
                    "The path-file is a pytorch-state-dict, i.e., you must also " +
                    "pass a suitable pytorch model-instance."
                )
        clf = PyTorchClassifier(
            model=model,
            max_epochs=max_epochs,
            device=device,
        )
        clf.is_pretrained_model = True
        return clf





class DenseNN(nn.Module):
    """An easy fully connected / dense network for classification with
    dropout regularization. Implements a `base_forward` method such that it is
    usable (but does not have to be used) with the [PyTorchClassifier][qutools.core.classifier.PyTorchClassifier]
    wrapper.
    """
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        p_dropout: float=0.1,
        n_hidden_nodes: int=64,
        n_hidden_layers: int=1
    ):
        """Sets up a simple dense-classifier using PyTorch.

        Parameters
        ----------
        in_features : int
            The number of input features.
        n_classes : int
            The number of possible classes for the labels.
        p_dropout : floa
            The dropout ratio
        n_hidden_nodes : int
            The number of hidden notes per layer
        n_hidden_layers : int
            The number of hidden layers
        """
        super().__init__()
        self.dense_in = nn.Linear(in_features=in_features, out_features=n_hidden_nodes)
        self.dropout_in = nn.Dropout(p_dropout)

        self.hidden = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden.append(nn.Linear(in_features=n_hidden_nodes, out_features=n_hidden_nodes))
            self.dropouts.append(nn.Dropout(p_dropout))

        self.dense_out = nn.Linear(in_features=n_hidden_nodes, out_features=n_classes)

    def base_forward(self, X: torch.Tensor) -> torch.Tensor:
        """A method to provide intermediate representations of the input data,
        in this case the output of the last hidden layer.

        Parameters
        ----------
        X : torch.Tensor
            The (typically numeric) features as a $N_{\\mathrm{samples}} \\times N_{\\mathrm{features}}$
            `torch.Tensor`.

        Returns
        -------
        torch.Tensor
            The intermediate representations as a  $N_{\\mathrm{samples}} \\times N_{\\mathrm{hidden nodes}}$
            `torch.Tensor`.
        """
        X = self.dense_in(X)
        X = F.relu(X)
        X = self.dropout_in(X)

        for lyr, drp in zip(self.hidden, self.dropouts):
            X = lyr(X)
            X = F.relu(X)
            X = drp(X)

        return X

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """The full `forward` method, that makes use of the `base_forward` method.

        Parameters
        ----------
        X : torch.Tensor
            The (typically numeric) features as a $N_{\\mathrm{samples}} \\times N_{\\mathrm{features}}$
            `torch.Tensor`.

        Returns
        -------
        torch.Tensor
            The classification logits as a $N_{\\mathrm{samples}} \\times N_{\\mathrm{classes}}$
            `torch.Tensor`.
        """
        X = self.base_forward(X)
        X = self.dense_out(X)
        X = F.softmax(X, dim=-1)
        return X
