"""
# Wrappers for Scikit-Cluster models

This submodule introduces a unified interface for clustering models, particularly
those from scikit-learn. It defines an abstract base class, `ClusterWrapper`, which wraps around scikit-learn's
clustering algorithms to provide a consistent method of accessing labels, fitting models, and (where applicable)
making predictions on new data. This approach simplifies the use of different clustering models within a
larger application by standardizing the interface across various clustering algorithms.

Usage
-----
The `ClusterWrapper` class is designed to be subclassed for specific clustering algorithms. Subclasses must
implement the `can_predict`, `transform`, `clusters`, and `n_clusters` abstract methods. The `can_predict` method should return a boolean
indicating whether the model can predict cluster labels for unseen data. The `transform` method should provide
the mechanism for predicting new cluster allocations for unseen data, raising an error or warning if the model
does not support this functionality.

Example
-------
Below is an example of how to subclass `ClusterWrapper` for the Gaussian-Mixture-Model clustering algorithm, implementing the
required abstract methods.

```python

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from qutools.clustering.cluster_wrapper import ClusterWrapper

class GMWrapper(ClusterWrapper):
    def __init__(self, kernel: GaussianMixture):
        self.kernel = kernel
        self._trn_labels = None

    def fit(self, X_trn: np.ndarray) -> None:
        self.kernel.fit(X_trn)
        self._trn_labels = self.kernel.predict(X_trn)

    def can_predict(self) -> bool:
        return True

    def transform(self, X_pre: np.ndarray) -> np.ndarray:
        return self.kernel.predict(X_pre)

    def clusters(self) -> np.ndarray:
        return self._trn_labels

    def n_clusters(self) -> tuple[int, bool]:
        return self.kernel.n_components, False

X, y = make_blobs(n_samples=1000, n_features=2, centers=3, random_state=5555)
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2)
df = pd.DataFrame(X_trn, columns=["x", "y"])
df["cls"] = y_trn
df.plot.scatter(x="x", y="y", c="cls", colormap="viridis")

gmm = GaussianMixture(n_components=3)
wrapper = GMWrapper(gmm)
wrapper.fit(X_trn)
y_tst_pred = wrapper.transform(X_tst)

df_pred = pd.DataFrame(X_tst, columns=["x", "y"])
df_pred["cls"] = y_tst_pred
df_pred.plot.scatter(x="x", y="y", c="cls", colormap="viridis")

```

<p align="center" width="80%">
   <img src="../../../assets/img/cluster_wrapper.png" width="80%">
</p>

"""

import numpy as np

from abc import ABC, abstractmethod
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture


class ClusterWrapper(ABC):
    def __init__(self, kernel: ClusterMixin) -> None:
        """A wrapper around (typically) scikit-learn cluster models that is
        intended to provide a unified interface to labels and predictions.

        Parameters
        ----------
        kernel : ClusterMixin
            The actual cluster method.
        """
        self.kernel = kernel

    def __str__(self):
        return f"{self.__class__.__name__}[{self.kernel.__str__()}]"

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.kernel.__repr__()}]"

    def fit(self, X: np.ndarray):
        """Fits the cluster model.

        Parameters
        ----------
        X : np.ndarray
            Data to fit the cluster model to.
        """
        self._X_fit = X
        self.kernel.fit(X)

    @abstractmethod
    def can_predict(self) -> bool:
        """Denoting wether the model can predict cluster labels for unseen data.

        Returns
        -------
        bool
        """

    @abstractmethod
    def transform(self, X_pre: np.ndarray) -> np.ndarray:
        """The method to "predict" new cluster-allocations for unseen data
        `X_pre`. It is suggested to raise an error or at least send a warning,
        for wrappers of cluster-models which are not able to do such predictions.
        Alternative, some models might not implement this by default, but there
        might be suitable ways to implement this yourself. For some discussion
        on this topic, maybe take a look at [this informative stackoverflow thread](https://stackoverflow.com/q/27822752).

        Parameters
        ----------
        X_pre : np.ndarray
            New unseen data to generate the cluster allocations for.

        Returns
        -------
        np.ndarray
        """
        pass

    @abstractmethod
    def clusters(self) -> np.ndarray:
        """Method to return the clusters observed / generated at fit-time. This
        is provided by every cluster model. Some models have a `.predict` method
        that can be applied to the fit-data (`self._X_fit`), some models
        (e. g. DBSCAN and HDBSCAN) might alternatively internally store the
        `self.labels_`-attribute which contains exactly these allocations.

        Returns
        -------
        np.ndarray
        """
        pass

    @abstractmethod
    def n_clusters(self) -> tuple[int, bool]:
        """Returns the number of cluster, and whether there existis a
        noise cluster.

        Returns
        -------
        tuple[int, bool]
            Total number of clusters (including noise, if it exists) and
            whether a noise cluster exists.
        """
        pass



class KMeansWrapper(ClusterWrapper):
    """A ClusterWrapper for the KMeans clustering algorithm."""

    def __init__(self, kernel: KMeans) -> None:
        """Initializes the Wrapper. Takes a sci-kit learn KMeans object.

        Parameters
        ----------
        kernel : KMeans
            The KMeans object to wrap.
        """
        self.kernel = kernel

    def transform(self, X_pre: np.ndarray) -> np.ndarray:
        """Predicts the clusters for new instances. In case of KMeans, this is
        equivalent to calling the `.predict` method.

        Parameters
        ----------
        X_pre : np.ndarray
            Prepared score-data of the same shape and type as the data used
            for model setup, i. e. during `.fit`.
        """
        return self.kernel.predict(X_pre)

    def predict(self, X_pre: np.ndarray) -> np.ndarray:
        """Predicts the clusters for new instances.

        Parameters
        ----------
        X_pre : np.ndarray
            Prepared score-data of the same shape and type as the data used
            for model setup, i. e. during `.fit`.

        Returns
        -------
        np.ndarray
            The predicted clusters.
        """
        return self.transform(X_pre=X_pre)

    def can_predict(self) -> bool:
        """For the use within the Clustering approaches. Is `True` for KMeans.

        Returns
        -------
        bool
            Always `True`.
        """
        return True

    def clusters(self) -> np.ndarray:
        """Returns the cluster-allocations calculated at fit-time for the fit-data.

        Returns
        -------
        np.ndarray
            The cluster labels of the fit-data.
        """
        return self.kernel.predict(self._X_fit)

    def n_clusters(self) -> tuple[int, bool]:
        """Returns the number of cluster, and whether there existis a noise cluster.

        Returns
        -------
        tuple[int, bool]
            Total number of clusters (including noise, if it exists) and
            whether a noise cluster exists.
        """
        return self.kernel.n_clusters, False



class DBSCANWrapper(ClusterWrapper):
    """A ClusterWrapper for the DBSCAN and HDBSCAN clustering algorithm."""

    def __init__(self, kernel: DBSCAN | HDBSCAN) -> None:
        """Initializes the Wrapper. Takes a scikit-learn DBSCAN or HDBSCAN object.

        Parameters
        ----------
        kernel : DBSCAN | HDBSCAN
            The (H)DBSCAN object to wrap.
        """
        self.kernel = kernel

    def transform(self, X_pre: np.ndarray) -> np.ndarray:
        """Predicts the clusters for new instances. In case of (H)DBSCAN, this is
        not implemented by default. If you whish to predict clusters for new
        instances, you might want to consider to specify your own ClusterWrapper
        or augment this one. Refer to the following stackoverflow thread for
        inspiration an discussion: https://stackoverflow.com/q/27822752.

        Raises
        ------
        NotImplementedError
            If the method is called.
        """
        raise NotImplementedError(
            "Predicting clusters for new instances is not implemented for the" +
            "\n\tDBSCAN clustering method."
        )

    def can_predict(self) -> bool:
        """For the use within the Clustering approaches. Is `False` for the current
        implementation. Refer to the `.transform` method documentation for furhter
        details.

        Returns
        -------
        bool
            Always `False`.
        """
        return False

    def fit(self, X):
        """Fits the cluster model.

        Parameters
        ----------
        X : np.ndarray
            Data to fit the cluster model to.
        """
        super().fit(X)
        unq_lbl = set(self.kernel.labels_)
        self._n_cls = len(unq_lbl)
        self._noise_cls = -1 in unq_lbl

    def clusters(self) -> np.ndarray:
        """Returns the cluster-allocations calculated at fit-time for the fit-data.

        Returns
        -------
        np.ndarray
            The cluster labels of the fit-data.
        """
        return self.kernel.labels_

    def n_clusters(self) -> tuple[int, bool]:
        """Returns the number of cluster, and whether there existis a noise cluster.

        Returns
        -------
        tuple[int, bool]
            Total number of clusters (including noise, if it exists) and
            whether a noise cluster exists.
        """
        return self._n_cls, self._noise_cls


class GMWrapper(ClusterWrapper):
    """A ClusterWrapper for the Gaussian-Mixture-Model clustering algorithm.
    Can be used for Latent-Profile/Class-"style"-Analyses.
    """

    def __init__(self, kernel: GaussianMixture):
        """Initializes the Wrapper. Takes a scikit-learn GaussianMixture object.

        Parameters
        ----------
        kernel : GaussianMixture
            The GaussianMixture object to wrap.
        """
        self.kernel = kernel
        self._trn_labels = None

    def fit(self, X: np.ndarray) -> None:
        """Fits the cluster model.

        Parameters
        ----------
        X : np.ndarray
            Data to fit the cluster model to.
        """
        self.kernel.fit(X)
        self._trn_labels = self.kernel.predict(X)

    def can_predict(self) -> bool:
        """Whether the model can predict cluster labels for unseen data. For
        GaussianMixture, this is `True`.

        Returns
        -------
        bool
            Always `True`.
        """
        return True

    def transform(self, X_pre: np.ndarray) -> np.ndarray:
        """Transforms the data `X_pre` to cluster allocations. In case of
        GaussianMixture, this is equivalent to calling the `.predict` method.

        Parameters
        ----------
        X_pre : np.ndarray
            New unseen data to generate the cluster allocations for.
        """
        return self.kernel.predict(X_pre)

    def clusters(self) -> np.ndarray:
        """Returns the cluster-allocations calculated at fit-time for the fit-data.

        Returns
        -------
        np.ndarray
            The cluster labels of the fit-data.
        """
        return self._trn_labels

    def n_clusters(self) -> tuple[int, bool]:
        """Returns the number of cluster, and whether there existis a noise cluster.

        Returns
        -------
        tuple[int, bool]
            Total number of clusters (including noise, if it exists) and
            whether a noise cluster exists.
        """
        return self.kernel.n_components, False
