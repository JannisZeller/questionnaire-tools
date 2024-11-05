"""
# Exploratory questionnaire score-cluster analyses

This submodule is used for exploratory cluster analyses of questionnaire scores.
It implements various ways to evaluate and visualize the clusters and models,
including the use of dimensionality reduction techniques for the visualization
of the cluster centroids and silhouette-score- and elbow-plots for k-selection
when using the K-Means algorithm. It is relatively strongly taylored towards the
K-Means algorithm. Some of the methods might not work with other clustering
algorithms, and might be need to be adapted or extended for other algorithms.

Example
-------
Below is an example for applying the `QuScoreClusters` class to a `QuData` instance.
Additionally some subscales are applied prior to applying the clustering model
usign a `QuSubscales` instance. The data used has been setup synthetically
such that it can be inclueded in the package. Real data can not be included
here, because it might violate data privacy regulations. Be advised to take
a look at the synthetic data used here (find it in the `qutools/_toydata`-directory)
to gain a better understanding of the required data-structure and the setup.

```python
from qutools.data.config import QuConfig
from qutools.data.data import QuData
from qutools.data.subscales import QuSubscales
from qutools.clusters import QuScoreClusters, quclst_silhouette_plot

quconfig = QuConfig.from_yaml("qutools/_toydata/quconfig.yaml")

qusubscales = QuSubscales(
    quconfig=quconfig,
    df_cat="qutools/_toydata/qusubscales.csv",
)

qudata = QuData(
    quconfig=quconfig,
    df_scr="qutools/_toydata/df_scr_syn.csv",
)
```
```output
> All scores in correct ranges. ✓
> Validated score-columns. ✓
```
```python
quclst = QuScoreClusters(
    qudata=qudata,
    qusubscales=qusubscales,
    n_clusters=4,
)

quclst_silhouette_plot(quclst=quclst)
```
<p align="center">
   <img src="../../../assets/img/clusters1.png" width="80%">
</p>

```python
quclst.set_cluster_labels({1: "low", 2: "mid-c", 3: "mid-a", 4: "high"})
quclst.centroid_lineplot()
```
<p align="center">
   <img src="../../../assets/img/clusters2.png" width="90%">
</p>

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from joblib import load as jl_load
from joblib import dump as jl_dump
from json import load as json_load
from json import dump as json_dump

from pathlib import Path
from typing import Literal, Callable, Self
from copy import deepcopy


from .cluster_wrapper import ClusterWrapper, DBSCANWrapper, KMeansWrapper
from ..data.config import QuConfig
from ..data.data import QuData
from ..data.subscales import QuSubscales
from ..data.subscales import SubscalesDimReducer

from ..core.io import empty_or_create_dir, path_suffix_warning



class ScoreClustersError(Exception):
    """An exception class for QuScoreClusters-objects.
    """
    pass




class QuScoreClusters:

    def __init__(
        self,
        qudata: QuData,
        qusubscales: QuSubscales=None,
        scaling: Literal["social", "absolute", "none"]="social",
        cluster_method: Literal["KMeans", "DBSCAN"]|ClusterWrapper="KMeans",
        drop_incomplete: bool=True,
        drop_earlystoppers: bool=True,
        **kwargs,
    ):
        """A class for the analysis of clusters in the questionnaire-scores.

        Parameters
        ----------
        qudata : QuestionnaireData
            A `QuestionnaireData`-instance.
        qusubscales: QuestionnaireSubscales
            An optional `QuestionnaireSubscales`-instance. The `QuestionnaireConfig`
            instances of `qudata` and `qusubscales` must be identical.
        scaling : Literal["social", "absolute", "none"]
            Whether the scores should (not) be scaled to the best achieved value
            ("social") the maximum achievable score ("absolut").
        cluster_method : Literal["KMeans", "DBSCAN", "HDBSCAN"]|CustomClusterWrapper
            The cluster method to apply. If a CustomClusterKernel is used refer
            to its documentation on how to wrap Scikit-Cluster models (
            [sklearn.clusters](https://scikit-learn.org/stable/modules/clustering.html)).
        drop_incomplete : bool
            Whether for the fitting of the cluster model incomplete instances
            should be dropped.
        drop_earlystoppers : bool
            Whether for the fitting of the cluster model earlystopped instances
            should be dropped.
        """
        self.qudata = qudata
        self.quconfig: QuConfig = deepcopy(qudata.quconfig)
        self.scaling = scaling
        self.cluster_method = cluster_method
        self.drop_incomplete = drop_incomplete
        self.drop_earlystoppers = drop_earlystoppers
        self.verbose=kwargs.pop("verbose", False)
        self.incomplete_threshold: float = kwargs.pop("incomplete_threshold", 0.5)
        self.earlystopping_threshold: float = kwargs.pop("earlystopping_threshold", 0.25)
        self.random_state = kwargs.pop("random_state", 42)
        cluster_labels = kwargs.pop("cluster_labels", None)

        self.pipeline = Pipeline([])
        self.qusubscales = qusubscales

        if not kwargs.pop("__supress_fit", False):
            if qusubscales is not None:
                if not qusubscales.quconfig == qudata.quconfig:
                    raise ScoreClustersError(
                        "The `QuestionnaireConfig`s of the passed data (`qudata`) " +
                        "\n\tand the passed subscales (`qusubscales`) are not equal."
                    )
                self.__set_dim_reducer()
                if self.verbose:
                    print("Setup subscales dim-reducer. ✓")

            if self.scaling != "none":
                self.__set_scaler()
                if self.verbose:
                    print(f"Setup scaler (strategy: {scaling}). ✓")

            self._clusterer_kwargs = kwargs
            self.__set_clusterer(**kwargs)
            if self.verbose:
                print(f"Setup clusterer model (cluster method: {cluster_method}). ✓")

            self.__sort_cluster_labels()
            if self.verbose:
                print("Setup clusterer-sorter. ✓")

            self.noise_cls = self._has_noise_cls()


            if cluster_labels is not None:
                self.set_cluster_labels(cluster_labels)



    ## Internals
    def __str__(self) -> str:
        s = "QuScoreClusters("
        s += f"\n\t- Config: {str(self.quconfig)}"
        s += f"\n\t- Data: {self.qudata.short_str()}"
        if self.drop_incomplete:
            s += f"\n\t- Incomplete dropped (<{self.incomplete_threshold})"
        if self.drop_earlystoppers:
            s += f"\n\t- Earlystoppers dropped (<{self.earlystopping_threshold})"
        s += f"\n\t- Cluster-Model: {self.clusterer.__str__()}"
        if self.qusubscales is not None:
            s += f"\n\t- Subscales: {self.qusubscales.get_subscales()}"
        if hasattr(self, "_cluster_label_dict"):
            s += f"\n\t- Cluster-Labels: {list(self._cluster_label_dict.values())}"
        s += "\n)"
        return s

    def __repr__(self) -> str:
        return self.__str__()

    #   General utils
    def __drop_id(self, df: pd.DataFrame) -> pd.DataFrame:
        # Dropping some IDs from a dataframe
        if self.quconfig.id_col in df.columns:
            df = df.drop(columns=self.quconfig.id_col)
        return df

    def _get_fit_scores(self, with_id: bool=False) -> pd.DataFrame:
        # Returns the scores used for fitting
        df_scr = self.qudata.get_scr(
            verbose=self.verbose,
            drop_incomplete=self.drop_incomplete,
            drop_earlystoppers=self.drop_earlystoppers,
            incomplete_threshold=self.incomplete_threshold,
            earlystopping_threshold=self.earlystopping_threshold,
        )
        if not with_id:
            df_scr = df_scr.drop(columns=self.quconfig.id_col)
        return df_scr.fillna(0)

    def _apply_pre_pipeline(self, X_scr: np.ndarray) -> np.ndarray:
        # Apply the pipeline, excluding the actual cluster model (i.e., subscales / scalers).
        idx = 0
        if hasattr(self, "dim_reducer"):
            if (
                self.dim_reducer in self.pipeline.steps or
                self.dim_reducer in self.pipeline
            ):
                idx += 1
        if hasattr(self, "scaler"):
            if (
                self.scaler in self.pipeline.steps or
                self.scaler in self.pipeline
            ):
                idx += 1
        if idx > 0:
            X_scr = self.pipeline[:idx].transform(X_scr)
        return X_scr

    def _has_noise_cls(self) -> bool:
        # Wether the model contains a noise cluster
        return -1 in self._cluster_sort_dict

    def _get_cluster_label_dict(self) -> dict:
        # Returns the cluster label dict if it is set
        if hasattr(self, "_cluster_label_dict"):
            return self._cluster_label_dict
        else:
            return None

    #   Pipeline-utils
    def __set_dim_reducer(self) -> None:
        # Sets the subscales-dim-reducer
        self.dim_reducer = SubscalesDimReducer(self.qusubscales)
        self.pipeline.steps.append(("DimReducer", self.dim_reducer))

    def __set_scaler(self) -> None:
        # Sets the scaler
        df_scr = self._get_fit_scores(with_id=True)

        if self.scaling == "absolute":
            df_max = pd.DataFrame(self.quconfig.get_max_scores(), index=["max"])
            df_max = df_max.reset_index(names=self.quconfig.id_col)
            df_scr = pd.concat([df_scr, df_max]).reset_index(drop=True)
            df_scr = df_scr.fillna(0)
            scaler_name = "AbsoluteScaler"

        if self.scaling == "social":
            scaler_name = "SocialScaler"

        X_scr = df_scr.drop(columns=self.quconfig.id_col).values
        if hasattr(self, "dim_reducer"):
            X_scr = self.pipeline.transform(X_scr)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(X_scr)
        self.pipeline.steps.append((scaler_name, self.scaler))

    def __set_clusterer(self, **kwargs) -> None:
        # Sets the actual cluster model
        if isinstance(self.cluster_method, str):
            if self.cluster_method not in ["KMeans", "DBSCAN", "HDBSCAN"]:
                raise NotImplementedError(
                    "Currently only the `cluster_method`s \"KMeans\", \"DBSCAN\" and \"HDBSCAN\""+
                    "\n\tare implemented. Other cluster methods can be appended using custom"
                    "\n\tScikit-Learn models."
                )

        df_scr = self._get_fit_scores()
        X_scr = self._apply_pre_pipeline(df_scr.values)

        if self.cluster_method == "KMeans":
            if "n_clusters" not in kwargs:
                print(
                    "Info: No number of clusters (`n_clusters`-argument) passed, but `cluster_method==\"KMeans\"`." +
                    "\n\tDefaulting to 4."
                )
            n_clusters = kwargs.pop("n_clusters", 4)
            self.clusterer = KMeansWrapper(KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                **kwargs
            ))
        elif self.cluster_method == "DBSCAN":
            self.clusterer = DBSCANWrapper(DBSCAN(**kwargs))
        elif self.cluster_method == "HDBSCAN":
            try:
                from sklearn.cluster import HDBSCAN
            except ModuleNotFoundError as e:
                raise ScoreClustersError(
                    "To enable the usage of HDBSCAN as a cluster-method, you have to install the hdbscan-package." +
                    f"\n Original Exception: {e}"
                )
            self.clusterer = DBSCANWrapper(HDBSCAN(**kwargs))
        else:
            self.clusterer = self.cluster_method

        self.clusterer.fit(X_scr)
        self.pipeline.steps.append((self.cluster_method, self.clusterer))

    def __sort_cluster_labels(self) -> None:
        # Stores a dict to sort the cluster labels with increasing total score
        df_scr = self._get_fit_scores()
        df_scr["total"] = df_scr.sum(axis=1)

        preds = self.clusters()["cluster"].values
        df_scr["cluster"] = preds
        means: dict = (df_scr[["total", "cluster"]]
            .groupby("cluster")
            .mean()
            .to_dict()["total"]
        )

        noise_cls = False
        if -1 in means.keys():
            noise_cls = True
            del means[-1]

        means = dict(sorted(means.items(), key=lambda item: item[1]))
        replacement = np.arange(1, len(means) + 1).tolist()
        sort_dict = dict(zip(list(means.keys()),  [str(int(x)) for x in replacement]))

        if noise_cls:
            sort_dict[-1] = "noise"

        self._cluster_sort_dict = sort_dict

    def __apply_cluster_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Applies the cluster-sort-dict and cluster-label-dict, the latter only if present
        if hasattr(self, "_cluster_sort_dict"):
            df["cluster"] = df["cluster"].astype(float).replace(self._cluster_sort_dict)
        if hasattr(self, "_cluster_label_dict"):
            str_dict = {str(k): v for k, v in self._cluster_label_dict.items()}
            df["cluster"] = df["cluster"].astype(str).replace(str_dict)
        return df


    ## Instance-wise results
    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def pre_cluster_scores(self) -> pd.DataFrame:
        """Returns pre-cluster scores (potentially scaled and subscaled)
        only for the data used for setting up the model, i.e., dropout filters
        might be applied.
        Is identical to `.pre_cluster_scores_all`, if no dropout was used initially.

        Returns
        -------
        pd.DataFrame
        """
        df_scr = self._get_fit_scores(with_id=True)
        return self._compute_pre_cluster_scores(df_scr)


    def pre_cluster_scores_all(self) -> pd.DataFrame:
        """Returns the pre-cluster scores (potentially scaled and subscaled) of
        the initial data, without any dropout etc.

        Returns
        -------
        pd.DataFrame
        """
        df_scr = self.qudata.get_scr(
            verbose=self.verbose,
            drop_incomplete=False,
            drop_earlystoppers=False,
        )
        return self._compute_pre_cluster_scores(df_scr)


    def clusters(self) -> pd.DataFrame:
        """Returns labels only for the data used for setting up the model.
        Is identical to `.predict_all`, if no dropout was used initially.

        Returns
        -------
        pd.DataFrame
        """
        df_scr = self._get_fit_scores(with_id=True)

        df = self._compute_pre_cluster_scores(df_scr)
        df["cluster"] = self.clusterer.clusters()
        df = self.__apply_cluster_labels(df)

        return df


    def clusters_all(self, mute_warning: bool=False) -> pd.DataFrame:
        """Get cluster predictions for initial input data complete input data,
        i.e., without potential dropout filtering. Is not possible with all
        cluster models because not all enable the allocation of samples that
        were not included in the model fitting.

        Parameters
        ----------
        mute_warning : bool
            Whether to mute the "clustering is not supervised learning" warning.

        Returns
        -------
        pd.DataFrame
        """
        return self.transform(self.qudata, return_clusters=True, mute_warning=mute_warning)


    def clusters_most(self, qudata: QuData=None) -> pd.DataFrame:
        """Gets the "most" cluster labels possible. If the internal cluster
        model has the capability to predict labels for unseen data, the
        `.predict` method is used if `qudata` is passed or the `clusters_all`
        method is used. Else only the
        `.clusters` method is used and potentially dropped out instances are
        not returned.

        Parameters
        ----------
        qudata : QuData
            Questionnaire data to predict the clusters for. If `None` the
            clusters are returned by ID.

        Returns
        -------
        pd.DataFrame
        """
        if self.clusterer.can_predict():
            if qudata is not None:
                return self.transform(qudata=qudata, return_clusters=True)
            else:
                return self.clusters_all(mute_warning=True)
        else:
            if qudata is not None:
                print(
                    "Warning: You requested `cluster_most` for some QuData passed. \n" +
                    "The cluster model can only return the cluster allocations \n" +
                    "for instances seen when setting up the model tough. \n" +
                    "Therefore, only these will be returned and matched to the data."
                )
            return self.clusters()


    def _compute_pre_cluster_scores(self, df_scr: pd.DataFrame) -> pd.DataFrame:
        """Applys the pipeline until (not including) the clustering step, i.e.,
        applys dimensionality reduction using the `QuestionnaireSubscales`
        and scaling (both if specified in the model-initialization).

        Parameters
        ----------
        df_scr : pd.DataFrame
            The data to pass through.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the preprocessed scores (potentially scaled
            and subscaled).
        """
        id_col = self.quconfig.id_col
        X_scr = self.__drop_id(df_scr).fillna(0).values

        if hasattr(self, "dim_reducer"):
            subscales = self.qusubscales.get_subscales()

        X_scr = self._apply_pre_pipeline(X_scr)
        df_ret = pd.DataFrame(X_scr)

        if hasattr(self, "dim_reducer"):
            df_ret.columns = subscales
        else:
            df_ret.columns = list(self.quconfig.get_task_names())

        if id_col in df_scr:
            ret_cols = df_ret.columns.to_list()
            df_ret[id_col] = df_scr[id_col].values
            df_ret = df_ret[[id_col] + ret_cols]

        return df_ret


    def transform(
        self,
        qudata: QuData,
        return_clusters: bool=False,
        mute_warning: bool=False,
    ) -> pd.DataFrame:
        """Transforms the score data accoring to the cluster-preprocessing
        pipeline. If `return_clusters` also trys to predict clusters, but this
        is not possible / might not make sense with all cluster models.

        Parameters
        ----------
        qudata : QuData,
            Questionnaire Data to transform.
        return_clusters : bool
            Wether clusters should be returned for the passed data.
        mute_warning : bool
            Whether to mute the "clustering is not supervised learning" warning.

        Returns
        -------
        pd.DataFrame
            Input dataframe transformed accoring to the pipeline, optionally
            with new "cluster" column containing the "predictions".
        """
        df_scr = qudata.get_scr(
            verbose=self.verbose,
            drop_incomplete=False,
            drop_earlystoppers=False,
        )
        df_scr = df_scr.fillna(0)
        df = self._compute_pre_cluster_scores(df_scr)
        if return_clusters:
            if not mute_warning:
                print(
                    "Warning: Clustering is not supervised learning. Prediction of\n" +
                    "\tclasslabels for new or unseen data might be inappropiate\n" +
                    "\tand might not even work, depending on the cluster model."
                )
            df_scr = self.__drop_id(df_scr)
            X_scr = df_scr.values
            X_pre = self._apply_pre_pipeline(X_scr)
            df["cluster"] = self.clusterer.transform(X_pre)
            df = self.__apply_cluster_labels(df)

        return df


    def predict(self, X_scr: np.ndarray) -> np.ndarray:
        X_pre = self._apply_pre_pipeline(X_scr)
        df = pd.DataFrame()
        df["cluster"] = self.clusterer.transform(X_pre)
        df = self.__apply_cluster_labels(df=df)
        return df["cluster"].values


    def n_clusters(self) -> tuple[int, bool]:
        """Returns the number of clusters.

        Returns
        -------
        tuple[int, bool]
            The first entry indicates the total number of clusters (including an
            optional noise cluster), the second entry denotes whether a
            noise-clusters is present (depends on used cluster-model).
        """
        return self.clusterer.n_clusters()


    ## Labelling
    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def set_cluster_labels(self, cluster_labels: dict) -> Self:
        """Set labels for the clusters using a dict. The labels are set replacing
        any currently existing labels. If you want to label the
        (potential) noise-cluster, use the key `"noise"`. Keys and values
        get string-casted for comparison and for the presentation using
        pandas dataframes.
        This is primarily intended for interactive usage replacing potentially
        already set labels.

        Parameters
        ----------
        cluster_labels : dict

        Returns
        -------
        Self
            For convenience.
        """
        cluster_labels = {str(k): str(v) for k, v in cluster_labels.items()}
        new_keys = list(cluster_labels.keys())
        new_labels = list(cluster_labels.values())
        if len(set(new_labels)) != len(cluster_labels):
            raise ScoreClustersError(
                "The values for cluster-renaming must be unique."
            )

        if not hasattr(self, "_cluster_label_dict"):
            cluster_numbers = list(self._cluster_sort_dict.values())

            missing_keys = [
                key for key in new_keys if key not in cluster_numbers
            ]
            if len(missing_keys) > 0:
                raise ScoreClustersError(
                    f"The key(s) {missing_keys} of the renaming dict are not all in the available cluster numbers."
                )

            self._cluster_label_dict = cluster_labels
            for k in cluster_numbers:
                if k not in self._cluster_label_dict:
                    self._cluster_label_dict[k] = k

        else:
            old_labels = list(self._cluster_label_dict.values())
            if any([key not in old_labels for key in new_keys]):
                raise ScoreClustersError(
                    "The keys of the renaming dict are not all in the available old labels."
                )
            remap_dict = {}
            for k, v in self._cluster_label_dict.items():
                if v in new_keys:
                    remap_dict[k] = cluster_labels[v]
                else:
                    remap_dict[k] = v

            self._cluster_label_dict = remap_dict

            info_dict = {k: v for k, v in remap_dict.items() if k!=v}
            print(
                "Info: You have set new cluster labels multiple times. \n" +
                "This can lead to unexpected behavior, when running the code in \n" +
                "a non-interactive fashion. The current cluster-labels can be set \n" +
                "directly using:\n" +
                "```python\n" +
                f"\t<ScoreCluster-instance name>.set_cluster_labels({info_dict})\n"
                "```"
            )

        return self


    def drop_cluster_labels(self) -> Self:
        """Drops the current cluster labels.

        Returns
        -------
        Self
            For convenience.
        """
        del self._cluster_label_dict


    ## Results
    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def get_cluster_order(self) -> list:
        """Returns the cluster order in a list.

        Returns
        -------
        list
        """
        if hasattr(self, "_cluster_label_dict"):
            order = {}
            for k, v in self._cluster_label_dict.items():
                if k == "noise":
                    order[v] = np.inf
                else:
                    order[v] = int(k)
            order_list = list(order.keys())
            order_list.sort(key=lambda k: order[k])
            return order_list

        else:
            return list(self._cluster_sort_dict.values())


    def get_cluster_aggregations(
        self,
        funcs: list[str|Callable[[np.ndarray], float]]=["mean", "std"],
    ) -> pd.DataFrame:
        """Returns the cluster centers as a dataframe. Here, the data used to
        setup the model, i.e., potentially with applied dropout filters, is used.

        Parameters
        ----------
        funcs: list[str|Callable[[np.ndarray], float]]
            Functions to aggregate by. Typical choices are `"mean"` and `"std"`.
            Optionally, the list elements can be tuples of the structure
            `tuple[str, str|Callable]`, where the first entry defines the column
            name of the aggregation in the second entry. If a function is passed
            it must map an array-like to a single float.
            For more infotmation on what can be passed see
            [pandas.core.groupby.DataFrameGroupBy.aggregate](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.aggregate.html).

        Returns
        -------
        pd.DataFrame
        """
        df = self.clusters()
        df = pd.merge(df, self.qudata.get_total_score())
        df = df.drop(columns=self.quconfig.id_col)

        df_aggs = df.groupby("cluster").aggregate(funcs)
        ds_counts = df.groupby("cluster").size()
        df_counts = pd.DataFrame(
            ds_counts,
            columns=pd.MultiIndex.from_tuples([("count", "")]),
        )

        df_ret = pd.concat([df_counts, df_aggs], axis=1)
        df_ret = df_ret.reset_index(names=["cluster"])

        if hasattr(self, "_cluster_label_dict"):
            order = {}
            for k, v in self._cluster_label_dict.items():
                if k == "noise":
                    order[v] = np.inf
                else:
                    order[v] = int(k)

            df_ret = df_ret.sort_values(
                by=["cluster"],
                key=lambda col: col.map(lambda idx: order[idx]),
            )
            df_ret = df_ret.reset_index(drop=True)

        return df_ret


    def get_df_centroids(self) -> pd.DataFrame:
        """Returns the centroids (means) of the clusters by aggregating the
        data.

        Returns
        -------
        pd.DataFrame
        """
        df = self.get_cluster_aggregations(["mean"])
        df.columns = [col[0] for col in df.columns]
        df = df.drop(columns=["total_score", "count"])
        return df


    def store_2dim_data(
        self,
        method: Literal["PCA", "TSNE"]="PCA",
    ) -> dict[str, pd.DataFrame|PCA]:
        """Returns a two-dimensional representation of the data. Currently only
        implemented with PCA-dimensionality reduction

        Returns
        -------
        pd.DataFrame
            _description_
        """
        df_pre = self.clusters()
        df_ctr = self.get_df_centroids()
        if method == "PCA":
            return _quclst_pca_data(self, df_pre, df_ctr)
        elif method == "TSNE":
            return _quclst_tsne_data(self, df_pre, df_ctr)
        else:
            raise NotImplementedError(
                "2-Dimensional data reduction is only implemented for `methon in [ \"PCA\"], \"TSNE\"]`."
            )


    ## Plots
    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def scatter_plot(
        self,
        dim_reduce_method: Literal["PCA", "TSNE"]="PCA",
        savepath: str=None,
        **kwargs
    ) -> Figure:
        """2-dimensional scatterplot of the cluster results.

        Parameters
        ----------
        dim_reduce_method : Literal["PCA", "TSNE"]
            The dimensionality reduction method to apply.
        savepath : str
            A path to save the plot to. If `None` the plot will not be saved.

        Returns
        -------
        Figure
        """
        return quclst_scatter_plot(
            quclst=self,
            dim_reduce_method=dim_reduce_method,
            savepath=savepath,
            **kwargs,
        )


    def centroid_lineplot(
        self,
        subscales: list[str]=None,
        norm_to_highest: bool=False,
        savepath: str=None,
        **kwargs
    ) -> Figure:
        """A lineplot of the centroids.

        Parameters
        ----------
        subscales: list[str]
            The subscales to include. If `None` all will be included. Can also
            be used to change the order in which the subscales appear on the
            x-axis.
        norm_to_highest: bool
            Whether the values should be normed to the cluster with the highest
            average total score.
        savepath : str
            A path to save the plot to. If `None` the plot will not be saved.

        Returns
        -------
        Figure
        """
        return quclst_centroid_lineplot(
            quclst=self,
            subscales=subscales,
            norm_to_highest=norm_to_highest,
            savepath=savepath,
            **kwargs,
        )


    def transformed_centroid_lineplot(
        self,
        qusub: QuSubscales,
        savepath: str=None,
        **kwargs
    ) -> Figure:
        """A lineplot of the centroids, transformed according to the passed
        subscales. Only applicable if the cluster pipeline does not contain
        a dimensionality reduction, i.e., no QuestionnaireSubscales have
        been passed for the setup of the cluster model. This plot is
        automatically normed to the cluster with the highest total score.

        Parameters
        ----------
        qusub : QuestionnaireSubscales
            The subscales to be applied.
        savepath : str
            A path to save the plot to. If `None` the plot will not be saved.

        Returns
        -------
        Figure
        """
        return quclst_transform_centroid_lineplot(quclst=self, qusub=qusub, savepath=savepath, **kwargs)



    ## IO
    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def to_pickle(self, path: str) -> None:
        """Stores the `QuScoreClusters` object as a pickle file using joblib.
        Using `.to_dir` might be a more robust and storage efficient option.

        Parameters
        ----------
        path : str
        """
        jl_dump(self, path)

    @staticmethod
    def from_pickle(path: str) -> "QuScoreClusters":
        """Loads the `QuScoreClusters` from a pickle file.

        Parameters
        ----------
        path : str
            The path to the pickle file.

        Returns
        -------
        QuScoreClusters
        """
        cls_model = jl_load(path)
        print("The following exploratory QuScoreClusters-Model has been found:")
        print(cls_model)
        return cls_model

    def to_dir(self, path: str) -> None:
        """Stores the `QuScoreClusters` object to a directory.

        Parameters
        ----------
        path : str
        """
        save_quclst(self, path)

    @staticmethod
    def from_dir(path: str) -> "QuScoreClusters":
        """Loads a `QuScoreClusters` object from a directory.

        Parameters
        ----------
        path : str

        Returns
        -------
        QuScoreClusters
        """
        return load_quclst(path)





## IO
## -----------------------------------------------------------------------------

def save_quclst(quclst: QuScoreClusters, path: str) -> None:
    """Stores a `QuScoreClusters` object to a directory.

    Parameters
    ----------
    quclst : QuScoreClusters
    path : str
    """
    dir_ = Path(path)
    path_suffix_warning(dir_.suffix, obj="QuScoreClusters")
    empty_or_create_dir(dir_)

    settings_dict = {
        "scaling": quclst.scaling,
        "cluster_method": quclst.cluster_method,
        "drop_incomplete": quclst.drop_incomplete,
        "drop_earlystoppers": quclst.drop_earlystoppers,
        "verbose": quclst.verbose,
        "incomplete_threshold": quclst.incomplete_threshold,
        "earlystopping_threshold": quclst.earlystopping_threshold,
        "random_state": quclst.random_state,
        "_clusterer_kwargs": quclst._clusterer_kwargs,
        "_cluster_sort_dict": quclst._cluster_sort_dict,
    }
    if hasattr(quclst, "_cluster_label_dict"):
        settings_dict["_cluster_label_dict"] = quclst._cluster_label_dict

    quclst.qudata.to_dir(dir_ / "qudata")
    if quclst.qusubscales is not None:
        quclst.qusubscales.to_dir(dir_ / "qusub")

    pipeline_steps_names = list(quclst.pipeline.named_steps.keys())
    if hasattr(quclst, "dim_reducer"):
        jl_dump(quclst.dim_reducer, dir_ / "dim_reducer.pkl")
        settings_dict["_dim_reducer_name"] = pipeline_steps_names.pop(0)
    if hasattr(quclst, "scaler"):
        jl_dump(quclst.scaler, dir_ / "scaler.pkl")
        settings_dict["_scaler_name"] = pipeline_steps_names.pop(0)
    jl_dump(quclst.clusterer, dir_ / "clusterer.pkl")
    settings_dict["_clusterer_name"] = pipeline_steps_names.pop(0)

    with open(dir_ / "settings.json", "w") as f:
        json_dump(settings_dict, f, indent=2)


def load_quclst(path: str) -> QuScoreClusters:
    """Loads a `QuScoreClusters` object from a directory.

    Parameters
    ----------
    path : str

    Returns
    -------
    QuScoreClusters
    """
    dir_ = Path(path)
    path_suffix_warning(dir_.suffix, obj="QuScoreClusters")

    print("Loading QuData:")
    print("---------------")
    qudata = QuData.from_dir(dir_ / "qudata")
    print(f"Scores-Shape: {qudata.get_scr(mc_scored=False, verbose=False).shape}")
    print("---------------")
    try:
        qusub = QuSubscales.from_dir(dir_ / "qusub")
    except FileNotFoundError:
        qusub = None

    with open(dir_ / "settings.json", "r") as f:
        settings_dict: dict = json_load(f)

    _cluster_label_dict: dict = settings_dict.pop("_cluster_label_dict", None)
    _cluster_sort_dict_ = settings_dict.pop("_cluster_sort_dict")

    _clusterer_kwargs = settings_dict.pop("_clusterer_kwargs")
    _dim_reducer_name = settings_dict.pop("_dim_reducer_name", "DimReducer")
    _scaler_name = settings_dict.pop("_scaler_name", "Scaler")
    _clusterer_name = settings_dict.pop("_clusterer_name", "Clusterer")

    qusc = QuScoreClusters(
        qudata=qudata,
        qusubscales=qusub,
        **settings_dict,
        **_clusterer_kwargs,
        __supress_fit=True,
    )
    qusc._cluster_label_dict = _cluster_label_dict

    if _cluster_sort_dict_ is not None:
        _cluster_scort_dict = {}
        for k, v in _cluster_sort_dict_.items():
            _cluster_scort_dict[int(k)] = v
        qusc._cluster_sort_dict = _cluster_scort_dict

    qusc.pipeline = Pipeline([])
    try:
        dim_reducer = jl_load(dir_ / "dim_reducer.pkl")
        qusc.dim_reducer = dim_reducer
        qusc.pipeline.steps.append((_dim_reducer_name, qusc.dim_reducer))
    except FileNotFoundError:
        pass
    try:
        scaler = jl_load(dir_ / "scaler.pkl")
        qusc.scaler = scaler
        qusc.pipeline.steps.append((_scaler_name, qusc.scaler))
    except FileNotFoundError:
        pass
    clusterer = jl_load(dir_ / "clusterer.pkl")
    qusc.clusterer = clusterer
    qusc.pipeline.steps.append((_clusterer_name, qusc.clusterer))

    print("The following exploratory QuScoreClusters-Model has been found:")
    print(qusc)

    return qusc





## Plotting
## -----------------------------------------------------------------------------

def get_cluster_colors(n_cluster: int, noise_cluster: bool) -> list[str]:
    """Returns a unified list of colors to use wherever the clusters are in
    use. Returns the colors as a list of strings containing the colors in hex format.

    Parameters
    ----------
    n_cluster : int
        Number of clusters (discrete colors).
    noise_cluster : bool
        Whether a greyish color should be appended to the end of the list
        to be used for a noise-cluster
    """
    if n_cluster > 10:
        palette = list(sns.color_palette("magma", n_cluster).as_hex())

    else:
        palette = list(sns.color_palette("hls", n_cluster).as_hex())

    if noise_cluster:
        palette[-1] = "#a9a9a9"

    return palette


## Scatter-KDE-plot

def _quclst_pca_data(
    quclst: QuScoreClusters,
    df_pre: pd.DataFrame,
    df_ctr: pd.DataFrame,
) -> dict[str, pd.DataFrame|PCA]:
    """Generates PCA 2-dimensional data for visualization"""
    if hasattr(QuScoreClusters, "pca_data"):
        return quclst.pca_data

    dim_names = ["pca-" + str(i+1) for i in range(2)]

    X_pre = df_pre.drop(columns=[quclst.quconfig.id_col, "cluster"]).values
    X_ctr = df_ctr.drop(columns="cluster").values

    model_pca = PCA(n_components=2)
    X_pre_ = model_pca.fit_transform(X_pre)

    df_pre_ = pd.DataFrame(X_pre_, columns=dim_names)
    df_pre_[quclst.quconfig.id_col] = df_pre[quclst.quconfig.id_col]
    df_pre_["cluster"] = df_pre["cluster"]
    df_pre_ = df_pre_[[quclst.quconfig.id_col, "cluster"] + dim_names]

    X_ctr_ = model_pca.transform(X_ctr)
    df_ctr_ = pd.DataFrame(X_ctr_, columns=dim_names)
    df_ctr_["cluster"] = df_ctr["cluster"]
    df_ctr_ = df_ctr_[["cluster"] + dim_names]

    quclst.pca_data = {
        'df': df_pre_,
        'df_centroids': df_ctr_,
        'model_pca': model_pca
    }
    return quclst.pca_data


def _quclst_tsne_data(
    quclst: QuScoreClusters,
    df_pre: pd.DataFrame,
    df_ctr: pd.DataFrame,
) -> dict[str, pd.DataFrame|TSNE]:
    """Generates TSNE 2-dimensional data for visualization"""
    if hasattr(quclst, "tsne_data"):
        return quclst.tsne_data

    dim_names = ["tsne-" + str(i+1) for i in range(2)]

    X_pre = df_pre.drop(columns=[quclst.quconfig.id_col, "cluster"]).values
    X_ctr = df_ctr.drop(columns="cluster").values
    X_tsne = np.concatenate([X_pre, X_ctr], axis=0)

    model_tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=50,
        n_iter=2000,
        init="pca",
        learning_rate="auto"
    )
    X_tsne_ = model_tsne.fit_transform(X_tsne)

    X_pre_ = X_tsne_[:-quclst.n_clusters()[0]]
    X_ctr_ = X_tsne_[-quclst.n_clusters()[0]:]

    df_pre_ = pd.DataFrame(X_pre_, columns=dim_names)
    df_pre_[quclst.quconfig.id_col] = df_pre[quclst.quconfig.id_col]
    df_pre_["cluster"] = df_pre["cluster"]
    df_pre_ = df_pre_[[quclst.quconfig.id_col, "cluster"] + dim_names]

    df_ctr_ = pd.DataFrame(X_ctr_, columns=dim_names)
    df_ctr_["cluster"] = df_ctr["cluster"]
    df_ctr_ = df_ctr_[["cluster"] + dim_names]

    quclst.tsne_data = {
        'df': df_pre_,
        'df_centroids': df_ctr_,
        'model_tsne': model_tsne
    }

    return quclst.tsne_data


def quclst_scatter_plot(
    quclst: QuScoreClusters,
    dim_reduce_method: Literal["PCA", "TSNE"]="PCA",
    savepath: str=None,
    **kwargs
) -> Figure:
    """2-dimensional scatterplot of the cluster results.

    Parameters
    ----------
    quclst : QuScoreClusters
        The `QuScoreClusters` instance.
    dim_reduce_method : Literal["PCA", "TSNE"]
        The dimensionality reduction method to apply.
    savepath : str
        A path to save the plot to. If `None` the plot will not be saved.

    Returns
    -------
    Figure
    """
    df, df_ctr, model = quclst.store_2dim_data(method=dim_reduce_method).values()
    n_cls, noise_cls = quclst.n_clusters()
    colors = get_cluster_colors(n_cls, noise_cls)
    hue_order=df_ctr["cluster"].to_list()

    xlabel = f"{dim_reduce_method}-Dim 1"
    ylabel = f"{dim_reduce_method}-Dim 2"
    if dim_reduce_method == "PCA":
        xlabel = f"{xlabel} ({100 * model.explained_variance_[0]:.1f} %)"
        ylabel = f"{ylabel} ({100 * model.explained_variance_[1]:.1f} %)"

    fig = plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=df.values[:, -2],
        y=df.values[:, -1],
        hue="cluster",
        palette=colors,
        hue_order=hue_order,
        data=df,
        legend="brief",
    )

    if "highlight_ids" in kwargs:
        hid = kwargs["highlight_ids"]
        df_hid = df[df["ID"].isin(hid)]
        sns.scatterplot(
            x=df_hid.values[:, -2],
            y=df_hid.values[:, -1],
            zorder=99,
            color="black",
            marker="X",
            s=150,
        )

    if not kwargs.get("supress_kde", False):
        sns.kdeplot(
            x=df.values[:, -2],
            y=df.values[:, -1],
            hue="cluster",
            palette=colors,
            hue_order=hue_order,
            data=df,
            bw_adjust=1.1,
            fill=True,
            alpha=0.3,
        )
        sns.kdeplot(
            x=df.values[:, -2],
            y=df.values[:, -1],
            hue="cluster",
            palette=colors,
            hue_order=hue_order,
            data=df,
            bw_adjust=1.1,
            fill=False,
            alpha=0.75,
        )
    if df_ctr is not None:
        if n_cls > 5 or kwargs.get("color_centroids", False):
            chue="cluster"
            cpalette=colors
            chue_order=hue_order
            ccolor=None
        else:
            chue=None
            cpalette=None
            chue_order=None
            ccolor="black"

        sns.scatterplot(
            x=df_ctr.values[:, -2],
            y=df_ctr.values[:, -1],
            hue=chue,
            color=ccolor,
            palette=cpalette,
            hue_order=chue_order,
            data=df_ctr,
            s=200,
            zorder=98,
            legend=False,
        )
    plt.title(kwargs.get("title", "2D-Visualization (PCA) of the Clusters"), y=1.0)
    plt.legend(title=kwargs.get("legend_title", "Cluster"))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if savepath is not None:
        plt.savefig(savepath, dpi=kwargs.get("dpi", 300))

    return fig



## Centroid Lineplots

def _centroid_lineplot_from_dfs(
    df_plot: pd.DataFrame,
    df_mns: pd.DataFrame,
    noise_cls_exists: bool,
    cluster_sort_dict: dict,
    _cluster_label_dict: dict,
    id_col: str,
    palette: list[str],
    subscales: list[str]=None,
    norm_to_highest: bool=False,
    savepath: str=None,
    **kwargs
) -> Figure:
    """Workhorse function for the centroid lineplots."""
    dims = df_plot.drop(columns=[id_col, "cluster"]).columns.to_list()
    if subscales is not None:
        surplus_scl = [sscl for sscl in subscales if sscl not in dims]
        if surplus_scl is not None:
            if len(surplus_scl) != 0:
                raise ScoreClustersError(
                    f"The columns {surplus_scl} are not availible in the pre-cluster " +
                    "data columns."
                )
            dims = subscales

    if noise_cls_exists:
        palette = palette[:-1]
        df_plot = df_plot[df_plot["cluster"] != "noise"]
        if _cluster_label_dict is not None:
            df_plot = df_plot[df_plot["cluster"] != _cluster_label_dict["noise"]]
    df_plot = df_plot[["cluster"] + dims].copy()

    if norm_to_highest:
        if noise_cls_exists:
            df_mns = df_mns[df_mns["cluster"] != "noise"]
            if _cluster_label_dict is not None:
                df_mns = df_mns[df_mns["cluster"] !=  _cluster_label_dict["noise"]]

        df_max = df_mns[df_mns["total_score"]==df_mns["total_score"].max()]
        X_max = df_max[dims].values

        df_plot[dims] = df_plot[dims] / X_max

    cluster_rename_dict = kwargs.get("cluster_rename_dict", None)
    if cluster_rename_dict is not None:
        df_plot["cluster"] = df_plot["cluster"].replace(cluster_rename_dict)

    cluster_col_name = kwargs.get("cluster_col_name", "Cluster")
    df_plot = df_plot.rename(columns={'cluster': cluster_col_name})

    if cluster_rename_dict is not None:
        hue_order = list(cluster_rename_dict.values())
    else:
        if _cluster_label_dict is not None:
            hue_order = deepcopy(_cluster_label_dict)
            if "noise" in hue_order:
                del hue_order["noise"]
        else:
            hue_order = deepcopy(cluster_sort_dict)
            if -1 in hue_order:
                del hue_order[-1]
        hue_order = list(hue_order.values())


    fig = plt.figure(figsize=(10, 7)) # (6, 5)
    sns.lineplot(
        data=df_plot.melt(id_vars=[cluster_col_name], var_name='cat'),
        x='cat',
        y='value',
        hue=cluster_col_name,
        hue_order=hue_order,
        palette=palette,
        errorbar=kwargs.get("errorbar", "ci"), # 95 % confidence interval
    )
    plt.xticks(rotation=0) # 15
    plt.xlabel("")

    ylabel = kwargs.get("ylabel", "Average scores")
    plt.ylabel(ylabel)

    title = kwargs.get("title", "Lineplot of Cluster-Averages")
    plt.title(title)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=kwargs.get("xtick_rotation", 20))

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=kwargs.get("dpi", 300))

    return fig


def quclst_centroid_lineplot(
    quclst: QuScoreClusters,
    subscales: list[str]=None,
    norm_to_highest : bool=False,
    savepath: str=None,
    **kwargs,
) -> Figure:
    """A lineplot of the centroids.

    Parameters
    ----------
    quclst : QuScoreClusters
        The `QuScoreClusters` instance to provide the plot for.
    subscales : list[str]
        The subscales to include. If `None` all will be included. Can also
        be used to change the order in which the subscales appear on the
        x-axis.
    norm_to_highest : bool
        Whether the values should be normed to the cluster with the highest
        average total score.
    savepath : str
        A path to save the plot to. If `None` the plot will not be saved.

    Returns
    -------
    Figure
    """
    palette = get_cluster_colors(*quclst.n_clusters())

    df_plot = quclst.clusters()

    df_mns = quclst.get_cluster_aggregations(["mean"])
    df_mns.columns = [col[0] for col in df_mns.columns]

    noise_cls_exists = quclst._has_noise_cls()
    cluster_sort_dict = quclst._cluster_sort_dict
    _cluster_label_dict = quclst._get_cluster_label_dict()
    id_col = quclst.quconfig.id_col

    return _centroid_lineplot_from_dfs(
        df_plot=df_plot,
        df_mns=df_mns,
        noise_cls_exists=noise_cls_exists,
        cluster_sort_dict=cluster_sort_dict,
        _cluster_label_dict=_cluster_label_dict,
        id_col=id_col,
        palette=palette,
        subscales=subscales,
        norm_to_highest=norm_to_highest,
        savepath=savepath,
        **kwargs,
    )


def quclst_transform_centroid_lineplot(
    quclst: QuScoreClusters,
    qusub: QuSubscales,
    savepath: str=None,
    **kwargs,
) -> Figure:
    """A lineplot of the centroids, transformed according to the passed
    subscales. Only applicable if the cluster pipeline does not contain
    a dimensionality reduction, i.e., no QuestionnaireSubscales have
    been passed for the setup of the cluster model. This plot is
    automatically normed to the cluster with the highest total score.

    Parameters
    ----------
    quclst : QuScoreClusters
        The `QuScoreClusters` instance to provide the plot for.
    qusub : QuestionnaireSubscales
        The subscales to be applied.
    savepath : str
        A path to save the plot to. If `None` the plot will not be saved.

    Returns
    -------
    Figure
    """
    if hasattr(quclst, "dim_reducer"):
        raise ScoreClustersError(
            "A transformed centroid lineplot is only possible if the cluster pipeline does\n" +
            "not contain a dimensionality reduction, i.e., no QuestionnaireSubscales have\n" +
            "been passed for the setup of the cluster model."
        )

    palette = get_cluster_colors(*quclst.n_clusters())

    df_plot = quclst.clusters()

    df_mns = quclst.get_cluster_aggregations(["mean"])
    df_mns.columns = [col[0] for col in df_mns.columns]

    noise_cls_exists = quclst._has_noise_cls()
    cluster_sort_dict = quclst._cluster_sort_dict
    _cluster_label_dict = quclst._get_cluster_label_dict()
    id_col = quclst.quconfig.id_col

    df_plot_prt = df_plot[[id_col, "cluster"]].copy()
    df_plot = df_plot.drop(columns=[id_col, "cluster"])
    df_plot = qusub.apply_to_dataframe(df_plot)
    df_plot = pd.concat([df_plot_prt, df_plot], axis=1)

    df_mns_prt = df_mns[["cluster", "total_score"]].copy()
    df_mns = df_mns.drop(columns=["cluster", "total_score", "count"])
    df_mns = qusub.apply_to_dataframe(df_mns)
    df_mns = pd.concat([df_mns_prt, df_mns], axis=1)

    return _centroid_lineplot_from_dfs(
        df_plot=df_plot,
        df_mns=df_mns,
        noise_cls_exists=noise_cls_exists,
        cluster_sort_dict=cluster_sort_dict,
        _cluster_label_dict=_cluster_label_dict,
        id_col=id_col,
        palette=palette,
        subscales=None,
        norm_to_highest=True,
        savepath=savepath,
        **kwargs,
    )



## K-Means-specific plots

def _quclst_is_kmeans(quclst: QuScoreClusters) -> bool:
    if quclst.cluster_method == "KMeans":
        return True
    if isinstance(quclst.cluster_method, KMeansWrapper):
        return True
    if isinstance(quclst.cluster_method, ClusterWrapper):
        if isinstance(quclst.cluster_method.kernel, KMeans):
            return True
    return False



def compute_kmeans_bic(kmeans: KMeans, X: np.ndarray) -> float:
    # https://link.springer.com/chapter/10.1007/978-3-540-88458-3_60, p. 666-667
    """Compute Bayesian Information Criterion (BIC) for a
    [`sklearn.cluster.KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    object..

    Parameters
    ----------
    kmeans : sklearn.cluster.KMeans
        Fitted scikit-learn `KMeans` object.
    X : np.ndarray
        The ($N_{sample} \\times d_{dim}$) data to calculate the BIC for
        (typically the same data `kmeans` was fittet to).

    Returns
    -------
    float
        BIC as a flat number.
    """
    centers = kmeans.cluster_centers_
    labels  = kmeans.labels_
    n_clusters = kmeans.n_clusters
    n_per_cluster = np.bincount(labels)
    n_sample, d_dim = X.shape

    within_sse = [
        np.linalg.norm(X[np.where(labels == i)] - centers[i, :])**2
        for i in range(n_clusters)
    ]

    bic = (
        np.sum([
            n_per_cluster[i] * np.log(n_per_cluster[i]) -
            n_per_cluster[i] * np.log(within_sse[i]) / 2
            for i in range(n_clusters)
            ]) -
        n_sample * np.log(n_sample)  -
        n_sample * d_dim * np.log(2*np.pi) / 2 -
        (n_sample - n_clusters**2) / 2 -
        n_clusters * np.log(n_sample) / 2
    )

    return bic


def quclst_elbow_plot(
    quclst: QuScoreClusters,
    savepath: str=None,
    plot_bic: bool=True,
    n_cluster_range=[1, 10],
    **kwargs
) -> Figure:
    """Generates an elbow plot. Only avalable for the kmeans cluster method.

    Parameters
    ----------
    quclst : QuScoreClusters
        The `QuScoreClusters` instance to provide the elbow plot for.
    savepath : str
        Path to save the plot to.
    plot_bic : bool
        Whether the BIC (Bayesian Information Criterion) should be included.
    n_cluster_range : list
        The range of the considered number of clusters.

    Returns
    -------
    Figure
        A matplotlib figure.

    Raises
    ------
    NotImplementedError
    """
    if not _quclst_is_kmeans(quclst):
        raise NotImplementedError("The elbow-analysis is only implemented for `cluster_method` \"kmeans\".")
    X_pre = quclst._get_fit_scores().values
    X_pre = quclst._apply_pre_pipeline(X_pre)

    cluster_range = np.arange(n_cluster_range[0], n_cluster_range[1]+1)
    sses = []
    bics = []

    for k in cluster_range:
        clusterer = KMeans(n_clusters=k)
        clusterer.fit(X_pre)
        sse = clusterer.inertia_
        sses.append(sse)
        bics.append(compute_kmeans_bic(clusterer, X_pre))

    ax1: Axes
    figsize = kwargs.get("figsize", (7, 4))
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.set_xticks(cluster_range)
    xlabel = kwargs.get("xlabel", 'Preset Number of Clusters')
    ax1.set_xlabel(xlabel)
    ylabel_sse = kwargs.get("ylabel_sse", 'Resulting Sum of Squared Errors')
    ax1.set_ylabel(ylabel_sse, color="black")
    ax1.plot(cluster_range, sses, color="black")
    ax1.grid()
    ax1.tick_params(axis='y', labelcolor="black")

    if plot_bic:
        ax2: Axes = ax1.twinx()
        ylabel_bic = kwargs.get("ylabel_bic", 'BIC (Bayesian Information Criterion)')
        ax2.set_ylabel(ylabel_bic, color="tab:blue")
        ax2.plot(cluster_range, bics, color="tab:blue")
        ax2.tick_params(axis='y', labelcolor="tab:blue")

    title = kwargs.get("title", "$K$-Means Elbow-Plot")
    fig.suptitle(title)
    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=kwargs.get("dpi", 300))

    return fig


def quclst_silhouette_plot(
    quclst: QuScoreClusters,
    savepath: str=None,
    n_cluster_range=[2, 10],
    **kwargs
) -> Figure:
    """Generates a plot of silhouette score means.

    Parameters
    ----------
    quclst : QuScoreClusters
        The `QuScoreClusters` instance to provide the silhouette plot for.
    savepath : str
        Path to save the plot to.
    n_cluster_range : list
        The range of the considered number of clusters.

    Returns
    -------
    Figure
        A matplotlib figure.

    Raises
    ------
    NotImplementedError
    """
    if not _quclst_is_kmeans(quclst):
        raise NotImplementedError("The elbow-analysis is only implemented for `cluster_method` \"kmeans\".")
    X_pre = quclst._get_fit_scores().values
    X_pre = quclst._apply_pre_pipeline(X_pre)

    cluster_range = np.arange(n_cluster_range[0], n_cluster_range[1]+1)
    sil_avgs = []

    for k in cluster_range:
        clusterer = KMeans(n_clusters=k)
        cluster_labels = clusterer.fit_predict(X_pre)
        sil_avg = silhouette_score(X_pre, cluster_labels)
        sil_avgs.append(sil_avg)

    ax1: Axes
    figsize = kwargs.get("figsize", (7, 4))
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.set_xticks(cluster_range)
    xlabel = kwargs.get("xlabel", 'Preset Number of Clusters')
    ax1.set_xlabel(xlabel)
    ylabel_sil = kwargs.get("ylabel_sse", 'Resulting (average) Silhouette Score')
    ax1.set_ylabel(ylabel_sil, color="black")
    ax1.plot(cluster_range, sil_avgs, color="black")
    ax1.grid()
    ax1.tick_params(axis='y', labelcolor="black")
    title = kwargs.get("title", "$K$-Means Silhouette-Plot")
    fig.suptitle(title)
    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=kwargs.get("dpi", 300))

    return fig
