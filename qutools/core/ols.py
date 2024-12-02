"""
# Submodule for R-style Linear Regression Analyses

This submodule contains a class for linear regression analyses, which is
compatible with the [`lm` function in `R`](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm).


Usage
-----
The main component of this submodule is the `LinearModel` class, that
can be used standalone and is not dependent on any other submodule. Is is
however used in the `qutools.scorer_results_regressor` submodules to evaluate
the performance of the regression models (see [here][qutools.scorer_results_regressor.sr_regressor_results]).
It provides various fit statistics and confidence intervals for the coefficients.

Example
-------
```python
import pandas as pd
import seaborn as sns

from sklearn.datasets import make_regression
from qutools.core.ols import LinearModel

X, y = make_regression(
    n_samples=100,
    n_features=3,
    n_informative=2,
    bias=1,
    noise=30,
    random_state=5555
)

features = [f"x{i}" for i in range(X.shape[1])]
target = "y"

df = pd.DataFrame(X, columns=features)
df[target] = y

lm = LinearModel()
lm.fit(df, features=features, target=target)
```
```
> Fittet Linear Model
>  - 100 Samples and 4 Features
>  - F-Statistic: F(3, 96)=303.975 (p=0.000)
>  - R²: 0.905
```
```python
lm.get_fit_stats()
```
<p align="center">
   <img src="../../../assets/img/ols-tab1.png" width="40%">
</p>
```python
lm.get_coef_result_df()
```
<p align="center">
   <img src="../../../assets/img/ols-tab2.png" width="80%">
</p>
```python
df[target+"_pred"] = lm.predict(df=df, features=features)
_ = sns.scatterplot(x=target, y=target+"_pred", data=df)
```
<p align="center">
   <img src="../../../assets/img/ols-plot.png" width="80%">
</p>

"""

import numpy as np
import pandas as pd

from scipy import stats



class LinearModel:

    coef: np.ndarray = None
    stats: dict[str, str|float] = {}

    def __init__(self) -> None:
        """Initializes the linear model - does basically nothing."""
        pass

    def _is_fitted(self):
        if self.coef is None:
            raise ValueError("Model not fitted yet.")

    def _get_features(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        return df[features].values

    def _get_target(self, df: pd.DataFrame, target: str) -> np.ndarray:
        return df[target].values

    def _append_intercept(self, X: np.ndarray) -> np.ndarray:
        return np.c_[np.ones(X.shape[0]), X]


    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Predicts the target variable based on the fitted model. Can be used
        for the fit-data as well as for new data.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data.
        features : list[str]
            The list of features to use for the prediction. Must be the same length
            as the `features` used for the fitting.
        """
        self._is_fitted()
        X = self._get_features(df, features)
        X = self._append_intercept(X)
        return X @ self.coef


    def get_residuals(self, df: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
        """Calculates the residuals of the model.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data.
        features : list[str]
            The list of features to use for the prediction. Must be the same length
            as the `features` used for the fitting.
        target : str
            The target variable to predict.
        """
        self._is_fitted()
        targs = self._get_target(df, target)
        preds = self.predict(df, features)
        res = targs - preds
        return res


    def fit(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        alpha_ci: float=0.05,
        verbose: bool=True,
    ) -> pd.DataFrame:
        """Performs the linear regression analysis. The "maths" is carried out
        using the `numpy` and `scipy` libraries. The results are stored in the
        class attributes `coef`, `coef_err`, `features`, and `stats`.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data.
        features : list[str]
            The list of features to use for the prediction. Must be the same length
            as the `features` used for the fitting.
        target : str
            The target variable to predict.
        alpha_ci : float
            The alpha level for the confidence intervals.
        verbose : bool
            Whether to print the results to the stdout or not.
        """

        X = self._get_features(df, features)
        X = self._append_intercept(X)
        y = self._get_target(df, target)

        n_samples = X.shape[0]
        n_features = X.shape[1]
        dof = n_samples - n_features

        cov_x = X.T @ X
        cov_x_inv = np.linalg.inv(cov_x)

        w = cov_x_inv @ X.T @ y
        res = y - (X @ w)

        var = res.T @ res / dof

        w_cov = var * cov_x_inv
        w_err = w_cov.diagonal()**0.5
        Tw = w / w_err
        pw = stats.t.sf(np.abs(Tw), dof) * 2
        gap = np.abs(stats.t.isf(1 - alpha_ci/2, dof)) * w_err
        w_low = w - gap
        w_high = w + gap

        R_sq = 1 - (y.T @ y - w.T @ X.T @ y) / (y.T @ y - n_samples * y.mean()**2)

        F = R_sq * dof / ((1-R_sq) * (n_features-1))
        pF = stats.f.sf(F, n_features-1, dof)

        features = ["Intercept"] + features

        self.coef = w
        self.coef_err = w_err
        self.features = features
        self.stats["dof"] = dof
        self.stats["F-str"] = f"F({n_features - 1}, {dof})"
        self.stats["F"] = F
        self.stats["pF"] = pF
        self.stats["rsq"] = R_sq
        self.stats["coef-t"] = Tw
        self.stats["coef-p"] = pw
        self.stats["coef-lower-ci"] = w_low
        self.stats["coef-upper-ci"] = w_high
        self.stats["alpha-ci"] = alpha_ci

        if verbose:
            print("Fittet Linear Model")
            print(f" - {n_samples} Samples and {n_features} Features")
            print(f" - F-Statistic: F({n_features - 1}, {dof})={F:.3f} (p={pF:.3f})")
            print(f" - R²: {R_sq:.3f}")


    def get_coef_result_df(self) -> pd.DataFrame:
        """Returns the coefficients of the model as a DataFrame. The DataFrame
        contains the coefficients, the standard errors, the t-values, the p-values,
        the Cohen d, and the confidence intervals.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the coefficients.
        """
        self._is_fitted()
        return pd.DataFrame({
            "Value": self.coef,
            "Std. Error": self.coef_err,
            "T": self.stats["coef-t"],
            "P > |T|": self.stats["coef-p"],
            'lower': self.stats["coef-lower-ci"],
            'upper': self.stats["coef-upper-ci"],
        }, index=self.features)


    def get_fit_stats(self) -> pd.DataFrame:
        """Returns the fit statistics of the model as a DataFrame. The DataFrame
        contains the F-statistic, the p-value of the F-statistic, and the R².

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the fit statistics.
        """
        self._is_fitted()
        return pd.DataFrame({
            "F": [self.stats["F"]],
            "pF": [self.stats["pF"]],
            "R²": [self.stats["rsq"]],
        }, index=[self.stats["F-str"]])
