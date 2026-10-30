"""
Abstract base class for all model wrappers in horizon_forecaster.

Every wrapper (ProphetWrapper, SARIMAXWrapper, CatBoostWrapper,
BayesianPyMCWrapper, NeuralProphetWrapper) must inherit from BaseModel and
implement `fit()` and `predict()`. This is the single most important
interface in the library — every wrapper depends on it, and
`forecaster.py`'s orchestration loop is written entirely against this
contract, never against any concrete model class directly.

Exception contract (must be honored by every subclass):
    fit()     may raise ModelPredictionError (adapter-contract/shape
              check, before any fitting is attempted; read it as a
              contract error), InsufficientDataError or ModelConvergenceError.
    predict() may raise ModelPredictionError.
    Both may let unexpected/unrelated exceptions propagate as-is — only
    wrap exceptions that are genuinely InsufficientData/Convergence/
    Prediction failures; don't swallow real bugs into these types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, Optional

import pandas as pd

from horizon_forecaster.exceptions import ModelPredictionError


class BaseModel(ABC):
    """
    Abstract base class that every model wrapper must inherit from.

    Args:
        frequency: "daily" or "weekly" — determines which default
            hyperparameter set is loaded from config/default_params.yaml.
        params: optional dict of hyperparameter overrides. If provided,
            these are merged on top of the frequency's defaults (user
            overrides win). If None, the frequency's defaults are used
            as-is.

    Attributes:
        model_name: short string identifier (e.g. "prophet", "sarimax"),
            set by each subclass. Used in exception messages and in the
            rejected-models report.
        is_fitted: True once fit() has completed successfully.
        accepted_params: the hyperparameter names this wrapper reads, for
            wrappers that read their params one by one (SARIMAX, PyMC):
            there an unknown key would otherwise be ignored without a
            word, so the forecaster rejects it up front. None for the
            wrappers that pass params straight to their library's
            constructor (Prophet, CatBoost, NeuralProphet), which rejects
            unknown names itself.
    """

    model_name: str = "base"
    accepted_params: Optional[FrozenSet[str]] = None

    def __init__(self, frequency: str, params: Optional[Dict[str, Any]] = None):
        self.frequency = frequency
        self.params: Dict[str, Any] = params or {}
        self.is_fitted: bool = False
        self._fitted_model: Any = None  # holds the underlying library's fitted object

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, target_column: str) -> "BaseModel":
        """
        Fit the model on training data.

        Args:
            train_df: training DataFrame, already shaped by the
                corresponding data_adapter for this model.
            target_column: accepted for interface consistency only. Every
                v1 wrapper reads the adapter's canonical "y" column and
                ignores this argument -- passing another name does NOT
                retarget the fit, it just fails the shape check below.

        Returns:
            self (to allow method chaining, e.g. model.fit(df, "y").predict(...)).

        Raises:
            ModelPredictionError: if train_df is missing the columns the
                adapter is contracted to supply ("ds"/"y"). Raised before
                any fitting is attempted; read it as a shape/contract
                error rather than a prediction one.
            InsufficientDataError: if train_df does not have enough rows
                for this model to train reliably.
            ModelConvergenceError: if the underlying library's fit
                procedure fails to converge or raises an internal
                numerical error.

        Implementations MUST set self.is_fitted = True on success, and
        MUST NOT set it on failure.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int, future_exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate a forecast for `horizon` future periods.

        Args:
            horizon: number of future periods to forecast (in the model's
                configured frequency — days or weeks).
            future_exog: DataFrame of future independent variable values,
                already shaped by the data_adapter and carrying at least
                a "ds" column. REQUIRED in practice: every v1 wrapper
                raises ModelPredictionError on None, because this method
                does not invent future exog data itself — the caller
                (forecaster.py) must have already applied a
                future-exogenous-data strategy (last_year,
                carry_forward, seasonal_naive, etc.). It is typed
                Optional only so the signature stays uniform with
                callers that pass the value through positionally.

        Returns:
            DataFrame with, at minimum, columns ["ds", "yhat"] — one row
            per forecasted period.

            Optionally, wrappers whose underlying library natively
            supports it (Prophet, SARIMAX, PyMC) MAY also include
            "yhat_lower" and "yhat_upper" columns (model-native
            uncertainty/credible intervals). Wrappers that do not
            support this (CatBoost, NeuralProphet) should simply omit
            these columns rather than fabricating them.

            IMPORTANT — these model-native bounds are only what a
            wrapper returns when used on its own. HorizonForecaster
            discards them and gives every model and ensemble rule the
            same calibrated 95% band, built from that forecast's own
            validation errors (see evaluation/bands.py). The native
            intervals mean different things across models and cover too
            little, and averaging them for an ensemble has no
            coherent interpretation, so ensemble.py operates on the
            "yhat" column only.

        Raises:
            ModelPredictionError: if called before fit(), or if the
                underlying library fails during prediction (e.g. shape
                mismatch with future_exog).
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        """Return the current hyperparameter dict (defaults + any overrides)."""
        return dict(self.params)

    def set_params(self, **kwargs) -> "BaseModel":
        """
        Update hyperparameters. Does not re-fit automatically — the
        caller must call fit() again after changing params.

        CAVEAT: this reliably re-targets only the wrappers that read
        self.params at fit time (Prophet, CatBoost). SARIMAXWrapper,
        BayesianPyMCWrapper and NeuralProphetWrapper's `freq` snapshot
        their hyperparameters into attributes in __init__, so a later
        set_params() changes what get_params() reports without changing
        what the next fit() actually uses. Construct a fresh wrapper
        instead — which is what forecaster.py does on every fit, so this
        never bites through the orchestrator. Unifying the two is
        deferred past 1.0 because it moves fitting behaviour.
        """
        self.params.update(kwargs)
        return self

    def _check_is_fitted(self) -> None:
        """
        Convenience guard for subclasses to call at the top of predict().
        Raises ModelPredictionError with a consistent message if the
        model hasn't been fitted yet.
        """
        if not self.is_fitted:
            raise ModelPredictionError(
                f"{self.model_name} model must be fitted before calling predict().",
                model_name=self.model_name,
            )

    def save(self, path: str) -> None:
        """
        Persist this fitted model to disk, so it can be reloaded in a
        future session without refitting.

        Default implementation: generic pickle of the whole wrapper
        instance. This is fine for wrappers whose underlying library
        pickles reliably (e.g. SARIMAX/statsmodels).

        IMPORTANT: wrappers whose underlying library has known
        serialization caveats (Prophet's Stan/cmdstanpy backend, PyMC's
        InferenceData, NeuralProphet's torch state) MUST override this
        method with that library's own recommended native serialization
        approach instead of relying on this default — generic pickling
        of those objects can silently produce a file that fails to load
        correctly in a different environment or library version.
        """
        self._check_is_fitted()
        import pickle  # nosec B403 -- model save/load, trusted files only

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Reload a previously saved model from disk.

        Default implementation: generic unpickle. See the note on
        save() — wrappers that override save() with library-native
        serialization must also override load() to match.
        """
        import pickle  # nosec B403 -- model save/load, trusted files only

        with open(path, "rb") as f:
            return pickle.load(f)  # nosec B301 -- trusted: written by save()
