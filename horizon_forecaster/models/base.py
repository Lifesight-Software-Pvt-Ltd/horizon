"""
Abstract base class for all model wrappers in horizon_forecaster.

Every wrapper (ProphetWrapper, SARIMAXWrapper, CatBoostWrapper,
PyMCWrapper, NeuralProphetWrapper) must inherit from BaseModel and
implement `fit()` and `predict()`. This is the single most important
interface in the library — every wrapper depends on it, and
`forecaster.py`'s orchestration loop is written entirely against this
contract, never against any concrete model class directly.

Exception contract (must be honored by every subclass):
    fit()     may raise InsufficientDataError or ModelConvergenceError.
    predict() may raise ModelPredictionError.
    Both may let unexpected/unrelated exceptions propagate as-is — only
    wrap exceptions that are genuinely InsufficientData/Convergence/
    Prediction failures; don't swallow real bugs into these types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

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
    """

    model_name: str = "base"

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
            target_column: name of the target/dependent variable column.

        Returns:
            self (to allow method chaining, e.g. model.fit(df, "y").predict(...)).

        Raises:
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
            future_exog: optional DataFrame of future independent variable
                values, already shaped by the data_adapter. If None, the
                caller (forecaster.py) is responsible for having already
                applied a future-exogenous-data strategy (carry_forward,
                seasonal_naive, historical_average, etc.) before calling
                predict — this method does not invent future exog data
                itself.

        Returns:
            DataFrame with (at minimum) columns ["ds", "yhat"] — one row
            per forecasted period.

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