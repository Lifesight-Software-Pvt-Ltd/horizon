import pandas as pd
import pytest

from horizon_forecaster.exceptions import (
    InsufficientDataError,
    ModelConvergenceError,
    ModelPredictionError,
)
from horizon_forecaster.models.base import BaseModel

class DummyModel(BaseModel):
    """
    Minimal concrete implementation of BaseModel, used only for testing
    the base contract. Real wrappers (ProphetWrapper, SARIMAXWrapper,
    etc.) should follow this same structural pattern.
    """

    model_name = "dummy"

    def __init__(self, frequency, params=None, min_rows_required=3, fail_to_converge=False):
        super().__init__(frequency, params)
        self.min_rows_required = min_rows_required
        self.fail_to_converge = fail_to_converge

    def fit(self, train_df: pd.DataFrame, target_column: str) -> "DummyModel":
        if len(train_df) < self.min_rows_required:
            raise InsufficientDataError(
                f"DummyModel requires at least {self.min_rows_required} rows.",
                model_name=self.model_name,
                required_periods=self.min_rows_required,
                available_periods=len(train_df),
            )
        if self.fail_to_converge:
            raise ModelConvergenceError(
                "DummyModel simulated a convergence failure.",
                model_name=self.model_name,
            )

        # "fit" by just remembering the mean of the target column
        self._fitted_model = train_df[target_column].mean()
        self.is_fitted = True
        return self

    def predict(self, horizon: int, future_exog=None) -> pd.DataFrame:
        self._check_is_fitted()
        return pd.DataFrame({
            "ds": range(horizon),
            "yhat": [self._fitted_model] * horizon,
        })

# ---------- cannot instantiate BaseModel directly ----------

def test_base_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseModel(frequency="daily")

# ---------- constructor / params ----------

def test_dummy_model_initializes_with_frequency_and_empty_params():
    model = DummyModel(frequency="daily")
    assert model.frequency == "daily"
    assert model.get_params() == {}
    assert model.is_fitted is False

def test_dummy_model_accepts_param_overrides():
    model = DummyModel(frequency="weekly", params={"alpha": 0.1})
    assert model.get_params() == {"alpha": 0.1}

def test_set_params_updates_and_returns_self():
    model = DummyModel(frequency="daily")
    result = model.set_params(alpha=0.5, beta=2)
    assert result is model  # chaining works
    assert model.get_params() == {"alpha": 0.5, "beta": 2}

# ---------- fit() contract ----------

def test_fit_returns_self_for_chaining():
    model = DummyModel(frequency="daily")
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
    result = model.fit(df, target_column="y")
    assert result is model
    assert model.is_fitted is True

def test_fit_raises_insufficient_data_error_and_does_not_set_is_fitted():
    model = DummyModel(frequency="daily", min_rows_required=10)
    df = pd.DataFrame({"y": [1, 2, 3]})
    with pytest.raises(InsufficientDataError) as exc_info:
        model.fit(df, target_column="y")
    assert model.is_fitted is False
    assert exc_info.value.required_periods == 10
    assert exc_info.value.available_periods == 3

def test_fit_raises_convergence_error_and_does_not_set_is_fitted():
    model = DummyModel(frequency="daily", fail_to_converge=True)
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
    with pytest.raises(ModelConvergenceError):
        model.fit(df, target_column="y")
    assert model.is_fitted is False

# ---------- predict() contract ----------

def test_predict_before_fit_raises_model_prediction_error():
    model = DummyModel(frequency="daily")
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=5)

def test_predict_after_fit_returns_expected_shape():
    model = DummyModel(frequency="daily")
    df = pd.DataFrame({"y": [10, 20, 30]})
    model.fit(df, target_column="y")
    forecast = model.predict(horizon=4)
    assert list(forecast.columns) == ["ds", "yhat"]
    assert len(forecast) == 4
    assert (forecast["yhat"] == 20.0).all()  # mean of [10,20,30]

# ---------- method chaining end-to-end ----------

def test_fit_predict_chaining_works_end_to_end():
    df = pd.DataFrame({"y": [5, 10, 15, 20]})
    forecast = DummyModel(frequency="weekly").fit(df, target_column="y").predict(horizon=2)
    assert len(forecast) == 2