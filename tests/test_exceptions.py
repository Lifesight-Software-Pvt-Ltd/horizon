import pytest

from horizon_forecaster.exceptions import (
    HorizonForecasterError,
    InsufficientDataError,
    ModelConvergenceError,
    ModelPredictionError,
    ThresholdRejectionError,
    AllModelsRejectedError,
    InvalidConfigError,
    UnsupportedFrequencyError,
)


def test_base_exception_message_without_model_name():
    err = HorizonForecasterError("something went wrong")
    assert str(err) == "something went wrong"


def test_base_exception_message_with_model_name():
    err = HorizonForecasterError("something went wrong", model_name="prophet")
    assert str(err) == "[prophet] something went wrong"


def test_all_custom_exceptions_are_catchable_as_base():
    """
    Critical contract: catching HorizonForecasterError must catch every
    subclass. This is what lets forecaster.py wrap each model's fit/predict
    call in a single except clause.
    """
    exceptions_to_test = [
        InsufficientDataError("insufficient data"),
        ModelConvergenceError("did not converge"),
        ModelPredictionError("prediction failed"),
        ThresholdRejectionError("mape out of range"),
        AllModelsRejectedError("all rejected"),
        InvalidConfigError("bad config"),
        UnsupportedFrequencyError("bad frequency"),
    ]
    for exc in exceptions_to_test:
        with pytest.raises(HorizonForecasterError):
            raise exc


def test_insufficient_data_error_carries_period_info():
    err = InsufficientDataError(
        "not enough history",
        model_name="sarimax",
        required_periods=104,
        available_periods=30,
    )
    assert err.model_name == "sarimax"
    assert err.required_periods == 104
    assert err.available_periods == 30


def test_model_convergence_error_carries_original_exception():
    original = ValueError("singular matrix")
    err = ModelConvergenceError(
        "SARIMAX failed to converge",
        model_name="sarimax",
        original_exception=original,
    )
    assert err.original_exception is original
    assert isinstance(err.original_exception, ValueError)


def test_threshold_rejection_error_carries_mape_and_bounds():
    err = ThresholdRejectionError(
        "MAPE outside allowed range",
        model_name="catboost",
        mape_value=150.0,
        allowed_min=1.0,
        allowed_max=100.0,
    )
    assert err.mape_value == 150.0
    assert err.allowed_min == 1.0
    assert err.allowed_max == 100.0


def test_all_models_rejected_error_carries_rejected_list():
    rejected = [
        {"model_name": "prophet", "reason": "threshold", "mape": 120.0},
        {"model_name": "sarimax", "reason": "convergence", "mape": None},
    ]
    err = AllModelsRejectedError("no models survived", rejected_models=rejected)
    assert err.rejected_models == rejected
    assert len(err.rejected_models) == 2


def test_all_models_rejected_error_defaults_to_empty_list():
    err = AllModelsRejectedError("no models survived")
    assert err.rejected_models == []


def test_invalid_config_error_carries_param_context():
    err = InvalidConfigError(
        "training_size must be between 50 and 100",
        param_name="training_size",
        provided_value=30,
        valid_range=(50, 100),
    )
    assert err.param_name == "training_size"
    assert err.provided_value == 30
    assert err.valid_range == (50, 100)


def test_unsupported_frequency_error_carries_provided_value():
    err = UnsupportedFrequencyError("only daily/weekly supported", provided_frequency="monthly")
    assert err.provided_frequency == "monthly"
