import pandas as pd
import pytest

from horizon_forecaster.evaluation.ensemble import EnsembleResult, build_ensemble
from horizon_forecaster.exceptions import ThresholdRejectionError

def make_predictions(ds, values):
    return pd.DataFrame({"ds": ds, "yhat": values})

# ---------- weighting math ----------

def test_inverse_mape_weighting_favors_lower_mape_model():
    ds = pd.date_range("2023-01-01", periods=3, freq="D")
    result = build_ensemble(
        per_model_mape={"a": 10.0, "b": 30.0},
        per_model_predictions={
            "a": make_predictions(ds, [1, 2, 3]),
            "b": make_predictions(ds, [4, 5, 6]),
        },
        mape_threshold=100.0,
        weighting="inverse_mape",
    )
    assert result.weights["a"] > result.weights["b"]
    assert abs(sum(result.weights.values()) - 1.0) < 1e-9


def test_equal_weighting_ignores_mape_differences():
    ds = pd.date_range("2023-01-01", periods=3, freq="D")
    result = build_ensemble(
        per_model_mape={"a": 10.0, "b": 30.0},
        per_model_predictions={
            "a": make_predictions(ds, [1, 2, 3]),
            "b": make_predictions(ds, [4, 5, 6]),
        },
        mape_threshold=100.0,
        weighting="equal",
    )
    assert result.weights == {"a": 0.5, "b": 0.5}


def test_inverse_mape_weighting_does_not_divide_by_zero_on_perfect_mape():
    """
    A literal 0% MAPE is included in the ensemble -- mape_threshold is an
    upper bound only, so an unusually low score is never rejected. That
    means _compute_weights must not divide by zero when it takes the
    inverse-MAPE weight of a perfect model; this exercises that guard.
    """
    ds = pd.date_range("2023-01-01", periods=2, freq="D")
    result = build_ensemble(
        per_model_mape={"a": 0.0, "b": 5.0},
        per_model_predictions={
            "a": make_predictions(ds, [1, 2]),
            "b": make_predictions(ds, [3, 4]),
        },
        mape_threshold=100.0,
        weighting="inverse_mape",
    )
    assert result.combined is not None
    assert result.weights["a"] > result.weights["b"]


def test_unknown_weighting_strategy_raises_value_error():
    ds = pd.date_range("2023-01-01", periods=2, freq="D")
    with pytest.raises(ValueError):
        build_ensemble(
            per_model_mape={"a": 10.0},
            per_model_predictions={"a": make_predictions(ds, [1, 2])},
            mape_threshold=100.0,
            weighting="bogus",
        )

# ---------- combination correctness ----------

def test_combined_yhat_is_the_correct_weighted_sum():
    ds = pd.date_range("2023-01-01", periods=2, freq="D")
    result = build_ensemble(
        per_model_mape={"a": 10.0, "b": 10.0},  # equal MAPE -> equal inverse-mape weight
        per_model_predictions={
            "a": make_predictions(ds, [10.0, 20.0]),
            "b": make_predictions(ds, [30.0, 40.0]),
        },
        mape_threshold=100.0,
        weighting="inverse_mape",
    )
    # equal MAPE -> 0.5/0.5 weights -> simple average
    assert list(result.combined["yhat"]) == pytest.approx([20.0, 30.0])


def test_single_surviving_model_ensemble_equals_that_models_own_predictions():
    ds = pd.date_range("2023-01-01", periods=2, freq="D")
    result = build_ensemble(
        per_model_mape={"a": 10.0},
        per_model_predictions={"a": make_predictions(ds, [10.0, 20.0])},
        mape_threshold=100.0,
    )
    assert result.weights == {"a": 1.0}
    assert list(result.combined["yhat"]) == [10.0, 20.0]


def test_misaligned_ds_between_models_raises_value_error():
    ds_a = pd.date_range("2023-01-01", periods=3, freq="D")
    ds_b = pd.date_range("2023-06-01", periods=3, freq="D")  # no overlap at all
    with pytest.raises(ValueError):
        build_ensemble(
            per_model_mape={"a": 10.0, "b": 10.0},
            per_model_predictions={
                "a": make_predictions(ds_a, [1, 2, 3]),
                "b": make_predictions(ds_b, [4, 5, 6]),
            },
            mape_threshold=100.0,
        )

# ---------- threshold rejection ----------

def test_model_above_threshold_is_rejected():
    ds = pd.date_range("2023-01-01", periods=2, freq="D")
    result = build_ensemble(
        per_model_mape={"good": 5.0, "bad": 90.0},
        per_model_predictions={
            "good": make_predictions(ds, [1, 2]),
            "bad": make_predictions(ds, [1, 2]),
        },
        mape_threshold=50.0,
    )
    assert "bad" in result.rejected_models
    assert isinstance(result.rejected_models["bad"], ThresholdRejectionError)
    assert result.rejected_models["bad"].mape_value == 90.0
    assert result.rejected_models["bad"].allowed_max == 50.0
    assert result.weights == {"good": 1.0}


def test_unusually_low_mape_is_kept_not_rejected():
    """
    mape_threshold is an upper bound only. A suspiciously low MAPE -- which
    the old symmetric [min_mape, max_mape] band used to reject -- is now
    kept, since there is no lower bound anymore.
    """
    ds = pd.date_range("2023-01-01", periods=2, freq="D")
    result = build_ensemble(
        per_model_mape={"suspiciously_good": 0.5, "normal": 10.0},
        per_model_predictions={
            "suspiciously_good": make_predictions(ds, [1, 2]),
            "normal": make_predictions(ds, [1, 2]),
        },
        mape_threshold=100.0,
    )
    assert "suspiciously_good" not in result.rejected_models
    assert set(result.weights.keys()) == {"suspiciously_good", "normal"}


def test_nan_mape_is_rejected_not_crashed_on():
    ds = pd.date_range("2023-01-01", periods=2, freq="D")
    result = build_ensemble(
        per_model_mape={"nan_model": float("nan"), "normal": 10.0},
        per_model_predictions={
            "nan_model": make_predictions(ds, [1, 2]),
            "normal": make_predictions(ds, [1, 2]),
        },
        mape_threshold=100.0,
    )
    assert "nan_model" in result.rejected_models
    assert result.weights == {"normal": 1.0}


def test_all_models_rejected_returns_empty_result_without_raising():
    """
    Deliberately does NOT raise (unlike AllModelsRejectedError elsewhere
    in this codebase) -- see evaluation/ensemble.py's module docstring
    for why: the per-model results that produced these MAPEs are still
    valid and returned by evaluate()/forecast(), only the ensemble
    combination step itself has nothing to combine.
    """
    ds = pd.date_range("2023-01-01", periods=2, freq="D")
    result = build_ensemble(
        per_model_mape={"a": 90.0, "b": 95.0},
        per_model_predictions={
            "a": make_predictions(ds, [1, 2]),
            "b": make_predictions(ds, [1, 2]),
        },
        mape_threshold=50.0,
    )
    assert isinstance(result, EnsembleResult)
    assert result.combined is None
    assert result.weights == {}
    assert set(result.rejected_models.keys()) == {"a", "b"}

# ---------- ensemble MAPE scoring ----------

def test_ensemble_mape_is_computed_when_actuals_provided():
    ds = pd.date_range("2023-01-01", periods=3, freq="D")
    actuals = pd.DataFrame({"ds": ds, "y": [10.0, 20.0, 30.0]})
    result = build_ensemble(
        per_model_mape={"a": 10.0},
        per_model_predictions={"a": make_predictions(ds, [10.0, 20.0, 30.0])},
        mape_threshold=100.0,
        actuals=actuals,
    )
    assert result.mape is not None
    assert result.mape.value == pytest.approx(0.0)


def test_ensemble_mape_is_none_without_actuals():
    ds = pd.date_range("2023-01-01", periods=2, freq="D")
    result = build_ensemble(
        per_model_mape={"a": 10.0},
        per_model_predictions={"a": make_predictions(ds, [1, 2])},
        mape_threshold=100.0,
        actuals=None,
    )
    assert result.mape is None


def test_ensemble_mape_raises_on_misaligned_actuals():
    ds = pd.date_range("2023-01-01", periods=3, freq="D")
    misaligned_ds = pd.date_range("2024-01-01", periods=3, freq="D")
    actuals = pd.DataFrame({"ds": misaligned_ds, "y": [10.0, 20.0, 30.0]})
    with pytest.raises(ValueError):
        build_ensemble(
            per_model_mape={"a": 10.0},
            per_model_predictions={"a": make_predictions(ds, [10.0, 20.0, 30.0])},
            mape_threshold=100.0,
            actuals=actuals,
        )
