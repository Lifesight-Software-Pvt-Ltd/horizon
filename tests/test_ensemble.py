import numpy as np
import pandas as pd
import pytest

from horizon_forecaster.evaluation.ensemble import (
    DEFAULT_ENSEMBLE_METHOD,
    ENSEMBLE_METHODS,
    build_ensemble,
    build_ensembles,
)


def _ds(n=2):
    return pd.date_range("2023-01-01", periods=n)


def _preds(values_by_model, n=None):
    """{"a": [1, 2], ...} -> {"a": DataFrame(ds, yhat), ...}"""
    n = n or len(next(iter(values_by_model.values())))
    ds = _ds(n)
    return {name: pd.DataFrame({"ds": ds, "yhat": vals}) for name, vals in values_by_model.items()}


# ---------- weighting math ----------

def test_inverse_mape_weighting_favors_lower_mape_model():
    result = build_ensemble(
        per_model_selection_mape={"a": 10.0, "b": 30.0},
        per_model_predictions=_preds({"a": [10.0, 20.0], "b": [30.0, 40.0]}),
        method="inverse_mape",
    )
    assert result.weights["a"] > result.weights["b"]
    assert result.weights["a"] + result.weights["b"] == pytest.approx(1.0)


def test_equal_weighting_ignores_mape_differences():
    result = build_ensemble(
        per_model_selection_mape={"a": 10.0, "b": 30.0},
        per_model_predictions=_preds({"a": [10.0, 20.0], "b": [30.0, 40.0]}),
        method="equal",
    )
    assert result.weights == {"a": 0.5, "b": 0.5}


def test_inverse_mape_weighting_does_not_divide_by_zero_on_perfect_mape():
    result = build_ensemble(
        per_model_selection_mape={"a": 0.0, "b": 5.0},
        per_model_predictions=_preds({"a": [10.0, 20.0], "b": [30.0, 40.0]}),
        method="inverse_mape",
    )
    assert result.weights["a"] == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(list(result.weights.values())).all()


def test_combined_yhat_is_the_correct_weighted_sum():
    result = build_ensemble(
        per_model_selection_mape={"a": 10.0, "b": 10.0},  # equal MAPE -> equal weight
        per_model_predictions=_preds({"a": [10.0, 20.0], "b": [30.0, 40.0]}),
        method="inverse_mape",
    )
    assert result.combined["yhat"].tolist() == [20.0, 30.0]


def test_unknown_method_raises_value_error():
    with pytest.raises(ValueError):
        build_ensemble(
            per_model_selection_mape={"a": 10.0},
            per_model_predictions=_preds({"a": [1.0, 2.0]}),
            method="bogus",
        )


# ---------- the trimmed rule ----------

def test_trimmed_is_the_default_method():
    result = build_ensemble(
        per_model_selection_mape={"a": 10.0, "b": 20.0, "c": 30.0},
        per_model_predictions=_preds({"a": [1.0, 1.0], "b": [2.0, 2.0], "c": [3.0, 3.0]}),
    )
    assert result.method == DEFAULT_ENSEMBLE_METHOD == "trimmed"


def test_trimmed_discards_the_high_and_low_member_per_period():
    # Members 'lo' and 'hi' are wild in opposite directions; the middle
    # three all sit on the truth. Trimming should land exactly on it.
    result = build_ensemble(
        per_model_selection_mape={n: 10.0 for n in ("lo", "a", "b", "c", "hi")},
        per_model_predictions=_preds({
            "lo": [1.0, 1.0], "a": [10.0, 10.0], "b": [12.0, 12.0],
            "c": [14.0, 14.0], "hi": [900.0, 900.0],
        }),
        method="trimmed",
    )
    assert result.combined["yhat"].tolist() == [12.0, 12.0]
    assert result.rule_applied == "trimmed"
    assert result.degraded is False


def test_trimmed_carries_no_weights():
    result = build_ensemble(
        per_model_selection_mape={n: 10.0 for n in ("a", "b", "c")},
        per_model_predictions=_preds({"a": [1.0, 1.0], "b": [2.0, 2.0], "c": [3.0, 3.0]}),
        method="trimmed",
    )
    # None, NOT an empty dict: {} already means "nothing survived", and an
    # equal split would imply an averaging that did not happen.
    assert result.weights is None


def test_four_members_is_named_median_not_trimmed():
    result = build_ensemble(
        per_model_selection_mape={n: 10.0 for n in ("a", "b", "c", "d")},
        per_model_predictions=_preds({
            "a": [1.0, 1.0], "b": [2.0, 2.0], "c": [4.0, 4.0], "d": [100.0, 100.0],
        }),
        method="trimmed",
    )
    assert result.rule_applied == "median"
    assert result.combined["yhat"].tolist() == [3.0, 3.0]  # mean of the middle two
    assert result.degraded is False


def test_trim_shares_identify_the_misbehaving_member():
    result = build_ensemble(
        per_model_selection_mape={n: 10.0 for n in ("a", "b", "runaway")},
        per_model_predictions=_preds({
            "a": [10.0, 10.0], "b": [11.0, 11.0], "runaway": [999.0, 999.0],
        }),
        method="trimmed",
    )
    shares = result.trim_shares.set_index("model")
    assert shares.loc["runaway", "trimmed_high"] == 2
    assert shares.loc["runaway", "trimmed_fraction"] == pytest.approx(1.0)
    assert shares.loc["b", "trimmed_total"] == 0


# ---------- the degradation ladder ----------

def test_three_members_still_trims():
    result = build_ensemble(
        per_model_selection_mape={n: 10.0 for n in ("a", "b", "c")},
        per_model_predictions=_preds({"a": [1.0, 1.0], "b": [5.0, 5.0], "c": [90.0, 90.0]}),
        method="trimmed",
    )
    assert result.rule_applied == "trimmed"
    assert result.degraded is False
    assert result.combined["yhat"].tolist() == [5.0, 5.0]


def test_two_members_falls_back_to_plain_mean_and_reports_degraded():
    # Trimming two values is undefined -- removing the high and the low
    # leaves nothing. This is an arithmetic boundary, not a tuned cutoff.
    result = build_ensemble(
        per_model_selection_mape={"a": 10.0, "b": 10.0},
        per_model_predictions=_preds({"a": [10.0, 10.0], "b": [20.0, 20.0]}),
        method="trimmed",
    )
    assert result.rule_applied == "mean"
    assert result.degraded is True
    assert result.combined["yhat"].tolist() == [15.0, 15.0]


def test_single_member_returns_that_models_predictions_and_reports_degraded():
    result = build_ensemble(
        per_model_selection_mape={"a": 10.0},
        per_model_predictions=_preds({"a": [10.0, 20.0]}),
        method="trimmed",
    )
    assert result.rule_applied == "single"
    assert result.degraded is True
    assert result.combined["yhat"].tolist() == [10.0, 20.0]


def test_no_members_returns_empty_result_without_raising():
    result = build_ensemble(
        per_model_selection_mape={"a": 90.0, "b": 95.0},
        per_model_predictions=_preds({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
        mape_threshold=50.0,
        method="trimmed",
    )
    assert result.combined is None
    assert result.rule_applied == "none"
    assert result.members == []
    assert set(result.rejected_models) == {"a", "b"}


def test_weighted_rules_never_report_degraded():
    # A weighted average of two (or one) is perfectly well defined, so the
    # ladder is specific to trimming.
    result = build_ensemble(
        per_model_selection_mape={"a": 10.0, "b": 20.0},
        per_model_predictions=_preds({"a": [10.0, 10.0], "b": [20.0, 20.0]}),
        method="inverse_mape",
    )
    assert result.degraded is False
    assert result.rule_applied == "inverse_mape"


# ---------- screening ----------

def test_model_above_threshold_is_rejected():
    result = build_ensemble(
        per_model_selection_mape={"good": 5.0, "bad": 90.0},
        per_model_predictions=_preds({"good": [1.0, 2.0], "bad": [100.0, 200.0]}),
        mape_threshold=50.0,
    )
    assert "bad" in result.rejected_models
    assert result.members == ["good"]
    assert result.rejected_models["bad"].mape_value == 90.0
    assert result.rejected_models["bad"].allowed_max == 50.0


def test_unusually_low_mape_is_kept_not_rejected():
    result = build_ensemble(
        per_model_selection_mape={"suspiciously_good": 0.5, "normal": 10.0},
        per_model_predictions=_preds({"suspiciously_good": [1.0, 2.0], "normal": [3.0, 4.0]}),
        mape_threshold=50.0,
    )
    assert result.rejected_models == {}


def test_threshold_none_admits_every_model():
    result = build_ensemble(
        per_model_selection_mape={"a": 5.0, "b": 300.0, "c": 12.0},
        per_model_predictions=_preds({"a": [1.0, 1.0], "b": [9.0, 9.0], "c": [2.0, 2.0]}),
        mape_threshold=None,
    )
    assert result.rejected_models == {}
    assert set(result.members) == {"a", "b", "c"}


def test_threshold_screening_can_degrade_the_trimmed_rule():
    # Screening and trimming both spend robustness. Screening five members
    # down to two leaves trimming undefined -- documented behaviour, and a
    # reason not to treat the threshold as an accuracy screen.
    result = build_ensemble(
        per_model_selection_mape={"a": 5.0, "b": 6.0, "c": 80.0, "d": 90.0, "e": 95.0},
        per_model_predictions=_preds({n: [1.0, 1.0] for n in ("a", "b", "c", "d", "e")}),
        mape_threshold=10.0,
        method="trimmed",
    )
    assert result.members == ["a", "b"]
    assert result.rule_applied == "mean"
    assert result.degraded is True


# ---------- NaN selection MAPE ----------

def test_nan_mape_model_is_kept_by_the_trimmed_rule():
    # A model whose MAPE is undefined (every actual in the window was zero)
    # genuinely learned the series -- only the metric broke. Trimming needs
    # no MAPE, so it should still contribute.
    result = build_ensemble(
        per_model_selection_mape={"nan_model": float("nan"), "b": 10.0, "c": 12.0},
        per_model_predictions=_preds({
            "nan_model": [1.0, 1.0], "b": [2.0, 2.0], "c": [3.0, 3.0],
        }),
        method="trimmed",
    )
    assert "nan_model" in result.members
    assert result.rejected_models == {}


def test_nan_mape_model_is_rejected_by_weighted_rules():
    result = build_ensemble(
        per_model_selection_mape={"nan_model": float("nan"), "normal": 10.0},
        per_model_predictions=_preds({"nan_model": [1.0, 2.0], "normal": [3.0, 4.0]}),
        method="inverse_mape",
    )
    assert "nan_model" in result.rejected_models
    assert result.members == ["normal"]


# ---------- alignment ----------

def test_misaligned_ds_between_models_raises_value_error():
    a = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=2), "yhat": [1.0, 2.0]})
    b = pd.DataFrame({"ds": pd.date_range("2023-06-01", periods=2), "yhat": [3.0, 4.0]})
    with pytest.raises(ValueError, match="don't share the same 'ds'"):
        build_ensemble(
            per_model_selection_mape={"a": 10.0, "b": 10.0},
            per_model_predictions={"a": a, "b": b},
            method="inverse_mape",
        )


# ---------- ensemble scoring ----------

def test_ensemble_mape_is_computed_when_actuals_provided():
    ds = _ds(2)
    result = build_ensemble(
        per_model_selection_mape={"a": 10.0, "b": 10.0},
        per_model_predictions=_preds({"a": [10.0, 20.0], "b": [30.0, 40.0]}),
        method="inverse_mape",
        actuals=pd.DataFrame({"ds": ds, "y": [20.0, 30.0]}),
    )
    assert result.mape is not None
    assert result.mape.value == pytest.approx(0.0)


def test_ensemble_mape_is_none_without_actuals():
    result = build_ensemble(
        per_model_selection_mape={"a": 10.0},
        per_model_predictions=_preds({"a": [10.0, 20.0]}),
    )
    assert result.mape is None


def test_ensemble_mape_raises_on_misaligned_actuals():
    misaligned = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=2), "y": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="Cannot score ensemble MAPE"):
        build_ensemble(
            per_model_selection_mape={"a": 10.0},
            per_model_predictions=_preds({"a": [10.0, 20.0]}),
            actuals=misaligned,
        )


# ---------- build_ensembles ----------

def test_build_ensembles_returns_one_result_per_method():
    results = build_ensembles(
        methods=ENSEMBLE_METHODS,
        per_model_selection_mape={n: 10.0 for n in ("a", "b", "c")},
        per_model_predictions=_preds({"a": [1.0, 1.0], "b": [2.0, 2.0], "c": [30.0, 30.0]}),
    )
    assert set(results) == set(ENSEMBLE_METHODS)
    assert results["trimmed"].weights is None
    assert results["equal"].weights == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    # The outlier drags the averaging rules but not the trimmed one.
    assert results["trimmed"].combined["yhat"].tolist() == [2.0, 2.0]
    assert results["equal"].combined["yhat"].tolist() == [11.0, 11.0]


def test_build_ensembles_is_always_a_dict_even_for_one_method():
    results = build_ensembles(
        methods=["trimmed"],
        per_model_selection_mape={"a": 10.0},
        per_model_predictions=_preds({"a": [1.0, 2.0]}),
    )
    assert isinstance(results, dict)
    assert list(results) == ["trimmed"]
