import warnings

import pandas as pd
import pytest

from horizon_forecaster.exceptions import (
    AllModelsRejectedError,
    InsufficientDataError,
    InvalidConfigError,
    ModelConvergenceError,
    UnsupportedFrequencyError,
)
from horizon_forecaster.forecaster import HorizonForecaster, EvaluationResult, ForecastResult, MODEL_REGISTRY

def make_raw_df(n=60, freq="D", with_spend=False, with_zero_variance_col=False):
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq=freq),
        "orders": [100 + i * 2 for i in range(n)],
    })
    if with_spend:
        df["spend"] = [50 + i for i in range(n)]
    if with_zero_variance_col:
        df["constant_col"] = 5  # same value every row -> zero variance
    return df

def make_noisy_raw_df(n=90, freq="D", seed=7):
    """
    Only for ensemble tests below. make_raw_df()'s perfectly clean,
    noiseless linear trend lets both models fit it almost exactly
    (MAPE well under 1%) -- which is BELOW the default min_mape=1.0
    threshold and gets threshold-rejected as "suspiciously good" (see
    evaluation/ensemble.py). A little real noise keeps MAPE in a
    realistic, comfortably-inside-[1,100] range without changing
    make_raw_df() itself and risking other tests' assumptions.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    trend = 100 + 2 * np.arange(n)
    noise = rng.normal(0, 3, n)
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq=freq),
        "orders": trend + noise,
    })

# ---------- constructor validation ----------

def test_constructor_validates_frequency():
    df = make_raw_df()
    with pytest.raises(UnsupportedFrequencyError):
        HorizonForecaster(df, date_column="date", target_column="orders", frequency="monthly", models=["prophet"])

def test_constructor_validates_training_size():
    df = make_raw_df()
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(
            df, date_column="date", target_column="orders", frequency="daily", training_size=30, models=["prophet"]
        )

def test_constructor_default_training_size_is_80_percent():
    df = make_raw_df()
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    assert forecaster.training_fraction == 0.8

def test_constructor_drops_zero_variance_independent_columns():
    df = make_raw_df(with_zero_variance_col=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["constant_col"],
        models=["prophet"],
    )
    assert "constant_col" in forecaster.dropped_zero_variance_columns
    # Critically: also removed from independent_columns, so the adapter's
    # hard-error never fires against a column WE dropped, not the user.
    assert "constant_col" not in forecaster.independent_columns

def test_constructor_sorts_data_chronologically():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
        "orders": [30, 10, 20],
    })
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    assert forecaster.raw_df["orders"].tolist() == [10, 20, 30]

def test_constructor_warns_when_declared_frequency_mismatches_actual_data():
    """
    The likely real-world bug this check exists for: a user declares
    frequency="weekly" but their actual date column is daily-spaced (or
    vice versa) - infer_frequency() (built in Phase 1) is used here to
    catch this, rather than silently proceeding with a mismatch.
    """
    daily_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=60, freq="D"),
        "orders": [100 + i * 2 for i in range(60)],
    })
    with pytest.warns(UserWarning, match="most common date spacing looks like 'daily'"):
        HorizonForecaster(daily_df, date_column="date", target_column="orders", frequency="weekly", models=["prophet"])

def test_constructor_does_not_warn_when_frequency_matches_data():
    daily_df = make_raw_df(n=60, freq="D")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HorizonForecaster(daily_df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])

def test_constructor_warns_but_does_not_block_on_irregular_data():
    """
    Real-world data is rarely perfectly regular - irregular spacing that
    doesn't cleanly match daily or weekly should warn, not raise, so
    legitimate (if slightly messy) data isn't rejected outright.
    """
    irregular_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-10", "2026-01-25"]),
        "orders": [10, 20, 30, 40],
    })
    with pytest.warns(UserWarning):
        forecaster = HorizonForecaster(
            irregular_df, date_column="date", target_column="orders", frequency="weekly", models=["prophet"]
        )
    assert forecaster is not None  # confirms it did NOT raise, just warned

def test_constructor_raises_immediately_when_training_size_leaves_no_test_rows():
    """
    training_size leaving zero test rows is a dataset-level configuration
    problem, identical for every model -- validated at construction time
    (fails fast, before any model even runs), not discovered later inside
    evaluate()'s per-model try/except.
    """
    df = make_raw_df(n=1)
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(
            df, date_column="date", target_column="orders", frequency="daily", training_size=100, models=["prophet"]
        )

# ---------- models= parameter ----------

def test_models_defaults_to_every_registered_model():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    assert set(forecaster.models) == set(MODEL_REGISTRY.keys())
    assert set(forecaster.models) == {"prophet", "catboost"}

def test_models_can_be_restricted_to_a_subset():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    assert forecaster.models == ["prophet"]

def test_models_raises_on_unknown_model_name():
    df = make_raw_df(n=60)
    with pytest.raises(InvalidConfigError) as exc_info:
        HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet", "sarimax"])
    assert "sarimax" in str(exc_info.value)

def test_models_raises_on_empty_list():
    df = make_raw_df(n=60)
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=[])

# ---------- config loading ----------

def test_config_loaded_correctly_for_daily():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    assert forecaster._configs["prophet"]["wrapper_config"]["min_rows_required"] == 14
    assert forecaster._configs["prophet"]["base_hyperparameters"]["daily_seasonality"] is False
    assert len(forecaster._configs["prophet"]["hyperparameter_grid"]) == 6

def test_config_loaded_correctly_for_weekly():
    df = make_raw_df(n=60, freq="7D")
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="weekly", models=["prophet"])
    assert forecaster._configs["prophet"]["wrapper_config"]["min_rows_required"] == 8
    assert len(forecaster._configs["prophet"]["hyperparameter_grid"]) == 6

def test_config_loaded_independently_per_model():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    assert forecaster._configs["prophet"]["wrapper_config"]["min_rows_required"] == 14
    assert forecaster._configs["catboost"]["wrapper_config"]["min_rows_required"] == 14
    assert len(forecaster._configs["catboost"]["hyperparameter_grid"]) == 18
    # prophet and catboost have genuinely different hyperparameter keys
    assert "seasonality_mode" in forecaster._configs["prophet"]["hyperparameter_grid"][0]
    assert "iterations" in forecaster._configs["catboost"]["hyperparameter_grid"][0]


def test_model_params_override_skips_grid_search_entirely():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet"],
        model_params={"prophet": {"seasonality_mode": "multiplicative", "yearly_seasonality": 20}},
    )
    assert forecaster._use_grid_search["prophet"] is False
    # Resolved immediately at construction, not lazily
    assert forecaster._selected_hyperparameters["prophet"]["seasonality_mode"] == "multiplicative"
    assert forecaster._selected_hyperparameters["prophet"]["yearly_seasonality"] == 20
    # base_hyperparameters (not part of the override) still applied
    assert forecaster._selected_hyperparameters["prophet"]["daily_seasonality"] is False


def test_model_params_override_actually_reaches_the_fitted_prophet_model():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet"],
        model_params={"prophet": {"seasonality_mode": "multiplicative"}},
    )
    result = forecaster.evaluate()["prophet"]
    assert result.fitted_model.params["seasonality_mode"] == "multiplicative"
    assert result.grid_search_summary is None  # no search happened, nothing to report


def test_no_model_params_uses_grid_search_mode():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    assert forecaster._use_grid_search["prophet"] is True
    # Not resolved yet — lazy, until evaluate()/forecast() actually runs it
    assert forecaster._selected_hyperparameters["prophet"] is None


def test_model_params_are_independent_per_model():
    """
    Giving prophet an explicit override should not affect catboost's own
    (independent) decision to use grid search, and vice versa.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        model_params={"prophet": {"seasonality_mode": "multiplicative"}},  # catboost gets no entry
    )
    assert forecaster._use_grid_search["prophet"] is False
    assert forecaster._use_grid_search["catboost"] is True


# ---------- hyperparameter grid search ----------

def test_grid_search_evaluates_all_six_candidates():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    result = forecaster.evaluate()["prophet"]
    assert result.grid_search_summary is not None
    assert len(result.grid_search_summary) == 6


def test_grid_search_summary_is_sorted_best_to_worst():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    result = forecaster.evaluate()["prophet"]
    mapes = [c["mape"] for c in result.grid_search_summary]
    assert mapes == sorted(mapes)


def test_grid_search_winner_matches_evaluate_result():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    result = forecaster.evaluate()["prophet"]
    assert result.mape.value == result.grid_search_summary[0]["mape"]


def test_grid_search_runs_only_once_across_repeated_evaluate_calls():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    result1 = forecaster.evaluate()["prophet"]
    result2 = forecaster.evaluate()["prophet"]
    # Same cached result object, not recomputed
    assert result1 is result2


def test_forecast_reuses_grid_search_winner_from_evaluate():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    forecaster.evaluate()
    winning_params = forecaster._selected_hyperparameters["prophet"]

    forecaster.forecast(horizon=5)
    assert forecaster._forecast_models["prophet"].params == winning_params


def test_forecast_runs_grid_search_itself_when_called_before_evaluate():
    """
    forecast() must be independently capable of selecting hyperparameters
    if evaluate() was never called first in this session.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    assert forecaster._selected_hyperparameters["prophet"] is None

    forecaster.forecast(horizon=5)
    assert forecaster._selected_hyperparameters["prophet"] is not None
    assert "prophet" in forecaster._grid_search_results


# ---------- evaluate() ----------

def test_evaluate_returns_structured_result():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    results = forecaster.evaluate()
    result = results["prophet"]

    assert isinstance(results, dict)
    assert isinstance(result, EvaluationResult)
    assert result.model_name == "prophet"
    assert result.n_train == 48  # 60 * 0.8
    assert result.n_test == 12
    assert len(result.test_predictions) == 12
    assert result.fitted_model.is_fitted is True


def test_evaluate_produces_low_mape_on_clean_trend():
    """
    A clean, noise-free linear trend should be near-trivially predictable
    -> confirms the whole wired pipeline (split -> adapt -> fit ->
    predict -> MAPE) produces a numerically sensible result, not just
    code that runs without error.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    result = forecaster.evaluate()["prophet"]
    assert result.mape.value < 5.0  # should be very low on a clean trend


def test_evaluate_test_predictions_have_expected_columns():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    result = forecaster.evaluate()["prophet"]
    assert set(["ds", "y", "yhat", "yhat_lower", "yhat_upper"]).issubset(
        set(result.test_predictions.columns)
    )


def test_evaluate_works_with_independent_columns():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
    )
    result = forecaster.evaluate()["prophet"]
    assert result.fitted_model._regressor_columns == ["spend"]


def test_evaluate_all_models_fail_raises_all_models_rejected_error():
    """
    A single requested model failing is treated identically to multiple
    models failing: if NOTHING succeeded, AllModelsRejectedError is
    raised rather than returning a silently-empty dict. The real
    underlying exception is still available via evaluate_errors.
    """
    df = make_raw_df(n=10)  # 80% train -> 8 rows, below daily floor of 14
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    with pytest.raises(AllModelsRejectedError) as exc_info:
        forecaster.evaluate()
    assert exc_info.value.rejected_models == ["prophet"]
    assert isinstance(forecaster.evaluate_errors["prophet"], InsufficientDataError)


def test_evaluate_weekly_frequency_end_to_end():
    df = make_raw_df(n=40, freq="7D")
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="weekly", models=["prophet"])
    result = forecaster.evaluate()["prophet"]
    assert result.model_name == "prophet"
    assert result.mape.value < 10.0


# ---------- multi-model evaluate()/forecast() ----------

def test_evaluate_runs_both_models_by_default():
    df = make_raw_df(n=90)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"iterations": 100, "depth": 3}},
    )
    results = forecaster.evaluate()
    assert set(results.keys()) == {"prophet", "catboost"}
    assert isinstance(results["prophet"], EvaluationResult)
    assert isinstance(results["catboost"], EvaluationResult)


def test_forecast_runs_both_models_by_default():
    df = make_raw_df(n=90)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"iterations": 100, "depth": 3}},
    )
    results = forecaster.forecast(horizon=5)
    assert set(results.keys()) == {"prophet", "catboost"}
    for result in results.values():
        assert isinstance(result, ForecastResult)
        assert len(result.forecast) == 5


def test_evaluate_partial_failure_returns_successful_models_and_records_errors():
    """
    CatBoostAdapter's redesign removed the lag-warmup mechanism that
    used to give CatBoost a higher EFFECTIVE min-rows floor than
    Prophet's (see CHANGELOG) -- both models now use the exact same raw
    row-count check. So a genuine partial failure here needs a real
    model-family failure, not a data-size coincidence: depth=-1 is a
    confirmed-empirically-real invalid CatBoost hyperparameter (see
    test_catboost.py), used here to make CatBoost fail while Prophet
    (given valid params) succeeds on the same dataset.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"depth": -1}},
    )
    results = forecaster.evaluate()
    assert set(results.keys()) == {"prophet"}
    assert set(forecaster.evaluate_errors.keys()) == {"catboost"}
    assert isinstance(forecaster.evaluate_errors["catboost"], ModelConvergenceError)


def test_evaluate_errors_reset_fresh_on_each_call_not_accumulated():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"depth": -1}},
    )
    forecaster.evaluate()
    errors_first_call = dict(forecaster.evaluate_errors)
    forecaster.evaluate()
    errors_second_call = dict(forecaster.evaluate_errors)
    assert set(errors_first_call.keys()) == set(errors_second_call.keys()) == {"catboost"}


def test_evaluate_all_models_fail_raises_all_models_rejected_error_multi_model():
    df = make_raw_df(n=10)  # below both models' floors
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    with pytest.raises(AllModelsRejectedError) as exc_info:
        forecaster.evaluate()
    assert set(exc_info.value.rejected_models) == {"prophet", "catboost"}


def test_grid_search_is_independent_per_model():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        model_params={"prophet": {"seasonality_mode": "additive"}},  # catboost gets no override
    )
    results = forecaster.evaluate()
    assert results["prophet"].grid_search_summary is None
    assert results["catboost"].grid_search_summary is not None
    assert len(results["catboost"].grid_search_summary) == 18


def test_catboost_test_split_features_are_anchored_to_train_history_not_test_split():
    """
    Direct regression test for the Phase 3 architecture fix, updated for
    CatBoostAdapter's redesigned (lag/rolling-free) feature set: the
    test split's features must be built via transform_future() (reusing
    the adapter's retained reference_date from transform_train()), not
    transform_train() on the test split alone -- the latter would reset
    time_idx back to 0 at the start of the test split instead of
    continuing the trend seamlessly from where training left off.
    """
    df = make_raw_df(n=90)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        models=["catboost"], model_params={"catboost": {"iterations": 50, "depth": 3}},
    )
    n_train, n_test, adapted_train, adapted_test_features, y_test = forecaster._prepare_split("catboost")

    assert adapted_test_features["time_idx"].iloc[0] == adapted_train["time_idx"].iloc[-1] + 1
    # and time_idx must vary per row across the test split, not be frozen
    assert adapted_test_features["time_idx"].nunique() == n_test


# ---------- forecast() ----------

def test_forecast_returns_structured_result():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    results = forecaster.forecast(horizon=10)
    result = results["prophet"]

    assert isinstance(results, dict)
    assert isinstance(result, ForecastResult)
    assert result.model_name == "prophet"
    assert result.horizon == 10
    assert len(result.forecast) == 10
    assert result.used_default_future_strategy is False  # no independent columns at all


def test_forecast_continues_trend_correctly():
    """
    orders = 100 + i*2, so trend is +2/period. We check the diffs are
    CLOSE to +2 rather than exactly == 2.0: whichever grid candidate wins
    (picked by lowest test-set MAPE) may be a multiplicative-seasonality
    one, which introduces tiny (sub-0.01) numerical deviation from a pure
    linear trend even on clean data. Exact equality made this test flaky
    depending on which of the 6 real candidates the grid search selects.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    result = forecaster.forecast(horizon=5)["prophet"]
    yhats = result.forecast["yhat"].tolist()
    diffs = [yhats[i + 1] - yhats[i] for i in range(len(yhats) - 1)]
    assert all(1.9 <= d <= 2.1 for d in diffs), diffs


def test_forecast_raises_on_non_positive_horizon():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    with pytest.raises(InvalidConfigError):
        forecaster.forecast(horizon=0)
    with pytest.raises(InvalidConfigError):
        forecaster.forecast(horizon=-5)


def test_forecast_refits_only_once_and_caches_across_calls():
    """
    Critical design contract: the full-data refit must happen exactly
    once per instance, regardless of how many times forecast() is
    called afterward (even with different horizons).
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])

    assert "prophet" not in forecaster._forecast_models
    forecaster.forecast(horizon=5)
    model_after_first_call = forecaster._forecast_models["prophet"]
    assert model_after_first_call is not None

    forecaster.forecast(horizon=10)  # different horizon
    model_after_second_call = forecaster._forecast_models["prophet"]

    assert model_after_first_call is model_after_second_call  # same object, not refit


def test_forecast_uses_default_strategy_when_independent_data_not_supplied():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
    )
    result = forecaster.forecast(horizon=5)["prophet"]
    assert result.used_default_future_strategy is True
    assert len(result.forecast) == 5


def test_forecast_uses_supplied_future_independent_data_when_provided():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
    )
    future_spend = pd.DataFrame({"spend": [500, 510, 520, 530, 540]})
    result = forecaster.forecast(horizon=5, future_independent_data=future_spend)["prophet"]
    assert result.used_default_future_strategy is False
    assert len(result.forecast) == 5


def test_default_future_strategy_reuses_last_n_periods_when_horizon_fits_within_history():
    df = make_raw_df(n=60, with_spend=True)  # spend = 50 + i, last values: ..., 107, 108, 109
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
    )
    default_future = forecaster._build_default_future_independent_data(horizon=3)
    # last 3 actual spend values in a 60-row series starting at 50: 107, 108, 109
    assert default_future["spend"].tolist() == [107, 108, 109]


def test_default_future_strategy_cycles_when_horizon_exceeds_history():
    df = make_raw_df(n=20, with_spend=True)  # spend = 50..69 (20 values)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
    )
    # horizon of 45 > 20 rows of history -> must cycle, not crash
    default_future = forecaster._build_default_future_independent_data(horizon=45)
    assert len(default_future) == 45
    assert not default_future["spend"].isna().any()


def test_forecast_warns_when_horizon_far_exceeds_history():
    df = make_raw_df(n=20, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
    )
    with pytest.warns(UserWarning, match="Forecasting 45 periods from only 20 periods"):
        forecaster.forecast(horizon=45)


def test_forecast_does_not_warn_when_horizon_is_reasonable():
    """
    Isolates specifically the horizon-vs-history warning (see
    test_forecast_warns_when_horizon_far_exceeds_history above) from
    the unrelated Phase 4 "running evaluate() internally" warning --
    evaluate() is called first, outside the strict block, so its
    one-time warning doesn't get mistaken for a horizon warning that
    shouldn't have fired.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    forecaster.evaluate()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        forecaster.forecast(horizon=10)  # well within 2x of 60 rows of history

def test_forecast_weekly_frequency_produces_correct_future_dates():
    df = make_raw_df(n=40, freq="7D")
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="weekly", models=["prophet"])
    result = forecaster.forecast(horizon=3)["prophet"]
    last_train_date = df["date"].max()
    expected_first_future_date = last_train_date + pd.Timedelta(days=7)
    assert result.forecast["ds"].iloc[0] == expected_first_future_date


def test_forecast_all_models_fail_raises_all_models_rejected_error():
    df = make_raw_df(n=10)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", models=["prophet"])
    with pytest.raises(AllModelsRejectedError):
        forecaster.forecast(horizon=5)
    assert isinstance(forecaster.forecast_errors["prophet"], InsufficientDataError)


def test_forecast_partial_failure_returns_successful_models():
    """
    See test_evaluate_partial_failure_returns_successful_models_and_records_errors
    for why depth=-1 (a genuine model-family failure) is used here rather
    than a data-size coincidence.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"depth": -1}},
    )
    results = forecaster.forecast(horizon=3)
    assert set(results.keys()) == {"prophet"}
    assert set(forecaster.forecast_errors.keys()) == {"catboost"}


# ---------- ensemble (Phase 4) ----------

def test_constructor_defaults_for_ensemble_params():
    df = make_raw_df()
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    assert forecaster.min_mape == 1.0
    assert forecaster.max_mape == 100.0
    assert forecaster.ensemble_weighting == "inverse_mape"
    assert forecaster.evaluate_ensemble is None
    assert forecaster.forecast_ensemble is None


def test_constructor_validates_mape_threshold_range():
    df = make_raw_df()
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", min_mape=50, max_mape=10)


def test_constructor_validates_ensemble_weighting():
    df = make_raw_df()
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily", ensemble_weighting="bogus")


def test_evaluate_populates_ensemble_with_default_permissive_threshold():
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"iterations": 100, "depth": 3}},
    )
    forecaster.evaluate()
    ensemble = forecaster.evaluate_ensemble
    assert ensemble is not None
    assert ensemble.combined is not None
    assert set(ensemble.weights.keys()) == {"prophet", "catboost"}
    assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-6
    assert ensemble.mape is not None  # actuals are available in the evaluate() context
    assert set(ensemble.combined.columns) == {"ds", "yhat"}


def test_evaluate_ensemble_uses_configured_weighting_strategy():
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"iterations": 100, "depth": 3}},
        ensemble_weighting="equal",
    )
    forecaster.evaluate()
    assert forecaster.evaluate_ensemble.weights["prophet"] == pytest.approx(0.5)
    assert forecaster.evaluate_ensemble.weights["catboost"] == pytest.approx(0.5)


def test_evaluate_ensemble_rejects_models_outside_impossible_threshold():
    """
    A min_mape of 99 rejects the model from the ensemble regardless of
    its actual (necessarily much lower, for any real fit) MAPE -- but
    does NOT affect evaluate()'s own return value -- the per-model
    result is still returned, only evaluate_ensemble reflects the
    rejection. This is the key distinction from AllModelsRejectedError
    (see evaluation/ensemble.py's module docstring). A high min_mape is
    used here (rather than a near-zero max_mape) specifically because
    it's valid under validate_mape_threshold_range's [1, 100] floor and
    doesn't depend on knowing the real MAPE precisely in advance.
    """
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        models=["prophet"], model_params={"prophet": {"seasonality_mode": "additive"}},
        min_mape=99.0, max_mape=100.0,
    )
    results = forecaster.evaluate()
    assert "prophet" in results
    assert forecaster.evaluate_ensemble.combined is None
    assert forecaster.evaluate_ensemble.weights == {}
    assert forecaster.evaluate_ensemble.mape is None
    assert "prophet" in forecaster.evaluate_ensemble.rejected_models


def test_forecast_ensemble_reuses_evaluate_weights_and_warns_when_evaluate_not_called_first():
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"iterations": 100, "depth": 3}},
    )
    assert forecaster._evaluate_mape_cache is None
    with pytest.warns(UserWarning, match=r"running evaluate\(\) internally"):
        forecaster.forecast(horizon=5)

    assert forecaster._evaluate_mape_cache is not None
    assert forecaster.evaluate_ensemble is not None  # populated as a side effect
    assert forecaster.forecast_ensemble is not None
    assert forecaster.forecast_ensemble.weights == forecaster.evaluate_ensemble.weights
    assert set(forecaster.forecast_ensemble.combined.columns) == {"ds", "yhat"}
    assert len(forecaster.forecast_ensemble.combined) == 5
    assert forecaster.forecast_ensemble.mape is None  # no actuals exist for the future


def test_forecast_does_not_rewarn_or_recompute_evaluate_on_second_call():
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"iterations": 100, "depth": 3}},
    )
    forecaster.forecast(horizon=5)
    cache_after_first_call = forecaster._evaluate_mape_cache

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning now fails the test
        forecaster.forecast(horizon=8)

    assert forecaster._evaluate_mape_cache is cache_after_first_call  # same object, not recomputed


def test_forecast_skips_internal_evaluate_when_already_called_explicitly():
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"iterations": 100, "depth": 3}},
    )
    forecaster.evaluate()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        forecaster.forecast(horizon=5)  # must not warn -- evaluate() already ran


def test_forecast_ensemble_is_none_when_internal_evaluate_totally_fails():
    """
    n=16 -> evaluate()'s 80% train split is 12 rows, below the daily
    floor of 14 (fails outright), but forecast()'s own full-data refit
    uses all 16 rows, which clears the floor. This confirms forecast()
    still attempts (and succeeds at) its own fits even when the
    internal evaluate() call it runs for ensemble weights finds nothing
    at all -- a model failing on the smaller train/test split doesn't
    block forecast() from succeeding with more data.
    """
    df = make_raw_df(n=16)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        models=["prophet"], model_params={"prophet": {"seasonality_mode": "additive"}},
    )
    with pytest.warns(UserWarning, match=r"running evaluate\(\) internally"):
        results = forecaster.forecast(horizon=3)

    assert "prophet" in results
    assert forecaster._evaluate_mape_cache == {}
    assert forecaster.forecast_ensemble is None
