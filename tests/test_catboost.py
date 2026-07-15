import pandas as pd
import pytest

from horizon_forecaster.exceptions import (
    InsufficientDataError,
    ModelConvergenceError,
    ModelPredictionError,
)
from horizon_forecaster.models.catboost_wrapper import CatBoostWrapper

def make_train_df(n=30):
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "ds": dates,
        "y": [10 + i * 0.5 for i in range(n)],
        "day_of_week": [d.dayofweek for d in dates],
        "feature_a": [100 + i * 2.0 for i in range(n)],
    })

def make_future_df(n=5, start="2026-01-31"):
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "ds": dates,
        "day_of_week": [d.dayofweek for d in dates],
        "feature_a": [160.0 + i * 2 for i in range(n)],
    })

# ---------- fit() ----------

def test_fit_succeeds_on_sufficient_data():
    model = CatBoostWrapper(frequency="daily")
    result = model.fit(make_train_df())
    assert result is model
    assert model.is_fitted is True

def test_fit_raises_insufficient_data_error_below_min_rows():
    model = CatBoostWrapper(frequency="daily")
    small_df = make_train_df(n=5)
    with pytest.raises(InsufficientDataError) as exc_info:
        model.fit(small_df)
    assert model.is_fitted is False
    assert exc_info.value.available_periods == 5

def test_fit_raises_exactly_one_row_below_min_rows_boundary():
    model = CatBoostWrapper(frequency="daily")
    df = make_train_df(n=model.min_rows_required - 1)
    with pytest.raises(InsufficientDataError):
        model.fit(df)

def test_fit_succeeds_at_exact_min_rows_boundary():
    model = CatBoostWrapper(frequency="daily")
    df = make_train_df(n=model.min_rows_required)
    model.fit(df)
    assert model.is_fitted is True

def test_min_rows_required_is_frequency_aware_by_default():
    daily_model = CatBoostWrapper(frequency="daily")
    weekly_model = CatBoostWrapper(frequency="weekly")
    assert daily_model.min_rows_required == 14
    assert weekly_model.min_rows_required == 8
    assert daily_model.min_rows_required != weekly_model.min_rows_required

def test_min_rows_required_can_be_explicitly_overridden():
    model = CatBoostWrapper(frequency="daily", min_rows_required=100)
    assert model.min_rows_required == 100
    df = make_train_df(n=50)
    with pytest.raises(InsufficientDataError):
        model.fit(df)  # 50 < explicit override of 100

def test_random_seed_defaults_to_42_and_can_be_overridden():
    default_model = CatBoostWrapper(frequency="daily")
    custom_model = CatBoostWrapper(frequency="daily", random_seed=123)
    assert default_model.random_seed == 42
    assert custom_model.random_seed == 123

def test_fit_raises_model_prediction_error_when_missing_ds_or_y_columns():
    model = CatBoostWrapper(frequency="daily")
    bad_df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=15), "value": range(15)})
    with pytest.raises(ModelPredictionError):
        model.fit(bad_df)

def test_fit_records_feature_columns_excluding_ds_and_y():
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())
    assert set(model._feature_columns) == {"day_of_week", "feature_a"}

def test_fit_marks_no_columns_as_categorical():
    """
    No categorical features are declared at all (see class docstring):
    all calendar features are trained as plain numeric, matching a
    proven real-data reference notebook that used no explicit
    categorical designation. This holds regardless of which columns are
    present in train_df.
    """
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())
    assert model._categorical_feature_names == []

def test_fit_records_target_min_for_clipping_decisions():
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())  # y = 10 + i*0.5, always >= 10
    assert model._target_min == 10.0

def test_fit_accepts_param_overrides():
    model = CatBoostWrapper(frequency="daily", params={"depth": 3, "iterations": 50})
    model.fit(make_train_df())
    assert model.is_fitted is True

def test_fit_raises_model_convergence_error_on_genuine_catboost_failure():
    """
    An invalid CatBoost hyperparameter (negative tree depth) causes a
    genuine CatBoostError from the underlying library (confirmed
    empirically, not simulated) — exercises the real exception-
    translation path in fit(), not a fabricated one.
    """
    model = CatBoostWrapper(frequency="daily", params={"depth": -1})
    with pytest.raises(ModelConvergenceError) as exc_info:
        model.fit(make_train_df())
    assert model.is_fitted is False
    assert exc_info.value.original_exception is not None

# ---------- predict() ----------

def test_predict_before_fit_raises():
    model = CatBoostWrapper(frequency="daily")
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=5, future_exog=make_future_df())

def test_predict_with_none_future_exog_raises():
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=5, future_exog=None)

def test_predict_returns_ds_and_yhat_only_no_uncertainty_bounds():
    """
    Unlike ProphetWrapper, CatBoost has no native uncertainty output —
    BaseModel's contract says wrappers without it must omit the columns
    rather than fabricate them.
    """
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())
    forecast = model.predict(horizon=5, future_exog=make_future_df())
    assert list(forecast.columns) == ["ds", "yhat"]
    assert len(forecast) == 5

def test_predict_raises_when_horizon_does_not_match_future_exog_length():
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())
    future = make_future_df(n=7)
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=10, future_exog=future)  # mismatch: 7 rows vs horizon=10

def test_predict_raises_when_future_exog_missing_ds_column():
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())
    future = make_future_df().drop(columns=["ds"])
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=5, future_exog=future)

def test_predict_raises_naming_missing_required_feature_in_future_exog():
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())  # trained with day_of_week + feature_a
    future = make_future_df().drop(columns=["feature_a"])
    with pytest.raises(ModelPredictionError) as exc_info:
        model.predict(horizon=5, future_exog=future)
    assert "feature_a" in str(exc_info.value)

def test_predict_succeeds_with_all_required_features_present():
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())
    forecast = model.predict(horizon=5, future_exog=make_future_df())
    assert len(forecast) == 5
    assert not forecast["yhat"].isna().any()

def test_predict_ignores_extra_columns_not_used_at_fit_time():
    """
    future_exog may contain columns beyond what was trained on (e.g. a
    caller passing more context than strictly needed) -- predict()
    should select exactly self._feature_columns, not choke on extras.
    """
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())
    future = make_future_df()
    future["unused_extra_column"] = "ignore_me"
    forecast = model.predict(horizon=5, future_exog=future)
    assert len(forecast) == 5

def test_predict_raises_model_prediction_error_on_genuine_catboost_failure():
    """
    A non-numeric value in a column that was numeric at fit time causes
    a genuine CatBoost prediction failure (confirmed empirically) --
    exercises the real exception-translation path, not a fabricated one.
    Note this passes our own wrapper-level validation (the column IS
    present), so it genuinely reaches CatBoost's own internal predict()
    call before failing.
    """
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())
    future = make_future_df()
    future["feature_a"] = ["not_a_number"] * len(future)
    with pytest.raises(ModelPredictionError) as exc_info:
        model.predict(horizon=5, future_exog=future)
    assert exc_info.value.original_exception is not None

def test_predict_clips_negative_predictions_when_training_target_was_non_negative():
    """
    Data-driven clipping: if fit() only ever saw non-negative target
    values, predict() clips its raw model output to >= 0. The
    underlying fitted model's predict() is monkeypatched to return a
    controlled array (including negative values) so this test is
    deterministic, rather than depending on CatBoost happening to
    produce a negative raw value on its own.
    """
    model = CatBoostWrapper(frequency="daily")
    model.fit(make_train_df())  # y always >= 10, so _target_min >= 0
    assert model._target_min >= 0

    model._fitted_model.predict = lambda X: pd.Series([-5.0, 3.0, -1.0, 0.0, 10.0]).to_numpy()
    forecast = model.predict(horizon=5, future_exog=make_future_df(n=5))
    assert forecast["yhat"].tolist() == [0.0, 3.0, 0.0, 0.0, 10.0]

def test_predict_does_not_clip_when_training_target_was_negative():
    """
    A target that legitimately goes negative during training (e.g.
    profit/margin data) must NOT be force-clipped -- that would silently
    corrupt a real, valid prediction range.
    """
    df = make_train_df(n=30)
    df["y"] = df["y"] - 1000  # shift entirely negative
    model = CatBoostWrapper(frequency="daily")
    model.fit(df)
    assert model._target_min < 0

    model._fitted_model.predict = lambda X: pd.Series([-5.0, 3.0, -1.0]).to_numpy()
    forecast = model.predict(horizon=3, future_exog=make_future_df(n=3))
    assert forecast["yhat"].tolist() == [-5.0, 3.0, -1.0]  # untouched

def test_fit_predict_chaining_end_to_end():
    forecast = (
        CatBoostWrapper(frequency="daily")
        .fit(make_train_df())
        .predict(horizon=3, future_exog=make_future_df(n=3))
    )
    assert len(forecast) == 3

# ---------- save() / load() ----------

def test_save_raises_if_called_before_fit(tmp_path):
    model = CatBoostWrapper(frequency="daily")
    with pytest.raises(ModelPredictionError):
        model.save(str(tmp_path / "model.json"))

def test_save_and_load_roundtrip_preserves_prediction(tmp_path):
    save_path = str(tmp_path / "catboost_model.json")

    original = CatBoostWrapper(frequency="daily")
    original.fit(make_train_df())
    original.save(save_path)

    reloaded = CatBoostWrapper.load(save_path)
    assert reloaded.is_fitted is True
    assert reloaded.frequency == "daily"
    assert reloaded._feature_columns == original._feature_columns

    future = make_future_df()
    original_forecast = original.predict(horizon=5, future_exog=future)
    reloaded_forecast = reloaded.predict(horizon=5, future_exog=future)

    pd.testing.assert_frame_equal(
        original_forecast.reset_index(drop=True),
        reloaded_forecast.reset_index(drop=True),
    )

def test_save_and_load_preserves_categorical_feature_names(tmp_path):
    save_path = str(tmp_path / "catboost_model.json")

    original = CatBoostWrapper(frequency="daily")
    original.fit(make_train_df())
    original.save(save_path)

    reloaded = CatBoostWrapper.load(save_path)
    assert reloaded._categorical_feature_names == []

def test_save_and_load_preserves_target_min_for_clipping(tmp_path):
    """
    Regression test: target_min must survive the save/load roundtrip,
    or a reloaded model would silently lose its clipping behavior (fall
    back to _target_min=None, which skips clipping regardless of what
    the original model would have done).
    """
    save_path = str(tmp_path / "catboost_model.json")

    original = CatBoostWrapper(frequency="daily")
    original.fit(make_train_df())  # y always >= 10
    original.save(save_path)

    reloaded = CatBoostWrapper.load(save_path)
    assert reloaded._target_min == original._target_min == 10.0

def test_load_of_pre_target_min_envelope_falls_back_to_none(tmp_path):
    """
    Backward compatibility: an envelope saved before target_min existed
    (no such key) must load cleanly with _target_min=None (no clipping),
    not raise a KeyError.
    """
    import json

    save_path = str(tmp_path / "catboost_model.json")
    original = CatBoostWrapper(frequency="daily")
    original.fit(make_train_df())
    original.save(save_path)

    with open(save_path) as f:
        envelope = json.load(f)
    del envelope["target_min"]
    with open(save_path, "w") as f:
        json.dump(envelope, f)

    reloaded = CatBoostWrapper.load(save_path)
    assert reloaded._target_min is None

def test_loaded_model_is_a_fresh_instance_not_shared_state(tmp_path):
    save_path = str(tmp_path / "catboost_model.json")
    original = CatBoostWrapper(frequency="daily")
    original.fit(make_train_df())
    original.save(save_path)

    reloaded = CatBoostWrapper.load(save_path)
    assert reloaded is not original
    assert reloaded._fitted_model is not original._fitted_model

def test_save_and_load_preserves_wrapper_level_config(tmp_path):
    """
    min_rows_required and random_seed are wrapper-level config, not
    CatBoost hyperparameters -- confirm they survive the save/load
    roundtrip too, not just the underlying CatBoost model.
    """
    save_path = str(tmp_path / "model.json")
    original = CatBoostWrapper(frequency="weekly", min_rows_required=12, random_seed=99)
    original.fit(make_train_df(n=20))
    original.save(save_path)

    reloaded = CatBoostWrapper.load(save_path)
    assert reloaded.min_rows_required == 12
    assert reloaded.random_seed == 99
    assert reloaded.frequency == "weekly"
