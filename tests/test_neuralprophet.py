"""
NeuralProphet wrapper tests.

These require the real NeuralProphet (torch / PyTorch Lightning), so the
whole module is skipped where it isn't installed. The adapter is covered
separately, and without this dependency, in
tests/test_data_adapter_neuralprophet.py.

Training is deliberately kept tiny (few epochs, short series) so the suite
stays tolerable: NeuralProphet is by far the slowest model registered.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("neuralprophet")  # local-only: skips where NP isn't installed

from horizon_forecaster.data_adapter.neuralprophet_adapter import NeuralProphetAdapter
from horizon_forecaster.models.neuralprophet_wrapper import NeuralProphetWrapper
from horizon_forecaster.exceptions import (
    InsufficientDataError,
    ModelPredictionError,
)

FAST_PARAMS = {
    "n_lags": 0,
    "epochs": 5,  # tiny: these tests check contract, not accuracy
    "learning_rate": 0.1,
    "yearly_seasonality": False,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "accelerator": "cpu",
    # Load-bearing, and it mirrors what default_params.yaml sets for BOTH
    # frequencies (asserted by test_data_adapter_neuralprophet.py).
    #
    # Metrics-on makes NeuralProphet attach a PyTorch Lightning logger, and
    # that logger is a landmine whose blast radius depends on what else is
    # installed in the environment. Measured by running this exact suite on
    # two Colab images:
    #
    #   Python 3.12  omegaconf >= 2.2 rejects ConfigTrain's
    #                loss_func: Union[str, _Loss, Callable] inside
    #                save_hparams_to_yaml()      -> ConfigValueError
    #   Python 3.13  the TensorBoard logger probes tf.io.gfile, importing
    #                TensorFlow -> jax -> np.dtypes.StringDType, which does
    #                not exist under the numpy 1.x this project pins
    #
    # Same trigger, different explosion, the same ten tests down in both.
    # Neither reproduces on a machine lacking omegaconf >= 2.2 or
    # TensorFlow, which is why this went unnoticed locally.
    #
    # The forecaster ALWAYS merges the YAML's collect_metrics=false on top
    # of any model_params, so metrics-on is a configuration Horizon never
    # ships. Driving the wrapper with it enabled was therefore testing a
    # state that cannot occur in production, while importing a whole class
    # of environmental fragility. (The integration test at the bottom of
    # this file passed throughout for exactly this reason -- it goes via
    # the forecaster, so it received the merge.)
    "collect_metrics": False,
}


def make_adapted(n=80, with_regressors=True, test_size=10, seed=0, negative=False):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    y = 100 + np.arange(n) + rng.normal(0, 3, n)
    if negative:
        y = y - 500  # force a signed target, to exercise the clipping decision
    spend = np.abs(rng.normal(200, 30, n)) + 1.0
    raw = pd.DataFrame({"date": dates, "orders": y, "spend": spend})

    cols = ["spend"] if with_regressors else []
    adapter = NeuralProphetAdapter("date", "orders", cols, "daily")
    train, test = raw.iloc[: n - test_size], raw.iloc[n - test_size :]
    adapted_train = adapter.transform_train(train)
    adapted_future = adapter.transform_future(
        pd.Series(test["date"].to_numpy()), test[cols] if cols else None
    )
    return adapted_train, adapted_future, test["orders"].to_numpy()


@pytest.fixture(scope="module")
def fitted():
    train, future, y_test = make_adapted()
    model = NeuralProphetWrapper("daily", dict(FAST_PARAMS), min_rows_required=30, random_seed=42)
    model.fit(train)
    return model, train, future, y_test


# ---------- fit / predict basics ----------


def test_fit_sets_is_fitted_and_records_regressors(fitted):
    model, *_ = fitted
    assert model.is_fitted
    assert model._regressor_columns == ["spend"]


def test_predict_returns_ds_and_yhat_only_without_bounds(fitted):
    model, _, future, _ = fitted
    out = model.predict(horizon=len(future), future_exog=future)
    # v1 deliberately emits no uncertainty bounds (see BaseModel contract)
    assert list(out.columns) == ["ds", "yhat"]
    assert len(out) == len(future)
    assert out["yhat"].notna().all()


def test_predict_needs_no_training_history():
    """
    With n_lags=0 the future frame is built directly from future_exog, so
    predict() must work without the wrapper retaining any training data.
    """
    train, future, _ = make_adapted()
    model = NeuralProphetWrapper("daily", dict(FAST_PARAMS), min_rows_required=30)
    model.fit(train)
    assert not hasattr(model, "_train_df")
    assert len(model.predict(horizon=len(future), future_exog=future)) == len(future)


# ---------- error contract ----------


def test_predict_before_fit_raises():
    model = NeuralProphetWrapper("daily", dict(FAST_PARAMS), min_rows_required=30)
    with pytest.raises(ModelPredictionError):
        model.predict(5, future_exog=None)


def test_insufficient_data_raises():
    train, _, _ = make_adapted()
    model = NeuralProphetWrapper("daily", dict(FAST_PARAMS), min_rows_required=30)
    with pytest.raises(InsufficientDataError):
        model.fit(train.iloc[:10])


def test_predict_with_none_future_exog_raises(fitted):
    model, *_ = fitted
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=5, future_exog=None)


def test_predict_horizon_mismatch_raises(fitted):
    model, _, future, _ = fitted
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=len(future) + 3, future_exog=future)


def test_predict_missing_regressor_raises(fitted):
    model, _, future, _ = fitted
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=len(future), future_exog=future.drop(columns=["spend"]))


# ---------- clipping + determinism ----------


def test_clips_negatives_when_training_target_was_non_negative(fitted):
    model, _, future, _ = fitted
    out = model.predict(horizon=len(future), future_exog=future)
    assert (out["yhat"] >= 0).all()


def test_does_not_clip_when_training_target_was_negative():
    train, future, _ = make_adapted(negative=True)
    model = NeuralProphetWrapper("daily", dict(FAST_PARAMS), min_rows_required=30)
    model.fit(train)
    assert model._target_min < 0  # clipping decision is data-driven, so it must be off


def test_same_seed_gives_identical_predictions():
    """NeuralProphet is stochastic; without seeding, repeat runs drift."""
    train, future, _ = make_adapted()
    preds = []
    for _ in range(2):
        model = NeuralProphetWrapper(
            "daily", dict(FAST_PARAMS), min_rows_required=30, random_seed=7
        )
        model.fit(train)
        preds.append(model.predict(horizon=len(future), future_exog=future)["yhat"].to_numpy())
    np.testing.assert_allclose(preds[0], preds[1], rtol=1e-5)


# ---------- persistence ----------


def test_save_load_roundtrip_preserves_prediction(tmp_path, fitted):
    model, _, future, _ = fitted
    path = str(tmp_path / "np_model")
    model.save(path)
    loaded = NeuralProphetWrapper.load(path)
    before = model.predict(len(future), future)["yhat"].to_numpy()
    after = loaded.predict(len(future), future)["yhat"].to_numpy()
    np.testing.assert_allclose(before, after, rtol=1e-5)
    assert loaded._regressor_columns == model._regressor_columns


# ---------- integration through the orchestrator ----------


def test_neuralprophet_is_registered_and_flows_through_forecaster():
    from horizon_forecaster.forecaster import HorizonForecaster, MODEL_REGISTRY

    assert "neuralprophet" in MODEL_REGISTRY

    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=n, freq="D"),
            "orders": np.clip(200 + 1.5 * np.arange(n) + rng.normal(0, 10, n), 1, None),
        }
    )
    f = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["neuralprophet"],
        mape_threshold=100,
        model_params={"neuralprophet": dict(FAST_PARAMS)},  # skip the grid: keep it fast
    )
    results = f.evaluate()
    assert "neuralprophet" in results
    assert "yhat_lower" not in results["neuralprophet"].test_predictions.columns

    forecasts = f.forecast(horizon=7)
    assert len(forecasts["neuralprophet"].forecast) == 7


# ---------- NeuralProphet's import must not reroute the caller's warnings ----------


def test_first_import_inside_a_warning_recorder_still_records_horizon_warnings():
    """
    neuralprophet/__init__.py calls logging.captureWarnings(True). Before the
    fix, a catch_warnings(record=True) (or pytest.warns) block that was open
    when NeuralProphet was first imported stopped recording, and every later
    warning in the process printed as a log line. Run in a fresh interpreter:
    NeuralProphet is already imported in this one.
    """
    import subprocess
    import sys

    script = (
        "import warnings, io, contextlib\n"
        "from horizon_forecaster.models.neuralprophet_wrapper import _import_neuralprophet\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    _import_neuralprophet()\n"
        "    warnings.warn('HORIZON-TEST')\n"
        "assert any('HORIZON-TEST' in str(w.message) for w in caught), 'not recorded'\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stderr(buf):\n"
        "    warnings.warn('AFTER')\n"
        "assert 'UserWarning: AFTER' in buf.getvalue(), buf.getvalue()\n"
        "assert 'py.warnings' not in buf.getvalue(), buf.getvalue()\n"
    )
    result = subprocess.run(  # nosec B603 -- fixed script, current interpreter
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
