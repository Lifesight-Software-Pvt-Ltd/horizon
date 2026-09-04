import numpy as np
import pandas as pd
import pytest

from horizon_forecaster.data_adapter.neuralprophet_adapter import NeuralProphetAdapter
from horizon_forecaster.exceptions import InvalidConfigError


def make_raw(n=40, freq="D"):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=n, freq=freq),
            "orders": np.arange(n) + 50.0,
            "spend": np.abs(rng.normal(100, 20, n)) + 1.0,
            "promo": rng.integers(0, 2, n),
        }
    )


def test_transform_train_renames_to_ds_y_and_keeps_regressors():
    adapter = NeuralProphetAdapter("date", "orders", ["spend", "promo"], "daily")
    out = adapter.transform_train(make_raw())
    assert list(out.columns) == ["ds", "y", "spend", "promo"]
    assert len(out) == 40


def test_transform_train_passes_regressors_through_raw():
    """
    Unlike SARIMAXAdapter (which log1p's, standardises and clamps), this
    adapter must NOT transform anything -- NeuralProphet normalises
    internally, so touching the data here would double-normalise it.
    """
    raw = make_raw()
    adapter = NeuralProphetAdapter("date", "orders", ["spend", "promo"], "daily")
    out = adapter.transform_train(raw)
    np.testing.assert_allclose(out["spend"].to_numpy(), raw["spend"].to_numpy())
    np.testing.assert_allclose(out["y"].to_numpy(), raw["orders"].to_numpy())


def test_transform_train_sorts_by_date():
    raw = make_raw().sample(frac=1, random_state=1)  # shuffle
    adapter = NeuralProphetAdapter("date", "orders", [], "daily")
    out = adapter.transform_train(raw)
    assert out["ds"].is_monotonic_increasing


def test_transform_train_works_with_no_regressors():
    adapter = NeuralProphetAdapter("date", "orders", [], "daily")
    out = adapter.transform_train(make_raw())
    assert list(out.columns) == ["ds", "y"]


def test_transform_train_raises_on_missing_column():
    adapter = NeuralProphetAdapter("date", "orders", ["spend"], "daily")
    with pytest.raises(InvalidConfigError):
        adapter.transform_train(make_raw().drop(columns=["spend"]))


def test_transform_future_shape_and_columns():
    adapter = NeuralProphetAdapter("date", "orders", ["spend", "promo"], "daily")
    adapter.transform_train(make_raw())
    future_dates = pd.Series(pd.date_range("2022-02-10", periods=4, freq="D"))
    future_ind = pd.DataFrame({"spend": [10.0, 11.0, 12.0, 13.0], "promo": [0, 1, 0, 1]})
    out = adapter.transform_future(future_dates, future_ind)
    assert list(out.columns) == ["ds", "spend", "promo"]
    assert len(out) == 4


def test_transform_future_raises_when_regressors_declared_but_missing():
    adapter = NeuralProphetAdapter("date", "orders", ["spend"], "daily")
    with pytest.raises(InvalidConfigError):
        adapter.transform_future(pd.Series(pd.date_range("2022-02-10", periods=3, freq="D")), None)


def test_transform_future_is_stateless_and_needs_no_prior_train_call():
    """
    Deliberate contrast with SARIMAXAdapter/CatBoostAdapter, which raise
    if transform_future() is called first: this adapter carries no state.
    """
    adapter = NeuralProphetAdapter("date", "orders", [], "daily")
    out = adapter.transform_future(pd.Series(pd.date_range("2022-02-10", periods=3, freq="D")))
    assert list(out.columns) == ["ds"]
    assert len(out) == 3


def test_collect_metrics_is_disabled_for_every_frequency():
    """
    Correctness requirement, not tuning. With metrics enabled NeuralProphet
    attaches a Lightning logger, which makes trainer.fit() run
    _log_hyperparams() -> save_hparams_to_yaml() -> OmegaConf.save().
    OmegaConf rejects ConfigTrain's `loss_func: Union[str, _Loss, Callable]`
    and fit() dies before training.

    Asserted per frequency because the first attempt at this fix landed in
    the daily block only -- and weekly was the frequency actually failing.
    """
    import yaml
    from pathlib import Path

    import horizon_forecaster

    config_path = Path(horizon_forecaster.__file__).parent / "config" / "default_params.yaml"
    neuralprophet = yaml.safe_load(config_path.read_text())["neuralprophet"]

    assert set(neuralprophet) >= {"daily", "weekly"}
    for frequency, block in neuralprophet.items():
        assert (
            block["base_hyperparameters"]["collect_metrics"] is False
        ), f"{frequency} must disable collect_metrics, or NeuralProphet cannot fit"
