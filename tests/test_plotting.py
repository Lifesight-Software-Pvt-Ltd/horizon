import matplotlib

matplotlib.use("Agg")  # headless: no display needed, and never blocks
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from types import SimpleNamespace

from horizon_forecaster.plotting import (
    plot_evaluation,
    plot_evaluation_per_model,
    plot_forecast,
    plot_forecast_per_model,
    plot_model_comparison,
    save_evaluation_plots,
    save_forecast_plots,
)
from horizon_forecaster.evaluation.ensemble import EnsembleResult
from horizon_forecaster.evaluation.metrics import MapeResult


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _ds(n=3):
    return pd.date_range("2023-01-01", periods=n, freq="D")


def make_eval_result(mape_value, *, with_bounds=False, n=3):
    ds = _ds(n)
    cols = {"ds": ds, "y": [10, 20, 30][:n], "yhat": [11, 19, 31][:n]}
    if with_bounds:
        cols["yhat_lower"] = [9, 17, 28][:n]
        cols["yhat_upper"] = [13, 21, 34][:n]
    return SimpleNamespace(
        test_predictions=pd.DataFrame(cols),
        mape=SimpleNamespace(value=mape_value),
    )


def make_forecast_result(*, with_bounds=False, n=3):
    ds = _ds(n)
    cols = {"ds": ds, "yhat": [40, 42, 44][:n]}
    if with_bounds:
        cols["yhat_lower"] = [38, 40, 42][:n]
        cols["yhat_upper"] = [42, 44, 46][:n]
    return SimpleNamespace(forecast=pd.DataFrame(cols))


def make_ensemble(*, combined=True, with_mape=True, n=3):
    ds = _ds(n)
    combined_df = pd.DataFrame({"ds": ds, "yhat": [12, 20, 32][:n]}) if combined else None
    mape = MapeResult(value=8.0, excluded_zero_actual_count=0, total_count=n) if with_mape else None
    return EnsembleResult(
        combined=combined_df,
        weights={"prophet": 0.6, "catboost": 0.4} if combined else {},
        rejected_models={},
        weighting="inverse_mape",
        mape=mape,
    )


# ---------- plot_evaluation ----------

def test_plot_evaluation_draws_actual_plus_one_line_per_model():
    results = {"prophet": make_eval_result(5.0), "catboost": make_eval_result(12.0)}
    ax = plot_evaluation(results)
    # 1 actual + 2 model lines
    assert len(ax.get_lines()) == 3
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "Actual" in labels
    assert any("prophet" in lbl and "5.00%" in lbl for lbl in labels)


def test_plot_evaluation_overlays_ensemble_with_its_mape():
    results = {"prophet": make_eval_result(5.0)}
    ax = plot_evaluation(results, ensemble=make_ensemble())
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert any(lbl.startswith("ensemble predicted") and "8.00%" in lbl for lbl in labels)


def test_plot_evaluation_skips_ensemble_when_not_formed():
    results = {"prophet": make_eval_result(5.0)}
    ax = plot_evaluation(results, ensemble=make_ensemble(combined=False))
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert not any("ensemble" in lbl for lbl in labels)


def test_plot_evaluation_shades_bounds_only_when_present():
    with_b = {"prophet": make_eval_result(5.0, with_bounds=True)}
    without_b = {"catboost": make_eval_result(12.0, with_bounds=False)}
    assert len(plot_evaluation(with_b).collections) == 1
    assert len(plot_evaluation(without_b).collections) == 0


def test_plot_evaluation_empty_raises():
    with pytest.raises(ValueError):
        plot_evaluation({})


def _has_no_ensemble_note(ax):
    return any("No ensemble formed" in t.get_text() for t in ax.texts)


def test_plot_evaluation_flags_when_no_ensemble_formed():
    results = {"prophet": make_eval_result(5.0)}
    ax = plot_evaluation(results, ensemble=make_ensemble(combined=False))
    assert _has_no_ensemble_note(ax)
    # individual model line is still drawn (diagnosis, not suppressed)
    assert any("prophet" in ln.get_label() for ln in ax.get_lines())


def test_plot_evaluation_no_note_when_ensemble_formed_or_absent():
    results = {"prophet": make_eval_result(5.0)}
    assert not _has_no_ensemble_note(plot_evaluation(results, ensemble=make_ensemble()))
    assert not _has_no_ensemble_note(plot_evaluation(results))


# ---------- plot_forecast ----------

def test_plot_forecast_one_line_per_model_no_history():
    results = {"prophet": make_forecast_result(), "catboost": make_forecast_result()}
    ax = plot_forecast(results)
    assert len(ax.get_lines()) == 2


def test_plot_forecast_includes_history_and_ensemble():
    results = {"prophet": make_forecast_result()}
    history = pd.DataFrame({"date": _ds(5), "orders": [1, 2, 3, 4, 5]})
    ax = plot_forecast(
        results, history=history, date_column="date", target_column="orders",
        ensemble=make_ensemble(),
    )
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "History" in labels
    assert "ensemble forecast" in labels
    # history + 1 model + ensemble
    assert len(ax.get_lines()) == 3


def test_plot_forecast_empty_raises():
    with pytest.raises(ValueError):
        plot_forecast({})


def test_plot_forecast_flags_when_no_ensemble_formed():
    results = {"prophet": make_forecast_result()}
    ax = plot_forecast(results, ensemble=make_ensemble(combined=False))
    assert _has_no_ensemble_note(ax)


# ---------- plot_model_comparison ----------

def test_plot_model_comparison_one_bar_per_model():
    results = {"prophet": make_eval_result(5.0), "catboost": make_eval_result(12.0)}
    ax = plot_model_comparison(results)
    assert len(ax.patches) == 2


def test_plot_model_comparison_adds_ensemble_bar_when_scored():
    results = {"prophet": make_eval_result(5.0), "catboost": make_eval_result(12.0)}
    ax = plot_model_comparison(results, ensemble=make_ensemble())
    assert len(ax.patches) == 3
    assert "ensemble" in [t.get_text() for t in ax.get_xticklabels()]


def test_plot_model_comparison_no_ensemble_bar_without_mape():
    results = {"prophet": make_eval_result(5.0)}
    ax = plot_model_comparison(results, ensemble=make_ensemble(with_mape=False))
    assert len(ax.patches) == 1


def test_plot_model_comparison_flags_when_no_ensemble_formed():
    results = {"prophet": make_eval_result(5.0)}
    # all-rejected: no combined series and no ensemble MAPE
    ax = plot_model_comparison(results, ensemble=make_ensemble(combined=False, with_mape=False))
    assert _has_no_ensemble_note(ax)
    assert len(ax.patches) == 1  # only the individual model bar, no ensemble bar


def test_plot_model_comparison_empty_raises():
    with pytest.raises(ValueError):
        plot_model_comparison({})


# ---------- dedicated per-model figures ----------

def test_plot_evaluation_per_model_one_figure_each():
    results = {"prophet": make_eval_result(5.0), "catboost": make_eval_result(12.0)}
    figs = plot_evaluation_per_model(results)
    assert set(figs) == {"prophet", "catboost"}  # one full-size figure per model
    assert all(isinstance(f, plt.Figure) for f in figs.values())


def test_plot_evaluation_per_model_adds_ensemble_figure_when_formed():
    results = {"prophet": make_eval_result(5.0), "catboost": make_eval_result(12.0)}
    figs = plot_evaluation_per_model(results, ensemble=make_ensemble())
    assert set(figs) == {"prophet", "catboost", "ensemble"}


def test_plot_evaluation_per_model_flags_and_omits_ensemble_when_not_formed():
    results = {"prophet": make_eval_result(5.0)}
    figs = plot_evaluation_per_model(results, ensemble=make_ensemble(combined=False))
    assert set(figs) == {"prophet"}  # no ensemble figure
    ax = figs["prophet"].axes[0]
    assert any("No ensemble formed" in t.get_text() for t in ax.texts)


def test_plot_evaluation_per_model_empty_raises():
    with pytest.raises(ValueError):
        plot_evaluation_per_model({})


def test_plot_forecast_per_model_one_figure_each_plus_ensemble():
    results = {"prophet": make_forecast_result(), "catboost": make_forecast_result()}
    figs = plot_forecast_per_model(results, ensemble=make_ensemble())
    assert set(figs) == {"prophet", "catboost", "ensemble"}


def test_plot_forecast_per_model_empty_raises():
    with pytest.raises(ValueError):
        plot_forecast_per_model({})


# ---------- saving to disk ----------

def test_save_evaluation_plots_writes_overlay_comparison_and_per_model(tmp_path):
    results = {"prophet": make_eval_result(5.0), "catboost": make_eval_result(12.0)}
    written = save_evaluation_plots(results, make_ensemble(), output_dir=str(tmp_path))
    assert set(written) == {"overlay", "mape_comparison", "prophet", "catboost", "ensemble"}
    for path in written.values():
        assert path.endswith(".png")
        assert (tmp_path / path.split("/")[-1]).exists()


def test_save_forecast_plots_writes_overlay_and_per_model(tmp_path):
    results = {"prophet": make_forecast_result()}
    written = save_forecast_plots(results, ensemble=make_ensemble(), output_dir=str(tmp_path))
    assert set(written) == {"overlay", "prophet", "ensemble"}
    for path in written.values():
        assert (tmp_path / path.split("/")[-1]).exists()


def test_save_forecast_plots_creates_nested_dir(tmp_path):
    target = tmp_path / "plots" / "run1"
    save_forecast_plots({"prophet": make_forecast_result()}, output_dir=str(target))
    assert (target / "forecast_overlay.png").exists()


def test_save_evaluation_plots_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        save_evaluation_plots({}, output_dir=str(tmp_path))


# ---------- shared behavior ----------

def test_plot_functions_draw_on_supplied_ax():
    fig, ax = plt.subplots()
    returned = plot_evaluation({"prophet": make_eval_result(5.0)}, ax=ax)
    assert returned is ax
