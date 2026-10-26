import matplotlib

matplotlib.use("Agg")  # headless: no display needed, and never blocks
import matplotlib.pyplot as plt
from pathlib import Path
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
from horizon_forecaster.exceptions import InvalidConfigError


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _ds(n=3):
    return pd.date_range("2023-01-01", periods=n, freq="D")


def _only(axes):
    """
    Unwrap the single-chart case.

    plot_evaluation/plot_forecast return {key: Axes} -- one chart per
    combination rule -- and most tests here build exactly one rule, so
    this asserts that shape and hands back the Axes. The assertion is the
    point: a test that silently indexed [0] would still pass if a rule
    were dropped, which is the defect this shape exists to prevent.
    """
    assert len(axes) == 1, f"expected exactly one chart, got {sorted(axes)}"
    return next(iter(axes.values()))


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
        method="inverse_mape",
        rule_applied="inverse_mape" if combined else "none",
        members=["prophet", "catboost"] if combined else [],
        weights={"prophet": 0.6, "catboost": 0.4} if combined else None,
        rejected_models={},
        mape=mape,
    )


def make_trimmed_ensemble(*, n=3, degraded=False):
    """A weightless (trimmed) result -- the DEFAULT rule, which the
    inverse_mape fixture above never exercises."""
    return EnsembleResult(
        combined=pd.DataFrame({"ds": _ds(n), "yhat": [12, 20, 32][:n]}),
        method="trimmed",
        rule_applied="mean" if degraded else "trimmed",
        members=["prophet", "catboost"] if degraded else ["prophet", "catboost", "sarimax"],
        degraded=degraded,
        weights=None,
        mape=MapeResult(value=8.0, excluded_zero_actual_count=0, total_count=n),
    )


# ---------- plot_evaluation ----------


def test_plot_evaluation_draws_actual_plus_one_line_per_model():
    results = {"prophet": make_eval_result(5.0), "catboost": make_eval_result(12.0)}
    ax = _only(plot_evaluation(results))
    # 1 actual + 2 model lines
    assert len(ax.get_lines()) == 3
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "Actual" in labels
    assert any("prophet" in lbl and "5.00%" in lbl for lbl in labels)


def test_plot_evaluation_overlays_ensemble_with_its_mape():
    results = {"prophet": make_eval_result(5.0)}
    ax = _only(plot_evaluation(results, ensemble=make_ensemble()))
    labels = [ln.get_label() for ln in ax.get_lines()]
    # The line is named for the rule that actually ran, not the one requested.
    assert any(lbl.startswith("ensemble (") and "8.00%" in lbl for lbl in labels)


def test_plot_evaluation_skips_ensemble_when_not_formed():
    results = {"prophet": make_eval_result(5.0)}
    ax = _only(plot_evaluation(results, ensemble=make_ensemble(combined=False)))
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert not any("ensemble" in lbl for lbl in labels)


def test_plot_evaluation_shades_bounds_only_when_present():
    with_b = {"prophet": make_eval_result(5.0, with_bounds=True)}
    without_b = {"catboost": make_eval_result(12.0, with_bounds=False)}
    assert len(_only(plot_evaluation(with_b)).collections) == 1
    assert len(_only(plot_evaluation(without_b)).collections) == 0


def test_plot_evaluation_empty_raises():
    with pytest.raises(ValueError):
        plot_evaluation({})


def _has_no_ensemble_note(ax):
    return any("No ensemble formed" in t.get_text() for t in ax.texts)


def test_plot_evaluation_flags_when_no_ensemble_formed():
    results = {"prophet": make_eval_result(5.0)}
    ax = _only(plot_evaluation(results, ensemble=make_ensemble(combined=False)))
    assert _has_no_ensemble_note(ax)
    # individual model line is still drawn (diagnosis, not suppressed)
    assert any("prophet" in ln.get_label() for ln in ax.get_lines())


def test_plot_evaluation_no_note_when_ensemble_formed_or_absent():
    results = {"prophet": make_eval_result(5.0)}
    assert not _has_no_ensemble_note(_only(plot_evaluation(results, ensemble=make_ensemble())))
    assert not _has_no_ensemble_note(_only(plot_evaluation(results)))


# ---------- plot_forecast ----------


def test_plot_forecast_one_line_per_model_no_history():
    results = {"prophet": make_forecast_result(), "catboost": make_forecast_result()}
    ax = _only(plot_forecast(results))
    assert len(ax.get_lines()) == 2


def test_plot_forecast_includes_history_and_ensemble():
    results = {"prophet": make_forecast_result()}
    history = pd.DataFrame({"date": _ds(5), "orders": [1, 2, 3, 4, 5]})
    ax = _only(
        plot_forecast(
            results,
            history=history,
            date_column="date",
            target_column="orders",
            ensemble=make_ensemble(),
        )
    )
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "History" in labels
    # The forecast line names the rule that ran, matching the evaluation charts.
    assert "ensemble (inverse_mape) forecast" in labels
    # history + 1 model + ensemble
    assert len(ax.get_lines()) == 3


def test_plot_forecast_empty_raises():
    with pytest.raises(ValueError):
        plot_forecast({})


def test_plot_forecast_flags_when_no_ensemble_formed():
    results = {"prophet": make_forecast_result()}
    ax = _only(plot_forecast(results, ensemble=make_ensemble(combined=False)))
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


def test_plot_model_comparison_tolerates_a_nan_mape():
    # A test window of nothing but zero actuals gives a NaN MAPE. Passed to
    # set_ylim it would raise -- and take save_evaluation_plots down with it.
    results = {"prophet": make_eval_result(float("nan")), "catboost": make_eval_result(12.0)}
    ax = plot_model_comparison(results)
    assert len(ax.patches) == 2
    assert ax.get_ylim() == pytest.approx((0, 12.0 * 1.15))

    ax = plot_model_comparison({"prophet": make_eval_result(float("nan"))})
    assert ax.get_ylim() == (0, 1.0)


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
    assert set(figs) == {"prophet", "catboost", "ensemble_inverse_mape"}


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
    assert set(figs) == {"prophet", "catboost", "ensemble_inverse_mape"}


def test_plot_forecast_per_model_empty_raises():
    with pytest.raises(ValueError):
        plot_forecast_per_model({})


# ---------- saving to disk ----------


def test_save_evaluation_plots_writes_overlay_comparison_and_per_model(tmp_path):
    results = {"prophet": make_eval_result(5.0), "catboost": make_eval_result(12.0)}
    written = save_evaluation_plots(results, make_ensemble(), output_dir=str(tmp_path))
    assert set(written) == {
        "overlay_inverse_mape",
        "mape_comparison",
        "prophet",
        "catboost",
        "ensemble_inverse_mape",
    }
    for path in written.values():
        assert path.endswith(".png")
        assert (tmp_path / path.split("/")[-1]).exists()


def test_save_forecast_plots_writes_overlay_and_per_model(tmp_path):
    results = {"prophet": make_forecast_result()}
    written = save_forecast_plots(results, ensemble=make_ensemble(), output_dir=str(tmp_path))
    assert set(written) == {"overlay_inverse_mape", "prophet", "ensemble_inverse_mape"}
    for path in written.values():
        assert (tmp_path / path.split("/")[-1]).exists()


def test_save_forecast_plots_creates_nested_dir(tmp_path):
    target = tmp_path / "plots" / "run1"
    save_forecast_plots({"prophet": make_forecast_result()}, output_dir=str(target))
    # No ensemble was given, so the single overlay is keyed "models".
    assert (target / "forecast_overlay_models.png").exists()


def test_save_evaluation_plots_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        save_evaluation_plots({}, output_dir=str(tmp_path))


# ---------- shared behavior ----------


def test_plot_functions_draw_on_supplied_ax():
    fig, ax = plt.subplots()
    returned = plot_evaluation({"prophet": make_eval_result(5.0)}, ax=ax)
    assert _only(returned) is ax


# ---------- one chart per combination rule ----------


def _three_rules():
    """The shape a real ensemble_method=None run produces: all three rules."""
    return {
        "trimmed": make_trimmed_ensemble(),
        "inverse_mape": make_ensemble(),
        "equal": EnsembleResult(
            combined=pd.DataFrame({"ds": _ds(3), "yhat": [12, 20, 32]}),
            method="equal",
            rule_applied="equal",
            members=["prophet", "catboost"],
            weights={"prophet": 0.5, "catboost": 0.5},
            mape=MapeResult(value=9.0, excluded_zero_actual_count=0, total_count=3),
        ),
    }


def test_plot_evaluation_draws_one_chart_per_rule_against_every_model():
    """
    What this shape exists to prevent: with three rules built, charts that
    showed only the first would hide two combinations the caller had paid
    to compute. Each rule gets its own chart carrying the SAME models and
    actuals, which is what makes the rules
    comparable against the truth rather than only against each other.
    """
    results = {"prophet": make_eval_result(5.0), "catboost": make_eval_result(12.0)}
    axes = plot_evaluation(results, _three_rules())

    assert set(axes) == {"trimmed", "inverse_mape", "equal"}
    for method, ax in axes.items():
        labels = [ln.get_label() for ln in ax.get_lines()]
        # actuals + both models + exactly one ensemble line, every time
        assert "Actual" in labels
        assert any("prophet" in lbl for lbl in labels)
        assert any("catboost" in lbl for lbl in labels)
        assert sum(lbl.startswith("ensemble (") for lbl in labels) == 1
        assert method in ax.get_title()


def test_plot_forecast_draws_one_chart_per_rule():
    forecasts = {"prophet": make_forecast_result(), "catboost": make_forecast_result()}
    axes = plot_forecast(forecasts, ensemble=_three_rules())

    assert set(axes) == {"trimmed", "inverse_mape", "equal"}
    for ax in axes.values():
        labels = [ln.get_label() for ln in ax.get_lines()]
        assert sum(lbl.startswith("ensemble (") for lbl in labels) == 1


def test_per_model_figures_cover_every_model_and_every_rule():
    """Five models and three rules give eight figures, not six."""
    results = {name: make_eval_result(5.0) for name in ("a", "b", "c", "d", "e")}
    figs = plot_evaluation_per_model(results, _three_rules())
    assert set(figs) == {
        "a",
        "b",
        "c",
        "d",
        "e",
        "ensemble_trimmed",
        "ensemble_inverse_mape",
        "ensemble_equal",
    }


def test_saved_files_carry_the_method_suffix_even_for_a_single_rule(tmp_path):
    """
    The suffix is unconditional for the same reason the CSV exporter
    suffixes final_ensemble_{method}.csv: a filename that changed shape
    with the number of rules requested could not be globbed downstream.
    """
    written = save_forecast_plots(
        {"prophet": make_forecast_result()},
        ensemble={"trimmed": make_trimmed_ensemble()},
        output_dir=str(tmp_path),
    )
    assert (tmp_path / "forecast_overlay_trimmed.png").exists()
    assert (tmp_path / "forecast_ensemble_trimmed.png").exists()
    assert all(Path(path).exists() for path in written.values())


def test_supplied_ax_is_refused_for_several_rules():
    """
    Drawing three rules onto one caller-supplied Axes would overlay them,
    and quietly drawing one would lose the other two -- so it is an error
    the caller can act on.
    """
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="several combination methods"):
        plot_evaluation({"prophet": make_eval_result(5.0)}, _three_rules(), ax=ax)


def test_no_ensemble_still_produces_one_chart():
    """The return shape never depends on whether an ensemble was asked for."""
    axes = plot_evaluation({"prophet": make_eval_result(5.0)})
    assert set(axes) == {"models"}


def test_bar_chart_labels_every_rule_and_rotates_them():
    """
    All three rules share one Axes here -- comparing them head-to-head is
    what a bar chart is for. Rotation is what keeps the long ensemble
    labels from colliding with their neighbours at that bar count.
    """
    results = {"prophet": make_eval_result(5.0), "catboost": make_eval_result(12.0)}
    ax = plot_model_comparison(results, _three_rules())

    assert len(ax.patches) == 5  # two models + three rules
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "ensemble (trimmed)" in labels
    assert "ensemble (inverse_mape)" in labels
    assert all(t.get_rotation() > 0 for t in ax.get_xticklabels())


def test_bar_chart_note_only_when_no_rule_formed_at_all():
    """
    A blanket "no ensemble formed" note would contradict a chart that
    still shows a real ensemble bar, so it appears only when EVERY
    requested rule came back empty.
    """
    results = {"prophet": make_eval_result(5.0)}
    mixed = {
        "trimmed": make_trimmed_ensemble(),
        "equal": make_ensemble(combined=False, with_mape=False),
    }
    assert not _has_no_ensemble_note(plot_model_comparison(results, mixed))

    all_empty = {
        "trimmed": make_ensemble(combined=False, with_mape=False),
        "equal": make_ensemble(combined=False, with_mape=False),
    }
    assert _has_no_ensemble_note(plot_model_comparison(results, all_empty))


# ---------- the dict shape the forecaster actually hands over ----------


def test_every_plot_entry_point_accepts_the_forecaster_dict():
    """
    HorizonForecaster.evaluate_ensemble / forecast_ensemble are
    {method: EnsembleResult} dicts. An entry point that did not normalise
    that would die with AttributeError on a perfectly ordinary run -- and
    through save_* so would exporters.export_forecast(save_plots=True), the
    documented one-call export. Every
    entry point must accept the dict.
    """
    results = {"prophet": make_eval_result(5.0), "catboost": make_eval_result(9.0)}
    forecasts = {"prophet": make_forecast_result(), "catboost": make_forecast_result()}
    as_dict = {"trimmed": make_trimmed_ensemble()}

    # Assert the ensemble actually reached the chart, not merely that the
    # call didn't raise: these functions always return an Axes, so
    # `is not None` would still pass if the dict were silently dropped.
    eval_ax = _only(plot_evaluation(results, as_dict))
    assert any("ensemble" in ln.get_label() for ln in eval_ax.get_lines())

    forecast_ax = _only(plot_forecast(forecasts, ensemble=as_dict))
    assert any("ensemble" in ln.get_label() for ln in forecast_ax.get_lines())

    # One bar per model, plus one for the ensemble.
    comparison_ax = plot_model_comparison(results, as_dict)
    assert len(comparison_ax.patches) == len(results) + 1

    assert "ensemble_trimmed" in plot_evaluation_per_model(results, as_dict)
    assert "ensemble_trimmed" in plot_forecast_per_model(forecasts, ensemble=as_dict)


def test_save_helpers_accept_the_forecaster_dict(tmp_path):
    results = {"prophet": make_eval_result(5.0)}
    forecasts = {"prophet": make_forecast_result()}
    as_dict = {"trimmed": make_trimmed_ensemble()}

    written = save_evaluation_plots(results, as_dict, output_dir=str(tmp_path))
    assert "ensemble_trimmed" in written
    written = save_forecast_plots(forecasts, ensemble=as_dict, output_dir=str(tmp_path))
    assert "ensemble_trimmed" in written
    assert all(Path(p).exists() for p in written.values())


def test_line_charts_label_the_rule_that_actually_ran():
    """A two-member plain mean must not be labelled 'trimmed'."""
    results = {"prophet": make_eval_result(5.0)}
    ax = _only(plot_evaluation(results, {"trimmed": make_trimmed_ensemble(degraded=True)}))
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert any("(mean)" in lbl and "degraded" in lbl for lbl in labels)


def test_weightless_ensemble_still_plots():
    """The default rule carries weights=None; nothing may assume a dict."""
    results = {"prophet": make_eval_result(5.0)}
    ax = _only(plot_evaluation(results, make_trimmed_ensemble()))
    assert any("ensemble" in ln.get_label() for ln in ax.get_lines())
    assert "ensemble_trimmed" in plot_evaluation_per_model(results, make_trimmed_ensemble())


def test_forecast_charts_label_the_rule_that_actually_ran():
    """
    Same guarantee as the evaluation charts: a degraded two-member mean is a
    materially weaker output than a five-member trimmed mean and must not be
    drawn under the same name. The returned dict KEY stays the plain
    "ensemble_<method>", keyed by the rule REQUESTED -- it is a stable
    handle and a filename -- so only the drawn label carries the rule that
    actually ran.
    """
    forecasts = {"prophet": make_forecast_result()}
    degraded = {"trimmed": make_trimmed_ensemble(degraded=True)}

    ax = _only(plot_forecast(forecasts, ensemble=degraded))
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert any("(mean)" in lbl and "degraded" in lbl for lbl in labels)

    figs = plot_forecast_per_model(forecasts, ensemble=degraded)
    assert "ensemble_trimmed" in figs
    fig_labels = [ln.get_label() for ln in figs["ensemble_trimmed"].axes[0].get_lines()]
    assert any("(mean)" in lbl and "degraded" in lbl for lbl in fig_labels)


def test_no_ensemble_note_is_shown_through_the_forecaster_dict():
    """
    The all-rejected run is when the "diagnosis only" warning matters most,
    and it arrives as a dict like every other real run -- so the note must
    survive normalisation, not just the bare-result path.
    """
    results = {"prophet": make_eval_result(5.0)}
    ax = _only(plot_evaluation(results, {"trimmed": make_ensemble(combined=False)}))
    assert _has_no_ensemble_note(ax)


# ---------- argument checks ----------


def test_plots_say_which_results_they_expect():
    forecast_results = {"prophet": make_forecast_result()}
    evaluation_results = {"prophet": make_eval_result(5.0)}
    with pytest.raises(InvalidConfigError, match=r"plot_forecast expects .*forecaster\.forecast"):
        plot_forecast(evaluation_results)
    with pytest.raises(InvalidConfigError, match=r"plot_evaluation expects .*forecaster\.evaluate"):
        plot_evaluation(forecast_results)
    with pytest.raises(InvalidConfigError, match="plot_model_comparison expects"):
        plot_model_comparison(forecast_results)
    with pytest.raises(InvalidConfigError, match="got list"):
        plot_forecast_per_model([make_forecast_result()])


def test_plot_forecast_names_missing_history_columns():
    history = pd.DataFrame({"date": _ds(3), "orders": [1, 2, 3]})
    with pytest.raises(InvalidConfigError, match=r"history has no column\(s\) \['ds', 'y'\]"):
        plot_forecast({"prophet": make_forecast_result()}, history=history)


def test_empty_results_raise_the_librarys_error_which_is_still_a_value_error():
    with pytest.raises(InvalidConfigError, match="nothing to plot"):
        plot_evaluation({})
    assert issubclass(InvalidConfigError, ValueError)


def test_plot_forecast_reads_text_dates_in_history():
    # Text dates straight from a CSV would otherwise crash with a raw matplotlib TypeError.
    history = pd.DataFrame(
        {"date": ["2022-12-29", "2022-12-30", "2022-12-31"], "orders": [1, 2, 3]}
    )
    axes = plot_forecast(
        {"prophet": make_forecast_result()},
        history=history,
        date_column="date",
        target_column="orders",
    )
    assert len(axes) == 1


# ---------- WAPE and the seasonal-naive baseline ----------


def _scored(mape_value, wape_value):
    result = make_eval_result(mape_value)
    result.wape = wape_value
    return result


def test_line_labels_give_wape_next_to_mape():
    ax = _only(plot_evaluation({"prophet": _scored(5.0, 7.5)}))
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "prophet predicted (MAPE 5.00%, WAPE 7.50%)" in labels


def test_comparison_chart_draws_the_chosen_metric_and_the_baseline():
    results = {"prophet": _scored(5.0, 30.0), "catboost": _scored(10.0, 20.0)}
    baseline = _scored(8.0, 25.0)
    ax = plot_model_comparison(results, metric="wape", baseline=baseline)
    heights = [round(p.get_height(), 2) for p in ax.patches]
    assert heights == [30.0, 20.0, 25.0]
    assert [t.get_text() for t in ax.get_xticklabels()][-1] == "seasonal naive"
    assert ax.get_ylabel() == "WAPE (%)"
    assert "WAPE" in ax.get_title()


def test_comparison_chart_rejects_an_unknown_metric():
    with pytest.raises(InvalidConfigError, match="metric must be one of"):
        plot_model_comparison({"prophet": _scored(5.0, 7.5)}, metric="rmse")


def test_save_evaluation_plots_names_the_comparison_by_metric(tmp_path):
    written = save_evaluation_plots(
        {"prophet": _scored(5.0, 7.5)}, output_dir=str(tmp_path), metric="wape"
    )
    assert Path(written["wape_comparison"]).name == "evaluation_wape_comparison.png"
    assert "mape_comparison" not in written


def test_comparison_metric_accepts_any_case(tmp_path):
    ax = plot_model_comparison({"prophet": _scored(5.0, 7.5)}, metric="WAPE")
    assert ax.get_ylabel() == "WAPE (%)"
    written = save_evaluation_plots(
        {"prophet": _scored(5.0, 7.5)}, output_dir=str(tmp_path), metric="Wape"
    )
    assert Path(written["wape_comparison"]).name == "evaluation_wape_comparison.png"


def test_overlay_shades_only_the_ensembles_band_when_every_model_has_one():
    results = {
        "prophet": make_eval_result(5.0, with_bounds=True),
        "sarimax": make_eval_result(6.0, with_bounds=True),
    }
    ensemble = make_ensemble()
    ensemble.combined["yhat_lower"] = [10, 18, 30]
    ensemble.combined["yhat_upper"] = [14, 22, 34]
    ax = _only(plot_evaluation(results, ensemble=ensemble))
    assert len(ax.collections) == 1  # five overlapping bands would hide the lines
    assert ax.collections[0].get_label() == "Ensemble 95% band"
