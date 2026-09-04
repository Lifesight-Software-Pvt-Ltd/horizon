"""
Plotting helpers for horizon_forecaster.

Turns the outputs of HorizonForecaster.evaluate()/forecast() into
matplotlib charts. Two families are provided at each stage:

- Overlay charts (plot_evaluation / plot_forecast) put every model on a
  single Axes -- compact, good for a quick side-by-side, but they get
  crowded as models are added or when models agree and their lines sit
  on top of each other.
- Dedicated per-model figures (plot_evaluation_per_model /
  plot_forecast_per_model) give each model (and the ensemble) its own
  full-size figure, so every series stays readable no matter how many
  models there are. These return a {label: Figure} dict rather than a
  single Axes.

plot_model_comparison sits alongside both as a one-number-per-model MAPE
bar chart. Like evaluation/metrics.py and evaluation/ensemble.py,
this is a pure, standalone module: it never imports forecaster.py or any
model/adapter class, so importing it (e.g. just to draw a chart) does not
pull in the heavy Prophet/CatBoost stack. It reaches into the result
objects only through their public attributes (test_predictions, forecast,
mape.value, and the EnsembleResult fields), which is why the type hints
below are TYPE_CHECKING-only -- at runtime the inputs are duck-typed.

Design choices, all deliberate:

- Every function DRAWS onto an Axes and RETURNS it; none of them calls
  plt.show(). Showing/saving is the caller's decision (a notebook calls
  plt.show(); a script or a test saves or inspects the figure). This
  keeps the module non-blocking and directly unit-testable under a
  headless "Agg" backend.
- Colors are auto-cycled from matplotlib's property cycle rather than a
  hardcoded {"prophet": ..., "catboost": ...} map, so a newly
  registered model charts correctly with no change here. The ensemble
  is the one fixed style -- a red dashed line
  -- so it always stands out from the individual models.
- Uncertainty bounds (yhat_lower/yhat_upper) are drawn as a shaded band
  only for the models that actually provide them (Prophet, SARIMAX and
  Bayesian PyMC do; CatBoost and NeuralProphet do not). The ensemble
  never has bounds, by design (see evaluation/ensemble.py) -- it is
  always a single line.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import matplotlib.pyplot as plt

from horizon_forecaster.evaluation.ensemble import as_ensemble_dict

if TYPE_CHECKING:  # hints only; never imported at runtime (avoids the heavy stack)
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from horizon_forecaster.evaluation.ensemble import EnsembleResult
    from horizon_forecaster.forecaster import EvaluationResult, ForecastResult

_ENSEMBLE_COLOR = "tab:red"
_ACTUAL_COLOR = "black"
_HISTORY_COLOR = "gray"
_BOUND_ALPHA = 0.15

# Shown when an ensemble was requested but none could be formed -- every
# model exceeded mape_threshold, or (under a weighted rule) every model
# had a NaN selection MAPE. The individual-model lines are still
# drawn -- a rejected model still produced a real forecast -- but this
# note makes it unmistakable that those lines are for diagnosis, NOT a
# blessed/trusted result, so a rejected model isn't mistaken for one.
_NO_ENSEMBLE_NOTE = (
    "No ensemble formed — no model was admitted;\n" "individual forecasts shown for diagnosis only"
)


def _new_ax(ax: "Optional[Axes]", figsize=(12, 5)) -> "Axes":
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _finalize(ax: "Axes", title: str) -> "Axes":
    ax.set_title(title)
    ax.legend()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.figure.tight_layout()
    return ax


def _draw_bounds(ax: "Axes", df: "pd.DataFrame", color) -> None:
    """Shade yhat_lower..yhat_upper if the model produced them."""
    if "yhat_lower" in df.columns and "yhat_upper" in df.columns:
        ax.fill_between(
            df["ds"], df["yhat_lower"], df["yhat_upper"], alpha=_BOUND_ALPHA, color=color
        )


def _primary_ensemble(ensemble) -> "Optional[EnsembleResult]":
    """
    The single ensemble to draw on a line chart.

    Overlaying several combination rules on one time-series axes buries
    the per-model lines the chart exists to show, so the line plots draw
    the FIRST requested method -- which is the configured default unless
    the caller deliberately reordered it. plot_model_comparison() is the
    exception: comparing rules is exactly what a bar chart is for, so it
    shows one bar per method.
    """
    ensembles = as_ensemble_dict(ensemble)
    if not ensembles:
        return None
    return next(iter(ensembles.values()))


def _ensemble_label(result: "EnsembleResult") -> str:
    """
    Name the combined line by the rule that was ACTUALLY applied, not the
    one requested -- a two-member plain mean labelled "trimmed" would
    misrepresent the chart.
    """
    label = f"ensemble ({result.rule_applied})"
    return f"{label} [degraded]" if result.degraded else label


def _annotate_no_ensemble(ax: "Axes", ensemble: "Optional[EnsembleResult]") -> None:
    """
    Draw the "no ensemble formed" note when an ensemble was requested but
    every model was rejected (ensemble.combined is None). Draws nothing
    when no ensemble was passed at all, or when one was formed -- so it is
    always safe to call unconditionally.
    """
    if ensemble is not None and ensemble.combined is None:
        # Reserve headroom at the top so the note never sits on top of the
        # bars/lines (or their value labels); then drop the note into it.
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + (top - bottom) * 0.25)
        ax.text(
            0.5,
            0.98,
            _NO_ENSEMBLE_NOTE,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=_ENSEMBLE_COLOR,
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=_ENSEMBLE_COLOR, alpha=0.9),
        )


def plot_evaluation(
    evaluation_results: "Dict[str, EvaluationResult]",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    *,
    ax: "Optional[Axes]" = None,
    title: str = "Test set: actual vs predicted, by model",
) -> "Axes":
    """
    Plot held-out actuals against each model's test-set predictions.

    Draws the real actual series once (it is identical across models --
    same held-out split), then one line per model with its MAPE in the
    legend label, plus a shaded uncertainty band for any model that has
    one. If `ensemble` is given and an ensemble was actually formed
    (combined is not None), it is overlaid as a red dashed line, with its
    own MAPE in the label when available. If an ensemble was given but
    none could be formed (every model exceeded mape_threshold), the
    individual lines are still drawn and a visible red note flags that
    they are for diagnosis only -- not a trusted/blended result.

    Args:
        evaluation_results: the dict returned by HorizonForecaster.evaluate(),
            {model_name: EvaluationResult}. Each result's test_predictions
            must have columns ["ds", "y", "yhat"] (+ optional bounds).
        ensemble: the forecaster's evaluate_ensemble (EnsembleResult), or
            None to skip the ensemble overlay.
        ax: an existing Axes to draw on; a new figure/Axes is created if
            omitted.
        title: chart title.

    Returns:
        The Axes drawn on (never shown -- the caller decides).

    Raises:
        ValueError: if evaluation_results is empty (nothing to plot).
    """
    if not evaluation_results:
        raise ValueError("plot_evaluation: evaluation_results is empty, nothing to plot.")

    ax = _new_ax(ax)

    first = next(iter(evaluation_results.values()))
    ax.plot(
        first.test_predictions["ds"],
        first.test_predictions["y"],
        label="Actual",
        color=_ACTUAL_COLOR,
        marker="o",
    )

    for name, result in evaluation_results.items():
        preds = result.test_predictions
        (line,) = ax.plot(
            preds["ds"],
            preds["yhat"],
            label=f"{name} predicted (MAPE {result.mape.value:.2f}%)",
            marker="o",
        )
        _draw_bounds(ax, preds, line.get_color())

    ensemble = _primary_ensemble(ensemble)
    if ensemble is not None and ensemble.combined is not None:
        mape_label = f" (MAPE {ensemble.mape.value:.2f}%)" if ensemble.mape is not None else ""
        ax.plot(
            ensemble.combined["ds"],
            ensemble.combined["yhat"],
            label=f"{_ensemble_label(ensemble)}{mape_label}",
            color=_ENSEMBLE_COLOR,
            linestyle="--",
            marker="x",
        )
    else:
        _annotate_no_ensemble(ax, ensemble)

    return _finalize(ax, title)


def plot_forecast(
    forecast_results: "Dict[str, ForecastResult]",
    *,
    history: "Optional[pd.DataFrame]" = None,
    date_column: str = "ds",
    target_column: str = "y",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    ax: "Optional[Axes]" = None,
    title: str = "History + forecast, by model",
) -> "Axes":
    """
    Plot the future forecast per model, optionally on top of history.

    Args:
        forecast_results: the dict returned by HorizonForecaster.forecast(),
            {model_name: ForecastResult}. Each result's forecast must have
            columns ["ds", "yhat"] (+ optional bounds).
        history: optional raw history DataFrame to draw behind the
            forecast, so trend continuation can be eyeballed. If given,
            date_column/target_column name its date and value columns.
        date_column, target_column: column names in `history` (ignored
            when history is None).
        ensemble: the forecaster's forecast_ensemble (EnsembleResult), or
            None to skip the ensemble overlay. The forecast ensemble never
            carries its own MAPE (no future actuals exist), so no MAPE is
            shown in its label. If an ensemble was given but none could be
            formed, a visible red note flags the individual forecasts as
            diagnosis-only (see plot_evaluation).
        ax: an existing Axes to draw on; a new figure/Axes is created if
            omitted.
        title: chart title.

    Returns:
        The Axes drawn on (never shown -- the caller decides).

    Raises:
        ValueError: if forecast_results is empty (nothing to plot).
    """
    if not forecast_results:
        raise ValueError("plot_forecast: forecast_results is empty, nothing to plot.")

    ax = _new_ax(ax)

    if history is not None:
        ax.plot(
            history[date_column],
            history[target_column],
            label="History",
            color=_HISTORY_COLOR,
        )

    for name, result in forecast_results.items():
        fc = result.forecast
        (line,) = ax.plot(fc["ds"], fc["yhat"], label=f"{name} forecast")
        _draw_bounds(ax, fc, line.get_color())

    ensemble = _primary_ensemble(ensemble)
    if ensemble is not None and ensemble.combined is not None:
        ax.plot(
            ensemble.combined["ds"],
            ensemble.combined["yhat"],
            label=f"{_ensemble_label(ensemble)} forecast",
            color=_ENSEMBLE_COLOR,
            linestyle="--",
        )
    else:
        _annotate_no_ensemble(ax, ensemble)

    return _finalize(ax, title)


def plot_model_comparison(
    evaluation_results: "Dict[str, EvaluationResult]",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    *,
    ax: "Optional[Axes]" = None,
    title: str = "Model comparison: MAPE (lower is better)",
) -> "Axes":
    """
    Bar chart comparing each model's test-window MAPE, and the ensemble's.

    These are the REPORTED scores, not the validation scores models were
    selected on -- so a model that looks strong here earned it on data no
    selection path was allowed to see.

    A quick side-by-side of the same numbers plot_evaluation puts in its
    legend -- easier to compare at a glance, and it makes the ensemble's
    "not automatically better than the best individual model" behavior
    (see evaluation/ensemble.py) visible directly. The ensemble bar is
    only added when an ensemble was formed AND scored (evaluate() context,
    where held-out actuals exist); it is drawn in the same red as the
    ensemble line in the other charts.

    Args:
        evaluation_results: {model_name: EvaluationResult} from evaluate().
        ensemble: evaluate_ensemble (EnsembleResult), or None. Its bar is
            shown only when ensemble.mape is not None.
        ax: existing Axes, or None to create one.
        title: chart title.

    Returns:
        The Axes drawn on (never shown).

    Raises:
        ValueError: if evaluation_results is empty.
    """
    if not evaluation_results:
        raise ValueError("plot_model_comparison: evaluation_results is empty, nothing to plot.")

    names = list(evaluation_results.keys())
    values = [evaluation_results[n].mape.value for n in names]
    colors = ["tab:blue"] * len(names)

    # One bar per combination method. Comparing rules head-to-head is
    # precisely what this chart is for, so unlike the line plots it does
    # not collapse to a single primary ensemble.
    ensembles = as_ensemble_dict(ensemble)
    for method, result in ensembles.items():
        if result.mape is None:
            continue
        names.append(f"ensemble\n({result.rule_applied})" if len(ensembles) > 1 else "ensemble")
        values.append(result.mape.value)
        colors.append(_ENSEMBLE_COLOR)
    ensemble = _primary_ensemble(ensemble)

    ax = _new_ax(ax, figsize=(8, 5))
    bars = ax.bar(names, values, color=colors)
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title)
    _annotate_no_ensemble(ax, ensemble)
    ax.figure.tight_layout()
    return ax


def plot_evaluation_per_model(
    evaluation_results: "Dict[str, EvaluationResult]",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    *,
    figsize=(11, 4.5),
) -> "Dict[str, Figure]":
    """
    One full-size figure per model (and one for the ensemble when formed),
    each showing the held-out actuals against that single series with its
    bounds and MAPE. Preferred over the single-axes overlay and the old
    faceted grid when there are several models -- each series gets its own
    readable, normal-sized plot instead of a cramped shared panel.

    Args:
        evaluation_results: {model_name: EvaluationResult} from evaluate().
        ensemble: evaluate_ensemble (EnsembleResult), or None. A formed
            ensemble gets its own figure; when one was requested but not
            formed, each model figure carries the red "no ensemble" note.
        figsize: size of each individual figure.

    Returns:
        {label: Figure} -- one entry per model, plus "ensemble" when a
        combination was formed. Never shown; the caller decides.

    Raises:
        ValueError: if evaluation_results is empty.
    """
    if not evaluation_results:
        raise ValueError("plot_evaluation_per_model: evaluation_results is empty, nothing to plot.")

    ensemble = _primary_ensemble(ensemble)
    actual = next(iter(evaluation_results.values())).test_predictions[["ds", "y"]]
    figures: "Dict[str, Figure]" = {}

    for name, result in evaluation_results.items():
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(actual["ds"], actual["y"], label="Actual", color=_ACTUAL_COLOR, marker="o")
        (line,) = ax.plot(
            result.test_predictions["ds"],
            result.test_predictions["yhat"],
            label=f"{name} predicted",
            marker="o",
        )
        _draw_bounds(ax, result.test_predictions, line.get_color())
        _annotate_no_ensemble(ax, ensemble)
        _finalize(ax, f"Test set: actual vs {name} (MAPE {result.mape.value:.2f}%)")
        figures[name] = fig

    if ensemble is not None and ensemble.combined is not None:
        mape_label = f" (MAPE {ensemble.mape.value:.2f}%)" if ensemble.mape is not None else ""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(actual["ds"], actual["y"], label="Actual", color=_ACTUAL_COLOR, marker="o")
        ax.plot(
            ensemble.combined["ds"],
            ensemble.combined["yhat"],
            label=_ensemble_label(ensemble),
            color=_ENSEMBLE_COLOR,
            linestyle="--",
            marker="x",
        )
        _finalize(ax, f"Test set: actual vs ensemble{mape_label}")
        figures["ensemble"] = fig

    return figures


def _savefig(fig: "Figure", path: Path, dpi: int) -> str:
    """Write a figure to disk (tight bbox) and close it to free memory."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_forecast_per_model(
    forecast_results: "Dict[str, ForecastResult]",
    *,
    history: "Optional[pd.DataFrame]" = None,
    date_column: str = "ds",
    target_column: str = "y",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    figsize=(11, 4.5),
) -> "Dict[str, Figure]":
    """
    One full-size figure per model (and one for the ensemble when formed),
    each showing that single model's forecast over the optional history.
    Args and return mirror plot_evaluation_per_model.

    Raises:
        ValueError: if forecast_results is empty.
    """
    if not forecast_results:
        raise ValueError("plot_forecast_per_model: forecast_results is empty, nothing to plot.")

    ensemble = _primary_ensemble(ensemble)
    figures: "Dict[str, Figure]" = {}

    def _figure(label, fc, is_ensemble):
        # `label` names the rule that ACTUALLY ran for the ensemble figure,
        # while the returned dict key stays the plain "ensemble" -- the key
        # is a stable handle (and a filename in save_forecast_plots), the
        # label is what a reader sees and must not call a degraded
        # two-member mean "trimmed".
        fig, ax = plt.subplots(figsize=figsize)
        if history is not None:
            ax.plot(
                history[date_column], history[target_column], label="History", color=_HISTORY_COLOR
            )
        color = _ENSEMBLE_COLOR if is_ensemble else None
        (line,) = ax.plot(
            fc["ds"],
            fc["yhat"],
            label=f"{label} forecast",
            color=color,
            linestyle="--" if is_ensemble else "-",
        )
        _draw_bounds(ax, fc, line.get_color())
        if not is_ensemble:
            _annotate_no_ensemble(ax, ensemble)
        _finalize(ax, f"History + {label} forecast")
        return fig

    for name, result in forecast_results.items():
        figures[name] = _figure(name, result.forecast, False)
    if ensemble is not None and ensemble.combined is not None:
        figures["ensemble"] = _figure(_ensemble_label(ensemble), ensemble.combined, True)

    return figures


def save_evaluation_plots(
    evaluation_results: "Dict[str, EvaluationResult]",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    *,
    output_dir: str = ".",
    prefix: str = "evaluation",
    dpi: int = 100,
) -> Dict[str, str]:
    """
    Render and save the evaluation charts to disk (PNG).

    Writes into output_dir (created if missing): the overlay
    ("{prefix}_overlay.png"), the MAPE bar comparison
    ("{prefix}_mape_comparison.png"), and one dedicated full-size figure
    per model plus the ensemble ("{prefix}_<model>.png",
    "{prefix}_ensemble.png"). Each figure is closed after writing.

    Returns:
        {"overlay": path, "mape_comparison": path, "<model>": path, ...}.

    Raises:
        ValueError: if evaluation_results is empty.
    """
    if not evaluation_results:
        raise ValueError("save_evaluation_plots: evaluation_results is empty, nothing to save.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}
    written["overlay"] = _savefig(
        plot_evaluation(evaluation_results, ensemble).figure, out / f"{prefix}_overlay.png", dpi
    )
    written["mape_comparison"] = _savefig(
        plot_model_comparison(evaluation_results, ensemble).figure,
        out / f"{prefix}_mape_comparison.png",
        dpi,
    )
    for label, fig in plot_evaluation_per_model(evaluation_results, ensemble).items():
        written[label] = _savefig(fig, out / f"{prefix}_{label}.png", dpi)
    return written


def save_forecast_plots(
    forecast_results: "Dict[str, ForecastResult]",
    *,
    history: "Optional[pd.DataFrame]" = None,
    date_column: str = "ds",
    target_column: str = "y",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    output_dir: str = ".",
    prefix: str = "forecast",
    dpi: int = 100,
) -> Dict[str, str]:
    """
    Render and save the forecast charts to disk (PNG).

    Writes into output_dir (created if missing): the overlay
    ("{prefix}_overlay.png") and one dedicated full-size figure per model
    plus the ensemble ("{prefix}_<model>.png", "{prefix}_ensemble.png").
    Each figure is closed after writing.

    Returns:
        {"overlay": path, "<model>": path, ...}.

    Raises:
        ValueError: if forecast_results is empty.
    """
    if not forecast_results:
        raise ValueError("save_forecast_plots: forecast_results is empty, nothing to save.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}
    written["overlay"] = _savefig(
        plot_forecast(
            forecast_results,
            history=history,
            date_column=date_column,
            target_column=target_column,
            ensemble=ensemble,
        ).figure,
        out / f"{prefix}_overlay.png",
        dpi,
    )
    for label, fig in plot_forecast_per_model(
        forecast_results,
        history=history,
        date_column=date_column,
        target_column=target_column,
        ensemble=ensemble,
    ).items():
        written[label] = _savefig(fig, out / f"{prefix}_{label}.png", dpi)
    return written
