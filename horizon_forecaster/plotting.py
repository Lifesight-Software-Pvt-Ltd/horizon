"""
Plotting helpers for horizon_forecaster (Phase 5, output layer).

Turns the outputs of HorizonForecaster.evaluate()/forecast() into
matplotlib charts. Like evaluation/metrics.py and evaluation/ensemble.py,
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
  hardcoded {"prophet": ..., "catboost": ...} map, so a third/fourth
  model (Phase 6: SARIMAX/PyMC/NeuralProphet) charts correctly with no
  change here. The ensemble is the one fixed style -- a red dashed line
  -- so it always stands out from the individual models.
- Uncertainty bounds (yhat_lower/yhat_upper) are drawn as a shaded band
  only for the models that actually provide them (Prophet does; CatBoost
  does not). The ensemble never has bounds, by design (see
  evaluation/ensemble.py) -- it is always a single line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import matplotlib.pyplot as plt

if TYPE_CHECKING:  # hints only; never imported at runtime (avoids the heavy stack)
    import pandas as pd
    from matplotlib.axes import Axes

    from horizon_forecaster.evaluation.ensemble import EnsembleResult
    from horizon_forecaster.forecaster import EvaluationResult, ForecastResult

_ENSEMBLE_COLOR = "tab:red"
_ACTUAL_COLOR = "black"
_HISTORY_COLOR = "gray"
_BOUND_ALPHA = 0.15


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


def plot_evaluation(
    evaluation_results: "Dict[str, EvaluationResult]",
    ensemble: "Optional[EnsembleResult]" = None,
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
    own MAPE in the label when available.

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
        line, = ax.plot(
            preds["ds"],
            preds["yhat"],
            label=f"{name} predicted (MAPE {result.mape.value:.2f}%)",
            marker="o",
        )
        _draw_bounds(ax, preds, line.get_color())

    if ensemble is not None and ensemble.combined is not None:
        mape_label = (
            f" (MAPE {ensemble.mape.value:.2f}%)" if ensemble.mape is not None else ""
        )
        ax.plot(
            ensemble.combined["ds"],
            ensemble.combined["yhat"],
            label=f"ensemble predicted{mape_label}",
            color=_ENSEMBLE_COLOR,
            linestyle="--",
            marker="x",
        )

    return _finalize(ax, title)


def plot_forecast(
    forecast_results: "Dict[str, ForecastResult]",
    *,
    history: "Optional[pd.DataFrame]" = None,
    date_column: str = "ds",
    target_column: str = "y",
    ensemble: "Optional[EnsembleResult]" = None,
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
            shown in its label.
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
        line, = ax.plot(fc["ds"], fc["yhat"], label=f"{name} forecast")
        _draw_bounds(ax, fc, line.get_color())

    if ensemble is not None and ensemble.combined is not None:
        ax.plot(
            ensemble.combined["ds"],
            ensemble.combined["yhat"],
            label="ensemble forecast",
            color=_ENSEMBLE_COLOR,
            linestyle="--",
        )

    return _finalize(ax, title)


def plot_model_comparison(
    evaluation_results: "Dict[str, EvaluationResult]",
    ensemble: "Optional[EnsembleResult]" = None,
    *,
    ax: "Optional[Axes]" = None,
    title: str = "Model comparison: MAPE (lower is better)",
) -> "Axes":
    """
    Bar chart comparing each model's test-set MAPE, and the ensemble's.

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

    if ensemble is not None and ensemble.mape is not None:
        names.append("ensemble")
        values.append(ensemble.mape.value)
        colors.append(_ENSEMBLE_COLOR)

    ax = _new_ax(ax, figsize=(8, 5))
    bars = ax.bar(names, values, color=colors)
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title)
    ax.figure.tight_layout()
    return ax
