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
from horizon_forecaster.exceptions import InvalidConfigError
from horizon_forecaster.utils import check_history, check_results_dict

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


def _charts_for(ensemble) -> "list":
    """
    The (key, EnsembleResult) pairs to draw, ONE CHART PER COMBINATION RULE.

    Overlaying several rules on a single time-series axes buries the
    per-model lines the chart exists to show, so each rule gets its own
    chart against the same models and actuals rather than sharing one.
    Returning a list of pairs keeps that decision in one place: every line
    chart iterates it, so none of them can silently drop a rule the caller
    asked for.

    With no ensemble at all there is still one chart to draw -- the models
    on their own -- keyed "models" so the return shape never depends on
    whether an ensemble was requested.
    """
    ensembles = as_ensemble_dict(ensemble)
    if not ensembles:
        return [("models", None)]
    return list(ensembles.items())


def _chart_title(base: str, result: "Optional[EnsembleResult]") -> str:
    """Name a per-rule chart after the rule that actually ran."""
    if result is None:
        return base
    return f"{base} - {_ensemble_label(result)}"


def _reject_shared_ax(caller: str, ax: "Optional[Axes]", n_charts: int) -> None:
    """
    A supplied Axes can only receive one chart.

    Drawing several rules onto one caller-supplied Axes would reintroduce
    exactly the overlaying this module avoids, and silently ignoring the
    extra rules is what this function exists to prevent. So it is an error
    the caller can act on rather than a surprise in the output.
    """
    if ax is not None and n_charts > 1:
        raise InvalidConfigError(
            f"{caller}: `ax` cannot be used when several combination methods were "
            f"built ({n_charts} here) -- each rule is drawn on its own Axes. Omit "
            f"`ax` to let each chart create its own figure, or pass a single "
            f"EnsembleResult to draw one rule onto the Axes you supplied."
        )


def _ensemble_label(result: "EnsembleResult") -> str:
    """
    Name the combined line by the rule that was ACTUALLY applied, not the
    one requested -- a two-member plain mean labelled "trimmed" would
    misrepresent the chart.
    """
    label = f"ensemble ({result.rule_applied})"
    return f"{label} [degraded]" if result.degraded else label


def _no_ensemble_marker(ensembles: "Dict[str, EnsembleResult]") -> "Optional[EnsembleResult]":
    """
    A result to hand _annotate_no_ensemble on a chart that carries EVERY
    rule at once (the bar chart).

    The note is only warranted when NO rule produced a combination: if one
    rule formed and another did not, the chart still shows a real
    ensemble bar, and a blanket "no ensemble formed" note would
    contradict it. So this returns an unformed result only when every
    requested rule came back empty, and None otherwise.
    """
    if not ensembles:
        return None
    unformed = [r for r in ensembles.values() if r.combined is None]
    if len(unformed) == len(ensembles):
        return unformed[0]
    return None


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
) -> "Dict[str, Axes]":
    """
    Plot held-out actuals against each model's test-set predictions,
    ONE CHART PER COMBINATION RULE.

    Each chart draws the real actual series once (it is identical across
    models -- same held-out split), then one line per model with its MAPE
    in the legend label, plus a shaded uncertainty band for any model that
    has one, and finally that chart's own ensemble overlaid as a red
    dashed line with its MAPE where available. Asking for three rules
    therefore gives three charts of the same five models, each against a
    different combination -- which is what makes the rules comparable
    against the actuals rather than only against each other.

    Where a rule was requested but no ensemble could be formed (every
    model exceeded mape_threshold), that chart still draws the individual
    lines and adds a visible red note flagging them as diagnosis only --
    not a trusted, blended result.

    Args:
        evaluation_results: the dict returned by HorizonForecaster.evaluate(),
            {model_name: EvaluationResult}. Each result's test_predictions
            must have columns ["ds", "y", "yhat"] (+ optional bounds).
        ensemble: the forecaster's evaluate_ensemble
            ({method: EnsembleResult}), a single EnsembleResult, or None to
            draw the models alone.
        ax: an existing Axes to draw on. Only valid when a single chart
            will be produced -- see Raises.
        title: base chart title; each chart appends the rule it drew.

    Returns:
        {key: Axes} -- one entry per combination method, keyed by method
        name, or the single key "models" when no ensemble was given.
        ALWAYS a dict, even for one method, matching the convention
        evaluate()/forecast()/evaluate_ensemble already follow. Never
        shown -- the caller decides.

    Raises:
        InvalidConfigError (a ValueError): if evaluation_results is empty (nothing to plot), or
            if `ax` is supplied for more than one combination method.
    """
    check_results_dict(evaluation_results, "evaluation", "plot_evaluation")
    if not evaluation_results:
        raise InvalidConfigError("plot_evaluation: evaluation_results is empty, nothing to plot.")

    charts = _charts_for(ensemble)
    _reject_shared_ax("plot_evaluation", ax, len(charts))

    actual = next(iter(evaluation_results.values())).test_predictions
    axes: "Dict[str, Axes]" = {}

    for key, result in charts:
        current = _new_ax(ax)
        current.plot(
            actual["ds"],
            actual["y"],
            label="Actual",
            color=_ACTUAL_COLOR,
            marker="o",
        )

        for name, model_result in evaluation_results.items():
            preds = model_result.test_predictions
            (line,) = current.plot(
                preds["ds"],
                preds["yhat"],
                label=f"{name} predicted (MAPE {model_result.mape.value:.2f}%)",
                marker="o",
            )
            _draw_bounds(current, preds, line.get_color())

        if result is not None and result.combined is not None:
            mape_label = f" (MAPE {result.mape.value:.2f}%)" if result.mape is not None else ""
            current.plot(
                result.combined["ds"],
                result.combined["yhat"],
                label=f"{_ensemble_label(result)}{mape_label}",
                color=_ENSEMBLE_COLOR,
                linestyle="--",
                marker="x",
            )
        else:
            _annotate_no_ensemble(current, result)

        axes[key] = _finalize(current, _chart_title(title, result))

    return axes


def plot_forecast(
    forecast_results: "Dict[str, ForecastResult]",
    *,
    history: "Optional[pd.DataFrame]" = None,
    date_column: str = "ds",
    target_column: str = "y",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    ax: "Optional[Axes]" = None,
    title: str = "History + forecast, by model",
) -> "Dict[str, Axes]":
    """
    Plot the future forecast per model, optionally on top of history,
    ONE CHART PER COMBINATION RULE (see plot_evaluation).

    Args:
        forecast_results: the dict returned by HorizonForecaster.forecast(),
            {model_name: ForecastResult}. Each result's forecast must have
            columns ["ds", "yhat"] (+ optional bounds).
        history: optional raw history DataFrame to draw behind the
            forecast, so trend continuation can be eyeballed. If given,
            date_column/target_column name its date and value columns.
        date_column, target_column: column names in `history` (ignored
            when history is None).
        ensemble: the forecaster's forecast_ensemble
            ({method: EnsembleResult}), a single EnsembleResult, or None to
            draw the models alone. A forecast ensemble never carries its
            own MAPE (no future actuals exist), so no MAPE is shown in its
            label. Where a rule was requested but none could be formed, a
            visible red note flags the individual forecasts as diagnosis
            only (see plot_evaluation).
        ax: an existing Axes to draw on. Only valid when a single chart
            will be produced -- see Raises.
        title: base chart title; each chart appends the rule it drew.

    Returns:
        {key: Axes} -- one entry per combination method, or the single key
        "models" when no ensemble was given. Always a dict.

    Raises:
        InvalidConfigError (a ValueError): if forecast_results is empty (nothing to plot), or if
            `ax` is supplied for more than one combination method.
    """
    check_results_dict(forecast_results, "forecast", "plot_forecast")
    history = check_history(history, date_column, target_column, "plot_forecast")
    if not forecast_results:
        raise InvalidConfigError("plot_forecast: forecast_results is empty, nothing to plot.")

    charts = _charts_for(ensemble)
    _reject_shared_ax("plot_forecast", ax, len(charts))

    axes: "Dict[str, Axes]" = {}

    for key, result in charts:
        current = _new_ax(ax)

        if history is not None:
            current.plot(
                history[date_column],
                history[target_column],
                label="History",
                color=_HISTORY_COLOR,
            )

        for name, model_result in forecast_results.items():
            fc = model_result.forecast
            (line,) = current.plot(fc["ds"], fc["yhat"], label=f"{name} forecast")
            _draw_bounds(current, fc, line.get_color())

        if result is not None and result.combined is not None:
            current.plot(
                result.combined["ds"],
                result.combined["yhat"],
                label=f"{_ensemble_label(result)} forecast",
                color=_ENSEMBLE_COLOR,
                linestyle="--",
            )
        else:
            _annotate_no_ensemble(current, result)

        axes[key] = _finalize(current, _chart_title(title, result))

    return axes


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
        ensemble: evaluate_ensemble ({method: EnsembleResult}), a single
            EnsembleResult, or None. A rule's bar is shown only when its
            mape is not None.
        ax: existing Axes, or None to create one.
        title: chart title.

    Returns:
        The Axes drawn on (never shown).

    Raises:
        InvalidConfigError (a ValueError): if evaluation_results is empty.
    """
    check_results_dict(evaluation_results, "evaluation", "plot_model_comparison")
    if not evaluation_results:
        raise InvalidConfigError(
            "plot_model_comparison: evaluation_results is empty, nothing to plot."
        )

    names = list(evaluation_results.keys())
    values = [evaluation_results[n].mape.value for n in names]
    colors = ["tab:blue"] * len(names)

    # One bar per combination method. Comparing rules head-to-head is
    # precisely what this chart is for, so this is the one chart that puts
    # every rule on a single Axes; the line charts give each its own.
    ensembles = as_ensemble_dict(ensemble)
    for result in ensembles.values():
        if result.mape is None:
            continue
        names.append(f"ensemble ({result.rule_applied})" if len(ensembles) > 1 else "ensemble")
        values.append(result.mape.value)
        colors.append(_ENSEMBLE_COLOR)

    # Widen with the bar count and rotate the labels. Ensemble labels
    # ("ensemble (inverse_mape)") are several times longer than a model
    # name, so at a fixed width they run into their neighbours and become
    # unreadable -- horizontal labels only fit while every name is short.
    # Rotating anchors each label's right end under its own bar, so length
    # no longer decides whether it collides.
    ax = _new_ax(ax, figsize=(max(8.0, 1.35 * len(names)), 5))
    bars = ax.bar(names, values, color=colors)
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    # Headroom for the value labels, which sit above the tallest bar. Only
    # finite values count: a model whose test window held nothing but zero
    # actuals has a NaN MAPE (no bar), and a NaN limit makes matplotlib raise.
    finite = [v for v in values if v == v and abs(v) != float("inf")]
    top = max(finite) if finite else 0.0
    ax.set_ylim(0, top * 1.15 if top > 0 else 1.0)
    _annotate_no_ensemble(ax, _no_ensemble_marker(ensembles))
    ax.figure.tight_layout()
    return ax


def plot_evaluation_per_model(
    evaluation_results: "Dict[str, EvaluationResult]",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    *,
    figsize=(11, 4.5),
) -> "Dict[str, Figure]":
    """
    One full-size figure per model, plus one per combination rule that
    formed, each showing the held-out actuals against that single series
    with its bounds and MAPE.

    Preferred over the overlay when there are several models: each series
    gets its own readable, normal-sized plot instead of a cramped shared
    panel. Five models and three rules therefore give eight figures.

    Args:
        evaluation_results: {model_name: EvaluationResult} from evaluate().
        ensemble: evaluate_ensemble ({method: EnsembleResult}), a single
            EnsembleResult, or None. Every rule that formed gets its own
            figure; when rules were requested and NONE formed, each model
            figure carries the red "no ensemble" note.
        figsize: size of each individual figure.

    Returns:
        {label: Figure} -- one entry per model keyed by model name, plus
        one per formed rule keyed "ensemble_<method>". Never shown; the
        caller decides.

    Raises:
        InvalidConfigError (a ValueError): if evaluation_results is empty.
    """
    check_results_dict(evaluation_results, "evaluation", "plot_evaluation_per_model")
    if not evaluation_results:
        raise InvalidConfigError(
            "plot_evaluation_per_model: evaluation_results is empty, nothing to plot."
        )

    ensembles = as_ensemble_dict(ensemble)
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
        _annotate_no_ensemble(ax, _no_ensemble_marker(ensembles))
        _finalize(ax, f"Test set: actual vs {name} (MAPE {result.mape.value:.2f}%)")
        figures[name] = fig

    for method, result in ensembles.items():
        if result.combined is None:
            continue
        mape_label = f" (MAPE {result.mape.value:.2f}%)" if result.mape is not None else ""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(actual["ds"], actual["y"], label="Actual", color=_ACTUAL_COLOR, marker="o")
        ax.plot(
            result.combined["ds"],
            result.combined["yhat"],
            label=_ensemble_label(result),
            color=_ENSEMBLE_COLOR,
            linestyle="--",
            marker="x",
        )
        _finalize(ax, f"Test set: actual vs {_ensemble_label(result)}{mape_label}")
        # Keyed by the REQUESTED method, while the title names the rule
        # that actually ran: the key is a stable handle (and a filename in
        # save_evaluation_plots), so it must not change when a rule
        # degrades, whereas the title must not call a two-member plain
        # mean "trimmed".
        figures[f"ensemble_{method}"] = fig

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
    One full-size figure per model, plus one per combination rule that
    formed, each showing that single forecast over the optional history.
    Args and return mirror plot_evaluation_per_model; ensemble figures are
    keyed "ensemble_<method>".

    Raises:
        InvalidConfigError (a ValueError): if forecast_results is empty.
    """
    check_results_dict(forecast_results, "forecast", "plot_forecast_per_model")
    history = check_history(history, date_column, target_column, "plot_forecast_per_model")
    if not forecast_results:
        raise InvalidConfigError(
            "plot_forecast_per_model: forecast_results is empty, nothing to plot."
        )

    ensembles = as_ensemble_dict(ensemble)
    unformed = _no_ensemble_marker(ensembles)
    figures: "Dict[str, Figure]" = {}

    def _figure(label, fc, is_ensemble):
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
            _annotate_no_ensemble(ax, unformed)
        _finalize(ax, f"History + {label} forecast")
        return fig

    for name, result in forecast_results.items():
        figures[name] = _figure(name, result.forecast, False)

    for method, result in ensembles.items():
        if result.combined is None:
            continue
        # Label by the rule that ran, key by the rule that was requested
        # -- see the matching note in plot_evaluation_per_model.
        figures[f"ensemble_{method}"] = _figure(_ensemble_label(result), result.combined, True)

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

    Writes into output_dir (created if missing):

      {prefix}_overlay_{method}.png    one per combination rule: every
                                       model, the actuals, and that rule
                                       ({prefix}_overlay_models.png when
                                       no ensemble was given)
      {prefix}_mape_comparison.png     every model and rule as bars
      {prefix}_{model}.png             one per model
      {prefix}_ensemble_{method}.png   one per rule that formed

    The method suffix is present even when a single rule was built, for
    the same reason the CSV exporter suffixes final_ensemble_{method}.csv:
    a filename that changed shape with the number of rules requested
    cannot be globbed by a downstream job. Each figure is closed after
    writing.

    Returns:
        {key: path} using the same keys as the plotting functions --
        "overlay_<method>", "mape_comparison", "<model>",
        "ensemble_<method>".

    Raises:
        InvalidConfigError (a ValueError): if evaluation_results is empty.
    """
    check_results_dict(evaluation_results, "evaluation", "save_evaluation_plots")
    if not evaluation_results:
        raise InvalidConfigError(
            "save_evaluation_plots: evaluation_results is empty, nothing to save."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}
    for key, axes in plot_evaluation(evaluation_results, ensemble).items():
        written[f"overlay_{key}"] = _savefig(axes.figure, out / f"{prefix}_overlay_{key}.png", dpi)
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

    Writes into output_dir (created if missing): one overlay per
    combination rule ("{prefix}_overlay_{method}.png", or
    "{prefix}_overlay_models.png" with no ensemble), one figure per model
    ("{prefix}_{model}.png"), and one per rule that formed
    ("{prefix}_ensemble_{method}.png"). See save_evaluation_plots on why
    the method suffix is always present. Each figure is closed after
    writing.

    Returns:
        {key: path} using the same keys as the plotting functions.

    Raises:
        InvalidConfigError (a ValueError): if forecast_results is empty.
    """
    check_results_dict(forecast_results, "forecast", "save_forecast_plots")
    history = check_history(history, date_column, target_column, "save_forecast_plots")
    if not forecast_results:
        raise InvalidConfigError("save_forecast_plots: forecast_results is empty, nothing to save.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}
    for key, axes in plot_forecast(
        forecast_results,
        history=history,
        date_column=date_column,
        target_column=target_column,
        ensemble=ensemble,
    ).items():
        written[f"overlay_{key}"] = _savefig(axes.figure, out / f"{prefix}_overlay_{key}.png", dpi)
    for label, fig in plot_forecast_per_model(
        forecast_results,
        history=history,
        date_column=date_column,
        target_column=target_column,
        ensemble=ensemble,
    ).items():
        written[label] = _savefig(fig, out / f"{prefix}_{label}.png", dpi)
    return written
