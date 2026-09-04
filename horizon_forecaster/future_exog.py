"""
Future-exogenous-data strategies.

When forecast() is called without (complete) future values for the declared
independent columns, these strategies generate them -- model-agnostically,
ONCE, from the raw history -- so every model receives the same assumed
future drivers. This is a SCENARIO, not a prediction: the forecast is only
as good as these assumptions, which is why forecast() warns and exposes the
resolved values as `forecaster.future_exog_used`.

The design mirrors evaluation/ensemble.py: a `FUTURE_EXOG_STRATEGIES`
registry of pure functions with no dependency on forecaster/model/adapter
classes, so it is standalone and unit-testable.

Strategies (all defined for any horizon -- short takes the head, long tiles):
    carry_forward       last observed value held flat.
    recent_average      mean/median of the last N periods.
    historical_average  SEASON-AWARE mean/median (by day-of-week for daily,
                        week-of-year for weekly), not one flat mean.
    seasonal_naive      the seasonal cycle repeated, PLUS drift (the fitted
                        historical trend), so growing seasonal drivers keep
                        growing. Falls back to carry_forward under one cycle
                        of history.
    linear_trend        a fitted trend line projected forward.
    cycle               legacy: replay/tile the trailing history, the
                        pre-1.0 default, kept for backward compatibility.

Type-aware: continuous drivers use the chosen strategy; binary 0/1 event
flags (promos, holidays) default to 0 -- "no event unless you supply one" --
EXCEPT under seasonal_naive, where they follow the cycle so recurring events
still repeat. Flags come from an explicit `event_columns` list if given,
else are auto-detected (<=2 distinct, non-negative values). The `cycle`
legacy strategy is exempt from type-awareness and tiles every column as-is,
to reproduce the old behaviour exactly.

Robustness: history is coerced to numeric and all aggregations are NaN-aware;
any continuous output for a non-negative driver is clipped at 0 (so a
projected trend can't go negative); out-of-range *clamping* is left to the
model adapters, which already do it.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from horizon_forecaster.exceptions import InvalidConfigError

_SEASONAL_PERIOD = {"daily": 7, "weekly": 52}
AGGREGATORS: Dict[str, Callable] = {"mean": np.nanmean, "median": np.nanmedian}


@dataclass
class _Ctx:
    """Shared, column-independent context computed once per resolve() call."""

    horizon: int
    hist_steps: np.ndarray  # history time index, in frequency steps from the first date
    future_steps: np.ndarray  # future time index, same origin
    hist_phase: np.ndarray  # seasonal phase per history row (dow / week-of-year)
    future_phase: np.ndarray  # seasonal phase per future row
    seasonal_period: int
    aggregator: Callable  # np.nanmean or np.nanmedian
    window: int  # N for recent_average


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _phase(dates, frequency: str) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    if frequency == "daily":
        return d.dt.dayofweek.to_numpy()  # 0..6
    return d.dt.isocalendar().week.to_numpy().astype(int) - 1  # 0..52


def _last_valid(hist: np.ndarray) -> float:
    finite = hist[np.isfinite(hist)]
    return float(finite[-1]) if finite.size else 0.0


def _fit_trend(hist: np.ndarray, steps: np.ndarray) -> Tuple[float, float]:
    """Least-squares (slope, intercept) of value vs frequency-step, NaN-safe."""
    m = np.isfinite(hist) & np.isfinite(steps)
    if m.sum() < 2:
        return 0.0, _last_valid(hist)
    slope, intercept = np.polyfit(steps[m], hist[m], 1)
    return float(slope), float(intercept)


# --------------------------------------------------------------------------
# continuous strategy functions: (hist, ctx) -> np.ndarray[horizon]
# --------------------------------------------------------------------------


def _carry_forward(hist, ctx):
    return np.full(ctx.horizon, _last_valid(hist))


def _recent_average(hist, ctx):
    finite = hist[np.isfinite(hist)]
    tail = finite[-ctx.window :] if finite.size else finite
    val = float(ctx.aggregator(tail)) if tail.size else 0.0
    return np.full(ctx.horizon, val)


def _historical_average(hist, ctx):
    finite_mask = np.isfinite(hist)
    overall = float(ctx.aggregator(hist[finite_mask])) if finite_mask.any() else 0.0
    out = np.empty(ctx.horizon)
    for i, p in enumerate(ctx.future_phase):
        m = finite_mask & (ctx.hist_phase == p)
        out[i] = float(ctx.aggregator(hist[m])) if m.any() else overall
    return out


def _seasonal_naive(hist, ctx):
    finite_mask = np.isfinite(hist)
    # Under one full cycle of history there is no seasonal profile to speak of.
    if finite_mask.sum() < ctx.seasonal_period:
        return _carry_forward(hist, ctx)
    slope, intercept = _fit_trend(hist, ctx.hist_steps)
    resid = hist - (intercept + slope * ctx.hist_steps)  # detrended history
    profile = {}
    for p in np.unique(ctx.future_phase):
        m = finite_mask & (ctx.hist_phase == p)
        profile[p] = float(ctx.aggregator(resid[m])) if m.any() else 0.0
    trend = intercept + slope * ctx.future_steps  # drift
    season = np.array([profile[p] for p in ctx.future_phase])
    return trend + season


def _linear_trend(hist, ctx):
    slope, intercept = _fit_trend(hist, ctx.hist_steps)
    return intercept + slope * ctx.future_steps


def _cycle(hist, ctx):
    """Legacy: tile the trailing history to cover the horizon (the pre-1.0 default)."""
    n = hist.size
    if n == 0:
        return np.zeros(ctx.horizon)
    reps = int(np.ceil(ctx.horizon / n))
    return np.tile(hist, reps)[-ctx.horizon :]


FUTURE_EXOG_STRATEGIES: Dict[str, Callable] = {
    "carry_forward": _carry_forward,
    "recent_average": _recent_average,
    "historical_average": _historical_average,
    "seasonal_naive": _seasonal_naive,
    "linear_trend": _linear_trend,
    "cycle": _cycle,
}


# --------------------------------------------------------------------------
# flag handling
# --------------------------------------------------------------------------


def _looks_binary(hist: np.ndarray) -> bool:
    finite = hist[np.isfinite(hist)]
    return finite.size > 0 and np.unique(finite).size <= 2 and finite.min() >= 0


def _is_flag(col: str, hist: np.ndarray, event_columns: Optional[List[str]]) -> bool:
    if event_columns is not None:  # explicit declaration wins; no auto-detect
        return col in event_columns
    return _looks_binary(hist)


def _flag_seasonal(hist, ctx):
    """A binary flag under seasonal_naive: most recent value at each future phase."""
    out = np.zeros(ctx.horizon)
    finite = np.isfinite(hist)
    for i, p in enumerate(ctx.future_phase):
        idx = np.where(finite & (ctx.hist_phase == p))[0]
        out[i] = float(hist[idx[-1]]) if idx.size else 0.0
    return out


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def resolve_future_exog(
    history_df: pd.DataFrame,
    date_column: str,
    independent_columns: List[str],
    future_dates: pd.Series,
    frequency: str,
    strategy: str = "carry_forward",
    aggregator: str = "mean",
    window: int = 28,
    event_columns: Optional[List[str]] = None,
    supplied: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Resolve future values for every independent column, honouring PARTIAL
    FILL: a column present in `supplied` is used as-is; any other column is
    GENERATED via `strategy`.

    Returns (future_df, generated_columns):
        future_df           a DataFrame of `independent_columns`, length == horizon.
        generated_columns   the columns that were auto-generated (vs supplied).

    Raises InvalidConfigError on an unknown strategy/aggregator.
    """
    if strategy not in FUTURE_EXOG_STRATEGIES:
        raise InvalidConfigError(
            f"Unknown future_exog_strategy '{strategy}'. "
            f"Available: {sorted(FUTURE_EXOG_STRATEGIES)}.",
            param_name="future_exog_strategy",
            provided_value=strategy,
        )
    if aggregator not in AGGREGATORS:
        raise InvalidConfigError(
            f"Unknown future_exog_aggregator '{aggregator}'. Available: {sorted(AGGREGATORS)}.",
            param_name="future_exog_aggregator",
            provided_value=aggregator,
        )

    horizon = len(future_dates)
    hist_dates = pd.to_datetime(history_df[date_column]).reset_index(drop=True)
    ref = hist_dates.min()
    per = 1.0 if frequency == "daily" else 7.0

    def _steps(dates):
        return (
            pd.to_datetime(pd.Series(dates).reset_index(drop=True)) - ref
        ).dt.days.to_numpy() / per

    ctx = _Ctx(
        horizon=horizon,
        hist_steps=_steps(hist_dates),
        future_steps=_steps(future_dates),
        hist_phase=_phase(hist_dates, frequency),
        future_phase=_phase(future_dates, frequency),
        seasonal_period=_SEASONAL_PERIOD.get(frequency, 7),
        aggregator=AGGREGATORS[aggregator],
        window=int(window),
    )

    supplied_cols = set(supplied.columns) if supplied is not None else set()
    out: Dict[str, np.ndarray] = {}
    generated: List[str] = []

    for col in independent_columns:
        if col in supplied_cols:
            out[col] = pd.to_numeric(supplied[col], errors="coerce").to_numpy(dtype=float)
            continue

        hist = pd.to_numeric(history_df[col], errors="coerce").to_numpy(dtype=float)

        if strategy == "cycle":
            # legacy: exempt from type-awareness; tile every column as-is
            out[col] = _cycle(hist, ctx)
        elif _is_flag(col, hist, event_columns):
            out[col] = (
                _flag_seasonal(hist, ctx) if strategy == "seasonal_naive" else np.zeros(horizon)
            )
        else:
            vals = FUTURE_EXOG_STRATEGIES[strategy](hist, ctx)
            finite = hist[np.isfinite(hist)]
            if finite.size and finite.min() >= 0:  # non-negative driver -> no negative future
                vals = np.clip(vals, 0.0, None)
            out[col] = vals
        generated.append(col)

    future_df = pd.DataFrame({c: out[c] for c in independent_columns})
    return future_df, generated
