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

Strategies (all defined for any horizon; `cycle` takes the tail of the
history for a short horizon and tiles it for a long one):
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
else are auto-detected (columns whose values are only 0 and 1). The `cycle`
legacy strategy is exempt from type-awareness and tiles every column as-is,
to reproduce the old behaviour exactly.

Robustness: history is coerced to numeric and all aggregations are NaN-aware;
any continuous output for a non-negative driver is clipped at 0 (so a
projected trend can't go negative); out-of-range *clamping* is left to the
SARIMAX and Bayesian PyMC adapters, which already do it (the other models
take values as given -- see _OUT_OF_RANGE_BEHAVIOUR).
"""

import difflib
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from horizon_forecaster.exceptions import InvalidConfigError
from horizon_forecaster.utils import (
    _DAY_FIRST_FORMATS,
    _MONTH_FIRST_FORMATS,
    parse_date_column,
    parse_driver_column,
)

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
    """
    A 0/1 column -- the only shape auto-detected as an event flag. Any other
    two-value column is a quantity, not an on/off switch: TV spend of 0 or
    5,000, a price of 9.99 or 12.99, a 0% or 20% discount. Treating those as
    flags set their future to 0 -- a free product, no spend at all.
    """
    finite = hist[np.isfinite(hist)]
    return finite.size > 0 and set(np.unique(finite)) <= {0.0, 1.0}


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


# --------------------------------------------------------------------------
# warning about generated values that are likely wrong
# --------------------------------------------------------------------------

# carry_forward holds the last value; this many preceding periods give the
# "usual" level it is compared with.
_SPIKE_LOOKBACK = {"daily": 28, "weekly": 13}
_SPIKE_HIGH = 1.5  # last value at least this many times the usual level ...
_SPIKE_LOW = 0.5  # ... or at most this fraction of it
# An event last year matches a forecast date within this many days of the
# same weekday one year (364 days) on -- wide enough for holidays that move
# by a few days (Black Friday), not for ones that move by weeks (Easter).
_EVENT_TOLERANCE_DAYS = 7


def _num(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:g}"


def describe_generation_risks(
    history_df: pd.DataFrame,
    date_column: str,
    independent_columns: List[str],
    generated: List[str],
    future_dates: pd.Series,
    frequency: str,
    strategy: str,
    event_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Warning messages for two ways generated future drivers go wrong.

    1. A spike held flat. carry_forward repeats the last value, so a last
       period that was a promotion -- spend 2.6x its usual level, or an event
       flag of 1 -- sets the level for the whole forecast. In testing that
       lifted an 8-week forecast by 15%; on real retail data the forecast
       came out at 1.6-2.5x the same period last year.
    2. A known event left out. A generated flag is 0 throughout the forecast
       (or, under seasonal_naive, repeated by week number or weekday, which
       puts a moving holiday in the wrong period), so an event that happened
       at this time last year is missed unless supplied -- Black Friday, in
       one real run.

    Returns [] when nothing applies. The `cycle` legacy strategy tiles the
    history as-is and is exempt, as it is from flag handling.
    """
    if strategy == "cycle" or not generated:
        return []
    unit = "day" if frequency == "daily" else "week"
    future_dates = pd.Series(future_dates).reset_index(drop=True)
    hist_dates = pd.to_datetime(history_df[date_column]).reset_index(drop=True)
    numeric = {
        c: pd.to_numeric(history_df[c], errors="coerce").to_numpy(dtype=float)
        for c in independent_columns
    }
    flags = [c for c in independent_columns if _is_flag(c, numeric[c], event_columns)]
    messages = []

    if strategy == "carry_forward":
        lookback = _SPIKE_LOOKBACK.get(frequency, 13)
        continuous = [c for c in generated if c not in flags]
        items = []
        for col in continuous:
            hist = numeric[col]
            usual = hist[-1 - lookback : -1]
            usual = usual[np.isfinite(usual)]
            last = hist[-1]
            if not usual.size or not np.isfinite(last):
                continue
            median = float(np.median(usual))
            if median > 0 and (last >= _SPIKE_HIGH * median or last <= _SPIKE_LOW * median):
                items.append(
                    f"'{col}' is {_num(last)}, {last / median:.1f}x its median over the "
                    f"previous {lookback} {unit}s ({_num(median)})"
                )
            elif median == 0 and last != 0:
                items.append(
                    f"'{col}' is {_num(last)}, while its median over the previous "
                    f"{lookback} {unit}s is 0"
                )
        on_last = [c for c in flags if numeric[c][-1] == 1]
        if continuous and on_last:
            items.append(
                f"{', '.join(repr(c) for c in on_last)} {'is' if len(on_last) == 1 else 'are'} "
                f"1, so it was an event {unit}"
            )
        if items:
            messages.append(
                f"The last {unit} of history ({hist_dates.iloc[-1]:%Y-%m-%d}) looks unusual, "
                f"and carry_forward holds its values for all {len(future_dates)} forecast "
                f"{unit}s: {'; '.join(items)}. If that {unit} was a promotion or an event, "
                f"supply planned values in future_independent_data, or use "
                f"future_exog_strategy='recent_average'."
            )

    tolerance = pd.Timedelta(days=_EVENT_TOLERANCE_DAYS)
    year = pd.Timedelta(days=364)
    matches = []
    for col in [c for c in flags if c in generated]:
        event_dates = hist_dates[numeric[col] == 1]
        found = []
        for when in event_dates:
            target = when + year
            gap = (future_dates - target).abs()
            if gap.min() <= tolerance:
                found.append(f"{when:%Y-%m-%d} (now {future_dates[gap.idxmin()]:%Y-%m-%d})")
        if found:
            more = f" and {len(found) - 3} more" if len(found) > 3 else ""
            matches.append(f"'{col}' was 1 on {', '.join(found[:3])}{more}")
    if matches:
        if strategy == "seasonal_naive":
            phase = "week of the year" if frequency == "weekly" else "day of the week"
            how = (
                f"seasonal_naive repeats past flags by {phase}, so an event whose date "
                f"moves, like Black Friday, can land in the wrong {unit}"
            )
        else:
            how = "they are set to 0 throughout the forecast, so these events are left out"
        messages.append(
            f"Event flags were generated, not supplied, for a forecast that covers the same "
            f"time of year as past events: {'; '.join(matches)}. Because they were generated, "
            f"{how}. Supply the dates of known events in future_independent_data."
        )
    return messages


# --------------------------------------------------------------------------
# checking a caller-supplied future table
# --------------------------------------------------------------------------

# What each model does with a future driver value outside the range it was
# trained on. They genuinely disagree, which is why an out-of-range scenario
# warns: the ensemble then blends very different answers.
_OUT_OF_RANGE_BEHAVIOUR = {
    "prophet": "Prophet extends the driver's effect in a straight line",
    "neuralprophet": "NeuralProphet extends the driver's effect in a straight line",
    "catboost": "CatBoost treats it like the most extreme value it has seen",
    "sarimax": "SARIMAX caps it at the historical range",
    "bayesianpymc": "Bayesian PyMC caps it at the historical range",
}

_PARAM = "future_independent_data"
_MAX_EXAMPLES = 3


def _fmt_dates(dates: pd.Series, mask: np.ndarray) -> str:
    picked = pd.Series(dates).reset_index(drop=True)[mask]
    shown = ", ".join(f"{d:%Y-%m-%d}" for d in picked.iloc[:_MAX_EXAMPLES])
    more = len(picked) - _MAX_EXAMPLES
    return shown + (f" and {more} more" if more > 0 else "")


def _same_dates(a: pd.Series, b: pd.Series) -> bool:
    a = pd.Series(a).reset_index(drop=True)
    b = pd.Series(b).reset_index(drop=True)
    return len(a) == len(b) and bool(
        np.array_equal(a.to_numpy("datetime64[ns]"), b.to_numpy("datetime64[ns]"))
    )


def _read_table_dates(
    values: pd.Series, label: str, expected: pd.Series, date_format: Optional[str]
) -> pd.Series:
    """
    Parse the future table's dates with the same rules as the history's.
    One extra allowance: a short future table can be ambiguous on its own
    (14 days that all fall on days 1-12 read equally well day-first or
    month-first), so when the normal parse fails, a day-first or
    month-first reading that reproduces the forecast dates exactly is
    accepted -- the expected dates settle the ambiguity.
    """
    try:
        return parse_date_column(values.reset_index(drop=True), label, date_format=date_format)
    except InvalidConfigError:
        if date_format is None and values.dtype == object:
            text = values.reset_index(drop=True).astype(str).str.strip()
            for fmt in _DAY_FIRST_FORMATS + _MONTH_FIRST_FORMATS:
                parsed = pd.to_datetime(text, format=fmt, errors="coerce")
                if parsed.notna().all() and _same_dates(parsed, expected):
                    return parsed
        raise


def _unknown_column_hint(col, declared: Sequence[str], history_columns) -> str:
    name = str(col)
    same_ignoring_case = [d for d in declared if str(d).lower() == name.lower()]
    close = same_ignoring_case or difflib.get_close_matches(
        name, [str(d) for d in declared], n=1, cutoff=0.8
    )
    if close:
        return f"'{name}' (did you mean '{close[0]}'?)"
    if col in history_columns:
        return (
            f"'{name}' (it is in your data, but not in independent_columns, so no model "
            f"uses it -- add it there if it should be a driver)"
        )
    return f"'{name}'"


def _skips_days(last_date, future_dates: pd.Series, frequency: str) -> bool:
    """True when daily forecast dates skip days: an operating calendar."""
    if frequency != "daily":
        return False
    dates = pd.to_datetime(pd.Series([last_date, *list(future_dates)]))
    return bool((dates.diff().dropna() > pd.Timedelta(days=1)).any())


def check_future_table(
    table,
    *,
    history_df: pd.DataFrame,
    date_column: str,
    target_column: str,
    independent_columns: List[str],
    declared_columns: List[str],
    event_columns: Optional[List[str]],
    future_dates: pd.Series,
    frequency: str,
    date_format: Optional[str] = None,
    models: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Check a caller-supplied future driver table before any model sees it.

    Every model reads this table by POSITION -- row i is forecast period i --
    and each model used to react to a bad table differently: too few rows
    made Prophet and SARIMAX fail while CatBoost carried on alone; a
    misspelled column was ignored and the real driver silently
    auto-generated; a blank made two models drop out; an infinite value gave
    Prophet a NaN forecast; a date column, if present, was never looked at.
    The run still finished, so the damage looked like a forecast. Here a
    table either matches the forecast exactly or is rejected with the reason.

    Errors (InvalidConfigError):
      - not a DataFrame;
      - a table supplied when no independent_columns were declared;
      - duplicate column names;
      - a column that is not a declared driver (the date and target columns
        are allowed and ignored);
      - a row count different from the horizon;
      - dates (a date column, or a DatetimeIndex) that are blank, out of
        order, or not exactly the forecast dates. Without dates the rows
        are taken in order;
      - a blank, non-numeric or infinite value in a driver column;
      - a negative value in a driver that is never negative in the history;
      - an event flag with a value it never takes in the history (a 0/1
        flag must stay 0/1). Booleans are read as 0/1.

    Warnings:
      - a supplied column for a declared driver that was dropped for never
        changing in the history (its future values cannot be used);
      - a driver value outside its historical range, naming what each
        requested model does with it.

    Returns the supplied, still-active driver columns as floats on a
    0..horizon-1 index, ready for resolve_future_exog(supplied=...).
    """
    if not isinstance(table, pd.DataFrame):
        raise InvalidConfigError(
            f"{_PARAM} must be a pandas DataFrame, got {type(table).__name__}. "
            f"Wrap it first, e.g. pd.DataFrame(...).",
            param_name=_PARAM,
            provided_value=type(table).__name__,
        )

    if not declared_columns:
        raise InvalidConfigError(
            f"{_PARAM} was supplied, but no independent_columns were declared when the "
            f"forecaster was created, so no model would use it. Declare the drivers in "
            f"independent_columns, or leave {_PARAM} out.",
            param_name=_PARAM,
            provided_value=list(table.columns),
        )

    columns = list(table.columns)
    duplicated = list(dict.fromkeys(c for c in columns if columns.count(c) > 1))
    if duplicated:
        raise InvalidConfigError(
            f"{_PARAM} has more than one column named {duplicated}. Each driver must "
            f"appear once.",
            param_name=_PARAM,
            provided_value=duplicated,
        )

    allowed = set(declared_columns) | {date_column, target_column}
    unknown = [c for c in columns if c not in allowed]
    if unknown:
        hints = [_unknown_column_hint(c, declared_columns, history_df.columns) for c in unknown]
        raise InvalidConfigError(
            f"{_PARAM} has column(s) that are not drivers of this forecaster: "
            f"{'; '.join(hints)}. The drivers are {list(declared_columns)}; the table may "
            f"also contain the date column '{date_column}'.",
            param_name=_PARAM,
            provided_value=[str(c) for c in unknown],
        )

    future_dates = pd.Series(future_dates).reset_index(drop=True)
    horizon = len(future_dates)
    span = f"{future_dates.iloc[0]:%Y-%m-%d} to {future_dates.iloc[-1]:%Y-%m-%d}"
    if len(table) != horizon:
        raise InvalidConfigError(
            f"{_PARAM} has {len(table)} row(s), but horizon={horizon}. It needs exactly one "
            f"row per forecast period, in order: {span}.",
            param_name=_PARAM,
            provided_value=len(table),
        )

    if date_column in table.columns:
        given_dates, label = table[date_column], date_column
        where = f"the date column '{date_column}'"
    elif isinstance(table.index, pd.DatetimeIndex):
        given_dates, label = pd.Series(table.index), f"index of {_PARAM}"
        where = f"the index of {_PARAM}"
    else:
        given_dates = None
    if given_dates is not None:
        given = _read_table_dates(given_dates, label, future_dates, date_format)
        if given.isna().any():
            raise InvalidConfigError(
                f"Some dates in {where} are blank (row(s) "
                f"{[int(i) + 1 for i in np.flatnonzero(given.isna())][:_MAX_EXAMPLES]}). "
                f"Every row needs its date: {span}.",
                param_name=_PARAM,
            )
        if not _same_dates(given, future_dates):
            if _same_dates(given.sort_values(), future_dates):
                raise InvalidConfigError(
                    f"The dates in {where} are the right dates but not in order. Sort the "
                    f"table by date first: the rows are matched to forecast periods by "
                    f"position.",
                    param_name=_PARAM,
                )
            last = history_df[date_column].max()
            row = int(
                np.flatnonzero(
                    given.to_numpy("datetime64[ns]") != future_dates.to_numpy("datetime64[ns]")
                )[0]
            )
            skipping = (
                " and skipping the closed days"
                if _skips_days(last, future_dates, frequency)
                else ""
            )
            raise InvalidConfigError(
                f"Row {row + 1} of {_PARAM} is dated {given.iloc[row]:%Y-%m-%d}, but forecast "
                f"period {row + 1} is {future_dates.iloc[row]:%Y-%m-%d}. The forecast runs "
                f"{span}, starting the period after the last date in your data "
                f"({last:%Y-%m-%d}){skipping}.",
                param_name=_PARAM,
                provided_value=str(given.iloc[row]),
            )

    supplied = [c for c in columns if c in independent_columns]
    ignored = [c for c in columns if c in declared_columns and c not in independent_columns]

    # Text first, with the same reading as the history: on/off words become
    # 1/0, numbers written as text become numbers, and other text gets a
    # specific error per column.
    parsed: Dict[str, pd.Series] = {}
    text_errors: List[str] = []
    for col in supplied:
        try:
            parsed[col] = parse_driver_column(table[col], str(col), dates=future_dates)
        except InvalidConfigError as e:
            text_errors.append(e.message)
    if text_errors:
        raise InvalidConfigError(
            f"{_PARAM} has text the models cannot use. " + " ".join(text_errors),
            param_name=_PARAM,
        )

    cleaned: Dict[str, pd.Series] = {}
    problems: List[str] = []
    out_of_range: List[str] = []
    for col in supplied:
        values = pd.to_numeric(parsed[col], errors="coerce").reset_index(drop=True)
        blank = values.isna().to_numpy()
        infinite = np.isinf(values.to_numpy(dtype=float))
        if blank.any():
            problems.append(f"'{col}' is blank for {_fmt_dates(future_dates, blank)}")
        if infinite.any():
            problems.append(f"'{col}' is infinite for {_fmt_dates(future_dates, infinite)}")
        if blank.any() or infinite.any():
            continue

        hist = pd.to_numeric(history_df[col], errors="coerce").to_numpy(dtype=float)
        hist = hist[np.isfinite(hist)]
        vals = values.to_numpy(dtype=float)
        negative = vals < 0
        if hist.size and hist.min() >= 0 and negative.any():
            problems.append(
                f"'{col}' is negative for {_fmt_dates(future_dates, negative)}, but it is "
                f"never negative in your data"
            )
            continue

        declared_flag = event_columns is not None and col in event_columns
        detected_flag = event_columns is None and hist.size and set(np.unique(hist)) <= {0, 1}
        if declared_flag or detected_flag:
            seen = np.unique(hist)
            unseen = ~np.isin(vals, seen)
            if unseen.any():
                shown = " and ".join(f"{v:g}" for v in seen)
                problems.append(
                    f"'{col}' is an on/off flag that is only ever {shown} in your data, but "
                    f"it is {list(dict.fromkeys(f'{v:g}' for v in vals[unseen]))} for "
                    f"{_fmt_dates(future_dates, unseen)}"
                )
                continue
        elif hist.size and (vals.max() > hist.max() or vals.min() < hist.min()):
            out_of_range.append(
                f"'{col}' goes from {vals.min():g} to {vals.max():g}; your data has "
                f"{hist.min():g} to {hist.max():g}"
            )
        cleaned[col] = pd.Series(vals)

    if problems:
        raise InvalidConfigError(
            f"{_PARAM} has values the models cannot use: " + "; ".join(problems) + ".",
            param_name=_PARAM,
        )

    if ignored:
        warnings.warn(
            f"{_PARAM} includes {ignored}, but these drivers never change in your data, so "
            f"they were dropped when the forecaster was created and their future values "
            f"are ignored. A driver that is constant in the past cannot be learned, "
            f"whatever its future values.",
            stacklevel=3,
        )

    if out_of_range:
        behaviour = [_OUT_OF_RANGE_BEHAVIOUR[m] for m in models if m in _OUT_OF_RANGE_BEHAVIOUR]
        how = f" ({'; '.join(behaviour)})" if behaviour else ""
        warnings.warn(
            f"Some future driver values are outside anything in your data: "
            f"{'; '.join(out_of_range)}. The models disagree about values they have never "
            f"seen{how}, so treat this forecast as a scenario and compare the models "
            f"before relying on the ensemble.",
            stacklevel=3,
        )

    return pd.DataFrame(cleaned, index=pd.RangeIndex(horizon))
