"""
Evaluation metrics for horizon_forecaster.

MAPE is the default selection metric: the grid search, the mape_threshold
screen, the inverse-MAPE weights and the exported best model are decided
on it. WAPE is reported next to it everywhere and replaces it for those
decisions when the forecaster's selection_metric resolves to "wape" (a
target with many zeros, or one that goes negative). RMSE and MAE are
supplementary only.

The scores are pure functions: they take numpy-array-like actuals and
predictions and return plain floats -- or, for MAPE, a MapeResult
dataclass carrying the zero-actual exclusion metadata alongside the
value -- with no dependency on any model class. seasonal_naive() and
error_correlation() work on dates, so they take pandas objects.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from horizon_forecaster.exceptions import InvalidConfigError


@dataclass
class MapeResult:
    """
    Structured MAPE result. Returning a dataclass instead of a bare float
    lets callers see how many rows were excluded due to zero-actuals,
    so a warning can be surfaced when too much of the series was skipped.
    """

    value: float
    excluded_zero_actual_count: int
    total_count: int

    @property
    def excluded_fraction(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.excluded_zero_actual_count / self.total_count

    def __float__(self) -> float:
        return self.value


def mape(
    y_true,
    y_pred,
    zero_handling: str = "exclude",
    epsilon: float = 1e-9,
) -> MapeResult:
    """
    Mean Absolute Percentage Error, with explicit handling for zero
    actuals — a common failure mode for MAPE on sparse daily data (e.g.
    days with 0 installs/orders). The test is for an exact zero, not a
    tolerance: a very small non-zero actual is kept and divided by, and
    will produce a correspondingly enormous percentage error.

    Args:
        y_true: array-like of actual values.
        y_pred: array-like of predicted values, same length as y_true.
        zero_handling: strategy for rows where y_true == 0:
            - "exclude" (default): drop these rows entirely from the MAPE
              calculation, and report how many were dropped.
            - "epsilon": replace zero actuals with `epsilon` before
              dividing, keeping all rows but heavily inflating their
              individual percentage error (use with caution).
        epsilon: only used when zero_handling="epsilon".

    Returns:
        MapeResult with .value (the MAPE, as a percentage, e.g. 12.5 for
        12.5%), .excluded_zero_actual_count, and .total_count.

    Raises:
        InvalidConfigError (a ValueError): if y_true and y_pred have different lengths, or if
            zero_handling is not a recognized strategy.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise InvalidConfigError(
            f"y_true and y_pred must have the same shape. " f"Got {y_true.shape} vs {y_pred.shape}."
        )

    total_count = len(y_true)

    if zero_handling == "exclude":
        nonzero_mask = y_true != 0
        excluded_count = total_count - int(nonzero_mask.sum())

        if nonzero_mask.sum() == 0:
            # every single actual was zero — MAPE is undefined
            return MapeResult(
                value=float("nan"),
                excluded_zero_actual_count=excluded_count,
                total_count=total_count,
            )

        y_true_used = y_true[nonzero_mask]
        y_pred_used = y_pred[nonzero_mask]
        pct_errors = np.abs((y_true_used - y_pred_used) / y_true_used)
        value = float(np.mean(pct_errors) * 100)
        return MapeResult(
            value=value,
            excluded_zero_actual_count=excluded_count,
            total_count=total_count,
        )

    elif zero_handling == "epsilon":
        safe_y_true = np.where(y_true == 0, epsilon, y_true)
        pct_errors = np.abs((y_true - y_pred) / safe_y_true)
        value = float(np.mean(pct_errors) * 100)
        return MapeResult(
            value=value,
            excluded_zero_actual_count=0,
            total_count=total_count,
        )

    else:
        raise InvalidConfigError(
            f"Unknown zero_handling strategy: '{zero_handling}'. "
            f"Expected 'exclude' or 'epsilon'."
        )


def wape(y_true, y_pred) -> float:
    """
    Weighted Absolute Percentage Error: sum(|actual - pred|) / sum(|actual|).

    Unlike MAPE, this does not blow up on individual zero-actual rows,
    since the denominator is a sum over the whole series rather than a
    per-row division. It replaces MAPE for selection, screening and
    weighting when HorizonForecaster's selection_metric resolves to
    'wape', and is reported alongside MAPE either way.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise InvalidConfigError(
            f"y_true and y_pred must have the same shape. " f"Got {y_true.shape} vs {y_pred.shape}."
        )

    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return float("nan")

    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100)


def rmse(y_true, y_pred) -> float:
    """
    Root Mean Squared Error. Supplementary metric only.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise InvalidConfigError(
            f"y_true and y_pred must have the same shape. " f"Got {y_true.shape} vs {y_pred.shape}."
        )

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """
    Mean Absolute Error. Supplementary metric only.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise InvalidConfigError(
            f"y_true and y_pred must have the same shape. " f"Got {y_true.shape} vs {y_pred.shape}."
        )

    return float(np.mean(np.abs(y_true - y_pred)))


def seasonal_naive(history_dates, history_values, future_dates, season_days: int):
    """
    The seasonal-naive forecast: each future date repeats the most recent
    value from a whole number of seasons earlier -- the same weekday last
    week for season_days=7, the same week last year for season_days=364. A
    model that cannot beat this adds nothing, which makes it the baseline
    every model and ensemble is compared with.

    Matched by date, not by row, so if the dates passed in have gaps it
    still compares Monday with Monday. (HorizonForecaster rejects missing
    dates, so its own baseline is never computed on gappy data.) A future
    date with no such value in the history (less than one season of it)
    repeats the last value instead.
    """
    hist = pd.Series(
        np.asarray(history_values, dtype=float), index=pd.to_datetime(pd.Index(history_dates))
    ).sort_index()
    if hist.empty:
        raise InvalidConfigError("seasonal_naive needs at least one past value.")
    if season_days < 1:
        raise InvalidConfigError(f"seasonal_naive needs season_days >= 1 (got {season_days}).")
    # Dates a whole number of seasons apart share the same day number modulo
    # the season; the last value seen for each one is "a season earlier".
    latest = hist.groupby(_day_number(hist.index) % season_days).last()
    future = pd.to_datetime(pd.Index(future_dates))
    found = latest.reindex(_day_number(future) % season_days).to_numpy()
    return np.where(np.isnan(found), hist.iloc[-1], found)


def _day_number(dates: "pd.DatetimeIndex") -> np.ndarray:
    """Whole days since the epoch, for comparing dates a season apart."""
    return (dates.normalize() - pd.Timestamp(0)).days.to_numpy()


def error_correlation(evaluation_results) -> "pd.DataFrame":
    """
    Pairwise correlation of the models' signed errors (actual - forecast) on
    the test window, from the dict evaluate() returns.

    An ensemble only gains from members that are wrong in different ways.
    Two models whose errors correlate near 1 are making the same mistakes,
    so averaging them adds little; a member with low or negative correlation
    to the rest is the one that actually diversifies the combination. On a
    real daily series Prophet and NeuralProphet correlated at 1.00 while
    SARIMAX sat at 0.73 against both.

    Returns a symmetric DataFrame indexed and columned by model name (empty
    when fewer than two models are given). A model whose errors never vary
    has an undefined correlation (NaN).
    """
    errors = {}
    for name, result in evaluation_results.items():
        tp = result.test_predictions
        errors[name] = pd.Series(
            tp["y"].to_numpy(dtype=float) - tp["yhat"].to_numpy(dtype=float),
            index=pd.to_datetime(tp["ds"]),
        )
    if len(errors) < 2:
        return pd.DataFrame()
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame(errors).corr()
