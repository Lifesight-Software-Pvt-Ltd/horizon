"""
Calibrated 95% bands for every model and ensemble rule.

Every forecast HorizonForecaster returns, from any model or ensemble rule,
carries the same kind of band: one built from that forecast's own errors on
the validation window (split conformal prediction). The models' own
intervals are not used. They mean different things (Prophet's trend
simulation, SARIMAX's state-space variance, PyMC's posterior), CatBoost,
NeuralProphet and the ensembles have none, and the ones that exist cover
too little: on 80 test windows, Prophet's own band set to 95% held 86% of
actuals over the first 28 days and SARIMAX's 76%, against 91% and 87% for
these bands.

The band, for a forecast yhat_h made h periods after the last fitted date:

    errors      e_i = |y_i - yhat_i| over the L validation periods, leaving
                out unexpected zero days (outages that would inflate the
                width for every other day)
    width       q = the k-th smallest of the n errors kept,
                k = ceil(0.95 * (n + 1))
    band        yhat_h -/+ q * sqrt(max(1, h / L)), lower edge floored at 0
                when the target is never negative

The "+1" in k is what makes this conformal rather than a plain percentile:
if future errors behave like the validation errors, a new error is at most
q with at least 95% probability. It also sets the minimum: k <= n first
holds at n = 19, so with fewer errors there is no band.

The validation errors are slightly optimistic, though: they come from the
grid candidate chosen because its validation errors were smallest. In the
study, the grid-searched models covered less over the first four weeks
(Prophet 91%, SARIMAX 87%) than the two run with fixed settings (CatBoost
and NeuralProphet 93%), consistent with this. CatBoost's default
18-candidate grid was not part of the study, so its coverage in a default
run may be a little lower than measured.

The width is flat for as far ahead as the validation window reaches, which
is all the errors describe, and then grows with the square root of the
distance, as a random-walk forecast's error does. Without the widening,
daily coverage fell to 69-77% between 3 and 16 months ahead; with it,
82-88%.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

BAND_LEVEL = 0.95

# The fewest validation errors a 95% band can be calibrated from:
# ceil(0.95 * (n + 1)) <= n first holds at n = 19.
MIN_CALIBRATION_ERRORS = 19


@dataclass
class BandCalibration:
    """
    How one model's or ensemble rule's band was calibrated.

    Attributes:
        width: q, the band's half-width for the first `validation_length`
            periods ahead; None when there is no band (see `reason`).
        n_errors: the validation errors the width was taken from.
        n_zero_days_left_out: unexpected zero days left out of those errors.
        validation_length: L, the validation window's length in periods;
            the band widens past this many periods ahead.
        reason: why there is no band, when `width` is None.
    """

    width: Optional[float]
    n_errors: int
    n_zero_days_left_out: int
    validation_length: int
    reason: Optional[str] = None


def conformal_width(errors: np.ndarray, level: float = BAND_LEVEL) -> Optional[float]:
    """
    The split-conformal half-width: the ceil(level * (n + 1))-th smallest
    absolute error, or None when there are too few errors for that rank.
    """
    scores = np.sort(np.abs(np.asarray(errors, dtype=float)))
    scores = scores[np.isfinite(scores)]
    rank = int(np.ceil(level * (len(scores) + 1)))
    if len(scores) == 0 or rank > len(scores):
        return None
    return float(scores[rank - 1])


def calibrate(
    actuals: np.ndarray,
    forecasts: np.ndarray,
    zero_days: Optional[np.ndarray] = None,
) -> BandCalibration:
    """
    Calibrate a band from validation actuals and forecasts.

    Args:
        actuals, forecasts: the validation window, same length L.
        zero_days: True where the actual is an unexpected zero (an outage);
            those days are left out of the errors.
    """
    y = np.asarray(actuals, dtype=float)
    yhat = np.asarray(forecasts, dtype=float)
    keep = np.ones(len(y), dtype=bool) if zero_days is None else ~np.asarray(zero_days, bool)
    n = int(keep.sum())
    left_out = int(len(y) - n)
    if n < MIN_CALIBRATION_ERRORS:
        return BandCalibration(
            width=None,
            n_errors=n,
            n_zero_days_left_out=left_out,
            validation_length=len(y),
            reason=(
                f"{n} validation errors, fewer than the {MIN_CALIBRATION_ERRORS} a 95% band "
                f"needs"
            ),
        )
    return BandCalibration(
        width=conformal_width(y[keep] - yhat[keep]),
        n_errors=n,
        n_zero_days_left_out=left_out,
        validation_length=len(y),
    )


def widening(steps_ahead: np.ndarray, validation_length: int) -> np.ndarray:
    """sqrt(max(1, h / L)): 1 up to the validation length, then square-root growth."""
    h = np.asarray(steps_ahead, dtype=float)
    return np.sqrt(np.maximum(1.0, h / max(int(validation_length), 1)))


def band(
    forecasts: np.ndarray, calibration: BandCalibration, floor_at_zero: bool
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    (lower, upper) around forecasts made 1, 2, ... periods after the last
    fitted date, or None when the calibration has no width.
    """
    if calibration.width is None:
        return None
    yhat = np.asarray(forecasts, dtype=float)
    half = calibration.width * widening(np.arange(1, len(yhat) + 1), calibration.validation_length)
    lower, upper = yhat - half, yhat + half
    if floor_at_zero:
        lower = np.maximum(lower, 0.0)
    return lower, upper


def coverage(actuals, lower, upper) -> Tuple[float, int]:
    """(percent of actuals inside [lower, upper], number of points)."""
    y = np.asarray(actuals, dtype=float)
    inside = (y >= np.asarray(lower, dtype=float)) & (y <= np.asarray(upper, dtype=float))
    return (float(inside.mean() * 100) if len(y) else float("nan")), int(len(y))
