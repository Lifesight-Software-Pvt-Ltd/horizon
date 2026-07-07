"""
Evaluation metrics for horizon_forecaster.

MAPE is the canonical metric per project spec — every model is evaluated
and every ensemble threshold decision is made using MAPE. WAPE and RMSE
are provided as supplementary metrics only (e.g. for future reporting),
never used for threshold/ensemble decisions.

All functions are pure: they take numpy-array-like actuals/predictions
and return plain floats (or a dict with metadata for MAPE), with no
dependency on any model class.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

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
    Mean Absolute Percentage Error, with explicit handling for zero (or
    near-zero) actuals — a common failure mode for MAPE on sparse daily
    data (e.g. days with 0 installs/orders).

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
        ValueError: if y_true and y_pred have different lengths, or if
            zero_handling is not a recognized strategy.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} vs {y_pred.shape}."
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
        raise ValueError(
            f"Unknown zero_handling strategy: '{zero_handling}'. "
            f"Expected 'exclude' or 'epsilon'."
        )

def wape(y_true, y_pred) -> float:
    """
    Weighted Absolute Percentage Error: sum(|actual - pred|) / sum(|actual|).

    Unlike MAPE, this does not blow up on individual zero-actual rows,
    since the denominator is a sum over the whole series rather than a
    per-row division. Provided as a supplementary metric only (not used
    for threshold/ensemble decisions per project spec).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} vs {y_pred.shape}."
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
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} vs {y_pred.shape}."
        )

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred) -> float:
    """
    Mean Absolute Error. Supplementary metric only.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} vs {y_pred.shape}."
        )

    return float(np.mean(np.abs(y_true - y_pred)))