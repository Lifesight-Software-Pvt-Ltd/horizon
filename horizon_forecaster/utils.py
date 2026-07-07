"""
Generic helper utilities used across horizon_forecaster.

These functions are intentionally free of any model-specific logic —
they operate purely on pandas DataFrames/Series and raw config values.
"""

from typing import List, Optional, Tuple

import pandas as pd

from horizon_forecaster.exceptions import (
    InvalidConfigError,
    UnsupportedFrequencyError,
)

SUPPORTED_FREQUENCIES = ("daily", "weekly")

def clean_zero_variance(
    df: pd.DataFrame, exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns whose values are constant across the entire dataset
    (zero variance). These carry no predictive signal and can cause
    numerical issues in some models (e.g. singular exog matrices in
    SARIMAX).

    Args:
        df: input DataFrame (e.g. independent variables + date + target).
        exclude_columns: columns to never drop regardless of variance
            (typically the date column and the target column).

    Returns:
        (cleaned_df, dropped_column_names)
    """
    exclude_columns = exclude_columns or []
    dropped = []
    cleaned = df.copy()

    for col in df.columns:
        if col in exclude_columns:
            continue
        # nunique(dropna=False) so an all-NaN column also counts as zero-variance
        if cleaned[col].nunique(dropna=False) <= 1:
            dropped.append(col)

    cleaned = cleaned.drop(columns=dropped)
    return cleaned, dropped

def infer_frequency(date_series: pd.Series) -> str:
    """
    Infer whether a date series represents daily or weekly granularity,
    based on the most common gap between consecutive sorted dates.

    Raises:
        UnsupportedFrequencyError: if the inferred gap doesn't cleanly
            match daily (1 day) or weekly (7 days) spacing.
    """
    sorted_dates = pd.to_datetime(date_series).sort_values().drop_duplicates()

    if len(sorted_dates) < 2:
        raise InvalidConfigError(
            "Need at least 2 distinct dates to infer frequency.",
            param_name="date_column",
            provided_value=len(sorted_dates),
        )

    diffs = sorted_dates.diff().dropna()
    most_common_gap = diffs.mode()[0]

    if most_common_gap == pd.Timedelta(days=1):
        return "daily"
    elif most_common_gap == pd.Timedelta(days=7):
        return "weekly"
    else:
        raise UnsupportedFrequencyError(
            f"Detected date spacing of {most_common_gap}, which does not match "
            f"a supported frequency ({', '.join(SUPPORTED_FREQUENCIES)}).",
            provided_frequency=str(most_common_gap),
        )

def validate_frequency(freq: str) -> str:
    """
    Validate a user-declared frequency string against supported values.
    Case-insensitive; returns the normalized lowercase value.
    """
    normalized = freq.strip().lower()
    if normalized not in SUPPORTED_FREQUENCIES:
        raise UnsupportedFrequencyError(
            f"Frequency '{freq}' is not supported. "
            f"Supported values: {', '.join(SUPPORTED_FREQUENCIES)}.",
            provided_frequency=freq,
        )
    return normalized

def validate_training_size(training_size: float) -> float:
    """
    Validate the training_size percentage against the allowed 50-100 range
    (per project spec: anything outside this range should raise/warn).

    Accepts either a fraction (0.5-1.0) or a percentage (50-100) and
    normalizes to a fraction (0.5-1.0) for internal use.
    """
    value = training_size

    # Normalize: if given as a percentage (e.g. 80), convert to fraction (0.8)
    if value > 1:
        value = value / 100.0

    if not (0.5 <= value <= 1.0):
        raise InvalidConfigError(
            f"training_size must be between 50% and 100% (got {training_size}).",
            param_name="training_size",
            provided_value=training_size,
            valid_range=(50, 100),
        )
    return value

def validate_mape_threshold_range(min_mape: float, max_mape: float) -> Tuple[float, float]:
    """
    Validate the ensemble MAPE threshold range against the allowed 1-100
    range (per project spec).
    """
    for name, value in (("min_mape", min_mape), ("max_mape", max_mape)):
        if not (1.0 <= value <= 100.0):
            raise InvalidConfigError(
                f"{name} must be between 1 and 100 (got {value}).",
                param_name=name,
                provided_value=value,
                valid_range=(1, 100),
            )

    if min_mape >= max_mape:
        raise InvalidConfigError(
            f"min_mape ({min_mape}) must be less than max_mape ({max_mape}).",
            param_name="min_mape/max_mape",
            provided_value=(min_mape, max_mape),
        )

    return min_mape, max_mape