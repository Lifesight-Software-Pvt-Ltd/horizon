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


def validate_no_missing_values(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    independent_columns: Optional[List[str]] = None,
) -> None:
    """
    Hard-fail if ANY cell of the data used for training/testing is missing.

    Every value in the date column, the target column, and each independent
    column that is actually used must be present -- a single NaN, NaT, or
    non-numeric entry raises InvalidConfigError. Missing data is NOT dropped or
    imputed: forecasting requires complete input, so the caller must supply it
    (project convention -- fail loudly rather than silently forecasting on a
    quietly-shrunk or gap-filled series).

    Columns present in the DataFrame but not used (i.e. not the date, target,
    or a surviving independent column) are ignored.
    """
    independent_columns = independent_columns or []

    # Column existence first, so a missing/misspelled column is a clear error
    # rather than a raw KeyError from the value checks below.
    missing_columns = [
        c for c in [date_column, target_column, *independent_columns]
        if c not in df.columns
    ]
    if missing_columns:
        raise InvalidConfigError(
            f"Column(s) not found in the data: {missing_columns}. "
            f"Available columns: {list(df.columns)}.",
            param_name="columns",
            provided_value=missing_columns,
        )

    problems = {}

    n_bad_dates = int(pd.to_datetime(df[date_column], errors="coerce").isna().sum())
    if n_bad_dates:
        problems[date_column] = n_bad_dates

    for col in [target_column, *independent_columns]:
        n_missing = int(pd.to_numeric(df[col], errors="coerce").isna().sum())
        if n_missing:
            problems[col] = n_missing

    if problems:
        raise InvalidConfigError(
            f"The data has missing or non-numeric values in column(s) used for "
            f"training/testing: {problems}. Every cell of the date, target and "
            f"independent columns must be present -- clean or impute the data "
            f"before forecasting (missing values are not dropped automatically).",
            param_name="raw_df",
            provided_value=problems,
        )

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
        InvalidConfigError: if fewer than 2 distinct dates are provided
            (too few to infer any spacing).
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

# The user-facing mape_threshold is constrained to this range. 0.01 and
# 100 are fixed internal bounds, NOT user-settable: a floor of 0.01%
# rejects a threshold of 0 (and negatives), which would admit only a
# literally perfect model and is almost never what's intended, while
# leaving every realistic low threshold valid; a MAPE above 100% already
# signals a degenerate fit, so a higher ceiling would add nothing useful.
MAPE_THRESHOLD_MIN = 0.01
MAPE_THRESHOLD_MAX = 100.0

def validate_mape_threshold(mape_threshold: float) -> float:
    """
    Validate the ensemble `mape_threshold` against the allowed
    [MAPE_THRESHOLD_MIN, MAPE_THRESHOLD_MAX] (0.01-100) range.

    `mape_threshold` is an upper cutoff: a model is included in the
    ensemble only if its MAPE is at or below it. The bound itself must be
    a sensible percentage within [0.01, 100].
    """
    if not (MAPE_THRESHOLD_MIN <= mape_threshold <= MAPE_THRESHOLD_MAX):
        raise InvalidConfigError(
            f"mape_threshold must be between {MAPE_THRESHOLD_MIN:g} and "
            f"{MAPE_THRESHOLD_MAX:g} (got {mape_threshold}).",
            param_name="mape_threshold",
            provided_value=mape_threshold,
            valid_range=(MAPE_THRESHOLD_MIN, MAPE_THRESHOLD_MAX),
        )
    return mape_threshold