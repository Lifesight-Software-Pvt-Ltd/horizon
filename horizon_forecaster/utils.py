"""
Generic helper utilities used across horizon_forecaster.

These functions are intentionally free of any model-specific logic —
they operate purely on pandas DataFrames/Series and raw config values.
"""

import warnings
from typing import List, NamedTuple, Optional, Tuple, Union

import pandas as pd

from horizon_forecaster.exceptions import (
    InvalidConfigError,
    UnsupportedFrequencyError,
)

SUPPORTED_FREQUENCIES = ("daily", "weekly")

# Default validation/test window sizes, in PERIODS, per frequency.
#
# Deliberately absolute rather than a percentage of history. A percentage
# makes both windows scale with however much history the caller happens to
# have, which is the wrong driver for either of them:
#
#   - The TEST window is an estimate of the error you will experience when
#     you actually forecast. It should therefore be about the length of a
#     horizon somebody would really forecast and act on -- 4 weeks of daily
#     data, a quarter of weekly data. At 15% of a 1,095-row daily series a
#     percentage rule instead produces a 164-day test window: nobody
#     forecasts 164 days and trusts it, the score is dominated by
#     far-horizon degradation, and it is not comparable to the same 15% of
#     a 120-row weekly series.
#
#   - The VALIDATION window is a selection instrument. It has to rank 3-18
#     grid candidates against each other, so it wants stability, which
#     means more data. Hence it defaults to TWICE the test window rather
#     than the same size -- the two windows measure different things and
#     there is no reason for them to match.
#
# These are Horizon's own defaults, chosen for users -- deliberately NOT
# the window fractions used by any particular evaluation corpus.
DEFAULT_EVALUATION_WINDOWS = {
    "daily": {"validation": 56, "test": 28},   # 8 weeks / 4 weeks
    "weekly": {"validation": 26, "test": 13},  # ~6 months / ~1 quarter
}

# When the caller takes the percentage route (`training_size`), whatever is
# left after training has to be divided between validation and test. It is
# split in this ratio, for the same reason validation defaults to twice
# test above: selection wants stability, reporting wants a realistic
# horizon.
_PERCENTAGE_ROUTE_VALIDATION_SHARE = 2 / 3

# The most of a dataset the DEFAULT windows are ever allowed to consume
# between them. On a long series the absolute defaults are already a small
# fraction and this never binds (it starts binding below ~210 daily rows,
# ~98 weekly). On a short one it stops the defaults from eating the
# training data alive: 56 + 28 out of 60 rows is not a split, it's an
# error message.
#
# Applies to DEFAULTED windows only. A size the caller asked for
# explicitly is honoured exactly, or raises -- silently shrinking a
# window somebody deliberately chose would be worse than refusing.
_MAX_DEFAULTED_EVALUATION_SHARE = 0.4


class SplitSizes(NamedTuple):
    """Row counts for the three chronological windows, in order."""

    n_train: int
    n_validation: int
    n_test: int


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


def _resolve_window(
    value: Union[int, float, None], n_total: int, param_name: str
) -> Optional[int]:
    """
    Turn one window size into a row count.

    An `int` is a number of PERIODS; a `float` strictly between 0 and 1 is
    a FRACTION of the dataset. `None` means "use the frequency default"
    and is returned as-is for the caller to substitute.

    A float of 1.0 or more is rejected rather than guessed at: `28` and
    `28.0` would otherwise mean the same thing while `0.28` meant
    something completely different, which is exactly the kind of silent
    unit confusion that produces a wrong split nobody notices.
    """
    if value is None:
        return None

    # bool is a subclass of int in Python, so True would otherwise sail
    # through as "1 period".
    if isinstance(value, bool):
        raise InvalidConfigError(
            f"{param_name} must be an int (periods) or a float between 0 and 1 "
            f"(a fraction of the data), got a bool.",
            param_name=param_name,
            provided_value=value,
        )

    if isinstance(value, int):
        if value < 1:
            raise InvalidConfigError(
                f"{param_name} must be at least 1 period (got {value}).",
                param_name=param_name,
                provided_value=value,
            )
        return value

    if isinstance(value, float):
        if not (0 < value < 1):
            raise InvalidConfigError(
                f"{param_name} was given as the float {value}, which is ambiguous. "
                f"Use an int for a number of periods (e.g. {param_name}=28) or a "
                f"float strictly between 0 and 1 for a fraction of the data "
                f"(e.g. {param_name}=0.15).",
                param_name=param_name,
                provided_value=value,
            )
        return max(1, round(value * n_total))

    raise InvalidConfigError(
        f"{param_name} must be an int (periods), a float between 0 and 1 "
        f"(a fraction), or None (the frequency default). Got {type(value).__name__}.",
        param_name=param_name,
        provided_value=value,
    )


def resolve_split(
    n_total: int,
    frequency: str,
    training_size: Union[int, float, None] = None,
    validation_size: Union[int, float, None] = None,
    test_size: Union[int, float, None] = None,
) -> SplitSizes:
    """
    Work out the train / validation / test row counts for a dataset.

    There are two mutually exclusive routes into this, because mixing them
    creates an ambiguity that cannot be resolved safely:

    1. **The window route** (the default). Give `validation_size` and/or
       `test_size` as periods or fractions; anything omitted falls back to
       the frequency default in DEFAULT_EVALUATION_WINDOWS. Training takes
       whatever is left. This is the recommended route -- see that
       constant's comment for why the windows are absolute.

    2. **The percentage route.** Give `training_size` as a percentage
       (50-100) or fraction (0.5-1.0), exactly as it has always meant, and
       the remainder is divided between validation and test in the
       _PERCENTAGE_ROUTE_VALIDATION_SHARE ratio.

    Passing `training_size` *and* either window size raises, rather than
    silently letting one win. Note also that `training_size` keeps its
    original percentage-only meaning: `training_size=80` is 80%, never 80
    periods. The new window parameters are the ones that accept periods,
    so no existing call quietly changes meaning.

    Returns:
        SplitSizes(n_train, n_validation, n_test).

    Raises:
        InvalidConfigError: if both routes are used at once, if a size is
            malformed, or if the requested windows do not leave at least
            one row in every window.
    """
    explicit_windows = [
        name
        for name, value in (("validation_size", validation_size), ("test_size", test_size))
        if value is not None
    ]

    if training_size is not None and explicit_windows:
        raise InvalidConfigError(
            f"training_size cannot be combined with {' or '.join(explicit_windows)}. "
            f"Either give training_size on its own (the percentage route -- the "
            f"remainder is split between validation and test automatically), or give "
            f"validation_size/test_size and let training take what is left.",
            param_name="training_size",
            provided_value=training_size,
        )

    if training_size is not None:
        training_fraction = validate_training_size(training_size)
        n_train = int(n_total * training_fraction)
        remainder = n_total - n_train
        if remainder < 2:
            raise InvalidConfigError(
                f"training_size of {training_fraction * 100:.0f}% leaves {remainder} "
                f"row(s) out of {n_total} for evaluation, but both a validation and a "
                f"test window are needed (at least 2 rows). Use a smaller "
                f"training_size, or a larger dataset.",
                param_name="training_size",
                provided_value=training_size,
                valid_range=(50, 100),
            )
        n_validation = max(1, round(remainder * _PERCENTAGE_ROUTE_VALIDATION_SHARE))
        n_validation = min(n_validation, remainder - 1)  # always leave a test row
        n_test = remainder - n_validation
        return SplitSizes(n_train, n_validation, n_test)

    defaults = DEFAULT_EVALUATION_WINDOWS[frequency]
    n_validation = _resolve_window(validation_size, n_total, "validation_size")
    n_test = _resolve_window(test_size, n_total, "test_size")

    both_defaulted = n_validation is None and n_test is None
    if n_validation is None:
        n_validation = defaults["validation"]
    if n_test is None:
        n_test = defaults["test"]

    # Short-series fallback: scale the defaults down proportionally rather
    # than refusing to split at all. Only when BOTH were defaulted -- if
    # the caller named either window, the pair they asked for is what they
    # get. The 2:1 validation:test ratio is preserved through the scaling,
    # since the reasoning behind it doesn't change with dataset size.
    budget = int(n_total * _MAX_DEFAULTED_EVALUATION_SHARE)
    if both_defaulted and (n_validation + n_test) > budget:
        original = (n_validation, n_test)
        n_validation = max(1, round(budget * _PERCENTAGE_ROUTE_VALIDATION_SHARE))
        n_test = max(1, budget - n_validation)
        warnings.warn(
            f"{n_total} rows is too short for the default {frequency} evaluation "
            f"windows ({original[0]} validation + {original[1]} test), which would "
            f"leave {n_total - sum(original)} rows to train on. Scaled them down to "
            f"{n_validation} validation + {n_test} test. Scores from windows this "
            f"small are noisy -- treat them as indicative, and note they are not "
            f"comparable to scores from a longer series, which uses the full "
            f"windows. Pass validation_size/test_size explicitly to choose your own.",
            stacklevel=3,
        )

    n_train = n_total - n_validation - n_test
    if n_train < 1:
        used_defaults = [
            name
            for name, value in (("validation_size", validation_size), ("test_size", test_size))
            if value is None
        ]
        default_note = (
            f" ({' and '.join(used_defaults)} defaulted to the {frequency} window "
            f"size; pass them explicitly to use smaller windows, or use training_size "
            f"for a percentage split)"
            if used_defaults
            else ""
        )
        raise InvalidConfigError(
            f"The requested evaluation windows do not leave any rows for training: "
            f"{n_total} total rows - {n_validation} validation - {n_test} test = "
            f"{n_train}{default_note}.",
            param_name="validation_size/test_size",
            provided_value=(validation_size, test_size),
        )

    return SplitSizes(n_train, n_validation, n_test)

# The user-facing mape_threshold is constrained to this range. 0.01 and
# 100 are fixed internal bounds, NOT user-settable: a floor of 0.01%
# rejects a threshold of 0 (and negatives), which would admit only a
# literally perfect model and is almost never what's intended, while
# leaving every realistic low threshold valid; a MAPE above 100% already
# signals a degenerate fit, so a higher ceiling would add nothing useful.
MAPE_THRESHOLD_MIN = 0.01
MAPE_THRESHOLD_MAX = 100.0

def validate_mape_threshold(mape_threshold: Optional[float]) -> Optional[float]:
    """
    Validate the ensemble `mape_threshold` against the allowed
    [MAPE_THRESHOLD_MIN, MAPE_THRESHOLD_MAX] (0.01-100) range.

    `mape_threshold` is an upper cutoff: a model is included in the
    ensemble only if its MAPE is at or below it. The bound itself must be
    a sensible percentage within [0.01, 100].

    `None` is a distinct, legal value meaning "no screening at all", and
    is returned unchanged. Note this is NOT the same as 0, which stays
    invalid: 0 would admit only a literally perfect model, which is
    almost never what anybody means. Because both are falsy, every
    downstream check must test `is None` rather than truthiness -- a
    plain `if not mape_threshold` would silently treat a rejected 0 and
    a deliberate None identically.
    """
    if mape_threshold is None:
        return None

    if not (MAPE_THRESHOLD_MIN <= mape_threshold <= MAPE_THRESHOLD_MAX):
        raise InvalidConfigError(
            f"mape_threshold must be between {MAPE_THRESHOLD_MIN:g} and "
            f"{MAPE_THRESHOLD_MAX:g} (got {mape_threshold}).",
            param_name="mape_threshold",
            provided_value=mape_threshold,
            valid_range=(MAPE_THRESHOLD_MIN, MAPE_THRESHOLD_MAX),
        )
    return mape_threshold