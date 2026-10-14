"""
Generic helper utilities used across horizon_forecaster.

These functions are intentionally free of any model-specific logic —
they operate purely on pandas DataFrames/Series and raw config values.
"""

import datetime
import warnings
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
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
    "daily": {"validation": 56, "test": 28},  # 8 weeks / 4 weeks
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


# Text date formats the date column is parsed with when no `date_format` is
# given. Each is tried strictly against the WHOLE column; a format is only
# used if every value matches it. Grouped by which field comes first, because
# that is what decides whether a column is ambiguous: "03/04/2024" fits both
# the day-first and the month-first pattern, and no rule can say which was
# meant. Year-first dates have no such problem.
_YEAR_FIRST_FORMATS = ("%Y-%m-%d", "%Y/%m/%d")
_DAY_FIRST_FORMATS = ("%d-%m-%Y", "%d/%m/%Y")
_MONTH_FIRST_FORMATS = ("%m-%d-%Y", "%m/%d/%Y")

# Full ISO 8601 timestamps ("2024-01-15 00:00:00", "2024-01-15T00:00:00"),
# which is how many exports -- BigQuery among them -- write a date. Tried
# after the plain year-first formats, so "2024-01-15" uses those.
_ISO_TIMESTAMP = "ISO8601"

ACCEPTED_DATE_FORMATS = (
    "YYYY-MM-DD",
    "YYYY/MM/DD",
    "DD-MM-YYYY",
    "DD/MM/YYYY",
    "MM-DD-YYYY",
    "MM/DD/YYYY",
)

_FORMAT_LABELS = {
    "%Y-%m-%d": "YYYY-MM-DD",
    "%Y/%m/%d": "YYYY/MM/DD",
    "%d-%m-%Y": "DD-MM-YYYY",
    "%d/%m/%Y": "DD/MM/YYYY",
    "%m-%d-%Y": "MM-DD-YYYY",
    "%m/%d/%Y": "MM/DD/YYYY",
}

_PERIOD_DAYS = {"daily": 1, "weekly": 7}


def _try_format(text: pd.Series, fmt: str) -> pd.Series:
    """Parse `text` with one format; values that do not match become NaT."""
    try:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
    except (ValueError, TypeError):
        # e.g. ISO timestamps carrying different UTC offsets in one column.
        return pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _ambiguous_example(text: pd.Series) -> str:
    """A value whose day and month differ, to show why the column is ambiguous."""
    for value in text:
        parts = value.replace("-", "/").split("/")
        if len(parts) == 3 and parts[0].lstrip("0") != parts[1].lstrip("0"):
            return value
    return text.iloc[0]


def parse_date_column(
    values: pd.Series, column_name: str, date_format: Optional[str] = None
) -> pd.Series:
    """
    Turn the user's date column into real datetimes, BEFORE anything sorts
    or splits on it.

    Sorting a column of text dates sorts it alphabetically, so "01/15/2023"
    lands before "02/01/2022" -- the training, validation and test windows
    then mix years, and the forecast starts from the wrong date, with no
    error anywhere. Parsing first is what makes every later step see time.

    Accepted input:
      - a datetime column (passed through unchanged; timezones are dropped,
        keeping the local time);
      - a column of date/datetime objects;
      - text in one of ACCEPTED_DATE_FORMATS, the same format on every row;
      - full ISO 8601 timestamps such as "2024-01-15 00:00:00".

    With `date_format` (a strftime pattern such as "%d/%m/%Y"), that format
    alone is used -- this is how a caller settles an ambiguous column or
    supplies any format outside the accepted list.

    Missing cells (NaN, None, blank text) come back as NaT and are left for
    validate_no_missing_values() to report alongside the other columns.

    Raises:
        InvalidConfigError: for a numeric date column, text that matches no
            accepted format, a column mixing formats, a column that is
            equally valid day-first and month-first, or values that do not
            match an explicit `date_format`.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        if getattr(values.dt, "tz", None) is not None:
            return values.dt.tz_localize(None)
        return values

    if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
        raise InvalidConfigError(
            f"The date column '{column_name}' is numeric ({values.dtype}), so it cannot be "
            f"read as dates unambiguously -- a number such as 20240115 could be a date, an "
            f"Excel serial or a timestamp. Convert it to datetimes (pd.to_datetime) or to "
            f"text in one of {list(ACCEPTED_DATE_FORMATS)} first.",
            param_name="date_column",
            provided_value=str(values.dtype),
        )

    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    present = values.notna() & (values.astype(str).str.strip() != "")
    if not present.any():
        return result

    present_values = values[present]
    if all(isinstance(v, (datetime.date, np.datetime64)) for v in present_values):
        parsed = pd.to_datetime(present_values)
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
        result[present] = parsed.to_numpy()
        return result

    text = present_values.astype(str).str.strip()

    if date_format is not None:
        parsed = _try_format(text, date_format)
        failed = text[parsed.isna()]
        if not failed.empty:
            raise InvalidConfigError(
                f"{len(failed)} value(s) in the date column '{column_name}' do not match "
                f"date_format='{date_format}', e.g. {list(failed.unique()[:3])}.",
                param_name="date_format",
                provided_value=date_format,
            )
        result[present] = parsed.to_numpy()
        return result

    attempts = {
        fmt: _try_format(text, fmt)
        for fmt in (
            *_YEAR_FIRST_FORMATS,
            _ISO_TIMESTAMP,
            *_DAY_FIRST_FORMATS,
            *_MONTH_FIRST_FORMATS,
        )
    }
    fits_all = [fmt for fmt, parsed in attempts.items() if parsed.notna().all()]

    year_first = [f for f in fits_all if f in _YEAR_FIRST_FORMATS or f == _ISO_TIMESTAMP]
    day_first = [f for f in fits_all if f in _DAY_FIRST_FORMATS]
    month_first = [f for f in fits_all if f in _MONTH_FIRST_FORMATS]

    if year_first:
        chosen = year_first[0]
    elif day_first and month_first:
        example = _ambiguous_example(text)
        raise InvalidConfigError(
            f"The dates in '{column_name}' could be read either day-first or month-first: "
            f"every value has a day and a month of 12 or less, so '{example}' could mean "
            f"either. Pass date_format='{day_first[0]}' (day first) or "
            f"date_format='{month_first[0]}' (month first) to say which.",
            param_name="date_format",
            provided_value=example,
        )
    elif day_first:
        chosen = day_first[0]
    elif month_first:
        chosen = month_first[0]
    else:
        matches_some = pd.concat([p.notna() for p in attempts.values()], axis=1).any(axis=1)
        if matches_some.all():
            raise InvalidConfigError(
                f"The date column '{column_name}' mixes date formats -- every value is a "
                f"valid date, but no single format fits them all (e.g. "
                f"{list(text.drop_duplicates().iloc[[0, -1]])}). Use one format throughout, "
                f"or convert the column with pd.to_datetime first.",
                param_name="date_column",
                provided_value=list(text.unique()[:3]),
            )
        unreadable = list(text[~matches_some].unique()[:3])
        raise InvalidConfigError(
            f"Could not read the date column '{column_name}' as dates, e.g. {unreadable}. "
            f"Accepted formats: {', '.join(ACCEPTED_DATE_FORMATS)} (with a 4-digit year), or "
            f"a full ISO 8601 timestamp. For any other format pass date_format, e.g. "
            f"date_format='%d.%m.%Y'.",
            param_name="date_column",
            provided_value=unreadable,
        )

    result[present] = attempts[chosen].to_numpy()
    return result


def find_duplicate_dates(dates: pd.Series) -> List[pd.Timestamp]:
    """The distinct dates that appear more than once, in date order."""
    duplicated = dates[dates.duplicated(keep=False)]
    return sorted(duplicated.unique())


# The longest run of consecutive missing dates accepted, in days: 28 daily
# periods or 4 weekly ones. Scattered gaps and one long hole hurt the models
# in different ways. A hole of more than four weeks removes a whole stretch
# of trend and seasonality in one piece, and SARIMAX -- which treats
# consecutive rows as consecutive periods -- sees the dates either side of it
# as neighbours. Absolute rather than relative to the history: a two-month
# hole does the same damage in four years of data as in two. 28 days is also
# the default daily test window.
MAX_CONTINUOUS_GAP_DAYS = 28


class DateSpacing(NamedTuple):
    """How a sorted, de-duplicated date column departs from one row per period."""

    one_period_share: float  # share of steps that are exactly one period
    typical_step: pd.Timedelta  # the most common step
    irregular_steps: int  # steps that are not a whole number of periods
    irregular_example: Optional[Tuple[pd.Timestamp, pd.Timestamp]]
    missing_periods: int  # whole periods absent, across all gaps
    gap_count: int  # places where one or more periods are missing
    longest_gap: Optional[Tuple[pd.Timestamp, pd.Timestamp]]  # dates either side
    longest_gap_periods: int  # periods missing in a row in the longest gap
    coverage: float  # rows present / periods spanned


def describe_date_spacing(sorted_dates: pd.Series, frequency: str) -> DateSpacing:
    """
    Measure the spacing of a sorted, unique date column with at least two
    dates. A step of exactly one period is regular; several whole periods
    means periods are missing; a step that is not a whole number of periods
    (6 or 8 days on weekly data) is irregular.
    """
    period = pd.Timedelta(days=_PERIOD_DAYS[frequency])
    dates = pd.Series(sorted_dates).reset_index(drop=True)
    steps = dates.diff().iloc[1:]

    whole = (steps % period) == pd.Timedelta(0)
    gaps = steps[whole & (steps > period)]
    irregular = steps[~whole]

    longest_gap, longest_gap_periods = None, 0
    if not gaps.empty:
        i = gaps.idxmax()
        longest_gap = (dates[i - 1], dates[i])
        longest_gap_periods = int(gaps[i] / period) - 1
    irregular_example = None
    if not irregular.empty:
        i = irregular.index[0]
        irregular_example = (dates[i - 1], dates[i])

    spanned = (dates.iloc[-1] - dates.iloc[0]) / period + 1
    return DateSpacing(
        one_period_share=float((steps == period).mean()),
        typical_step=steps.mode().iloc[0],
        irregular_steps=len(irregular),
        irregular_example=irregular_example,
        missing_periods=int(sum(int(g / period) - 1 for g in gaps)),
        gap_count=len(gaps),
        longest_gap=longest_gap,
        longest_gap_periods=longest_gap_periods,
        coverage=float(len(dates) / spanned),
    )


# Text a driver column may use for an on/off flag, matched case-insensitively
# after trimming spaces. A fixed list on purpose: for any other pair of words
# ("Promo"/"None", "A"/"B") Horizon would have to guess which one means "on",
# and a wrong guess silently flips the sign of every effect the models learn.
YES_NO_WORDS = {
    "true": 1,
    "false": 0,
    "t": 1,
    "f": 0,
    "yes": 1,
    "no": 0,
    "y": 1,
    "n": 0,
    "on": 1,
    "off": 0,
}
_YES_NO_LABEL = "true/false, yes/no, y/n, t/f, on/off"


def parse_driver_column(
    values: pd.Series, column_name: str, dates: Optional[pd.Series] = None
) -> pd.Series:
    """
    Turn one driver column into numbers, before anything checks or models it.

      - numbers pass through; booleans become 1/0;
      - text that is a number ("12.5") becomes that number;
      - the on/off words in YES_NO_WORDS become 1/0, in any case and any mix
        ("Yes" next to "N" is fine -- each word means one thing);
      - blanks stay missing, for the missing-value check to report.

    Any other text raises InvalidConfigError saying what is unclear:
      - two text values outside the list ("Promo"/"None"): an on/off column
        whose "on" value cannot be told;
      - several text values ("Low"/"Mid"/"High"): categories, which must be
        converted to numbers first;
      - text among numbers ("1,200" in a spend column).

    `dates`, aligned with `values`, lets an error say where the text is.
    """
    values = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_bool_dtype(values):
        return values.astype(int)
    if pd.api.types.is_numeric_dtype(values):
        return values

    text = values.map(lambda v: v if pd.isna(v) else str(v).strip())
    blank = text.isna() | (text == "")
    numbers = pd.to_numeric(text.where(~blank), errors="coerce")
    is_text = ~blank & numbers.isna()
    if not is_text.any():
        return numbers

    words = text[~blank].str.lower()
    if words.isin(list(YES_NO_WORDS) + ["0", "1", "0.0", "1.0"]).all():
        out = pd.Series(np.nan, index=values.index)
        out[~blank] = words.map(lambda w: YES_NO_WORDS[w] if w in YES_NO_WORDS else float(w))
        return out.astype(float)

    distinct = list(dict.fromkeys(text[is_text]))
    has_numbers = bool((~blank & ~is_text).any())
    if not has_numbers and len(distinct) == 1:
        message = (
            f"'{column_name}' has a single text value {distinct} and blanks, so it can't be "
            f"read as numbers. If it marks the periods when something is on, use 1 there "
            f"and 0 everywhere else (a CSV reader may also have turned words such as "
            f"'None' or 'NA' into blanks)."
        )
    elif not has_numbers and len(distinct) == 2:
        message = (
            f"'{column_name}' looks like an on/off flag, but its two values "
            f'{sorted(distinct)} are unclear: Horizon can\'t tell which one means "on". Use '
            f"1/0, or one of {_YES_NO_LABEL}."
        )
    elif not has_numbers:
        shown = sorted(distinct)[:6]
        more = f" and {len(distinct) - 6} more" if len(distinct) > 6 else ""
        message = (
            f"'{column_name}' has text values {shown}{more}, not numbers. Drivers must be "
            f"numeric: convert categories to numbers yourself, or to one 0/1 column per "
            f"category."
        )
    else:
        first = int(np.flatnonzero(is_text.to_numpy())[0])
        where = ""
        if dates is not None:
            when = pd.Series(dates).reset_index(drop=True).iloc[first]
            if not pd.isna(when):
                where = f" on {pd.Timestamp(when):%Y-%m-%d}"
        message = (
            f"'{column_name}' has values that aren't numbers, e.g. {distinct[:3]}{where}. "
            f"Remove thousands separators, currency signs and other text before passing "
            f"the data."
        )
    raise InvalidConfigError(message, param_name=column_name, provided_value=distinct[:6])


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
        c for c in [date_column, target_column, *independent_columns] if c not in df.columns
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


def _resolve_window(value: Union[int, float, None], n_total: int, param_name: str) -> Optional[int]:
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
