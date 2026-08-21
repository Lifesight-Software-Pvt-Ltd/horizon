import pandas as pd
import pytest

import numpy as np

from horizon_forecaster.exceptions import InvalidConfigError, UnsupportedFrequencyError
from horizon_forecaster.utils import (
    resolve_split,
    clean_zero_variance,
    infer_frequency,
    validate_frequency,
    validate_no_missing_values,
    validate_training_size,
    validate_mape_threshold,
)


# ---------- validate_no_missing_values (complete-data requirement) ----------

def _complete_df(n=100):
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n),
        "y": np.arange(n, dtype=float),
        "spend": np.arange(n, dtype=float) + 1,
    })


def test_validate_no_missing_passes_on_complete_data():
    validate_no_missing_values(_complete_df(), "date", "y", ["spend"])   # no raise


def test_validate_no_missing_raises_on_a_single_nan_target():
    df = _complete_df(); df.loc[5, "y"] = np.nan                          # ONE missing cell
    with pytest.raises(InvalidConfigError):
        validate_no_missing_values(df, "date", "y", ["spend"])


def test_validate_no_missing_raises_on_nan_independent():
    df = _complete_df(); df.loc[5, "spend"] = np.nan
    with pytest.raises(InvalidConfigError):
        validate_no_missing_values(df, "date", "y", ["spend"])


def test_validate_no_missing_raises_on_non_numeric_value():
    df = _complete_df(); df["y"] = df["y"].astype(object); df.loc[0, "y"] = "x"
    with pytest.raises(InvalidConfigError):
        validate_no_missing_values(df, "date", "y", ["spend"])


def test_validate_no_missing_raises_on_missing_date():
    df = _complete_df(); df["date"] = df["date"].astype(object); df.loc[0, "date"] = None
    with pytest.raises(InvalidConfigError):
        validate_no_missing_values(df, "date", "y", ["spend"])


def test_validate_no_missing_ignores_unused_columns():
    df = _complete_df(); df["extra"] = np.nan     # not date/target/independent -> ignored
    validate_no_missing_values(df, "date", "y", ["spend"])   # no raise


def test_validate_no_missing_raises_clearly_on_absent_column():
    df = _complete_df()
    with pytest.raises(InvalidConfigError):        # missing target -> clear error, not KeyError
        validate_no_missing_values(df, "date", "does_not_exist", ["spend"])
    with pytest.raises(InvalidConfigError):        # missing independent
        validate_no_missing_values(df, "date", "y", ["nope"])

# ---------- clean_zero_variance ----------

def test_clean_zero_variance_drops_constant_columns():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5),
        "spend": [10, 20, 30, 40, 50],
        "promo_flag": [0, 0, 0, 0, 0],  # zero variance
        "y": [1, 2, 3, 4, 5],
    })
    cleaned, dropped = clean_zero_variance(df, exclude_columns=["date", "y"])
    assert dropped == ["promo_flag"]
    assert "promo_flag" not in cleaned.columns
    assert "spend" in cleaned.columns

def test_clean_zero_variance_respects_exclude_columns():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3),
        "y": [5, 5, 5],  # constant, but excluded
    })
    cleaned, dropped = clean_zero_variance(df, exclude_columns=["date", "y"])
    assert dropped == []
    assert "y" in cleaned.columns

def test_clean_zero_variance_all_nan_column_is_dropped():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3),
        "organic": [None, None, None],
    })
    cleaned, dropped = clean_zero_variance(df, exclude_columns=["date"])
    assert "organic" in dropped

# ---------- infer_frequency ----------

def test_infer_frequency_detects_daily():
    dates = pd.Series(pd.date_range("2026-01-01", periods=10, freq="D"))
    assert infer_frequency(dates) == "daily"

def test_infer_frequency_detects_weekly():
    dates = pd.Series(pd.date_range("2026-01-01", periods=10, freq="7D"))
    assert infer_frequency(dates) == "weekly"

def test_infer_frequency_rejects_monthly():
    dates = pd.Series(pd.date_range("2026-01-01", periods=6, freq="30D"))
    with pytest.raises(UnsupportedFrequencyError):
        infer_frequency(dates)

def test_infer_frequency_raises_on_insufficient_dates():
    dates = pd.Series(pd.to_datetime(["2026-01-01"]))
    with pytest.raises(InvalidConfigError):
        infer_frequency(dates)

# ---------- validate_frequency ----------

def test_validate_frequency_accepts_daily_and_weekly_case_insensitive():
    assert validate_frequency("Daily") == "daily"
    assert validate_frequency("WEEKLY") == "weekly"

def test_validate_frequency_rejects_unsupported():
    with pytest.raises(UnsupportedFrequencyError):
        validate_frequency("monthly")

# ---------- validate_training_size ----------

@pytest.mark.parametrize("value,expected", [
    (50, 0.5),
    (80, 0.8),
    (100, 1.0),
    (0.6, 0.6),
    (1.0, 1.0),
])
def test_validate_training_size_accepts_valid_values(value, expected):
    assert validate_training_size(value) == expected

@pytest.mark.parametrize("value", [10, 49, 0.3, 150])
def test_validate_training_size_rejects_out_of_range(value):
    with pytest.raises(InvalidConfigError):
        validate_training_size(value)

# ---------- validate_mape_threshold ----------

def test_validate_mape_threshold_accepts_valid_values():
    assert validate_mape_threshold(0.5) == 0.5
    assert validate_mape_threshold(40) == 40  # the default
    assert validate_mape_threshold(100) == 100

def test_validate_mape_threshold_rejects_out_of_bounds():
    with pytest.raises(InvalidConfigError):
        validate_mape_threshold(0)       
    with pytest.raises(InvalidConfigError):
        validate_mape_threshold(150)  # above the internal ceiling of 100

# ---------- resolve_split ----------

def test_resolve_split_uses_frequency_defaults_on_a_long_series():
    assert resolve_split(400, "daily") == (316, 56, 28)
    assert resolve_split(300, "weekly") == (261, 26, 13)


def test_resolve_split_treats_ints_as_periods_and_floats_as_fractions():
    assert resolve_split(400, "daily", validation_size=40, test_size=20) == (340, 40, 20)
    # 0.1 * 400 = 40, 0.05 * 400 = 20 -- same split, expressed as fractions.
    assert resolve_split(400, "daily", validation_size=0.1, test_size=0.05) == (340, 40, 20)


@pytest.mark.parametrize("value", [1.0, 28.0, 2.5])
def test_resolve_split_rejects_ambiguous_floats(value):
    """
    `28` and `28.0` would otherwise mean periods while `0.28` meant a
    fraction -- exactly the unit confusion that produces a wrong split
    nobody notices. A float >= 1 is refused rather than guessed at.
    """
    with pytest.raises(InvalidConfigError, match="ambiguous"):
        resolve_split(400, "daily", test_size=value)


@pytest.mark.parametrize("value", [0, -5, 0.0, True])
def test_resolve_split_rejects_nonsensical_window_sizes(value):
    # bool is a subclass of int, so True must not sail through as "1 period".
    with pytest.raises(InvalidConfigError):
        resolve_split(400, "daily", validation_size=value)


def test_resolve_split_percentage_route_splits_remainder_two_to_one():
    n_train, n_validation, n_test = resolve_split(1000, "daily", training_size=80)
    assert n_train == 800
    assert n_validation + n_test == 200
    # Validation is the selection instrument and wants stability, so it
    # takes two-thirds rather than an even split.
    assert n_validation == 133 and n_test == 67


def test_resolve_split_refuses_to_mix_the_two_routes():
    with pytest.raises(InvalidConfigError, match="cannot be combined"):
        resolve_split(400, "daily", training_size=80, test_size=20)


def test_resolve_split_scales_defaults_down_on_a_short_series():
    with pytest.warns(UserWarning, match="too short for the default"):
        n_train, n_validation, n_test = resolve_split(60, "daily")
    # Capped at 40% of the data between them, keeping the 2:1 ratio.
    assert (n_train, n_validation, n_test) == (36, 16, 8)
    assert n_validation + n_test == int(60 * 0.4)


def test_resolve_split_never_silently_shrinks_an_explicit_window():
    """A size the caller chose is honoured or refused -- never quietly
    reduced, unlike the defaults."""
    with pytest.raises(InvalidConfigError, match="do not leave any rows for training"):
        resolve_split(60, "daily", validation_size=50, test_size=20)


def test_resolve_split_scaling_applies_only_when_both_windows_defaulted():
    # test_size named explicitly -> no scaling, so the daily default
    # validation window of 56 is used as-is and there is no room left.
    with pytest.raises(InvalidConfigError):
        resolve_split(60, "daily", test_size=5)
