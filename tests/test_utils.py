import pandas as pd
import pytest

import numpy as np

from horizon_forecaster.exceptions import InvalidConfigError, UnsupportedFrequencyError
from horizon_forecaster.utils import (
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