import pandas as pd
import pytest

import numpy as np

from horizon_forecaster.exceptions import InvalidConfigError, UnsupportedFrequencyError
from horizon_forecaster.utils import (
    ACCEPTED_DATE_FORMATS,
    as_name_list,
    describe_date_spacing,
    describe_weekdays,
    find_closed_weekdays,
    open_day_positions,
    find_duplicate_dates,
    parse_date_column,
    parse_driver_column,
    resolve_split,
    clean_zero_variance,
    validate_frequency,
    validate_no_missing_values,
    validate_training_size,
    validate_mape_threshold,
)

# ---------- validate_no_missing_values (complete-data requirement) ----------


def _complete_df(n=100):
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=n),
            "y": np.arange(n, dtype=float),
            "spend": np.arange(n, dtype=float) + 1,
        }
    )


def test_validate_no_missing_passes_on_complete_data():
    validate_no_missing_values(_complete_df(), "date", "y", ["spend"])  # no raise


def test_validate_no_missing_raises_on_a_single_nan_target():
    df = _complete_df()
    df.loc[5, "y"] = np.nan  # ONE missing cell
    with pytest.raises(InvalidConfigError):
        validate_no_missing_values(df, "date", "y", ["spend"])


def test_validate_no_missing_raises_on_nan_independent():
    df = _complete_df()
    df.loc[5, "spend"] = np.nan
    with pytest.raises(InvalidConfigError):
        validate_no_missing_values(df, "date", "y", ["spend"])


def test_validate_no_missing_raises_on_non_numeric_value():
    df = _complete_df()
    df["y"] = df["y"].astype(object)
    df.loc[0, "y"] = "x"
    with pytest.raises(InvalidConfigError):
        validate_no_missing_values(df, "date", "y", ["spend"])


def test_validate_no_missing_raises_on_missing_date():
    df = _complete_df()
    df["date"] = df["date"].astype(object)
    df.loc[0, "date"] = None
    with pytest.raises(InvalidConfigError):
        validate_no_missing_values(df, "date", "y", ["spend"])


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_validate_no_missing_raises_on_infinite_values(bad):
    df = _complete_df()
    df.loc[5, "spend"] = bad
    with pytest.raises(InvalidConfigError, match="'spend' is infinite for 2022-01-06") as info:
        validate_no_missing_values(df, "date", "y", ["spend"])
    assert info.value.provided_value == {"spend": 1}


def test_validate_no_missing_counts_blank_text_as_blank():
    df = _complete_df()
    df["spend"] = df["spend"].astype(object)
    df.loc[5, "spend"] = "  "
    with pytest.raises(InvalidConfigError, match="'spend' is blank for 2022-01-06"):
        validate_no_missing_values(df, "date", "y", ["spend"])


def test_validate_no_missing_names_rows_when_the_date_is_unreadable():
    df = _complete_df()
    df["date"] = df["date"].astype(object)
    df.loc[0, "date"] = None
    df.loc[0, "y"] = None
    with pytest.raises(InvalidConfigError) as info:
        validate_no_missing_values(df, "date", "y", ["spend"])
    assert "'date' is blank or not a date in row 1" in str(info.value)
    assert "'y' is blank for row 1" in str(info.value)


def test_validate_no_missing_ignores_unused_columns():
    df = _complete_df()
    df["extra"] = np.nan  # not date/target/independent -> ignored
    validate_no_missing_values(df, "date", "y", ["spend"])  # no raise


def test_validate_no_missing_raises_clearly_on_absent_column():
    df = _complete_df()
    with pytest.raises(InvalidConfigError):  # missing target -> clear error, not KeyError
        validate_no_missing_values(df, "date", "does_not_exist", ["spend"])
    with pytest.raises(InvalidConfigError):  # missing independent
        validate_no_missing_values(df, "date", "y", ["nope"])


# ---------- clean_zero_variance ----------


def test_clean_zero_variance_drops_constant_columns():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5),
            "spend": [10, 20, 30, 40, 50],
            "promo_flag": [0, 0, 0, 0, 0],  # zero variance
            "y": [1, 2, 3, 4, 5],
        }
    )
    cleaned, dropped = clean_zero_variance(df, exclude_columns=["date", "y"])
    assert dropped == ["promo_flag"]
    assert "promo_flag" not in cleaned.columns
    assert "spend" in cleaned.columns


def test_clean_zero_variance_respects_exclude_columns():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "y": [5, 5, 5],  # constant, but excluded
        }
    )
    cleaned, dropped = clean_zero_variance(df, exclude_columns=["date", "y"])
    assert dropped == []
    assert "y" in cleaned.columns


def test_clean_zero_variance_all_nan_column_is_dropped():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "organic": [None, None, None],
        }
    )
    cleaned, dropped = clean_zero_variance(df, exclude_columns=["date"])
    assert "organic" in dropped


# ---------- closed weekdays and open-day spacing ----------


def test_find_closed_weekdays_names_the_weekdays_that_never_appear():
    weekdays = pd.Series(pd.bdate_range("2024-01-01", periods=40))
    assert find_closed_weekdays(weekdays) == (5, 6)
    no_sundays = pd.Series([d for d in pd.date_range("2024-01-01", periods=60) if d.dayofweek != 6])
    assert find_closed_weekdays(no_sundays) == (6,)


def test_find_closed_weekdays_finds_none_in_a_full_week_or_a_short_span():
    assert find_closed_weekdays(pd.Series(pd.date_range("2024-01-01", periods=60))) == ()
    assert find_closed_weekdays(pd.Series(pd.bdate_range("2024-01-01", periods=15))) == ()


def test_dates_all_on_one_weekday_are_weekly_data_not_a_one_day_week():
    assert find_closed_weekdays(pd.Series(pd.date_range("2024-01-07", periods=30, freq="7D"))) == ()


def test_a_weekday_that_appears_once_is_not_closed():
    dates = list(pd.bdate_range("2024-01-01", periods=40)) + [pd.Timestamp("2024-01-06")]
    assert find_closed_weekdays(pd.Series(sorted(dates))) == (6,)


def test_open_day_positions_step_over_closed_days():
    dates = pd.Series(pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"]))  # Thu, Fri, Mon
    assert list(np.diff(open_day_positions(dates, (5, 6)))) == [1, 1]


def test_weekends_are_not_gaps_on_a_weekday_calendar():
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=40))
    spacing = describe_date_spacing(dates, "daily", (5, 6))
    assert spacing.missing_periods == 0 and spacing.coverage == 1.0
    missing_wednesday = dates.drop(index=7).reset_index(drop=True)
    spacing = describe_date_spacing(missing_wednesday, "daily", (5, 6))
    assert spacing.missing_periods == 1 and spacing.gap_count == 1


def test_describe_weekdays_reads_naturally():
    assert describe_weekdays([6]) == "Sunday"
    assert describe_weekdays([5, 6]) == "Saturday and Sunday"
    assert describe_weekdays([0, 2, 4]) == "Monday, Wednesday and Friday"
    assert describe_weekdays([5, 6], "or", plural=True) == "Saturdays or Sundays"


# ---------- validate_frequency ----------


@pytest.mark.parametrize("freq", [None, 7, ["daily"]])
def test_validate_frequency_rejects_non_text(freq):
    with pytest.raises(UnsupportedFrequencyError, match="frequency must be one of"):
        validate_frequency(freq)


def test_validate_frequency_accepts_daily_and_weekly_case_insensitive():
    assert validate_frequency("Daily") == "daily"
    assert validate_frequency("WEEKLY") == "weekly"


def test_validate_frequency_rejects_unsupported():
    with pytest.raises(UnsupportedFrequencyError):
        validate_frequency("monthly")


# ---------- validate_training_size ----------


@pytest.mark.parametrize(
    "value,expected",
    [
        (50, 0.5),
        (80, 0.8),
        (100, 1.0),
        (0.6, 0.6),
        (1.0, 1.0),
    ],
)
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
    assert resolve_split(400, "daily") == (288, 84, 28)
    assert resolve_split(300, "weekly") == (248, 39, 13)


def test_resolve_split_keeps_full_weekly_windows_on_a_typical_weekly_series():
    # 122 weeks (~2.3 years) is common for weekly marketing data. The
    # defaulted-window budget is 45% of rows, so it keeps the full 39 + 13
    # rather than shrinking the test window below a quarter.
    assert resolve_split(122, "weekly") == (70, 39, 13)


def test_resolve_split_treats_ints_as_periods_and_floats_as_fractions():
    assert resolve_split(400, "daily", validation_size=40, test_size=20) == (340, 40, 20)
    # 0.1 * 400 = 40, 0.05 * 400 = 20 -- same split, expressed as fractions.
    assert resolve_split(400, "daily", validation_size=0.1, test_size=0.05) == (340, 40, 20)


def test_resolve_split_accepts_numpy_integers_as_periods():
    # A size computed with numpy (np.int64) is still a number of periods.
    split = resolve_split(400, "daily", validation_size=np.int64(40), test_size=np.int32(20))
    assert split == (340, 40, 20)
    assert all(type(n) is int for n in split)


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


def test_resolve_split_percentage_route_splits_remainder_three_to_one():
    n_train, n_validation, n_test = resolve_split(1000, "daily", training_size=80)
    assert n_train == 800
    assert n_validation + n_test == 200
    # Validation is the selection instrument and wants stability, so it
    # takes three-quarters, matching the default windows' 3:1 ratio.
    assert n_validation == 150 and n_test == 50


def test_resolve_split_refuses_to_mix_the_two_routes():
    with pytest.raises(InvalidConfigError, match="cannot be combined"):
        resolve_split(400, "daily", training_size=80, test_size=20)


def test_resolve_split_scales_defaults_down_on_a_short_series():
    with pytest.warns(UserWarning, match="too short for the default"):
        n_train, n_validation, n_test = resolve_split(60, "daily")
    # Capped at 45% of the data between them, keeping the 3:1 ratio.
    assert (n_train, n_validation, n_test) == (33, 20, 7)
    assert n_validation + n_test == int(60 * 0.45)


def test_resolve_split_never_silently_shrinks_an_explicit_window():
    """A size the caller chose is honoured or refused -- never quietly
    reduced, unlike the defaults."""
    with pytest.raises(InvalidConfigError, match="do not leave any rows for training"):
        resolve_split(60, "daily", validation_size=50, test_size=20)


def test_resolve_split_scaling_applies_only_when_both_windows_defaulted():
    # test_size named explicitly -> no scaling, so the daily default
    # validation window of 84 is used as-is and there is no room left.
    with pytest.raises(InvalidConfigError):
        resolve_split(60, "daily", test_size=5)


# ---------- parse_date_column ----------

_JAN = pd.date_range("2024-01-10", periods=10, freq="D")


@pytest.mark.parametrize(
    "fmt",
    ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m-%d-%Y", "%m/%d/%Y"],
)
def test_parse_date_column_reads_each_accepted_format(fmt):
    # Days 10-19 include values above 12, so day-first and month-first are
    # each unambiguous.
    text = pd.Series(_JAN.strftime(fmt))
    parsed = parse_date_column(text, "date")
    assert list(parsed) == list(_JAN)


def test_accepted_formats_list_matches_the_six_documented_formats():
    assert len(ACCEPTED_DATE_FORMATS) == 6


def test_parse_date_column_passes_datetimes_through_unchanged():
    dates = pd.Series(_JAN)
    assert parse_date_column(dates, "date").equals(dates)


def test_parse_date_column_drops_timezone_keeping_local_time():
    dates = pd.Series(_JAN.tz_localize("Asia/Kolkata"))
    parsed = parse_date_column(dates, "date")
    assert parsed.dt.tz is None
    assert list(parsed) == list(_JAN)


def test_parse_date_column_reads_full_iso_timestamps():
    text = pd.Series(["2024-01-15 00:00:00", "2024-01-16 00:00:00"])
    parsed = parse_date_column(text, "date")
    assert list(parsed) == list(pd.to_datetime(["2024-01-15", "2024-01-16"]))


def test_parse_date_column_reads_date_objects():
    import datetime

    values = pd.Series([datetime.date(2024, 1, 15), datetime.date(2024, 1, 16)])
    parsed = parse_date_column(values, "date")
    assert list(parsed) == list(pd.to_datetime(["2024-01-15", "2024-01-16"]))


def test_parse_date_column_refuses_to_guess_an_ambiguous_column():
    # Every day and month is 12 or less: 03/04/2024 is 3 April or 4 March.
    text = pd.Series(["03/04/2024", "04/04/2024", "05/04/2024"])
    with pytest.raises(InvalidConfigError, match="day-first or month-first"):
        parse_date_column(text, "date")


@pytest.mark.parametrize(
    "date_format, expected",
    [("%d/%m/%Y", "2024-04-03"), ("%m/%d/%Y", "2024-03-04")],
)
def test_date_format_settles_an_ambiguous_column(date_format, expected):
    text = pd.Series(["03/04/2024", "04/04/2024"])
    parsed = parse_date_column(text, "date", date_format=date_format)
    assert parsed.iloc[0] == pd.Timestamp(expected)


def test_date_format_supports_formats_outside_the_accepted_list():
    parsed = parse_date_column(pd.Series(["15.01.2024"]), "date", date_format="%d.%m.%Y")
    assert parsed.iloc[0] == pd.Timestamp("2024-01-15")


def test_date_format_reports_values_that_do_not_match():
    with pytest.raises(InvalidConfigError, match="do not match date_format"):
        parse_date_column(pd.Series(["15/01/2024", "2024-01-16"]), "date", date_format="%d/%m/%Y")


def test_parse_date_column_rejects_mixed_formats():
    text = pd.Series(["15/01/2024", "2024-01-16"])
    with pytest.raises(InvalidConfigError, match="mixes date formats"):
        parse_date_column(text, "date")


@pytest.mark.parametrize("bad", [["15.01.2024"], ["15/01/24", "16/01/24"], ["not a date"]])
def test_parse_date_column_rejects_unrecognised_text(bad):
    with pytest.raises(InvalidConfigError, match="Accepted formats"):
        parse_date_column(pd.Series(bad), "date")


def test_parse_date_column_rejects_a_numeric_date_column():
    with pytest.raises(InvalidConfigError, match="is numeric"):
        parse_date_column(pd.Series([20240115, 20240116]), "date")


def test_parse_date_column_leaves_missing_cells_for_the_missing_value_check():
    parsed = parse_date_column(pd.Series(["2024-01-15", None, ""]), "date")
    assert parsed.iloc[0] == pd.Timestamp("2024-01-15")
    assert parsed.iloc[1:].isna().all()


# ---------- find_duplicate_dates / find_date_gaps ----------


def test_find_duplicate_dates_lists_each_repeated_date_once():
    dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-03"]))
    assert find_duplicate_dates(dates) == [pd.Timestamp("2024-01-02")]


def test_find_duplicate_dates_is_empty_for_unique_dates():
    assert find_duplicate_dates(pd.Series(_JAN)) == []


def test_describe_date_spacing_is_clean_for_a_regular_series():
    spacing = describe_date_spacing(pd.Series(_JAN), "daily")
    assert spacing.one_period_share == 1.0
    assert spacing.missing_periods == 0 and spacing.irregular_steps == 0
    assert spacing.coverage == 1.0


def test_describe_date_spacing_counts_gaps_and_the_longest_run():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-15"]))
    spacing = describe_date_spacing(dates, "daily")
    assert spacing.missing_periods == 2 + 9
    assert spacing.gap_count == 2
    assert spacing.longest_gap == (pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-15"))
    assert spacing.longest_gap_periods == 9
    assert spacing.coverage == pytest.approx(4 / 15)


def test_describe_date_spacing_reports_the_typical_step_and_its_share():
    dates = pd.Series(pd.date_range("2024-01-01", periods=10, freq="2D"))
    spacing = describe_date_spacing(dates, "daily")
    assert spacing.typical_step == pd.Timedelta(days=2)
    assert spacing.one_period_share == 0.0


def test_describe_date_spacing_flags_weekly_dates_on_different_weekdays():
    # Sunday, Sunday, then a Monday: 8 days is not a whole number of weeks.
    dates = pd.Series(pd.to_datetime(["2024-01-07", "2024-01-14", "2024-01-22"]))
    spacing = describe_date_spacing(dates, "weekly")
    assert spacing.irregular_steps == 1
    assert spacing.missing_periods == 0


# ---------- parse_driver_column ----------


def test_parse_driver_reads_yes_no_words_in_any_case_and_mix():
    out = parse_driver_column(pd.Series(["Yes", "no", " Y ", "N", "TRUE", "f", "On", "off"]), "p")
    assert list(out) == [1, 0, 1, 0, 1, 0, 1, 0]


def test_parse_driver_passes_numbers_and_turns_booleans_and_numeric_text_into_numbers():
    assert list(parse_driver_column(pd.Series([True, False]), "p")) == [1, 0]
    assert list(parse_driver_column(pd.Series(["3.5", "2"]), "p")) == [3.5, 2.0]
    assert list(parse_driver_column(pd.Series([1.5, 2.5]), "p")) == [1.5, 2.5]


def test_parse_driver_keeps_blanks_missing():
    out = parse_driver_column(pd.Series(["Yes", None, ""]), "p")
    assert out.iloc[0] == 1 and out.iloc[1:].isna().all()


def test_parse_driver_rejects_an_unclear_pair():
    with pytest.raises(InvalidConfigError, match=r"two values \['None', 'Promo'\] are unclear"):
        parse_driver_column(pd.Series(["Promo", "None", "Promo"]), "promo")


def test_parse_driver_rejects_categories():
    with pytest.raises(InvalidConfigError, match="one 0/1 column per category"):
        parse_driver_column(pd.Series(["Low", "Mid", "High"]), "tier")


def test_parse_driver_rejects_a_single_text_value_with_blanks():
    with pytest.raises(InvalidConfigError, match="single text value"):
        parse_driver_column(pd.Series(["Promo", None, None]), "promo")


def test_parse_driver_rejects_text_among_numbers_with_its_date():
    dates = pd.Series(pd.date_range("2024-03-01", periods=3))
    with pytest.raises(InvalidConfigError, match=r"\['1,200'\] on 2024-03-02"):
        parse_driver_column(pd.Series(["100", "1,200", "300"]), "spend", dates=dates)


# ---------- as_name_list ----------


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("spend", ["spend"]),
        (["a", "b"], ["a", "b"]),
        (("a",), ["a"]),
        (pd.Index(["a"]), ["a"]),
        (np.array(["a", "b"]), ["a", "b"]),
    ],
)
def test_as_name_list_reads_names(value, expected):
    assert as_name_list(value, "independent_columns") == expected


@pytest.mark.parametrize("value", [5, {"a": 1}, np.array([["a"]])])
def test_as_name_list_rejects_non_names(value):
    with pytest.raises(InvalidConfigError, match="independent_columns must be"):
        as_name_list(value, "independent_columns")


def test_validate_mape_threshold_rejects_text_and_bool():
    for bad in ("40", True):
        with pytest.raises(InvalidConfigError, match="mape_threshold must be a number"):
            validate_mape_threshold(bad)


def test_check_history_reads_text_dates_on_a_copy():
    from horizon_forecaster.utils import check_history

    history = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "y": [1, 2]})
    out = check_history(history, "date", "y", "test")
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert history["date"].dtype == object  # the caller's frame is untouched
