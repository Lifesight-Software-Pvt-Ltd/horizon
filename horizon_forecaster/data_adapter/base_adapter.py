"""
Abstract base class for all data adapters in horizon_forecaster.

Adapters are responsible for ALL model-specific data reshaping. Model
wrappers (ProphetWrapper, SARIMAXWrapper, etc.) never see raw user data —
they only ever receive data that has already been shaped by the
corresponding adapter. This keeps wrappers focused purely on model-fitting
logic, and keeps all "how does model X want its input shaped" knowledge
in one place per model.

Contract:
    transform_train()  raw historical data -> model-ready training data
    transform_future()  future dates (+ optional future independent
                         variables) -> model-ready future/exog data,
                         ready to be passed as `future_exog` to the
                         corresponding wrapper's predict().
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from horizon_forecaster.exceptions import InvalidConfigError


class BaseAdapter(ABC):
    """
    Abstract base class every model-specific data adapter must inherit
    from.

    Args:
        date_column: name of the date column in the user's raw data.
        target_column: name of the target/dependent variable column in
            the user's raw data.
        independent_columns: names of independent variable columns
            (spend, impressions, clicks, promos, organic, etc.) to carry
            through, if any. May be empty/None if the user provides no
            independent variables.
        frequency: "daily" or "weekly", as validated by
            HorizonForecaster. Accepted here (rather than passed
            separately/specially) so forecaster.py's MODEL_REGISTRY-driven
            instantiation can construct every adapter identically
            (`AdapterClass(date_column=..., target_column=...,
            independent_columns=..., frequency=...)`), without
            special-casing any one model. Adapters that don't need it
            (e.g. ProphetAdapter, which has no frequency-dependent
            feature logic) simply ignore it via this base class.
    """

    def __init__(
        self,
        date_column: str,
        target_column: str,
        independent_columns: Optional[List[str]] = None,
        frequency: Optional[str] = None,
    ):
        self.date_column = date_column
        self.target_column = target_column
        self.independent_columns = independent_columns or []
        self.frequency = frequency

    @abstractmethod
    def transform_train(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape raw historical data into the format the corresponding
        model wrapper's fit() expects.

        Args:
            raw_df: user's raw DataFrame, containing at least
                self.date_column and self.target_column, plus any of
                self.independent_columns that are present.

        Returns:
            A DataFrame shaped for this specific model's fit() method.
            The exact shape is model-specific (e.g. Prophet wants
            "ds"/"y"; SARIMAX wants a separate exog matrix concept;
            CatBoost wants a flat table of trend-index and cyclical/
            calendar features) — see each concrete adapter's docstring
            for its specific output shape.
        """
        raise NotImplementedError

    @abstractmethod
    def transform_future(
        self, future_dates: pd.Series, future_independent_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Reshape future dates (and optional future independent variable
        values) into the format the corresponding model wrapper's
        predict() expects as `future_exog`.

        Args:
            future_dates: the future dates to forecast for.
            future_independent_df: DataFrame of future independent
                variable values, indexed/aligned to future_dates.
                Required (non-None) if this adapter has any
                independent_columns declared — the caller (forecaster.py)
                is responsible for having already resolved future values,
                either from real user-provided data or by applying a
                fallback strategy (carry_forward, seasonal_naive,
                historical_average), BEFORE calling this method. This
                method does not invent future values itself, and treats
                a missing DataFrame (when columns were declared) as a
                hard error rather than silently proceeding without them.

        Returns:
            A DataFrame shaped for this specific model's predict() method.

        Raises:
            InvalidConfigError: if independent_columns were declared but
                future_independent_df is None, or is missing one or more
                of the declared columns.
        """
        raise NotImplementedError

    @staticmethod
    def _validate_columns_present(df: pd.DataFrame, required_columns: list) -> None:
        """
        Hard-fail when a declared column is absent from the input.

        Shared by every adapter, so the message and exception type have a
        single definition rather than five that can drift apart.

        Deliberately raises rather than dropping the column: a missing
        declared column means the caller's data does not match what they
        told the forecaster to expect, and silently forecasting without a
        driver they asked for is worse than refusing.

        Raises:
            InvalidConfigError: naming the missing column(s).
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise InvalidConfigError(
                f"Missing required column(s) in input data: {missing}",
                param_name="raw_df columns",
                provided_value=list(df.columns),
            )
