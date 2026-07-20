"""
Data adapter for NeuralProphet.

NeuralProphet shares Prophet's "ds"/"y" column convention and references
future regressors by their original column names, so this adapter is
deliberately near-identical to ProphetAdapter -- and, unlike
SARIMAXAdapter or CatBoostAdapter, it is completely STATELESS.

Why no feature engineering here (a real difference from the SARIMAX
adapter, which log1p's, standardises and range-clamps its exog):
NeuralProphet normalises internally (`normalize='auto'`), for both the
target and every registered regressor. Transforming the data here on top
of that would double-normalise it -- so the target and regressors are
passed through RAW and NeuralProphet is left to do its own scaling.

Seasonality, trend and changepoints are likewise NeuralProphet's job (it
learns them), so none of CatBoostAdapter's calendar/cyclical feature
engineering is needed or wanted.
"""

from typing import Optional

import pandas as pd

from horizon_forecaster.data_adapter.base_adapter import BaseAdapter
from horizon_forecaster.exceptions import InvalidConfigError


class NeuralProphetAdapter(BaseAdapter):
    """
    Reshapes raw user data into NeuralProphet's expected ds/y format,
    passing independent columns through unchanged under their own names
    (NeuralProphet's add_future_regressor() references them by name).
    """

    def transform_train(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        required_columns = [self.date_column, self.target_column] + self.independent_columns
        self._validate_columns_present(raw_df, required_columns)

        df = pd.DataFrame()
        df["ds"] = pd.to_datetime(raw_df[self.date_column])
        df["y"] = raw_df[self.target_column]

        for col in self.independent_columns:
            df[col] = raw_df[col]

        df = df.sort_values("ds").reset_index(drop=True)
        return df

    def transform_future(
        self, future_dates: pd.Series, future_independent_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        df = pd.DataFrame()
        df["ds"] = pd.to_datetime(future_dates).reset_index(drop=True)

        if self.independent_columns:
            if future_independent_df is None:
                raise InvalidConfigError(
                    f"independent_columns {self.independent_columns} were declared, "
                    f"but no future_independent_df was provided.",
                    param_name="future_independent_df",
                    provided_value=None,
                )
            self._validate_columns_present(future_independent_df, self.independent_columns)
            for col in self.independent_columns:
                df[col] = future_independent_df[col].reset_index(drop=True)

        return df

    @staticmethod
    def _validate_columns_present(df: pd.DataFrame, required_columns: list) -> None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise InvalidConfigError(
                f"Missing required column(s) in input data: {missing}",
                param_name="raw_df columns",
                provided_value=list(df.columns),
            )
