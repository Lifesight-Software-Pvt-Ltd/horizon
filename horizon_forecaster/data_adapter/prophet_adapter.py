"""
Data adapter for Prophet.

Prophet requires training data with exact column names "ds" (date) and
"y" (target). Independent/regressor variables are kept under their
original names, since Prophet's add_regressor() API references columns
by name and there's no requirement to rename them.
"""

from typing import Optional

import pandas as pd

from horizon_forecaster.data_adapter.base_adapter import BaseAdapter
from horizon_forecaster.exceptions import InvalidConfigError

class ProphetAdapter(BaseAdapter):
    """
    Reshapes raw user data into Prophet's expected ds/y format.
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