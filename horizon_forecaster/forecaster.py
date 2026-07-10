"""
HorizonForecaster: the main orchestration class for horizon_forecaster.

This is the first place that wires together data adapters, model
wrappers, and config loading — everything built so far (Phase 1's
contracts, Phase 2's Prophet adapter/wrapper/config) was built precisely
so this file could stay thin, since all the hard decisions were already
made and tested in isolation before reaching here.

Current scope (Phase 2, step 5): single model (Prophet), evaluate() only.
No ensemble, no multi-model selection, no future forecast() yet — those
are Phase 4 and a following Phase 2 step, respectively.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from horizon_forecaster.data_adapter.prophet_adapter import ProphetAdapter
from horizon_forecaster.evaluation.metrics import MapeResult, mape
from horizon_forecaster.exceptions import InvalidConfigError
from horizon_forecaster.models.base import BaseModel
from horizon_forecaster.models.prophet_wrapper import ProphetWrapper
from horizon_forecaster.utils import clean_zero_variance, validate_frequency, validate_training_size

_CONFIG_PATH = Path(__file__).parent / "config" / "default_params.yaml"

@dataclass
class EvaluationResult:
    """
    Structured result from HorizonForecaster.evaluate().

    Attributes:
        model_name: which model this result is for (e.g. "prophet").
        mape: MapeResult (value + zero-actual exclusion metadata), not a
            bare float — see evaluation/metrics.py for why.
        n_train: number of rows used for training.
        n_test: number of rows held out for evaluation.
        test_predictions: DataFrame with columns ["ds", "y", "yhat"]
            (plus "yhat_lower"/"yhat_upper" if the model provides them),
            one row per test-set period — kept for later plotting/export
            (Phase 5), not just the summary MAPE number.
        fitted_model: the wrapper instance fitted on the TRAIN SPLIT ONLY.
            Kept for introspection/debugging. NOT the model used for any
            real future forecast — forecast() (a following step) refits
            on the full dataset separately, per the project's "refit
            once, on everything, before the real forecast" design.
    """

    model_name: str
    mape: MapeResult
    n_train: int
    n_test: int
    test_predictions: pd.DataFrame
    fitted_model: BaseModel

class HorizonForecaster:
    """
    Main orchestrator. Takes raw user data once at construction, and
    exposes evaluate() (train/test split + MAPE) as a first step, with a
    separate forecast() step (full-data refit + real future prediction)
    to follow.

    Args:
        raw_df: user's raw DataFrame containing at least date_column and
            target_column, plus any of independent_columns that apply.
        date_column: name of the date column in raw_df.
        target_column: name of the target/dependent variable column.
        frequency: "daily" or "weekly".
        training_size: percentage (50-100) or fraction (0.5-1.0) of data
            to use for training; the remainder is held out for
            evaluation. Defaults to 80 (an 80/20 split), a common
            convention — override explicitly for anything else.
        independent_columns: names of independent variable columns
            (spend, impressions, promos, etc.) to use, if any.
    """

    def __init__(
        self,
        raw_df: pd.DataFrame,
        date_column: str,
        target_column: str,
        frequency: str,
        training_size: float = 80,
        independent_columns: Optional[List[str]] = None,
    ):
        self.frequency = validate_frequency(frequency)
        self.training_fraction = validate_training_size(training_size)
        self.date_column = date_column
        self.target_column = target_column

        requested_independent_columns = independent_columns or []
        cleaned_df, dropped_columns = clean_zero_variance(
            raw_df, exclude_columns=[date_column, target_column]
        )
        self.dropped_zero_variance_columns = dropped_columns

        # Any independent column WE dropped for zero variance must also
        # be removed from the requested list — otherwise the adapter's
        # (deliberately strict) hard-error-on-missing-column would fire
        # against a column we silently removed ourselves, not something
        # the user actually got wrong.
        self.independent_columns = [
            c for c in requested_independent_columns if c not in dropped_columns
        ]

        self.raw_df = cleaned_df.sort_values(date_column).reset_index(drop=True)
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load this frequency's Prophet hyperparameters + wrapper_config."""
        with open(_CONFIG_PATH) as f:
            full_config = yaml.safe_load(f)
        return full_config["prophet"][self.frequency]

    def evaluate(self) -> EvaluationResult:
        """
        Chronologically split the data, fit Prophet on the training
        portion only, predict on the held-out test portion, and compute
        MAPE.

        Raises:
            InvalidConfigError: if training_size leaves zero rows for
                the test split (e.g. training_size=100 on a dataset
                where the resulting split rounds to using every row).
            InsufficientDataError: propagated from ProphetWrapper.fit()
                if the training split doesn't meet the frequency-aware
                minimum row floor.
            ModelConvergenceError / ModelPredictionError: propagated
                from ProphetWrapper if the underlying Prophet fit or
                predict genuinely fails.
        """
        n_total = len(self.raw_df)
        n_train = int(n_total * self.training_fraction)
        n_test = n_total - n_train

        if n_test == 0:
            raise InvalidConfigError(
                f"training_size of {self.training_fraction * 100:.0f}% leaves no rows for "
                f"evaluation ({n_total} total rows -> {n_train} train, 0 test). "
                f"Use a smaller training_size, or a larger dataset.",
                param_name="training_size",
                provided_value=self.training_fraction,
            )

        train_raw = self.raw_df.iloc[:n_train]
        test_raw = self.raw_df.iloc[n_train:]

        adapter = ProphetAdapter(
            date_column=self.date_column,
            target_column=self.target_column,
            independent_columns=self.independent_columns,
        )

        adapted_train = adapter.transform_train(train_raw)
        # transform_train() is reused for the test slice too, not just
        # train: it just renames/reshapes/validates columns and sorts by
        # date — nothing about its logic is train-specific. The "train"
        # in its name refers to it operating on data that INCLUDES real
        # target values (unlike transform_future(), which doesn't
        # require them), which is exactly what evaluation needs: real
        # y values to compare predictions against.
        adapted_test = adapter.transform_train(test_raw)

        model = ProphetWrapper(
            frequency=self.frequency,
            params=self._config["hyperparameters"],
            min_rows_required=self._config["wrapper_config"]["min_rows_required"],
            random_seed=self._config["wrapper_config"]["random_seed"],
        )
        model.fit(adapted_train, target_column="y")

        future_exog = adapted_test.drop(columns=["y"])
        forecast = model.predict(horizon=len(test_raw), future_exog=future_exog)

        mape_result = mape(adapted_test["y"].values, forecast["yhat"].values)

        test_predictions = adapted_test[["ds", "y"]].copy()
        test_predictions["yhat"] = forecast["yhat"].values
        if "yhat_lower" in forecast.columns:
            test_predictions["yhat_lower"] = forecast["yhat_lower"].values
            test_predictions["yhat_upper"] = forecast["yhat_upper"].values

        return EvaluationResult(
            model_name=model.model_name,
            mape=mape_result,
            n_train=n_train,
            n_test=n_test,
            test_predictions=test_predictions,
            fitted_model=model,
        )