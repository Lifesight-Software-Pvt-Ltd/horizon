"""
CatBoost model wrapper.

Expects train_df already shaped by CatBoostAdapter (columns: "ds", "y",
plus calendar features, lag/rolling-window features, and any
independent columns — see data_adapter/catboost_adapter.py).
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from horizon_forecaster.exceptions import (
    InsufficientDataError,
    ModelConvergenceError,
    ModelPredictionError,
)
from horizon_forecaster.models.base import BaseModel

class CatBoostWrapper(BaseModel):
    """
    Wraps CatBoostRegressor to conform to the BaseModel contract.

    Unlike ProphetWrapper, this wrapper never produces yhat_lower/
    yhat_upper — CatBoost has no native uncertainty/credible-interval
    output, and BaseModel's contract (see models/base.py) explicitly
    says wrappers without native uncertainty support should simply omit
    those columns rather than fabricate them. predict() therefore
    always returns exactly ["ds", "yhat"].

    Design decision: calendar features that are genuinely categorical
    or cyclical (day_of_week, month, quarter, is_month_start,
    is_month_end, is_weekend) are passed to CatBoost as declared
    categorical features, not raw numeric ones — a tree split like
    "day_of_week <= 3" implies an ordering (Wednesday is "less than"
    Thursday) that doesn't correspond to anything real, whereas
    CatBoost's categorical handling uses target statistics instead of
    assuming order. day_of_month, day_of_year, week_of_year are left
    numeric, since they carry real ordinal/magnitude information (e.g.
    day_of_month proximity to typical pay dates).

    KNOWN LIMITATION (confirmed via direct testing, not theoretical):
    tree-based models, including CatBoost, can only predict values
    within the range of leaf outputs seen during training. On a series
    with a genuine ongoing trend, once the forecast horizon moves past
    the max target value seen in training, predictions flatten out
    near the last few training values instead of continuing the trend
    — unlike Prophet, which models trend as an explicit continuous
    function. Verified directly: a clean +2/period synthetic trend
    produced a test MAPE of ~9% specifically because of this
    flattening, not noise or a fitting bug. No fix applied in this
    wrapper (would require adding an explicit numeric trend/time-index
    feature, or modeling the differenced series instead of the level)
    — documented here as a real, load-bearing finding for Phase 4's
    ensemble weighting to be aware of, not silently absorbed.
    """

    model_name = "catboost"

    # Mirrors ProphetWrapper's pattern: frequency-aware fallback floors,
    # used only if min_rows_required isn't explicitly provided. Counted
    # AFTER CatBoostAdapter's own lag/rolling warmup-row drop — i.e. this
    # is "rows with a fully-populated feature set", not raw input rows.
    _FREQUENCY_MIN_ROWS_DEFAULTS = {"daily": 14, "weekly": 8}
    _FALLBACK_MIN_ROWS = 10

    _CATEGORICAL_CALENDAR_FEATURES = (
        "day_of_week",
        "month",
        "quarter",
        "is_month_start",
        "is_month_end",
        "is_weekend",
    )

    _NON_FEATURE_COLUMNS = ("ds", "y")

    def __init__(
        self,
        frequency: str,
        params: Optional[Dict[str, Any]] = None,
        min_rows_required: Optional[int] = None,
        random_seed: int = 42,
    ):
        """
        Args:
            frequency: "daily" or "weekly".
            params: hyperparameters passed directly to CatBoostRegressor's
                constructor (e.g. n_estimators, learning_rate, depth).
                Should NOT contain wrapper-level settings like
                min_rows_required or random_seed.
            min_rows_required: minimum post-warmup training rows required
                to fit. If None, falls back to a frequency-aware default.
            random_seed: seed passed to CatBoostRegressor as random_seed,
                for reproducible training (CatBoost's own fit process has
                stochastic elements — subsampling, feature permutations —
                so this matters for reproducibility the same way
                ProphetWrapper's random_seed does for its uncertainty
                simulation).
        """
        super().__init__(frequency, params)
        if min_rows_required is None:
            min_rows_required = self._FREQUENCY_MIN_ROWS_DEFAULTS.get(
                frequency, self._FALLBACK_MIN_ROWS
            )
        self.min_rows_required = min_rows_required
        self.random_seed = random_seed
        self._feature_columns: Optional[List[str]] = None
        self._categorical_feature_names: Optional[List[str]] = None

    def fit(self, train_df: pd.DataFrame, target_column: str = "y") -> "CatBoostWrapper":
        """
        Fit CatBoostRegressor on already-adapted data.

        Note: `target_column` is accepted for interface consistency with
        BaseModel, but CatBoostAdapter always produces a "y" column, so
        this wrapper does not use the argument to rename anything — it
        simply validates that "y" is present (same convention as
        ProphetWrapper).
        """
        if "ds" not in train_df.columns or "y" not in train_df.columns:
            raise ModelPredictionError(
                "CatBoostWrapper.fit() expects train_df with 'ds' and 'y' "
                "columns (i.e. data already shaped by CatBoostAdapter).",
                model_name=self.model_name,
            )

        if len(train_df) < self.min_rows_required:
            raise InsufficientDataError(
                f"CatBoost requires at least {self.min_rows_required} rows "
                f"(after lag/rolling warmup rows are dropped) to fit reliably.",
                model_name=self.model_name,
                required_periods=self.min_rows_required,
                available_periods=len(train_df),
            )

        feature_columns = [c for c in train_df.columns if c not in self._NON_FEATURE_COLUMNS]
        categorical_feature_names = [
            c for c in feature_columns if c in self._CATEGORICAL_CALENDAR_FEATURES
        ]

        X = train_df[feature_columns]
        y = train_df["y"]

        try:
            # Imported here (not at module level) so that importing this
            # module doesn't require CatBoost's dependency chain unless
            # this specific wrapper is actually used — same rationale as
            # ProphetWrapper's lazy `from prophet import Prophet`.
            from catboost import CatBoostRegressor, Pool

            model = CatBoostRegressor(random_seed=self.random_seed, verbose=False, **self.params)
            train_pool = Pool(X, y, cat_features=categorical_feature_names)
            model.fit(train_pool)
        except Exception as e:
            raise ModelConvergenceError(
                "CatBoost failed to fit.",
                model_name=self.model_name,
                original_exception=e,
            )

        self._fitted_model = model
        self._feature_columns = feature_columns
        self._categorical_feature_names = categorical_feature_names
        self.is_fitted = True
        return self

    def predict(self, horizon: int, future_exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict using an already-adapted future_exog DataFrame (from
        CatBoostAdapter.transform_future()), containing "ds" and every
        feature column CatBoost was trained on.

        Like ProphetWrapper, a None future_exog is treated as a caller
        error (forecaster.py failed to build it), not a valid "predict
        with no context" request — CatBoost's features are entirely
        derived from dates/history, so there's no meaningful prediction
        without them.
        """
        self._check_is_fitted()

        if future_exog is None:
            raise ModelPredictionError(
                "CatBoost requires future_exog (a DataFrame with 'ds' and every "
                "feature column used at fit time) to predict — received None.",
                model_name=self.model_name,
            )

        if "ds" not in future_exog.columns:
            raise ModelPredictionError(
                "future_exog must contain a 'ds' column.",
                model_name=self.model_name,
            )

        if len(future_exog) != horizon:
            raise ModelPredictionError(
                f"future_exog has {len(future_exog)} rows but horizon={horizon} was requested.",
                model_name=self.model_name,
            )

        missing_features = [c for c in self._feature_columns if c not in future_exog.columns]
        if missing_features:
            raise ModelPredictionError(
                f"future_exog is missing required feature column(s): {missing_features}",
                model_name=self.model_name,
            )

        # Exact same column set AND order used at fit time — CatBoost
        # (like most sklearn-style estimators) infers feature identity
        # from position, not from DataFrame column names.
        X_future = future_exog[self._feature_columns]

        try:
            yhat = self._fitted_model.predict(X_future)
        except Exception as e:
            raise ModelPredictionError(
                "CatBoost failed during prediction.",
                model_name=self.model_name,
                original_exception=e,
            )

        return pd.DataFrame({"ds": future_exog["ds"].reset_index(drop=True), "yhat": yhat})

    def save(self, path: str) -> None:
        """
        Overrides BaseModel's generic pickle default. CatBoost ships its
        own native serialization (save_model/load_model), which is the
        library's recommended approach and more portable across
        versions than pickling the estimator object directly — same
        rationale as ProphetWrapper's model_to_json()/model_from_json()
        override. Wrapper-level state (feature columns, categorical
        feature names, frequency, params, min_rows_required,
        random_seed) is stored in a companion JSON envelope, since
        CatBoost's native format only captures the underlying model
        itself, not this wrapper's bookkeeping.
        """
        import json

        self._check_is_fitted()

        model_path = f"{path}.cbm"
        self._fitted_model.save_model(model_path)

        envelope = {
            "frequency": self.frequency,
            "params": self.params,
            "min_rows_required": self.min_rows_required,
            "random_seed": self.random_seed,
            "feature_columns": self._feature_columns,
            "categorical_feature_names": self._categorical_feature_names,
            "model_path": model_path,
        }
        with open(path, "w") as f:
            json.dump(envelope, f)

    @classmethod
    def load(cls, path: str) -> "CatBoostWrapper":
        """
        Reload a CatBoostWrapper previously saved with save(), using
        CatBoost's own load_model() rather than generic unpickling.
        """
        import json

        from catboost import CatBoostRegressor

        with open(path, "r") as f:
            envelope = json.load(f)

        instance = cls(
            frequency=envelope["frequency"],
            params=envelope["params"],
            min_rows_required=envelope["min_rows_required"],
            random_seed=envelope["random_seed"],
        )
        model = CatBoostRegressor()
        model.load_model(envelope["model_path"])
        instance._fitted_model = model
        instance._feature_columns = envelope["feature_columns"]
        instance._categorical_feature_names = envelope["categorical_feature_names"]
        instance.is_fitted = True
        return instance
