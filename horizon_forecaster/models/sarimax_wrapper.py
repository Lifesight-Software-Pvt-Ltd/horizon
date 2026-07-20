"""
SARIMAX model wrapper (statsmodels).

Expects train_df already shaped by SARIMAXAdapter: columns "ds", raw "y",
and an exog matrix (standardised independent columns + annual Fourier
terms). This wrapper owns everything model-specific:

- Target transform: fits in log1p space (variance stabilisation for a
  strictly-positive metric) and inverts with expm1 on predict, so yhat is
  returned in RAW units -- forecaster.py scores MAPE against the raw y.

- Order selection: ADF picks the differencing order d, then a small
  exhaustive order space is scored on a validation tail carved out of the
  TRAINING data (not the reported test split). The lowest-MAPE order wins
  -- selecting on forecast accuracy, not AIC, which was the root cause of
  the occasional very-high-MAPE fits in the original notebook. AIC is kept
  only as a fallback if every candidate fails to produce a scorable
  validation forecast.

- Explosion control at the source: the winning order is refit with
  enforce_stationarity=True (falling back to False only if that fit
  fails), so explosive AR roots -- the real cause of exponential blow-ups
  after expm1 -- are prevented up front. A log-space clip to
  (train_max + headroom) remains as a last-resort guard.

- Bias correction: expm1 of the log-space mean systematically
  under-predicts (it returns the median, not the mean, of a lognormal).
  The point forecast adds +sigma^2/2 using the model's CONSTANT innovation
  variance (not the horizon-growing forecast variance, which would feed
  the very explosion the clip fights). Consequence, by design: the
  bias-corrected point can sit slightly above the centre of the
  (uncorrected) prediction interval -- correct lognormal behaviour, since
  mean > median.

SARIMAX has no forecaster-level grid search (its config carries no
`hyperparameter_grid`): the internal ADF + validation-MAPE search IS its
tuning, so it fits once per evaluate()/forecast() and self-selects.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from horizon_forecaster.evaluation.metrics import mape
from horizon_forecaster.exceptions import (
    InsufficientDataError,
    ModelConvergenceError,
    ModelPredictionError,
)
from horizon_forecaster.models.base import BaseModel


class SARIMAXWrapper(BaseModel):
    """Wraps statsmodels SARIMAX to the BaseModel contract."""

    model_name = "sarimax"

    # Higher than Prophet/CatBoost's floors: order selection carves a
    # validation tail out of train, so we need enough rows to both fit and
    # validate. ~5 weeks daily / ~4 months weekly.
    _FREQUENCY_MIN_ROWS_DEFAULTS = {"daily": 35, "weekly": 18}
    _FALLBACK_MIN_ROWS = 20

    def __init__(
        self,
        frequency: str,
        params: Optional[Dict[str, Any]] = None,
        min_rows_required: Optional[int] = None,
        random_seed: int = 42,
    ):
        super().__init__(frequency, params)
        if min_rows_required is None:
            min_rows_required = self._FREQUENCY_MIN_ROWS_DEFAULTS.get(
                frequency, self._FALLBACK_MIN_ROWS
            )
        self.min_rows_required = min_rows_required
        self.random_seed = random_seed

        # All wrapper-interpreted config arrives via params (SARIMAX's
        # base_hyperparameters); nothing here is passed to a library
        # constructor blindly. Defaults keep the wrapper usable standalone.
        p = self.params
        self.alpha = p.get("alpha", 0.05)
        self.log_clip_headroom = p.get("log_clip_headroom", 1.5)
        self.p_max = p.get("p_max", 1)
        self.q_max = p.get("q_max", 1)
        self.seasonal_cap = p.get("seasonal_cap", 1)
        self.seasonal_period = p.get("seasonal_period", 7)
        self.use_seasonal = p.get("use_seasonal", frequency == "daily")
        self.validation_fraction = p.get("validation_fraction", 0.15)
        self.min_validation_rows = p.get("min_validation_rows", 7 if frequency == "daily" else 4)
        self.maxiter_search = p.get("maxiter_search", 30)
        self.maxiter_final = p.get("maxiter_final", 150)
        self.enforce_stationarity_final = p.get("enforce_stationarity_final", True)
        # Anchor the explosion clip on a high QUANTILE of the training target
        # rather than its raw max: on spiky data a single outlier max inflates
        # the cap enormously (a real series with a lone 849k spike gave a 3.8e6
        # cap, so a diverged forecast could still sit at 3.8e6 before tripping
        # any guard). A quantile anchor keeps the cap tied to the bulk of the
        # data. Set to 1.0 to restore the old max-anchored behaviour.
        self.clip_anchor_quantile = p.get("clip_anchor_quantile", 0.99)
        # Lognormal bias correction (+sigma^2/2). ON by default, but note the
        # tension: MAPE -- which this library selects on and reports -- is
        # minimised by the conditional MEDIAN, and expm1(mean_log) already IS
        # the median; the correction targets the MEAN. Turn this off for
        # strictly MAPE-optimal point forecasts.
        self.bias_correction = p.get("bias_correction", True)

    # ---------- fit ----------

    def fit(self, train_df: pd.DataFrame, target_column: str = "y") -> "SARIMAXWrapper":
        if "ds" not in train_df.columns or "y" not in train_df.columns:
            raise ModelPredictionError(
                "SARIMAXWrapper.fit() expects train_df with 'ds' and 'y' columns "
                "(i.e. data already shaped by SARIMAXAdapter).",
                model_name=self.model_name,
            )

        n = len(train_df)
        if n < self.min_rows_required:
            raise InsufficientDataError(
                f"SARIMAX requires at least {self.min_rows_required} rows to fit reliably.",
                model_name=self.model_name,
                required_periods=self.min_rows_required,
                available_periods=n,
            )

        y_raw = train_df["y"].to_numpy(dtype=float)
        y_log = np.log1p(y_raw)
        self._exog_columns: List[str] = [c for c in train_df.columns if c not in ("ds", "y")]
        exog = train_df[self._exog_columns].to_numpy(dtype=float) if self._exog_columns else None

        d = self._adf_differencing_order(y_log)
        candidates = self._candidate_orders(d)

        order, seasonal_order = self._select_order_by_validation(y_log, exog, y_raw, candidates)
        if order is None:
            raise ModelConvergenceError(
                "SARIMAX: every candidate order failed to fit or produced no scorable forecast.",
                model_name=self.model_name,
            )

        results = self._fit_final(y_log, exog, order, seasonal_order)
        if results is None:
            raise ModelConvergenceError(
                "SARIMAX: the selected order failed to fit on the full training data.",
                model_name=self.model_name,
            )

        self._results = results
        self._fitted_model = results
        self._order = order
        self._seasonal_order = seasonal_order
        self._d = d
        self._clip_anchor_log = float(np.nanquantile(y_log, self.clip_anchor_quantile))
        self._resid_var = self._innovation_variance(results)
        self.is_fitted = True
        return self

    # ---------- predict ----------

    def predict(self, horizon: int, future_exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self._check_is_fitted()

        if self._exog_columns:
            if future_exog is None:
                raise ModelPredictionError(
                    "SARIMAX requires future_exog (a DataFrame with 'ds' and the exog "
                    "columns produced by SARIMAXAdapter) to predict -- received None.",
                    model_name=self.model_name,
                )
            if "ds" not in future_exog.columns:
                raise ModelPredictionError(
                    "future_exog must contain a 'ds' column.", model_name=self.model_name
                )
            missing = [c for c in self._exog_columns if c not in future_exog.columns]
            if missing:
                raise ModelPredictionError(
                    f"future_exog is missing required exog column(s): {missing}",
                    model_name=self.model_name,
                )

        if future_exog is None or len(future_exog) != horizon:
            got = 0 if future_exog is None else len(future_exog)
            raise ModelPredictionError(
                f"future_exog has {got} rows but horizon={horizon} was requested.",
                model_name=self.model_name,
            )

        exog = future_exog[self._exog_columns].to_numpy(dtype=float) if self._exog_columns else None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_log = np.asarray(
                    self._results.get_forecast(steps=horizon, exog=exog).predicted_mean,
                    dtype=float,
                )
        except Exception as e:
            raise ModelPredictionError(
                "SARIMAX failed during prediction.",
                model_name=self.model_name,
                original_exception=e,
            )

        # Prediction interval from the CONSTANT innovation variance (sigma^2),
        # NOT get_forecast()'s horizon-growing forecast variance. Over a long
        # horizon the growing variance makes the log-space interval enormous,
        # and back-transforming it with expm1 produces absurd raw-scale upper
        # bounds (e.g. a [8k, 700k] band around an 80k point forecast, then
        # mangled further by the clip). A constant band gives a readable,
        # roughly-multiplicative "typical error" interval instead -- the same
        # constant-variance philosophy used for the bias correction. Trade-off:
        # it does not widen formally with horizon; it is a typical-error band,
        # not a textbook expanding predictive interval. (The ensemble ignores
        # bounds entirely, so this only affects per-model plots/CSVs.)
        from scipy.stats import norm

        z = norm.ppf(1.0 - self.alpha / 2.0)
        sigma = float(np.sqrt(self._resid_var))
        cap = self._clip_anchor_log + self.log_clip_headroom
        corrected_log = mean_log + self._bias_shift()

        # Divergence guard. A SARIMAX fit can blow up in log space -- an
        # (near-)explosive AR root, or an exogenous coefficient extrapolating
        # on out-of-range future values -- so the log forecast runs far past
        # the explosion clip and expm1() would emit a degenerate flat line
        # pinned at the cap (observed on real spiky weekly data: a ~3.8e6 flat
        # forecast, 1400% MAPE). The clip prevents infinities but not garbage.
        # When the clip binds on a majority of the horizon, the fit has
        # genuinely diverged, so we reject it as a prediction failure (the
        # orchestrator records it in evaluate_errors/forecast_errors and drops
        # the model) rather than returning a misleading "result". This is
        # distinct from a merely-inaccurate model, which stays a real result
        # and is handled by the ensemble's MAPE threshold.
        n_saturated = int(np.sum(corrected_log >= cap))
        if n_saturated >= max(1, int(np.ceil(0.5 * len(corrected_log)))):
            raise ModelPredictionError(
                f"SARIMAX forecast diverged: the log-space forecast exceeded the "
                f"explosion clip on {n_saturated}/{len(corrected_log)} of the horizon, "
                f"so the back-transformed prediction would be a degenerate flat line "
                f"(pinned at ~{np.expm1(cap):.3g}). Rejecting this fit rather than "
                f"emitting garbage. Likely an (near-)explosive AR root or exogenous "
                f"extrapolation on this data.",
                model_name=self.model_name,
            )

        yhat = np.expm1(np.clip(corrected_log, 0.0, cap))
        yhat_lower = np.expm1(np.clip(mean_log - z * sigma, 0.0, cap))
        yhat_upper = np.expm1(np.clip(mean_log + z * sigma, 0.0, cap))

        return pd.DataFrame(
            {
                "ds": future_exog["ds"].reset_index(drop=True),
                "yhat": yhat,
                "yhat_lower": yhat_lower,
                "yhat_upper": yhat_upper,
            }
        )

    # ---------- internals ----------

    def _adf_differencing_order(self, y_log: np.ndarray) -> int:
        """ADF test on the log target: stationary (p<=0.05) -> d=0, else d=1."""
        from statsmodels.tsa.stattools import adfuller

        series = pd.Series(y_log).dropna()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_value = adfuller(series, autolag="AIC")[1]
        except Exception:
            return 1  # default to one difference if the test itself fails
        return 0 if p_value <= 0.05 else 1

    def _candidate_orders(self, d: int) -> List[tuple]:
        p_range = range(0, self.p_max + 1)
        q_range = range(0, self.q_max + 1)
        if self.use_seasonal:
            big_d = min(d, 1)
            seasonal_orders = [
                (P, big_d, Q, self.seasonal_period)
                for P in range(0, self.seasonal_cap + 1)
                for Q in range(0, self.seasonal_cap + 1)
            ]
        else:
            seasonal_orders = [(0, 0, 0, 0)]
        return [
            ((p, d, q), seasonal)
            for p in p_range
            for q in q_range
            for seasonal in seasonal_orders
        ]

    def _fit_one(self, y_log, exog, order, seasonal_order, enforce_stationarity, maxiter):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_log,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=False,
            )
            return model.fit(disp=False, maxiter=maxiter, method="lbfgs")

    def _select_order_by_validation(self, y_log, exog, y_raw, candidates):
        """
        Score each candidate by MAPE on a validation tail carved out of
        train, and return the best (order, seasonal_order). Falls back to
        the lowest-AIC candidate if none produced a scorable forecast.
        Returns (None, None) if nothing fit at all.
        """
        n = len(y_log)
        val_size = max(self.min_validation_rows, int(np.ceil(self.validation_fraction * n)))
        val_size = min(val_size, max(1, n // 3))  # keep enough rows for the fit slice
        fit_n = n - val_size

        y_fit_log = y_log[:fit_n]
        y_val_raw = y_raw[fit_n:]
        exog_fit = exog[:fit_n] if exog is not None else None
        exog_val = exog[fit_n:] if exog is not None else None
        fit_max_log = float(np.nanmax(y_fit_log))

        best_mape = np.inf
        best_order = None
        best_aic = np.inf
        best_order_by_aic = None

        for order, seasonal_order in candidates:
            try:
                res = self._fit_one(
                    y_fit_log, exog_fit, order, seasonal_order,
                    enforce_stationarity=False, maxiter=self.maxiter_search,
                )
            except Exception:
                continue

            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic = res.aic
                best_order_by_aic = (order, seasonal_order)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean_log = np.asarray(
                        res.get_forecast(steps=val_size, exog=exog_val).predicted_mean, dtype=float
                    )
            except Exception:
                continue

            yhat_val = self._back_transform(mean_log, self._innovation_variance(res), fit_max_log)
            score = mape(y_val_raw, yhat_val).value
            if np.isfinite(score) and score < best_mape:
                best_mape = score
                best_order = (order, seasonal_order)

        chosen = best_order if best_order is not None else best_order_by_aic
        return chosen if chosen is not None else (None, None)

    def _fit_final(self, y_log, exog, order, seasonal_order):
        """Refit the chosen order on full train, enforcing stationarity first."""
        if self.enforce_stationarity_final:
            try:
                return self._fit_one(
                    y_log, exog, order, seasonal_order,
                    enforce_stationarity=True, maxiter=self.maxiter_final,
                )
            except Exception:
                pass  # fall back to the unconstrained fit below
        try:
            return self._fit_one(
                y_log, exog, order, seasonal_order,
                enforce_stationarity=False, maxiter=self.maxiter_final,
            )
        except Exception:
            return None

    def _bias_shift(self, resid_var: Optional[float] = None) -> float:
        """+sigma^2/2 lognormal correction, or 0 when bias_correction is off."""
        if not self.bias_correction:
            return 0.0
        return (self._resid_var if resid_var is None else resid_var) / 2.0

    def _back_transform(self, mean_log, resid_var, train_max_log) -> np.ndarray:
        cap = train_max_log + self.log_clip_headroom
        return np.expm1(np.clip(mean_log + self._bias_shift(resid_var), 0.0, cap))

    @staticmethod
    def _innovation_variance(results) -> float:
        """
        Constant innovation variance (sigma^2) for the lognormal bias
        correction. Prefers the model's estimated sigma2 parameter; falls
        back to the residual variance if unavailable.
        """
        try:
            sigma2 = float(results.params[results.param_names.index("sigma2")])
            if np.isfinite(sigma2) and sigma2 > 0:
                return sigma2
        except (ValueError, KeyError, TypeError):
            pass
        resid_var = float(np.nanvar(results.resid))
        return resid_var if np.isfinite(resid_var) and resid_var > 0 else 0.0
