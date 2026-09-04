"""
SARIMAX model wrapper (statsmodels).

Expects train_df already shaped by SARIMAXAdapter: columns "ds", raw "y",
and standardised independent columns. Everything model-specific lives
here -- the log transform, the annual Fourier terms, and the back-transform
safeguards.

Selection: SARIMAX uses the SAME forecaster-level grid search as every
other model. It previously self-tuned via an internal search over a
validation tail it carved out of train itself, which made it the odd one
out: every other model was selected by the orchestrator, on the
orchestrator's window. Since 1.0 the orchestrator scores every model's
candidates on a dedicated validation window that no reported number comes
from, so the concern that motivated SARIMAX's private search -- that
shared selection meant selecting on the reported window -- no longer
applies to anyone. One selection mechanism for every model is the more
useful property here, so the grid varies (p, q), the seasonal (P, Q), and
the annual Fourier order, while `d` stays ADF-derived (a property of the
data, not a hyperparameter).

Annual Fourier lives here, not in the adapter, because that is the only
way the grid can choose it. Real data made this necessary: on a ~3-year
daily series annual Fourier was a clear win, while on a ~2-year weekly
series it was actively harmful (and diverged outright at high order). No
fixed default serves both.

Key safeguards, all kept from the validation-search era:

- log1p fit / expm1 predict, so yhat is returned in RAW units.
- enforce_stationarity=True on the final fit (falling back to False),
  stopping explosive AR roots at the source.
- A log-space clip anchored on a high QUANTILE of the training target
  (not its raw max, which a single spike inflates enormously).
- +sigma^2/2 lognormal bias correction, optional via `bias_correction`.
- A divergence guard: a forecast pinned to the clip is rejected rather
  than emitted as a degenerate flat line. With the grid now tolerating
  per-candidate failures, a diverging candidate is simply skipped.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from horizon_forecaster.exceptions import (
    InsufficientDataError,
    ModelConvergenceError,
    ModelPredictionError,
)
from horizon_forecaster.models.base import BaseModel

# Annual period in DAYS. Fourier terms are built from absolute day offsets
# for both frequencies -- a year is a year regardless of sampling rate --
# which also makes them robust to gaps, unlike a positional index.
_ANNUAL_PERIOD_DAYS = 365.25


class SARIMAXWrapper(BaseModel):
    """Wraps statsmodels SARIMAX to the BaseModel contract."""

    model_name = "sarimax"

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

        p = self.params
        # --- grid-selected structure ---
        self.p = p.get("p", 1)
        self.q = p.get("q", 1)
        self.seasonal_p = p.get("P", 0)
        self.seasonal_q = p.get("Q", 0)
        self.fourier_order_annual = p.get("fourier_order_annual", 0)
        self.use_seasonal = p.get("use_seasonal", frequency == "daily")
        self.seasonal_period = p.get("seasonal_period", 7)
        # --- back-transform / fitting behaviour ---
        self.alpha = p.get("alpha", 0.05)
        self.log_clip_headroom = p.get("log_clip_headroom", 1.5)
        self.clip_anchor_quantile = p.get("clip_anchor_quantile", 0.99)
        self.bias_correction = p.get("bias_correction", True)
        self.enforce_stationarity_final = p.get("enforce_stationarity_final", True)
        self.maxiter_final = p.get("maxiter_final", 150)

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

        self._independent_columns: List[str] = [c for c in train_df.columns if c not in ("ds", "y")]
        self._reference_date = pd.to_datetime(train_df["ds"]).iloc[0]
        exog = self._build_exog(train_df)

        d = self._adf_differencing_order(y_log)
        order = (self.p, d, self.q)
        seasonal_order = (
            (self.seasonal_p, min(d, 1), self.seasonal_q, self.seasonal_period)
            if self.use_seasonal
            else (0, 0, 0, 0)
        )

        results = self._fit_final(y_log, exog, order, seasonal_order)
        if results is None:
            raise ModelConvergenceError(
                f"SARIMAX failed to fit with order={order} seasonal_order={seasonal_order}.",
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

        if future_exog is None:
            raise ModelPredictionError(
                "SARIMAX requires future_exog (a DataFrame with 'ds' and the columns "
                "produced by SARIMAXAdapter) to predict -- received None.",
                model_name=self.model_name,
            )
        if "ds" not in future_exog.columns:
            raise ModelPredictionError(
                "future_exog must contain a 'ds' column.", model_name=self.model_name
            )
        missing = [c for c in self._independent_columns if c not in future_exog.columns]
        if missing:
            raise ModelPredictionError(
                f"future_exog is missing required column(s): {missing}",
                model_name=self.model_name,
            )
        if len(future_exog) != horizon:
            raise ModelPredictionError(
                f"future_exog has {len(future_exog)} rows but horizon={horizon} was requested.",
                model_name=self.model_name,
            )

        exog = self._build_exog(future_exog)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_log = np.asarray(
                    self._results.get_forecast(steps=horizon, exog=exog).predicted_mean,
                    dtype=float,
                )
        except Exception as e:
            raise ModelPredictionError(
                f"SARIMAX failed during prediction ({type(e).__name__}: {e}).",
                model_name=self.model_name,
                original_exception=e,
            )

        # Interval from the CONSTANT innovation variance, not
        # get_forecast()'s horizon-growing forecast variance: the latter
        # makes the log-space interval enormous over a long horizon, and
        # expm1 turns that into an absurd raw-scale upper bound.
        from scipy.stats import norm

        z = norm.ppf(1.0 - self.alpha / 2.0)
        sigma = float(np.sqrt(self._resid_var))
        cap = self._clip_anchor_log + self.log_clip_headroom
        corrected_log = mean_log + self._bias_shift()

        # Divergence guard: a forecast pinned to the clip across most of the
        # horizon is a blown-up fit, not a prediction. Rejecting it lets the
        # grid search skip this candidate and keep the ones that worked.
        n_saturated = int(np.sum(corrected_log >= cap))
        if n_saturated >= max(1, int(np.ceil(0.5 * len(corrected_log)))):
            raise ModelPredictionError(
                f"SARIMAX forecast diverged: the log-space forecast exceeded the "
                f"explosion clip on {n_saturated}/{len(corrected_log)} of the horizon "
                f"(order={self._order}, seasonal={self._seasonal_order}, "
                f"fourier_order={self.fourier_order_annual}), so the back-transformed "
                f"prediction would be a degenerate flat line pinned at "
                f"~{np.expm1(cap):.3g}. Rejecting this fit.",
                model_name=self.model_name,
            )

        # All three go through _back_transform so the clip/floor/expm1
        # chain lives in exactly one place. The bounds pass resid_var=0.0
        # to suppress the +sigma^2/2 lift: it corrects the MEAN of the
        # lognormal, and shifting a quantile by it would just move the
        # whole interval without making it cover anything better.
        yhat = self._back_transform(mean_log, None, self._clip_anchor_log)
        yhat_lower = self._back_transform(mean_log - z * sigma, 0.0, self._clip_anchor_log)
        yhat_upper = self._back_transform(mean_log + z * sigma, 0.0, self._clip_anchor_log)

        return pd.DataFrame(
            {
                "ds": future_exog["ds"].reset_index(drop=True),
                "yhat": yhat,
                "yhat_lower": yhat_lower,
                "yhat_upper": yhat_upper,
            }
        )

    # ---------- exog / Fourier ----------

    def _build_exog(self, frame: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Assemble the exog matrix: the adapter's standardised independent
        columns, plus annual Fourier terms generated here.

        Fourier terms are computed from the absolute day offset against the
        reference date captured at fit time, so train and future stay
        phase-aligned and a gap in the date sequence cannot shift the phase
        (a positional index would).
        """
        blocks = []
        if self._independent_columns:
            blocks.append(frame[self._independent_columns].to_numpy(dtype=float))

        fourier = self._fourier_terms(frame["ds"])
        if fourier is not None:
            blocks.append(fourier)

        if not blocks:
            return None
        return np.hstack(blocks)

    def _fourier_terms(self, ds: pd.Series) -> Optional[np.ndarray]:
        order = int(self.fourier_order_annual or 0)
        if order <= 0:
            return None
        offset_days = pd.to_datetime(ds).reset_index(drop=True) - self._reference_date
        t = offset_days.dt.total_seconds().to_numpy() / 86400.0
        terms = []
        for k in range(1, order + 1):
            angle = 2.0 * np.pi * k * t / _ANNUAL_PERIOD_DAYS
            terms.append(np.sin(angle))
            terms.append(np.cos(angle))
        return np.column_stack(terms)

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
            return 1
        return 0 if p_value <= 0.05 else 1

    @staticmethod
    def _trend_for(order, seasonal_order) -> str:
        """
        Choose the deterministic trend term for a given differencing.

        This is not cosmetic. statsmodels' SARIMAX defaults to no constant
        at all, which is only correct when the series is differenced --
        differencing removes the level, so a constant would imply a drift
        nobody asked for. With d=0 and D=0, however, "no constant" means a
        ZERO-MEAN process, and this wrapper fits in log1p space where a
        typical business series sits around 7 (log1p of ~1,100). Asked to
        fit a zero-mean model to data centred at 7, the optimiser has only
        one way out: push the AR root to ~1 so the level decays as slowly
        as possible. The fit then looks respectable in-sample and the
        forecast mean-reverts toward zero, gently at first and then not.

        Observed directly on a trending daily series: ADF returned d=0, and
        the resulting 80-period forecast fell from ~945 to 3.3 against
        actuals near 1,300 (97.7% MAPE). Over a shorter 28-period horizon
        the same defect merely halved the forecast -- which is far more
        dangerous, because it reads as a plausible downturn rather than an
        obvious failure, and the existing divergence guard only watches for
        explosions upward.

        The term used when nothing is differenced is "ct" (constant AND
        linear trend), not just "c". A constant alone stops the collapse to
        zero but leaves the forecast reverting to the training mean, which
        on a growing series is still a confident prediction of decline
        (measured at ~30% MAPE on the same trending series, versus ~7% with
        the trend term). ADF with its default constant-only regression will
        happily call a trending series stationary, so "not differenced"
        cannot be taken to mean "has no trend" -- the model needs a level
        AND a slope to be stationary around. On a genuinely flat series the
        slope is estimated at essentially zero and costs nothing (4.1% with
        "c" vs 3.9% with "ct" on flat synthetic data).

        Deliberately narrow: this changes only the branch that was broken.
        The differencing decision itself is untouched, so every series that
        currently gets d>=1 -- random walks, level shifts, exponential
        growth -- fits exactly as it did before.
        """
        differenced = order[1] > 0 or seasonal_order[1] > 0
        return "n" if differenced else "ct"

    def _fit_one(self, y_log, exog, order, seasonal_order, enforce_stationarity, maxiter):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_log,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                trend=self._trend_for(order, seasonal_order),
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=False,
            )
            return model.fit(disp=False, maxiter=maxiter, method="lbfgs")

    def _fit_final(self, y_log, exog, order, seasonal_order):
        """Fit with stationarity enforced, falling back to unconstrained."""
        if self.enforce_stationarity_final:
            try:
                return self._fit_one(
                    y_log,
                    exog,
                    order,
                    seasonal_order,
                    enforce_stationarity=True,
                    maxiter=self.maxiter_final,
                )
            except Exception:
                pass
        try:
            return self._fit_one(
                y_log,
                exog,
                order,
                seasonal_order,
                enforce_stationarity=False,
                maxiter=self.maxiter_final,
            )
        except Exception:
            return None

    def _bias_shift(self, resid_var: Optional[float] = None) -> float:
        """+sigma^2/2 lognormal correction, or 0 when bias_correction is off."""
        if not self.bias_correction:
            return 0.0
        return (self._resid_var if resid_var is None else resid_var) / 2.0

    def _back_transform(self, mean_log, resid_var, anchor_log) -> np.ndarray:
        """
        Log space back to raw units: apply the lognormal bias correction,
        floor at 0 (expm1(0)), cap at the explosion clip, then expm1.

        The single place this conversion is written. `predict()` calls it
        for the point forecast and both interval bounds -- the bounds pass
        `resid_var=0.0` because the +sigma^2/2 correction targets the mean
        of the distribution and must not be applied to a quantile of it.
        """
        cap = anchor_log + self.log_clip_headroom
        return np.expm1(np.clip(mean_log + self._bias_shift(resid_var), 0.0, cap))

    @staticmethod
    def _innovation_variance(results) -> float:
        """
        Constant innovation variance (sigma^2) for the lognormal bias
        correction and the prediction band.
        """
        try:
            sigma2 = float(results.params[results.param_names.index("sigma2")])
            if np.isfinite(sigma2) and sigma2 > 0:
                return sigma2
        except (ValueError, KeyError, TypeError):
            pass
        resid_var = float(np.nanvar(results.resid))
        return resid_var if np.isfinite(resid_var) and resid_var > 0 else 0.0
