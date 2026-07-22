"""
Bayesian PyMC model wrapper.

A Bayesian structural time-series model: a linear trend, a Fourier
seasonal basis (annual for both frequencies, plus a weekly cycle for
daily data), and a linear regression on the user's independent variables,
observed through a Student-T likelihood (heavy tails absorb spikes that
would drag a Gaussian fit). The posterior is sampled with NUTS.

Expects train_df already shaped by BayesianPyMCAdapter: columns "ds", raw
"y", and standardised independent columns. Everything model-specific lives
here -- the target standardisation, the Fourier terms, the priors, the
sampler, and the back-transform.

Design, and how it maps onto the reference notebook:

- The notebook built one Deterministic `mu` over the WHOLE series (train +
  test) and sliced the test portion out of the posterior. That only works
  when train and test are sampled together. Under the BaseModel contract,
  fit() sees only the training rows and predict() sees only future dates,
  so instead we sample the posterior on TRAIN once, keep the posterior
  draws of every coefficient, and in predict() reconstruct `mu` at the
  future dates as a pure-numpy linear combination of those draws. Same
  model, same numbers, but honouring the fit-once / predict-any-horizon
  split every other wrapper obeys.

- Target standardisation (y_mean/y_std from TRAIN) lives here, so "y" keeps
  one raw meaning across models and yhat comes back in raw units for MAPE.

- Fourier terms are built from absolute day offsets against the reference
  date captured at fit time (like SARIMAX), so train and future stay
  phase-aligned and a gap in the dates cannot shift the phase. The Fourier
  order (`n_fourier`) is an ordinary grid hyperparameter.

- Coefficient priors: an independent column that looks like a continuous
  driver (more than two distinct standardised values -- e.g. spend,
  impressions) gets an INFORMATIVE prior centred on its ordinary-least-
  squares slope against the standardised target, mirroring the notebook's
  polyfit-based spend prior. A binary flag (<= two distinct values) gets a
  plain zero-centred prior. The classification keys off the standardised
  column the adapter hands over, so no metadata crosses the boundary.

- Uncertainty: yhat_lower/upper are posterior percentiles of `mu` (the
  latent mean), matching the notebook's 3rd/97th-percentile band. Like
  every wrapper these bounds are individual-model-only and never enter the
  ensemble (see BaseModel.predict docstring).

Selection: PyMC uses the SAME forecaster-level grid search as the other
models. Because NUTS sampling is expensive, the default grid is small and
varies only `n_fourier` and `target_accept`; every candidate is sampled at
full draws (no cheap-selection shortcut), so the reported MAPE reflects the
real posterior each candidate would produce in a forecast.
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

# Annual period in DAYS. The long Fourier cycle is built from absolute day
# offsets for both frequencies -- a year is a year regardless of sampling
# rate -- which also makes it robust to gaps, unlike a positional index.
_ANNUAL_PERIOD_DAYS = 365.25

# Weekly period in DAYS, for the short (intra-week) cycle used on daily data.
_WEEKLY_PERIOD_DAYS = 7.0

# A column with more than this many distinct standardised values is treated
# as a continuous driver (informative prior); at or below it, a binary/near-
# binary flag (zero-centred prior). 2 cleanly separates 0/1 flags from real
# continuous regressors.
_CONTINUOUS_MIN_DISTINCT = 2


class BayesianPyMCWrapper(BaseModel):
    """Wraps a PyMC Bayesian structural model to the BaseModel contract."""

    model_name = "bayesianpymc"

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
        self.n_fourier = int(p.get("n_fourier", 4))
        self.target_accept = float(p.get("target_accept", 0.95))
        # --- sampler settings ---
        self.draws = int(p.get("draws", 2000))
        self.tune = int(p.get("tune", 2000))
        self.chains = int(p.get("chains", 2))
        self.cores = int(p.get("cores", 1))
        self.max_treedepth = int(p.get("max_treedepth", 15))
        self.init = p.get("init", "adapt_diag")
        # --- prior scales ---
        self.slope_sigma = float(p.get("slope_sigma", 1.0))
        self.intercept_sigma = float(p.get("intercept_sigma", 2.0))
        self.beta_fourier_sigma = float(p.get("beta_fourier_sigma", 1.0))
        self.driver_coef_sigma = float(p.get("driver_coef_sigma", 2.0))
        self.flag_coef_sigma = float(p.get("flag_coef_sigma", 1.0))
        self.sigma_sigma = float(p.get("sigma_sigma", 1.0))
        self.nu_alpha = float(p.get("nu_alpha", 2.0))
        self.nu_beta = float(p.get("nu_beta", 0.1))
        # --- back-transform / bands ---
        self.ci_lower_pct = float(p.get("ci_lower_pct", 3.0))
        self.ci_upper_pct = float(p.get("ci_upper_pct", 97.0))
        # Short (weekly) seasonal cycle is only meaningful for daily data.
        self.use_short_seasonal = bool(p.get("use_short_seasonal", frequency == "daily"))

    # ---------- fit ----------

    def fit(self, train_df: pd.DataFrame, target_column: str = "y") -> "BayesianPyMCWrapper":
        if "ds" not in train_df.columns or "y" not in train_df.columns:
            raise ModelPredictionError(
                "BayesianPyMCWrapper.fit() expects train_df with 'ds' and 'y' columns "
                "(i.e. data already shaped by BayesianPyMCAdapter).",
                model_name=self.model_name,
            )

        n = len(train_df)
        if n < self.min_rows_required:
            raise InsufficientDataError(
                f"Bayesian PyMC requires at least {self.min_rows_required} rows to fit reliably.",
                model_name=self.model_name,
                required_periods=self.min_rows_required,
                available_periods=n,
            )

        train_df = train_df.sort_values("ds").reset_index(drop=True)
        self._reference_date = pd.to_datetime(train_df["ds"]).iloc[0]
        self._independent_columns: List[str] = [
            c for c in train_df.columns if c not in ("ds", "y")
        ]

        y_raw = train_df["y"].to_numpy(dtype=float)
        self._y_mean = float(np.nanmean(y_raw))
        self._y_std = float(np.nanstd(y_raw)) + 1e-8
        y_sc = (y_raw - self._y_mean) / self._y_std

        t = self._time_steps(train_df["ds"])
        ff_long, ff_short = self._fourier_terms(train_df["ds"])
        exog = (
            train_df[self._independent_columns].to_numpy(dtype=float)
            if self._independent_columns
            else np.empty((n, 0))
        )

        # Prior means, straight from the reference notebook.
        slope_est, _ = np.polyfit(t, y_sc, 1)
        tail = min(52, n)
        intercept_est = float(y_sc[-tail:].mean())

        # Per-column coefficient prior means: OLS slope of the standardised
        # target on a continuous driver (informative), else 0 for a flag.
        self._driver_mask = [
            np.unique(exog[:, j][np.isfinite(exog[:, j])]).size > _CONTINUOUS_MIN_DISTINCT
            for j in range(exog.shape[1])
        ]
        coef_prior_means = []
        for j in range(exog.shape[1]):
            if self._driver_mask[j] and np.nanstd(exog[:, j]) > 1e-12:
                slope_j, _ = np.polyfit(exog[:, j], y_sc, 1)
                coef_prior_means.append(float(slope_j))
            else:
                coef_prior_means.append(0.0)

        try:
            trace = self._sample(
                y_sc, t, ff_long, ff_short, exog,
                slope_est=float(slope_est),
                intercept_est=intercept_est,
                coef_prior_means=coef_prior_means,
            )
        except Exception as e:  # noqa: BLE001 -- surfaced as a convergence failure
            raise ModelConvergenceError(
                f"Bayesian PyMC sampling failed ({type(e).__name__}: {e}).",
                model_name=self.model_name,
                original_exception=e,
            )

        self._extract_posterior(trace, n_exog=exog.shape[1])
        self.trace = trace
        self._fitted_model = trace
        self.sample_stats = self._convergence_summary(trace)
        self.is_fitted = True
        return self

    # ---------- predict ----------

    def predict(self, horizon: int, future_exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self._check_is_fitted()

        if self._independent_columns:
            if future_exog is None:
                raise ModelPredictionError(
                    "Bayesian PyMC requires future_exog (a DataFrame with 'ds' and the "
                    "columns produced by BayesianPyMCAdapter) to predict -- received None.",
                    model_name=self.model_name,
                )
            missing = [c for c in self._independent_columns if c not in future_exog.columns]
            if missing:
                raise ModelPredictionError(
                    f"future_exog is missing required column(s): {missing}",
                    model_name=self.model_name,
                )
        if future_exog is None or "ds" not in future_exog.columns:
            raise ModelPredictionError(
                "future_exog must contain a 'ds' column.", model_name=self.model_name
            )
        if len(future_exog) != horizon:
            raise ModelPredictionError(
                f"future_exog has {len(future_exog)} rows but horizon={horizon} was requested.",
                model_name=self.model_name,
            )

        future_exog = future_exog.reset_index(drop=True)
        t = self._time_steps(future_exog["ds"])
        ff_long, ff_short = self._fourier_terms(future_exog["ds"])
        exog = (
            future_exog[self._independent_columns].to_numpy(dtype=float)
            if self._independent_columns
            else np.empty((horizon, 0))
        )

        try:
            # mu draws at the future dates: (n_draws, horizon), pure numpy.
            mu = (
                self._post["intercept"][:, None]
                + self._post["slope"][:, None] * t[None, :]
                + self._post["beta_long"] @ ff_long.T
            )
            if ff_short is not None and "beta_short" in self._post:
                mu = mu + self._post["beta_short"] @ ff_short.T
            for j, col in enumerate(self._independent_columns):
                mu = mu + self._post["coef"][:, j][:, None] * exog[:, j][None, :]
        except Exception as e:  # noqa: BLE001
            raise ModelPredictionError(
                f"Bayesian PyMC failed during prediction ({type(e).__name__}: {e}).",
                model_name=self.model_name,
                original_exception=e,
            )

        def back(z: np.ndarray) -> np.ndarray:
            return np.clip(z * self._y_std + self._y_mean, 0.0, None)

        yhat = back(mu.mean(axis=0))
        yhat_lower = back(np.percentile(mu, self.ci_lower_pct, axis=0))
        yhat_upper = back(np.percentile(mu, self.ci_upper_pct, axis=0))

        return pd.DataFrame(
            {
                "ds": future_exog["ds"].reset_index(drop=True),
                "yhat": yhat,
                "yhat_lower": yhat_lower,
                "yhat_upper": yhat_upper,
            }
        )

    # ---------- time / Fourier ----------

    def _time_steps(self, ds: pd.Series) -> np.ndarray:
        """
        Position of each date along the series, in frequency steps from the
        reference date (days for daily, weeks for weekly). Used as the trend
        regressor; matches the notebook's positional time index but is
        computed from real dates so it stays consistent between fit and
        predict and is robust to gaps.
        """
        offset_days = (pd.to_datetime(ds).reset_index(drop=True) - self._reference_date)
        days = offset_days.dt.total_seconds().to_numpy() / 86400.0
        return days if self.frequency == "daily" else days / 7.0

    def _fourier_terms(self, ds: pd.Series):
        """
        Return (ff_long, ff_short). ff_long is the annual cycle for both
        frequencies; ff_short is the weekly cycle for daily data (else None).
        Both are built from absolute day offsets against the reference date.
        Column layout matches the notebook: [cos_1..cos_k, sin_1..sin_k].
        """
        offset_days = (pd.to_datetime(ds).reset_index(drop=True) - self._reference_date)
        day = offset_days.dt.total_seconds().to_numpy() / 86400.0
        ff_long = self._fourier_basis(day, _ANNUAL_PERIOD_DAYS)
        ff_short = (
            self._fourier_basis(day, _WEEKLY_PERIOD_DAYS) if self.use_short_seasonal else None
        )
        return ff_long, ff_short

    def _fourier_basis(self, day: np.ndarray, period_days: float) -> np.ndarray:
        k = np.arange(1, self.n_fourier + 1)
        ang = 2.0 * np.pi * day[:, None] * k[None, :] / period_days
        return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)

    # ---------- sampling / posterior ----------

    def _sample(self, y_sc, t, ff_long, ff_short, exog, *, slope_est, intercept_est, coef_prior_means):
        import pymc as pm  # lazy: keeps module import free of the heavy dependency

        n_fourier2 = 2 * self.n_fourier
        with pm.Model():
            slope = pm.Normal("slope", mu=slope_est, sigma=self.slope_sigma)
            intercept = pm.Normal("intercept", mu=intercept_est, sigma=self.intercept_sigma)
            mu_det = intercept + slope * t

            beta_long = pm.Normal("beta_long", mu=0.0, sigma=self.beta_fourier_sigma, shape=n_fourier2)
            mu_det = mu_det + pm.math.dot(ff_long, beta_long)

            if ff_short is not None:
                beta_short = pm.Normal(
                    "beta_short", mu=0.0, sigma=self.beta_fourier_sigma, shape=n_fourier2
                )
                mu_det = mu_det + pm.math.dot(ff_short, beta_short)

            for j in range(exog.shape[1]):
                sigma_j = self.driver_coef_sigma if self._driver_mask[j] else self.flag_coef_sigma
                coef = pm.Normal(f"coef_{j}", mu=coef_prior_means[j], sigma=sigma_j)
                mu_det = mu_det + coef * exog[:, j]

            sigma = pm.HalfNormal("sigma", sigma=self.sigma_sigma)
            nu = pm.Gamma("nu", alpha=self.nu_alpha, beta=self.nu_beta)
            pm.StudentT("y_obs", nu=nu, mu=mu_det, sigma=sigma, observed=y_sc)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    cores=self.cores,
                    target_accept=self.target_accept,
                    max_treedepth=self.max_treedepth,
                    init=self.init,
                    progressbar=False,
                    return_inferencedata=True,
                    random_seed=self.random_seed,
                    compute_convergence_checks=False,
                )
        return trace

    def _extract_posterior(self, trace, n_exog: int) -> None:
        """
        Flatten the posterior of every coefficient to plain numpy arrays
        (chains x draws collapsed to a single sample axis). predict() uses
        only these, so it never needs pymc imported and the reconstruction
        is pure numpy.
        """
        post = trace.posterior

        def flat(name: str) -> np.ndarray:
            arr = post[name].values
            # (chain, draw, *param) -> (chain*draw, *param)
            return arr.reshape(-1, *arr.shape[2:]) if arr.ndim > 2 else arr.reshape(-1)

        self._post: Dict[str, np.ndarray] = {
            "slope": flat("slope"),
            "intercept": flat("intercept"),
            "beta_long": flat("beta_long"),
        }
        if "beta_short" in post:
            self._post["beta_short"] = flat("beta_short")
        if n_exog:
            self._post["coef"] = np.column_stack([flat(f"coef_{j}") for j in range(n_exog)])
        else:
            self._post["coef"] = np.empty((self._post["slope"].shape[0], 0))

    @staticmethod
    def _convergence_summary(trace) -> Dict[str, float]:
        """
        A small, picklable diagnostics summary (max R-hat, divergence count)
        kept on the fitted wrapper for the user to inspect, computed lazily
        so a missing arviz never breaks fit().
        """
        summary: Dict[str, float] = {}
        try:
            import arviz as az

            rhat = az.rhat(trace)
            summary["max_rhat"] = float(max(float(v.max()) for v in rhat.data_vars.values()))
        except Exception:  # noqa: BLE001 -- diagnostics are best-effort
            summary["max_rhat"] = float("nan")
        try:
            summary["num_divergences"] = int(trace.sample_stats["diverging"].values.sum())
        except Exception:  # noqa: BLE001
            summary["num_divergences"] = -1
        return summary

    # ---------- serialization ----------

    def save(self, path: str) -> None:
        """
        Persist the fitted wrapper. Per BaseModel's note, PyMC's
        InferenceData is not pickled directly: it is written to a NetCDF
        sidecar ("<path>.nc") via arviz, and the rest of the wrapper (the
        numpy posterior draws predict() actually uses, plus scalars) is
        pickled. predict() depends only on the pickled numpy state, so a
        reload works even if the sidecar is absent.
        """
        self._check_is_fitted()
        import pickle

        trace = getattr(self, "trace", None)
        self.trace = None
        self._fitted_model = None
        try:
            if trace is not None:
                try:
                    trace.to_netcdf(f"{path}.nc")
                except Exception:  # noqa: BLE001 -- sidecar is best-effort
                    pass
            with open(path, "wb") as f:
                pickle.dump(self, f)
        finally:
            self.trace = trace
            self._fitted_model = trace

    @classmethod
    def load(cls, path: str) -> "BayesianPyMCWrapper":
        import os
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)
        sidecar = f"{path}.nc"
        if os.path.exists(sidecar):
            try:
                import arviz as az

                model.trace = az.from_netcdf(sidecar)
                model._fitted_model = model.trace
            except Exception:  # noqa: BLE001
                pass
        return model
