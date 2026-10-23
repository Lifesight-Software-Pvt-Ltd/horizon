"""
Bayesian PyMC model wrapper.

A Bayesian structural time-series model: a linear trend, a Fourier
seasonal basis (annual for both frequencies once a fit spans 183 days,
plus a weekly cycle for daily data), and a linear regression on the user's independent variables,
observed through a Student-T likelihood (heavy tails absorb spikes that
would drag a Gaussian fit). The posterior is sampled with NUTS.

Expects train_df already shaped by BayesianPyMCAdapter: columns "ds", raw
"y", and standardised independent columns. Everything model-specific lives
here -- the target standardisation, the Fourier terms, the priors, the
sampler, and the back-transform.

Design, and how it maps onto the reference implementation this was ported
from:

- That implementation built one Deterministic `mu` over the WHOLE series (train +
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
  order (`n_fourier`) is an ordinary grid hyperparameter. It sets the annual
  order; the weekly cycle stops at order 3 (higher orders alias lower ones on
  whole days), and a fit spanning fewer than ANNUAL_MIN_SPAN_DAYS has no
  annual terms at all.

- Coefficient priors: an independent column that looks like a continuous
  driver (more than two distinct standardised values -- e.g. spend,
  impressions) gets an INFORMATIVE prior centred on its ordinary-least-
  squares slope against the standardised target, mirroring the reference
  implementation's polyfit-based spend prior. A binary flag (<= two distinct values) gets a
  plain zero-centred prior. The classification keys off the standardised
  column the adapter hands over, so no metadata crosses the boundary.

- Uncertainty: yhat_lower/upper are a posterior-PREDICTIVE interval -- the
  3rd/97th percentiles of draws from `y ~ StudentT(nu, mu, sigma)`, not of
  `mu` alone. The reference implementation (and an earlier cut of this
  wrapper) banded `mu`, the latent mean, which omits the observation noise and badly under-covers
  (only ~34% of real actuals fell inside its nominal 94% band on daily
  data); sampling the likelihood adds that noise back so the band covers
  ~(ci_upper - ci_lower)% of outcomes. These bounds are what the wrapper
  returns when used on its own: HorizonForecaster discards them and gives
  this model, like every other, the calibrated 95% band built from its
  validation errors (evaluation/bands.py; see BaseModel.predict docstring).

Selection: PyMC uses the SAME forecaster-level grid search as the other
models. Because NUTS sampling is expensive, the default grid is small: daily
varies `n_fourier` x `beta_fourier_sigma` at a fixed `target_accept`,
weekly varies `n_fourier` x `target_accept`. Every candidate is sampled at
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
from horizon_forecaster.models._pytensor_compat import ensure_macos_linker_compat
from horizon_forecaster.models.base import BaseModel

# Annual period in DAYS. The long Fourier cycle is built from absolute day
# offsets for both frequencies -- a year is a year regardless of sampling
# rate -- which also makes it robust to gaps, unlike a positional index.
_ANNUAL_PERIOD_DAYS = 365.25

# Weekly period in DAYS, for the short (intra-week) cycle used on daily data.
_WEEKLY_PERIOD_DAYS = 7.0

# Highest Fourier order used for the weekly cycle. On whole-day dates order 4
# is order 3 mirrored (cos(2*pi*4d/7) == cos(2*pi*3d/7), sin(2*pi*4d/7) ==
# -sin(2*pi*3d/7)), so orders above 3 add only perfectly collinear columns:
# no new shape, just a ridge NUTS crawls along. n_fourier still sets the
# annual order.
_WEEKLY_MAX_ORDER = 3

# A fit whose dates span fewer days than this has no annual terms. From less
# than half a year the model has to guess the other half of the yearly cycle,
# and on real data that guess swung forecasts wildly: test WAPE 11.5 with 158
# days of training, 73.9 with 150, 49.8 with 136 (13.5-17.4 without annual
# terms). From about 7 months up the annual terms paid off whenever the
# series had a yearly cycle and cost little when it did not.
ANNUAL_MIN_SPAN_DAYS = 183

# A fit is reported as poorly converged above this R-hat or below this bulk
# effective sample size. Deliberately looser than the textbook 1.01 / 400,
# which flagged a quarter of fits whose forecasts were fine; these flag only
# clearly broken sampling.
CONVERGENCE_MAX_RHAT = 1.05
CONVERGENCE_MIN_ESS = 100


def span_days(ds) -> int:
    """Calendar days from the first to the last date, inclusive (0 when empty)."""
    dates = pd.to_datetime(pd.Series(ds))
    if dates.empty:
        return 0
    return int((dates.max() - dates.min()).days) + 1


# A column with more than this many distinct standardised values is treated
# as a continuous driver (informative prior); at or below it, a binary/near-
# binary flag (zero-centred prior). 2 cleanly separates 0/1 flags from real
# continuous regressors.
_CONTINUOUS_MIN_DISTINCT = 2


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    NaN-safe least-squares slope of y on x. Returns 0.0 if fewer than two
    finite (x, y) pairs remain. Used for prior means, so a NaN in a driver
    can't leak a raw LinAlgError out of fit() before the sampler's guard.
    """
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


class BayesianPyMCWrapper(BaseModel):
    """Wraps a PyMC Bayesian structural model to the BaseModel contract."""

    model_name = "bayesianpymc"

    _FREQUENCY_MIN_ROWS_DEFAULTS = {"daily": 35, "weekly": 18}
    # Every key __init__ reads from params; keep in step with it.
    accepted_params = frozenset(
        {
            "n_fourier",
            "target_accept",
            "draws",
            "tune",
            "chains",
            "cores",
            "max_treedepth",
            "init",
            "slope_sigma",
            "intercept_sigma",
            "beta_fourier_sigma",
            "driver_coef_sigma",
            "flag_coef_sigma",
            "sigma_sigma",
            "nu_alpha",
            "nu_beta",
            "ci_lower_pct",
            "ci_upper_pct",
            "use_short_seasonal",
        }
    )
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
        self._weekly_order = min(self.n_fourier, _WEEKLY_MAX_ORDER)
        # Decided in fit() from the span of the training dates; see
        # ANNUAL_MIN_SPAN_DAYS.
        self.annual_terms = True
        # Set from the training target in fit(); None here so an unfitted
        # (or older, unpickled) wrapper degrades to "don't clip" rather
        # than raising AttributeError inside predict().
        self._target_min: Optional[float] = None

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
        self.annual_terms = span_days(train_df["ds"]) >= ANNUAL_MIN_SPAN_DAYS
        self._independent_columns: List[str] = [c for c in train_df.columns if c not in ("ds", "y")]

        y_raw = train_df["y"].to_numpy(dtype=float)
        self._y_mean = float(np.nanmean(y_raw))
        self._y_std = float(np.nanstd(y_raw)) + 1e-8
        # Recorded so predict() can decide whether flooring at zero is
        # appropriate for THIS target, mirroring CatBoostWrapper's
        # _target_min. See the note in back() below.
        self._target_min = float(np.nanmin(y_raw))
        y_sc = (y_raw - self._y_mean) / self._y_std

        t = self._time_steps(train_df["ds"])
        ff_long, ff_short = self._fourier_terms(train_df["ds"])
        exog = (
            train_df[self._independent_columns].to_numpy(dtype=float)
            if self._independent_columns
            else np.empty((n, 0))
        )

        # Prior means, from the reference implementation (NaN-safe, so a
        # missing value in a driver can't leak a raw LinAlgError out of fit()).
        slope_est = _ols_slope(t, y_sc)
        tail = min(52, n)
        intercept_est = float(np.nanmean(y_sc[-tail:]))

        # Per-column coefficient prior means: OLS slope of the standardised
        # target on a continuous driver (informative), else 0 for a flag.
        self._driver_mask = [
            np.unique(exog[:, j][np.isfinite(exog[:, j])]).size > _CONTINUOUS_MIN_DISTINCT
            for j in range(exog.shape[1])
        ]
        coef_prior_means = [
            (
                _ols_slope(exog[:, j], y_sc)
                if (self._driver_mask[j] and np.nanstd(exog[:, j]) > 1e-12)
                else 0.0
            )
            for j in range(exog.shape[1])
        ]

        try:
            trace = self._sample(
                y_sc,
                t,
                ff_long,
                ff_short,
                exog,
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
            """
            Undo the target z-scoring, flooring at zero only when the
            training target itself never went negative.

            The floor used to be unconditional, which silently corrupted
            any target that legitimately goes below zero -- net revenue
            after refunds, inventory deltas, week-on-week change. Such a
            series would have every negative prediction quietly rewritten
            to 0, which is not a conservative approximation but a wrong
            number presented with full confidence. Now data-driven, the
            same rule CatBoostWrapper and NeuralProphetWrapper already
            apply: a target observed to be non-negative keeps the floor,
            and one that dips negative is returned unaltered.
            """
            raw = z * self._y_std + self._y_mean
            if self._target_min is not None and self._target_min >= 0:
                return np.clip(raw, 0.0, None)
            return raw

        # Point forecast: posterior mean of the latent mean mu (= the
        # predictive mean for a Student-T with nu > 1), so MAPE is unchanged.
        yhat = back(mu.mean(axis=0))

        # Bounds are a POSTERIOR-PREDICTIVE interval, not a band on mu. A
        # percentile of mu alone answers "how uncertain is the mean curve"
        # (narrow on long histories) and omits the per-period observation
        # noise, so it badly under-covers -- real actuals include that noise.
        # Drawing y ~ StudentT(nu, mu, sigma) per posterior draw adds it back,
        # giving an interval that actually covers ~(ci_upper - ci_lower)% of
        # outcomes. Seeded so predict() stays reproducible.
        rng = np.random.default_rng(self.random_seed)
        nu = np.broadcast_to(self._post["nu"][:, None], mu.shape)
        y_pred = mu + self._post["sigma"][:, None] * rng.standard_t(nu)
        yhat_lower = back(np.percentile(y_pred, self.ci_lower_pct, axis=0))
        yhat_upper = back(np.percentile(y_pred, self.ci_upper_pct, axis=0))

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
        regressor; matches the reference implementation's positional time
        index but is
        computed from real dates so it stays consistent between fit and
        predict and is robust to gaps.
        """
        offset_days = pd.to_datetime(ds).reset_index(drop=True) - self._reference_date
        days = offset_days.dt.total_seconds().to_numpy() / 86400.0
        return days if self.frequency == "daily" else days / 7.0

    def _fourier_terms(self, ds: pd.Series):
        """
        Return (ff_long, ff_short). ff_long is the annual cycle for both
        frequencies, with no columns when the fit had too short a span for
        it; ff_short is the weekly cycle for daily data (else None), at most
        order _WEEKLY_MAX_ORDER. Both are built from absolute day offsets
        against the reference date. Column layout is [cos_1..cos_k,
        sin_1..sin_k].

        The getattr defaults keep a model pickled before these settings
        existed predicting exactly as it was fitted.
        """
        offset_days = pd.to_datetime(ds).reset_index(drop=True) - self._reference_date
        day = offset_days.dt.total_seconds().to_numpy() / 86400.0
        if getattr(self, "annual_terms", True):
            ff_long = self._fourier_basis(day, _ANNUAL_PERIOD_DAYS, self.n_fourier)
        else:
            ff_long = np.empty((len(day), 0))
        weekly_order = getattr(self, "_weekly_order", self.n_fourier)
        ff_short = (
            self._fourier_basis(day, _WEEKLY_PERIOD_DAYS, weekly_order)
            if self.use_short_seasonal
            else None
        )
        return ff_long, ff_short

    @staticmethod
    def _fourier_basis(day: np.ndarray, period_days: float, order: int) -> np.ndarray:
        k = np.arange(1, order + 1)
        ang = 2.0 * np.pi * day[:, None] * k[None, :] / period_days
        return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)

    # ---------- sampling / posterior ----------

    def _sample(
        self, y_sc, t, ff_long, ff_short, exog, *, slope_est, intercept_est, coef_prior_means
    ):
        # Must run before PyMC builds anything: on some macOS toolchains PyTensor's
        # '-ld64' flag breaks every compile. See _pytensor_compat for details.
        ensure_macos_linker_compat()
        import pymc as pm  # lazy: keeps module import free of the heavy dependency

        with pm.Model():
            slope = pm.Normal("slope", mu=slope_est, sigma=self.slope_sigma)
            intercept = pm.Normal("intercept", mu=intercept_est, sigma=self.intercept_sigma)
            mu_det = intercept + slope * t

            if ff_long.shape[1]:
                beta_long = pm.Normal(
                    "beta_long", mu=0.0, sigma=self.beta_fourier_sigma, shape=ff_long.shape[1]
                )
                mu_det = mu_det + pm.math.dot(ff_long, beta_long)

            if ff_short is not None:
                beta_short = pm.Normal(
                    "beta_short", mu=0.0, sigma=self.beta_fourier_sigma, shape=ff_short.shape[1]
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

        slope = flat("slope")
        self._post: Dict[str, np.ndarray] = {
            "slope": slope,
            "intercept": flat("intercept"),
            # No annual terms -> no columns, so predict()'s matrix product
            # contributes zero without a special case.
            "beta_long": (
                flat("beta_long") if "beta_long" in post else np.empty((slope.shape[0], 0))
            ),
            # sigma/nu are the Student-T likelihood's scale and tails. They are
            # NOT part of the mean function, but predict() needs them to build a
            # posterior-PREDICTIVE band (observation noise), not just a band on
            # the mean -- so they are retained here alongside the coefficients.
            "sigma": flat("sigma"),
            "nu": flat("nu"),
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
        A small, picklable diagnostics summary (max R-hat, smallest bulk
        effective sample size, divergence count) kept on the fitted wrapper
        for the user to inspect, computed at the end of fit(). The
        forecaster warns from it (see CONVERGENCE_MAX_RHAT). The arviz
        import is deferred so a missing arviz never breaks fit(); R-hat is
        NaN for a single chain.
        """
        summary: Dict[str, float] = {}
        try:
            import arviz as az

            rhat = az.rhat(trace)
            summary["max_rhat"] = float(max(float(v.max()) for v in rhat.data_vars.values()))
        except Exception:  # noqa: BLE001 -- diagnostics are best-effort
            summary["max_rhat"] = float("nan")
        try:
            import arviz as az

            ess = az.ess(trace, method="bulk")
            summary["min_ess_bulk"] = float(min(float(v.min()) for v in ess.data_vars.values()))
        except Exception:  # noqa: BLE001 -- diagnostics are best-effort
            summary["min_ess_bulk"] = float("nan")
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
        import pickle  # nosec B403 -- model save/load, trusted files only

        trace = getattr(self, "trace", None)
        self.trace = None
        self._fitted_model = None
        try:
            if trace is not None:
                try:
                    trace.to_netcdf(f"{path}.nc")
                except Exception:  # noqa: BLE001  # nosec B110 -- sidecar is best-effort
                    pass
            with open(path, "wb") as f:
                pickle.dump(self, f)
        finally:
            self.trace = trace
            self._fitted_model = trace

    @classmethod
    def load(cls, path: str) -> "BayesianPyMCWrapper":
        import os
        import pickle  # nosec B403 -- model save/load, trusted files only

        with open(path, "rb") as f:
            model = pickle.load(f)  # nosec B301 -- trusted: written by save()
        sidecar = f"{path}.nc"
        if os.path.exists(sidecar):
            try:
                import arviz as az

                model.trace = az.from_netcdf(sidecar)
                model._fitted_model = model.trace
            except Exception:  # noqa: BLE001  # nosec B110 -- sidecar is optional
                pass
        return model
