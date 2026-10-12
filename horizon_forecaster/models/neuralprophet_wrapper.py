"""
NeuralProphet model wrapper.

Expects train_df already shaped by NeuralProphetAdapter (columns "ds",
"y", plus any regressor columns under their original names).

Key design decisions, several of them deliberate departures from the
reference implementation this was ported from:

- **n_lags = 0 (no autoregression).** That implementation grid-searched
  n_lags in {0, 7} and predicted on a frame CONTAINING the test period's
  real `y` values, which with n_lags > 0 makes each test point an
  effectively one-step-ahead prediction that has seen recent actuals.
  That is not comparable to Prophet/CatBoost/SARIMAX (which forecast the
  whole window blind), and feeding that flattered MAPE into inverse-MAPE
  ensemble weighting would systematically over-weight this model. It also
  can't be expressed through this library's contract, which has no actuals
  to hand predict(). Separately, NeuralProphet fixes `n_forecasts` at
  construction, whereas BaseModel is fit-once-then-predict-any-horizon --
  supporting AR properly would mean refitting per horizon and breaking the
  cached-refit design. So AR is deferred; v1 is a pure decomposition model.

- **The future frame is built directly, not via make_future_dataframe().**
  With n_lags = 0 a prediction depends only on ds + regressors, so the
  frame handed to predict() is just future_exog with y = NaN. This is why
  the wrapper does NOT need to retain the training history, which in turn
  keeps save()/load() to a clean model-only operation.

- **CPU is pinned and logging is silenced.** NeuralProphet runs on
  PyTorch Lightning and will otherwise pick up an accelerator (MPS on a
  Mac, CPU elsewhere), producing different numbers for identical code. It is
  also extremely chatty; progress=False alone does not quiet it.

- **Both epochs and learning_rate are pinned** via the config grid. If
  either is left None, NeuralProphet runs a learning-rate range test and
  auto-sizes epochs from the data -- slow, and it makes two runs on the
  same data non-comparable.

- **yearly_seasonality defaults to "auto", not True.** Forcing yearly
  seasonality on short history is a failure mode this project has already
  hit twice: Prophet produced negative forecasts, and SARIMAX's annual
  Fourier terms swung MAPE from ~6% to ~65% on under half a year of data.

- **collect_metrics is disabled** (in default_params.yaml). This is a
  correctness requirement, not a tuning choice. NeuralProphet attaches a
  Lightning logger whenever metrics are enabled, and a logger makes
  trainer.fit() run _log_hyperparams() -> save_hparams_to_yaml() ->
  OmegaConf.save(). OmegaConf then rejects ConfigTrain's
  `loss_func: Union[str, _Loss, Callable]` with ConfigValueError and fit()
  dies before training. Disabling metrics makes configure_trainer() set
  `logger = False`, so the path never runs. Costs nothing here: MAPE is
  computed by the orchestrator, and NeuralProphet's own docs note that
  collecting metrics slows training.

- **No uncertainty bounds in v1**, per models/base.py's contract note.
  NeuralProphet can produce them via `quantiles=[...]`, which is a natural
  follow-up now that the plotting/export layers handle bounds, but it adds
  output heads and training cost.
"""

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from horizon_forecaster.exceptions import (
    InsufficientDataError,
    ModelConvergenceError,
    ModelPredictionError,
)
from horizon_forecaster.models.base import BaseModel

# NeuralProphet/Lightning emit a large volume of logging that would otherwise
# drown the orchestrator's own output. Silenced at CRITICAL rather than ERROR
# because NeuralProphet logs genuinely harmless conditions AT error level --
# e.g. "NP.plotly: Importing plotly failed. Interactive plots will not work.",
# which is irrelevant here since this library plots via matplotlib. Child
# loggers are listed explicitly rather than relying on parent inheritance, in
# case NeuralProphet sets levels on them directly.
_NOISY_LOGGERS = (
    "NP",
    "NP.plotly",
    "NP.utils",
    "NP.config",
    "NP.df_utils",
    "neuralprophet",
    "pytorch_lightning",
    "lightning",
    "lightning.pytorch",
)

# Wrapper-level settings that must NOT be forwarded to the NeuralProphet
# constructor (it would raise on unexpected keyword arguments).
_NON_CONSTRUCTOR_PARAMS = ("freq",)


class NeuralProphetWrapper(BaseModel):
    """Wraps NeuralProphet to conform to the BaseModel contract."""

    model_name = "neuralprophet"

    # Higher than Prophet's 14/8: this is a neural model, and fitting one
    # on a fortnight of data produces noise rather than a forecast. A
    # higher floor makes it decline cleanly instead.
    _FREQUENCY_MIN_ROWS_DEFAULTS = {"daily": 30, "weekly": 16}
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
        # "auto" lets NeuralProphet infer the spacing. Deriving a pandas
        # alias instead is a trap for weekly data: "W" anchors to a
        # specific weekday, which need not match 7-day-spaced rows, and a
        # mismatch makes NeuralProphet silently resample/impute.
        self.freq = self.params.get("freq", "auto")

    # ---------- fit ----------

    def fit(self, train_df: pd.DataFrame, target_column: str = "y") -> "NeuralProphetWrapper":
        if "ds" not in train_df.columns or "y" not in train_df.columns:
            raise ModelPredictionError(
                "NeuralProphetWrapper.fit() expects train_df with 'ds' and 'y' columns "
                "(i.e. data already shaped by NeuralProphetAdapter).",
                model_name=self.model_name,
            )

        if len(train_df) < self.min_rows_required:
            raise InsufficientDataError(
                f"NeuralProphet requires at least {self.min_rows_required} rows to fit reliably.",
                model_name=self.model_name,
                required_periods=self.min_rows_required,
                available_periods=len(train_df),
            )

        regressor_columns = [c for c in train_df.columns if c not in ("ds", "y")]
        constructor_params = {
            k: v for k, v in self.params.items() if k not in _NON_CONSTRUCTOR_PARAMS
        }

        try:
            # Imported here (not at module level) so importing this module
            # doesn't require NeuralProphet's heavy torch/Lightning chain
            # unless this specific wrapper is actually used -- same pattern
            # as ProphetWrapper's cmdstanpy-backed import.
            from neuralprophet import NeuralProphet, set_random_seed

            self._silence_loggers()
            set_random_seed(self.random_seed)

            model = NeuralProphet(**constructor_params)
            for col in regressor_columns:
                model.add_future_regressor(col)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train_df, freq=self.freq, progress=False)
        except Exception as e:
            # Surface the underlying cause in the message, not just on the
            # exception object: the most likely first-run failure is a
            # constructor keyword this NeuralProphet version doesn't accept,
            # and "failed to fit" alone would send you looking for a
            # convergence problem that isn't there.
            raise ModelConvergenceError(
                f"NeuralProphet failed to fit ({type(e).__name__}: {e}).",
                model_name=self.model_name,
                original_exception=e,
            )

        self._fitted_model = model
        self._regressor_columns = regressor_columns
        # Data-driven clipping decision, mirroring CatBoostWrapper: only
        # clip negative predictions when the training target was itself
        # never negative. Blanket clipping would corrupt a genuinely
        # signed series.
        target_values = pd.to_numeric(train_df["y"], errors="coerce").dropna()
        self._target_min = float(target_values.min()) if len(target_values) else None
        self.is_fitted = True
        return self

    # ---------- predict ----------

    def predict(self, horizon: int, future_exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self._check_is_fitted()

        if future_exog is None:
            raise ModelPredictionError(
                "NeuralProphet requires future_exog (a DataFrame with 'ds' and any "
                "required regressor columns) to predict -- received None.",
                model_name=self.model_name,
            )

        if "ds" not in future_exog.columns:
            raise ModelPredictionError(
                "future_exog must contain a 'ds' column.", model_name=self.model_name
            )

        if len(future_exog) != horizon:
            raise ModelPredictionError(
                f"future_exog has {len(future_exog)} rows but horizon={horizon} was requested.",
                model_name=self.model_name,
            )

        missing_regressors = [c for c in self._regressor_columns if c not in future_exog.columns]
        if missing_regressors:
            raise ModelPredictionError(
                f"future_exog is missing required regressor column(s): {missing_regressors}",
                model_name=self.model_name,
            )

        # With n_lags = 0 there is no autoregressive dependency, so the
        # frame NeuralProphet needs is exactly future_exog plus an empty
        # target column -- no training history required.
        future_frame = future_exog[["ds"] + self._regressor_columns].copy()
        future_frame["y"] = np.nan

        try:
            self._silence_loggers()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = self._fitted_model.predict(future_frame)
        except Exception as e:
            raise ModelPredictionError(
                f"NeuralProphet failed during prediction ({type(e).__name__}: {e}).",
                model_name=self.model_name,
                original_exception=e,
            )

        yhat_column = self._resolve_yhat_column(forecast)
        yhat = pd.to_numeric(forecast[yhat_column], errors="coerce").to_numpy(dtype=float)

        if self._target_min is not None and self._target_min >= 0:
            yhat = np.clip(yhat, 0.0, None)

        return pd.DataFrame({"ds": future_exog["ds"].reset_index(drop=True), "yhat": yhat})

    # ---------- internals ----------

    @staticmethod
    def _resolve_yhat_column(forecast_df: pd.DataFrame) -> str:
        """
        NeuralProphet's prediction column name is version- and
        configuration-dependent: "yhat1" in current versions, "yhat" in
        older ones, and yhat1..yhatN when multi-step forecasting is on.
        Prefer the canonical names, else the first yhat* column.
        """
        for candidate in ("yhat1", "yhat"):
            if candidate in forecast_df.columns:
                return candidate
        yhat_columns = sorted(c for c in forecast_df.columns if c.startswith("yhat"))
        if yhat_columns:
            return yhat_columns[0]
        raise ModelPredictionError(
            f"NeuralProphet returned no yhat column. Got: {list(forecast_df.columns)}",
            model_name=NeuralProphetWrapper.model_name,
        )

    @staticmethod
    def _silence_loggers() -> None:
        # CRITICAL, not ERROR: NeuralProphet logs harmless conditions at
        # ERROR level (the plotly import probe), so ERROR would still let
        # them through.
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.CRITICAL)

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """
        Overrides BaseModel's generic pickle default. NeuralProphet's
        fitted state is a PyTorch model, which BaseModel's docstring
        explicitly calls out as unsafe to pickle generically. This uses
        NeuralProphet's own save utility for the model, plus a sidecar
        pickle holding the wrapper-level state needed to rebuild this
        object (frequency, params, min_rows_required, random_seed,
        regressor columns, and the clipping decision).
        """
        import pickle  # nosec B403 -- model save/load, trusted files only

        from neuralprophet import save as np_save

        self._check_is_fitted()

        np_save(self._fitted_model, path)
        envelope = {
            "frequency": self.frequency,
            "params": self.params,
            "min_rows_required": self.min_rows_required,
            "random_seed": self.random_seed,
            "regressor_columns": self._regressor_columns,
            "target_min": self._target_min,
        }
        with open(f"{path}.wrapper", "wb") as f:
            pickle.dump(envelope, f)

    @classmethod
    def load(cls, path: str) -> "NeuralProphetWrapper":
        """
        Reload a NeuralProphetWrapper saved with save(), using
        NeuralProphet's own load utility plus the sidecar envelope.
        """
        import pickle  # nosec B403 -- model save/load, trusted files only

        with open(f"{path}.wrapper", "rb") as f:
            envelope = pickle.load(f)  # nosec B301 -- trusted: written by save()

        instance = cls(
            frequency=envelope["frequency"],
            params=envelope["params"],
            min_rows_required=envelope["min_rows_required"],
            random_seed=envelope["random_seed"],
        )
        instance._fitted_model = cls._load_torch_model(path)
        instance._regressor_columns = envelope["regressor_columns"]
        instance._target_min = envelope["target_min"]
        instance.is_fitted = True
        return instance

    @staticmethod
    def _load_torch_model(path: str):
        """
        Load the NeuralProphet model, tolerating PyTorch's weights_only flip.

        NeuralProphet's own load() delegates to torch.load(). PyTorch >= 2.6
        changed torch.load's `weights_only` default from False to True, and
        NeuralProphet saves a full Python object (not a tensor state dict),
        so the new default refuses to unpickle it:

            UnpicklingError: Unsupported global: GLOBAL
            neuralprophet.forecaster.NeuralProphet was not an allowed global

        This is genuinely version-dependent -- it passed local testing on an
        older torch and only surfaced in CI on torch >= 2.6.

        We try NeuralProphet's own loader first (correct on older torch), then
        fall back to torch.load with weights_only=False. Allowlisting classes
        via add_safe_globals() was the other option, but a saved
        NeuralProphet pulls in a whole tree of its own config classes, so that
        becomes whack-a-mole across versions. weights_only=False is
        appropriate here because the file was written by this library's own
        save() -- a trusted source, which is precisely the condition PyTorch's
        documentation attaches to that flag.

        **The fallback must also restore the trainer.** NeuralProphet's
        `save()` deliberately sets `forecaster.trainer` and
        `forecaster.model._trainer` to None before writing, because a
        PyTorch Lightning `Trainer` is not picklable; its `load()` then calls
        `restore_trainer()` to rebuild one. A bare `torch.load()` skips that
        step, so the model comes back with `trainer = None` and the next
        `predict()` dies deep inside NeuralProphet with a bare
        `AttributeError: 'NoneType' object has no attribute 'predict'`.

        This is the second CI-only failure on this same method, from the same
        cause: the fallback path silently diverged from what NeuralProphet's
        own loader does. Runs on torch < 2.6 never reach the fallback, which
        is why tests/test_neuralprophet_compat.py stubs torch to exercise it
        on any version.
        """
        from neuralprophet import load as np_load

        try:
            return np_load(path)
        except Exception:
            import torch

            model = torch.load(path, map_location="cpu", weights_only=False)  # nosec B614
            # Mirror np_load()'s own post-load step. "cpu" (rather than
            # None, which would defer to the model's stored accelerator) to
            # stay consistent with map_location above -- this branch has
            # just pinned everything to CPU.
            restore_trainer = getattr(model, "restore_trainer", None)
            if restore_trainer is not None:  # pragma: no branch - upstream rename guard
                restore_trainer(accelerator="cpu")
            return model
