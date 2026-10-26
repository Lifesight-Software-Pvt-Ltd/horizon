"""
horizon_forecaster: ensemble time-series forecasting library.

Fits a diverse panel of models on a single time series with optional
exogenous drivers, then combines them, so you don't have to know in
advance which model suits your series.

The public API is re-exported here, so the common case is a single
import:

    from horizon_forecaster import HorizonForecaster

Submodules stay importable for everything else (plotting, exporters, the
individual wrappers and adapters). Note that importing this package does
NOT pull in Prophet, CatBoost, statsmodels, torch or PyMC -- each wrapper
imports its own backend lazily, inside fit(), so a script that only needs
the metrics or the ensemble maths pays none of that cost.
"""

from horizon_forecaster.evaluation.bands import BAND_LEVEL, BandCalibration
from horizon_forecaster.evaluation.ensemble import (
    DEFAULT_ENSEMBLE_METHOD,
    ENSEMBLE_METHODS,
    EnsembleResult,
    build_ensemble,
    build_ensembles,
)
from horizon_forecaster.evaluation.metrics import (
    MapeResult,
    error_correlation,
    mae,
    mape,
    rmse,
    seasonal_naive,
    wape,
)
from horizon_forecaster.exceptions import (
    AllModelsRejectedError,
    HorizonForecasterError,
    InsufficientDataError,
    InvalidConfigError,
    ModelConvergenceError,
    ModelPredictionError,
    ThresholdRejectionError,
    UnsupportedFrequencyError,
)
from horizon_forecaster.forecaster import (
    MODEL_REGISTRY,
    EvaluationResult,
    ForecastResult,
    HorizonForecaster,
)
from horizon_forecaster.utils import DEFAULT_EVALUATION_WINDOWS, SUPPORTED_FREQUENCIES

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # orchestration
    "HorizonForecaster",
    "EvaluationResult",
    "ForecastResult",
    "MODEL_REGISTRY",
    # ensembling
    "EnsembleResult",
    "build_ensemble",
    "build_ensembles",
    "ENSEMBLE_METHODS",
    "DEFAULT_ENSEMBLE_METHOD",
    # bands
    "BandCalibration",
    "BAND_LEVEL",
    # metrics
    "MapeResult",
    "mape",
    "wape",
    "rmse",
    "mae",
    "seasonal_naive",
    "error_correlation",
    # configuration
    "SUPPORTED_FREQUENCIES",
    "DEFAULT_EVALUATION_WINDOWS",
    # exceptions
    "HorizonForecasterError",
    "InsufficientDataError",
    "ModelConvergenceError",
    "ModelPredictionError",
    "ThresholdRejectionError",
    "AllModelsRejectedError",
    "InvalidConfigError",
    "UnsupportedFrequencyError",
]
