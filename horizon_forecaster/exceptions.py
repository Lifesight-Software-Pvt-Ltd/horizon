"""
Custom exception hierarchy for horizon_forecaster.

Design principle: every exception carries structured, machine-readable
attributes in addition to a human-readable message -- `model_name` on
the base class, plus per-subclass detail such as required_periods/
available_periods, original_exception, mape_value/allowed_max,
param_name/provided_value/valid_range, and provided_frequency. This lets
`evaluation/ensemble.py` build the "rejected models" report, and
`exporters.py` turn it into a table, programmatically rather than by
parsing exception strings.
"""

from typing import Optional


class HorizonForecasterError(Exception):
    """
    Base exception for all horizon_forecaster errors.
    """

    def __init__(self, message: str, model_name: Optional[str] = None):
        self.model_name = model_name
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.model_name:
            return f"[{self.model_name}] {self.message}"
        return self.message


class InsufficientDataError(HorizonForecasterError):
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        required_periods: Optional[int] = None,
        available_periods: Optional[int] = None,
    ):
        self.required_periods = required_periods
        self.available_periods = available_periods
        super().__init__(message, model_name)


class ModelConvergenceError(HorizonForecasterError):
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.original_exception = original_exception
        super().__init__(message, model_name)


class ModelPredictionError(HorizonForecasterError):
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.original_exception = original_exception
        super().__init__(message, model_name)


class ThresholdRejectionError(HorizonForecasterError):
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        mape_value: Optional[float] = None,
        allowed_min: Optional[float] = None,
        allowed_max: Optional[float] = None,
    ):
        self.mape_value = mape_value
        self.allowed_min = allowed_min
        self.allowed_max = allowed_max
        super().__init__(message, model_name)


class AllModelsRejectedError(HorizonForecasterError):
    def __init__(self, message: str, rejected_models: Optional[list] = None):
        self.rejected_models = rejected_models or []
        super().__init__(message)


class InvalidConfigError(HorizonForecasterError):
    def __init__(
        self,
        message: str,
        param_name: Optional[str] = None,
        provided_value=None,
        valid_range: Optional[tuple] = None,
    ):
        self.param_name = param_name
        self.provided_value = provided_value
        self.valid_range = valid_range
        super().__init__(message)


class UnsupportedFrequencyError(HorizonForecasterError):
    def __init__(self, message: str, provided_frequency: Optional[str] = None):
        self.provided_frequency = provided_frequency
        super().__init__(message)
