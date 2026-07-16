"""
Ensemble combination logic for horizon_forecaster (Phase 4).

Combines multiple models' per-period "yhat" predictions into a single
weighted ensemble, after filtering out any model whose MAPE exceeds an
accuracy cutoff (`mape_threshold`). Deliberately a pure,
standalone module -- like evaluation/metrics.py, it has no dependency
on forecaster.py, HorizonForecaster, or any model/adapter class.
forecaster.py (HorizonForecaster.evaluate()/forecast()) is the only
caller.

Design constraint carried over from models/base.py's BaseModel.predict()
contract: this module operates on the "yhat" column ONLY. Uncertainty
bounds (yhat_lower/yhat_upper) are never combined -- averaging
confidence/credible intervals across statistically different model
types (Prophet's trend simulation vs. CatBoost's point estimate, and
eventually PyMC's posterior) has no coherent interpretation. Per-model
bounds remain available on each model's own EvaluationResult/
ForecastResult for individual-model plots/CSVs.

Why a model can be "rejected" from the ensemble without evaluate()/
forecast() failing: a model that fits and predicts successfully but
scores above mape_threshold is not a failure in the same sense
as InsufficientDataError/ModelConvergenceError (see
forecaster.py's evaluate_errors/forecast_errors) -- it produced a real
result, it just isn't accurate enough to be trusted in the combined
ensemble. This module reports that distinction via `rejected_models`
(a dict of ThresholdRejectionError, one per excluded model) rather than
raising. If EVERY model is rejected, `build_ensemble()` still does not
raise -- it returns an EnsembleResult with `combined=None` and empty
`weights`, so the caller's own per-model results (which may still be
perfectly valid) are never discarded just because no ensemble could be
formed. This is a deliberate difference from AllModelsRejectedError,
which forecaster.py reserves for a stricter case: not one single model
produced ANY result at all.

KNOWN LIMITATION (confirmed via direct testing, not theoretical): a
weighted ensemble is NOT guaranteed to beat the single best individual
model's MAPE, even with inverse-MAPE weighting. Verified directly on a
synthetic dataset: Prophet alone scored 1.31% MAPE, CatBoost alone
scored 12.77%, and the inverse-MAPE-weighted ensemble of the two
scored 1.58% -- worse than Prophet alone, despite Prophet receiving
~91% of the weight. This happens because the weights are static across
the whole horizon, while each model's actual per-period error is not
-- a model that's usually excellent but occasionally very wrong (or
vice versa) can drag down an otherwise-strong ensemble on the specific
periods where it's wrong, in a way a single fixed weight can't correct
for. This is why evaluate()'s EnsembleResult.mape is computed and
surfaced directly (see build_ensemble()'s `actuals` param) rather than
assumed -- always compare it against each individual model's own MAPE
before trusting the ensemble output over the best single model.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from horizon_forecaster.evaluation.metrics import MapeResult, mape
from horizon_forecaster.exceptions import ThresholdRejectionError

WEIGHTING_STRATEGIES = ("inverse_mape", "equal")

# A literal 0% MAPE would divide-by-zero under inverse-MAPE weighting.
# Real-world MAPE is never exactly 0; this floor only guards against
# that edge case (e.g. a synthetic/test dataset) without materially
# changing behavior for any realistic MAPE value.
_INVERSE_MAPE_FLOOR = 1e-9

@dataclass
class EnsembleResult:
    """
    Structured result of combining multiple models' predictions into
    one weighted ensemble.

    Attributes:
        combined: DataFrame with columns ["ds", "yhat"] -- the weighted
            ensemble prediction, a genuinely new value (not any single
            model's own yhat). None if no model survived the
            mape_threshold, i.e. no ensemble could be formed.
        weights: normalized weight (summing to 1.0) per surviving
            model, e.g. {"prophet": 0.55, "catboost": 0.45}. Empty if
            combined is None.
        rejected_models: ThresholdRejectionError per model whose MAPE
            exceeded mape_threshold (or was NaN -- see
            build_ensemble()). Distinct from forecaster.py's
            evaluate_errors/forecast_errors: these models fit and
            predicted successfully, they just weren't accurate enough
            to trust in the ensemble.
        weighting: which strategy was used ("inverse_mape" or "equal").
        mape: the ensemble's OWN MapeResult, scored against real
            actuals. Only set when build_ensemble() was given
            `actuals` (the evaluate() context, where held-out actuals
            exist) -- always None in the forecast() context, where no
            ground truth for the future exists yet.
    """

    combined: Optional[pd.DataFrame]
    weights: Dict[str, float] = field(default_factory=dict)
    rejected_models: Dict[str, ThresholdRejectionError] = field(default_factory=dict)
    weighting: str = "inverse_mape"
    mape: Optional[MapeResult] = None

def _compute_weights(surviving_mapes: Dict[str, float], weighting: str) -> Dict[str, float]:
    if weighting == "equal":
        n = len(surviving_mapes)
        return {name: 1.0 / n for name in surviving_mapes}

    if weighting == "inverse_mape":
        inverse = {name: 1.0 / max(m, _INVERSE_MAPE_FLOOR) for name, m in surviving_mapes.items()}
        total = sum(inverse.values())
        return {name: v / total for name, v in inverse.items()}

    raise ValueError(
        f"Unknown weighting strategy: '{weighting}'. Expected one of {WEIGHTING_STRATEGIES}."
    )

def _combine_predictions(
    per_model_predictions: Dict[str, pd.DataFrame], weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Weighted sum of "yhat" across every surviving model, aligned on
    "ds" (not positional order) -- a merge, not a zip, so a genuine
    date misalignment between models raises instead of silently
    combining mismatched rows.
    """
    combined_df: Optional[pd.DataFrame] = None
    for model_name, weight in weights.items():
        model_df = per_model_predictions[model_name][["ds", "yhat"]].reset_index(drop=True)
        weighted_df = model_df.assign(yhat=model_df["yhat"] * weight)

        if combined_df is None:
            combined_df = weighted_df
            continue

        merged = combined_df.merge(weighted_df, on="ds", how="inner", suffixes=("", "_other"))
        if len(merged) != len(combined_df):
            raise ValueError(
                f"Cannot build ensemble: '{model_name}' predictions don't share the "
                f"same 'ds' values as the other surviving model(s) ({len(merged)} "
                f"aligned rows vs. {len(combined_df)} expected)."
            )
        merged["yhat"] = merged["yhat"] + merged["yhat_other"]
        combined_df = merged[["ds", "yhat"]]

    return combined_df

def build_ensemble(
    per_model_mape: Dict[str, float],
    per_model_predictions: Dict[str, pd.DataFrame],
    mape_threshold: float,
    weighting: str = "inverse_mape",
    actuals: Optional[pd.DataFrame] = None,
) -> EnsembleResult:
    """
    Combine multiple models' predictions into one weighted ensemble.

    Args:
        per_model_mape: {"prophet": 18.68, "catboost": 18.86, ...} --
            each model's MAPE.value. In forecaster.py, this is ALWAYS
            sourced from evaluate()'s held-out MAPE, even when building
            the forecast() ensemble -- forecast() has no actuals of its
            own to score against, so it reuses evaluate()'s weights.
        per_model_predictions: {"prophet": df, ...}, each df containing
            at least ["ds", "yhat"], for the SAME set of ds values
            across every model (the same test split, or the same
            future horizon dates).
        mape_threshold: accuracy cutoff (validated upstream by
            utils.validate_mape_threshold). A model whose MAPE exceeds
            this threshold is excluded and reported in rejected_models.
            This is an upper bound only -- an unusually low MAPE is NOT
            rejected; every model at or below the threshold is kept.
        weighting: "inverse_mape" (lower-MAPE models get more weight)
            or "equal" (every surviving model weighted equally).
        actuals: optional DataFrame with ["ds", "y"] real actual
            values, used to score the combined ensemble's own MAPE
            (the evaluate() context). None in the forecast() context.

    Returns:
        EnsembleResult. Never raises for "no model survived the
        threshold" -- see module docstring for why. Only genuine
        programming/data-shape errors (unknown weighting strategy,
        misaligned "ds" values between models) raise ValueError.
    """
    surviving_mapes: Dict[str, float] = {}
    rejected: Dict[str, ThresholdRejectionError] = {}

    for model_name, mape_value in per_model_mape.items():
        is_nan = mape_value != mape_value  # NaN check; same idiom already used in forecaster.py
        if is_nan:
            rejected[model_name] = ThresholdRejectionError(
                f"{model_name}'s MAPE is NaN (likely every test-set actual was zero), "
                f"so it can't be compared against the mape_threshold of {mape_threshold}%.",
                model_name=model_name,
                mape_value=mape_value,
                allowed_max=mape_threshold,
            )
            continue

        if mape_value > mape_threshold:
            rejected[model_name] = ThresholdRejectionError(
                f"{model_name}'s MAPE ({mape_value:.2f}%) exceeds the mape_threshold "
                f"of {mape_threshold}%.",
                model_name=model_name,
                mape_value=mape_value,
                allowed_max=mape_threshold,
            )
            continue

        surviving_mapes[model_name] = mape_value

    if not surviving_mapes:
        return EnsembleResult(
            combined=None,
            weights={},
            rejected_models=rejected,
            weighting=weighting,
            mape=None,
        )

    weights = _compute_weights(surviving_mapes, weighting)
    combined_df = _combine_predictions(per_model_predictions, weights)

    ensemble_mape: Optional[MapeResult] = None
    if actuals is not None:
        merged_actuals = combined_df.merge(actuals[["ds", "y"]], on="ds", how="inner")
        if len(merged_actuals) != len(combined_df):
            raise ValueError(
                f"Cannot score ensemble MAPE: 'actuals' don't share the same 'ds' "
                f"values as the combined ensemble ({len(merged_actuals)} aligned rows "
                f"vs. {len(combined_df)} expected)."
            )
        ensemble_mape = mape(merged_actuals["y"].values, merged_actuals["yhat"].values)

    return EnsembleResult(
        combined=combined_df,
        weights=weights,
        rejected_models=rejected,
        weighting=weighting,
        mape=ensemble_mape,
    )
