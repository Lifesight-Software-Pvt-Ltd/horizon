"""
Ensemble combination logic for horizon_forecaster.

Combines multiple models' per-period "yhat" predictions into a single
series. Deliberately a pure, standalone module -- like
evaluation/metrics.py, it has no dependency on forecaster.py,
HorizonForecaster, or any model/adapter class. forecaster.py
(HorizonForecaster.evaluate()/forecast()) is the only caller.

Three combination rules are available, and the DEFAULT IS `trimmed`:

  trimmed       Per period, sort the members, discard the highest and the
                lowest, and average what remains. Over five members that
                is the mean of the middle three; over four it is the
                median -- the rule reports which of those it actually did
                via `rule_applied`, rather than calling everything a
                "trimmed mean" and leaving the caller to work it out.

  inverse_mape  Weight each member by 1/MAPE, normalised.
  equal         Weight every member identically.

Why trimming is the default, when the two weighted rules came first:
estimated weights have to be estimated FROM something, and everything
available to estimate them from is a finite sample of past error. A
trimmed mean consumes no such quantity at all, so it cannot leak
information from a scoring window by construction, and it has no weights
to get wrong. On an out-of-fold evaluation over 99 daily commercial
series at 3 rolling origins, inverse-error weighting was also the worse
performer on tail behaviour specifically (maximum error ratio 3.74
vs 2.63, maximum regret 2.98 vs 1.59 against a trimmed mean) -- the
weighted rules are kept because they remain reasonable choices and some
callers will want them, not because anything recommends them as a
default.

TWO DIFFERENT MAPES, AND WHY THEY ARE SEPARATE ARGUMENTS. build_ensemble()
takes `per_model_selection_mape` (used for screening and for
inverse-MAPE weights) and, optionally, `actuals` (used to score the
combined result). The first must come from a window that was allowed to
influence decisions -- Horizon's validation window. The second is
measured on the reporting window. Collapsing them into one argument is
what allowed the pre-1.0 library to weight an ensemble using the very
numbers it then reported.

Design constraint carried over from models/base.py's BaseModel.predict()
contract: this module operates on the "yhat" column ONLY. Uncertainty
bounds (yhat_lower/yhat_upper) are never combined -- averaging
confidence/credible intervals across statistically different model
types (Prophet's trend simulation vs. CatBoost's point estimate vs.
PyMC's posterior) has no coherent interpretation. Per-model bounds
remain available on each model's own EvaluationResult/ForecastResult for
individual-model plots/CSVs.

KNOWN LIMITATION (confirmed via direct testing, not theoretical): a
combined ensemble is NOT guaranteed to beat the single best individual
model. Verified directly on a synthetic dataset: Prophet alone scored
1.31% MAPE, CatBoost alone scored 12.77%, and the inverse-MAPE-weighted
ensemble of the two scored 1.58% -- worse than Prophet alone, despite
Prophet receiving ~91% of the weight. Weights are static across the
whole horizon while each model's actual per-period error is not, so a
model that's usually excellent but occasionally very wrong can drag down
an otherwise-strong ensemble in a way a single fixed weight can't
correct for. Trimming is considerably more robust to this specific
failure (it discards the offending member on exactly the periods where
it offends), but is still not a guarantee. This is why the ensemble's
own MAPE is computed and surfaced directly (see build_ensemble()'s
`actuals` param) rather than assumed -- always compare it against each
individual model's own MAPE before trusting the ensemble output over the
best single model.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from horizon_forecaster.evaluation.metrics import MapeResult, mape
from horizon_forecaster.exceptions import InvalidConfigError, ThresholdRejectionError

# Every combination rule this module can apply. Order matters only for
# presentation (it is the order build_ensembles() returns them in when
# asked for all of them); `trimmed` is first because it is the default.
ENSEMBLE_METHODS = ("trimmed", "inverse_mape", "equal")
DEFAULT_ENSEMBLE_METHOD = "trimmed"

# The subset of ENSEMBLE_METHODS that produce weights. Load-bearing, not
# a naming leftover: _screen() uses it to decide that a NaN selection MAPE
# disqualifies a member from the WEIGHTED rules only, since trimming needs
# no MAPE at all. Removing it would silently change which members survive.
WEIGHTING_STRATEGIES = ("inverse_mape", "equal")

# Below this many members, trimming is not defined -- discarding the
# highest and the lowest of two values leaves nothing to average. This is
# an arithmetic boundary rather than a tuned constant, which is why the
# degradation ladder in _combine() hinges on it.
_MIN_MEMBERS_FOR_TRIMMING = 3

# A literal 0% MAPE would divide-by-zero under inverse-MAPE weighting.
# Real-world MAPE is never exactly 0; this floor only guards against
# that edge case (e.g. a synthetic/test dataset) without materially
# changing behavior for any realistic MAPE value.
_INVERSE_MAPE_FLOOR = 1e-9


@dataclass
class EnsembleResult:
    """
    Structured result of combining multiple models' predictions.

    Attributes:
        combined: DataFrame with columns ["ds", "yhat"] -- the combined
            prediction, a genuinely new series (not any single model's own
            yhat). None when no member was available to combine.
        method: the combination rule that was REQUESTED ("trimmed",
            "inverse_mape" or "equal").
        rule_applied: the rule that actually ran. Usually equal to
            `method`, but the trimmed rule degrades gracefully when too
            few members survive -- see `degraded`. Values: "trimmed",
            "median", "mean", "single" or "none" for the trimmed rule,
            and "inverse_mape" or "equal" echoed back for the weighted
            rules, which never degrade. Branch on this rather than on
            `method` when what matters is the statistic produced.
        members: the models that fed the combination, in the order they
            were combined.
        degraded: True when `rule_applied` is a fallback rather than the
            rule asked for, i.e. when too few members survived for
            trimming to be defined. Surfaced as a field rather than a
            warning so callers can branch on it -- a two-member plain mean
            is a materially weaker output than a five-member trimmed mean
            and should not look identical in a result object.
        weights: normalized weight (summing to 1.0) per member, e.g.
            {"prophet": 0.55, "catboost": 0.45}. **None for weightless
            rules** such as `trimmed`, which is the honest representation
            -- there is no weight, and reporting an equal split would
            imply an averaging that did not happen.
        trim_shares: for trimming only, a per-member summary of how often
            each member was discarded as the high or the low value. More
            diagnostically useful than weights: it identifies which model
            misbehaves, and on how much of the horizon. None for weighted
            rules.
        rejected_models: ThresholdRejectionError per model excluded before
            combination. Distinct from forecaster.py's evaluate_errors/
            forecast_errors: these models fit and predicted successfully,
            they just weren't admitted to the combination.
        mape: the ensemble's OWN MapeResult, scored against real actuals
            on the reporting window. Only set when build_ensemble() was
            given `actuals` (the evaluate() context, where held-out
            actuals exist) -- always None in the forecast() context, where
            no ground truth for the future exists yet.
    """

    combined: Optional[pd.DataFrame]
    method: str = DEFAULT_ENSEMBLE_METHOD
    rule_applied: str = "none"
    members: List[str] = field(default_factory=list)
    degraded: bool = False
    weights: Optional[Dict[str, float]] = None
    trim_shares: Optional[pd.DataFrame] = None
    rejected_models: Dict[str, ThresholdRejectionError] = field(default_factory=dict)
    mape: Optional[MapeResult] = None


def as_ensemble_dict(
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]",
) -> Dict[str, "EnsembleResult"]:
    """
    Normalise an ensemble argument into a {method: EnsembleResult} dict.

    HorizonForecaster.evaluate_ensemble/forecast_ensemble are keyed by
    combination method, but a single EnsembleResult is still accepted so a
    caller holding one result doesn't have to wrap it, and None maps to an
    empty dict so consumers need no special case.

    Lives here, next to EnsembleResult, rather than in either consumer.
    It previously existed as identical copies in plotting.py and
    exporters.py, and the two drifted: exporters normalised its input
    while half of plotting's entry points did not, so passing the dict a
    forecaster actually produces raised AttributeError from
    export_forecast(save_plots=True). Note it cannot live in plotting.py
    either -- that module imports matplotlib at module scope, and
    exporters.py deliberately keeps matplotlib off the CSV-only path.
    """
    if ensemble is None:
        return {}
    if isinstance(ensemble, EnsembleResult):
        return {ensemble.method: ensemble}
    if isinstance(ensemble, dict):
        wrong = {
            k: type(v).__name__ for k, v in ensemble.items() if not isinstance(v, EnsembleResult)
        }
        if not wrong:
            return ensemble
        detail = f"its values {wrong} are not EnsembleResults"
    else:
        detail = f"got {type(ensemble).__name__}"
    raise InvalidConfigError(
        f"ensemble must be forecaster.evaluate_ensemble or forecaster.forecast_ensemble (a "
        f"dict of method name to EnsembleResult), a single EnsembleResult, or None; {detail}.",
        param_name="ensemble",
        provided_value=detail,
    )


def _align_members(
    per_model_predictions: Dict[str, pd.DataFrame], members: Sequence[str]
) -> "tuple[pd.Series, np.ndarray]":
    """
    Line every member's "yhat" up on "ds" and return (ds, matrix), where
    matrix has shape (n_members, n_periods) in `members` order.

    Alignment is a MERGE, not a positional zip, so a genuine date
    misalignment between models raises instead of silently combining
    mismatched rows.

    This is deliberately separate from the rules that consume it. In the
    pre-1.0 module, alignment was fused into the weighted-sum loop --
    which meant the loop was driven by the weights dict, and a rule with
    no weights (like trimming) had no way to align anything at all.
    """
    aligned: Optional[pd.DataFrame] = None
    expected_rows: Optional[int] = None

    for model_name in members:
        model_df = (
            per_model_predictions[model_name][["ds", "yhat"]]
            .reset_index(drop=True)
            .rename(columns={"yhat": model_name})
        )
        if aligned is None:
            aligned = model_df
            expected_rows = len(model_df)
            continue

        aligned = aligned.merge(model_df, on="ds", how="inner")
        if len(aligned) != expected_rows:
            raise InvalidConfigError(
                f"Cannot build ensemble: '{model_name}' predictions don't share the "
                f"same 'ds' values as the other member(s) ({len(aligned)} aligned "
                f"rows vs. {expected_rows} expected)."
            )

    matrix = aligned[list(members)].to_numpy(dtype=float).T
    return aligned["ds"].reset_index(drop=True), matrix


def _compute_weights(member_mapes: Dict[str, float], weighting: str) -> Dict[str, float]:
    if weighting == "equal":
        n = len(member_mapes)
        return {name: 1.0 / n for name in member_mapes}

    if weighting == "inverse_mape":
        inverse = {name: 1.0 / max(m, _INVERSE_MAPE_FLOOR) for name, m in member_mapes.items()}
        total = sum(inverse.values())
        return {name: v / total for name, v in inverse.items()}

    raise InvalidConfigError(
        f"Unknown weighting strategy: '{weighting}'. Expected one of {WEIGHTING_STRATEGIES}."
    )


def _build_trim_shares(members: Sequence[str], order: np.ndarray, n_periods: int) -> pd.DataFrame:
    """
    Summarise how often each member was discarded as the period's high or
    low value. `order` is argsort output of shape (n_members, n_periods).
    """
    low_counts = np.bincount(order[0], minlength=len(members))
    high_counts = np.bincount(order[-1], minlength=len(members))

    return pd.DataFrame(
        {
            "model": list(members),
            "trimmed_low": low_counts,
            "trimmed_high": high_counts,
            "trimmed_total": low_counts + high_counts,
            "n_periods": n_periods,
            "trimmed_fraction": (low_counts + high_counts) / n_periods if n_periods else 0.0,
        }
    )


def _combine(
    method: str,
    members: List[str],
    per_model_predictions: Dict[str, pd.DataFrame],
    member_mapes: Dict[str, float],
):
    """
    Apply one combination rule to the surviving members.

    Returns (combined_df, rule_applied, degraded, weights, trim_shares).

    The degradation ladder for `trimmed` is:

        3+ members  trim the high and the low, average the rest
        2 members   plain mean          (degraded -- trimming undefined)
        1 member    that member's own forecast (degraded)
        0 members   no combination at all

    Three is where the arithmetic stops working, not a tuned threshold:
    remove the highest and lowest of two values and nothing is left. The
    weighted rules have no such boundary -- a weighted average of two, or
    of one, is perfectly well defined -- so they never report `degraded`.
    """
    ds, matrix = _align_members(per_model_predictions, members)
    n_members = len(members)

    if method == "trimmed":
        if n_members >= _MIN_MEMBERS_FOR_TRIMMING:
            # A STABLE sort matters beyond tidiness: _build_trim_shares
            # attributes each period's discard to order[0]/order[-1], so
            # when two members predict the same value, which one gets
            # blamed for being the high or the low is decided entirely by
            # the sort being stable and by `members` order. An unstable
            # sort would make the diagnostic non-reproducible run to run.
            order = np.argsort(matrix, axis=0, kind="stable")
            ordered = np.take_along_axis(matrix, order, axis=0)
            values = ordered[1:-1].mean(axis=0)
            # Over exactly four members, trimming the high and the low
            # leaves two values whose mean IS the median. Naming it
            # accurately matters: a reader comparing runs across series
            # should not have to recompute the member count to know which
            # statistic they are looking at.
            rule_applied = "median" if n_members == 4 else "trimmed"
            trim_shares = _build_trim_shares(members, order, len(ds))
            return (
                pd.DataFrame({"ds": ds, "yhat": values}),
                rule_applied,
                False,
                None,
                trim_shares,
            )

        if n_members == 2:
            return (
                pd.DataFrame({"ds": ds, "yhat": matrix.mean(axis=0)}),
                "mean",
                True,
                None,
                None,
            )

        return (
            pd.DataFrame({"ds": ds, "yhat": matrix[0]}),
            "single",
            True,
            None,
            None,
        )

    weights = _compute_weights({name: member_mapes[name] for name in members}, method)
    weight_vector = np.array([weights[name] for name in members])[:, None]
    return (
        pd.DataFrame({"ds": ds, "yhat": (matrix * weight_vector).sum(axis=0)}),
        method,
        False,
        weights,
        None,
    )


def _screen(
    per_model_selection_mape: Dict[str, float],
    mape_threshold: Optional[float],
    method: str,
) -> "tuple[List[str], Dict[str, float], Dict[str, ThresholdRejectionError]]":
    """
    Decide which models are admitted to the combination.

    Two independent things can exclude a model here, and they are not the
    same test:

      1. `mape_threshold`, when set. This is a GARBAGE FILTER, not an
         accuracy screen -- a model at 300% MAPE should not get a vote
         even though trimming would mostly absorb it. Set it to None to
         admit everything. Note that screening and trimming both spend
         robustness: screen aggressively AND trim, and five members can
         become three, which is the median. That is a legitimate choice,
         but it should be a deliberate one.

      2. A NaN selection MAPE, but ONLY for the weighted rules. For
         'inverse_mape' that is arithmetic -- there is no 1 / NaN.
         'equal' could weight such a member (it only reads the member
         COUNT), and it is excluded on a different ground: a model with
         no validation MAPE was never screened against mape_threshold
         either, and the weighted rules hand every member a full,
         untrimmed share of the result, so admitting an unvetted model
         there lets it move the number with no check on it at all.

         Trimming keeps it, because trimming itself bounds how far any
         one member can move the result -- it discards the period's high
         and low regardless of who they came from. So a model whose MAPE
         is undefined (typically because every actual in the window was
         zero -- sparse conversions, installs) still contributes to a
         trimmed combination. It genuinely learned the series; only the
         metric broke.
    """
    members: List[str] = []
    member_mapes: Dict[str, float] = {}
    rejected: Dict[str, ThresholdRejectionError] = {}

    for model_name, mape_value in per_model_selection_mape.items():
        is_nan = mape_value != mape_value  # NaN check

        if is_nan and method in WEIGHTING_STRATEGIES:
            rejected[model_name] = ThresholdRejectionError(
                f"{model_name}'s validation MAPE is NaN (likely every actual in the "
                f"validation window was zero), so it was never screened and the "
                f"'{method}' rule cannot admit it to a full, untrimmed share of the "
                f"result. It would still be usable by the 'trimmed' rule, which needs "
                f"no MAPE and bounds any single member's influence itself.",
                model_name=model_name,
                mape_value=mape_value,
                # Deliberately not passing allowed_max: this rejection is not a
                # threshold breach, so echoing the threshold back would invite a
                # reader to conclude the model scored above it.
            )
            continue

        if mape_threshold is not None and not is_nan and mape_value > mape_threshold:
            rejected[model_name] = ThresholdRejectionError(
                f"{model_name}'s validation MAPE ({mape_value:.2f}%) exceeds the "
                f"mape_threshold of {mape_threshold}%.",
                model_name=model_name,
                mape_value=mape_value,
                allowed_max=mape_threshold,
            )
            continue

        members.append(model_name)
        member_mapes[model_name] = mape_value

    return members, member_mapes, rejected


def build_ensemble(
    per_model_selection_mape: Dict[str, float],
    per_model_predictions: Dict[str, pd.DataFrame],
    mape_threshold: Optional[float] = None,
    method: str = DEFAULT_ENSEMBLE_METHOD,
    actuals: Optional[pd.DataFrame] = None,
) -> EnsembleResult:
    """
    Combine multiple models' predictions using one rule.

    Args:
        per_model_selection_mape: {"prophet": 18.68, ...} -- each model's
            MAPE **on the window used for decisions** (Horizon's
            validation window). Used for threshold screening and, for the
            weighted rules, for the weights themselves. Never the score
            being reported: see the module docstring.
        per_model_predictions: {"prophet": df, ...}, each df containing at
            least ["ds", "yhat"], for the SAME set of ds values across
            every model (the same test window, or the same future
            horizon).
        mape_threshold: degenerate-fit filter applied to the selection MAPE, or
            None to admit every model. See _screen().
        method: "trimmed" (default), "inverse_mape" or "equal".
        actuals: optional DataFrame with ["ds", "y"] real actual values,
            used to score the combined result's own MAPE (the evaluate()
            context). None in the forecast() context.

    Returns:
        EnsembleResult. Never raises for "no model was admitted" -- it
        returns combined=None, since the per-model results that produced
        those MAPEs are still perfectly valid and should not be discarded
        just because no combination could be formed. Only genuine
        programming/data-shape errors (unknown method, misaligned "ds"
        values between models) raise.
    """
    if method not in ENSEMBLE_METHODS:
        raise InvalidConfigError(
            f"Unknown ensemble method: '{method}'. Expected one of {ENSEMBLE_METHODS}."
        )

    members, member_mapes, rejected = _screen(per_model_selection_mape, mape_threshold, method)

    if not members:
        return EnsembleResult(
            combined=None,
            method=method,
            rule_applied="none",
            members=[],
            degraded=False,
            weights=None,
            trim_shares=None,
            rejected_models=rejected,
            mape=None,
        )

    combined_df, rule_applied, degraded, weights, trim_shares = _combine(
        method, members, per_model_predictions, member_mapes
    )

    ensemble_mape: Optional[MapeResult] = None
    if actuals is not None:
        merged_actuals = combined_df.merge(actuals[["ds", "y"]], on="ds", how="inner")
        if len(merged_actuals) != len(combined_df):
            raise InvalidConfigError(
                f"Cannot score ensemble MAPE: 'actuals' don't share the same 'ds' "
                f"values as the combined ensemble ({len(merged_actuals)} aligned rows "
                f"vs. {len(combined_df)} expected)."
            )
        ensemble_mape = mape(merged_actuals["y"].values, merged_actuals["yhat"].values)

    return EnsembleResult(
        combined=combined_df,
        method=method,
        rule_applied=rule_applied,
        members=members,
        degraded=degraded,
        weights=weights,
        trim_shares=trim_shares,
        rejected_models=rejected,
        mape=ensemble_mape,
    )


def build_ensembles(
    methods: Sequence[str],
    per_model_selection_mape: Dict[str, float],
    per_model_predictions: Dict[str, pd.DataFrame],
    mape_threshold: Optional[float] = None,
    actuals: Optional[pd.DataFrame] = None,
) -> Dict[str, EnsembleResult]:
    """
    Build one EnsembleResult per requested rule, keyed by rule name.

    Always a dict, even for a single method -- matching the convention
    HorizonForecaster.evaluate()/forecast() already follow for per-model
    results, so the shape of what comes back never depends on how many
    things were asked for.

    Combination itself is arithmetic on predictions that already exist;
    the expensive part (fitting the models) has long since happened. So
    asking for all three rules costs essentially nothing beyond the first,
    which is why `ensemble_method=None` offering all of them is a
    reasonable thing for a caller to want.
    """
    return {
        method: build_ensemble(
            per_model_selection_mape=per_model_selection_mape,
            per_model_predictions=per_model_predictions,
            mape_threshold=mape_threshold,
            method=method,
            actuals=actuals,
        )
        for method in methods
    }
