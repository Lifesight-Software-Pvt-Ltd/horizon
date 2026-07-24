# Horizon Forecaster — Build Log: Phase 7 (Future-exogenous-data handling)

Phase 7 replaces the single hardcoded "replay recent history" rule for
inventing future values of the independent columns with a small registry of
selectable strategies, plus type-aware handling of event flags and partial
fill. It was unblocked by Phase 6: the logic has to behave consistently for
every model, which is only meaningful once all five adapters exist.

## The architectural point that shaped the whole phase

The roadmap framed this as logic that "plugs into every adapter's
future-dataframe construction." In practice the Phase 1–2 contract already
had the right shape for it: future exog is generated **once, model-agnostically**
in the orchestrator, and the *same* values feed every model's
`transform_future()`. So Phase 7 is an extension of one existing orchestrator
helper into a standalone module — **no per-adapter changes**, and the gate
rule held again (nothing forced a change to a model's contract).

## What was built

- `horizon_forecaster/future_exog.py` — a `FUTURE_EXOG_STRATEGIES` registry
  of pure functions with no dependency on forecaster/model/adapter classes
  (mirroring `evaluation/ensemble.py`), plus a single public entry point
  `resolve_future_exog()`.
- `forecaster.py` wiring: four constructor params (`future_exog_strategy`,
  `event_columns`, `future_exog_aggregator`, `recent_average_window`),
  validated like `ensemble_weighting`; a `_resolve_future_exog()` that runs
  once per `forecast()` and feeds every model; `future_exog_used` exposed on
  the forecaster and on `ForecastResult`. The old
  `_build_default_future_independent_data()` is gone.

Strategies: `carry_forward` (default), `recent_average` (mean/median of the
last N), `historical_average` (season-aware), `seasonal_naive` (seasonal
cycle **plus drift**), `linear_trend`, and `cycle` (the retained Phase 2
behaviour).

## Key design decisions

- **Default changed from `cycle` to `carry_forward`.** The old behaviour
  replayed the last `horizon` periods of history, which injects the recent
  pattern — including any one-off spike — into the future. `carry_forward` is
  the least-surprising neutral baseline: hold the last level flat. `cycle` is
  kept for anyone who wants the old behaviour explicitly. This is a behaviour
  change on `forecast()` output when drivers aren't supplied; two tests that
  asserted the old cycling were re-pointed at `future_exog_strategy="cycle"`.

- **Type-aware flag handling.** Carrying a binary promo flag forward would
  assume a *permanent* promo; averaging it yields a nonsensical fractional
  flag. So continuous drivers use the chosen strategy, while binary 0/1 event
  flags default to **0** ("no event unless you supply one") — except under
  `seasonal_naive`, where flags follow the cycle so genuinely recurring events
  (holidays) still repeat. Flags come from an explicit `event_columns` list if
  given, else are auto-detected (≤2 distinct, non-negative values). The
  `cycle` legacy strategy is deliberately exempt from all of this and tiles
  every column as-is, so it reproduces the old output exactly.

- **`seasonal_naive` carries drift.** A pure seasonal-naive repeats last
  year's level, which is wrong for a growing driver over a long horizon. The
  implementation fits a trend, adds the phase-wise seasonal residual on top,
  and projects both forward — so a flat driver degrades to pure seasonal
  naive, and a trending one keeps trending. Under one full cycle of history it
  falls back to `carry_forward`.

- **Robustness baked in.** History is coerced to numeric and every
  aggregation is NaN-aware; any continuous output for a **non-negative** driver
  is clipped at 0 (so `linear_trend`/`seasonal_naive` can't project a spend
  column negative), while a driver that genuinely goes negative in history is
  left unclipped. Out-of-range *clamping* is still left to the model adapters,
  which already do it — the strategy module stays separation-clean.

- **Partial fill.** `forecast()` no longer treats future data as
  all-or-nothing: a column present in the supplied frame is used as-is, and
  only the missing columns are generated. This matches the real case — "I know
  next quarter's planned spend, but not the promo calendar." The
  auto-generation warning now names only the generated columns and the strategy
  used.

- **Transparency.** The resolved future exog (supplied + generated, with a
  `ds` column) is exposed as `forecaster.future_exog_used` and on each
  `ForecastResult`, so the assumptions behind a scenario can be inspected or
  plotted rather than trusted blind.

## How this was verified

- `tests/test_future_exog.py` — 23 pure unit tests: each strategy's numeric
  behaviour, season-awareness, drift, the under-one-cycle fallback,
  non-negativity clipping (and its non-clipping of genuinely-negative drivers),
  flag defaulting and the `seasonal_naive` flag exception, explicit vs
  auto-detected flags, partial fill, NaN history, horizon-longer-than-history,
  a weekly case, and the validation errors.
- `tests/test_forecaster.py` — forecaster-level wiring without a model fit:
  the `cycle` strategy reproduces the old replay numbers, `carry_forward` is
  the default and holds flat, partial fill generates only the missing column,
  explicit `event_columns` force a flag to 0, `future_exog_used` is exposed
  with a `ds` column, and the four new params reject bad values.
- All of the above run in the sandbox, alongside the existing pure suites and
  SARIMAX (229 passed). The prophet/catboost/neuralprophet/pymc-backed
  `forecast()` tests remain the local/CI gate, as before.

## Deliberately deferred

User-facing docs (`quickstart.md`, `supported_models.md`) are not updated
here — per the roadmap, the interface locks first and Phase 8 documents it
once. Only this build log and the CHANGELOG are updated live.
