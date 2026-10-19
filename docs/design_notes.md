# Design notes

Why Horizon Forecaster is built the way it is. This is the reference for
anyone changing the library rather than using it: the contracts that hold the
five models together, the reasoning behind the choices that are easy to
mistake for arbitrary, and the failure modes that shaped them.

For usage, start with [`quickstart.md`](quickstart.md). For the release
record, see [`CHANGELOG.md`](../CHANGELOG.md).

---

## 1. The shape of the library

Everything follows from one decision: a model is an **(adapter, wrapper)
pair**, and the orchestrator knows nothing else about it.

- A **data adapter** (`data_adapter/`, subclasses `BaseAdapter`) does all
  model-specific reshaping. `transform_train()` turns raw user data into that
  model's training input; `transform_future()` turns future dates plus
  resolved future driver values into its prediction input.
- A **model wrapper** (`models/`, subclasses `BaseModel`) does
  `fit()` / `predict()`, plus `save()` / `load()` where the backend needs
  something better than pickle.
- `HorizonForecaster` (`forecaster.py`) wires
  `adapter.transform_train() → wrapper.fit()` and
  `adapter.transform_future() → wrapper.predict()` identically for every
  entry in `MODEL_REGISTRY`.

A wrapper never calls an adapter, and an adapter never sees a model. Each
side is testable in isolation — the adapter against plain DataFrames, the
wrapper against pre-shaped input.

**The gate rule.** Adding a model should mean adding one `MODEL_REGISTRY`
entry and nothing else. If a new model forces a change to the orchestrator's
core logic, that is evidence of a gap in the `BaseAdapter`/`BaseModel`
contract — fix the contract, do not special-case the orchestrator. All five
current models were absorbed under this rule without an orchestrator change,
including the two that came closest to breaking it: NeuralProphet's
autoregression and PyMC's whole-series posterior (see §5).

**Pure modules stay pure.** `evaluation/metrics.py`,
`evaluation/ensemble.py`, `future_exog.py`, `plotting.py` and `exporters.py`
import no model, adapter, or `forecaster.py`. Result objects are duck-typed
and type hints are `TYPE_CHECKING`-only. Verified rather than assumed:
importing `plotting` and `exporters` pulls in neither Prophet, CatBoost, nor
the forecaster. Drawing a chart or writing a CSV never pays the heavy-import
cost.

**Errors are structured.** Every raised error is a typed subclass of
`HorizonForecasterError` (`exceptions.py`) carrying machine-readable
attributes (`model_name`, `mape_value`, `required_periods`, …), not just a
message. That is what lets `evaluate_errors`, `forecast_errors` and
`EnsembleResult.rejected_models` be built programmatically instead of by
parsing exception text. One base class also lets the orchestrator catch
library failures with a single `except` while genuine bugs still surface as
raw Python exceptions.

**Failures are isolated, per model.** Any exception from one model — typed or
not — is recorded against that model and the others continue. A non-typed
failure is wrapped as `ModelConvergenceError` with the original preserved on
`original_exception`. This exists because a raw `numpy.linalg.LinAlgError`
from PyMC's prior estimation once took four healthy models down with it; the
isolation is the general guarantee, not a patch for that one case.

---

## 2. Evaluation protocol

### The date column is checked, not trusted

Every later step — sorting, the window split, the forecast dates, each
model's seasonal settings — assumes the date column is real, regular dates
at the declared frequency. When that was assumed rather than checked, bad
dates produced confident wrong answers instead of errors: text dates such as
`01/15/2023` were sorted alphabetically, so the windows mixed years and the
forecast started from the wrong date; monthly data declared daily finished
with a 0.67% MAPE and a daily-dated forecast; duplicate dates crashed
`evaluate()` late with an unrelated error. So the constructor now parses the
dates first, on a copy, and enforces an explicit contract (the table in
[`quickstart.md`](quickstart.md)): one of six text formats or a datetime,
each date once, weekly dates on one weekday, more than half the steps
exactly one period, and no more than 28 days missing in a row.

Three choices in it are worth recording:

- **Ambiguous dates raise rather than being guessed.** A column whose every
  value reads equally well day-first or month-first (`03/04/2024`) has no
  safe default, and real exports had been read month-first silently.
  `date_format` settles it.
- **The gap limit is absolute, not a share of the history.** A hole of
  three months removes the same stretch of trend and seasonality from four
  years of data as from two, and SARIMAX treats the dates either side of it
  as neighbours. Four weeks is the line; anything longer raises.
- **Shorter, scattered gaps only warn.** They are common (an occasional
  missing day, a holiday on an open day), the models still run, and Robyn —
  the MMM reference this library sits next to — does not reject them
  either. The warning says what they cost; Horizon never fills them.

### Closed weekdays are an operating calendar, not gaps

Daily data with no weekend rows used to pass as "daily with gaps", and the
damage showed up later. `forecast(10)` produced Saturday and Sunday dates at
weekday levels (4 of the 10 rows), and NeuralProphet's were nonsense (Saturday
1,770, Sunday 2,614 on a series around 2,000), since it had never seen a
weekend. SARIMAX counted 7 rows as a week, so its Friday peak
drifted onto other days and its test WAPE went from 5.03 to 9.56 (Mon–Fri) and
from 5.77 to 14.85 (Mon–Sat). The gap warning told people to "add the missing
periods", which is wrong for days the business is shut.

A weekday that never appears, in data spanning at least four weeks, is now a
closed day (`utils.find_closed_weekdays`). Periods become open days:
`describe_date_spacing` steps over closed days, forecast dates come from a
`CustomBusinessDay` on the open weekdays, SARIMAX's `seasonal_period` is the
number of open days, and the 4-week gap limit is 4 weeks of open days. The
models that work from dates (Prophet, NeuralProphet, CatBoost, Bayesian PyMC)
need no change: their test scores on Mon–Fri data were within a point of the
7-day ones, and once the forecast dates are open days only, nothing is asked
of them about a weekend. The seasonal-naive baseline already
matches by date.

Adding the closed days as 0 was measured as the alternative and lost: it
forces the models to learn an on/off pattern, Prophet and NeuralProphet then
forecast about 300 a day for closed days, and their weekday WAPE rose (4.74 to
6.23, and 4.43 to 5.48). The rule is strict — one Sunday in the data makes
Sunday an open day — because guessing "mostly closed" would drop real
observations; that case falls back to ordinary gaps. The accepted cost is that
holidays on open days are still gaps, which SARIMAX handles worst (9.2 against
5.0 WAPE with about 3% of weekdays missing); Horizon still does not fill them.

### Three windows, not two

Data splits chronologically into **train / validation / test**.

- **Validation** carries every decision: hyperparameter grid search,
  `mape_threshold` screening, `inverse_mape` weighting.
- **Test** is reported and read by no selection path.
- The winning candidate is refit on train + validation before the test window
  is scored.

The reported `mape` is therefore an out-of-sample number rather than the
score a model was chosen for; the selection score stays alongside it as
`validation_mape`.

The refit costs one extra fit per model and is deliberate. Fitting on train
alone and predicting across both windows is cheaper but measures a model
trained on less data than would ever be deployed. Refitting on everything
available makes the reported MAPE an estimate of the same procedure that
produces the real forecast. `EvaluationResult.fitted_model` is consequently
that refit — the model that actually produced `test_predictions`.

### Window sizes are absolute, not percentages

Defaults are **84 validation / 28 test** for daily, **39 / 13** for weekly,
with training taking the remainder. Percentages would make both windows scale
with however much history the caller happens to have, which is the wrong
driver for either: a test window estimates the error you will experience when
you forecast, so it should be about the length of a horizon somebody would
really act on, regardless of whether you hold two years of history or ten.

`validation_size`/`test_size` accept an `int` (periods), a `float` strictly
between 0 and 1 (a fraction), or `None` for the default. `training_size`
survives as a percentage-only alternative route (50–100, or 0.5–1.0) with the
remainder split three-quarters to validation, and is **mutually exclusive** with
the other two — passing both raises `InvalidConfigError` rather than letting
one silently win.

On a series too short for the defaults they scale down proportionally, capped
at 45% of rows between them, preserving the 3:1 ratio and warning with the
resulting sizes. Explicitly requested sizes are never silently shrunk.

**Validation is three times the test window, and why not a rolling
backtest.** Validation started at twice the test window. Testing showed that
the model it picked often did badly on test (validation-best ranked 4th or
5th of five in all four real runs), so a 3-fold rolling backtest was the
planned fix. Before building it, the two were compared on 19 real series (5
weekly, 14 daily, up to 4 test origins each, 4 models at fixed settings):
today's window, one window three test-lengths long, and three rolling folds
of one test-length each, all judged by how well they picked the model that
then did best on test.

| | single (2×) | long (3×) | 3 folds |
|---|---|---|---|
| Weekly: picked the test-best model | 29% | 29% | 36% |
| Weekly: rank correlation with test | 0.01 | 0.20 | 0.20 |
| Weekly: mean / median regret, MAPE points | 9.1 / 3.0 | 7.3 / 2.1 | 5.7 / 1.9 |
| Daily: picked the test-best model | 40% | 44% | 38% |
| Daily: mean / median regret, MAPE points | 1.9 / 0.13 | 1.2 / 0.14 | 1.2 / 0.18 |

On daily data none of them changed the choice much. On weekly data the folds
helped, but a single longer window captured most of it at no extra fitting
cost, while the folds triple the selection runtime — so the longer window
became the default and the rolling backtest stays a deliberate gap (§9).
Every score beat a random pick by a wide margin (median regret 0.2 against
2.9 points); what none of them fixes is the handful of series where no score
predicts the test window. Choosing on the most recent window instead (the
test window, then forecasting the next one) was also measured and was worse
on weekly data (picked the best model 11% of the time against 33%), so
selection stays on validation. The budget for defaulted windows rose from 40%
to 45% of rows at the same time, so a typical 122-week series keeps the full
39 + 13 rather than shrinking its test window below a quarter.

### A driver that never changes inside a fit window

A driver that varies over the full history survives the zero-variance check,
yet it can be one value throughout the window a model is fitted on — a price
change or a new promotion that first appears in the validation or test
window. No model can learn an effect from such a column, and they failed
in different ways when given one: NeuralProphet discarded it at fit and then
rejected it at prediction, so it dropped out of the run (reported on client
data with a `Price_Hike` flag); Prophet, when the constant was not 0, fitted
an arbitrary effect to it (+31,403 on a series around 3,500 whose true effect
was −400); CatBoost, SARIMAX and Bayesian PyMC gave it no effect. The
orchestrator now finds such drivers once per fit window and leaves them out
of that fit for every model, so all of them give the same, honest answer —
no effect yet — and stay comparable. A warning names the driver and the
score that includes the effect no model could see.

It leaves the split alone rather than moving it to include the change: a
split that depended on the drivers would mean something different from run
to run. A rolling backtest will meet this in every fold whose training ends
before the change, which is why the check is one function that runs per fit
window.

### Selection by MAPE, and its limits

Candidates are scored by validation MAPE rather than an information
criterion. AIC rewards in-sample fit, not out-of-sample accuracy, and
selecting SARIMAX orders by AIC was a direct cause of occasional very high
MAPE in the pre-library version of that model.

MAPE is undefined against zero actuals, so `mape()` leaves those rows out and
returns a `MapeResult` saying how many. On a series with many zeros that
means scoring a model on the days it finds easiest: on one real daily series
the models' test MAPE was 238–420% while their WAPE was 34–41%. MAPE also
explodes as actuals cross zero.

MAPE stays the default, because it is the metric the people reading these
results use. `selection_metric="auto"` switches every decision to WAPE
(`sum|y - yhat| / sum|y|`, which counts every row) only when the target is 0
in at least 10% of periods or has a negative value, with a warning; both
scores are always reported. The 10% line comes from 42 series (21 real retail
series and 21 synthetic): the two metrics chose the same models on 35, and
where they differed WAPE's choice did better on test in 6 of 7. The median
WAPE/MAPE ratio was 1.02, so `mape_threshold` keeps its meaning on either.

At 50% zeros or more the target is refused outright. Forecasting 0 every
period has a WAPE of exactly 100%, and on the two series tested above that
share (70% and 53% zeros) every model scored 103–129%. Daily data like that
is usually fine summed to weeks — those two series were 0 in 9% and 6% of
weeks — so the error gives that share and points to `frequency="weekly"`.
Horizon does not aggregate on its own: summing spend is right but summing a
price or an index is not, and only the caller knows which columns are which.
The accepted cost is that a sparse daily series with a strong weekday
pattern is refused even where a model could have forecast it.

Neither metric says whether a model beats a naive forecast, so `evaluate()`
scores one — the same weekday last week, or the same week last year — on the
same windows (`naive_baseline`) and warns about anything that does worse on
test WAPE. On the same 42 series the default trimmed ensemble lost to it in
13 of the 35 where it formed. The baseline is reported, not used to select
or weight; a benchmark-relative metric (RelMAE, MASE) that does that is still
open (§9).

### Uncertainty bounds are per-model only

`yhat_lower`/`yhat_upper` are optional in the `BaseModel` contract, provided
by Prophet, SARIMAX and Bayesian PyMC, and flow into per-model plots and
CSVs. They are **never combined into the ensemble**. Averaging intervals
produced by a trend simulation, a Gaussian innovation band and an MCMC
posterior has no coherent interpretation. This is enforced by a test, not
only documented.

---

## 3. The ensemble

`build_ensembles()` combines the point forecasts of every model whose
validation MAPE is at or below `mape_threshold`, under one or more rules,
returning one `EnsembleResult` per rule.

### Why a trimmed mean is the default

Estimated weights have to be estimated from something, and everything
available is a finite sample of past error. A trimmed mean — per period,
discard the highest and lowest member, average the rest — spends no
estimation budget at all and is robust exactly where static weights are
fragile.

Static weights fail in a specific, confirmed way. On a synthetic test Prophet
alone scored 1.31% MAPE, CatBoost 12.77%, and their ~91%/9% inverse-MAPE
blend scored **1.58%** — worse than Prophet alone despite Prophet carrying
almost all the weight. Weights are fixed across the horizon while each
model's per-period error is not, so a strong ensemble can still be dragged
off on precisely the periods where a weak member is wrong. Trimming drops
the offending member on the periods where it offends.

**A combined forecast is still not guaranteed to beat the best single
model.** That is a real property of combination, not a bug, which is why
`evaluate()` scores every ensemble against actuals and the exporter writes
both a `final_ensemble_<method>` and a `final_best_model` CSV. The choice is
made per run, with the evidence in hand. This is not hypothetical: on one
real run the ensemble scored 22.40% against CatBoost's 21.32%. On another —
weekly, three comparably-accurate models with partly independent errors — the
ensemble at 34.41% beat every member. Both outcomes are normal.

### Member counts and degradation

For the trimmed rule: 3+ members trim normally; 2 fall back to a plain mean;
1 returns that model's own forecast; 0 gives `combined=None`. Three is where
the arithmetic stops working, not a tuned threshold — remove the highest and
lowest of two values and nothing remains. Degradation is reported as a
first-class `degraded` field, and `rule_applied` names what actually ran:
over exactly four members, trimming leaves two values whose mean *is* the
median, and it says `"median"`. It also warns, but only when the fallback
comes from losing models the caller asked for (a failure, the threshold, no
validation MAPE): asking for two models and getting their mean is not a
loss. A rule left with no members at all gets its own, louder warning,
since its `combined` is `None` and there is no ensemble forecast.

### `mape_threshold` filters degenerate fits

It defaults to 40 and accepts `None` to admit everything. It is there to drop
degenerate fits, not to screen for accuracy: sweeping the equivalent cutoff
from 0.20 to 0.80 on an external evaluation moved the screened rule's mean
error by 0.018 total with no monotone trend, and it trailed a plain trimmed
mean at every value. The trimmed rule already discards the worst member every
period, so screening hard *and* trimming spends the same robustness budget
twice.

Threshold rejection never raises, even when every model is rejected —
`combined` is `None` and the per-model results are still returned. A model
too inaccurate for the blend is a different situation from a model that
failed to fit, and discarding valid per-model results because the combination
step had nothing to do would be strictly worse for the caller.

### Weights are optional

`EnsembleResult.weights` is `Optional` and is **`None`** for weightless
rules. This is not cosmetic: `weights={}` already meant "no member survived",
so a trimmed ensemble returning an empty dict would have been
indistinguishable from total rejection. Code that assumes a populated dict
will break.

A model whose validation MAPE is NaN is kept by the trimmed rule and excluded
by the weighted ones. For `inverse_mape` that is arithmetic — there is no
`1 / NaN`. `equal` reads only the member count and could weight such a model,
but a model with no validation MAPE was never screened against
`mape_threshold` either, and the weighted rules are where an unscreened
member does the most damage.

Asking for several rules at once is close to free: the fits happened long
before, and combination is arithmetic over predictions already in memory.

---

## 4. Future driver values

When `forecast()` is not given future values for the declared independent
columns, they are generated **once, model-agnostically**, in the
orchestrator, and the same values feed every model's `transform_future()`.
`future_exog.py` holds a `FUTURE_EXOG_STRATEGIES` registry of pure functions
behind a single `resolve_future_exog()` entry point. Strategies:
`carry_forward` (default), `recent_average`, `historical_average`,
`seasonal_naive`, `linear_trend`, and `cycle`.

- **The default is `carry_forward`, not replay.** The original behaviour
  replayed the last `horizon` periods of history, which injects the recent
  pattern — including any one-off spike — into the future. Holding the last
  level flat is the least-surprising neutral baseline. `cycle` keeps the old
  behaviour for anyone who wants it.
- **Flags are type-aware.** Carrying a binary promo flag forward assumes a
  permanent promo; averaging one yields a nonsensical fraction. Continuous
  drivers use the chosen strategy; binary 0/1 event flags default to 0 ("no
  event unless supplied") — except under `seasonal_naive`, where they follow
  the cycle so recurring events still repeat. Flags come from an explicit
  `event_columns` list, else are auto-detected as columns whose values are
  only 0 and 1. The first rule here was "≤2 distinct, non-negative values",
  which caught quantities too: TV spend of 0 or 5,000, a 0%/20% discount and
  a 9.99/12.99 price were all forecast at 0 — the last one a free product.
  A two-value column is not necessarily an on/off switch; only 0/1 is.
  `cycle` is exempt from all of this and tiles every column as-is,
  reproducing the old output exactly.
- **On/off text is read from a fixed word list.** `true/false`, `yes/no`,
  `y/n`, `t/f` and `on/off` become 1/0, in the data and in the future table.
  Any other pair of words ("Promo"/"None") is an error, not a guess: guessing
  which word means "on" wrong would flip the sign of every effect the models
  learn, with nothing to show it.
- **Two warnings point at generated values that are probably wrong.**
  `carry_forward` repeats the last value, so a last period that was a
  promotion sets the level for the whole forecast; on real retail data that
  produced a forecast at 1.6–2.5× the same period last year. A warning fires
  when a generated driver's last value is ≥1.5× or ≤0.5× its median over the
  previous 13 weeks / 28 days, or an event flag is 1 there. It warns rather
  than substituting a "non-promo" value, because a substitute would be one
  more hidden assumption. Separately, a generated flag is 0 throughout (or
  repeated by week number under `seasonal_naive`, which puts a moving
  holiday in the wrong week), so an event that happened within a week of
  the same date last year triggers a warning naming those dates — one real
  forecast missed Black Friday this way.
- **`seasonal_naive` carries drift.** Pure seasonal naive repeats last year's
  level, which is wrong for a growing driver over a long horizon. The
  implementation fits a trend, adds the phase-wise seasonal residual, and
  projects both. A flat driver degrades to pure seasonal naive; under one
  full cycle of history it falls back to `carry_forward`.
- **Partial fill.** A supplied column is used as-is and only the missing
  columns are generated — matching the common case where next quarter's
  planned spend is known but the promotional calendar is not.
- **A supplied table is checked, not trusted.** Every model reads it by
  position, and each used to react to a bad one differently while the run
  still finished: too few rows made Prophet and SARIMAX fail and CatBoost
  forecast alone; a misspelled column was ignored and the real driver
  silently generated; a blank dropped two models; a date column was never
  read, so a table starting a week late was applied to the wrong periods.
  `check_future_table()` now rejects anything that does not match the
  forecast exactly — row count, columns, optional dates, values — before any
  model runs. Values outside a driver's historical range only warn, because
  scenario work needs them; the warning says what each model does with them
  (Prophet and NeuralProphet extend the effect linearly, SARIMAX and PyMC
  cap it, CatBoost treats it as the most extreme value seen), since that is
  where the models disagree most.
- **Transparency.** The resolved values (supplied + generated, with `ds`) are
  exposed as `forecaster.future_exog_used` and on each `ForecastResult`, so
  the assumptions behind a scenario can be inspected rather than trusted
  blind. A warning names the generated columns and the strategy used.
- Continuous output for a non-negative driver is clipped at 0, so
  `linear_trend` cannot project spend negative; a driver that genuinely goes
  negative in history is left alone. Out-of-range *clamping* stays in the
  adapters that already do it.

---

## 5. Per-model notes

### Prophet

The reference implementation of the contract — no lag features, no exog
matrix construction, no sampling to tune — which is why it was built first.
Independent columns keep their original names, since Prophet's
`add_regressor()` references columns by name.

- **Seeded, because it was not reproducible.** `yhat_lower`/`yhat_upper`
  varied between repeated `predict()` calls on the same fitted model —
  Prophet's uncertainty simulation draws from numpy's global random state.
  `predict()` now sets the seed (default 42, from `wrapper_config.random_seed`
  in the config) immediately before calling Prophet. The seed is an explicit
  wrapper constructor argument, never part of `params`, because `params` is
  passed straight to `Prophet(**params)`, which would reject it. Verified
  across separate processes, not just repeated calls.
- **`save()`/`load()` use `model_to_json()`, not pickle**, per Prophet's own
  documentation — the cmdstanpy backend does not deserialize reliably across
  environments. The JSON is wrapped in an envelope carrying wrapper state
  (regressor columns, frequency, params) so a reload reconstructs the
  wrapper, not merely the Prophet object.
- **`yearly_seasonality` fallbacks in the grid.** Tuned candidates hard-code
  a yearly Fourier order that assumes enough history to constrain a yearly
  cycle; on short history Prophet fits one anyway and extrapolates wildly
  (MAPE up to 802%, including negative predictions on a strictly positive
  metric). Rather than hand-code a data-span rule, the grid carries two safe
  `'auto'` candidates and lets selection favour them on short series.
- Prophet can still emit negative point forecasts on a low-level series. No
  output clamp is applied.

### CatBoost

Features are a **trend index plus cyclical seasonality**, not target lags.
`time_idx` is an absolute day/week offset from a fixed reference date (not a
row counter), so it stays continuous across the train/future boundary and is
gap-robust; sin/cos encodings cover yearly and half-yearly cycles, with
weekly and monthly added for daily data; calendar fields and independent
columns pass through.

This replaced a lag(1/2/3/7/14) + rolling(7/14) design, and the reason it
was replaced still applies. Because `predict()` is one batched call rather than a
per-step recursive loop, those features had to be frozen at train-end and
broadcast across the whole horizon. On real revenue data that scored 39.95%
MAPE against Prophet's 18.68%. Two plausible explanations were tested and
disproven — the tree-extrapolation ceiling (0 of 144 test rows exceeded the
training max) and monotonic staleness (the second half of the window scored
*better* than the first). Breaking the error down by known demand flags
found it: a December demand dip scored 67–90% MAPE against a 34–40%
baseline. A frozen snapshot of recent history cannot see a demand shift
coming; it keeps predicting the pre-shift baseline for the entire horizon.
After the redesign, that same segment scored 14.59% — the best segment
rather than the worst — with overall MAPE at 18.86%, matching Prophet.

Every feature is computed identically by `transform_train()` and
`transform_future()`, purely from dates, so nothing needs freezing. No
feature is declared categorical. Non-negative clipping is data-driven, and
applied only when the training target was itself always non-negative.

### SARIMAX

Fits `log1p(y)` and returns raw units via `expm1`, so `y` keeps one meaning
across every model and MAPE is scored consistently. The transform lives in
the wrapper for that reason; the adapter supplies raw `y` plus a
standardised exog matrix. `d` comes from an ADF test on the data; the grid
varies `(p, q)`, the seasonal `(P, Q)` at daily frequency, and
`fourier_order_annual`. Weekly disables the SARIMAX seasonal order — an
`s=52` term is prohibitively slow — and leans on the Fourier terms instead.

- **`trend` must be passed explicitly.** statsmodels defaults to no constant
  term, which is correct for a differenced series but means fitting a
  zero-mean process when `d=0` and `D=0`. This produced degenerate forecasts
  on every series the model did not difference.
- **Annual Fourier is a grid hyperparameter, not a fixed setting.** There is
  no correct default: on a ~3-year daily series annual Fourier is a clear
  win, and on a ~2-year weekly series it was actively harmful (82.81% at
  order 2 versus 48.74% with it off). It could not be grid-searched while it
  lived in the adapter, because the exog matrix is built once and reused by
  every candidate — so it moved into the wrapper, built from absolute day
  offsets against a fit-time reference date (gap-robust, phase-aligned).
  A related warning: synthetic data with a *clean* annual cycle pointed the
  opposite way (higher orders better); only the real series, whose annual
  pattern is weak across ~2 cycles, settled it. Worth remembering before
  trusting a generated fixture to justify a default.
- **Bias correction and intervals both use the constant innovation
  variance**, not `get_forecast()`'s horizon-growing forecast variance.
  `expm1` of the log-space mean returns the lognormal median, biasing low, so
  the point forecast adds `+σ²/2`. Using the growing variance would inflate
  later periods and feed the very explosion the clip guards against. The
  interval is `mean ± z·σ` in log space: a readable typical-error band whose
  width is stable across the horizon. The trade-off is real — it does not
  widen formally with horizon. With the growing variance, a 144-step forecast
  produced an `[8k, 700k]` band around an accurate ~80k point forecast. A
  documented consequence of the bias correction is that the corrected point
  can sit slightly above the interval centre; mean > median is correct
  lognormal behaviour.
- **Explosion is controlled at the source, with a divergence guard behind
  it.** The winning order is refit with `enforce_stationarity=True` (falling
  back to `False` only if that fit fails), which makes the hard clip a rare
  last resort rather than a routine crutch. It is still reachable: a
  near-explosive root or an exog coefficient extrapolating on out-of-range
  values can run the log forecast past the cap, and `expm1` then emits a
  degenerate flat line (observed: a ~3.8e6 flat forecast at ~1400% MAPE while
  other models sat at 20–30%). When the clip binds on a majority of the
  horizon, `predict()` raises `ModelPredictionError` and the model is dropped
  from the run. A clip-saturated forecast is a divergence, not a weak
  prediction, and is deliberately distinct from a model the threshold
  excludes.
- **Future exog is clamped to the training range.** A test-window spend spike
  far above any training value standardises to a large number and, times a
  fitted coefficient, blows up the log-linear predictor. `transform_train()`
  records the standardised min/max per column and `transform_future()` clamps
  to it. SARIMAX is linear in the exog, so extrapolating a coefficient beyond
  the trained range is unreliable anyway. Verified: a synthetic series with a
  40× spend spike across the test window, which diverged before, fits at ~17%.
- **The collinearity warning is diagnostic, not a fix.**
  `_warn_on_annual_collinearity()` flags any independent column whose
  variance is ≥30% explained by an annual sin/cos basis. It exists because
  adding drivers once made SARIMAX *worse* on a weekly series, and the cause
  turned out to be a stockout-weighting column that was itself seasonal
  (`corr(y) = -0.584`, `corr(annual sin) = +0.576`, `corr(trend) = -0.766`).
  On ~100 rows the fit cannot attribute that shared variance to both the
  column and six Fourier harmonics. That is an identifiability limit of the
  data, not a defect — the grid already resolves it by picking a low Fourier
  order once exog is present — so the right response is to make the cause
  visible, not to chase the number. The 0.30 threshold is calibrated against
  a frame mirroring the real data: it flags the seasonal stockout column at
  R²=0.95 and stays silent on spend columns and a sparse promo flag. Nothing
  is dropped.
- statsmodels pickles reliably, so SARIMAX needs no `save`/`load` override.

Known and unresolved: SARIMAX's daily MAPE has drifted between runs on
identical data with the identical winning candidate (17.73% vs 19.62%) while
the other four reproduce to the digit. Likely statsmodels MLE or BLAS
threading non-determinism — a hypothesis, not a finding, and left open rather
than "fixed" on a guess.

### NeuralProphet

A stateless adapter that renames and sorts, and nothing else: NeuralProphet
normalises internally, so the SARIMAX-style log/standardise/clamp treatment
would double-normalise. The future frame is built directly (`future_exog`
with `y=NaN`) rather than via `make_future_dataframe()`, which is why the
wrapper retains no training history and `save`/`load` stays a model-only
operation.

- **`n_lags=0` in 1.0, deliberately.** A reference implementation grid-searched
  `n_lags` in {0, 7} and predicted on a frame containing the test period's
  real `y`. With `n_lags>0` NeuralProphet is autoregressive, so every test
  point becomes an effectively one-step-ahead prediction that has already
  seen recent actuals. Ported as-is that would be (a) not comparable with
  four models forecasting the window blind, (b) corrupting — the flattered
  MAPE feeds weighting and screening, and (c) inexpressible: `predict(horizon,
  future_exog)` has no actuals to hand the model. There is also a genuine
  contract tension, since NeuralProphet fixes `n_forecasts` at construction
  while `BaseModel` is fit-once/predict-any-horizon; supporting AR properly
  means refitting per horizon. By the gate rule that is a contract discussion,
  not a wrapper special case.
- **Seeded and CPU-pinned.** It is stochastic, and Lightning otherwise picks
  MPS on Apple silicon and CPU elsewhere, producing different numbers from
  identical code.
  `epochs` and `learning_rate` are both pinned — leaving either `None` runs a
  learning-rate range test and auto-sizes epochs, which is slow and makes two
  runs on the same data non-comparable.
- **`collect_metrics: False` is load-bearing.** With metrics enabled
  NeuralProphet attaches a Lightning logger, and Lightning then runs
  `_log_hyperparams()` → `save_hparams_to_yaml()` → `OmegaConf.save()`, which
  rejects `ConfigTrain`'s `loss_func: Union[str, _Loss, Callable]` and kills
  `fit()` before a single epoch. On newer stacks the same logger fails
  differently, via TensorBoard → TensorFlow → jax → `np.dtypes.StringDType`.
  The config sets it false for both frequencies, and both the config and the
  test defaults assert it — the first attempt at this fix landed in the
  `daily` block only, and `weekly` was the frequency actually failing.
- **`weights_only=False` in the torch load fallback, with the trainer
  restored.** PyTorch ≥2.6 flipped `torch.load`'s `weights_only` default to
  `True`, and NeuralProphet saves a full Python object rather than a state
  dict, so the new default refuses to unpickle it. The wrapper tries
  NeuralProphet's own loader first and falls back. Allowlisting via
  `add_safe_globals()` was rejected — a saved NeuralProphet drags in a tree of
  its own config classes, so the allowlist would need revising with every
  version —
  and `weights_only=False` is appropriate for a file written by this
  library's own `save()`. The fallback must also call
  `restore_trainer(accelerator="cpu")`: `save()` deliberately nulls the
  Lightning trainer (it is not picklable), so a bare `torch.load` returns a
  structurally complete model that fails on the *next* `predict()`.
  `test_neuralprophet_compat.py` stubs `neuralprophet` and `torch` to drive
  this path on any torch version, because it never executes on an older one.
- **Config widened to stop under-fitting.** On a synthetic sweep
  NeuralProphet was the consistent outlier — 18.7% MAPE on a mid-series level
  shift against ~2% for the structural models, which fit clean parametric
  signals near-perfectly. A middle learning rate (0.03), higher epochs
  (150/300) and more changepoints (20 daily / 10 weekly) brought the level
  shift to ~1.9%. An interim `changepoints_range = 0.9` regressed a
  smooth-trend series (5.6% → 11.8%) by leaving a short, noisy final segment
  to extrapolate from; reverting to the 0.8 default kept the win.
- `yearly_seasonality="auto"`, not forced — forcing it on short history has
  produced unusable forecasts twice in this project. Non-negative clipping is
  data-driven, mirroring CatBoost. No uncertainty bounds in 1.0, though
  `quantiles=[...]` is a natural follow-up now the output layer handles them.

### Bayesian PyMC

A structural model — linear trend + annual Fourier (+ weekly for daily) + a
linear regression on standardised exog — observed through a Student-T
likelihood and sampled with NUTS. The adapter is close to SARIMAX's minus the
log1p and the collinearity warning; target standardisation and Fourier live
in the wrapper, mirroring that split.

- **`predict()` reconstructs `mu` from stored posterior draws, in pure
  numpy.** A reference implementation built one `Deterministic('mu')` over
  train + test together and sliced the test rows out of the posterior, which
  only works when both are sampled at once. Under `BaseModel`, `fit()` sees
  only training rows and `predict()` only future dates — so the wrapper
  samples on train once, keeps the flattened posterior draws of every
  coefficient, and reconstructs `mu` at future dates as a linear combination
  of them. Same model, same numbers, and `predict()` never imports pymc.
- **Priors are keyed off column shape, because there is no metadata
  channel.** The library takes arbitrary `independent_columns` and the
  adapter and wrapper are constructed independently, so nothing can tell the
  wrapper which column is spend. A continuous driver (>2 distinct values)
  gets an informative prior centred on its OLS slope against the standardised
  target; a binary flag (≤2 distinct values — this prior rule is the
  wrapper's own, separate from the future-driver flag detection in §4) gets a
  zero-centred prior. Prior
  estimation drops non-finite pairs.
- **Bounds sample the likelihood, not just `mu`.** Banding the latent mean
  captures parameter uncertainty but omits observation noise, and
  under-covered badly — only ~34% of real daily actuals fell inside a nominal
  94% band. Retaining the posterior `sigma`/`nu` draws and drawing Student-T
  noise restores coverage. The point forecast is still `mu.mean` (the
  predictive mean for `nu > 1`), so MAPE is unaffected; only the band widened.
- **No unconditional zero clamp.** The back-transform once applied
  `np.clip(..., 0.0, None)` regardless of the data, so any target that
  legitimately goes negative — net revenue after refunds, inventory deltas,
  week-on-week change — had every negative prediction silently rewritten to
  0. Clipping is now data-driven like CatBoost's.
- **`save`/`load` via a NetCDF sidecar.** `InferenceData` is not pickled
  directly: the trace is written to `<path>.nc` via arviz and the rest of the
  wrapper is pickled. `predict()` depends only on the pickled numpy state, so
  a reload reproduces predictions exactly even if the sidecar is absent.
- **The daily config is tuned for sampling cost. The constraint is speed,
  not validity.** Weekly (≈100 rows, one annual cycle) samples cleanly
  at 2000 tune / 2000 draws. Daily (≈1144 rows) took about an hour on a first
  full-grid run: an annual basis, a weekly basis, and drivers that themselves
  track the annual cycle make a poorly-conditioned ridge, and NUTS answers
  with very deep leapfrog trees. Three changes fixed it — halve draws/tune to
  1000 (a sweep showed point MAPE flat across a 4× range: 500 → 11.46%,
  1000 → 11.56%, 2000 → 11.65%, pure Monte-Carlo jitter, while time scaled
  linearly), keep the Fourier order low (3–4 on daily data, never 6, which was
  slowest and never won), and grid-select `beta_fourier_sigma` (0.3 tight vs
  1.0 loose) rather than hardcoding it. A tight Fourier prior regularises the ridge — on
  the collinear synthetic it halved sampling time *and* improved MAPE (1.0 →
  14.55%, 0.3 → 11.56%) — but it underfits strong, well-supported
  seasonality, so which wins is data-dependent and belongs in the grid.
  One correction is worth recording: an early diagnostic reported ~40%
  divergences, and the daily posterior was briefly judged untrustworthy. That
  figure was an artifact of an 80-step warmup in a throwaway test. With
  adequate warmup the posterior is clean (0 divergences, R-hat ≈ 1.0).
- **A macOS linker workaround runs before the first model is built.**
  PyTensor appends `-ld64` to compile commands on macOS 15+, and current
  Apple toolchains no longer have that linker, so every compile failed with
  `ld: library 'd64' not found` and Bayesian PyMC dropped out of every run on
  those machines — silently, because failure isolation kept the other four
  going. The PyTensor pin cannot move to pick up a fix (see §7), so
  `models/_pytensor_compat.py` probes the configured compiler once per
  process and drops that one flag only when the probe fails with exactly that
  error. Probe-then-patch rather than patch-always because PyTensor added the
  flag deliberately, to dodge bugs in Xcode 15's newer linker; a toolchain
  that still accepts it keeps it.
- It remains by far the slowest model — a full NUTS sample per grid
  candidate. That is inherent to the approach. `nutpie` was considered as a
  further speedup and not adopted in 1.0: it addresses speed only, and the
  grid-selected regularisation already brought daily back to a practical
  runtime.

---

## 6. Output layer

`plotting.py` offers two chart families at each stage: overlays
(`plot_evaluation()`, `plot_forecast()`) and dedicated per-model figures
(`plot_evaluation_per_model()`, `plot_forecast_per_model()`), plus
`plot_model_comparison()`, a bar chart of test MAPE or WAPE (with an
optional seasonal-naive bar) that makes "the ensemble is not
automatically better" visible at a glance. The per-model figures exist
because the overlay crowds at even two models when their lines coincide, and
a first faceted-grid attempt made each panel too small to read.

- **One chart per combination rule, not one chart for the first rule.** Both
  overlays return `{method: Axes}` — always a dict, matching the convention
  `evaluate()` and `evaluate_ensemble` already follow — and the per-model
  functions emit an `ensemble_<method>` figure for every rule that formed. An
  earlier cut collapsed to the first requested method on the grounds that
  several rules on one time axis bury the per-model lines. The premise was
  right and the conclusion was wrong: with `ensemble_method=None` a caller
  pays to build all three rules and could then see only one of them, with no
  indication the other two existed. Separate charts keep the axis readable
  *and* keep every rule visible, each against the same models and actuals,
  which is what makes the rules comparable against the truth rather than only
  against each other. A caller-supplied `ax` is consequently refused for more
  than one rule — silently ignoring the extras is the defect this shape
  exists to prevent.
- **The bar chart is the deliberate exception**: every rule shares one Axes,
  because comparing rules head-to-head is what a bar chart is for. Its labels
  are rotated, since an "ensemble (inverse_mape)" label is several times
  longer than a model name and collides with its neighbours at that bar count
  when drawn horizontally.

- **Charts draw and return; they never call `plt.show()`.** Rendering is the
  caller's decision, which keeps the functions testable under a headless
  backend and non-blocking in scripts.
- **Colors auto-cycle and only the ensemble has a fixed style.** No hardcoded
  per-model color map, so a newly registered model charts correctly with no
  change here — the same registry-generic stance as the ensemble.
- **The combined line is labelled by the rule that actually ran**, so a
  degraded two-member plain mean is distinguishable from a five-member
  trimmed mean.
- **"Ensemble requested but none formed" is labelled loudly, in red, while
  the individual model results are still drawn.** Rejection from the blend is
  a trust signal, not a reason to hide a real forecast.
- **Exporters never raise on empty input.** Each `build_*_table()` returns a
  correctly-columned empty DataFrame, and `export_forecast_csvs()` always
  writes every file. A run where everything was rejected produces predictable
  header-only CSVs rather than a crash or a missing file.
- **Both a `final_ensemble_<method>` and a `final_best_model` CSV are
  written**, always, with `model_summary` carrying validation MAPE, test
  MAPE, method, rule applied, weight, status and `best_model_check`. The
  export deliberately does not bake in the choice between them.
- **New summary columns go after the existing ones.** `validation_wape`,
  `wape`, `zero_actuals_excluded`, `beats_naive` and `selection_metric` were
  appended, and the baseline is an extra row, so a reader that selects
  columns by name or position, or filters rows by status, is unaffected.
  `rejected_models` keeps its `mape` column name even when it holds a WAPE;
  the new `selection_metric` column says which.
- **`forecaster=` on the exports.** Building the score dicts by hand was the
  usual way to get a file subtly wrong (a forgotten threshold, the test MAPE
  passed as validation). The exports read the forecaster's public attributes
  only, so `exporters.py` still never imports `forecaster.py`, and explicit
  arguments still win.
- **The best single model is chosen on validation, with test as a check.**
  It used to be the lowest test MAPE with no threshold, so the export was
  selected on the window that is meant only to be reported, and in one real
  run a model the ensemble had rejected was exported at 103% error. It is
  now the lowest validation MAPE among the models within `mape_threshold` on
  validation *and* on test. Ranking on test was measured and picked worse
  (§2, "Window sizes"), but as a veto it carries signal: in the same
  comparison, models that passed validation and then scored above 40% on
  test had a median error of 24.8% over the periods that followed, against
  5.6% for the rest. The cost is that the chosen model's test MAPE is
  slightly optimistic — test helped choose it — which `best_model_check`
  states. When no model passes, the file is empty and a warning lists each
  model's scores, consistent with "exporters never raise on empty input".
- The `export_forecast()` bundle imports matplotlib lazily, so
  `export_forecast_csvs()` alone stays matplotlib-free.

---

## 7. Packaging

Every runtime dependency is pinned to an exact tested version. Five pins and
declarations are load-bearing rather than tidy, and all are commented inline
in `pyproject.toml`:

- **`numpy==1.26.4`** — required by `neuralprophet==0.8.0`'s `numpy<2`
  constraint, not chosen. numpy ships no 3.13 wheel for its 1.x line, so on
  Python 3.13 pip compiles it from source. See
  [`installation.md`](installation.md).
- **`pymc==5.25.1`** — held back by the numpy pin, since later PyMC releases
  require numpy 2. It also fixes PyTensor below 2.32, which is why the macOS
  `-ld64` workaround (§5) is carried in this library rather than taken from
  an upgrade.
- **`matplotlib==3.10.9`** — held on the 3.10.x line: 3.11.x dropped Python
  3.10 support and broke installs on 3.10.
- **`setuptools>=68,<82`** — pytorch-lightning (via neuralprophet) imports
  `pkg_resources` at runtime, and setuptools 82 removed it entirely. A loose
  `>=68` was tried first and failed for exactly this reason: pip satisfied it
  with 82+.
- **`requires = ["setuptools>=77.0"]`** in `[build-system]` — the PEP 639
  bare-string `license = "Apache-2.0"` is only understood from 77.0.0.
  A plain `pip install .` hides this because pip resolves a current
  setuptools in its isolated build environment; it surfaces on a
  `--no-isolation` build.

Two more declarations exist because their absence was confirmed, not
theorised:

- `[tool.setuptools.packages.find] include` — without it, any stray
  top-level folder (a venv named `test_env`, a `catboost_info` artifact) is
  mistaken for a second package by flat-layout discovery and the build fails.
- `[tool.setuptools.package-data]` — without it, `config/*.yaml` is silently
  absent from a built wheel. Editable installs symlink the source tree and
  mask this completely; only inspecting a real wheel reveals it.

`requirements.txt` mirrors the runtime pins for tools that expect a plain
requirements file; there is deliberately no `requirements-dev.txt`, since the
`dev` extra is the single dev dependency list.

**Known and open:** 24 CVEs in `torch`/`pytorch-lightning`, forced by
`neuralprophet==0.8.0`'s own upper bounds. Adopting a NeuralProphet
pre-release would resolve them but carries breaking API changes and no
testing against this code; the pin stands until a vetted release is adopted.

---

## 8. How this library is tested

The suite fits real models rather than mocking them. Mocking is used only
where it is the only way to hit a path deterministically — CatBoost's
prediction-clipping tests monkeypatch `_fitted_model.predict` to return a
fixed array, the NeuralProphet torch-compat tests stub `torch` because
the fallback branch never runs on an older version, and the PyTensor
linker-workaround tests stub the platform and compiler because the patch
branch only runs on an affected Mac.

Beyond unit tests, three things repeatedly found what unit tests did not:

- **A synthetic edge-case sweep** with known ground truth — short and long
  history, flat, regime-shift and strong-seasonal shapes, collinear, trending
  and negative drivers, zero-inflated and spiky targets, gaps, NaNs, unsorted
  and zero-variance data. This is what surfaced the failure-isolation gap and
  NeuralProphet's under-fitting.
- **An invariant fuzzer** over random valid datasets — forecast length, bound
  ordering, weight sums, non-negativity, MAPE ≥ 0.
- **Real datasets.** Several decisions in this document exist only because
  synthetic data pointed the wrong way: SARIMAX's Fourier order, CatBoost's
  entire feature design, the plausibility-warning margin. Synthetic fixtures
  are clean parametric signals, which is exactly the regime where structural
  models look perfect and everything else looks broken.

Two practices are recommended for anyone working on this library, both
established after a defect that could have been avoided by them.

**Obtain the traceback before writing the fix.** The NeuralProphet
`ConfigValueError` was first diagnosed by reading library source rather than
by confirming against a stack trace. The resulting fix was plausible,
internally consistent, and wrong, and contrary evidence that was already
visible was rationalised rather than investigated.

**Check a proposed rule against real data before shipping it.** The first
version of the plausibility check used "entirely outside the historical
range", which both missed the case that motivated it (the forecast tail
decayed back inside the range) and would have raised false positives on every
healthy trend.

---

## 9. Deliberate gaps

None of these block a stable API; all are additive.

- **Rolling-origin backtesting.** Every score comes from a single test
  window, so it reflects one stretch of history rather than an average over
  several origins. Measured against a longer single validation window (§2,
  "Window sizes"), three folds improved weekly model choice only a little
  further at three times the selection cost, and not daily choice at all.
  Untested: whether folds help more during a full grid search, where one
  window may favour candidates that suit it specifically. The per-window
  driver check (§2) is already written as one function per fit window, so a
  backtest can call it per fold.
- **Benchmark-relative metrics (RelMAE) and MASE.** The seasonal-naive
  baseline is reported and warned about (§2), and selection can run on WAPE,
  but nothing is selected or weighted relative to the baseline. The
  metric-aware grid sorting, screen and weighting that WAPE needed are the
  path for it: `seasonal_naive()` in `evaluation/metrics.py` already gives
  the denominator, and `selection_metric` is where a third option would go.
- **Family de-duplication.** Prophet and NeuralProphet share an
  additive-decomposition methodology (signed-error correlation 0.787, the
  highest of any pair involving either), so a blend containing both is less
  diverse than its member count suggests. Nothing accounts for that.
- **Acting on error diversity.** `test_error_correlation` (and the
  `error_correlation` export) reports the pairwise correlation of the
  models' signed test errors, so a user can see whether there is anything to
  exploit on their own series; the ensemble does not use it to choose or
  weight members.
- **Non-negativity is not enforced library-wide.** Prophet can emit negative
  point forecasts; SARIMAX is non-negative structurally via its `log1p` fit,
  which is also why it is left out of any run whose target goes negative;
  CatBoost, NeuralProphet and Bayesian PyMC clamp only when the training
  target was itself non-negative.
- **NeuralProphet autoregression** (`n_lags > 0`), which needs the contract
  discussion described in §5.
