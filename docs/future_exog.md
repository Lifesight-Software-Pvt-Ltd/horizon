# Future-exogenous-data strategies

When you call `forecast()` with independent (driver) columns but do not supply
their future values, `HorizonForecaster` has to *assume* them. A
**future-exogenous-data strategy** decides how. The values are generated
once, model-agnostically, and fed to every model, so the comparison stays
fair. The result is a **scenario**, not a prediction — the forecast is only
as good as these assumed inputs, which is why `forecast()` warns and exposes
the assumptions as `forecaster.future_exog_used`.

Set the strategy once on the constructor:

```python
HorizonForecaster(
    ...,
    independent_columns=["spend", "promo"],
    future_exog_strategy="auto",            # default: last_year or carry_forward
    event_columns=["promo"],                # explicit 0/1 flags (optional)
    future_exog_aggregator="mean",          # "mean" | "median"
    recent_average_window=28,               # N for recent_average
)
```

## The strategies

The default, `"auto"`, is `last_year` when the history covers a year (364
days, or 52 weeks) and `carry_forward` when it is shorter. The strategy used
is stored in `forecaster.future_exog_strategy`. Pass any strategy below by name
to use it instead.

| Strategy | What it does | Good when |
|---|---|---|
| `last_year` | Copies each driver from the same weekday 364 days earlier (daily) or the same week 52 weeks earlier (weekly), so the latest year of history repeats for as long as the horizon runs. A driver that is live now but was 0 on every copied date is new since last year, and holds its last value instead. Falls back to `carry_forward` under a year of history. | Marketing data with a year or more of history, where this year's plan follows last year's. The default then. |
| `carry_forward` | Holds the last observed value flat across the horizon. | A neutral baseline; short horizons; slowly-changing drivers. The default under a year of history. |
| `recent_average` | Mean (or median) of the last `recent_average_window` periods. | A one-off recent spike or dip you do not want to project forward. |
| `historical_average` | **Season-aware** mean/median — by day-of-week (daily) or week-of-year (weekly), not one flat mean. | Drivers with a stable seasonal shape and no strong trend. |
| `seasonal_naive` | The seasonal cycle repeated, **plus drift** (the fitted historical trend). Falls back to `carry_forward` under one full cycle of history. | Long horizons on drivers that are both seasonal and trending. |
| `linear_trend` | A fitted trend line projected forward. | Steadily trending drivers (e.g. a spend ramp). |
| `cycle` | Replays/tiles the trailing history verbatim, and is exempt from the flag handling below. | Reproducing a naive seasonal repeat exactly. |

Why `last_year` is the default with a year of history: on the testers' real
series (8 daily and 6 weekly, 42 forecasts of 4 weeks to 6 months), a
forecast made with `last_year`'s drivers had a median WAPE 0.9 points above
the same forecast made with the drivers that actually happened on daily data,
and level with it on weekly data, against 3.6 and 6.9 points above for
`carry_forward`. The day it copies keeps the weekday, because spend follows
weekdays: copying the same calendar date a year back instead more than
doubled the error in the daily spend curve. It copies values exactly, with no
growth: if spend has grown since last year, the forecast assumes last year's
level.

All strategies are defined for **any horizon** — short ones take the tail,
long ones tile — so the same choice works whether you forecast 7 periods or
700.

## Type-aware handling of event flags

Carrying a binary promo flag forward would assume a *permanent* promo;
averaging it would produce a nonsensical fractional flag. So:

- **Continuous drivers** (spend, impressions) use the chosen strategy.
- **Binary 0/1 event flags** (promos, holidays) default to **0** — "no event
  unless you supply one" — *except* under `seasonal_naive`, where they follow
  the cycle so recurring events still repeat, by week of the year (weekly) or
  day of the week (daily), and under `last_year`, where they are copied from a
  year earlier like every other driver. A holiday whose date moves, such as
  Black Friday, can land a week off, and under `last_year` an event that
  happened once last year is assumed to happen again; see the warnings below.

Flags are taken from the explicit `event_columns` list if you give one;
otherwise they're auto-detected: a column whose values are only 0 and 1
(`True`/`False`, and the words `yes/no`, `y/n`, `t/f`, `on/off`, are read as
1/0 first). A column with two other values — TV spend of 0 or 5,000, prices
of 9.99 and 12.99, a 0% or 20% discount — is a quantity, not a flag, and
follows the chosen strategy. The `cycle` strategy is exempt from all of this
and tiles every column as-is.

## Warnings about generated values

These warnings point at generated values that are likely wrong:

- **A last period that looks like a promotion, held by `carry_forward`.**
  When a generated driver's last value is at least 1.5× (or at most 0.5×)
  its median over the previous 13 weeks / 28 days, or an event flag is 1 in
  the last period, carry-forward would hold that level for the whole
  forecast. The warning names the driver, its value and its usual level.
  Supply planned values, or use `recent_average`.
- **An event from last year left out or misplaced.** When a flag is
  generated and it was 1 within a week of the same date last year (364 days,
  so the weekday matches), the warning names those dates. Generated flags
  are 0 throughout — and under `seasonal_naive` they repeat by week number
  or weekday, which puts a holiday that moves (Black Friday) in the wrong
  period. Under `last_year` they are copied by date, so a holiday that moves
  lands in the wrong period and a one-off event repeats. Supply the dates of
  known events in `future_independent_data`.
- **A driver that started or stopped since last year, under `last_year`.**
  Each case gets its own warning, which names every driver and says exactly
  what was done to it:
  - **Held at its last value.** Above 0 in each of the last 28 days (4
    weeks) but 0 on every copied date: the driver is new since last year, and
    copying would set it to 0 for the whole forecast. The warning gives the
    value it is held at.
  - **Switched back on, as seasonal.** At 0 in each of the last 28 days (4
    weeks), and also at 0 in the same 28 days a year earlier, but above 0 on
    a copied date: an off-season. It is copied, and the warning gives the date
    it comes back on and its total over the forecast. On the testers' series,
    9 of 10 channels like this did come back.
  - **Switched back on, although it looks stopped.** At 0 in each of the last
    28 days (4 weeks), but running in the same 28 days a year earlier. It is
    still copied, because the data cannot tell a pause from a stop (4 of the
    11 channels like this on the testers' series came back), and the warning
    gives the date it comes back on and its total. If it has stopped, supply
    0 for it.
- **Under a year of history, under `last_year`.** There is no year to copy,
  so the drivers hold their last value, as `carry_forward` does, and the
  warning says so.

The auto-generation warning itself names the strategy, and under `last_year`
the dates whose values were copied.

## Partial fill

You rarely know every driver's future. You can supply future values for
*some* columns and let the strategy generate the rest — pass a DataFrame with
only the columns you know:

```python
# you have next month's planned spend, but not the promo calendar
planned = pd.DataFrame({"spend": [...]})        # len == horizon, "promo" omitted
forecasts = forecaster.forecast(horizon=30, future_independent_data=planned)
# spend is used as-is; promo is generated by the strategy
```

The auto-generation warning names only the columns that were generated.

The supplied table must have exactly one row per forecast period, only
declared drivers (plus, optionally, the date and target columns), and clean
numeric values (on/off words such as `yes/no` count as 1/0); see the table in
[`quickstart.md`](quickstart.md) for the full list of checks.

## Spend what-ifs

Supplying the spend you plan is the main use of `future_independent_data`.
Two limits apply to every such scenario.

**Spend above the highest value in your data cannot be estimated reliably.**
The models disagree about values they have never seen: Prophet and
NeuralProphet extend the driver's effect in a straight line, SARIMAX and
Bayesian PyMC cap the driver at its historical range, and CatBoost treats it
like the most extreme value it has seen. On synthetic data with a known spend
effect, with spend held for the whole forecast at a multiple of the highest
past value, the default trimmed ensemble captured this share of the true
uplift (1.00 is right):

| Spend effect | Frequency | 1× | 1.5× | 2× | 5× |
|---|---|---|---|---|---|
| Linear | daily | 0.88 | 0.66 | 0.57 | 0.42 |
| Linear | weekly | 0.92 | 0.67 | 0.57 | 0.42 |
| Diminishing returns | daily | 1.28 | 1.17 | 1.16 | 1.46 |
| Diminishing returns | weekly | 1.17 | 0.99 | 0.96 | 1.15 |

With a linear effect the forecast understates the uplift by about a third at
1.5× and about half at 2×; with diminishing returns it overstates it. Beyond
the highest past spend the data cannot tell the two apart, so no setting
fixes this. `forecast()` warns whenever a supplied value is outside the range
in your data.

**Carry-over is not modelled.** Every model uses the same period's spend only.
When part of spend's effect carries into later periods, a sustained change in
spend is understated even within the historical range: in the same test, with
half of each week's effect carrying into the next, the default ensemble
captured about 45% of the true uplift, and no model more than half.

## Inspecting the assumptions

After a forecast, the resolved future drivers (supplied **and** generated,
with a `ds` column) are available for inspection or plotting:

```python
forecasts = forecaster.forecast(horizon=30)
forecaster.future_exog_used            # DataFrame: ds + every driver used
forecasts["sarimax"].future_exog_used  # the same, also on each ForecastResult
```

Because the forecast is a scenario, always sanity-check these assumed inputs
before trusting the output — especially for an exogenous-variable-driven
model, where the assumed drivers can dominate the result.

## Robustness notes

- History is coerced to numeric and every aggregation is NaN-aware.
- A **non-negative** driver's generated future is clipped at 0, so a projected
  trend cannot push a spend column negative; a driver that genuinely goes
  negative in history is left unclipped.
- Out-of-range *clamping* (keeping a future value within the training range) is
  handled by the SARIMAX and Bayesian PyMC adapters — the only two that
  transform their exog — not here; the strategies stay separation-clean.
