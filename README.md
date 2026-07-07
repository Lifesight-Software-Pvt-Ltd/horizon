# Horizon Forecaster

An open-source ensemble time-series forecasting library that combines Prophet, NeuralProphet, SARIMAX, CatBoost, and Bayesian PyMC models with configurable ensemble weighting based on MAPE thresholds.

## Features
- Supports daily and weekly forecasting granularity
- Flexible target variables (orders, revenue, conversions, installs, etc.)
- Configurable independent variables (spend, impressions, clicks, promos, organic, etc.)
- Automatic model selection via MAPE threshold-based ensemble weighting
- Outputs both final ensemble and individual model forecasts (plots + CSV)
- Transparent reporting of rejected models with their MAPE scores

## Installation

```bash
pip install -e .
```

## Status

This project is under active development.

## License

See [LICENSE](LICENSE).
