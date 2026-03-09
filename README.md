# Nasdaq_analysis
# Stylized Facts & Regime-Filtered Momentum

A quantitative research project analysing 6 years of daily equity log returns (March 2020 – February 2026) and using empirical market properties to build a risk-aware momentum strategy.

## Overview

This project documents four canonical stylized facts of financial returns and translates them into a systematic trading strategy:

1. **Heavy tails** — Return kurtosis of 11.09; empirical 3-sigma events occur 5x more frequently than a Gaussian model predicts.
2. **Volatility clustering** — Autocorrelation of absolute returns remains significant beyond lag 50, motivating GARCH-based forecasting.
3. **Drawdown dynamics** — Maximum drawdown of -39.7% with a 932-day recovery; sharp crash vs. slow grind regimes behave very differently.
4. **Market regimes** — A two-state Hidden Markov Model separates a bull regime (ann. vol 17%, ann. return +35%) from a bear regime (ann. vol 43%, ann. return -55%).

## Strategy Results

| Strategy | CAGR | Sharpe | Sortino | Max DD |
|---|---|---|---|---|
| Buy & Hold | 13.07% | 0.52 | 0.66 | -39.73% |
| Momentum 12m | 7.63% | 0.50 | 0.51 | -29.37% |
| Regime-Filtered Momentum | 10.73% | 0.90 | 0.89 | -13.84% |

Regime filtering lifts the Sharpe ratio by 73% and cuts maximum drawdown by two-thirds relative to buy-and-hold.

## Methods

- **HMM**: Two-state Gaussian Hidden Markov Model estimated via Baum-Welch; states decoded with Viterbi.
- **GARCH(1,1)**: One-step-ahead conditional volatility forecast; RMSE vs 30-day realised vol = 0.00293.
- **Signal**: Long when 12-month momentum is positive AND HMM is in bull state AND GARCH forecast is below the 70th percentile threshold.

## Robustness

Results hold across momentum lookback windows (126d, 189d, 252d) and volatility threshold quantiles (Q60, Q70, Q80).


## Requirements

```
python >= 3.9
numpy, pandas, matplotlib
hmmlearn
arch
scipy, statsmodels
```
##paper
find paper called results
