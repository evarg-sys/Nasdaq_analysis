import warnings
warnings.filterwarnings("ignore")
import os

"""
End-to-end project:
- PHASE 1: Data loading & EDA (stylized facts)
- PHASE 2: Regime detection with HMM
- PHASE 3: Volatility forecasting with GARCH
- PHASE 4: Baseline momentum strategy
- PHASE 5: Regime-aware momentum strategy
- PHASE 6: Simple robustness checks
- PHASE 7: Summary of results

Run this file directly (Python 3.9+ recommended).

Required packages (install via pip if missing):
    pip install yfinance pandas numpy matplotlib seaborn scipy hmmlearn arch tabulate
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from tabulate import tabulate


plt.style.use("seaborn-v0_8-darkgrid")

# Fast run defaults: quick EDA first, no GUI plotting, and bounded row count.
SHOW_PLOTS = False
RUN_FULL_PIPELINE = True

MAX_ROWS = 1500
PLOT_DIR = "plot"
os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def download_data(ticker: str = "^IXIC",
                  start: str = "2000-01-01",
                  end: str = "2026-02-24") -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end)

    # yfinance may return either flat columns or a MultiIndex with ticker level.
    if isinstance(data.columns, pd.MultiIndex):
        top_cols = set(data.columns.get_level_values(0))
        if "Adj Close" in top_cols:
            price = data["Adj Close"]
        elif "Close" in top_cols:
            price = data["Close"]
        else:
            raise ValueError(f"Downloaded data missing 'Adj Close'/'Close'. Columns: {list(data.columns)}")

        # Single-ticker downloads can still be a 1-col DataFrame here.
        if isinstance(price, pd.DataFrame):
            price = price.iloc[:, 0]
    else:
        if "Adj Close" in data.columns:
            price = data["Adj Close"]
        elif "Close" in data.columns:
            price = data["Close"]
        else:
            raise ValueError(f"Downloaded data missing 'Adj Close'/'Close'. Columns: {list(data.columns)}")

    out = pd.DataFrame({"price": price}).dropna()
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute:
    - log returns
    - rolling volatility (30d, 252d)
    - cumulative returns
    - drawdowns
    - moving averages
    - momentum (12m, and alt windows for robustness)
    """
    prices = df["price"]
    rets = np.log(prices / prices.shift(1))
    df["ret"] = rets

    df["vol_30"] = df["ret"].rolling(30).std() * np.sqrt(252)
    df["vol_252"] = df["ret"].rolling(252).std() * np.sqrt(252)

    df["cum_ret"] = (1 + df["ret"]).cumprod()
    rolling_max = df["cum_ret"].cummax()
    df["drawdown"] = df["cum_ret"] / rolling_max - 1

    df["ma_50"] = prices.rolling(50).mean()
    df["ma_200"] = prices.rolling(200).mean()

    df["mom_6m"] = prices.pct_change(126)
    df["mom_9m"] = prices.pct_change(189)
    df["mom_12m"] = prices.pct_change(252)

    return df.dropna()


def limit_rows(df: pd.DataFrame, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.tail(max_rows).copy()


def quick_eda(df: pd.DataFrame):
    r = df["ret"].dropna()
    if r.empty:
        print("\n--- Quick EDA ---")
        print("No returns available after preprocessing.")
        return

    print("\n--- Quick EDA ---")
    print(f"Rows used: {len(df):,}")
    print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Mean daily log return: {r.mean():.5f}")
    print(f"Std daily log return:  {r.std():.5f}")
    print(f"Skewness:              {stats.skew(r):.3f}")
    print(f"Kurtosis (normal=3):   {stats.kurtosis(r, fisher=False):.3f}")
    print(f"Worst day:             {r.min():.2%}")
    print(f"Best day:              {r.max():.2%}")


def maybe_show_plot(filename: str):
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def performance_stats(returns: pd.Series, name: str = "") -> dict:
    if len(returns) == 0:
        return {"Name": name, "CAGR": np.nan, "Sharpe": np.nan,
                "Sortino": np.nan, "MaxDD": np.nan, "WinRate": np.nan}

    r = returns.dropna()
    if len(r) == 0:
        return {"Name": name, "CAGR": np.nan, "Sharpe": np.nan,
                "Sortino": np.nan, "MaxDD": np.nan, "WinRate": np.nan}

    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside != 0 else np.nan

    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    win_rate = (r > 0).mean()

    return {
        "Name": name,
        "CAGR": ann_ret,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "WinRate": win_rate,
    }


def print_stats_table(stats_list, title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    headers = ["Name", "CAGR", "Sharpe", "Sortino", "MaxDD", "WinRate"]
    table = []
    for s in stats_list:
        row = [
            s["Name"],
            f"{s['CAGR']:.2%}" if pd.notna(s["CAGR"]) else "NA",
            f"{s['Sharpe']:.2f}" if pd.notna(s["Sharpe"]) else "NA",
            f"{s['Sortino']:.2f}" if pd.notna(s["Sortino"]) else "NA",
            f"{s['MaxDD']:.2%}" if pd.notna(s["MaxDD"]) else "NA",
            f"{s['WinRate']:.2%}" if pd.notna(s["WinRate"]) else "NA",
        ]
        table.append(row)
    print(tabulate(table, headers=headers, tablefmt="github"))


# --------------------------------------------------------------------------------------
# PHASE 1: Stylized facts
# --------------------------------------------------------------------------------------

def analyze_heavy_tails(df: pd.DataFrame):
    r = df["ret"]
    mean, std = r.mean(), r.std()
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    jb_stat, jb_p = stats.jarque_bera(r)

    print("\n--- Heavy Tails: Summary ---")
    print(f"Mean daily return: {mean:.5f}")
    print(f"Std daily return:  {std:.5f}")
    print(f"Skewness:          {skew:.3f}")
    print(f"Kurtosis (normal=3): {kurt:.3f}")
    print(f"Jarque-Bera p-value: {jb_p:.3e}")

    three_sigma = 3 * std
    emp_extreme_prob = (np.abs(r) > three_sigma).mean()
    gauss_extreme_prob = 2 * (1 - stats.norm.cdf(3))
    print(f"Empirical P(|r|>3σ): {emp_extreme_prob:.3%}")
    print(f"Gaussian  P(|r|>3σ): {gauss_extreme_prob:.3%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(r, bins=80, stat="density", kde=False, ax=axes[0], color="steelblue")
    x = np.linspace(r.quantile(0.001), r.quantile(0.999), 200)
    axes[0].plot(x, stats.norm.pdf(x, loc=mean, scale=std), color="red", lw=2, label="Normal PDF")
    axes[0].set_title("Daily Log Returns vs Normal PDF")
    axes[0].legend()

    stats.probplot(r, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-Plot vs Normal")

    maybe_show_plot("01_heavy_tails_hist_qq.png")


def analyze_vol_clustering(df: pd.DataFrame):
    from statsmodels.graphics.tsaplots import plot_acf

    r = df["ret"]
    abs_r = np.abs(r)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(r.index, r, lw=0.5)
    axes[0].set_title("Daily Log Returns")
    axes[1].plot(abs_r.index, abs_r, lw=0.5, color="orange")
    axes[1].set_title("Absolute Returns (Volatility Proxy)")
    maybe_show_plot("02_returns_and_abs_returns.png")

    plt.figure(figsize=(6, 4))
    plot_acf(abs_r.dropna(), lags=50)
    plt.title("ACF of |Returns| (Volatility Clustering)")
    maybe_show_plot("03_abs_returns_acf.png")


def analyze_drawdowns(df: pd.DataFrame, top_n: int = 10):
    dd = df["drawdown"]
    in_dd = dd < 0

    episodes = []
    start = None
    trough_date = None
    trough_val = 0.0

    for date, val in dd.items():
        if in_dd.loc[date] and start is None:
            start = date
            trough_date = date
            trough_val = val
        elif in_dd.loc[date]:
            if val < trough_val:
                trough_date = date
                trough_val = val
        elif not in_dd.loc[date] and start is not None:
            recovery = date
            episodes.append((start, trough_date, recovery, trough_val))
            start = None

    if start is not None:
        episodes.append((start, trough_date, dd.index[-1], trough_val))

    episodes_sorted = sorted(episodes, key=lambda x: x[3])[:top_n]

    print("\n--- Top Drawdowns ---")
    headers = ["Rank", "Start", "Trough", "Recovery", "Depth", "Days to Trough", "Days to Recovery"]
    rows = []
    for i, (s, t, r_date, depth) in enumerate(episodes_sorted, 1):
        days_to_trough = (t - s).days
        days_to_rec = (r_date - s).days
        rows.append([
            i,
            s.date(),
            t.date(),
            r_date.date(),
            f"{depth:.2%}",
            days_to_trough,
            days_to_rec,
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))

    plt.figure(figsize=(12, 4))
    plt.plot(dd.index, dd, color="darkred")
    plt.title("Drawdown Over Time")
    plt.ylabel("Drawdown")
    maybe_show_plot("04_drawdowns.png")


# --------------------------------------------------------------------------------------
# PHASE 2: Regime detection with HMM
# --------------------------------------------------------------------------------------

def fit_hmm(df: pd.DataFrame, n_states: int = 2) -> pd.Series:
    from hmmlearn.hmm import GaussianHMM

    r = df["ret"].dropna()
    X = r.values.reshape(-1, 1)

    hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
    hmm.fit(X)
    hidden_states = hmm.predict(X)

    states = pd.Series(hidden_states, index=r.index, name="state")

    print("\n--- HMM Regime Summary ---")
    summary_rows = []
    for s in range(n_states):
        mask = states == s
        sub = r[mask]
        ann_ret = sub.mean() * 252
        ann_vol = sub.std() * np.sqrt(252)
        avg_dur = mask.groupby((mask != mask.shift()).cumsum()).sum().mean()
        summary_rows.append([s, f"{ann_ret:.2%}", f"{ann_vol:.2%}", f"{avg_dur:.1f}"])

    headers = ["State", "Ann. Mean Ret", "Ann. Vol", "Avg Duration (days)"]
    print(tabulate(summary_rows, headers=headers, tablefmt="github"))

    print("\nTransition matrix (rows: from, cols: to):")
    print(hmm.transmat_)

    low_vol_state = None
    best_mean = -1e9
    for s in range(n_states):
        mask = states == s
        sub = r[mask]
        ann_ret = sub.mean() * 252
        ann_vol = sub.std() * np.sqrt(252)
        if ann_vol > 0 and ann_ret / ann_vol > best_mean:
            best_mean = ann_ret / ann_vol
            low_vol_state = s

    print(f"\nChosen 'bull/low-vol' state: {low_vol_state}")

    fig, ax = plt.subplots(figsize=(12, 4))
    for s in range(n_states):
        ax.plot(df.index, df["price"], color="lightgray", alpha=0.4)
        mask = states == s
        ax.scatter(states.index[mask], df.loc[mask.index[mask], "price"],
                   s=4, label=f"State {s}")
    ax.set_title("Regime States over Price")
    ax.legend()
    maybe_show_plot("05_hmm_regimes_over_price.png")

    df["state"] = states.reindex(df.index).ffill()
    return states


# --------------------------------------------------------------------------------------
# PHASE 3: Volatility forecasting with GARCH
# --------------------------------------------------------------------------------------

def fit_garch(df: pd.DataFrame) -> pd.Series:
    from arch import arch_model

    r = df["ret"].dropna() * 100
    am = arch_model(r, vol="GARCH", p=1, q=1, mean="Constant", dist="normal")
    res = am.fit(disp="off")

    # Use in-sample conditional volatility so we have a full aligned series.
    sigma = (res.conditional_volatility / 100.0).reindex(df.index).ffill().bfill()
    df["garch_vol"] = sigma

    realized_30 = df["ret"].rolling(30).std()
    common = sigma.dropna().index.intersection(realized_30.dropna().index)
    rmse = np.sqrt(((sigma.loc[common] - realized_30.loc[common]) ** 2).mean())
    print(f"\n--- GARCH Forecasting ---")
    print(f"RMSE vs 30-day realized vol: {rmse:.5f}")

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, realized_30, label="Realized 30d vol")
    plt.plot(df.index, sigma, label="GARCH(1,1) forecast vol", alpha=0.8)
    plt.legend()
    plt.title("Realized vs GARCH Forecast Volatility")
    maybe_show_plot("06_garch_vs_realized_vol.png")

    return sigma


# --------------------------------------------------------------------------------------
# PHASE 4 & 5: Strategies
# --------------------------------------------------------------------------------------

def construct_momentum_strategy(df: pd.DataFrame, lookback_days: int = 252,
                                name: str = "Mom 12m") -> pd.Series:
    prices = df["price"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    mom = prices.pct_change(lookback_days)
    signal = (mom > 0).shift(1).fillna(False)
    strat_rets = signal.astype(float) * df["ret"].astype(float)
    strat_rets.name = name
    return strat_rets


def construct_regime_filtered_momentum(df: pd.DataFrame,
                                       lookback_days: int = 252,
                                       vol_quantile: float = 0.7,
                                       name: str = "Regime Mom 12m") -> pd.Series:
    prices = df["price"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    mom = prices.pct_change(lookback_days)
    mom_signal = (mom > 0)

    if "state" not in df.columns:
        raise ValueError("HMM state not found in df.")
    states = df["state"]

    state_means = df.groupby("state")["ret"].mean() * 252
    state_vols = df.groupby("state")["ret"].std() * np.sqrt(252)
    sharpe_by_state = state_means / state_vols
    bull_state = sharpe_by_state.idxmax()

    regime_filter = states == bull_state

    if "garch_vol" not in df.columns:
        raise ValueError("GARCH volatility forecast not found in df.")
    vol = df["garch_vol"]
    vol_thresh = vol.quantile(vol_quantile)
    low_vol_forecast = vol < vol_thresh

    combined_signal = (mom_signal & regime_filter & low_vol_forecast).shift(1).fillna(False)
    strat_rets = combined_signal.astype(float) * df["ret"].astype(float)
    strat_rets.name = name

    avg_exposure = combined_signal.mean()
    # In rare cases alignment can make this a Series (e.g. DataFrame boolean ops); reduce to scalar
    if isinstance(avg_exposure, pd.Series):
        avg_exposure = avg_exposure.mean()

    print(f"\nRegime-filtered momentum using:")
    print(f"- Bull/low-vol state: {bull_state}")
    print(f"- Vol forecast threshold (Q{int(vol_quantile*100)}): {vol_thresh:.4f}")
    print(f"- Average exposure: {avg_exposure:.2%}")

    return strat_rets


def segment_periods(returns: pd.Series):
    segs = {
        "2000-2008": returns.loc["2000-01-01":"2008-12-31"],
        "2009-2019": returns.loc["2009-01-01":"2019-12-31"],
        "2020-2026": returns.loc["2020-01-01":"2026-12-31"],
    }
    return segs


def run_backtests(df: pd.DataFrame):
    bh = df["ret"].copy()
    bh.name = "Buy & Hold"
    mom_12 = construct_momentum_strategy(df, 252, "Momentum 12m")
    reg_mom_12 = construct_regime_filtered_momentum(df, 252, 0.7, "Regime Mom 12m")

    all_rets = pd.concat([bh, mom_12, reg_mom_12], axis=1).dropna()

    cum = (1 + all_rets).cumprod()
    plt.figure(figsize=(12, 5))
    for c in cum.columns:
        plt.plot(cum.index, cum[c], label=c)
    plt.legend()
    plt.title("Equity Curves: Buy & Hold vs Momentum vs Regime-Filtered Momentum")
    maybe_show_plot("07_equity_curves.png")

    start_label = df.index.min().date()
    end_label = df.index.max().date()

    overall_stats = [
        performance_stats(bh, "Buy & Hold"),
        performance_stats(mom_12, "Momentum 12m"),
        performance_stats(reg_mom_12, "Regime Mom 12m"),
    ]
    print_stats_table(overall_stats, f"Overall Performance ({start_label} to {end_label})")

    for label, seg_range in [
        ("2000-2008", ("2000-01-01", "2008-12-31")),
        ("2009-2019", ("2009-01-01", "2019-12-31")),
        ("2020-2026", ("2020-01-01", "2026-12-31")),
    ]:
        s, e = seg_range
        seg_bh = bh.loc[s:e]
        seg_mom = mom_12.loc[s:e]
        seg_reg = reg_mom_12.loc[s:e]
        if len(seg_bh.dropna()) == 0 and len(seg_mom.dropna()) == 0 and len(seg_reg.dropna()) == 0:
            print(f"\nSkipping {label}: no data in current sample window.")
            continue
        seg_stats = [
            performance_stats(seg_bh, "Buy & Hold"),
            performance_stats(seg_mom, "Momentum 12m"),
            performance_stats(seg_reg, "Regime Mom 12m"),
        ]
        print_stats_table(seg_stats, f"Performance in {label}")

    return {
        "bh": bh,
        "mom_12": mom_12,
        "reg_mom_12": reg_mom_12,
    }


# --------------------------------------------------------------------------------------
# PHASE 6: Simple robustness checks
# --------------------------------------------------------------------------------------

def robustness_checks(df: pd.DataFrame):
    print("\n=== Robustness: Different momentum windows ===")
    windows = [126, 189, 252]
    for w in windows:
        mom = construct_momentum_strategy(df, w, name=f"Mom {w}d")
        reg_mom = construct_regime_filtered_momentum(df, w, 0.7, name=f"RegMom {w}d")
        stats_list = [
            performance_stats(mom, f"Mom {w}d"),
            performance_stats(reg_mom, f"RegMom {w}d"),
        ]
        print_stats_table(stats_list, f"Momentum Window = {w} days")

    print("\n=== Robustness: Different volatility thresholds (12m momentum) ===")
    for q in [0.6, 0.7, 0.8]:
        reg_mom_q = construct_regime_filtered_momentum(df, 252, q, name=f"RegMom 12m Q{int(q*100)}")
        stats_list = [performance_stats(reg_mom_q, f"RegMom 12m Q{int(q*100)}")]
        print_stats_table(stats_list, f"Vol Threshold Quantile = {q}")


# --------------------------------------------------------------------------------------
# PHASE 7: Narrative summary (print-only)
# --------------------------------------------------------------------------------------

def final_summary():
    """
    Placeholder for narrative summary of results.
    """
    pass

# --------------------------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------------------------

def main():
    df = download_data()
    df = compute_features(df)

    # Fast path: run lightweight EDA on a smaller, recent slice.
    df = limit_rows(df, MAX_ROWS)
    quick_eda(df)

    if not RUN_FULL_PIPELINE:
        print("\nSkipping full pipeline (set RUN_FULL_PIPELINE=True to run all phases).")
        return

    analyze_heavy_tails(df)
    analyze_vol_clustering(df)
    analyze_drawdowns(df)

    fit_hmm(df)
    fit_garch(df)

    run_backtests(df)

    robustness_checks(df)

    final_summary()


if __name__ == "__main__":
    main()

