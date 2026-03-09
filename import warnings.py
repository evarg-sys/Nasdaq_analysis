import warnings
warnings.filterwarnings("ignore")

"""
End-to-end project (NASDAQ-100 / NASDAQ Composite via yfinance):
- PHASE 1: Data loading & EDA (stylized facts)
- PHASE 2: Regime detection with HMM (on log returns)
- PHASE 3: Volatility forecasting with GARCH (on log returns)
- PHASE 4: Baseline momentum strategy (on simple returns)
- PHASE 5: Regime-aware momentum strategy (fixed alignment + nonzero exposure)
- PHASE 6: Robustness checks
- PHASE 7: Summary tables

IMPORTANT:
- This version SAVES plots (does NOT show them).
- Plots saved under ./plots/

Required packages:
    pip install yfinance pandas numpy matplotlib scipy hmmlearn arch tabulate statsmodels
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # save-only backend
import matplotlib.pyplot as plt

from scipy import stats
from hmmlearn.hmm import GaussianHMM
from arch import arch_model
from tabulate import tabulate


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

TICKER = "^IXIC"         # NASDAQ Composite (common proxy if you don't have NDX)
START = "2000-01-01"
END = "2026-02-24"

TRADING_DAYS = 252


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------------------
# Data + features
# --------------------------------------------------------------------------------------

def download_data(ticker: str = TICKER,
                  start: str = START,
                  end: str = END) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)

    # Prefer Adjusted Close if available, else fall back to Close
    if "Adj Close" in data.columns:
        price_col = "Adj Close"
    elif "Close" in data.columns:
        price_col = "Close"
    else:
        raise ValueError(f"Downloaded data missing 'Adj Close'/'Close'. Columns: {list(data.columns)}")

    df = data[[price_col]].rename(columns={price_col: "price"}).dropna()
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Key fix:
    - Keep BOTH log returns and simple returns.
    - Use SIMPLE returns for backtesting + CAGR/Sharpe math.
    - Use LOG returns for HMM/GARCH + stylized facts plots.
    """
    price = df["price"]

    df["log_ret"] = np.log(price / price.shift(1))
    df["ret"] = np.exp(df["log_ret"]) - 1  # simple return

    # Realized vol based on log returns (common in quant)
    df["realized_vol_30"] = df["log_ret"].rolling(30).std()
    df["realized_vol_252"] = df["log_ret"].rolling(252).std()

    # Equity curve + drawdowns should use simple returns
    df["cum_ret"] = (1 + df["ret"]).cumprod()
    df["drawdown"] = df["cum_ret"] / df["cum_ret"].cummax() - 1

    df["ma_50"] = price.rolling(50).mean()
    df["ma_200"] = price.rolling(200).mean()

    # Momentum signals (on price)
    df["mom_6m"] = price.pct_change(126)
    df["mom_9m"] = price.pct_change(189)
    df["mom_12m"] = price.pct_change(252)

    # Drop rows with missing core fields (keeps alignment clean)
    return df.dropna(subset=["price", "log_ret", "ret", "cum_ret", "drawdown"])


# --------------------------------------------------------------------------------------
# Performance metrics (on SIMPLE returns)
# --------------------------------------------------------------------------------------

def performance_stats(returns: pd.Series, name: str = "") -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {"Name": name, "CAGR": np.nan, "Sharpe": np.nan,
                "Sortino": np.nan, "MaxDD": np.nan, "WinRate": np.nan}

    # CAGR on simple returns
    ann_ret = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1

    # Annualized vol on daily simple returns (fine in practice for reporting)
    ann_vol = r.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0].std(ddof=0) * np.sqrt(TRADING_DAYS)
    sortino = ann_ret / downside if downside > 0 else np.nan

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
        table.append([
            s["Name"],
            f"{s['CAGR']:.2%}" if pd.notna(s["CAGR"]) else "NA",
            f"{s['Sharpe']:.2f}" if pd.notna(s["Sharpe"]) else "NA",
            f"{s['Sortino']:.2f}" if pd.notna(s["Sortino"]) else "NA",
            f"{s['MaxDD']:.2%}" if pd.notna(s["MaxDD"]) else "NA",
            f"{s['WinRate']:.2%}" if pd.notna(s["WinRate"]) else "NA",
        ])
    print(tabulate(table, headers=headers, tablefmt="github"))


# --------------------------------------------------------------------------------------
# PHASE 1: Stylized facts (on LOG returns)
# --------------------------------------------------------------------------------------

def analyze_heavy_tails(df: pd.DataFrame):
    r = df["log_ret"].dropna()

    mean, std = r.mean(), r.std(ddof=0)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)  # normal=3
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

    # Plot: histogram + normal PDF + QQ plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(r, bins=80, density=True, alpha=0.7)
    x = np.linspace(r.quantile(0.001), r.quantile(0.999), 200)
    axes[0].plot(x, stats.norm.pdf(x, loc=mean, scale=std), lw=2, label="Normal PDF")
    axes[0].set_title("Daily Log Returns vs Normal PDF")
    axes[0].set_xlabel("log return")
    axes[0].set_ylabel("density")
    axes[0].legend()

    stats.probplot(r, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-Plot vs Normal")

    savefig(os.path.join(PLOTS_DIR, "eda_heavy_tails_hist_qq.png"))


def analyze_vol_clustering(df: pd.DataFrame):
    from statsmodels.graphics.tsaplots import plot_acf

    r = df["log_ret"].dropna()
    abs_r = np.abs(r)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(r.index, r, lw=0.5)
    axes[0].set_title("Daily Log Returns")
    axes[1].plot(abs_r.index, abs_r, lw=0.5)
    axes[1].set_title("Absolute Returns (Volatility Proxy)")
    savefig(os.path.join(PLOTS_DIR, "eda_returns_abs_returns.png"))

    plt.figure(figsize=(7, 4))
    plot_acf(abs_r, lags=50)
    plt.title("ACF of |Returns| (Volatility Clustering)")
    savefig(os.path.join(PLOTS_DIR, "eda_acf_abs_returns.png"))


def analyze_drawdowns(df: pd.DataFrame, top_n: int = 10):
    dd = df["drawdown"].dropna()
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
        elif (not in_dd.loc[date]) and start is not None:
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
        rows.append([
            i,
            s.date(),
            t.date(),
            r_date.date(),
            f"{depth:.2%}",
            (t - s).days,
            (r_date - s).days,
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))

    plt.figure(figsize=(12, 4))
    plt.plot(dd.index, dd, lw=1.2)
    plt.title("Drawdown Over Time")
    plt.ylabel("Drawdown")
    savefig(os.path.join(PLOTS_DIR, "eda_drawdown_over_time.png"))


# --------------------------------------------------------------------------------------
# PHASE 2: Regime detection with HMM (on LOG returns)
# --------------------------------------------------------------------------------------

def fit_hmm(df: pd.DataFrame, n_states: int = 2) -> int:
    r = df["log_ret"].dropna()
    X = r.values.reshape(-1, 1)

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=42
    )
    hmm.fit(X)
    hidden_states = hmm.predict(X)

    states = pd.Series(hidden_states, index=r.index, name="state")
    df["state"] = states.reindex(df.index)  # NO forward fill

    print("\n--- HMM Regime Summary ---")
    summary_rows = []
    for s in range(n_states):
        mask = states == s
        sub = r[mask]
        ann_ret = sub.mean() * TRADING_DAYS
        ann_vol = sub.std(ddof=0) * np.sqrt(TRADING_DAYS)

        # average duration of consecutive runs in this state
        runs = mask.groupby((mask != mask.shift()).cumsum()).sum()
        avg_dur = runs[runs > 0].mean() if (runs > 0).any() else np.nan

        summary_rows.append([s, f"{ann_ret:.2%}", f"{ann_vol:.2%}", f"{avg_dur:.1f}"])

    print(tabulate(summary_rows,
                   headers=["State", "Ann. Mean Ret", "Ann. Vol", "Avg Duration (days)"],
                   tablefmt="github"))

    print("\nTransition matrix (rows: from, cols: to):")
    print(hmm.transmat_)

    # Choose bull state by highest Sharpe-like ratio (mean/vol)
    sharpe_like = {}
    for s in range(n_states):
        sub = r[states == s]
        m = sub.mean() * TRADING_DAYS
        v = sub.std(ddof=0) * np.sqrt(TRADING_DAYS)
        sharpe_like[s] = (m / v) if v > 0 else -np.inf
    bull_state = max(sharpe_like, key=sharpe_like.get)

    print(f"\nChosen 'bull/low-vol' state: {bull_state}")

    # Plot states over price (saved)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["price"], alpha=0.5, lw=1.0, label="Price")

    for s in range(n_states):
        idx = states.index[states == s]
        ax.scatter(idx, df.loc[idx, "price"], s=6, label=f"State {s}")

    ax.set_title("HMM Regime States over Price")
    ax.legend()
    savefig(os.path.join(PLOTS_DIR, "hmm_regimes_over_price.png"))

    return int(bull_state)


# --------------------------------------------------------------------------------------
# PHASE 3: Volatility forecasting with GARCH (on LOG returns)
# --------------------------------------------------------------------------------------

def fit_garch(df: pd.DataFrame) -> float:
    # Use log returns in percent for arch package
    r = (df["log_ret"].dropna() * 100)

    am = arch_model(r, vol="GARCH", p=1, q=1, mean="Constant", dist="normal")
    res = am.fit(disp="off")

    fcasts = res.forecast(horizon=1, reindex=False)
    sigma = np.sqrt(fcasts.variance.iloc[:, 0]) / 100.0  # back to raw units
    sigma = sigma.reindex(df.index)  # NO forward fill
    df["garch_vol"] = sigma

    realized_30 = df["log_ret"].rolling(30).std()

    common = sigma.dropna().index.intersection(realized_30.dropna().index)
    rmse = np.sqrt(((sigma.loc[common] - realized_30.loc[common]) ** 2).mean())

    print(f"\n--- GARCH Forecasting ---")
    print(f"RMSE vs 30-day realized vol: {rmse:.5f}")

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, realized_30, label="Realized 30d vol")
    plt.plot(df.index, sigma, label="GARCH(1,1) forecast vol", alpha=0.9)
    plt.title("Realized vs GARCH Forecast Volatility")
    plt.legend()
    savefig(os.path.join(PLOTS_DIR, "garch_vs_realized_vol.png"))

    return float(rmse)


# --------------------------------------------------------------------------------------
# PHASE 4 & 5: Strategies (on SIMPLE returns)
# --------------------------------------------------------------------------------------

def construct_momentum_strategy(df: pd.DataFrame,
                                lookback_days: int = 252,
                                name: str = "Momentum") -> pd.Series:
    price = df["price"]
    mom = price.pct_change(lookback_days)
    signal = (mom > 0)

    # shift(1) to avoid lookahead, fill NA as False so we don't lose everything
    signal = signal.shift(1).fillna(False)

    strat = signal.astype(float) * df["ret"]
    strat.name = name
    return strat


def construct_regime_filtered_momentum(df: pd.DataFrame,
                                       bull_state: int,
                                       lookback_days: int = 252,
                                       vol_quantile: float = 0.7,
                                       name: str = "Regime Mom") -> pd.Series:
    if "state" not in df.columns:
        raise ValueError("HMM state not found in df.")
    if "garch_vol" not in df.columns:
        raise ValueError("GARCH volatility forecast not found in df.")

    price = df["price"]
    mom = price.pct_change(lookback_days)
    mom_signal = (mom > 0)

    regime_filter = (df["state"] == bull_state)

    vol = df["garch_vol"]
    vol_thresh = vol.dropna().quantile(vol_quantile)
    low_vol_forecast = (vol <= vol_thresh)

    # Combine (no NA propagation)
    combined = (mom_signal & regime_filter & low_vol_forecast)
    combined = combined.shift(1).fillna(False)

    strat = combined.astype(float) * df["ret"]
    strat.name = name

    avg_exposure = float(combined.mean())
    print(f"\nRegime-filtered momentum using:")
    print(f"- Bull/low-vol state: {bull_state}")
    print(f"- Vol forecast threshold (Q{int(vol_quantile*100)}): {vol_thresh:.4f}")
    print(f"- Average exposure: {avg_exposure:.2%}")

    return strat


def run_backtests(df: pd.DataFrame, bull_state: int):
    bh = df["ret"].copy()
    bh.name = "Buy & Hold"

    mom_12 = construct_momentum_strategy(df, 252, "Momentum 12m")
    reg_mom_12 = construct_regime_filtered_momentum(df, bull_state, 252, 0.7, "Regime Mom 12m")

    # IMPORTANT: Align everything for stats + equity curves
    aligned = pd.concat([bh, mom_12, reg_mom_12], axis=1).dropna()

    # Equity curves saved
    cum = (1 + aligned).cumprod()
    plt.figure(figsize=(12, 5))
    for c in cum.columns:
        plt.plot(cum.index, cum[c], label=c)
    plt.legend()
    plt.title("Equity Curves: Buy & Hold vs Momentum vs Regime-Filtered Momentum")
    savefig(os.path.join(PLOTS_DIR, "equity_curves_main.png"))

    overall_stats = [
        performance_stats(aligned["Buy & Hold"], "Buy & Hold"),
        performance_stats(aligned["Momentum 12m"], "Momentum 12m"),
        performance_stats(aligned["Regime Mom 12m"], "Regime Mom 12m"),
    ]
    print_stats_table(overall_stats, "Overall Performance (2000–2026)")

    for label, (s, e) in [
        ("2000-2008", ("2000-01-01", "2008-12-31")),
        ("2009-2019", ("2009-01-01", "2019-12-31")),
        ("2020-2026", ("2020-01-01", "2026-12-31")),
    ]:
        seg = aligned.loc[s:e]
        seg_stats = [
            performance_stats(seg["Buy & Hold"], "Buy & Hold"),
            performance_stats(seg["Momentum 12m"], "Momentum 12m"),
            performance_stats(seg["Regime Mom 12m"], "Regime Mom 12m"),
        ]
        print_stats_table(seg_stats, f"Performance in {label}")

    return aligned


# --------------------------------------------------------------------------------------
# PHASE 6: Robustness
# --------------------------------------------------------------------------------------

def robustness_checks(df: pd.DataFrame, bull_state: int):
    print("\n=== Robustness: Different momentum windows ===")
    windows = [126, 189, 252]
    for w in windows:
        mom = construct_momentum_strategy(df, w, name=f"Mom {w}d")
        reg_mom = construct_regime_filtered_momentum(df, bull_state, w, 0.7, name=f"RegMom {w}d")
        aligned = pd.concat([mom, reg_mom], axis=1).dropna()

        stats_list = [
            performance_stats(aligned[f"Mom {w}d"], f"Mom {w}d"),
            performance_stats(aligned[f"RegMom {w}d"], f"RegMom {w}d"),
        ]
        print_stats_table(stats_list, f"Momentum Window = {w} days")

    print("\n=== Robustness: Different volatility thresholds (12m momentum) ===")
    for q in [0.6, 0.7, 0.8]:
        reg_mom_q = construct_regime_filtered_momentum(df, bull_state, 252, q, name=f"RegMom 12m Q{int(q*100)}")
        aligned = pd.concat([reg_mom_q], axis=1).dropna()
        stats_list = [performance_stats(aligned.iloc[:, 0], f"RegMom 12m Q{int(q*100)}")]
        print_stats_table(stats_list, f"Vol Threshold Quantile = {q}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    print(f"Downloading {TICKER} from {START} to {END} ...")
    df = download_data()
    df = compute_features(df)

    # EDA (saved)
    analyze_heavy_tails(df)
    analyze_vol_clustering(df)
    analyze_drawdowns(df)

    # Models (saved plots)
    bull_state = fit_hmm(df, n_states=2)
    fit_garch(df)

    # Backtests (saved equity curve)
    run_backtests(df, bull_state)

    # Robustness
    robustness_checks(df, bull_state)

    print(f"\nDone. Plots saved to: ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()