# ============================================================
# Regime-Aware Momentum Strategy (NASDAQ, 2000–2026) in R
# - HMM Regime Detection (depmixS4)
# - GARCH Vol Forecasting (rugarch)
# - Momentum + Regime + Vol filter backtests
# - ALL PLOTS SAVED (not shown) to ./plots
# ============================================================

suppressWarnings({
  suppressMessages({
    library(quantmod)
    library(xts)
    library(zoo)
    library(depmixS4)
    library(rugarch)
    library(PerformanceAnalytics)
    library(tseries)
    library(ggplot2)
    library(scales)
  })
})

# ----------------------------
# Config
# ----------------------------
PLOTS_DIR <- "plots"
dir.create(PLOTS_DIR, showWarnings = FALSE)

TICKER <- "^IXIC"              # NASDAQ Composite proxy
START  <- "2000-01-01"
END    <- "2026-02-24"
TRADING_DAYS <- 252

LOOKBACK_MAIN <- 252
VOL_Q <- 0.70

# ----------------------------
# Helpers
# ----------------------------
save_plot <- function(p, filename, w=10, h=5) {
  ggsave(
    filename = file.path(PLOTS_DIR, filename),
    plot = p, width = w, height = h, units = "in", dpi = 200
  )
}

safe_stats <- function(r, name="") {
  r <- r[is.finite(r)]
  r <- r[!is.na(r)]
  if (length(r) < 2) {
    return(data.frame(
      Name=name, CAGR=NA, Sharpe=NA, Sortino=NA, MaxDD=NA, WinRate=NA
    ))
  }
  
  # CAGR on simple returns
  ann_ret <- prod(1 + r)^(TRADING_DAYS/length(r)) - 1
  ann_vol <- sd(r) * sqrt(TRADING_DAYS)
  sharpe  <- ifelse(ann_vol > 0, ann_ret/ann_vol, NA)
  
  downside <- sd(r[r < 0]) * sqrt(TRADING_DAYS)
  sortino  <- ifelse(is.finite(downside) && downside > 0, ann_ret/downside, NA)
  
  cum <- cumprod(1 + r)
  dd  <- cum / cummax(cum) - 1
  maxdd <- min(dd, na.rm = TRUE)
  
  winrate <- mean(r > 0)
  
  data.frame(
    Name=name, CAGR=ann_ret, Sharpe=sharpe, Sortino=sortino, MaxDD=maxdd, WinRate=winrate
  )
}

print_table <- function(df, title) {
  cat("\n", paste(rep("=", 80), collapse=""), "\n", sep="")
  cat(title, "\n")
  cat(paste(rep("=", 80), collapse=""), "\n", sep="")
  
  fmt_pct <- function(x) ifelse(is.na(x), "NA", sprintf("%.2f%%", 100*x))
  fmt_num <- function(x) ifelse(is.na(x), "NA", sprintf("%.2f", x))
  
  out <- df
  out$CAGR   <- fmt_pct(out$CAGR)
  out$MaxDD  <- fmt_pct(out$MaxDD)
  out$WinRate<- fmt_pct(out$WinRate)
  out$Sharpe <- fmt_num(out$Sharpe)
  out$Sortino<- fmt_num(out$Sortino)
  
  print(out, row.names = FALSE)
}

align3 <- function(a, b, c) {
  m <- merge(a, b, c, join = "inner")
  colnames(m) <- c("A","B","C")
  m
}

# ----------------------------
# 1) Load data
# ----------------------------
cat(sprintf("Downloading %s from %s to %s ...\n", TICKER, START, END))
px <- getSymbols(TICKER,
                 src = "yahoo",
                 from = START,
                 to = END,
                 auto.assign = FALSE,
                 warnings = FALSE)

px <- Ad(px)
colnames(px) <- "price"
px <- px[!is.na(px$price)]

# Returns:
# log_ret for HMM/GARCH; simple ret for performance/backtests
log_ret <- diff(log(px$price))
simp_ret <- exp(log_ret) - 1

# realized volatility (log returns)
realized_vol_30  <- rollapply(log_ret, 30, sd, align="right", fill=NA)
realized_vol_252 <- rollapply(log_ret, 252, sd, align="right", fill=NA)

# equity curve + drawdown (simple returns)
cum <- cumprod(1 + simp_ret)
dd  <- cum / cummax(cum) - 1

# Momentum signals (on price)
mom_126 <- ROC(px$price, n = 126, type = "discrete")
mom_189 <- ROC(px$price, n = 189, type = "discrete")
mom_252 <- ROC(px$price, n = 252, type = "discrete")

# Ensure everything is xts and properly named BEFORE merge
# Just rename directly — no xts() wrapping

colnames(px) <- "price"
colnames(log_ret) <- "log_ret"
colnames(simp_ret) <- "ret"
colnames(realized_vol_30) <- "rv30"
colnames(realized_vol_252) <- "rv252"
colnames(cum) <- "cum"
colnames(dd) <- "dd"
colnames(mom_126) <- "mom126"
colnames(mom_189) <- "mom189"
colnames(mom_252) <- "mom252"

df <- merge(px, log_ret, simp_ret,
            realized_vol_30, realized_vol_252,
            cum, dd,
            mom_126, mom_189, mom_252)

df <- na.omit(df)
# ----------------------------
# 2) EDA (saved plots)
# ----------------------------
class(df)
colnames(df)

# Heavy tails summary
r <- as.numeric(df$log_ret)
jb <- jarque.bera.test(r)

cat("\n--- Heavy Tails: Summary ---\n")
cat(sprintf("Mean daily log return: %.5f\n", mean(r)))
cat(sprintf("Std daily log return:  %.5f\n", sd(r)))
cat(sprintf("Skewness:              %.3f\n", PerformanceAnalytics::skewness(r)))
cat(sprintf("Kurtosis (normal=3):   %.3f\n", PerformanceAnalytics::kurtosis(r) + 3)) # PE uses excess
cat(sprintf("Jarque-Bera p-value:   %.3e\n", jb$p.value))

three_sigma <- 3 * sd(r)
emp_extreme <- mean(abs(r) > three_sigma)
gauss_extreme <- 2 * (1 - pnorm(3))
cat(sprintf("Empirical P(|r|>3σ):   %.3f%%\n", 100*emp_extreme))
cat(sprintf("Gaussian  P(|r|>3σ):   %.3f%%\n", 100*gauss_extreme))

# Histogram + normal overlay
d1 <- data.frame(date=index(df), log_ret=as.numeric(df$log_ret))
p1 <- ggplot(d1, aes(x=log_ret)) +
  geom_histogram(aes(y=after_stat(density)), bins=80, alpha=0.7) +
  stat_function(fun=dnorm, args=list(mean=mean(r), sd=sd(r)), linewidth=1) +
  labs(title="Daily Log Returns vs Normal PDF", x="log return", y="density")
save_plot(p1, "eda_heavy_tails_hist.png", 10, 5)

# QQ plot
p2 <- ggplot(d1, aes(sample=log_ret)) +
  stat_qq() + stat_qq_line() +
  labs(title="QQ Plot: Log Returns vs Normal")
save_plot(p2, "eda_heavy_tails_qq.png", 7, 5)

# Returns + abs returns
d2 <- data.frame(date=index(df), log_ret=as.numeric(df$log_ret), abs_ret=abs(as.numeric(df$log_ret)))
p3 <- ggplot(d2, aes(x=date, y=log_ret)) +
  geom_line(linewidth=0.3) +
  labs(title="Daily Log Returns", x="", y="log return")
save_plot(p3, "eda_log_returns.png", 12, 4)

p4 <- ggplot(d2, aes(x=date, y=abs_ret)) +
  geom_line(linewidth=0.3) +
  labs(title="Absolute Log Returns (Volatility Proxy)", x="", y="|log return|")
save_plot(p4, "eda_abs_returns.png", 12, 4)

# Drawdown plot
d3 <- data.frame(date=index(df), dd=as.numeric(df$dd))
p5 <- ggplot(d3, aes(x=date, y=dd)) +
  geom_line(linewidth=0.7) +
  scale_y_continuous(labels=percent) +
  labs(title="Drawdown Over Time", x="", y="Drawdown")
save_plot(p5, "eda_drawdown.png", 12, 4)

# ----------------------------
# 3) HMM Regime Detection (2-state Gaussian)
# ----------------------------
# depmix wants a data.frame; use log returns only (common, simple)
hmm_data <- data.frame(r = as.numeric(df$log_ret))

set.seed(42)
mod <- depmix(r ~ 1, family = gaussian(), nstates = 2, data = hmm_data)
fit <- fit(mod, verbose = FALSE)

post <- posterior(fit)              # contains state posterior + most likely state
state <- post$state                 # 1..K
state_xts <- xts(state, order.by=index(df))
colnames(state_xts) <- "state"
df <- merge(df, state_xts, join="inner")

# Regime summary
summary_rows <- data.frame()
for (s in 1:2) {
  mask <- df$state == s
  sub <- df$log_ret[mask]
  ann_mean <- mean(sub) * TRADING_DAYS
  ann_vol  <- sd(sub) * sqrt(TRADING_DAYS)
  
  # avg run length
  runs <- rle(as.integer(mask))
  lens <- runs$lengths[runs$values == 1]
  avg_dur <- ifelse(length(lens) > 0, mean(lens), NA)
  
  summary_rows <- rbind(summary_rows, data.frame(
    State=s-1, AnnMeanRet=ann_mean, AnnVol=ann_vol, AvgDuration=avg_dur
  ))
}

# Choose bull state by Sharpe-like (mean/vol)
summary_rows$SharpeLike <- with(summary_rows, AnnMeanRet / AnnVol)
bull_state_raw <- summary_rows$State[which.max(summary_rows$SharpeLike)]
bull_state <- bull_state_raw + 1   # convert 0/1 to 1/2 for df$state

cat("\n--- HMM Regime Summary ---\n")
print(data.frame(
  State = summary_rows$State,
  AnnMeanRet = sprintf("%.2f%%", 100*summary_rows$AnnMeanRet),
  AnnVol     = sprintf("%.2f%%", 100*summary_rows$AnnVol),
  AvgDurationDays = sprintf("%.1f", summary_rows$AvgDuration)
), row.names = FALSE)
cat(sprintf("\nChosen 'bull/low-vol' state: %d\n", bull_state_raw))

# Plot regimes over price
dreg <- data.frame(date=index(df), price=as.numeric(df$price), state=as.factor(as.numeric(df$state)))
p6 <- ggplot(dreg, aes(x=date, y=price)) +
  geom_line(alpha=0.45) +
  geom_point(aes(color=state), size=0.6, alpha=0.8) +
  labs(title="HMM Regimes Over Price", x="", y="Price", color="State")
save_plot(p6, "hmm_regimes_over_price.png", 12, 4)

# ----------------------------
# 4) GARCH Vol Forecasting (GARCH(1,1))
# ----------------------------
# rugarch likes returns in decimal; we use log returns
garch_r <- as.numeric(df$log_ret)

spec <- ugarchspec(
  variance.model = list(model="sGARCH", garchOrder=c(1,1)),
  mean.model = list(armaOrder=c(0,0), include.mean=TRUE),
  distribution.model = "norm"
)

fit_g <- ugarchfit(spec = spec, data = garch_r, solver = "hybrid", solver.control=list(trace=0))

# One-step-ahead rolling forecast of sigma:
# We'll get fitted conditional sigma for each t (in-sample), which is a common proxy for "forecast"
sigma <- sigma(fit_g)   # conditional sd aligned to input length
sigma_xts <- xts(as.numeric(sigma), order.by=index(df))
colnames(sigma_xts) <- "garch_vol"

df <- merge(df, sigma_xts, join="inner")

# RMSE vs 30-day realized vol (log returns)
rv30 <- df$rv30
common_idx <- index(na.omit(merge(df$garch_vol, rv30)))
rmse <- sqrt(mean((as.numeric(df$garch_vol[common_idx]) - as.numeric(rv30[common_idx]))^2))

cat("\n--- GARCH Forecasting ---\n")
cat(sprintf("RMSE vs 30-day realized vol: %.5f\n", rmse))

dg <- data.frame(
  date=index(df),
  realized=as.numeric(df$rv30),
  garch=as.numeric(df$garch_vol)
)
p7 <- ggplot(dg, aes(x=date)) +
  geom_line(aes(y=realized, color="Realized 30d vol"), linewidth=0.7) +
  geom_line(aes(y=garch, color="GARCH sigma"), linewidth=0.7, alpha=0.85) +
  scale_color_discrete(name="") +
  labs(title="Realized vs GARCH Conditional Volatility", x="", y="Volatility")
save_plot(p7, "garch_vs_realized.png", 12, 4)

# ----------------------------
# 5) Strategies (FIXED alignment; no NA-killing; nonzero exposure)
# ----------------------------

momentum_strategy <- function(df, lookback=252, name="Momentum") {
  mom <- ROC(df$price, n = lookback, type="discrete")
  sig <- (mom > 0)
  sig <- lag(sig, 1)            # avoid lookahead
  sig[is.na(sig)] <- FALSE
  rets <- sig * df$ret
  colnames(rets) <- name
  rets
}

regime_filtered_momentum <- function(df, bull_state, lookback=252, vol_q=0.70, name="Regime Mom") {
  mom <- ROC(df$price, n = lookback, type="discrete")
  mom_sig <- (mom > 0)
  
  regime_ok <- (df$state == bull_state)
  
  # IMPORTANT: compute threshold on NON-NA vol only
  v <- df$garch_vol
  thresh <- as.numeric(quantile(na.omit(v), probs = vol_q))
  low_vol <- (v <= thresh)
  
  combined <- mom_sig & regime_ok & low_vol
  combined <- lag(combined, 1)
  combined[is.na(combined)] <- FALSE
  
  avg_expo <- mean(as.numeric(combined))
  cat("\nRegime-filtered momentum using:\n")
  cat(sprintf("- Bull/low-vol state: %d\n", bull_state - 1))
  cat(sprintf("- Vol forecast threshold (Q%d): %.4f\n", floor(vol_q*100), thresh))
  cat(sprintf("- Average exposure: %.2f%%\n", 100*avg_expo))
  
  rets <- combined * df$ret
  colnames(rets) <- name
  rets
}

bh <- df$ret; colnames(bh) <- "Buy & Hold"
mom12 <- momentum_strategy(df, LOOKBACK_MAIN, "Momentum 12m")
regmom12 <- regime_filtered_momentum(df, bull_state, LOOKBACK_MAIN, VOL_Q, "Regime Mom 12m")

# Align returns for fair stats + curves
R <- na.omit(merge(bh, mom12, regmom12))

# Equity curve plot
eq <- cumprod(1 + R)
deq <- data.frame(date=index(eq),
                  BuyHold=as.numeric(eq[,1]),
                  Mom=as.numeric(eq[,2]),
                  RegMom=as.numeric(eq[,3]))
deq_long <- reshape(deq, varying=2:4, v.names="value", timevar="Strategy",
                    times=c("Buy & Hold","Momentum 12m","Regime Mom 12m"),
                    direction="long")
p8 <- ggplot(deq_long, aes(x=date, y=value, color=Strategy)) +
  geom_line(linewidth=0.8) +
  labs(title="Equity Curves", x="", y="Growth of $1")
save_plot(p8, "equity_curves_main.png", 12, 5)

# Overall stats
overall <- rbind(
  safe_stats(as.numeric(R[,"Buy & Hold"]), "Buy & Hold"),
  safe_stats(as.numeric(R[,"Momentum 12m"]), "Momentum 12m"),
  safe_stats(as.numeric(R[,"Regime Mom 12m"]), "Regime Mom 12m")
)
print_table(overall, "Overall Performance (2000–2026)")

# Segment stats
segments <- list(
  "2000-2008" = c("2000-01-01","2008-12-31"),
  "2009-2019" = c("2009-01-01","2019-12-31"),
  "2020-2026" = c("2020-01-01","2026-12-31")
)

for (nm in names(segments)) {
  s <- segments[[nm]][1]; e <- segments[[nm]][2]
  Rs <- R[paste0(s,"/",e)]
  segtab <- rbind(
    safe_stats(as.numeric(Rs[,"Buy & Hold"]), "Buy & Hold"),
    safe_stats(as.numeric(Rs[,"Momentum 12m"]), "Momentum 12m"),
    safe_stats(as.numeric(Rs[,"Regime Mom 12m"]), "Regime Mom 12m")
  )
  print_table(segtab, paste0("Performance in ", nm))
}

# ----------------------------
# 6) Robustness checks
# ----------------------------
cat("\n=== Robustness: Different momentum windows ===\n")
for (w in c(126, 189, 252)) {
  m <- momentum_strategy(df, w, paste0("Mom ", w, "d"))
  rm <- regime_filtered_momentum(df, bull_state, w, VOL_Q, paste0("RegMom ", w, "d"))
  A <- na.omit(merge(m, rm))
  tab <- rbind(
    safe_stats(as.numeric(A[,1]), colnames(A)[1]),
    safe_stats(as.numeric(A[,2]), colnames(A)[2])
  )
  print_table(tab, paste0("Momentum Window = ", w, " days"))
}

cat("\n=== Robustness: Different volatility thresholds (12m momentum) ===\n")
for (q in c(0.60, 0.70, 0.80)) {
  rmq <- regime_filtered_momentum(df, bull_state, 252, q, paste0("RegMom 12m Q", floor(q*100)))
  A <- na.omit(merge(rmq))
  tab <- safe_stats(as.numeric(A[,1]), colnames(A)[1])
  print_table(tab, paste0("Vol Threshold Quantile = ", q))
}

cat(sprintf("\nDone. Plots saved to: ./%s/\n", PLOTS_DIR))

# ============================================================
# If packages are missing, install once:
#install.packages(c("quantmod","xts","zoo","depmixS4","rugarch",
                    "PerformanceAnalytics","tseries","ggplot2","scales"))
# ============================================================