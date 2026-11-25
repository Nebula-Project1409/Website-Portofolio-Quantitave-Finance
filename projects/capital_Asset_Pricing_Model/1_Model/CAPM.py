"""
Kalau aku berinvestasi pada suatu saham,
berapa tingkat keuntungan yang seharusnya aku harapkan
berdasarkan risiko saham tersebut?

CAPM:  E[R_i] = Rf + Beta_i * (E[R_m] - Rf)
Beta diestimasi dengan OLS:  (R_i - Rf) = Alpha + Beta * (R_m - Rf) + error
"""

from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# ============================================================
# 1. CONFIGURATION
# ============================================================

START_DATE = "2013-01-01"
END_DATE = "2026-01-01"  # End date is exclusive

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Semua grafik dibuat otomatis dari daftar ini.
# Menambah atau mengganti perusahaan cukup dilakukan di sini.
TICKERS = {
    "BCA": {"stock": "BBCA.JK", "market": "^JKSE", "risk_free_annual": 0.05},
    "MANDIRI": {"stock": "BMRI.JK", "market": "^JKSE", "risk_free_annual": 0.05},
    "INDOFOOD": {"stock": "INDF.JK", "market": "^JKSE", "risk_free_annual": 0.05},
    "ASTRA": {"stock": "ASII.JK", "market": "^JKSE", "risk_free_annual": 0.05},
}

# File imbal hasil obligasi (risk-free), disimpan di folder data/ di samping CAPM.py
MARKET_PREMIUM_ANNUAL = None
RF_FILE = Path(__file__).resolve().parent / "data" / "Data Historis Imbal Hasil Obligasi Indonesia 3 Bulan (1).csv"

sns.set_theme(style="whitegrid")

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.family"] = "DejaVu Sans"


# ============================================================
# 2. DOWNLOAD DATA FROM YAHOO FINANCE
# ============================================================


def download_data():

    all_tickers = []

    for company in TICKERS.values():
        all_tickers.append(company["stock"])
        all_tickers.append(company["market"])

    all_tickers = list(set(all_tickers))

    print("\nDownloading data from Yahoo Finance...")

    raw = yf.download(
        all_tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise ValueError("Yahoo Finance did not return any data.")

    prices = pd.DataFrame()

    for ticker in all_tickers:

        try:
            if isinstance(raw.columns, pd.MultiIndex):

                if ticker in raw.columns.get_level_values(0):
                    series = raw[ticker]["Close"]

                elif ticker in raw.columns.get_level_values(1):
                    series = raw.xs(ticker, axis=1, level=1)["Close"]

                else:
                    print(f"Ticker not found: {ticker}")
                    continue

            else:
                series = raw["Close"]

            prices[ticker] = series

        except (KeyError, TypeError):
            print(f"Could not extract data for {ticker}")

    prices = prices.dropna(how="all")

    print("\nAvailable data:")
    print(prices.notna().sum())

    return prices


# ============================================================
# 3. CALCULATE MONTHLY RETURNS
# ============================================================


def calculate_returns(prices):

    # Take the last available trading price each month
    monthly_prices = prices.resample("ME").last()

    # Calculate percentage changes
    monthly_returns = monthly_prices.pct_change()

    monthly_returns = monthly_returns.replace([np.inf, -np.inf], np.nan)

    return monthly_prices, monthly_returns

# ============================================================
# 3b. LOAD RISK-FREE RATE (BOND YIELD)
# ============================================================


def load_risk_free():

    rf = pd.read_csv(RF_FILE, encoding="utf-8-sig", dtype=str)

    # Format Indonesia: tanggal DD/MM/YYYY, desimal memakai koma ("6,346" = 6,346%)
    rf["Date"] = pd.to_datetime(rf["Tanggal"], format="%d/%m/%Y")
    rf["Yield"] = rf["Terakhir"].str.replace(",", ".", regex=False).astype(float) / 100

    # Tanggal di file = tanggal 1 tiap bulan; ubah ke akhir bulan
    # supaya cocok dengan resample("ME") pada return saham
    rf["Date"] = rf["Date"] + pd.offsets.MonthEnd(0)

    return rf.sort_values("Date").set_index("Date")["Yield"]


def check_risk_free(risk_free, monthly_returns):

    months = monthly_returns.dropna(how="all").index
    missing = months.difference(risk_free.index)

    print("\nRisk-free data (bond yield, annual):")
    print(f"  File period  : {risk_free.index.min().date()} to {risk_free.index.max().date()}")
    print(f"  Analysis     : {months.min().date()} to {months.max().date()} ({len(months)} months)")
    print(f"  Yield range  : {risk_free.min():.2%} to {risk_free.max():.2%}")
    print(f"  Average      : {risk_free.reindex(months).mean():.2%}")
    print(f"  Last month   : {risk_free.reindex(months).iloc[-1]:.2%}")

    if len(missing) > 0:
        raise ValueError(
            f"Risk-free data is missing for {len(missing)} analysis months, "
            f"for example: {[d.date() for d in missing[:3]]}"
        )


# ============================================================
# 4. CAPM ESTIMATION USING OLS
# ============================================================


def estimate_capm(monthly_returns, risk_free, company_name, config):

    stock = config["stock"]
    market = config["market"]

    if stock not in monthly_returns.columns:
        print(f"\nMissing stock data: {stock}")
        return None

    if market not in monthly_returns.columns:
        print(f"\nMissing market data: {market}")
        return None

    # Align stock and market returns
    # Risk-free bulanan (yield tahunan -> bulanan). Pakai yield akhir bulan SEBELUMNYA,
    # karena itulah suku bunga yang diketahui investor di awal bulan return tersebut.
    risk_free_monthly = ((1 + risk_free.shift(1, freq="ME")) ** (1 / 12) - 1).rename("RF")

    # Align stock, market, and risk-free data
    data = pd.concat(
        [
            monthly_returns[stock].rename("Stock"),
            monthly_returns[market].rename("Market"),
            risk_free_monthly,
        ],
        axis=1,
        sort=True,
    ).dropna()

    # Rf untuk expected return = yield terbaru di akhir periode data
    risk_free_annual = risk_free.reindex(data.index).iloc[-1]

    # Calculate excess returns (Rf berubah tiap bulan)
    data["Stock_Excess"] = data["Stock"] - data["RF"]

    data["Market_Excess"] = data["Market"] - data["RF"]
    # OLS:
    # Stock Excess Return = Alpha + Beta * Market Excess Return

    x = data["Market_Excess"].values
    y = data["Stock_Excess"].values

    beta, alpha = np.polyfit(x, y, 1)

    # Fitted values
    data["Predicted"] = alpha + beta * data["Market_Excess"]

    # R-squared
    residuals = y - data["Predicted"].values

    ss_res = np.sum(residuals**2)
    ss_total = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - ss_res / ss_total if ss_total != 0 else np.nan

    # CAPM expected return
    average_market_return = data["Market"].mean()

    historical_premium_annual = data["Market_Excess"].mean() * 12
        
    if historical_premium_annual <= 0 and MARKET_PREMIUM_ANNUAL is None:
        print(f"[WARNING] {company_name}: premi pasar historis = "
              f"{historical_premium_annual:.2%} (<= 0). CAPM dengan premi ini tidak "
              "bermakna secara ekonomi. Isi MARKET_PREMIUM_ANNUAL dengan angka dari sumber.")

    if MARKET_PREMIUM_ANNUAL is None:
        premium_used = historical_premium_annual
        premium_source = "historical"
    else:
        premium_used = MARKET_PREMIUM_ANNUAL
        premium_source = "assumed"

    expected_annual = risk_free_annual + beta * premium_used

    # Historical average annualized stock return
    historical_annual = data["Stock"].mean() * 12

    result = {
        "Company": company_name,
        "Ticker": stock,
        "Market Index": market,
        "Observations": len(data),
        "Beta": beta,
        "Alpha Monthly": alpha,
        "R-Squared": r_squared,
        "Risk-Free Average": risk_free.reindex(data.index).mean(),
        "Risk-Free Last": risk_free_annual,
        "Market Return Annual": average_market_return * 12,
        "Historical Premium Annual": historical_premium_annual,
        "Premium Used": premium_used,
        "Premium Source": premium_source,
        "CAPM Expected Annual": expected_annual,
        "Historical Return Annual": historical_annual,
    }

    return result, data


# ============================================================
# 5. RUN ALL COMPANIES
# ============================================================


def run_analysis():

    prices = download_data()

    monthly_prices, monthly_returns = calculate_returns(prices)
    
    risk_free = load_risk_free()
    check_risk_free(risk_free, monthly_returns)

    results = []
    regression_data = {}

    for company_name, config in TICKERS.items():

        output = estimate_capm(monthly_returns, risk_free, company_name, config)
        if output is None:
            continue

        result, data = output

        results.append(result)
        regression_data[company_name] = data

    if not results:
        raise ValueError("No valid CAPM results were produced.")

    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(OUTPUT_DIR / "capm_results.csv", index=False)

    monthly_prices.to_csv(OUTPUT_DIR / "monthly_prices.csv")

    monthly_returns.to_csv(OUTPUT_DIR / "monthly_returns.csv")

    print("\n================ CAPM RESULTS ================")

    print(results_df.round(4).to_string(index=False))

    return prices, monthly_prices, monthly_returns, results_df, regression_data


# ============================================================
# 6. VISUALIZATION 1: NORMALIZED STOCK PRICES
# ============================================================


def plot_normalized_prices(prices):

    # Hanya perusahaan yang datanya benar-benar ada di hasil download
    companies = {
        name: cfg["stock"]
        for name, cfg in TICKERS.items()
        if cfg["stock"] in prices.columns
    }

    if not companies:
        print("No stock price data available for plotting.")
        return

    stock_prices = prices[list(companies.values())].dropna()

    if stock_prices.empty:
        print("No overlapping stock price data.")
        return

    # Normalize all stocks to 100 at the first common date
    normalized = (stock_prices / stock_prices.iloc[0]) * 100

    plt.figure(figsize=(12, 6))

    for name, ticker in companies.items():
        plt.plot(normalized.index, normalized[ticker], label=name, linewidth=2)

    # Indeks pasar sebagai pembanding
    market = next(iter(TICKERS.values()))["market"]
    if market in prices.columns:
        market_prices = prices[market].reindex(stock_prices.index).dropna()
        if not market_prices.empty:
            market_norm = (market_prices / market_prices.iloc[0]) * 100
            plt.plot(
                market_norm.index,
                market_norm,
                label=f"Market ({market})",
                linewidth=2,
                color="black",
                linestyle=":",
            )

    plt.axhline(100, linestyle="--", alpha=0.6)

    plt.title("Normalized Stock Price vs Market", fontsize=15, fontweight="bold")

    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Base = 100)")

    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "01_normalized_prices.png", dpi=300)

    plt.show()


# ============================================================
# 7. VISUALIZATION 2: MONTHLY STOCK RETURNS
# ============================================================


def plot_monthly_returns(monthly_returns):

    companies = [
        (name, cfg["stock"])
        for name, cfg in TICKERS.items()
        if cfg["stock"] in monthly_returns.columns
    ]

    if not companies:
        print("No return data available for plotting.")
        return

    fig, axes = plt.subplots(
        len(companies), 1, figsize=(13, 3.2 * len(companies)), sharex=True
    )

    if len(companies) == 1:
        axes = [axes]

    for ax, (name, ticker) in zip(axes, companies):

        ax.plot(monthly_returns.index, monthly_returns[ticker] * 100, linewidth=1)

        ax.axhline(0, color="black", linewidth=0.8)

        ax.set_title(f"{name} Monthly Returns")
        ax.set_ylabel("Return (%)")

    axes[-1].set_xlabel("Date")

    #fig.suptitle("Monthly Stock Returns", fontsize=15, fontweight="bold")

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "02_monthly_returns.png", dpi=300)

    plt.show()


# ============================================================
# 8. VISUALIZATION 3: CAPM REGRESSION
# ============================================================


def plot_capm_regression(regression_data, results_df):

    companies = list(regression_data.keys())

    # Satu baris dua kolom supaya tetap terbaca untuk 4 perusahaan atau lebih
    n_cols = 2 if len(companies) > 1 else 1
    n_rows = int(np.ceil(len(companies) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 5.5 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    flat_axes = axes.ravel()

    for ax, company in zip(flat_axes, companies):

        data = regression_data[company]

        x = data["Market_Excess"] * 100
        y = data["Stock_Excess"] * 100

        beta = results_df.loc[results_df["Company"] == company, "Beta"].iloc[0]

        alpha = results_df.loc[results_df["Company"] == company, "Alpha Monthly"].iloc[
            0
        ]

        r2 = results_df.loc[results_df["Company"] == company, "R-Squared"].iloc[0]

        ax.scatter(x, y, alpha=0.65)

        x_line = np.linspace(x.min(), x.max(), 100)

        y_line = alpha * 100 + beta * x_line

        ax.plot(
            x_line,
            y_line,
            linewidth=2,
            label=f"OLS: Beta = {beta:.3f}, R² = {r2:.2f}",
        )

        ax.axhline(0, linestyle="--", alpha=0.4)
        ax.axvline(0, linestyle="--", alpha=0.4)

        ax.set_title(f"{company}: CAPM Regression", fontweight="bold")

        ax.set_xlabel("Market Excess Return (%)")
        ax.set_ylabel("Stock Excess Return (%)")

        ax.legend()

    # Sembunyikan kotak kosong jika jumlah perusahaan ganjil
    for ax in flat_axes[len(companies):]:
        ax.axis("off")

    plt.savefig(OUTPUT_DIR / "03_capm_regression.png", dpi=300)

    plt.show()


# ============================================================
# 9. VISUALIZATION 4: BETA COMPARISON
# ============================================================


def plot_beta_comparison(results_df):

    plt.figure(figsize=(8, 5))

    ax = sns.barplot(data=results_df, x="Company", y="Beta")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)

    plt.axhline(1, linestyle="--", color="black", label="Market Beta = 1")

    plt.title("Systematic Risk (Beta) by Company", fontsize=15, fontweight="bold")

    plt.ylabel("Estimated Beta")

    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "04_beta_comparison.png", dpi=300)

    plt.show()


# ============================================================
# 10. VISUALIZATION 5: EXPECTED RETURN COMPARISON
# ============================================================


def plot_expected_returns(results_df):

    plot_data = results_df.melt(
        id_vars="Company",
        value_vars=["CAPM Expected Annual", "Historical Return Annual"],
        var_name="Return Type",
        value_name="Return",
    )

    plot_data["Return"] *= 100

    plt.figure(figsize=(10, 6))

    sns.barplot(data=plot_data, x="Company", y="Return", hue="Return Type")

    plt.title(
        "CAPM Expected Return vs Historical Return", fontsize=15, fontweight="bold"
    )

    plt.ylabel("Annualized Return (%)")
    plt.xlabel("Company")

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "05_expected_returns.png", dpi=300)

    plt.show()


# ============================================================
# 11. MAIN PROGRAM
# ============================================================


def main():

    (prices, monthly_prices, monthly_returns, results_df, regression_data) = (
        run_analysis()
    )

    print("\nGenerating visualizations...")

    plot_normalized_prices(prices)

    plot_monthly_returns(monthly_returns)

    plot_capm_regression(regression_data, results_df)

    plot_beta_comparison(results_df)

    plot_expected_returns(results_df)

    print("\n==========================================")
    print("CAPM ANALYSIS COMPLETED")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("==========================================")


if __name__ == "__main__":
    main()