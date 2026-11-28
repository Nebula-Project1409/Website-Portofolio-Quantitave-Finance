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
import json


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
PREMIUM_SCENARIOS = [0.03, 0.05, 0.07]   # (isi seperti di pesan sebelumnya, beserta komentarnya)

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
    monthly_returns = monthly_prices.pct_change(fill_method=None)

    monthly_returns = monthly_returns.replace([np.inf, -np.inf], np.nan)

    return monthly_prices, monthly_returns

# ============================================================
# 3b. LOAD RISK-FREE RATE (BOND YIELD)
# ============================================================


def load_risk_free():

    rf = pd.read_csv(
        RF_FILE,
        encoding="utf-8-sig",
        dtype=str
    )

    # Parse tanggal dan nilai yield
    rf["Date"] = pd.to_datetime(
        rf["Tanggal"],
        format="%d/%m/%Y"
    )

    rf["Yield"] = (
        rf["Terakhir"]
        .str.replace(",", ".", regex=False)
        .astype(float) / 100
    )

    # Urutkan tanggal
    rf = rf.sort_values("Date")

    # Ubah label tanggal awal bulan menjadi akhir bulan
    rf["Date"] = rf["Date"] + pd.offsets.MonthEnd(0)

    # Jadikan tanggal sebagai index
    risk_free = rf.set_index("Date")["Yield"]

    return risk_free


def check_risk_free(risk_free, monthly_returns):

    # Risk-free rate bulan sebelumnya untuk return bulan berjalan
    risk_free_lagged = risk_free.shift(1, freq="ME")

    months = monthly_returns.dropna(how="all").index
    available_rf = risk_free_lagged.dropna().index

    missing = months.difference(available_rf)

    print("\nRisk-free data (bond yield, annual):")
    print(
        f"  File period : {risk_free.index.min().date()} "
        f"to {risk_free.index.max().date()}"
    )
    print(
        f"  Analysis    : {months.min().date()} "
        f"to {months.max().date()}"
    )
    print(f"  Yield range : {risk_free.min():.2%} to {risk_free.max():.2%}")

    if len(missing) > 0:
        raise ValueError(
            f"Risk-free data is missing for {len(missing)} months "
            "after applying the one-month lag. "
            f"Examples: {[d.date() for d in missing[:3]]}"
        )


# ============================================================
# 3c. MARKET RISK PREMIUM (dihitung SATU kali untuk semua saham)
# ============================================================
def estimate_market_premium(monthly_returns, risk_free, market):

    # Rf bulanan, pakai yield akhir bulan sebelumnya (sama seperti di estimate_capm)
    rf_monthly = ((1 + risk_free.shift(1, freq="ME")) ** (1 / 12) - 1).rename("RF")

    data = pd.concat(
        [monthly_returns[market].rename("Market"), rf_monthly], axis=1, sort=True
    ).dropna()

    excess = data["Market"] - data["RF"]

    premium_annual = excess.mean() * 12
    se_annual = excess.std(ddof=1) / np.sqrt(len(excess)) * 12

    return {
        "premium_annual": premium_annual,
        "se_annual": se_annual,
        "t_stat": premium_annual / se_annual,
        "months": len(excess),
    }

# ============================================================
# 4. CAPM ESTIMATION USING OLS
# ============================================================
def estimate_capm(monthly_returns, risk_free, market_premium, company_name, config):

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
    # Align previous month's annual yield with current month's return
    
    risk_free_annual_aligned = risk_free.shift(1, freq="ME")

# Convert annual yield into effective monthly rate
    risk_free_monthly = (
    (1 + risk_free_annual_aligned) ** (1 / 12) - 1
    ).rename("RF")

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

    X = sm.add_constant(data["Market_Excess"])
    Y = data["Stock_Excess"]

    model = sm.OLS(Y, X).fit()

    # Regression coefficients
    alpha = model.params["const"]
    beta = model.params["Market_Excess"]

    # Goodness of fit
    r_squared = model.rsquared

    # Standard Error
    alpha_se = model.bse["const"]
    beta_se = model.bse["Market_Excess"]

    # t-statistic
    alpha_t = model.tvalues["const"]
    beta_t = model.tvalues["Market_Excess"]

    # p-value
    alpha_p = model.pvalues["const"]
    beta_p = model.pvalues["Market_Excess"]

    # 95% Confidence Interval
    ci = model.conf_int(alpha=0.05)

    alpha_ci_lower = ci.loc["const"].iloc[0]
    alpha_ci_upper = ci.loc["const"].iloc[1]

    beta_ci_lower = ci.loc["Market_Excess"].iloc[0]
    beta_ci_upper = ci.loc["Market_Excess"].iloc[1]

    # Fitted values
    data["Predicted"] = model.fittedvalues



    # CAPM expected return
    average_market_return = data["Market"].mean()

    historical_premium_annual = market_premium["premium_annual"]
        
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

    # Alpha Statistical Inference
    "Alpha SE": alpha_se,
    "Alpha t-stat": alpha_t,
    "Alpha p-value": alpha_p,
    "Alpha CI 95% Lower": alpha_ci_lower,
    "Alpha CI 95% Upper": alpha_ci_upper,

    # Beta Statistical Inference
    "Beta SE": beta_se,
    "Beta t-stat": beta_t,
    "Beta p-value": beta_p,
    "Beta CI 95% Lower": beta_ci_lower,
    "Beta CI 95% Upper": beta_ci_upper,

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
    
    market = next(iter(TICKERS.values()))["market"]
    market_premium = estimate_market_premium(monthly_returns, risk_free, market)

    print("\nMarket risk premium (annual, computed once for all stocks):")
    print(f"  Premium       : {market_premium['premium_annual']:.2%}")
    print(f"  Std. error    : {market_premium['se_annual']:.2%}")
    print(f"  t-statistic   : {market_premium['t_stat']:.2f}")
    print(f"  Months        : {market_premium['months']}")

    results = []
    regression_data = {}

    for company_name, config in TICKERS.items():

        output = estimate_capm(monthly_returns, risk_free, market_premium, company_name, config)
        if output is None:
            continue

        result, data = output

        results.append(result)
        regression_data[company_name] = data

    if not results:
        raise ValueError("No valid CAPM results were produced.")
    
    results_df = pd.DataFrame(results)
    
    # ========================================================
    # EXPORT ALL RESULTS TO EXCEL
    # ========================================================

    excel_path = OUTPUT_DIR / "CAPM_Analysis_Results.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        # Sheet 1: CAPM Results
        results_df.to_excel(
            writer,
            sheet_name="CAPM Results",
            index=False
        )

        # Sheet 2: Monthly Prices
        monthly_prices.to_excel(
            writer,
            sheet_name="Monthly Prices"
        )

        # Sheet 3: Monthly Returns
        monthly_returns.to_excel(
            writer,
            sheet_name="Monthly Returns"
        )

        # Sheet 4 onwards: Regression Data
        for company_name, data in regression_data.items():

            sheet_name = f"{company_name} Regression"

            data.to_excel(
                writer,
                sheet_name=sheet_name
            )

    print(f"\nExcel file saved: {excel_path}")


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
# 10b. EXPORT RESULTS FOR THE WEBSITE (hasil.json)
# ============================================================
def export_json(results_df, regression_data):

    first = next(iter(regression_data.values()))

    columns = {
        "Company": "company",
        "Ticker": "ticker",
        "Observations": "observations",
        "Beta": "beta",
        "Beta SE": "beta_se",
        "Beta t-stat": "beta_t",
        "Beta p-value": "beta_p",
        "Beta CI 95% Lower": "beta_ci_low",
        "Beta CI 95% Upper": "beta_ci_high",
        "Alpha Monthly": "alpha_monthly",
        "Alpha SE": "alpha_se",
        "Alpha t-stat": "alpha_t",
        "Alpha p-value": "alpha_p",
        "Alpha CI 95% Lower": "alpha_ci_low",
        "Alpha CI 95% Upper": "alpha_ci_high",
        "R-Squared": "r2",
        "CAPM Expected Annual": "capm_expected_annual",
        "Historical Return Annual": "historical_return_annual",
    }

    stocks = results_df[list(columns)].rename(columns=columns)
    row = results_df.iloc[0]

    output = {
        "period": [str(first.index.min().date()), str(first.index.max().date())],
        "market_index": row["Market Index"],
        "risk_free_last": float(row["Risk-Free Last"]),
        "market_return_annual": float(row["Market Return Annual"]),
        "market_premium_annual": float(row["Premium Used"]),
        "premium_source": row["Premium Source"],
        "stocks": json.loads(stocks.to_json(orient="records")),
    }

    with open(OUTPUT_DIR / "hasil.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWebsite data saved: {OUTPUT_DIR / 'hasil.json'}")
    
    
    
# ============================================================
# 11. MAIN PROGRAM
# ============================================================


def main():

    (prices, monthly_prices, monthly_returns, results_df, regression_data) = (
        run_analysis()
    )
    
    export_json(results_df, regression_data)

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