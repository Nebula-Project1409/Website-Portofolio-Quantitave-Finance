

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1. CONFIGURATION


FILE_PATH = r"C:\Users\ghina\Downloads\fa581c85-dc12-477b-bade-fb824defc45e_Series - Metadata.csv"


COUNTRY = "Country"
YEAR = "Year"

Y = "NPL"

X = [
    "GDP_Growth",
    "Inflation",
    "Unemployment",
    "Lending_Rate",
    "CAR"
]

Q = "GDP_Growth"

TRIMMING = 0.15
GRID_POINTS = 40
BOOTSTRAP = 199

OUTPUT = "outputs"

os.makedirs(OUTPUT, exist_ok=True)

rng = np.random.default_rng(42)


# 2. LOAD DATA

def load_data():

    df = pd.read_csv(FILE_PATH)

    required = [COUNTRY, YEAR, Y] + X

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing}\n"
            f"Kolom tersedia: {df.columns.tolist()}"
        )

    df = df[required].copy()

    for col in [YEAR, Y] + X:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna(subset=required)

    df[COUNTRY] = df[COUNTRY].astype(str)
    df[YEAR] = df[YEAR].astype(int)

    if df.duplicated([COUNTRY, YEAR]).any():
        raise ValueError(
            "Terdapat duplikasi Country-Year."
        )

    df = df.sort_values(
        [COUNTRY, YEAR]
    ).reset_index(drop=True)

    if df[COUNTRY].nunique() < 2:
        raise ValueError(
            "Diperlukan minimal dua negara."
        )

    if df[Q].nunique() < 5:
        raise ValueError(
            "Variasi threshold variable terlalu sedikit."
        )

    print("\n===== DATA OVERVIEW =====")
    print("Observations:", len(df))
    print("Countries:", df[COUNTRY].nunique())
    print("Period:", df[YEAR].min(), "-", df[YEAR].max())
    print("\nObservations per country:")
    print(df.groupby(COUNTRY).size())

    print("\nDescriptive statistics:")
    print(df[[Y] + X].describe().round(3))

    df.to_csv(
        f"{OUTPUT}/cleaned_data.csv",
        index=False
    )

    return df



# 3. PREPARE FIXED EFFECTS DATA


def prepare_fe(df):

    # Country fixed effects are removed
    # using the within transformation.

    y = df[Y].to_numpy(dtype=float)

    x = df[X].to_numpy(dtype=float)

    group = df[COUNTRY].to_numpy()

    y_mean = (
        df.groupby(COUNTRY)[Y]
        .transform("mean")
        .to_numpy()
    )

    x_mean = (
        df.groupby(COUNTRY)[X]
        .transform("mean")
        .to_numpy()
    )

    y_within = y - y_mean
    x_within = x - x_mean

    return y_within, x_within, group



# 4. REGRESSION FUNCTION


def fit_model(y, matrix):

    beta, _, rank, _ = np.linalg.lstsq(
        matrix, y, rcond=None
    )

    residual = y - matrix @ beta

    ssr = float(residual @ residual)

    return {
        "beta": beta,
        "residual": residual,
        "SSR": ssr,
        "rank": rank
    }



# 5. OLS BASELINE


def run_ols(df):

    print("\n===== STAGE 1: ORDINARY LEAST SQUARES =====")

    matrix = sm.add_constant(
        df[X].astype(float)
    )

    model = sm.OLS(
        df[Y].astype(float),
        matrix
    ).fit()

    print(model.summary())

    print("\nOLS SSR:", round(
        np.sum(model.resid ** 2), 4
    ))

    print("OLS RMSE:", round(
        np.sqrt(np.mean(model.resid ** 2)), 4
    ))

    print("OLS F-statistic:", model.fvalue)
    print("OLS F-test p-value:", model.f_pvalue)

    return model



# 6. FIXED EFFECTS BASELINE


def run_fe(df):

    print("\n===== STAGE 2: COUNTRY FIXED EFFECTS =====")

    yw, xw, group = prepare_fe(df)

    result = fit_model(yw, xw)

    n = len(df)
    countries = df[COUNTRY].nunique()
    p = len(X)

    df_resid = n - countries - p

    rmse = np.sqrt(
        result["SSR"] / n
    )

    r_squared = (
        1 - result["SSR"] / np.sum(yw ** 2)
    )

    print("Within SSR:", round(result["SSR"], 4))
    print("Within R-squared:", round(r_squared, 4))
    print("RMSE:", round(rmse, 4))
    print("Residual degrees of freedom:", df_resid)

    table = pd.DataFrame({
        "Variable": X,
        "Coefficient": result["beta"]
    })

    print("\nFE Coefficients:")
    print(table.round(4))

    table.to_csv(
        f"{OUTPUT}/fe_coefficients.csv",
        index=False
    )

    return result, yw, xw, group


# 7. CREATE THRESHOLD REGIMES


def make_threshold_matrix(df, gamma):

    q = df[Q].to_numpy(dtype=float)

    low = (q <= gamma).astype(float)
    high = (q > gamma).astype(float)

    matrix = df[X].to_numpy(dtype=float)

    # Interact every explanatory variable
    # with the two threshold regimes.

    low_matrix = matrix * low[:, None]
    high_matrix = matrix * high[:, None]

    raw = np.column_stack([
        low_matrix,
        high_matrix
    ])

    # Within transformation by country

    temp = pd.DataFrame(raw)
    temp[COUNTRY] = df[COUNTRY].to_numpy()

    demeaned = (
        temp.groupby(COUNTRY)
        .transform(lambda col: col - col.mean())
        .to_numpy()
    )

    return demeaned, low, high



# 8. THRESHOLD SEARCH


def search_threshold(df, verbose=True):

    if verbose:
        print("\n===== STAGE 3: THRESHOLD SEARCH =====")

    yw, xw, group = prepare_fe(df)

    candidates = np.unique(
        np.quantile(
            df[Q],
            np.linspace(
                TRIMMING,
                1 - TRIMMING,
                GRID_POINTS
            )
        )
    )

    results = []

    for gamma in candidates:

        matrix, low, high = make_threshold_matrix(
            df, gamma
        )

        # Avoid regimes with insufficient observations

        if low.sum() < len(X) + 3:
            continue

        if high.sum() < len(X) + 3:
            continue

        result = fit_model(yw, matrix)

        # Check full column rank
        if result["rank"] < matrix.shape[1]:
            continue

        results.append({
            "Threshold": gamma,
            "SSR": result["SSR"],
            "Low_Observations": int(low.sum()),
            "High_Observations": int(high.sum())
        })

    if not results:
        raise ValueError(
            "Tidak ditemukan threshold yang valid. "
            "Periksa ukuran sampel dan variasi data."
        )

    table = pd.DataFrame(results)

    best = table.loc[table["SSR"].idxmin()]

    gamma = float(best["Threshold"])

    matrix, low, high = make_threshold_matrix(
        df, gamma
    )

    threshold_result = fit_model(yw, matrix)

    if verbose:

        print("\nOptimal threshold:", gamma)
        print("Minimum SSR:", threshold_result["SSR"])
        print("Low regime:", int(low.sum()))
        print("High regime:", int(high.sum()))

        table.to_csv(
            f"{OUTPUT}/threshold_search.csv",
            index=False
        )

        plt.figure(figsize=(9, 5))

        plt.plot(
            table["Threshold"],
            table["SSR"],
            marker="o"
        )

        plt.axvline(
            gamma,
            linestyle="--",
            label=f"Optimal threshold = {gamma:.3f}"
        )

        plt.xlabel("Candidate Threshold")
        plt.ylabel("Sum of Squared Residuals")
        plt.title("Threshold Search: SSR Minimization")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"{OUTPUT}/threshold_search.png",
            dpi=150
        )

        plt.show()

    return gamma, threshold_result, table



# 9. SUP-F TEST

def calculate_sup_f(ssr_null, ssr_alt, df_resid):

    extra_parameters = len(X)

    if ssr_alt <= 0 or df_resid <= 0:
        return np.nan

    return (
        (ssr_null - ssr_alt) / extra_parameters
    ) / (
        ssr_alt / df_resid
    )


def run_sup_f(df, fe_result, threshold_result):

    print("\n===== STAGE 4: SUP-F TEST =====")

    n = len(df)
    countries = df[COUNTRY].nunique()

    df_resid = (
        n - countries - 2 * len(X)
    )

    sup_f = calculate_sup_f(
        fe_result["SSR"],
        threshold_result["SSR"],
        df_resid
    )

    print("Null SSR:", fe_result["SSR"])
    print("Threshold SSR:", threshold_result["SSR"])
    print("Sup-F statistic:", sup_f)

    print("\nH0: No threshold effect.")
    print("H1: Threshold effect exists.")

    print(
        "Note: Sup-F is not compared with an ordinary "
        "F distribution. Bootstrap inference is required."
    )

    return sup_f



# 10. RESIDUAL BOOTSTRAP UNDER NULL


def bootstrap_test(
    df,
    observed_sup_f,
    fe_result,
    B=BOOTSTRAP
):

    print("\n===== STAGE 5: BOOTSTRAP TEST =====")

    print("Replications:", B)

    yw, xw, group = prepare_fe(df)

    # Null model fitted values in within-transformed space
    fitted = xw @ fe_result["beta"]

    residual = fe_result["residual"]

    n = len(df)
    countries = df[COUNTRY].nunique()

    df_resid = (
        n - countries - 2 * len(X)
    )

    if countries < 20:
        print(
            "WARNING: Only",
            countries,
            "country clusters. "
            "Bootstrap inference is exploratory."
        )

    bootstrap_stats = []

    # Residual resampling separately within each country.
    # This preserves country-specific residual distributions,
    # but assumes within-country exchangeability.

    indices = {
        country: np.where(group == country)[0]
        for country in np.unique(group)
    }

    for b in range(B):

        simulated_y = fitted.copy()

        for country, idx in indices.items():

            simulated_y[idx] += rng.choice(
                residual[idx],
                size=len(idx),
                replace=True
            )

        null_fit = fit_model(
            simulated_y,
            xw
        )

        null_ssr = null_fit["SSR"]

        best_ssr = np.inf

        q_values = np.unique(
            np.quantile(
                df[Q],
                np.linspace(
                    TRIMMING,
                    1 - TRIMMING,
                    GRID_POINTS
                )
            )
        )

        for gamma in q_values:

            matrix, low, high = make_threshold_matrix(
                df, gamma
            )

            candidate = fit_model(
                simulated_y,
                matrix
            )

            if candidate["rank"] < matrix.shape[1]:
                continue

            best_ssr = min(
                best_ssr,
                candidate["SSR"]
            )

        if np.isfinite(best_ssr):

            f_value = calculate_sup_f(
                null_ssr,
                best_ssr,
                df_resid
            )

            if np.isfinite(f_value):
                bootstrap_stats.append(f_value)

        if (b + 1) % 25 == 0:
            print(f"Progress: {b + 1}/{B}")

    bootstrap_stats = np.array(bootstrap_stats)

    if len(bootstrap_stats) == 0:
        raise RuntimeError(
            "Bootstrap gagal menghasilkan statistik."
        )

    p_value = (
        1 + np.sum(
            bootstrap_stats >= observed_sup_f
        )
    ) / (
        1 + len(bootstrap_stats)
    )

    print("\nBootstrap valid:", len(bootstrap_stats))
    print("Bootstrap p-value:", round(p_value, 4))

    print("\nInterpretation:")

    if p_value < 0.05:
        print(
            "Evidence of threshold effect at 5% level, "
            "subject to bootstrap assumptions and sample limitations."
        )
    else:
        print(
            "Insufficient evidence of threshold effect at 5% level."
        )

    pd.DataFrame({
        "Bootstrap_Sup_F": bootstrap_stats
    }).to_csv(
        f"{OUTPUT}/bootstrap_results.csv",
        index=False
    )

    return p_value


# 11. BEFORE AND AFTER VISUALIZATION


def plot_comparison(df, ols_model, gamma):

    print("\n===== STAGE 6: VISUAL COMPARISON =====")

    q_grid = np.linspace(
        df[Q].min(),
        df[Q].max(),
        150
    )

    scenario = pd.DataFrame({
        col: [df[col].median()] * len(q_grid)
        for col in X
    })

    scenario[Q] = q_grid

    # OLS predictions
    before = ols_model.predict(
        sm.add_constant(
            scenario[X],
            has_constant="add"
        )
    )

    # Threshold predictions using pooled scenario slopes.
    # For a visual illustration, use regime coefficients
    # estimated by the country-FE threshold model.

    yw, xw, group = prepare_fe(df)

    matrix, _, _ = make_threshold_matrix(
        df, gamma
    )

    # Re-estimate final threshold model
    # to retrieve regime coefficients
    result = fit_model(yw, matrix)

    p = len(X)

    beta_low = result["beta"][:p]
    beta_high = result["beta"][p:]

    # Use centered scenario values for within-model slopes.
    scenario_centered = scenario[X].to_numpy() - df[X].mean().to_numpy()

    after = []

    for i, value in enumerate(q_grid):

        beta = (
            beta_low if value <= gamma
            else beta_high
        )

        after.append(
            scenario_centered[i] @ beta
        )

    after = np.array(after)

    # Center OLS prediction for visual comparison
    before = np.asarray(before) - np.mean(ols_model.fittedvalues)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5)
    )

    axes[0].scatter(
        df[Q],
        df[Y],
        alpha=0.4
    )

    axes[0].plot(
        q_grid,
        before,
        linewidth=2
    )

    axes[0].set_title(
        "BEFORE: Linear OLS"
    )

    axes[0].set_xlabel(Q)
    axes[0].set_ylabel(
        "NPL (observed and centered fitted values)"
    )

    axes[1].scatter(
        df[Q],
        df[Y] - df.groupby(COUNTRY)[Y].transform("mean"),
        alpha=0.4
    )

    low = q_grid <= gamma
    high = q_grid > gamma

    axes[1].plot(
        q_grid[low],
        after[low],
        linewidth=2,
        label="Low regime"
    )

    axes[1].plot(
        q_grid[high],
        after[high],
        linewidth=2,
        label="High regime"
    )

    axes[1].axvline(
        gamma,
        linestyle="--",
        label=f"Threshold = {gamma:.3f}"
    )

    axes[1].set_title(
        "AFTER: Fixed Effects Threshold"
    )

    axes[1].set_xlabel(Q)
    axes[1].set_ylabel(
        "Within-transformed NPL"
    )

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT}/before_after.png",
        dpi=150
    )

    plt.show()



# 12. REGIME COEFFICIENTS

def coefficient_table(threshold_result):

    p = len(X)

    table = pd.DataFrame({

        "Variable": X,

        "Low_Regime": threshold_result["beta"][:p],

        "High_Regime": threshold_result["beta"][p:]

    })

    table["Difference"] = (
        table["High_Regime"] -
        table["Low_Regime"]
    )

    print("\n===== REGIME COEFFICIENTS =====")
    print(table.round(4))

    table.to_csv(
        f"{OUTPUT}/regime_coefficients.csv",
        index=False
    )

    return table



# 13. FINAL COMPARISON

def compare_models(
    fe_result,
    threshold_result,
    gamma,
    p_value
):

    table = pd.DataFrame({

        "Model": [
            "Country Fixed Effects",
            "Threshold Fixed Effects"
        ],

        "SSR": [
            fe_result["SSR"],
            threshold_result["SSR"]
        ]

    })

    table["SSR_Reduction"] = [
        0,
        fe_result["SSR"] - threshold_result["SSR"]
    ]

    print("\n===== FINAL MODEL COMPARISON =====")
    print(table.round(4))

    print("\nOptimal threshold:", gamma)
    print("Bootstrap p-value:", p_value)

    table.to_csv(
        f"{OUTPUT}/model_comparison.csv",
        index=False
    )



# 14. MAIN

def main():

    print("=" * 60)
    print("BANKING CREDIT RISK: THRESHOLD REGRESSION")
    print("=" * 60)

    df = load_data()

    ols_model = run_ols(df)

    fe_result, yw, xw, group = run_fe(df)

    gamma, threshold_result, search_table = (
        search_threshold(df)
    )

    observed_sup_f = run_sup_f(
        df,
        fe_result,
        threshold_result
    )

    p_value = bootstrap_test(
        df,
        observed_sup_f,
        fe_result
    )

    plot_comparison(
        df,
        ols_model,
        gamma
    )

    coefficient_table(
        threshold_result
    )

    compare_models(
        fe_result,
        threshold_result,
        gamma,
        p_value
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("Files saved in:", OUTPUT)
    print("=" * 60)


if __name__ == "__main__":
    main()
