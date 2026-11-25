
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import textwrap
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

try:  
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass



# 0. CONFIGURATION (edit here, or override with command-line flags)

BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "data_path": r"C:\Users\Latihan Ghina Python\NEBULA PROJECT\WEB PORTOFOLIO\projects\treshold-regression\1_Model\data\ASEAN_Banking_PanelData_2019_2025.xlsx",
    "countries": [],
    "year_start": 2019,
    "year_end": 2025,
}
    

    # Sesuaikan nama kolom dengan yang ada di Excel kamu

CONFIG.update({
    "y": "NPL",
    "x": ["ESG "],
    "q": "GDP",
    "controls": [
        "SIZE ",
        "CIR",
        "CAR",
        "LDR",
        "LR",
        "INF",
    ],

    "lag_regressors": 0,
    "missing_strategy": "drop",
    "interpolate_limit": 1,
    "balance": "keep",
    "min_obs_per_entity": 3,

    "trim": 0.15,
    "max_grid": 300,
    "n_bootstrap": 500,
    "bootstrap_scheme": "iid",
    "alpha": 0.05,
    "seed": 42,

    "output_dir": str(BASE_DIR / "outputs"),
    "show_plots": False,
    "educational": True,
})


import matplotlib  # noqa: E402

if not CONFIG["show_plots"]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # noqa: E402
    HAS_SNS = True
except Exception:  # seaborn is optional
    HAS_SNS = False

C1, C2, CL, CGREY = "#682127", "#d9780f", "#3a6ea5", "#8a7f7a"  # regime 1, regime 2, linear, neutral

# Metadata only for series whose meaning is certain. Anything else is NOT guessed.
KNOWN_SERIES = {
    "GFDD.SI.01": ("Bank_Z_Score", "Bank Z-score: distance to insolvency, higher = more stable banks."),
    "GFDD.SI.02": ("NPL_Ratio", "Bank nonperforming loans to gross loans (%). Credit-risk (asset quality) indicator."),
    "GFDD.SI.05": ("Regulatory_Capital_RWA", "Bank regulatory capital to risk-weighted assets (%). Capital adequacy ratio."),
    "GFDD.EI.05": ("Bank_ROA", "Bank return on assets (%, after tax). Profitability."),
    "GFDD.EI.06": ("Bank_ROE", "Bank return on equity (%, after tax). Profitability."),
    "GFDD.SI.03": ("Capital_to_Assets", "Bank capital to assets ratio (%). Loss-absorbing buffer."),
    "GFDD.SI.04": ("Credit_to_Deposits", "Bank credit to bank deposits (%). Funding / liquidity structure."),
    "GFDD.DI.01": ("Private_Credit_GDP", "Private credit by deposit money banks to GDP (%). Financial depth."),
    "NY.GDP.MKTP.KD.ZG": ("GDP_Growth", "GDP growth (annual %). Business-cycle indicator."),
    "FP.CPI.TOTL.ZG": ("Inflation", "Inflation, consumer prices (annual %)."),
    "SL.UEM.TOTL.ZS": ("Unemployment", "Unemployment, total (% of labour force, modelled ILO estimate)."),
    "FR.INR.LEND": ("Lending_Rate", "Lending interest rate (%). Price of credit."),
    "FR.INR.RINR": ("Real_Interest_Rate", "Real interest rate (%)."),
    "NY.GDP.PCAP.KD": ("GDP_per_Capita", "GDP per capita (constant US$)."),
    "PA.NUS.FCRF": ("Exchange_Rate", "Official exchange rate (local currency per US$, period average)."),
    "NE.TRD.GNFS.ZS": ("Trade_GDP", "Trade (% of GDP). Openness."),
}
NAME_HINTS = [  # weak hints from the NAME only; always labelled as guesses
    (r"nonperform|npl|impaired", "looks like an asset-quality / credit-risk indicator (candidate for Y)"),
    (r"z.?score", "looks like a bank-stability measure (candidate for Y)"),
    (r"capital", "looks like a capital-adequacy measure (candidate control)"),
    (r"gdp.*growth|growth", "looks like business-cycle activity (candidate for X or q)"),
    (r"inflation|cpi", "looks like a price-level variable (candidate for X)"),
    (r"unemploy", "looks like a labour-market variable (candidate for X)"),
    (r"interest|lending|rate", "looks like a price-of-credit / policy variable (candidate for X)"),
    (r"exchange", "looks like an external-sector variable (candidate for X)"),
    (r"credit|loan|deposit", "looks like a credit-volume / funding variable (candidate control)"),
]


# =============================================================================
# 1. SMALL UTILITIES
# =============================================================================
def hr(title: str | None = None) -> None:
    print("\n" + "=" * 80)
    if title:
        print(title)
        print("=" * 80)


def edu(title: str, text: str) -> None:
    """Educational note, printed only when CONFIG['educational'] is True."""
    if CONFIG["educational"]:
        body = textwrap.indent(textwrap.dedent(text).strip(), "    ")
        print(f"\n[LEARN] {title}\n{body}\n")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt_table(df: pd.DataFrame, digits: int = 4) -> str:
    return df.to_string(float_format=lambda v: f"{v:,.{digits}f}")


def slug(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")[:40] or "var"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def savefig(fig, name: str) -> str:
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    path = os.path.join(CONFIG["output_dir"], name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if CONFIG["show_plots"]:
        plt.show()
    plt.close(fig)
    print(f"   saved figure: {path}")
    return path


# =============================================================================
# 2. DATA LOADING
# =============================================================================
YEAR_RE = re.compile(r"^\s*(\d{4})(?:\s*\[YR\d{4}\])?\s*$")


def read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Download it from the World Bank DataBank (see README / the notes in this "
            "script) or run with --demo to use synthetic data."
        )
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, na_values=["..", ""])
    last_err = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, na_values=["..", "", "NA", "N/A"], low_memory=False)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err  # type: ignore[misc]


def _first(cols, candidates):
    low = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def to_panel_frame(raw: pd.DataFrame):
    """Return (long panel frame, metadata dict).

    Handles (a) the World Bank DataBank / WDI / GFDD 'wide' layout, where every
    row is one country x one series and the years are columns, and (b) an
    ordinary long file with one row per country-year.
    """
    cols = list(raw.columns)
    year_cols = [c for c in cols if YEAR_RE.match(str(c))]
    ctry = _first(cols, ["Country Name", "Country", "Economy", "Entity", "country_name"])
    ccode = _first(cols, ["Country Code", "ISO3", "ISO", "country_code"])
    scode = _first(cols, ["Series Code", "Indicator Code"])
    sname = _first(cols, ["Series Name", "Indicator Name"])
    meta: dict[str, dict] = {}

    if len(year_cols) >= 3 and (scode or sname) and ctry:
        print("   format detected: World Bank WIDE (years as columns) -> converting to LONG")
        defs: dict[str, str] = {}
        key_for_footer = scode or sname
        # DataBank appends notes ('Data from database', 'Last Updated', a definitions table);
        # they are not data rows, so keep only rows that carry a series identifier.
        is_def = raw[ctry].astype(str).str.strip().eq("Code").to_numpy()
        if is_def.any():
            at = int(np.argmax(is_def))
            foot = raw.iloc[at + 1:]
            for _, r in foot.iterrows():
                code = str(r.iloc[0]).strip()
                short = str(r.iloc[2]).strip() if len(r) > 2 and pd.notna(r.iloc[2]) else ""
                if code and code.lower() != "nan" and short and short.lower() != "nan":
                    defs[code] = short
            raw = raw.iloc[:at]   # everything from the definitions table downwards is not data
        raw = raw[raw[ctry].notna() & raw[key_for_footer].notna()].copy()
        if "Last Updated" in " ".join(raw[ctry].astype(str)):
            raw = raw[~raw[ctry].astype(str).str.startswith(("Last Updated", "Data from database"))]
        id_cols = [c for c in cols if c not in year_cols]
        m = raw.melt(id_vars=id_cols, value_vars=year_cols, var_name="_y", value_name="value")
        m["year"] = m["_y"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
        m["value"] = pd.to_numeric(m["value"], errors="coerce")
        key = scode or sname
        used: set[str] = set()
        mapping: dict[str, str] = {}
        for _, r in m[[key] + ([sname] if sname else [])].drop_duplicates().iterrows():
            k = str(r[key])
            nm = str(r[sname]) if sname else k
            if k in KNOWN_SERIES:
                short, desc = KNOWN_SERIES[k]
            else:
                short, desc = slug(nm), (defs.get(k) and f"[from the file's own metadata] {defs[k]}")
            while short in used:
                short += "_" + slug(k)[:6]
            used.add(short)
            mapping[k] = short
            meta[short] = {"code": k, "original_name": nm, "description": desc}
        m["variable"] = m[key].astype(str).map(mapping)
        wide = m.pivot_table(index=[ctry, "year"], columns="variable", values="value",
                             aggfunc="first", dropna=False)
        wide.columns.name = None
        df = wide.reset_index().rename(columns={ctry: "country"})
        if ccode:
            cm = raw[[ctry, ccode]].drop_duplicates().set_index(ctry)[ccode].to_dict()
            df["country_code"] = df["country"].map(cm)
        return df, meta

    ycol = _first(cols, ["year", "time", "date"])
    if ctry and ycol:
        print("   format detected: LONG panel (one row per country-year)")
        df = raw.rename(columns={ctry: "country", ycol: "year"}).copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["country", "year"])
        df["year"] = df["year"].astype(int)
        if ccode and ccode in df.columns:
            df = df.rename(columns={ccode: "country_code"})
        
        # Menggabungkan nama negara dan bank agar tiap bank dianggap entitas unik
        df["country"] = df["country"].ffill() + " - " + df["Bank name"].ffill() 
        for c in df.columns:
            if c not in ("country", "year", "country_code"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(axis=1, how="all")
        for c in df.columns:
            if c not in ("country", "year", "country_code"):
                meta[c] = {"code": None, "original_name": c, "description": None}
        return df, meta

    raise ValueError(
        "Could not recognise the file layout. Expected either a World Bank wide file "
        "(columns 'Country Name', 'Series Code'/'Indicator Code', '2005 [YR2005]', ...) "
        "or a long file with 'country' and 'year' columns."
    )


def select_countries_years(df, countries, y0, y1):
    if countries:
        wanted = {str(c).strip().lower() for c in countries}
        mask = df["country"].astype(str).str.lower().isin(wanted)
        if "country_code" in df.columns:
            mask |= df["country_code"].astype(str).str.lower().isin(wanted)
        present = set(df.loc[mask, "country"].str.lower()) | \
            (set(df.loc[mask, "country_code"].astype(str).str.lower()) if "country_code" in df.columns else set())
        lost = [c for c in countries if str(c).lower() not in present]
        if lost:
            warn(f"countries not found in the file: {lost}")
        df = df[mask]
    df = df[(df["year"] >= y0) & (df["year"] <= y1)]
    return df.sort_values(["country", "year"]).reset_index(drop=True)


def describe_variable(name: str, meta: dict) -> str:
    info = meta.get(name, {})
    if info.get("description"):
        return info["description"]
    label = info.get("original_name", name)
    for pat, hint in NAME_HINTS:
        if re.search(pat, str(label), flags=re.I):
            return f"[guess from the name only] {hint}"
    return "No description available. Check the World Bank metadata before using it."


# =============================================================================
# 3. STAGE 1 — EXPLORATION
# =============================================================================
def explore_data(df: pd.DataFrame, meta: dict) -> None:
    hr("STAGE 1 - DATA EXPLORATION AND PREPARATION")
    var_cols = [c for c in df.columns if c not in ("country", "country_code", "year")]
    print(f"Observations (country-year rows): {len(df)}")
    print(f"Variables: {len(var_cols)}   Countries: {df['country'].nunique()}   "
          f"Years: {df['year'].min()}-{df['year'].max()}")

    edu("Why a panel?", """
        A panel follows the same units (here: countries) over time. Observing many
        countries over many years lets us separate permanent country differences
        (fixed effects, mu_i) from the effect of changing macro conditions.
        """)

    print("\nVariables and what they might be:")
    rows = []
    for c in var_cols:
        nn = int(df[c].notna().sum())
        rows.append({"variable": c, "dtype": str(df[c].dtype), "non_missing": nn,
                     "missing_%": 100 * (1 - nn / max(len(df), 1)),
                     "meaning": describe_variable(c, meta)})
    info = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(info.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    dup = int(df.duplicated(["country", "year"]).sum())
    print(f"\nDuplicate country-year rows: {dup}" + ("  <-- investigate!" if dup else ""))

    span = df.groupby("country")["year"].agg(["min", "max", "count"])
    print("\nCoverage by country (rows in file):")
    print(span.to_string())

    edu("Missing values: what NOT to do", """
        Never fill missing values with 0: a missing NPL ratio is not an NPL ratio of 0,
        and zeros would pull means and regression slopes towards nonsense.
        Options used later in this script: 'drop' rows with a missing value in any
        selected variable (listwise deletion, safest, shrinks the sample) or
        'interpolate' very short gaps inside a country's own series (keeps more data
        but invents values, so it is only reasonable for 1-year gaps).
        """)

    num = df[var_cols].select_dtypes("number")
    if not num.empty:
        print("\nDescriptive statistics:")
        print(fmt_table(num.describe().T, 3))
        corr = num.corr()
        print("\nCorrelation matrix (pairwise-complete observations):")
        print(fmt_table(corr, 2))
        _plot_corr(corr)
    _plot_series(df, var_cols)


def _plot_corr(corr: pd.DataFrame) -> None:
    try:
        if corr.shape[0] < 2:
            return
        fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(corr)), max(4, 0.7 * len(corr))))
        if HAS_SNS:
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
        else:
            im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr)))
            ax.set_xticklabels(corr.columns, rotation=60, ha="right")
            ax.set_yticks(range(len(corr)))
            ax.set_yticklabels(corr.index)
            fig.colorbar(im, ax=ax)
        ax.set_title("Correlation matrix")
        savefig(fig, "01b_correlation_matrix.png")
    except Exception as e:  # plotting must never kill the pipeline
        warn(f"correlation plot skipped: {e}")


def _plot_series(df: pd.DataFrame, var_cols: list[str], max_vars: int = 9) -> None:
    try:
        cols = [c for c in var_cols if df[c].notna().sum() > 0][:max_vars]
        if not cols:
            return
        ncol = 3
        nrow = int(math.ceil(len(cols) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.1 * nrow), squeeze=False)
        for ax, c in zip(axes.ravel(), cols):
            for ctry, g in df.groupby("country"):
                ax.plot(g["year"], g[c], marker="o", ms=3, lw=1.3, label=ctry)
            ax.set_title(c, fontsize=10)
            ax.grid(alpha=.3)
        for ax in axes.ravel()[len(cols):]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=7)
        fig.suptitle("Development of each variable over time, by country", y=1.01)
        fig.tight_layout()
        savefig(fig, "01_series_by_country.png")
    except Exception as e:
        warn(f"time-series plot skipped: {e}")


# =============================================================================
# 4. STAGE 2 — VARIABLE CHOICE AND PANEL CONSTRUCTION
# =============================================================================
class Panel:
    """Stacked panel arrays sorted by (country, year)."""

    def __init__(self, df, y, xs, zs, q):
        self.df = df.reset_index(drop=True)
        self.y_name, self.x_names, self.z_names, self.q_name = y, list(xs), list(zs), q
        self.y = self.df[y].to_numpy(float)
        self.X = self.df[self.x_names].to_numpy(float)
        self.Z = self.df[self.z_names].to_numpy(float) if self.z_names else np.empty((len(self.df), 0))
        self.q = self.df[q].to_numpy(float)
        self.codes, uniques = pd.factorize(self.df["country"])
        self.entities = list(uniques)
        self.N = len(self.entities)
        self.n = len(self.df)
        self.counts = np.bincount(self.codes, minlength=self.N)

    @property
    def reg_names(self):
        return self.x_names + self.z_names


def choose_variables(df: pd.DataFrame, meta: dict, cfg: dict):
    """Use CONFIG values, or ask the user interactively."""
    cands = [c for c in df.columns if c not in ("country", "country_code", "year")
             and df[c].notna().sum() > 0]
    y, xs, zs, q = cfg["y"], list(cfg["x"]), list(cfg["controls"]), cfg["q"]
    if y and xs and q:
        return y, xs, zs, q
    if not sys.stdin.isatty():
        raise SystemExit("Variables not set. Set y / x / q in CONFIG or via --y --x --q "
                         "(or run in a terminal for interactive selection, or use --demo).")
    hr("VARIABLE SELECTION")
    for i, c in enumerate(cands, 1):
        print(f"  [{i:2d}] {c:28s} {describe_variable(c, meta)[:90]}")
    print("\nI will not choose for you: the roles must come from your research question.\n"
          "Typical roles: Y = a credit-risk indicator, X = macro conditions, q = a cycle variable.")

    def ask(prompt, many=False):
        while True:
            s = input(prompt).strip()
            if not s and many:
                return []
            try:
                idx = [int(t) for t in re.split(r"[,\s]+", s) if t]
                picked = [cands[i - 1] for i in idx]
                if picked or many:
                    return picked if many else picked[0]
            except (ValueError, IndexError):
                pass
            print("  please enter valid numbers from the list")

    y = ask("Dependent variable Y (one number): ")
    xs = ask("Regime-dependent X variables (numbers separated by commas): ", many=True)
    q = ask("Threshold variable q (one number): ")
    zs = ask("Controls with the same slope in both regimes (optional, Enter to skip): ", many=True)
    return y, xs, zs, q


def validate_selection(df, y, xs, zs, q):
    """Fatal problems raise ValueError; softer problems return warning strings."""
    problems, notes = [], []
    all_vars = [y] + xs + zs + [q]
    for v in all_vars:
        if v not in df.columns:
            problems.append(f"'{v}' is not a column of the data")
    if problems:
        raise ValueError("; ".join(problems))
    if y in xs + zs:
        raise ValueError("Y cannot also be a regressor.")
    if not xs:
        raise ValueError("Choose at least one X.")
    if q == y:
        raise ValueError("The threshold variable cannot be the dependent variable.")
    for v in all_vars:
        miss = df[v].isna().mean()
        if miss > 0.30:
            notes.append(f"'{v}' has {100 * miss:.0f}% missing values; the usable sample will shrink.")
    if q in xs + zs:
        notes.append(f"q ('{q}') is also a regressor. Allowed in Hansen's model, but read the "
                     "coefficients carefully: the same variable defines the regimes and enters the slope.")
    within_sd = df.groupby("country")[q].std().mean()
    total_sd = df[q].std()
    if total_sd and within_sd / total_sd < 0.25:
        notes.append(f"q ('{q}') varies mostly between countries, not over time "
                     f"(average within-country SD = {within_sd:.2f} vs overall SD = {total_sd:.2f}). "
                     "Regimes will mostly be 'country groups', not 'conditions that change'.")
    for v in xs + zs:
        if df[v].nunique() < 3:
            problems.append(f"'{v}' has fewer than 3 distinct values")
        if df.groupby("country")[v].std().fillna(0).max() == 0:
            problems.append(f"'{v}' never changes within a country; fixed effects would absorb it entirely")
    if problems:
        raise ValueError("; ".join(problems))
    return notes


def build_panel(df: pd.DataFrame, y, xs, zs, q, cfg) -> Panel:
    work = df[["country", "year"] + [y] + xs + zs + ([q] if q not in xs + zs else [])].copy()
    work = work.loc[:, ~work.columns.duplicated()]

    L = int(cfg["lag_regressors"])
    if L > 0:
        print(f"   lagging X, controls and q by {L} year(s) within each country")
        ren = {}
        for v in set(xs + zs + [q]):
            lag = work[["country", "year", v]].copy()
            lag["year"] = lag["year"] + L
            lag = lag.rename(columns={v: f"L{L}_{v}"})
            work = work.drop(columns=[v]).merge(lag, on=["country", "year"], how="left")
            ren[v] = f"L{L}_{v}"
        xs = [ren[v] for v in xs]
        zs = [ren[v] for v in zs]
        q = ren[q]
    use = [y] + xs + zs + [q]
    use = list(dict.fromkeys(use))

    before = len(work)
    if cfg["missing_strategy"] == "interpolate":
        work[use] = work.groupby("country")[use].transform(
            lambda s: s.interpolate(limit=cfg["interpolate_limit"], limit_area="inside"))
        print(f"   interpolated gaps of at most {cfg['interpolate_limit']} year(s) inside each country series")
    work = work.dropna(subset=use)
    print(f"   rows with complete data: {len(work)} of {before} ({100 * len(work) / max(before, 1):.0f}%)")

    if cfg["balance"] == "balance":
        ncty = work["country"].nunique()
        full_years = work.groupby("year")["country"].nunique()
        keep = full_years[full_years == ncty].index
        if len(keep) == 0:
            raise ValueError("No year has complete data for all countries; use balance='keep'.")
        work = work[work["year"].isin(keep)]
        print(f"   balanced panel: kept {len(keep)} common years")

    small = work.groupby("country").size()
    drop = small[small < cfg["min_obs_per_entity"]].index.tolist()
    if drop:
        warn(f"dropping countries with fewer than {cfg['min_obs_per_entity']} observations: {drop}")
        work = work[~work["country"].isin(drop)]
    work = work.sort_values(["country", "year"]).reset_index(drop=True)
    if len(work) < 20:
        raise ValueError(f"Only {len(work)} usable observations; too few for panel estimation.")
    return Panel(work, y, xs, zs, q)


def panel_checks(P: Panel, cfg: dict) -> None:
    """Section 8 of the brief: is the dataset adequate for panel threshold regression?"""
    hr("DATA ADEQUACY CHECK (honesty before modelling)")
    T = P.counts
    years = P.df["year"]
    balanced = len(set(T)) == 1 and T[0] == years.nunique()
    print(f"N (countries)      : {P.N}")
    print(f"T per country      : min {T.min()}, mean {T.mean():.1f}, max {T.max()}")
    print(f"Total observations : {P.n}")
    print(f"Panel is           : {'BALANCED' if balanced else 'UNBALANCED'}")
    print(f"Min obs per regime : {max(int(math.ceil(cfg['trim'] * P.n)), P.X.shape[1] + 2)} "
          f"(trim = {cfg['trim']:.0%})")
    print(f"Distinct q values  : {np.unique(P.q).size}")
    if not balanced:
        warn("Hansen (1999) is stated for BALANCED panels. The within transformation and the "
             "grid search still run, but the asymptotic theory is not formally covered. "
             "Set balance='balance' to test sensitivity.")
    if P.N < 10:
        warn(f"N = {P.N} is very small. Hansen's asymptotics need N large and T fixed; the "
             "bootstrap p-value and the confidence interval for gamma are NOT reliable here. "
             "Treat all threshold results as exploratory.")
    elif P.N < 30:
        warn(f"N = {P.N} is modest; interpret the confidence interval for gamma with caution.")
    if cfg["n_bootstrap"] < 500:
        warn("fewer than 500 bootstrap replications; the p-value is coarse.")
    if P.N < 10:
        print("\nBetter alternatives or complements when N is this small:")
        print("  - widen the sample (more emerging economies from the same World Bank database);")
        print("  - a simple interaction model: Y on X, q and X*q (needs no gamma, fewer assumptions);")
        print("  - fix the threshold from theory/regulation and test the slope difference with a dummy;")
        print("  - a time-series threshold model (Hansen 2000) country by country, if T is long enough.")


# =============================================================================
# 5. LINEAR ALGEBRA HELPERS AND LINEAR PANEL MODELS
# =============================================================================
def entity_means(A, codes, N):
    A = np.asarray(A, float)
    one_d = A.ndim == 1
    if one_d:
        A = A[:, None]
    cnt = np.bincount(codes, minlength=N).astype(float)
    out = np.empty((N, A.shape[1]))
    for j in range(A.shape[1]):
        out[:, j] = np.bincount(codes, weights=A[:, j], minlength=N) / cnt
    return out[:, 0] if one_d else out


def within(A, codes, N):
    """Within transformation: subtract each entity's own mean (this removes mu_i)."""
    A = np.asarray(A, float)
    m = entity_means(A, codes, N)
    return A - m[codes]


def cluster_cov(Xm, e, codes, N, Ainv):
    """Entity-clustered (Liang-Zeger) covariance with a small-sample factor."""
    n, K = Xm.shape
    S = np.zeros((N, K))
    for j in range(K):
        S[:, j] = np.bincount(codes, weights=Xm[:, j] * e, minlength=N)
    c = (N / (N - 1)) * ((n - 1) / max(n - K, 1)) if N > 1 else 1.0
    return c * Ainv @ (S.T @ S) @ Ainv


def coef_table(names, beta, se, dof, extra_se=None):
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    crit = stats.t.ppf(0.975, dof)
    tab = pd.DataFrame({"coef": beta, "std_err": se, "t_stat": t, "p_value": p,
                        "ci_low": beta - crit * se, "ci_high": beta + crit * se,
                        "sig": [stars(v) for v in p]}, index=names)
    if extra_se is not None:
        tab["se_clustered"] = extra_se
        tab["p_clustered"] = 2 * stats.t.sf(np.abs(beta / extra_se), dof)
    return tab


def fit_pooled(P: Panel) -> dict:
    W = np.hstack([np.ones((P.n, 1)), P.X, P.Z])
    K = W.shape[1]
    Ainv = np.linalg.pinv(W.T @ W)
    beta = Ainv @ (W.T @ P.y)
    fitted = W @ beta
    resid = P.y - fitted
    ssr = float(resid @ resid)
    tss = float(((P.y - P.y.mean()) ** 2).sum())
    dof = P.n - K
    s2 = ssr / dof
    se = np.sqrt(np.diag(s2 * Ainv))
    r2 = 1 - ssr / tss
    r2a = 1 - (1 - r2) * (P.n - 1) / dof
    Fstat = (r2 / (K - 1)) / ((1 - r2) / dof)
    names = ["const"] + P.reg_names
    return dict(name="Pooled OLS", names=names, beta=beta, table=coef_table(names, beta, se, dof),
                fitted=fitted, resid=resid, ssr=ssr, r2=r2, r2_adj=r2a, F=Fstat,
                F_p=float(stats.f.sf(Fstat, K - 1, dof)), rmse=math.sqrt(ssr / P.n),
                dof=dof, K=K, s2=s2)


def fit_fe(P: Panel) -> dict:
    W = np.hstack([P.X, P.Z])
    K = W.shape[1]
    ym = within(P.y, P.codes, P.N)
    Wm = within(W, P.codes, P.N)
    Ainv = np.linalg.pinv(Wm.T @ Wm)
    beta = Ainv @ (Wm.T @ ym)
    resid = ym - Wm @ beta
    ssr = float(resid @ resid)
    dof = P.n - P.N - K
    if dof <= 0:
        raise ValueError("Not enough observations for the number of regressors and countries.")
    s2 = ssr / dof
    se = np.sqrt(np.diag(s2 * Ainv))
    se_cl = np.sqrt(np.diag(cluster_cov(Wm, resid, P.codes, P.N, Ainv)))
    mu = entity_means(P.y, P.codes, P.N) - entity_means(W, P.codes, P.N) @ beta
    fitted = mu[P.codes] + W @ beta
    tss_w = float(ym @ ym)
    r2w = 1 - ssr / tss_w
    r2a = 1 - (1 - r2w) * (P.n - P.N) / dof
    Fstat = (r2w / K) / ((1 - r2w) / dof)
    names = P.reg_names
    return dict(name="Fixed Effects", names=names, beta=beta, mu=mu, fitted=fitted,
                resid_actual=P.y - fitted, resid=resid, ssr=ssr, dof=dof, K=K, s2=s2,
                r2=r2w, r2_adj=r2a, F=Fstat, F_p=float(stats.f.sf(Fstat, K, dof)),
                rmse=math.sqrt(ssr / P.n), Ainv=Ainv, Wm=Wm, ym=ym,
                r2_overall=float(np.corrcoef(P.y, fitted)[0, 1] ** 2),
                table=coef_table(names, beta, se, dof, se_cl), cov=s2 * Ainv)


def fit_re(P: Panel, fe: dict):
    """Swamy-Arora random effects (needed only for the Hausman and LM tests)."""
    N, n, K = P.N, P.n, fe["K"]
    if N - K - 1 < 1:
        return None
    W1 = np.hstack([np.ones((P.n, 1)), P.X, P.Z])
    s2e = fe["ssr"] / fe["dof"]
    yb = entity_means(P.y, P.codes, N)
    Wb = entity_means(W1, P.codes, N)
    bb = np.linalg.pinv(Wb.T @ Wb) @ (Wb.T @ yb)
    s2b = float(((yb - Wb @ bb) ** 2).sum()) / (N - K - 1)
    Th = N / np.sum(1.0 / P.counts)
    s2u = max(0.0, s2b - s2e / Th)
    theta = 1 - np.sqrt(s2e / (s2e + P.counts * s2u)) if s2u > 0 else np.zeros(N)
    ty = P.y - theta[P.codes] * yb[P.codes]
    tW = W1 - theta[P.codes][:, None] * Wb[P.codes]
    Ainv = np.linalg.pinv(tW.T @ tW)
    beta = Ainv @ (tW.T @ ty)
    return dict(beta=beta, cov=s2e * Ainv, sigma_u2=s2u, sigma_e2=s2e, theta=theta)


def test_row(name, h0, h1, stat, p, note=""):
    verdict = "reject H0" if p < 0.05 else "do not reject H0"
    return dict(test=name, H0=h0, H1=h1, statistic=stat, p_value=p, decision_5pct=verdict, note=note)


def print_tests(rows):
    for r in rows:
        print(f"\n* {r['test']}")
        print(f"    H0: {r['H0']}")
        print(f"    H1: {r['H1']}")
        print(f"    statistic = {r['statistic']:.4f},  p-value = {r['p_value']:.4f}  ->  {r['decision_5pct']} (5%)")
        if r["note"]:
            print(f"    note: {r['note']}")


# =============================================================================
# 6. STAGE 4 — DIAGNOSTICS
# =============================================================================
def vif_table(P: Panel) -> pd.DataFrame:
    W = np.hstack([P.X, P.Z])
    out = {}
    for j, nm in enumerate(P.reg_names):
        others = np.delete(W, j, axis=1)
        A = np.column_stack([np.ones(P.n), others]) if others.shape[1] else np.ones((P.n, 1))
        b = np.linalg.lstsq(A, W[:, j], rcond=None)[0]
        r = W[:, j] - A @ b
        r2 = 1 - float(r @ r) / float(((W[:, j] - W[:, j].mean()) ** 2).sum())
        out[nm] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.DataFrame({"VIF": out})


def wooldridge_serial(P: Panel):
    """Wooldridge (2002) test: in first differences, corr(e_t, e_t-1) = -0.5 under no serial correlation."""
    d = P.df.copy()
    W = np.hstack([P.X, P.Z])
    cols = P.reg_names
    for j, c in enumerate(cols):
        d[c + "_w"] = W[:, j]
    d["y_w"] = P.y
    d["code"] = P.codes
    d = d.sort_values(["code", "year"])
    g = d.groupby("code")
    ok = (g["year"].diff() == 1)
    dy = g["y_w"].diff()
    dX = np.column_stack([g[c + "_w"].diff() for c in cols])
    m = ok.to_numpy() & dy.notna().to_numpy() & ~np.isnan(dX).any(axis=1)
    if m.sum() < 15:
        return None
    b = np.linalg.lstsq(dX[m], dy.to_numpy()[m], rcond=None)[0]
    e = pd.Series(np.nan, index=d.index)
    e[m] = dy.to_numpy()[m] - dX[m] @ b
    e_l = e.groupby(d["code"]).shift(1)
    yrs_ok = (d["year"].groupby(d["code"]).diff() == 1) & (d["year"].groupby(d["code"]).diff().shift(1) == 1)
    mm = e.notna() & e_l.notna() & yrs_ok.fillna(False)
    if mm.sum() < 10:
        return None
    x = e_l[mm].to_numpy()
    yv = e[mm].to_numpy()
    codes = d.loc[mm, "code"].to_numpy()
    rho = (x @ yv) / (x @ x)
    u = yv - rho * x
    S = np.bincount(codes, weights=x * u)
    nc = len(np.unique(codes))
    var = (nc / max(nc - 1, 1)) * (S @ S) / (x @ x) ** 2
    t = (rho + 0.5) / math.sqrt(var)
    F = t * t
    return test_row("Wooldridge serial-correlation test",
                    "no first-order serial correlation in the idiosyncratic errors",
                    "first-order serial correlation present", F,
                    float(stats.f.sf(F, 1, max(nc - 1, 1))),
                    f"estimated rho(differenced) = {rho:.3f}; clustered by country, so unreliable when N is small")


def modified_wald_het(P: Panel, fe: dict):
    """Greene's modified Wald test for groupwise heteroskedasticity in the FE model."""
    e = fe["resid"]
    s2i = np.bincount(P.codes, weights=e ** 2, minlength=P.N) / P.counts
    s2 = fe["ssr"] / P.n
    v = np.zeros(P.N)
    for i in range(P.N):
        ei = e[P.codes == i]
        Ti = len(ei)
        v[i] = np.sum((ei ** 2 - s2i[i]) ** 2) / (Ti * (Ti - 1)) if Ti > 1 else np.nan
    W = float(np.nansum((s2i - s2) ** 2 / v))
    return test_row("Modified Wald test (groupwise heteroskedasticity, FE residuals)",
                    "equal error variance across countries", "error variance differs across countries",
                    W, float(stats.chi2.sf(W, P.N)),
                    "low power and size distortions when T is small")


def pesaran_cd(P: Panel, fe: dict):
    """Pesaran (2004) CD test for cross-sectional dependence in the FE residuals."""
    d = pd.DataFrame({"c": P.codes, "year": P.df["year"].to_numpy(), "e": fe["resid"]})
    M = d.pivot(index="year", columns="c", values="e")
    tot, cnt = 0.0, 0
    for i in range(M.shape[1]):
        for j in range(i + 1, M.shape[1]):
            pair = M.iloc[:, [i, j]].dropna()
            if len(pair) >= 3:
                r = np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1]
                tot += math.sqrt(len(pair)) * r
                cnt += 1
    if cnt == 0:
        return None
    cd = math.sqrt(2.0 / (P.N * (P.N - 1))) * tot
    return test_row("Pesaran CD test (cross-sectional dependence)",
                    "residuals are cross-sectionally independent",
                    "countries share common shocks", cd, float(2 * stats.norm.sf(abs(cd))),
                    "countries often share global shocks (e.g. 2008, 2020); a rejection is common and matters "
                    "for the bootstrap, which assumes independent draws")


_DF_CACHE: dict[int, np.ndarray] = {}


def _adf_stat(y):
    dy = np.diff(y)
    ylag = y[:-1]
    A = np.column_stack([np.ones_like(ylag), ylag])
    b = np.linalg.lstsq(A, dy, rcond=None)[0]
    r = dy - A @ b
    s2 = float(r @ r) / (len(dy) - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    return b[1] / math.sqrt(cov[1, 1])


def panel_unit_root(P: Panel, var: str, series: np.ndarray, rng, reps: int = 2000):
    """Fisher-type (Maddala-Wu) Dickey-Fuller test with SIMULATED p-values (no lags)."""
    pvals = []
    for i in range(P.N):
        y = series[P.codes == i]
        if len(y) < 8:
            continue
        T = len(y)
        if T not in _DF_CACHE:
            sims = np.array([_adf_stat(np.cumsum(rng.standard_normal(T))) for _ in range(reps)])
            _DF_CACHE[T] = np.sort(sims)
        pvals.append(np.searchsorted(_DF_CACHE[T], _adf_stat(y)) / reps)
    if len(pvals) < 2:
        return None
    stat = -2 * np.sum(np.log(np.clip(pvals, 1e-4, 1)))
    return test_row(f"Panel unit root, Fisher-DF on '{var}'",
                    "every country's series has a unit root (non-stationary)",
                    "at least one series is stationary", float(stat),
                    float(stats.chi2.sf(stat, 2 * len(pvals))),
                    "no lag augmentation, simulated critical values; very low power when T is about 15")


def run_linear_diagnostics(P: Panel, pooled: dict, fe: dict, rng) -> list:
    hr("STAGE 4 - DIAGNOSTICS AND PANEL-MODEL SELECTION")
    rows = []
    n, N, K = P.n, P.N, fe["K"]

    edu("Which linear panel model?", """
        Pooled OLS: one intercept for everybody (assumes countries are alike).
        Fixed effects: each country gets its own intercept mu_i (removes anything permanent
        about a country, e.g. banking structure) -- consistent even if mu_i is correlated with X.
        Random effects: mu_i treated as random noise uncorrelated with X (more efficient, riskier).
        Tests below choose between them. The threshold model that follows uses FIXED effects.
        """)

    F = ((pooled["ssr"] - fe["ssr"]) / (N - 1)) / (fe["ssr"] / fe["dof"])
    rows.append(test_row("F-test for fixed effects (Chow / redundant fixed effects)",
                         "all country intercepts are equal (pooled OLS is enough)",
                         "country intercepts differ (use fixed effects)", F,
                         float(stats.f.sf(F, N - 1, fe["dof"]))))
    rows.append(test_row("Overall F-test of the FE model", "all slope coefficients are zero",
                         "at least one slope differs from zero", fe["F"], fe["F_p"]))

    re = fit_re(P, fe)
    if re is not None:
        e = pooled["resid"]
        S = np.bincount(P.codes, weights=e, minlength=N)
        s2 = float(e @ e) / n
        lm = float((np.sum(S ** 2) - float(e @ e)) ** 2 / (2 * s2 ** 2 * np.sum(P.counts * (P.counts - 1))))
        rows.append(test_row("Breusch-Pagan LM test for random effects",
                             "variance of country effects is zero (pooled OLS is enough)",
                             "country effects exist (use RE or FE)", lm, float(stats.chi2.sf(lm, 1))))
        d = fe["beta"] - re["beta"][1:]
        V = fe["cov"] - re["cov"][1:, 1:]
        ev = np.linalg.eigvalsh((V + V.T) / 2)
        H = float(d @ np.linalg.pinv(V) @ d)
        note = "the covariance difference was not positive definite; interpret with care" if ev.min() < -1e-10 else ""
        rows.append(test_row("Hausman test (FE vs RE)",
                             "random effects are consistent (country effects uncorrelated with X)",
                             "country effects correlated with X (use fixed effects)", H,
                             float(stats.chi2.sf(H, K)), note))
    else:
        print("   (random-effects tests skipped: too few countries for the between regression)")

    print_tests(rows)
    het = modified_wald_het(P, fe)
    ser = wooldridge_serial(P)
    cd = pesaran_cd(P, fe)
    extra = [r for r in (het, ser, cd) if r is not None]
    print_tests(extra)
    rows += extra

    print("\n* Multicollinearity (VIF; above 5-10 is worrying)")
    vt = vif_table(P)
    print(fmt_table(vt, 2))

    print("\n* Stationarity (Fisher-type Dickey-Fuller, simulated p-values)")
    ur_rows = []
    for nm, arr in [(P.y_name, P.y), (P.q_name, P.q)] + [(c, P.X[:, j]) for j, c in enumerate(P.x_names)]:
        r = panel_unit_root(P, nm, arr, rng)
        if r:
            ur_rows.append(r)
    print_tests(ur_rows)
    rows += ur_rows
    if any(r["p_value"] > 0.05 for r in ur_rows):
        warn("some series may be non-stationary. With T about 15 the test has little power; "
             "if levels are persistent, consider first differences or a longer panel.")

    conclusions = []
    if rows[0]["p_value"] < 0.05:
        conclusions.append("Country intercepts differ: use fixed effects, not pooled OLS.")
    else:
        conclusions.append("No evidence that country intercepts differ; but FE remains the safe choice here.")
    for r in extra:
        if "heteroskedasticity" in r["test"] and r["p_value"] < 0.05:
            conclusions.append("Heteroskedasticity detected: prefer clustered SEs and the wild bootstrap.")
        if "serial" in r["test"] and r["p_value"] < 0.05:
            conclusions.append("Serial correlation detected: the i.i.d. bootstrap is questionable; "
                               "lag the regressors, check the wild bootstrap, treat p-values cautiously.")
        if "Pesaran" in r["test"] and r["p_value"] < 0.05:
            conclusions.append("Cross-sectional dependence detected: effective sample is smaller than N*T.")
    print("\nModel-selection summary:")
    for c in conclusions:
        print("  - " + c)
    return rows


# =============================================================================
# 7. STAGES 5-6 — PANEL THRESHOLD REGRESSION (Hansen, 1999)
# =============================================================================
class PanelThreshold:
    """
    Hansen (1999) panel threshold regression, one threshold, fixed effects.

    For a candidate gamma the regressors are  [X*1(q<=g), X*1(q>g), Z]  (Z = same-slope
    controls). The WITHIN transformation is applied to this whole constructed matrix
    (not to X first), then OLS gives SSR(g). The estimate is gamma_hat = argmin SSR(g).

    Speed trick: since demeaning is a projection M, X_g' y_m = Raw_g' y_m, so for every
    candidate we only need cumulative sums of x*y over observations sorted by q. That makes
    the grid search and every bootstrap replication cheap.
    """

    def __init__(self, P: Panel, trim=0.15, max_grid=300):
        self.P = P
        self.k, self.kz = P.X.shape[1], P.Z.shape[1]
        self.K = 2 * self.k + self.kz
        self.y_m = within(P.y, P.codes, P.N)
        W = np.hstack([P.X, P.Z])
        self.W_m = within(W, P.codes, P.N)
        self.A0_inv = np.linalg.pinv(self.W_m.T @ self.W_m)
        self.order = np.argsort(P.q, kind="mergesort")
        self.q_sorted = P.q[self.order]
        self.X_sorted = P.X[self.order]
        self.min_obs = max(int(math.ceil(trim * P.n)), self.k + 2)

        cand = []
        for g in np.unique(P.q):
            n_low = int(np.searchsorted(self.q_sorted, g, side="right"))
            if n_low >= self.min_obs and P.n - n_low >= self.min_obs:
                cand.append(g)
        if not cand:
            raise ValueError("No admissible threshold candidate: every candidate leaves a regime "
                             "with too few observations. Lower 'trim' or use more data.")
        if len(cand) > max_grid:
            idx = np.linspace(0, len(cand) - 1, max_grid).round().astype(int)
            cand = [cand[i] for i in np.unique(idx)]
        self.cand = np.array(cand)
        self.cut = np.searchsorted(self.q_sorted, self.cand, side="right") - 1
        self.A_inv = []
        for g in self.cand:
            Xg = self._design(g)
            self.A_inv.append(np.linalg.pinv(Xg.T @ Xg))

    def _raw(self, g):
        d = (self.P.q <= g).astype(float)[:, None]
        return np.hstack([self.P.X * d, self.P.X * (1 - d), self.P.Z])

    def _design(self, g):
        return within(self._raw(g), self.P.codes, self.P.N)

    def ssr_linear(self, y_m):
        b = self.W_m.T @ y_m
        return float(y_m @ y_m - b @ self.A0_inv @ b)

    def ssr_profile(self, y_m):
        yy = float(y_m @ y_m)
        cs = np.cumsum(self.X_sorted * y_m[self.order][:, None], axis=0)
        tot = cs[-1]
        zy = self.P.Z.T @ y_m
        out = np.empty(len(self.cand))
        for i, idx in enumerate(self.cut):
            low = cs[idx]
            b = np.concatenate([low, tot - low, zy])
            out[i] = yy - b @ self.A_inv[i] @ b
        return out

    def fit(self, alpha=0.05) -> dict:
        P, N, n = self.P, self.P.N, self.P.n
        prof = self.ssr_profile(self.y_m)
        i = int(np.argmin(prof))
        gamma, ssr1 = float(self.cand[i]), float(prof[i])
        Xg = self._design(gamma)
        beta = self.A_inv[i] @ (Xg.T @ self.y_m)
        resid = self.y_m - Xg @ beta
        dof = n - N - self.K
        s2 = float(resid @ resid) / dof
        cov = s2 * self.A_inv[i]
        se = np.sqrt(np.diag(cov))
        se_cl = np.sqrt(np.diag(cluster_cov(Xg, resid, P.codes, N, self.A_inv[i])))

        k, kz = self.k, self.kz
        names = ([f"{x} | q <= gamma" for x in P.x_names] + [f"{x} | q > gamma" for x in P.x_names]
                 + list(P.z_names))
        tab = coef_table(names, beta, se, dof, se_cl)

        diff_rows = []
        for j, x in enumerate(P.x_names):
            d = beta[j] - beta[k + j]
            v = cov[j, j] + cov[k + j, k + j] - 2 * cov[j, k + j]
            t = d / math.sqrt(v) if v > 0 else np.nan
            diff_rows.append({"variable": x, "coef_regime1": beta[j], "coef_regime2": beta[k + j],
                              "difference": d, "t_stat": t,
                              "p_value": float(2 * stats.t.sf(abs(t), dof)) if v > 0 else np.nan})
        diff = pd.DataFrame(diff_rows).set_index("variable")

        low = P.q <= gamma
        raw = self._raw(gamma)
        mu = entity_means(P.y, P.codes, N) - entity_means(raw, P.codes, N) @ beta
        fitted = mu[P.codes] + raw @ beta

        sigma2_h = ssr1 / (n - N)             # Hansen's sigma_hat^2 (n(T-1) in a balanced panel)
        ssr0 = self.ssr_linear(self.y_m)
        F1 = (ssr0 - ssr1) / sigma2_h
        lr = (prof - ssr1) / sigma2_h
        c = -2 * math.log(1 - math.sqrt(1 - alpha))
        inside = self.cand[lr <= c]
        ci = (float(inside.min()), float(inside.max()))
        tss_w = float(self.y_m @ self.y_m)

        return dict(gamma=gamma, ssr=ssr1, ssr_linear=ssr0, F=F1, sigma2=sigma2_h,
                    cand=self.cand, ssr_profile=prof, lr_profile=lr, lr_crit=c, ci=ci, alpha=alpha,
                    beta=beta, names=names, table=tab, diff=diff, dof=dof, K=self.K,
                    n_low=int(low.sum()), n_high=int((~low).sum()),
                    ent_low=int(len(np.unique(P.codes[low]))), ent_high=int(len(np.unique(P.codes[~low]))),
                    mu=mu, fitted=fitted, resid=resid, rmse=math.sqrt(ssr1 / n),
                    r2=1 - ssr1 / tss_w, r2_adj=1 - (1 - (1 - ssr1 / tss_w)) * (n - N) / dof,
                    n_params=self.K + 1)

    def bootstrap(self, lin: dict, B: int, scheme: str, rng) -> np.ndarray:
        """Hansen's bootstrap for the sup-F test under H0 (linear FE model).

        Regressors and q are held fixed. Pseudo-errors are drawn from the (rescaled) H0
        residuals ("iid") or are the residuals times random +/-1 ("wild"). Because the model
        is demeaned, y*_m = W_m b0 + M e*.
        """
        P, n, N = self.P, self.P.n, self.P.N
        fitted0 = self.W_m @ lin["beta"]
        res = lin["resid"] * math.sqrt(n / (n - N - lin["K"]))
        out = np.empty(B)
        for b in range(B):
            if scheme == "wild":
                e = res * rng.choice([-1.0, 1.0], size=n)
            else:
                e = rng.choice(res, size=n, replace=True)
            yb = fitted0 + within(e, P.codes, N)
            ssr0 = self.ssr_linear(yb)
            ssr1 = float(self.ssr_profile(yb).min())
            out[b] = (ssr0 - ssr1) / (ssr1 / (n - N))
        return out


def regime_tests_summary(thr: dict, boot: np.ndarray, scheme: str) -> dict:
    p = float((1 + np.sum(boot >= thr["F"])) / (1 + len(boot)))
    crit = {f"{int(100 * a)}%": float(np.quantile(boot, a)) for a in (0.90, 0.95, 0.99)}
    return dict(F=thr["F"], p_value=p, crit=crit, B=len(boot), scheme=scheme)


# =============================================================================
# 8. COMPARISON AND INTERPRETATION
# =============================================================================
def aic_bic(ssr, n, p):
    return n * math.log(ssr / n) + 2 * p, n * math.log(ssr / n) + p * math.log(n)


def comparison_table(P, fe, thr, test) -> pd.DataFrame:
    a0, b0 = aic_bic(fe["ssr"], P.n, fe["K"])
    a1, b1 = aic_bic(thr["ssr"], P.n, thr["n_params"])
    rows = {
        "Number of slope parameters (incl. gamma)": (fe["K"], thr["n_params"]),
        "SSR (within)": (fe["ssr"], thr["ssr"]),
        "RMSE": (fe["rmse"], thr["rmse"]),
        "R-squared within": (fe["r2"], thr["r2"]),
        "AIC (heuristic)": (a0, a1),
        "BIC (heuristic)": (b0, b1),
        "Threshold gamma": (np.nan, thr["gamma"]),
        "Obs in regime 1 / regime 2": ("-", f"{thr['n_low']} / {thr['n_high']}"),
        "Sup-F statistic": (np.nan, test["F"]),
        "Bootstrap p-value": (np.nan, test["p_value"]),
    }
    return pd.DataFrame(rows, index=["Linear FE", "Threshold FE"]).T


def interpret_linear(P, fe) -> list[str]:
    out = []
    for nm, r in fe["table"].iterrows():
        direction = "positive" if r["coef"] > 0 else "negative"
        sig = ("statistically significant at the 1% level" if r["p_value"] < 0.01 else
               "significant at the 5% level" if r["p_value"] < 0.05 else
               "significant at the 10% level" if r["p_value"] < 0.10 else
               "not statistically significant")
        out.append(f"{nm}: coefficient {r['coef']:+.4f} ({direction}, {sig}). A one-unit increase in {nm} "
                   f"is associated with a {r['coef']:+.4f}-unit change in {P.y_name}, holding the other "
                   f"regressors and the country effect constant. This is an association, not proof of causation.")
    return out


def interpret_threshold(P, thr, test) -> list[str]:
    g = thr["gamma"]
    out = [f"The estimated threshold in {P.q_name} is gamma = {g:.3f} "
           f"(approx. {100 * (1 - thr['alpha']):.0f}% interval {thr['ci'][0]:.3f} to {thr['ci'][1]:.3f}). "
           f"Regime 1 ({P.q_name} <= {g:.3f}) has {thr['n_low']} observations from {thr['ent_low']} countries; "
           f"regime 2 ({P.q_name} > {g:.3f}) has {thr['n_high']} observations from {thr['ent_high']} countries."]
    for x, r in thr["diff"].iterrows():
        p1 = thr["table"].loc[f"{x} | q <= gamma", "p_value"]
        p2 = thr["table"].loc[f"{x} | q > gamma", "p_value"]
        same_sign = (r["coef_regime1"] > 0) == (r["coef_regime2"] > 0)
        stronger = "regime 1" if abs(r["coef_regime1"]) > abs(r["coef_regime2"]) else "regime 2"
        shape = (f"The sign is the same in both regimes; the association is stronger in {stronger}."
                 if same_sign else "The sign changes between regimes.")
        sig = ("The difference between regimes is statistically significant"
               if r["p_value"] < 0.05 else "The difference between regimes is not statistically significant")
        out.append(
            f"{x}: a one-unit rise is associated with a {r['coef_regime1']:+.4f} change in {P.y_name} when "
            f"{P.q_name} <= {g:.3f} (p = {p1:.3f}) and a {r['coef_regime2']:+.4f} change when "
            f"{P.q_name} > {g:.3f} (p = {p2:.3f}). {shape} {sig} "
            f"(p = {r['p_value']:.3f}, conditional on the estimated gamma).")
    return out


def decide(test, N) -> str:
    p = test["p_value"]
    if N < 10:
        return ("INCONCLUSIVE: the bootstrap p-value is " + f"{p:.3f}" + " but N is far too small for it to be reliable. "
                "Treat the threshold as a hypothesis for a larger sample, not a finding.")
    if p < 0.05:
        return f"The data support a threshold effect (bootstrap p = {p:.3f} < 0.05)."
    if p < 0.10:
        return f"Weak evidence for a threshold effect (bootstrap p = {p:.3f}, significant only at 10%)."
    return f"No reliable evidence of a threshold effect (bootstrap p = {p:.3f}). The linear model is enough."


def managerial_implications(P, thr, test) -> list[str]:
    p = test["p_value"]
    if P.N < 10 or p >= 0.10:
        return ["The estimates do not provide statistically reliable regime-specific evidence, so no "
                "regime-specific policy recommendation is made. What the estimates do support is the "
                "linear association shown in the baseline table."]
    g = thr["gamma"]
    out = [f"Early-warning monitoring: track {P.q_name} against the estimated threshold ({g:.2f}); the "
           f"sensitivity of {P.y_name} to the other variables differs on either side of it."]
    for x, r in thr["diff"].iterrows():
        if r["p_value"] < 0.10:
            stronger = "regime 1" if abs(r["coef_regime1"]) > abs(r["coef_regime2"]) else "regime 2"
            out.append(f"Stress testing: calibrate shocks to {x} separately by regime; the estimated effect is larger in "
                       f"{stronger} (diff. p = {r['p_value']:.3f}).")
    out.append("Credit-risk monitoring and risk appetite: use the thresholds as a screening signal, "
               "not as a mechanical rule; the estimate comes from a small macro sample and has sampling uncertainty.")
    return out


# =============================================================================
# 9. VISUALISATION
# =============================================================================
def plot_ols(P, pooled, fe):
    try:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        ax[0].scatter(P.y, fe["fitted"], s=22, c=CL, alpha=.8)
        lo, hi = min(P.y.min(), fe["fitted"].min()), max(P.y.max(), fe["fitted"].max())
        ax[0].plot([lo, hi], [lo, hi], "--", c=CGREY)
        ax[0].set(xlabel=f"actual {P.y_name}", ylabel="fitted (FE)", title="Actual vs fitted values")
        ax[1].scatter(fe["fitted"], fe["resid_actual"], s=22, c=CL, alpha=.8)
        ax[1].axhline(0, c=CGREY, ls="--")
        ax[1].set(xlabel="fitted (FE)", ylabel="residual = actual - fitted", title="Residuals vs fitted")
        for i, ctry in enumerate(P.entities):
            m = P.codes == i
            ax[2].plot(P.df["year"][m], fe["resid_actual"][m], marker="o", ms=3, lw=1.2, label=ctry)
        ax[2].axhline(0, c=CGREY, ls="--")
        ax[2].set(xlabel="year", title="Residuals over time")
        ax[2].legend(fontsize=7)
        for a in ax:
            a.grid(alpha=.3)
        fig.tight_layout()
        savefig(fig, "02_ols_actual_fitted_residuals.png")
    except Exception as e:
        warn(f"OLS plot skipped: {e}")


def plot_ssr(thr):
    ssr_linear = thr["ssr_linear"]
    try:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        ax[0].plot(thr["cand"], thr["ssr_profile"], c=C1, lw=2)
        ax[0].axhline(ssr_linear, c=CL, ls="--", label=f"linear FE SSR = {ssr_linear:,.1f}")
        ax[0].axvline(thr["gamma"], c=C2, ls="-", label=f"gamma_hat = {thr['gamma']:.3f}")
        ax[0].axvspan(thr["ci"][0], thr["ci"][1], color=C2, alpha=.15, label="confidence interval")
        ax[0].set(xlabel="candidate threshold gamma", ylabel="SSR(gamma)",
                  title="Grid search: the minimum of SSR(gamma) is the estimate")
        ax[0].legend(fontsize=8)
        ax[1].plot(thr["cand"], thr["lr_profile"], c=C1, lw=2)
        ax[1].axhline(thr["lr_crit"], c=C2, ls="--", label=f"critical value {thr['lr_crit']:.2f}")
        ax[1].set_ylim(-0.5, max(8 * thr["lr_crit"], 20))   # zoom on the region near the minimum
        ax[1].set(xlabel="candidate threshold gamma", ylabel="LR(gamma)  (zoomed)",
                  title="LR statistic: candidates below the dashed line are plausible")
        ax[1].legend(fontsize=8)
        for a in ax:
            a.grid(alpha=.3)
        fig.tight_layout()
        savefig(fig, "04_ssr_vs_threshold.png")
    except Exception as e:
        warn(f"SSR plot skipped: {e}")


def partial_resid_data(P, fe, thr, focal: int):
    """Component-plus-residual data so a many-variable model can be drawn in 2-D.

    point_it = mu_bar + b_f*x_f,it + sum_{j!=f} b_j*mean(x_j) + e_it
    i.e. every other regressor is held at its sample mean and the country effect at its average.
    """
    W = np.hstack([P.X, P.Z])
    wbar = W.mean(axis=0)
    k = P.X.shape[1]
    others = [j for j in range(W.shape[1]) if j != focal]

    b = fe["beta"]
    adj = (W[:, others] - wbar[others]) @ b[others]
    pr_lin = P.y - fe["mu"][P.codes] + fe["mu"].mean() - adj
    line_lin = lambda x: fe["mu"].mean() + b[focal] * x + wbar[others] @ b[others]

    d = (P.q <= thr["gamma"])
    bt = thr["beta"]
    Cmat = np.empty_like(W)
    Cmat[:, :k] = np.where(d[:, None], bt[:k], bt[k:2 * k])
    Cmat[:, k:] = bt[2 * k:]
    adj_t = ((W[:, others] - wbar[others]) * Cmat[:, others]).sum(axis=1)
    pr_thr = P.y - thr["mu"][P.codes] + thr["mu"].mean() - adj_t
    cl = {True: bt[focal], False: bt[k + focal]}
    cother = {True: np.concatenate([bt[:k], bt[2 * k:]]), False: np.concatenate([bt[k:2 * k], bt[2 * k:]])}
    line_thr = lambda x, reg: thr["mu"].mean() + cl[reg] * x + wbar[others] @ cother[reg][others]
    return pr_lin, pr_thr, line_lin, line_thr, d


def plot_before_after(P, fe, thr, focal=0):
    """BEFORE: one slope for everyone.  AFTER: one slope per regime."""
    try:
        xf = P.X[:, focal]
        pr_lin, pr_thr, line_lin, line_thr, d = partial_resid_data(P, fe, thr, focal)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
        xs = np.linspace(xf.min(), xf.max(), 50)
        for a, pr, title in ((ax[0], pr_lin, "BEFORE: one linear slope for all observations"),
                             (ax[1], pr_thr, "AFTER: separate slopes below / above the threshold")):
            a.scatter(xf[d], pr[d], s=26, c=C1, alpha=.75, label=f"regime 1: {P.q_name} <= {thr['gamma']:.2f}")
            a.scatter(xf[~d], pr[~d], s=26, c=C2, alpha=.75, label=f"regime 2: {P.q_name} > {thr['gamma']:.2f}")
            a.set(xlabel=P.x_names[focal], title=title)
            a.grid(alpha=.3)
        ax[0].set_ylabel(f"{P.y_name} (other variables and country effects held fixed)")
        ax[0].plot(xs, line_lin(xs), c="black", lw=2.6, label=f"single slope = {fe['beta'][focal]:+.3f}")
        for reg, col, msk in ((True, C1, d), (False, C2, ~d)):
            if msk.sum() > 1:
                xr = np.linspace(xf[msk].min(), xf[msk].max(), 30)
                ax[1].plot(xr, line_thr(xr, reg), c=col, lw=2.8,
                           label=f"slope = {thr['beta'][focal if reg else P.X.shape[1] + focal]:+.3f}")
        ax[0].legend(fontsize=8)
        ax[1].legend(fontsize=8)
        fig.suptitle("Same points, different model: does one slope fit every regime?", y=1.02)
        fig.tight_layout()
        fig.text(0.5, -0.04, "Other regressors are held at their sample means. A vertical gap between the regimes "
                 "comes from those other variables having different slopes in each regime.",
                 ha="center", fontsize=8.5, color="#5d4442")
        savefig(fig, "05_before_after_threshold.png")
    except Exception as e:
        warn(f"before/after plot skipped: {e}")


def plot_regime_coefs(P, fe, thr):
    try:
        k = P.X.shape[1]
        tab = thr["table"]
        fig, ax = plt.subplots(figsize=(max(6, 2.6 * k), 4.6))
        xs = np.arange(k)
        w = 0.32
        for off, col, lab, suffix in ((-w / 2, C1, "regime 1", " | q <= gamma"), (w / 2, C2, "regime 2", " | q > gamma")):
            rows = [tab.loc[x + suffix] for x in P.x_names]
            ax.bar(xs + off, [r["coef"] for r in rows], w, color=col, label=lab,
                   yerr=[[r["coef"] - r["ci_low"] for r in rows], [r["ci_high"] - r["coef"] for r in rows]],
                   capsize=4)
        ax.scatter(xs, fe["beta"][:k], marker="D", c="black", zorder=5, label="linear FE")
        ax.axhline(0, c=CGREY, lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(P.x_names)
        ax.set(ylabel="coefficient (95% CI)", title="Coefficients by regime vs the single linear coefficient")
        ax.legend()
        ax.grid(alpha=.3, axis="y")
        savefig(fig, "06_regime_coefficients.png")
    except Exception as e:
        warn(f"coefficient plot skipped: {e}")


def plot_bootstrap(thr, boot, test):
    try:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.hist(boot, bins=40, color=CL, alpha=.8)
        ax.axvline(thr["F"], c=C1, lw=2.5, label=f"observed sup-F = {thr['F']:.2f}")
        ax.axvline(test["crit"]["95%"], c=C2, ls="--", label=f"bootstrap 95% value = {test['crit']['95%']:.2f}")
        ax.set(xlabel="sup-F in bootstrap samples generated WITHOUT a threshold",
               title=f"Is the observed sup-F unusual? bootstrap p = {test['p_value']:.3f}")
        ax.legend()
        savefig(fig, "07_bootstrap_distribution.png")
    except Exception as e:
        warn(f"bootstrap plot skipped: {e}")


def plot_comparison(cmp_tab):
    try:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
        for a, row in zip(ax, ["SSR (within)", "RMSE", "BIC (heuristic)"]):
            vals = [float(v) for v in cmp_tab.loc[row]]
            a.bar(["Linear FE", "Threshold FE"], vals, color=[CL, C1])
            a.set_title(row)
            for i, v in enumerate(vals):
                a.text(i, v, f"{v:,.2f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle("Linear vs threshold (lower is better; a lower SSR alone proves nothing, see the sup-F test)", y=1.04)
        fig.tight_layout()
        savefig(fig, "08_model_comparison.png")
    except Exception as e:
        warn(f"comparison plot skipped: {e}")


def plot_scatter_regimes(P, thr):
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        d = P.q <= thr["gamma"]
        for j, x in enumerate(P.x_names[:2]):
            ax[j].scatter(P.X[d, j], P.y[d], c=C1, s=22, alpha=.75, label="regime 1")
            ax[j].scatter(P.X[~d, j], P.y[~d], c=C2, s=22, alpha=.75, label="regime 2")
            ax[j].set(xlabel=x, ylabel=P.y_name, title=f"Raw scatter of {P.y_name} vs {x}")
            ax[j].grid(alpha=.3)
            ax[j].legend(fontsize=8)
        for a in ax[len(P.x_names[:2]):]:
            a.axis("off")
        fig.tight_layout()
        savefig(fig, "03_scatter_x_y_by_regime.png")
    except Exception as e:
        warn(f"scatter plot skipped: {e}")


# =============================================================================
# 10. EXPORT
# =============================================================================
def export_all(P, pooled, fe, tests, thr, test, boot, cmp_tab, text_blocks):
    out = CONFIG["output_dir"]
    os.makedirs(out, exist_ok=True)
    tabs = {
        "ols_pooled": pooled["table"], "ols_fixed_effects": fe["table"],
        "diagnostic_tests": pd.DataFrame(tests), "threshold_coefficients": thr["table"],
        "regime_differences": thr["diff"], "model_comparison": cmp_tab,
        "ssr_profile": pd.DataFrame({"gamma": thr["cand"], "SSR": thr["ssr_profile"], "LR": thr["lr_profile"]}),
    }
    for nm, t in tabs.items():
        t.to_csv(os.path.join(out, f"{nm}.csv"))
    try:
        with pd.ExcelWriter(os.path.join(out, "threshold_regression_results.xlsx"), engine="openpyxl") as xw:
            for nm, t in tabs.items():
                t.to_excel(xw, sheet_name=nm[:31])
            pd.DataFrame({"report": text_blocks}).to_excel(xw, sheet_name="interpretation", index=False)
        print(f"   saved Excel workbook: {os.path.join(out, 'threshold_regression_results.xlsx')}")
    except Exception as e:
        warn(f"Excel export skipped ({e}); CSV files were still written.")

    payload = {
        "meta": {"y": P.y_name, "x": P.x_names, "controls": P.z_names, "q": P.q_name, "N": P.N, "n_obs": P.n,
                 "countries": P.entities, "bootstrap_replications": test["B"], "bootstrap_scheme": test["scheme"]},
        "linear_fe": {"coefficients": fe["table"].reset_index().rename(columns={"index": "variable"}).to_dict("records"),
                      "ssr": fe["ssr"], "rmse": fe["rmse"], "r2_within": fe["r2"]},
        "threshold": {"gamma": thr["gamma"], "ci": thr["ci"], "ssr": thr["ssr"], "rmse": thr["rmse"],
                      "r2_within": thr["r2"], "n_regime1": thr["n_low"], "n_regime2": thr["n_high"],
                      "coefficients": thr["table"].reset_index().rename(columns={"index": "variable"}).to_dict("records"),
                      "differences": thr["diff"].reset_index().to_dict("records"),
                      "grid": {"gamma": thr["cand"], "ssr": thr["ssr_profile"], "lr": thr["lr_profile"]}},
        "test": {"sup_f": test["F"], "p_value": test["p_value"], "critical_values": test["crit"]},
        "comparison": cmp_tab.astype(str).reset_index().to_dict("records"),
    }
    with open(os.path.join(out, "threshold_results.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_blocks))
    print(f"   saved tables, JSON and report.txt in '{out}/'")


# =============================================================================
# 11. TOY EXAMPLE (10 rows, matches the website walkthrough)
# =============================================================================
def toy_example():
    hr("TOY EXAMPLE - 10 borrowers, no panel, worked by hand")
    q = np.array([20, 25, 30, 35, 40, 50, 55, 60, 65, 70], float)
    x = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], float)
    y = np.array([1.0, 1.3, 1.5, 1.9, 2.1, 3.0, 5.4, 8.0, 10.4, 13.0])

    def line(xv, yv):
        sxx = ((xv - xv.mean()) ** 2).sum()
        sxy = ((xv - xv.mean()) * (yv - yv.mean())).sum()
        b = sxy / sxx
        a = yv.mean() - b * xv.mean()
        return a, b, float(((yv - a - b * xv) ** 2).sum())

    edu("OLS by hand", """
        Slope b = Sxy / Sxx,  intercept a = mean(y) - b * mean(x).
        This choice of (a, b) makes the sum of squared residuals (SSR) as small as possible
        for ONE straight line. 'Residual' = actual y - fitted y.
        """)
    a0, b0, ssr0 = line(x, y)
    print(f"One line for all 10 rows : y = {a0:.2f} + {b0:.2f} x    SSR = {ssr0:.1f}")

    rows = []
    for g in (25, 30, 35, 40, 45, 50, 55, 60):
        L, R = q <= g, q > g
        aL, bL, sL = line(x[L], y[L])
        aR, bR, sR = line(x[R], y[R])
        rows.append({"gamma (q)": g, "obs left": int(L.sum()), "slope left": bL,
                     "slope right": bR, "SSR total": sL + sR})
    t = pd.DataFrame(rows)
    print("\nGrid search (each row = one candidate threshold):")
    print(t.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    best = t.loc[t["SSR total"].idxmin()]
    print(f"\nBest threshold: gamma = {best['gamma (q)']:.0f}  (SSR {best['SSR total']:.3f} vs {ssr0:.1f} with one line)")
    edu("Why the SSR drops so much", """
        With the wrong threshold, one line still has to serve observations from two different
        'worlds', so it misses a lot. At the right threshold each line serves one world.
        More flexible models ALWAYS lower the SSR a little, which is why real data need the
        sup-F bootstrap test (stage 6) before believing a threshold exists.
        """)
    try:
        g = best["gamma (q)"]
        L, R = q <= g, q > g
        aL, bL, _ = line(x[L], y[L])
        aR, bR, _ = line(x[R], y[R])
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        ax[0].scatter(x, y, c=CGREY, s=40)
        xs = np.linspace(0.5, 5.5, 20)
        ax[0].plot(xs, a0 + b0 * xs, c="black", lw=2.4)
        for xi, yi in zip(x, y):
            ax[0].plot([xi, xi], [yi, a0 + b0 * xi], c="#b4b2a9", lw=1)
        ax[0].set(title=f"BEFORE: one line, SSR = {ssr0:.1f}", xlabel="X: rise in instalment", ylabel="Y: rise in risk")
        ax[1].plot(t["gamma (q)"], t["SSR total"], marker="o", c=C1)
        ax[1].axvline(g, c=C2, ls="--")
        ax[1].set(title="Grid search: SSR for each candidate threshold", xlabel="candidate gamma (q, %)", ylabel="SSR")
        ax[2].scatter(x[L], y[L], c=C1, s=40, label="q <= gamma")
        ax[2].scatter(x[R], y[R], c=C2, s=40, label="q > gamma")
        ax[2].plot(xs, aL + bL * xs, c=C1, lw=2.4)
        ax[2].plot(xs, aR + bR * xs, c=C2, lw=2.4)
        ax[2].set(title=f"AFTER: two lines, SSR = {best['SSR total']:.3f}", xlabel="X: rise in instalment")
        ax[2].legend()
        for a in ax:
            a.grid(alpha=.3)
        fig.tight_layout()
        savefig(fig, "00_toy_before_after.png")
    except Exception as e:
        warn(f"toy plot skipped: {e}")


# =============================================================================
# 12. SYNTHETIC DEMO DATA  (clearly NOT real data)
# =============================================================================
def make_demo_panel(N=30, T=17, seed=1) -> pd.DataFrame:
    """Synthetic panel with a KNOWN threshold in GDP_Growth at 3.0.

    GDP growth = country level + a business cycle shared by all countries (with 2009 and
    2020 style crashes) + country-specific noise, so q moves over time within each country.
    The slope of Unemployment and Inflation on NPL differs below / above 3.0.
    """
    rng = np.random.default_rng(seed)
    common = np.zeros(T)
    for t in range(1, T):
        common[t] = 0.5 * common[t - 1] + rng.normal(0, 0.9)
    for yr, hit in ((2009, -2.5), (2020, -3.0)):
        if 0 <= yr - 2005 < T:
            common[yr - 2005] += hit
    rows = []
    for i in range(N):
        mu = rng.normal(4, 1.5)
        base = 4 + rng.normal(0, 0.6)
        idio = np.zeros(T)
        for t in range(1, T):
            idio[t] = 0.4 * idio[t - 1] + rng.normal(0, 0.9)
        gdp = base + common + idio
        infl = np.abs(3 + rng.normal(0, 1.0) + rng.normal(0, 1.4, T))
        unemp = np.abs(5 + rng.normal(0, 1.0) + rng.normal(0, 0.8, T))
        for t in range(T):
            low = gdp[t] <= 3.0
            b_u, b_i = (0.9, 0.10) if low else (0.2, 0.40)
            npl = mu + b_u * unemp[t] + b_i * infl[t] + rng.normal(0, 0.7)
            rows.append({"country": f"Country_{i + 1:02d}", "year": 2005 + t, "NPL_Ratio": npl,
                         "Inflation": infl[t], "Unemployment": unemp[t], "GDP_Growth": gdp[t]})
    return pd.DataFrame(rows)


# =============================================================================
# 13. MAIN PIPELINE
# =============================================================================
def run(cfg: dict, stage: int = 8, demo: bool = False) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    results: dict = {}

    # ---------------- STAGE 1
    if demo:
        warn("SYNTHETIC DEMO DATA. The 'true' threshold in GDP_Growth is 3.0. Do not mistake this for evidence.")
        df = make_demo_panel()
        meta = {c: {"code": None, "original_name": c, "description": "synthetic demo variable"}
                for c in df.columns if c not in ("country", "year")}
        cfg = dict(cfg, y="NPL_Ratio", x=["Inflation", "Unemployment"], q="GDP_Growth", controls=[])
    else:
        paths = cfg["data_path"] if isinstance(cfg["data_path"], (list, tuple)) else [cfg["data_path"]]
        df, meta = None, {}
        for pth in paths:
            print(f"   reading {pth}")
            d_i, m_i = to_panel_frame(read_table(pth))
            meta.update(m_i)
            if df is None:
                df = d_i
            else:  # merge extra files (e.g. GFDD + WDI) on country-year
                d_i = d_i.drop(columns=[c for c in ("country_code",) if c in d_i.columns])
                dup = [c for c in d_i.columns if c in df.columns and c not in ("country", "year")]
                df = df.merge(d_i.drop(columns=dup), on=["country", "year"], how="outer")
        var_cols0 = [c for c in df.columns if c not in ("country", "country_code", "year")]
        has_vals = df.dropna(subset=var_cols0, how="all")
        print(f"   file covers: {df['country'].nunique()} countries, years {df['year'].min()}-{df['year'].max()} "
              f"(years that actually contain values: {has_vals['year'].min()}-{has_vals['year'].max()}), "
              f"{len(var_cols0)} variables")
        df = select_countries_years(df, cfg["countries"], cfg["year_start"], cfg["year_end"])
        if df.empty or df.drop(columns=["country", "country_code", "year"], errors="ignore").notna().sum().sum() == 0:
            raise SystemExit(
                f"\nNo usable data remain for years {cfg['year_start']}-{cfg['year_end']}.\n"
                f"The file only contains values for {has_vals['year'].min()}-{has_vals['year'].max()}. "
                "Re-download the data with the right time range (see DATA DOWNLOAD at the top of this file), "
                "or change year_start / year_end in CONFIG.")
    explore_data(df, meta)
    if stage <= 1:
        return results

    # ---------------- STAGE 2
    hr("STAGE 2 - CHOOSE Y, X AND q")
    y, xs, zs, q = choose_variables(df, meta, cfg)
    for note in validate_selection(df, y, xs, zs, q):
        warn(note)
    print(f"\nY (dependent)          : {y}")
    print(f"X (regime-dependent)   : {xs}")
    print(f"Controls (same slope)  : {zs or '-'}")
    print(f"q (threshold variable) : {q}")
    edu("How to choose the roles", """
        Y should be the outcome you want to explain (a credit-risk indicator).
        X are the drivers whose EFFECT you suspect changes across regimes.
        q is the variable that decides the regime; pick it from economic reasoning (e.g. the
        business cycle), never by trying every variable and keeping the best SSR (data mining).
        Regressors must be exogenous: do not put a lagged Y among the X (Hansen's model forbids it).
        Banking data respond to the economy with a delay, so consider lag_regressors = 1.
        """)
    P = build_panel(df, y, xs, zs, q, cfg)
    panel_checks(P, cfg)
    print(f"\nHypotheses (threshold effect on {P.x_names}):")
    print(f"  H0: slopes are equal in both regimes ({P.q_name} <= gamma and > gamma).")
    print(f"  H1: slopes differ across regimes (a threshold effect exists).")
    if stage <= 2:
        return {"panel": P}

    # ---------------- STAGE 3
    hr("STAGE 3 - BASELINE LINEAR MODELS (OLS)")
    edu("What OLS does", """
        OLS chooses coefficients b that minimise the sum of squared residuals
        SSR = sum (y_it - fitted_it)^2. Residual = actual - fitted: observations are NOT expected
        to sit exactly on the line; the residual is everything the regressors do not explain.
        Fixed effects add one intercept per country by subtracting each country's mean
        (the 'within' transformation), so only variation over time identifies the slopes.
        """)
    pooled, fe = fit_pooled(P), fit_fe(P)
    def equation(lhs, names, beta):
        """Readable regression equation with proper signs."""
        parts = []
        for nm, b in zip(names, beta):
            sign = "-" if b < 0 else "+"
            term = f"{abs(b):.4f}" if nm == "const" else f"{abs(b):.4f}*{nm}"
            parts.append((sign, term))
        first_sign, first_term = parts[0]
        txt = (("-" if first_sign == "-" else "") + first_term)
        for sign, term in parts[1:]:
            txt += f" {sign} {term}"
        return f"{lhs} = {txt}"
    print("\nPooled OLS equation:")
    print("  " + equation(P.y_name, pooled["names"], pooled["beta"]))
    print("\nFixed-effects equation (mu_i = country-specific intercept):")
    print("  " + equation(f"{P.y_name}_it", ["mu_i"] + list(fe["names"]), [1.0] + list(fe["beta"])).replace("1.0000*mu_i", "mu_i"))
    print("\nPooled OLS:")
    print(fmt_table(pooled["table"]))
    print(f"  R2 = {pooled['r2']:.4f}, adj R2 = {pooled['r2_adj']:.4f}, F = {pooled['F']:.3f} (p = {pooled['F_p']:.4f}), "
          f"SSR = {pooled['ssr']:.3f}, RMSE = {pooled['rmse']:.4f}")
    print("\nFixed effects (std_err classical; se_clustered by country):")
    print(fmt_table(fe["table"]))
    print(f"  within R2 = {fe['r2']:.4f}, adj = {fe['r2_adj']:.4f}, overall R2 = {fe['r2_overall']:.4f}, "
          f"F = {fe['F']:.3f} (p = {fe['F_p']:.4f}), SSR = {fe['ssr']:.3f}, RMSE = {fe['rmse']:.4f}")
    print("\nEconomic interpretation (fixed effects):")
    for line in interpret_linear(P, fe):
        print("  - " + textwrap.fill(line, 100, subsequent_indent="    "))
    if P.N < 20:
        warn("clustered standard errors need many clusters; with so few countries, treat them as indicative only.")
    plot_ols(P, pooled, fe)
    results.update(panel=P, pooled=pooled, fe=fe)
    if stage <= 3:
        return results

    # ---------------- STAGE 4
    tests = run_linear_diagnostics(P, pooled, fe, rng)
    results["tests"] = tests
    if stage <= 4:
        return results

    # ---------------- STAGE 5
    hr("STAGE 5 - PANEL THRESHOLD REGRESSION (Hansen, 1999)")
    edu("Is the relationship always linear?", """
        In OLS each coefficient is constant across all observations. In reality the effect of,
        say, unemployment on NPLs may be weak in good times and strong in bad times.
        Threshold regression lets the slope CHANGE when q crosses gamma:
            y_it = mu_i + b1'x_it*1(q_it <= gamma) + b2'x_it*1(q_it > gamma) + e_it
        gamma is unknown, so we try many values (grid search) and keep the one with the smallest SSR.
        We choose by minimum SSR because the least-squares criterion is the same one OLS uses: the
        best gamma is the one whose two regimes describe the data most closely.
        Candidate values are trimmed so that each regime keeps at least a minimum share of the data.
        """)
    thr_model = PanelThreshold(P, trim=cfg["trim"], max_grid=cfg["max_grid"])
    thr = thr_model.fit(alpha=cfg["alpha"])
    print(f"\nCandidate thresholds : {len(thr['cand'])} (min {thr['cand'].min():.3f}, max {thr['cand'].max():.3f})")
    print(f"Estimated threshold  : gamma_hat = {thr['gamma']:.4f}")
    print(f"Approx. {100 * (1 - cfg['alpha']):.0f}% CI for gamma : [{thr['ci'][0]:.4f}, {thr['ci'][1]:.4f}]  (LR inversion, Hansen 1999)")
    print(f"Observations: regime 1 = {thr['n_low']} ({thr['ent_low']} countries), "
          f"regime 2 = {thr['n_high']} ({thr['ent_high']} countries)")
    print(f"SSR threshold model  : {thr['ssr']:.4f}   (linear FE: {thr['ssr_linear']:.4f})")
    print("\nRegime coefficients:")
    print(fmt_table(thr["table"]))
    print("\nRegime difference for each X (conditional on gamma_hat):")
    print(fmt_table(thr["diff"]))
    if min(thr["n_low"], thr["n_high"]) < 0.20 * P.n:
        warn("one regime holds less than 20% of the observations; its coefficients are imprecise.")
    plot_ssr(thr)
    plot_scatter_regimes(P, thr)
    plot_before_after(P, fe, thr, focal=0)
    plot_regime_coefs(P, fe, thr)
    results.update(thr=thr)
    if stage <= 5:
        return results

    # ---------------- STAGE 6
    hr("STAGE 6 - IS THE THRESHOLD EFFECT SIGNIFICANT? (sup-F with bootstrap)")
    edu("Why not an ordinary F test?", """
        A threshold model always fits at least as well as the linear one, so a smaller SSR proves nothing.
        Under H0 (no threshold) gamma is not identified: the grid search picks the BEST of many
        candidates, which makes sup-F larger than any standard F variable. Its distribution is
        non-standard and depends on the data, so Hansen (1999) simulates it by bootstrap:
        generate many datasets from the LINEAR model (regressors and q held fixed), recompute
        sup-F each time, and ask how often it is as large as the observed value.
        'iid' draws residuals with replacement (Hansen's own scheme: valid for independent,
        homoskedastic errors); 'wild' multiplies residuals by random +/-1 (robust to heteroskedasticity).
        Neither handles serial or cross-country dependence. See the diagnostics in stage 4.
        """)
    t0 = pd.Timestamp.now()
    boot = thr_model.bootstrap(fe, cfg["n_bootstrap"], cfg["bootstrap_scheme"], rng)
    test = regime_tests_summary(thr, boot, cfg["bootstrap_scheme"])
    print(f"Bootstrap: {cfg['n_bootstrap']} replications, scheme = {cfg['bootstrap_scheme']} "
          f"({(pd.Timestamp.now() - t0).total_seconds():.1f}s)")
    print(f"Sup-F statistic     : {thr['F']:.4f}")
    print(f"Bootstrap p-value   : {test['p_value']:.4f}")
    print("Bootstrap critical values  : " + ", ".join(f"{k} = {v:.3f}" for k, v in test["crit"].items()))
    print("H0: slope coefficients are equal in both regimes (no threshold effect)")
    print("H1: slopes differ across regimes")
    print("Decision: " + decide(test, P.N))
    plot_bootstrap(thr, boot, test)
    results.update(test=test, boot=boot)
    if stage <= 6:
        return results

    # ---------------- STAGE 7
    hr("STAGE 7 - LINEAR vs THRESHOLD")
    cmp_tab = comparison_table(P, fe, thr, test)
    print(cmp_tab.to_string())
    drop = 100 * (1 - thr["ssr"] / fe["ssr"])
    print(f"\nSSR falls by {drop:.1f}% (this is partly mechanical, because the threshold model is more flexible).")
    print("Do the coefficients change across regimes?  See the 'Regime difference' table above.")
    print("Decision: " + decide(test, P.N))
    print("Is the threshold model more informative?  " +
          ("Only tentatively." if P.N < 10 or test["p_value"] >= 0.05 else "Yes, supported by the bootstrap test."))
    plot_comparison(cmp_tab)
    thr_text = interpret_threshold(P, thr, test)
    print("\nInterpretation of the threshold model:")
    for line in thr_text:
        print("  - " + textwrap.fill(line, 100, subsequent_indent="    "))
    results.update(comparison=cmp_tab)
    if stage <= 7:
        return results

    # ---------------- STAGE 8
    hr("STAGE 8 - REPORT AND EXPORT")
    mgr = managerial_implications(P, thr, test)
    blocks = ["SUMMARY OF FINDINGS",
              f"Sample: {P.N} countries, {P.n} observations. Y = {P.y_name}; X = {', '.join(P.x_names)}; q = {P.q_name}.",
              "Linear (fixed effects) results:", *interpret_linear(P, fe),
              "Threshold results:", *thr_text,
              "Statistical decision: " + decide(test, P.N),
              "Managerial implications (only what the estimates support):", *mgr,
              "Limitations: association, not causation; small N and short T limit inference; "
              "results depend on the variable choice, trimming and the bootstrap scheme."]
    for b in blocks:
        print(textwrap.fill(b, 100))
    export_all(P, pooled, fe, tests, thr, test, boot, cmp_tab, blocks)
    results.update(report=blocks)
    return results


def main():
    ap = argparse.ArgumentParser(description="Interactive panel threshold regression (Hansen, 1999)")
    ap.add_argument("--data", nargs="+", help="path(s) to the CSV/XLSX file(s); several files are merged on country-year")
    ap.add_argument("--demo", action="store_true", help="run on synthetic demo data")
    ap.add_argument("--toy", action="store_true", help="run the 10-row worked example only")
    ap.add_argument("--stage", type=int, default=8, help="stop after this stage (1-8)")
    ap.add_argument("--y")
    ap.add_argument("--x", nargs="+")
    ap.add_argument("--q")
    ap.add_argument("--controls", nargs="*")
    ap.add_argument("--boot", type=int, help="bootstrap replications")
    ap.add_argument("--scheme", choices=["iid", "wild"])
    ap.add_argument("--lag", type=int, help="lag X and q by this many years")
    ap.add_argument("--balance", choices=["keep", "balance"])
    ap.add_argument("--quiet", action="store_true", help="hide the [LEARN] explanations")
    ap.add_argument("--out", help="output directory")
    a = ap.parse_args()

    cfg = dict(CONFIG)
    if a.data:
        cfg["data_path"] = a.data
    if a.y:
        cfg["y"] = a.y
    if a.x:
        cfg["x"] = a.x
    if a.q:
        cfg["q"] = a.q
    if a.controls is not None:
        cfg["controls"] = a.controls
    if a.boot:
        cfg["n_bootstrap"] = a.boot
    if a.scheme:
        cfg["bootstrap_scheme"] = a.scheme
    if a.lag is not None:
        cfg["lag_regressors"] = a.lag
    if a.balance:
        cfg["balance"] = a.balance
    if a.out:
        cfg["output_dir"] = a.out
    if a.quiet:
        cfg["educational"] = False
    CONFIG.update(cfg)

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if a.toy:
        toy_example()
        return
    run(cfg, stage=a.stage, demo=a.demo)


if __name__ == "__main__":
    main()