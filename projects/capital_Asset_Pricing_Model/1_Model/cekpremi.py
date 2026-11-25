from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
r = pd.read_csv(OUT / "monthly_returns.csv", index_col=0, parse_dates=True)["^JKSE"].dropna()

print("Return tahunan ^JKSE per tahun (%):")
print(((1 + r).groupby(r.index.year).prod() - 1).mul(100).round(1))

print("\nRata-rata tahunan, semua tahun (%):", round(r.mean() * 12 * 100, 2))
print("Rata-rata tahunan tanpa 2020 (%):  ", round(r[r.index.year != 2020].mean() * 12 * 100, 2))
