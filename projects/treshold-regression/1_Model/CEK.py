
import pandas as pd

file_path = r"C:\Users\Latihan Ghina Python\NEBULA PROJECT\WEB PORTOFOLIO\projects\treshold-regression\1_Model\data\ASEAN_Banking_PanelData_2019_2025.xlsx"

df = pd.read_excel(file_path)
print(df[["Bank name", "Year", "LR", "INF", "Covid Dummy"]].head(20).to_string(index=False))

# Cek duplikasi kombinasi bank dan tahun
duplikat = df[df.duplicated(
    subset=["Bank name", "Year"],
    keep=False
)]

print("\n=== DATA DUPLIKAT BANK-TAHUN ===")

if duplikat.empty:
    print("Tidak ada duplikasi.")
else:
    print(duplikat[
        ["Country", "Bank name", "Year"]
    ].sort_values(["Bank name", "Year"]).to_string(index=False))

print("\nJumlah baris duplikat:", len(duplikat))


"""print("\n=== PEMERIKSAAN IDENTITAS PANEL ===")

print(df[[
    "Country",
    "Bank name",
    "Year",
    "LR",
    "INF",
    "Covid Dummy"
]].head(30).to_string(index=False))

print("\nJumlah Country kosong:", df["Country"].isna().sum())
print("Jumlah Bank name kosong:", df["Bank name"].isna().sum())


variables = ["LR", "INF", "Covid Dummy"]

for var in variables:
    print(f"\n=== {var} ===")
    print("Jumlah nilai unik:", df[var].nunique())
    print("Nilai unik:", df[var].unique())
    print("\nPer bank:")
    print(df.groupby("Bank name")[var].nunique())"""
