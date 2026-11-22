import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize

daftar_saham = ["BBCA.JK", "BMRI.JK", "ASII.JK", "INDF.JK"]
tanggal_awal = "2021-01-01"
tanggal_akhir = "2026-01-01"

hari_bursa = 252
suku_bunga_bebas_risiko = 0.06
bobot_maksimal = 0.4

jumlah_simulasi = 1000
jumlah_hari = 100
modal_awal = 10_000_000

harga = yf.download(daftar_saham, start=tanggal_awal, end=tanggal_akhir, auto_adjust=True)["Close"]
return_harian = harga.pct_change().dropna()
rata_return = return_harian.mean()
matriks_kovarians = return_harian.cov()
print(f"matriks kovarians dari ke 4 saham {matriks_kovarians}")
nama_saham = [kode.replace(".JK", "") for kode in rata_return.index]
jumlah_saham = len(nama_saham)
judul_saham = ", ".join(nama_saham)

return_tahunan = rata_return.values * hari_bursa
kovarians_tahunan = matriks_kovarians.values * hari_bursa
volatilitas_tahunan = np.sqrt(np.diag(kovarians_tahunan))
sharpe_saham = (return_tahunan - suku_bunga_bebas_risiko) / volatilitas_tahunan


def sharpe_negatif(bobot):
    return_portofolio = bobot @ return_tahunan
    risiko_portofolio = np.sqrt(bobot @ kovarians_tahunan @ bobot)
    return -(return_portofolio - suku_bunga_bebas_risiko) / risiko_portofolio


hasil_optimasi = minimize(
    sharpe_negatif,
    x0=np.full(jumlah_saham, 1 / jumlah_saham),
    method="SLSQP",
    bounds=[(0, bobot_maksimal)] * jumlah_saham,
    constraints={"type": "eq", "fun": lambda bobot: bobot.sum() - 1},
)
bobot = hasil_optimasi.x

return_portofolio = bobot @ return_tahunan
risiko_portofolio = np.sqrt(bobot @ kovarians_tahunan @ bobot)
sharpe_portofolio = (return_portofolio - suku_bunga_bebas_risiko) / risiko_portofolio

hasil_simulasi = np.zeros((jumlah_hari + 1, jumlah_simulasi))
hasil_simulasi[0] = modal_awal
matriks_rata_return = np.tile(rata_return.values, (jumlah_hari, 1)).T
L = np.linalg.cholesky(matriks_kovarians)

for i in range(jumlah_simulasi):
    Z = np.random.normal(size=(jumlah_hari, jumlah_saham))
    return_simulasi = matriks_rata_return + L @ Z.T
    hasil_simulasi[1:, i] = np.cumprod(bobot @ return_simulasi + 1) * modal_awal

rata_waktu = hasil_simulasi.mean(axis=1)
persentil_5_waktu = np.percentile(hasil_simulasi, 5, axis=1)
persentil_95_waktu = np.percentile(hasil_simulasi, 95, axis=1)

nilai_akhir = hasil_simulasi[-1]
rata_akhir = nilai_akhir.mean()
median_akhir = np.median(nilai_akhir)
persentil_5 = np.percentile(nilai_akhir, 5)
persentil_95 = np.percentile(nilai_akhir, 95)
peluang_untung = (nilai_akhir > modal_awal).mean()
peluang_rugi = 1 - peluang_untung
var_95 = modal_awal - persentil_5

hari = np.arange(jumlah_hari + 1)
format_juta = FuncFormatter(lambda nilai, _: f"{nilai / 1e6:.0f}")

plt.figure(figsize=(11, 6))
plt.plot(hari, hasil_simulasi, alpha=0.4, linewidth=0.8)
plt.plot(hari, rata_waktu, color="black", linewidth=3, label="Rata-rata simulasi", zorder=5)
plt.plot(hari, persentil_5_waktu, color="red", linestyle="--", linewidth=3, label="Persentil 5", zorder=5)
plt.plot(hari, persentil_95_waktu, color="green", linestyle="--", linewidth=3, label="Persentil 95", zorder=5)
plt.axhline(modal_awal, color="gray", linestyle=":", linewidth=2, label="Modal awal", zorder=5)
plt.gca().yaxis.set_major_formatter(format_juta)
plt.title(f"Simulasi Monte Carlo Portofolio Saham {judul_saham}\n"
          f"(Bobot Markowitz Maksimum Sharpe, Data {tanggal_awal[:4]}-2025)")
plt.xlabel("Hari ke-")
plt.ylabel("Nilai Portofolio (juta Rp)")
plt.legend()
plt.tight_layout()
plt.savefig("grafik_jalur_monte_carlo.png", dpi=300)
plt.show()

plt.figure(figsize=(11, 6))
plt.hist(nilai_akhir, bins=60, color="tab:blue", alpha=0.75)
plt.axvline(modal_awal, color="gray", linestyle=":", linewidth=2, label=f"Modal awal {modal_awal:,.0f}")
plt.axvline(median_akhir, color="black", linewidth=2.5, label=f"Median {median_akhir:,.0f}")
plt.axvline(rata_akhir, color="orange", linewidth=2.5, label=f"Rata-rata {rata_akhir:,.0f}")
plt.axvline(persentil_5, color="red", linestyle="--", linewidth=2, label=f"P5 {persentil_5:,.0f}")
plt.axvline(persentil_95, color="green", linestyle="--", linewidth=2, label=f"P95 {persentil_95:,.0f}")
plt.gca().xaxis.set_major_formatter(format_juta)
plt.title(f"Sebaran Nilai Akhir Portofolio {judul_saham} ({jumlah_hari} hari)\n"
          f"Peluang untung {peluang_untung:.1%}")
plt.xlabel("Nilai Portofolio Akhir (juta Rp)")
plt.ylabel("Jumlah skenario")
plt.legend()
plt.tight_layout()
plt.savefig("grafik_sebaran_nilai_akhir.png", dpi=300)
plt.show()

posisi = np.arange(jumlah_saham)
fig, sumbu = plt.subplots(1, 4, figsize=(16, 5))

data_batang = [
    (return_tahunan * 100, "Return Tahunan (%)"),
    (volatilitas_tahunan * 100, "Volatilitas Tahunan (%)"),
    (sharpe_saham, "Sharpe Ratio"),
    (bobot * 100, "Bobot Markowitz (%)"),
]

for ax, (nilai, judul) in zip(sumbu, data_batang):
    batang = ax.bar(posisi, nilai, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_xticks(posisi)
    ax.set_xticklabels(nama_saham)
    ax.set_title(judul)
    ax.bar_label(batang, fmt="%.1f")

fig.suptitle(f"Perbandingan Karakteristik Saham {judul_saham} ({tanggal_awal[:4]}-2025)")
plt.tight_layout()
plt.savefig("grafik_perbandingan_saham.png", dpi=300)
plt.show()

tabel_saham = pd.DataFrame({
    "Return Tahunan (%)": return_tahunan * 100,
    "Volatilitas Tahunan (%)": volatilitas_tahunan * 100,
    "Sharpe Ratio": sharpe_saham,
    "Bobot Markowitz (%)": bobot * 100,
}, index=nama_saham).round(2)

print("=" * 60)
print("RINGKASAN HASIL")
print("=" * 60)
print(f"Periode data           : {harga.index.min().date()} s/d {harga.index.max().date()}")
print(f"Jumlah hari perdagangan: {len(return_harian)}")
print(f"Suku bunga bebas risiko: {suku_bunga_bebas_risiko:.1%}")

print("\n1. KARAKTERISTIK PER SAHAM")
print(tabel_saham)

print("\n2. KOMPOSISI PORTOFOLIO (MARKOWITZ MAKSIMUM SHARPE)")
for nama, b in zip(nama_saham, bobot):
    print(f"   {nama}: {b:.1%}")
print(f"   Return tahunan portofolio     : {return_portofolio:.2%}")
print(f"   Volatilitas tahunan portofolio: {risiko_portofolio:.2%}")
print(f"   Sharpe ratio portofolio       : {sharpe_portofolio:.2f}")

print(f"\n3. HASIL SIMULASI MONTE CARLO ({jumlah_simulasi} simulasi, {jumlah_hari} hari)")
print(f"   Modal awal            : Rp{modal_awal:,.0f}")
print(f"   Rata-rata nilai akhir : Rp{rata_akhir:,.0f}")
print(f"   Median nilai akhir    : Rp{median_akhir:,.0f}")
print(f"   Skenario buruk (P5)   : Rp{persentil_5:,.0f}")
print(f"   Skenario baik (P95)   : Rp{persentil_95:,.0f}")
print(f"   Peluang untung        : {peluang_untung:.1%}")
print(f"   Peluang rugi          : {peluang_rugi:.1%}")
print(f"   Value at Risk 95%     : Rp{var_95:,.0f}")

# hasil untuk kode lengkap di card 2

import json
from pathlib import Path

folder_output = Path(__file__).parent / "outputs"
folder_output.mkdir(exist_ok=True)

es_95 = modal_awal - nilai_akhir[nilai_akhir <= persentil_5].mean()

hasil = {
    "saham": nama_saham,
    "periode": [str(harga.index.min().date()), str(harga.index.max().date())],
    "jumlah_hari_data": int(len(return_harian)),
    "suku_bunga_bebas_risiko": float(suku_bunga_bebas_risiko),
    "jumlah_simulasi": int(jumlah_simulasi),
    "jumlah_hari": int(jumlah_hari),
    "modal_awal": float(modal_awal),
    "return_portofolio": float(return_portofolio),
    "volatilitas_portofolio": float(risiko_portofolio),
    "sharpe_portofolio": float(sharpe_portofolio),
    "rata_akhir": float(rata_akhir),
    "median_akhir": float(median_akhir),
    "persentil_5": float(persentil_5),
    "persentil_95": float(persentil_95),
    "peluang_untung": float(peluang_untung),
    "peluang_rugi": float(peluang_rugi),
    "var_95": float(var_95),
    "es_95": float(es_95),
    "tabel_saham": [
        {"nama": n, "return_tahunan": float(r), "volatilitas": float(v),
         "sharpe": float(s), "bobot": float(b)}
        for n, r, v, s, b in zip(nama_saham, return_tahunan,
                                 volatilitas_tahunan, sharpe_saham, bobot)
    ],
    "kovarians": matriks_kovarians.values.tolist(),
}

with open(folder_output / "hasil.json", "w", encoding="utf-8") as f:
    json.dump(hasil, f, ensure_ascii=False, indent=2)

print("\nhasil.json tersimpan di", folder_output)































