
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openpyxl.styles import Font, PatternFill
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# PARAMETER
# =============================================================================
FOLDER_KODE = Path(__file__).parent
PATH_DATA = FOLDER_KODE / "data" / "YANG BENER.xlsx"
FOLDER_HASIL = FOLDER_KODE / "outputs"
FOLDER_HASIL.mkdir(exist_ok=True)


NAMA_TARGET = "loan_status"
UMUR_MAKSIMAL = 80
MASA_KERJA_MAKSIMAL = 60

UKURAN_TEST = 0.2
RANDOM_STATE = 42
AMBANG = 0.5

SKOR_MINIMAL = 300
SKOR_MAKSIMAL = 850
BATAS_RISIKO_RENDAH = 740   # skor >= 740  -> Rendah
BATAS_RISIKO_SEDANG = 670   # skor >= 670  -> Sedang, di bawahnya Tinggi

SKOR_DASAR = 600            # skor scorecard ketika odds = ODDS_DASAR
ODDS_DASAR = 50
PDO = 20                    # kenaikan skor tiap odds berlipat dua
BATAS_PD = 1e-6             # mencegah odds tak hingga dan log(0)

LABEL_STATUS = {0: "Lancar", 1: "Gagal bayar"}
WARNA_RISIKO = {"Rendah": "C6EFCE", "Sedang": "FFEB9C", "Tinggi": "FFC7CE"}


# =============================================================================
# FUNGSI BANTU
# =============================================================================
def hitung_metrik(nama_model, y_asli, probabilitas, ambang=AMBANG):
    """Hitung metrik evaluasi pada data test."""
    y_prediksi = (probabilitas >= ambang).astype(int)
    fpr, tpr, _ = roc_curve(y_asli, probabilitas)
    auc = roc_auc_score(y_asli, probabilitas)
    return {
        "model": nama_model,
        "akurasi": accuracy_score(y_asli, y_prediksi),
        "presisi": precision_score(y_asli, y_prediksi),
        "recall": recall_score(y_asli, y_prediksi),
        "f1": f1_score(y_asli, y_prediksi),
        "AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": (tpr - fpr).max(),
    }


def pd_ke_skor(probabilitas):
    """PD menjadi skor skala 300-850."""
    return SKOR_MINIMAL + (1 - probabilitas) * (SKOR_MAKSIMAL - SKOR_MINIMAL)


def pd_ke_skor_scorecard(probabilitas):
    """PD menjadi skor scorecard (rumus odds) beserta odds-nya."""
    pd_aman = np.clip(probabilitas, BATAS_PD, 1 - BATAS_PD)
    odds = (1 - pd_aman) / pd_aman
    skor = SKOR_DASAR + (PDO / np.log(2)) * np.log(odds / ODDS_DASAR)
    return skor, odds


def tentukan_risk_level(skor):
    """Kelompokkan skor menjadi Rendah, Sedang, atau Tinggi."""
    return np.select(
        [skor >= BATAS_RISIKO_RENDAH, skor >= BATAS_RISIKO_SEDANG],
        ["Rendah", "Sedang"],
        default="Tinggi",
    )


def tampilkan_grafik(nama_file):
    """Simpan grafik aktif ke file, lalu tampilkan."""
    plt.tight_layout()
    plt.savefig(FOLDER_HASIL / nama_file, dpi=300, bbox_inches="tight")
    plt.show()


def simpan_excel(path, tabel_utama, lembar_tambahan):
    """Simpan ke Excel dengan kolom risk level berwarna."""
    with pd.ExcelWriter(path, engine="openpyxl") as penulis:
        tabel_utama.to_excel(penulis, sheet_name="Hasil", index=False)
        for nama, (tabel, pakai_index) in lembar_tambahan.items():
            tabel.to_excel(penulis, sheet_name=nama, index=pakai_index)

        lembar = penulis.sheets["Hasil"]
        for sel in lembar[1]:
            sel.font = Font(bold=True, color="FFFFFF")
            sel.fill = PatternFill("solid", fgColor="1F4E78")
        lembar.freeze_panes = "A2"

        for nama_kolom in tabel_utama.columns:
            if not nama_kolom.startswith("risk_level"):
                continue
            nomor_kolom = tabel_utama.columns.get_loc(nama_kolom) + 1
            for baris in lembar.iter_rows(
                min_row=2, min_col=nomor_kolom, max_col=nomor_kolom
            ):
                baris[0].fill = PatternFill("solid", fgColor=WARNA_RISIKO[baris[0].value])


# =============================================================================
# TAHAP 1: BACA DATA, KOLOM, TIPE, DISTRIBUSI TARGET
# =============================================================================
print("=" * 15, "TAHAP 1: PEMAHAMAN DATA", "=" * 15)

data = pd.read_excel(PATH_DATA)

print(f"Jumlah data: {data.shape[0]} baris, {data.shape[1]} kolom")
print(f"\n5 baris pertama:\n{data.head()}")
print(f"\nTipe data:\n{data.dtypes}")
print(f"\nJumlah tiap kelas target:\n{data[NAMA_TARGET].value_counts()}")
print(f"\nPersentase tiap kelas target:\n{data[NAMA_TARGET].value_counts(normalize=True)}")

# =============================================================================
# TAHAP 2: CEK KUALITAS DATA
# =============================================================================
print("\n" + "=" * 15, "TAHAP 2: KUALITAS DATA", "=" * 15)

print(f"Jumlah data kosong per kolom:\n{data.isnull().sum()}")
print(f"\nJumlah baris duplikat: {data.duplicated().sum()}")
print(f"\nRangkuman statistik:\n{data.describe().T.to_string()}")

jumlah_awal = len(data)
data = data.dropna(subset=[NAMA_TARGET]).drop_duplicates()
print(f"\nBaris dibuang (target kosong atau duplikat): {jumlah_awal - len(data)}")

# Perbandingan dengan NaN bernilai False, jadi baris kosong tidak ikut dibuang
# di sini. Nilai kosong pada fitur ditangani oleh imputasi di tahap 4.
outlier = (data["person_age"] > UMUR_MAKSIMAL) | (
    data["person_emp_length"] > MASA_KERJA_MAKSIMAL
)
print(f"Outlier umur atau masa kerja yang dibuang: {outlier.sum()}")
data = data[~outlier]

# Versi teks sudah terwakili oleh kolom _code, jadi tidak dipakai dua kali.
if "cb_person_default_on_file_code" in data.columns:
    data = data.drop(columns=["cb_person_default_on_file"], errors="ignore")

data = data.reset_index(drop=True)
print(f"Ukuran data setelah dibersihkan: {data.shape}")
print(f"Kolom yang dipakai: {data.columns.tolist()}")

# =============================================================================
# TAHAP 3: HEATMAP KORELASI
# =============================================================================
print("\n" + "=" * 15, "TAHAP 3: KORELASI", "=" * 15)

korelasi = data.corr(numeric_only=True)
print(f"Korelasi dengan target:\n{korelasi[NAMA_TARGET].drop(NAMA_TARGET).sort_values()}")

plt.figure(figsize=(10, 7))
sns.heatmap(korelasi, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Peta Korelasi Antar Variabel (Profil Nasabah)")
tampilkan_grafik("grafik_1_korelasi.png")

# =============================================================================
# TAHAP 4: SPLIT DULU, LALU IMPUTASI, ENCODING, SCALING DALAM PIPELINE
# =============================================================================
print("\n" + "=" * 15, "TAHAP 4: SPLIT DAN PREPROCESSING", "=" * 15)

X = data.drop(columns=[NAMA_TARGET])
y = data[NAMA_TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=UKURAN_TEST, random_state=RANDOM_STATE, stratify=y
)

print(f"Ukuran X_train: {X_train.shape}, X_test: {X_test.shape}")
print(
    f"Rata-rata default (semua, train, test): "
    f"{y.mean():.4f}, {y_train.mean():.4f}, {y_test.mean():.4f}"
)
print(f"\nData kosong di X_train:\n{X_train.isnull().sum()}")

kolom_kategori = X.select_dtypes(include=["object", "category"]).columns.tolist()
kolom_angka = [kolom for kolom in X.columns if kolom not in kolom_kategori]
print(f"\nKolom angka: {kolom_angka}")
print(f"Kolom kategori: {kolom_kategori}")

preprocessor = ColumnTransformer(
    [
        (
            "angka",
            Pipeline([
                ("imputasi", SimpleImputer(strategy="median")),
                ("skala", StandardScaler()),
            ]),
            kolom_angka,
        ),
        (
            "kategori",
            Pipeline([
                ("imputasi", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", sparse_output=False)),
            ]),
            kolom_kategori,
        ),
    ],
    verbose_feature_names_out=False,
)

X_train_pp = preprocessor.fit_transform(X_train)  # belajar hanya dari data training
X_test_pp = preprocessor.transform(X_test)        # data test hanya ditransformasi
X_semua_pp = preprocessor.transform(X)
nama_fitur = preprocessor.get_feature_names_out()

print(f"\nUkuran setelah preprocessing: {X_train_pp.shape}, {X_test_pp.shape}")
print(f"Nama fitur: {nama_fitur.tolist()}")
print(f"Sisa nilai kosong setelah imputasi: {np.isnan(X_train_pp).sum()}")

# =============================================================================
# TAHAP 5: LOGISTIC REGRESSION (KOEFISIEN, RASIO ODDS, PD)
# =============================================================================
print("\n" + "=" * 15, "TAHAP 5: LOGISTIC REGRESSION", "=" * 15)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_pp, y_train)

tabel_koefisien = pd.DataFrame({
    "fitur": nama_fitur,
    "koefisien": model_lr.coef_[0],
    "rasio_odds": np.exp(model_lr.coef_[0]),
}).sort_values("koefisien", ascending=False)

print(f"Intercept (b0): {model_lr.intercept_[0]:.4f}")
print(f"\nKoefisien dan rasio odds:\n{tabel_koefisien.to_string(index=False)}")

pd_lr_test = model_lr.predict_proba(X_test_pp)[:, 1]
print(
    f"\nPD data test: min {pd_lr_test.min():.4f}, "
    f"maks {pd_lr_test.max():.4f}, rata-rata {pd_lr_test.mean():.4f}"
)
print(f"Rata-rata default sebenarnya (test): {y_test.mean():.4f}")

# =============================================================================
# TAHAP 6: EVALUASI LR (CONFUSION MATRIX, ROC, AUC, GINI, KS)
# =============================================================================
print("\n" + "=" * 15, "TAHAP 6: EVALUASI LOGISTIC REGRESSION", "=" * 15)

y_pred_lr = (pd_lr_test >= AMBANG).astype(int)
matriks_lr = confusion_matrix(y_test, y_pred_lr)  # [[TN, FP], [FN, TP]]
metrik_lr = hitung_metrik("Logistic Regression", y_test, pd_lr_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, pd_lr_test)

print(f"Confusion matrix:\n{matriks_lr}")
for nama, nilai in metrik_lr.items():
    if nama != "model":
        print(f"{nama:8s}: {nilai:.4f}")

fig, sumbu = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(
    matriks_lr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=sumbu[0],
    xticklabels=["Lancar", "Gagal bayar"], yticklabels=["Lancar", "Gagal bayar"],
)
sumbu[0].set_title(f"Confusion Matrix LR (ambang {AMBANG})")
sumbu[0].set_xlabel("Prediksi")
sumbu[0].set_ylabel("Kenyataan")

sumbu[1].plot(fpr_lr, tpr_lr, linewidth=2.5, label=f"LR (AUC = {metrik_lr['AUC']:.3f})")
sumbu[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Acak")
sumbu[1].set_title(f"Kurva ROC LR (Gini = {metrik_lr['Gini']:.3f}, KS = {metrik_lr['KS']:.3f})")
sumbu[1].set_xlabel("False Positive Rate")
sumbu[1].set_ylabel("True Positive Rate")
sumbu[1].legend()
sumbu[1].grid(True, alpha=0.3)

tampilkan_grafik("grafik_2_evaluasi_lr.png")

# =============================================================================
# TAHAP 7: PD MENJADI CREDIT SCORE DAN RISK LEVEL
# =============================================================================
print("\n" + "=" * 15, "TAHAP 7: CREDIT SCORE DAN RISK LEVEL", "=" * 15)

pd_lr_semua = model_lr.predict_proba(X_semua_pp)[:, 1]
skor_scorecard_lr, _ = pd_ke_skor_scorecard(pd_lr_semua)
skor_lr = pd_ke_skor(pd_lr_semua)

hasil = pd.DataFrame(index=X.index)
hasil["No"] = X.index + 1
hasil["status_aktual"] = y.values
hasil["sumber"] = np.where(X.index.isin(X_test.index), "Test", "Training")
hasil["pd_lr"] = pd_lr_semua
hasil["skor_scorecard_lr"] = skor_scorecard_lr
hasil["skor_lr"] = skor_lr
hasil["risk_level_lr"] = tentukan_risk_level(skor_lr)

print(hasil[["pd_lr", "skor_scorecard_lr", "skor_lr"]].describe())
print(f"\nJumlah nasabah per risk level:\n{hasil['risk_level_lr'].value_counts()}")
print(
    "\nTingkat gagal bayar sebenarnya per risk level:\n"
    f"{hasil.groupby('risk_level_lr')['status_aktual'].mean()}"
)

# =============================================================================
# TAHAP 8: HISTOGRAM PD DAN SKOR (DEFAULT vs TIDAK DEFAULT)
# =============================================================================
print("\n" + "=" * 15, "TAHAP 8: HISTOGRAM", "=" * 15)

label_status = hasil["status_aktual"].map(LABEL_STATUS).rename("status")

fig, sumbu = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(
    x=hasil["pd_lr"], hue=label_status, bins=40, stat="density",
    common_norm=False, element="step", ax=sumbu[0],
)
sumbu[0].set_title("Sebaran PD (Logistic Regression)")
sumbu[0].set_xlabel("Probability of Default")
sumbu[0].set_ylabel("Kepadatan")

sns.histplot(
    x=hasil["skor_lr"], hue=label_status, bins=40, stat="density",
    common_norm=False, element="step", ax=sumbu[1],
)
sumbu[1].axvline(BATAS_RISIKO_SEDANG, color="orange", linestyle="--", label="Batas Sedang/Tinggi")
sumbu[1].axvline(BATAS_RISIKO_RENDAH, color="green", linestyle="--", label="Batas Rendah/Sedang")
sumbu[1].set_title(f"Sebaran Credit Score ({SKOR_MINIMAL}-{SKOR_MAKSIMAL})")
sumbu[1].set_xlabel("Credit Score")
sumbu[1].set_ylabel("Kepadatan")

tampilkan_grafik("grafik_3_histogram_pd_skor.png")

# =============================================================================
# TAHAP 9: RANDOM FOREST (PD DAN IMPORTANCE)
# =============================================================================
print("\n" + "=" * 15, "TAHAP 9: RANDOM FOREST", "=" * 15)

model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model_rf.fit(X_train_pp, y_train)
print("Model Random Forest selesai dilatih.")

pd_rf_test = model_rf.predict_proba(X_test_pp)[:, 1]
pd_rf_semua = model_rf.predict_proba(X_semua_pp)[:, 1]
skor_rf = pd_ke_skor(pd_rf_semua)

hasil["pd_rf"] = pd_rf_semua
hasil["skor_rf"] = skor_rf
hasil["risk_level_rf"] = tentukan_risk_level(skor_rf)

tabel_importance = (
    pd.DataFrame({"fitur": nama_fitur, "importance": model_rf.feature_importances_})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print(f"\n10 fitur terpenting:\n{tabel_importance.head(10).to_string()}")

importance_10 = tabel_importance.head(10).iloc[::-1]
plt.figure(figsize=(9, 5))
batang = plt.barh(importance_10["fitur"], importance_10["importance"], color="teal")
plt.bar_label(batang, fmt="%.3f", padding=3)
plt.title("10 Fitur Terpenting Random Forest")
plt.xlabel("Importance")
tampilkan_grafik("grafik_4_importance_rf.png")

# =============================================================================
# TAHAP 10: BANDINGKAN LR vs RF
# =============================================================================
print("\n" + "=" * 15, "TAHAP 10: PERBANDINGAN MODEL", "=" * 15)

metrik_rf = hitung_metrik("Random Forest", y_test, pd_rf_test)
perbandingan = pd.DataFrame([metrik_lr, metrik_rf]).set_index("model").round(4)
print(perbandingan)

matriks_rf = confusion_matrix(y_test, (pd_rf_test >= AMBANG).astype(int))
print(f"\nConfusion matrix Random Forest:\n{matriks_rf}")

fpr_rf, tpr_rf, _ = roc_curve(y_test, pd_rf_test)

plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, linewidth=2.5, label=f"Logistic Regression (AUC = {metrik_lr['AUC']:.3f})")
plt.plot(fpr_rf, tpr_rf, linewidth=2.5, label=f"Random Forest (AUC = {metrik_rf['AUC']:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Acak")
plt.title("Perbandingan Kurva ROC (Data Test)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, alpha=0.3)
tampilkan_grafik("grafik_5_roc_lr_vs_rf.png")

# =============================================================================
# TAHAP 11: TABEL HASIL DAN EXCEL BERWARNA
# =============================================================================
print("\n" + "=" * 15, "TAHAP 11: EXPORT EXCEL", "=" * 15)

hasil["risk_level_lr"] = hasil["risk_level_lr"].astype(str)
hasil["risk_level_rf"] = hasil["risk_level_rf"].astype(str)
hasil_test = hasil[hasil["sumber"] == "Test"]

nama_label = ["Aktual Lancar", "Aktual Gagal bayar"]
kolom_label = ["Prediksi Lancar", "Prediksi Gagal bayar"]
lembar_umum = {
    "Perbandingan_Model": (perbandingan, True),
    "Koefisien_LR": (tabel_koefisien, False),
    "Importance_RF": (tabel_importance, False),
}
lembar_test = {
    **lembar_umum,
    "Confusion_LR": (pd.DataFrame(matriks_lr, index=nama_label, columns=kolom_label), True),
    "Confusion_RF": (pd.DataFrame(matriks_rf, index=nama_label, columns=kolom_label), True),
}

path_semua = FOLDER_HASIL / "1_credit_scoring_ALL_NASABAH.xlsx"
path_test = FOLDER_HASIL / "2_prediksi_DATA_TESTING_SAJA.xlsx"

simpan_excel(path_semua, hasil, lembar_umum)
simpan_excel(path_test, hasil_test, lembar_test)

print(f"File 1: {path_semua} ({len(hasil)} baris)")
print(f"File 2: {path_test} ({len(hasil_test)} baris)")

# =============================================================================
# TAHAP 12: RINGKASAN AKHIR DAN BUKA FILE
# =============================================================================
print("\n" + "=" * 60)
print("RINGKASAN AKHIR")
print("=" * 60)
print(f"Jumlah nasabah dianalisis : {len(hasil)}")
print(f"Tingkat gagal bayar       : {y.mean():.1%}")
print(f"Ambang klasifikasi        : {AMBANG}")
print(f"\nPerbandingan model (data test):\n{perbandingan}")
print(f"\nRisk level Logistic Regression:\n{hasil['risk_level_lr'].value_counts()}")
print(f"\nRisk level Random Forest:\n{hasil['risk_level_rf'].value_counts()}")

if hasattr(os, "startfile"):
    for path_file in (path_semua, path_test):
        os.startfile(path_file)
        
# =============================================================================
# TAHAP 13: EKSPOR UNTUK WEBSITE
# =============================================================================
import json

pp_angka = preprocessor.named_transformers_["angka"]
imputer_angka = pp_angka.named_steps["imputasi"]
scaler_angka = pp_angka.named_steps["skala"]
koef = dict(zip(nama_fitur, model_lr.coef_[0]))

fitur_angka = []
for i, kolom in enumerate(kolom_angka):
    nilai_unik = set(X_train[kolom].dropna().unique())
    fitur_angka.append({
        "nama": kolom,
        "median": float(imputer_angka.statistics_[i]),
        "mean": float(scaler_angka.mean_[i]),
        "scale": float(scaler_angka.scale_[i]),
        "koefisien": float(koef[kolom]),
        "min": float(X[kolom].min()),
        "maks": float(X[kolom].max()),
        "biner": bool(nilai_unik <= {0, 1}),
    })

fitur_kategori = []
if kolom_kategori:
    pp_kat = preprocessor.named_transformers_["kategori"]
    onehot = pp_kat.named_steps["onehot"]
    modus = pp_kat.named_steps["imputasi"].statistics_
    for j, kolom in enumerate(kolom_kategori):
        kategori = [str(k) for k in onehot.categories_[j]]
        dasar = kategori[int(onehot.drop_idx_[j])]
        fitur_kategori.append({
            "nama": kolom,
            "kategori": kategori,
            "dasar": dasar,
            "modus": str(modus[j]),
            "frekuensi": {k: float((X_train[kolom] == k).mean()) for k in kategori},
            "koefisien": {k: float(koef[f"{kolom}_{k}"]) for k in kategori if k != dasar},
        })

model_json = {
    "intercept": float(model_lr.intercept_[0]),
    "angka": fitur_angka,
    "kategori": fitur_kategori,
        "skor": {
        "minimal": SKOR_MINIMAL, "maksimal": SKOR_MAKSIMAL,
        "batas_rendah": BATAS_RISIKO_RENDAH, "batas_sedang": BATAS_RISIKO_SEDANG,
        "dasar": SKOR_DASAR, "odds_dasar": ODDS_DASAR, "pdo": PDO, "batas_pd": BATAS_PD,
    },
}


def metrik_bersih(m):
    return {k: (v if k == "model" else float(v)) for k, v in m.items()}


per_level = hasil.groupby("risk_level_lr")["status_aktual"].agg(["count", "mean"])
hasil_json = {
    "jumlah_nasabah": int(len(hasil)),
    "tingkat_gagal_bayar": float(y.mean()),
    "jumlah_training": int(len(X_train)),
    "jumlah_test": int(len(X_test)),
    "ambang": float(AMBANG),
    "metrik": {"lr": metrik_bersih(metrik_lr), "rf": metrik_bersih(metrik_rf)},
    "confusion_lr": matriks_lr.tolist(),
    "confusion_rf": matriks_rf.tolist(),
    "koefisien": [
        {"fitur": r.fitur, "koefisien": float(r.koefisien), "rasio_odds": float(r.rasio_odds)}
        for r in tabel_koefisien.itertuples()
    ],
    "importance": [
        {"fitur": r.fitur, "importance": float(r.importance)}
        for r in tabel_importance.head(10).itertuples()
    ],
    "risk_level": {
        nama: {"jumlah": int(per_level.loc[nama, "count"]),
               "gagal_bayar": float(per_level.loc[nama, "mean"])}
        for nama in per_level.index
    },
}

for nama_file, isi in (("hasil.json", hasil_json), ("model.json", model_json)):
    with open(FOLDER_HASIL / nama_file, "w", encoding="utf-8") as f:
        json.dump(isi, f, ensure_ascii=False, indent=2)

print("\nhasil.json dan model.json tersimpan di", FOLDER_HASIL)
        
def pohon_ke_dict(pohon):
    t = pohon.tree_
    nilai = t.value[:, 0, :]
    return {
        "kiri": t.children_left.tolist(),
        "kanan": t.children_right.tolist(),
        "fitur": t.feature.tolist(),
        "batas": [round(float(x), 5) for x in t.threshold],
        "prob": [round(float(x), 4) for x in nilai[:, 1] / nilai.sum(axis=1)],
    }


rf_json = {
    "nama_fitur": nama_fitur.tolist(),
    "pohon": [pohon_ke_dict(p) for p in model_rf.estimators_],
}
with open(FOLDER_HASIL / "rf.json", "w", encoding="utf-8") as f:
    json.dump(rf_json, f, separators=(",", ":"))
print("rf.json tersimpan:", round((FOLDER_HASIL / "rf.json").stat().st_size / 1e6, 1), "MB")

contoh = X_test.head(3).copy()
contoh["pd_lr"] = pd_lr_test[:3].round(4)
contoh["pd_rf"] = pd_rf_test[:3].round(4)
print("\nContoh untuk menguji form website:\n", contoh.to_string())      


def pohon_ke_dict(pohon):
    t = pohon.tree_
    nilai = t.value[:, 0, :]
    return {
        "kiri": t.children_left.tolist(),
        "kanan": t.children_right.tolist(),
        "fitur": t.feature.tolist(),
        "batas": [round(float(x), 5) for x in t.threshold],
        "prob": [round(float(x), 4) for x in nilai[:, 1] / nilai.sum(axis=1)],
    }


rf_json = {
    "nama_fitur": nama_fitur.tolist(),
    "pohon": [pohon_ke_dict(p) for p in model_rf.estimators_],
}
with open(FOLDER_HASIL / "rf.json", "w", encoding="utf-8") as f:
    json.dump(rf_json, f, separators=(",", ":"))
print("rf.json tersimpan:", round((FOLDER_HASIL / "rf.json").stat().st_size / 1e6, 1), "MB")

contoh = X_test.head(3).copy()
contoh["pd_lr"] = pd_lr_test[:3].round(4)
contoh["pd_rf"] = pd_rf_test[:3].round(4)
print("\nContoh untuk menguji website:\n", contoh.to_string())