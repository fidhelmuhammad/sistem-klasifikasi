# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import inspect
import traceback
from datetime import datetime

st.set_page_config(page_title="Sistem Klasifikasi Penerima Bantuan Sosial", layout="wide")

# --- Safe import scikit-learn (tampilkan instruksi bila tidak terpasang) ---
SKLEARN_AVAILABLE = True
SKLEARN_IMPORT_ERROR = None
try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
except Exception as e:
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = e

# -----------------------
# Helper utilities
# -----------------------
def show_missing_sklearn_message():
    st.error(
        "Modul `scikit-learn` tidak ditemukan di environment. "
        "Silakan pasang dependency terlebih dahulu dengan menjalankan:\n\n"
        "`pip install -r requirements.txt`  atau `pip install scikit-learn`\n\n"
        "Lalu restart aplikasi. Error import asli:\n\n"
        f"```\n{SKLEARN_IMPORT_ERROR}\n```"
    )

def find_column_by_keywords(columns, keywords):
    """Return the first column name containing any keyword (case-insensitive)."""
    for kw in keywords:
        for c in columns:
            if kw.lower() in str(c).lower():
                return c
    return None

def read_uploaded_file(uploaded_file):
    """Read CSV or Excel into pandas DataFrame."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif filename.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("Format file tidak didukung. Unggah .csv atau .xlsx.")

def validate_dataset(df):
    """Validasi keberadaan kolom Status dan label yang diperbolehkan."""
    if df is None or df.shape[0] == 0:
        raise ValueError("Dataset kosong atau gagal dibaca.")
    if "Status" not in df.columns:
        raise ValueError("Kolom target 'Status' tidak ditemukan. Pastikan nama kolom tepat 'Status'.")
    labels = set(df["Status"].dropna().unique())
    allowed = {"Layak", "Tidak Layak"}
    if not labels.issubset(allowed):
        raise ValueError(f"Label 'Status' hanya boleh 'Layak' dan 'Tidak Layak'. Ditemukan: {sorted(list(labels))}")
    return True

def prepare_features(df, drop_cols=None):
    """Pisah X dan y, deteksi fitur numerik/kategorikal, dan kembalikan daftar kolom fitur."""
    if drop_cols is None: drop_cols = []
    y = df["Status"].copy()
    X = df.drop(columns=["Status"]).copy()
    # drop name/kampung dari fitur
    X_for_model = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')
    numeric_feats = X_for_model.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X_for_model.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    feature_columns = X_for_model.columns.tolist()
    return X_for_model, y, numeric_feats, cat_feats, feature_columns

def build_preprocessor(numeric_feats, cat_feats):
    """Buat ColumnTransformer dengan imputasi & encoding. Menangani beda signature OneHotEncoder."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn tidak tersedia")

    transformers = []
    if numeric_feats:
        num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
        transformers.append(("num", num_pipeline, numeric_feats))
    if cat_feats:
        # adaptif: beberapa versi sklearn punya sparse_output, beberapa punya sparse
        ohe_kwargs = {"handle_unknown": "ignore"}
        try:
            sig = inspect.signature(OneHotEncoder)
            if 'sparse_output' in sig.parameters:
                ohe_kwargs['sparse_output'] = False
            else:
                ohe_kwargs['sparse'] = False
        except Exception:
            ohe_kwargs['sparse'] = False
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**ohe_kwargs))
        ])
        transformers.append(("cat", cat_pipeline, cat_feats))

    if not transformers:
        # tidak ada fitur -> tidak bisa training, caller harus cek
        return None
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor

def train_nb_model(X, y, numeric_feats, cat_feats):
    """Latih pipeline GaussianNB dan kembalikan pipeline serta akurasi sederhana."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn tidak tersedia")

    preprocessor = build_preprocessor(numeric_feats, cat_feats)
    if preprocessor is None:
        raise ValueError("Tidak ada fitur untuk dilatih (semua kolom di-drop). Periksa dataset atau kolom yang dipakai.")

    clf = Pipeline([("pre", preprocessor), ("nb", GaussianNB())])

    # split: stratify jika kelas minoritas cukup besar
    try:
        if min(y.value_counts()) >= 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

    clf.fit(X_train, y_train)
    try:
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
    except Exception:
        acc = None
    return clf, acc, (X_train, X_test, y_train, y_test)

def explain_prediction_simple(df_original, status_series, exclude_cols=None):
    """
    Bangun tabel frekuensi & rata-rata numeric untuk memberikan penjelasan sederhana.
    exclude_cols: list kolom (mis. [name_col, kampung_col]) yang tidak dipakai dalam analisis.
    """
    if exclude_cols is None: exclude_cols = []
    freq_tables = {}
    numeric_stats = {}
    for col in df_original.columns:
        if col == "Status" or col in exclude_cols:
            continue
        # numeric
        if pd.api.types.is_numeric_dtype(df_original[col]):
            try:
                numeric_stats[col] = df_original.groupby("Status")[col].mean().to_dict()
            except Exception:
                numeric_stats[col] = {}
        else:
            try:
                freq = df_original.groupby(["Status", col]).size().rename("count").reset_index()
                pivot = freq.pivot(index=col, columns="Status", values="count").fillna(0)
                prop = pivot.copy()
                for c in prop.columns:
                    total = prop[c].sum()
                    if total > 0:
                        prop[c] = prop[c] / total
                    else:
                        prop[c] = 0
                freq_tables[col] = prop
            except Exception:
                continue
    return freq_tables, numeric_stats

def feature_contributors_for_row(row, predicted_class, freq_tables, numeric_stats, other_class):
    """Hitung skor sederhana per fitur yang mendukung kelas prediksi."""
    eps = 1e-9
    scores = []
    for col in row.index:
        val = row[col]
        if pd.isna(val):
            continue
        # numeric
        if col in numeric_stats:
            stats = numeric_stats.get(col, {})
            mean_pred = stats.get(predicted_class, np.nan)
            mean_other = stats.get(other_class, np.nan)
            if pd.isna(mean_pred) or pd.isna(mean_other):
                continue
            score = (abs(val - mean_other) - abs(val - mean_pred))
            reason = f"Nilai {val} lebih dekat ke rata-rata {predicted_class} ({mean_pred:.2f}) daripada {other_class} ({mean_other:.2f})" if score>0 else f"Nilai {val} kurang mendukung {predicted_class}"
            scores.append((col, float(score), reason))
        else:
            table = freq_tables.get(col)
            if table is None:
                continue
            try:
                # jika kategori tidak ada di table index, treat as 0
                if val not in table.index:
                    prop_pred = 0.0
                    prop_other = 0.0
                else:
                    prop_pred = float(table.loc[val, predicted_class]) if predicted_class in table.columns else 0.0
                    prop_other = float(table.loc[val, other_class]) if other_class in table.columns else 0.0
                score = np.log(prop_pred + eps) - np.log(prop_other + eps)
                reason = f"Kategori '{val}' lebih umum pada {predicted_class} (prop {prop_pred:.2f}) daripada {other_class} (prop {prop_other:.2f})" if score>0 else f"Kategori '{val}' tidak mendukung kuat {predicted_class}"
                scores.append((col, float(score), reason))
            except Exception:
                continue
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores_sorted

def prepare_X_for_predict(df_k, feature_columns, numeric_feats, cat_feats, drop_cols):
    """
    Align kolom fitur df_k sesuai feature_columns yang dipakai saat training.
    - Tambahkan kolom yang hilang dengan NaN
    - Hapus kolom ekstra
    - Konversi tipe numeric/categorical sesuai list
    """
    X = df_k.copy()
    # drop cols non-feature
    X = X.drop(columns=[c for c in drop_cols if c in X.columns] + ["Status", "Prediction"], errors='ignore')
    # ensure all feature_columns exist
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan
    # drop extras
    extra_cols = [c for c in X.columns if c not in feature_columns]
    if extra_cols:
        X = X.drop(columns=extra_cols, errors='ignore')
    # reorder
    X = X[feature_columns]
    # cast types
    for c in numeric_feats:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    for c in cat_feats:
        if c in X.columns:
            # ensure object dtype for consistent imputation/encoding
            X[c] = X[c].astype(object)
    return X

def save_archive(record, path="archive.csv"):
    df_rec = pd.DataFrame([record])
    if os.path.exists(path):
        df_rec.to_csv(path, mode='a', header=False, index=False)
    else:
        df_rec.to_csv(path, index=False)
    if "archive" not in st.session_state:
        st.session_state["archive"] = []
    st.session_state["archive"].append(record)

def load_archive(path="archive.csv"):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=["timestamp", "kampung", "num_layak", "num_tidak_layak", "details_path"])
    else:
        return pd.DataFrame(columns=["timestamp", "kampung", "num_layak", "num_tidak_layak", "details_path"])

# -----------------------
# Sidebar / State init
# -----------------------
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Training", "Prediksi", "Arsip/Riwayat"])

# session defaults
st.session_state.setdefault("df", None)
st.session_state.setdefault("df_raw", None)
st.session_state.setdefault("trained_pipeline", None)
st.session_state.setdefault("feature_info", None)  # (freq_tables, numeric_stats)
st.session_state.setdefault("drop_cols_for_model", [])
st.session_state.setdefault("last_training_summary", None)
st.session_state.setdefault("archive", [])
st.session_state.setdefault("feature_columns", None)
st.session_state.setdefault("numeric_feats", [])
st.session_state.setdefault("cat_feats", [])

# -----------------------
# Home
# -----------------------
if page == "Home":
    st.title("Sistem Klasifikasi Penerima Bantuan Sosial")
    st.markdown("""
    **Tujuan:** membantu menentukan siapa yang layak/tidak layak menerima bantuan sosial agar tepat sasaran.  
    **Metode klasifikasi:** *Naïve Bayes* (GaussianNB dari scikit-learn) dengan preprocessing otomatis.
    """)
    st.markdown("#### Langkah singkat penggunaan")
    st.write("""
    1. Ke halaman **Training** → unggah dataset (.csv/.xlsx) → klik *Mulai Training*.  
    2. Setelah model terlatih, ke halaman **Prediksi** → pilih nama kampung → jalankan prediksi.  
    3. Hasil prediksi disimpan ke **Arsip/Riwayat** dan dapat diunduh.
    """)
    st.info("Kolom target harus bernama **Status** dengan label hanya 'Layak' dan 'Tidak Layak'. Sistem akan mencoba mendeteksi kolom Nama (kata 'nama') dan Kampung (kata 'kampung'/'desa').")
    if not SKLEARN_AVAILABLE:
        st.warning("`scikit-learn` tidak ditemukan. Halaman Training/Prediksi tidak bisa digunakan sampai dependency dipasang.")

# ---
