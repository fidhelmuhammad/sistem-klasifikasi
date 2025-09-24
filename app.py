# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Sistem Klasifikasi Penerima Bantuan Sosial", layout="wide")

# --------------------------
# Helper functions
# --------------------------

def find_column_by_keywords(columns, keywords):
    """Return the first column name that contains any of the keywords (case-insensitive)."""
    cols = list(columns)
    for kw in keywords:
        for c in cols:
            if kw.lower() in c.lower():
                return c
    return None

def read_uploaded_file(uploaded_file):
    """Read CSV or Excel into pandas DataFrame."""
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif filename.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            raise ValueError("Format file tidak didukung. Unggah .csv atau .xlsx.")
    except Exception as e:
        raise

def validate_dataset(df):
    """Checks presence of 'Status' and appropriate labels."""
    if df is None or df.shape[0] == 0:
        raise ValueError("Dataset kosong atau gagal dibaca.")
    # Status column must exist exactly 'Status' (case-sensitive per requirement)
    if "Status" not in df.columns:
        raise ValueError("Kolom target 'Status' tidak ditemukan. Pastikan nama kolom tepat 'Status'.")
    allowed = set(df["Status"].dropna().unique())
    expected = set(["Layak", "Tidak Layak"])
    if not allowed.issubset(expected):
        raise ValueError(f"Label 'Status' hanya boleh berisi 'Layak' dan 'Tidak Layak'. Ditemukan: {sorted(list(allowed))}")
    return True

def prepare_features(df, drop_cols=None):
    """Separate X and y and detect numerical & categorical features automatically (excluding drop_cols)."""
    if drop_cols is None: drop_cols = []
    # target
    y = df["Status"].copy()
    X = df.drop(columns=["Status"]).copy()
    # drop specified cols (like Nama, Kampung) from features
    X_for_model = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')
    # detect dtypes
    numeric_feats = X_for_model.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X_for_model.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return X_for_model, y, numeric_feats, cat_feats, X

def build_preprocessor(numeric_feats, cat_feats):
    """Create ColumnTransformer with imputers and encoders."""
    transformers = []
    if numeric_feats:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean"))
        ])
        transformers.append(("num", num_pipeline, numeric_feats))
    if cat_feats:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])
        transformers.append(("cat", cat_pipeline, cat_feats))
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    else:
        preprocessor = None
    return preprocessor

def train_nb_model(X, y, numeric_feats, cat_feats):
    """Train pipeline with preprocessor and GaussianNB. Returns fitted pipeline and simple metrics."""
    preprocessor = build_preprocessor(numeric_feats, cat_feats)
    # NB expects dense input; we ensure OneHotEncoder sparse=False
    clf = Pipeline([
        ("pre", preprocessor),
        ("nb", GaussianNB())
    ])
    # split (careful with stratify - follow spec: do not stratify if minority too small)
    try:
        # decide stratify or not
        if min(y.value_counts()) >= 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return clf, acc, (X_train, X_test, y_train, y_test)

def explain_prediction_simple(df_original, status_series):
    """
    Build simple class-conditional frequency tables and numeric means to be used later for local explanations.
    Returns:
        - freq_tables: dict of {feature: DataFrame(counts / freq per class)}
        - numeric_stats: dict of {feature: {class: mean}}
    """
    freq_tables = {}
    numeric_stats = {}
    classes = status_series.unique()
    for col in df_original.columns:
        if col == "Status":
            continue
        if pd.api.types.is_numeric_dtype(df_original[col]):
            numeric_stats[col] = df_original.groupby("Status")[col].mean().to_dict()
        else:
            # frequency per class
            freq = (df_original.groupby(["Status", col]).size().rename("count").reset_index())
            # convert to pivot for quick lookup
            pivot = freq.pivot(index=col, columns="Status", values="count").fillna(0)
            # convert to proportions per class:
            prop = pivot.copy()
            for c in prop.columns:
                total = prop[c].sum()
                if total > 0:
                    prop[c] = prop[c] / total
                else:
                    prop[c] = 0
            freq_tables[col] = prop
    return freq_tables, numeric_stats

def feature_contributors_for_row(row, predicted_class, freq_tables, numeric_stats, other_class):
    """
    For a single row (Series), compute per-original-feature score favoring predicted_class.
    - For categorical: use log(prop_pred + eps) - log(prop_other + eps)
    - For numeric: compare distance to class means: (|x-mean_other| - |x-mean_pred|) normalized by std-like
    Returns sorted list of (feature, score, reason_text)
    """
    eps = 1e-9
    scores = []
    for col in row.index:
        val = row[col]
        if col in numeric_stats:
            stats = numeric_stats[col]
            mean_pred = stats.get(predicted_class, np.nan)
            mean_other = stats.get(other_class, np.nan)
            # if means nan, skip
            if pd.isna(mean_pred) or pd.isna(mean_other):
                continue
            # score: how much closer to predicted mean than other mean
            score = (abs(val - mean_other) - abs(val - mean_pred))
            reason = f"Nilai {val} lebih dekat ke rata-rata {predicted_class} ({mean_pred:.2f}) daripada {other_class} ({mean_other:.2f})" if score>0 else f"Nilai {val} tidak mendukung kuat {predicted_class}"
            scores.append((col, float(score), reason))
        else:
            # categorical
            table = freq_tables.get(col)
            if table is None:
                continue
            try:
                # get proportions for that category value
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
    # sort by descending score
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores_sorted

def save_archive(record, path="archive.csv"):
    """Append a prediction summary record to CSV archive and to session_state."""
    cols = ["timestamp", "kampung", "num_layak", "num_tidak_layak", "details_path"]
    df_rec = pd.DataFrame([record])
    if os.path.exists(path):
        df_rec.to_csv(path, mode='a', header=False, index=False)
    else:
        df_rec.to_csv(path, index=False)
    # also maintain in session state
    if "archive" not in st.session_state:
        st.session_state["archive"] = []
    st.session_state["archive"].append(record)

def load_archive(path="archive.csv"):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            return pd.DataFrame(columns=["timestamp", "kampung", "num_layak", "num_tidak_layak", "details_path"])
    else:
        return pd.DataFrame(columns=["timestamp", "kampung", "num_layak", "num_tidak_layak", "details_path"])

# --------------------------
# UI: Sidebar navigation
# --------------------------
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Training", "Prediksi", "Arsip/Riwayat"])

# Shared state for model and data
if "df" not in st.session_state:
    st.session_state["df"] = None
if "trained_pipeline" not in st.session_state:
    st.session_state["trained_pipeline"] = None
if "feature_info" not in st.session_state:
    st.session_state["feature_info"] = None  # (freq_tables, numeric_stats)
if "drop_cols_for_model" not in st.session_state:
    st.session_state["drop_cols_for_model"] = []
if "last_training_summary" not in st.session_state:
    st.session_state["last_training_summary"] = None
if "archive" not in st.session_state:
    st.session_state["archive"] = []

# --------------------------
# Home
# --------------------------
if page == "Home":
    st.title("Sistem Klasifikasi Penerima Bantuan Sosial")
    st.markdown("""
    **Tujuan:** membantu menentukan siapa yang layak/tidak layak menerima bantuan sosial agar tepat sasaran.  
    **Metode klasifikasi:** *Naïve Bayes* (menggunakan `scikit-learn` — `GaussianNB`) dengan preprocessing otomatis (penanganan missing value dan encoding kategorikal).
    """)
    st.markdown("#### Alur penggunaan singkat")
    st.write("""
    1. Masuk ke halaman **Training** dan unggah file dataset (.csv atau .xlsx).  
    2. Sistem otomatis memproses data, melatih model Naïve Bayes, dan menampilkan ringkasan training (jumlah data, distribusi kelas, akurasi sederhana).  
    3. Ke halaman **Prediksi**: pilih *Nama Kampung* (dipilih dari dataset yang diunggah). Sistem akan menampilkan daftar warga kampung tersebut yang diprediksi **Layak** dan **Tidak Layak**, beserta *alasan singkat* fitur yang paling berpengaruh.  
    4. Hasil prediksi disimpan ke **Arsip/Riwayat** dan dapat diunduh.
    """)
    st.info("Catatan: kolom target harus bernama **'Status'** dan hanya berisi 'Layak' atau 'Tidak Layak'. Pastikan dataset juga memuat kolom nama warga (mengandung kata 'nama') dan kolom kampung/desa (mengandung kata 'kampung' atau 'desa').")

# --------------------------
# Training
# --------------------------
elif page == "Training":
    st.title("Halaman Training")
    st.markdown("Unggah dataset (.csv atau .xlsx). Kolom target harus bernama **Status** dan hanya berisi 'Layak' dan 'Tidak Layak'.")

    uploaded_file = st.file_uploader("Upload file dataset (.csv / .xlsx)", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        try:
            df = read_uploaded_file(uploaded_file)
            st.session_state["df_raw"] = df.copy()
            # Basic show
            st.write("Preview dataset (5 baris):")
            st.dataframe(df.head())

            # Validate
            try:
                validate_dataset(df)
            except Exception as e:
                st.error(str(e))
                st.stop()

            # Try to find name & kampung columns
            name_col = find_column_by_keywords(df.columns, ["nama ", "nama", "name", "nm"])
            kampung_col = find_column_by_keywords(df.columns, ["kampung", "desa", "kelurahan", "village"])
            if name_col is None or kampung_col is None:
                st.warning("Sistem tidak otomatis menemukan kolom nama warga atau kolom kampung/desa.")
                st.write("Kolom yang terdeteksi pada file:", list(df.columns))
                name_col = st.selectbox("Pilih kolom untuk 'Nama Warga'", options=[None]+list(df.columns), index=0)
                kampung_col = st.selectbox("Pilih kolom untuk 'Nama Kampung/Desa'", options=[None]+list(df.columns), index=0)
            else:
                st.success(f"Deteksi otomatis: Nama warga = '{name_col}', Kampung = '{kampung_col}'")

            if st.button("Mulai Training"):
                # Validate again
                if name_col is None or kampung_col is None:
                    st.error("Kolom nama warga atau kampung belum dipilih.")
                    st.stop()

                # Store chosen drop columns for modeling
                st.session_state["drop_cols_for_model"] = [name_col, kampung_col]

                # Prepare features
                X_for_model, y, numeric_feats, cat_feats, X_full = prepare_features(df, drop_cols=[name_col, kampung_col])
                st.write(f"Jumlah baris: {df.shape[0]}; Jumlah fitur untuk model: {X_for_model.shape[1]}")
                st.write("Fitur numerik terdeteksi:", numeric_feats)
                st.write("Fitur kategorikal terdeteksi:", cat_feats)

                # Train model
                with st.spinner("Melatih model Naïve Bayes..."):
                    try:
                        clf_pipeline, acc, splits = train_nb_model(X_for_model, y, numeric_feats, cat_feats)
                    except Exception as e:
                        st.error("Gagal melatih model: " + str(e))
                        st.stop()

                st.session_state["trained_pipeline"] = clf_pipeline
                st.session_state["df"] = df.copy()
                st.session_state["name_col"] = name_col
                st.session_state["kampung_col"] = kampung_col
                st.session_state["last_training_summary"] = {
                    "n_rows": df.shape[0],
                    "class_counts": df["Status"].value_counts().to_dict(),
                    "accuracy": float(acc)
                }

                # prepare explanation tables
                freq_tables, numeric_stats = explain_prediction_simple(df)
                st.session_state["feature_info"] = (freq_tables, numeric_stats)

                st.success("Training selesai ✅")
                st.write("Ringkasan training:")
                st.json(st.session_state["last_training_summary"])

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

    else:
        st.info("Belum ada file diunggah. Silakan unggah dataset di sini.")

# --------------------------
# Prediksi
# --------------------------
elif page == "Prediksi":
    st.title("Halaman Prediksi")
    if st.session_state.get("df") is None or st.session_state.get("trained_pipeline") is None:
        st.warning("Belum ada model terlatih atau dataset. Silakan lakukan proses Training terlebih dahulu.")
        st.stop()

    df = st.session_state["df"]
    name_col = st.session_state["name_col"]
    kampung_col = st.session_state["kampung_col"]
    pipeline = st.session_state["trained_pipeline"]
    freq_tables, numeric_stats = st.session_state["feature_info"]

    st.write(f"Dataset terpakai: {df.shape[0]} baris. Kolom nama warga: '{name_col}', kolom kampung: '{kampung_col}'")

    kampungs = sorted(df[kampung_col].dropna().unique().tolist())
    kampung_selected = st.selectbox("Pilih Nama Kampung", options=kampungs)

    if st.button("Jalankan Prediksi untuk Kampung ini"):
        # filter rows
        df_k = df[df[kampung_col] == kampung_selected].copy()
        if df_k.shape[0] == 0:
            st.warning("Tidak ada data untuk kampung yang dipilih.")
            st.stop()

        # Prepare X for model (drop name and kampung)
        drop_cols = st.session_state["drop_cols_for_model"]
        X_for_model = df_k.drop(columns=[c for c in drop_cols if c in df_k.columns] + ["Status"], errors='ignore')
        # Ensure columns order same as training pipeline expects:
        # The pipeline's preprocessor expects the feature names present during training. We attempt to reorder/align.
        try:
            # Predict
            preds = pipeline.predict(X_for_model)
        except Exception as e:
            st.error("Gagal memprediksi. Periksa apakah struktur kolom dataset cocok dengan saat training. Error: " + str(e))
            st.stop()

        df_k = df_k.copy()
        df_k["Prediction"] = preds

        # Display lists
        layak_df = df_k[df_k["Prediction"] == "Layak"]
        tidak_df = df_k[df_k["Prediction"] == "Tidak Layak"]

        st.subheader(f"Hasil Prediksi untuk Kampung: {kampung_selected}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Layak** — jumlah: {layak_df.shape[0]}")
            if layak_df.shape[0] > 0:
                st.dataframe(layak_df[[name_col]].reset_index(drop=True))
        with col2:
            st.markdown(f"**Tidak Layak** — jumlah: {tidak_df.shape[0]}")
            if tidak_df.shape[0] > 0:
                st.dataframe(tidak_df[[name_col]].reset_index(drop=True))

        # Provide explanations (top 2 features) for each person (but keep UI reasonable — show top 3)
        st.markdown("### Penjelasan singkat mengapa tiap warga diklasifikasikan demikian (fitur teratas)")
        explanations = []
        other_class_map = {"Layak": "Tidak Layak", "Tidak Layak": "Layak"}
        max_show = st.number_input("Jumlah warga yang ingin ditunjukkan penjelasannya (per halaman):", min_value=1, max_value=100, value=10, step=1)
        # iterate rows (limit to max_show)
        for idx, row in df_k.reset_index(drop=True).iterrows():
            if idx >= max_show:
                break
            person_name = row[name_col] if name_col in row.index else f"Baris {idx}"
            pred = row["Prediction"]
            other = other_class_map[pred]
            # compute contributors using original features (exclude name, kampung, target, prediction)
            row_features = row.drop(labels=[name_col, kampung_col, "Status", "Prediction"], errors='ignore')
            contributors = feature_contributors_for_row(row_features, pred, freq_tables, numeric_stats, other)
            top = contributors[:3]
            st.markdown(f"**{person_name}** — Prediksi: **{pred}**")
            if top:
                for f, s, reason in top:
                    st.write(f"- {f}: {reason}")
            else:
                st.write("- Tidak dapat mengekstrak fitur penyebab (data mungkin tidak memadai).")
            st.write("---")
            explanations.append({
                "name": person_name,
                "prediction": pred,
                "top_reasons": [r for (_, _, r) in top]
            })

        # Save to archive (create details csv for kampung)
        timestamp = datetime.now().isoformat()
        details_filename = f"pred_{kampung_selected}_{timestamp.replace(':','-')}.csv"
        # save full df_k with predictions
        try:
            df_k.to_csv(details_filename, index=False)
            details_path = details_filename
        except Exception:
            details_path = ""

        record = {
            "timestamp": timestamp,
            "kampung": kampung_selected,
            "num_layak": int(layak_df.shape[0]),
            "num_tidak_layak": int(tidak_df.shape[0]),
            "details_path": details_path
        }
        try:
            save_archive(record)
            st.success("Hasil prediksi disimpan ke arsip.")
        except Exception as e:
            st.error("Gagal menyimpan arsip: " + str(e))

        # Offer download of full details
        csv_buf = io.StringIO()
        df_k.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode()
        st.download_button("Unduh hasil prediksi (CSV)", data=csv_bytes, file_name=f"hasil_prediksi_{kampung_selected}.csv", mime="text/csv")

# --------------------------
# Arsip / Riwayat
# --------------------------
elif page == "Arsip/Riwayat":
    st.title("Arsip / Riwayat Prediksi")
    archive_df = load_archive()
    # merge with session_state archive if exists (session entries may be duplicates — keep both)
    if len(st.session_state.get("archive", [])) > 0:
        extra = pd.DataFrame(st.session_state["archive"])
        if not extra.empty:
            archive_df = pd.concat([archive_df, extra], ignore_index=True)
            # drop duplicates
            archive_df = archive_df.drop_duplicates(subset=["timestamp", "kampung"], keep="first")
    if archive_df.empty:
        st.info("Belum ada hasil prediksi yang diarsipkan.")
    else:
        st.dataframe(archive_df.sort_values(by="timestamp", ascending=False).reset_index(drop=True))
        # allow download whole archive
        csv_bytes = archive_df.to_csv(index=False).encode()
        st.download_button("Unduh arsip lengkap (CSV)", data=csv_bytes, file_name="arsip_prediksi.csv", mime="text/csv")
