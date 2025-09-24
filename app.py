# app.py
import streamlit as st
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------
# Konfigurasi dasar
# -------------------------
st.set_page_config(page_title="Klasifikasi Bantuan Sosial - Desa Cikembar", layout="wide")
st.title("📊 Penerapan Algoritma Naïve Bayes")
st.write("Upload dataset (.csv / .xlsx) — sistem akan otomatis memilih kolom target dan melatih model.")

# Inisialisasi session state
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = []
if "feature_meta" not in st.session_state:
    st.session_state.feature_meta = {}
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

# -------------------------
# Util: pilih target otomatis
# -------------------------
def auto_select_target(df):
    """
    Heuristik memilih kolom target:
    - Cari kolom dengan jumlah kelas antara 2 dan max(50, 0.5 * n_rows)
    - Prioritaskan kolom dengan sedikit kelas (lebih mudah untuk klasifikasi)
    - Jika tidak ada kandidat, pakai kolom terakhir jika tidak merupakan ID unik untuk setiap baris
    - Jika tidak bisa, kembalikan None
    """
    n = len(df)
    max_unique = max(50, int(0.5 * n))
    candidates = []
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        if 2 <= nunique <= max_unique:
            candidates.append((col, nunique, df[col].dtype))
    if candidates:
        # urutkan: prefer kolom dengan kecil nunique, dan prefer object dtype (kategori)
        candidates.sort(key=lambda x: (x[1], 0 if str(x[2]).startswith("object") else 1))
        return candidates[0][0]
    # fallback: last column kalau tidak semua unik
    last = df.columns[-1]
    if df[last].nunique(dropna=True) < n:
        if df[last].nunique(dropna=True) >= 2:
            return last
    return None

# -------------------------
# Util: deteksi tipe fitur + konversi ringan
# -------------------------
def detect_and_prepare_X(X):
    """
    Mengembalikan:
    - X2: salinan X dengan beberapa konversi (numeric coercion jika banyak nilai numeric)
    - num_cols, cat_cols: daftar nama kolom numerik dan kategorikal
    - feature_meta: metadata per fitur (type, defaults, options)
    """
    X2 = X.copy()
    num_cols = []
    cat_cols = []
    feature_meta = {}
    for col in X2.columns:
        ser = X2[col]
        # rasio nilai yang bisa dikonversi ke numeric
        parsed = pd.to_numeric(ser, errors='coerce')
        frac_numeric = parsed.notna().mean()
        if frac_numeric >= 0.9:
            # anggap numeric -> gunakan parsed
            X2[col] = parsed
            num_cols.append(col)
            feature_meta[col] = {
                "type": "numeric",
                "mean": float(parsed.mean(skipna=True)) if parsed.notna().any() else 0.0,
                "min": float(parsed.min(skipna=True)) if parsed.notna().any() else 0.0,
                "max": float(parsed.max(skipna=True)) if parsed.notna().any() else 0.0
            }
        else:
            # anggap categorical -> ubah ke string (untuk kestabilan one-hot)
            X2[col] = ser.fillna("___NaN___").astype(str)
            cat_vals = pd.Series(X2[col].unique()).astype(str).tolist()
            # batasi jumlah options yang ditampilkan di UI (untuk selectbox)
            feature_meta[col] = {"type": "categorical", "values": cat_vals[:500]}
            cat_cols.append(col)
    return X2, num_cols, cat_cols, feature_meta

# -------------------------
# Halaman: Pelatihan (otomatis training on upload)
# -------------------------
st.header("📥 Upload & Training (otomatis)")

uploaded_file = st.file_uploader("Upload CSV atau Excel (.csv / .xlsx)", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # create simple file id to avoid retrain on every rerun unless file changed
    try:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    except Exception:
        file_id = uploaded_file.name

    # read file (handle openpyxl missing)
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            try:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            except ImportError:
                st.error("Paket `openpyxl` belum terpasang di environment. Tambahkan 'openpyxl' ke requirements.txt dan redeploy.")
                st.stop()
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    # info
    st.subheader("Info dataset")
    buf = io.StringIO()
    df.info(buf=buf)
    st.text(buf.getvalue())
    st.write("Jumlah baris:", df.shape[0], " | Jumlah kolom:", df.shape[1])
    st.write("Missing per kolom:")
    st.write(df.isnull().sum())

    # Auto-pick target
    target = auto_select_target(df)
    if target is None:
        st.error("Tidak dapat memilih kolom target secara otomatis. Kemungkinan semua kolom unik (seperti Nama/ID). "
                 "Solusi: pastikan dataset berisi kolom target (label) dengan 2..n kelas, atau tambahkan kolom label terakhir.")
        st.stop()

    st.write(f"🎯 Kolom target terpilih otomatis: **{target}**")
    features = [c for c in df.columns if c != target]
    st.write(f"🧩 Fitur yang dipakai (otomatis): {features}")

    # only retrain when new file uploaded
    if st.session_state.last_file_id != file_id:
        st.session_state.last_file_id = file_id
        # prepare data
        X_raw = df[features]
        y = df[target]

        # detect & prepare
        X, num_cols, cat_cols, feature_meta = detect_and_prepare_X(X_raw)

        # build transformers (only include if non-empty)
        transformers = []
        if len(num_cols) > 0:
            transformers.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols))
        if len(cat_cols) > 0:
            # OneHotEncoder compatibility
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe)
            ]), cat_cols))

        if not transformers:
            st.error("Tidak ada fitur numerik atau kategorikal yang valid untuk dilatih.")
            st.stop()

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        pipeline = Pipeline([("preprocessor", preprocessor), ("clf", GaussianNB())])

        # split data (try stratify, fallback if error)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        except Exception:
            st.warning("Stratify gagal (kelas tidak cukup). Membagi data tanpa stratify.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # training
        with st.spinner("Melatih model..."):
            try:
                pipeline.fit(X_train, y_train)
            except Exception as e:
                st.error(f"Gagal melatih model: {e}")
                st.stop()

        # evaluasi
        try:
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
        except Exception as e:
            st.error(f"Gagal evaluasi model setelah training: {e}")
            st.stop()

        st.success(f"Model terlatih — Akurasi (test): {acc:.3f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        try:
            clf = pipeline.named_steps["clf"]
            classes = clf.classes_
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=[str(c) for c in classes],
                        yticklabels=[str(c) for c in classes], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Gagal menampilkan confusion matrix: {e}")

        # simpan model & metadata ke session_state
        st.session_state.model = pipeline
        st.session_state.features = features
        st.session_state.feature_meta = feature_meta
        st.session_state.target_col = target
        st.info("Model dan metadata fitur disimpan di session. Buka tab Prediksi Data Baru untuk coba input manual.")

    else:
        st.info("File sama seperti sebelumnya — model sudah terlatih dan disimpan di session. Buka halaman Prediksi.")

# -------------------------
# Halaman: Prediksi Data Baru
# -------------------------
st.markdown("---")
st.header("🔮 Prediksi Data Baru")

if st.session_state.model is None:
    st.warning("Belum ada model terlatih. Upload dataset dan tunggu proses training otomatis di atas.")
else:
    model = st.session_state.model
    features = st.session_state.features
    meta = st.session_state.feature_meta
    target_col = st.session_state.target_col

    st.write(f"Model terlatih untuk target: **{target_col}**")
    st.subheader("Masukkan data (satu record) untuk diprediksi")

    input_data = {}
    cols_left = features[:]  # maintain order
    for col in cols_left:
        m = meta.get(col, {})
        if m.get("type") == "numeric":
            default = m.get("mean", 0.0)
            val = st.number_input(f"{col} (numeric)", value=float(default))
            input_data[col] = val
        else:
            opts = m.get("values", [])
            if opts and len(opts) <= 100:
                # tampilkan selectbox kalau opsi tidak terlalu banyak
                val = st.selectbox(f"{col} (kategori)", options=opts, index=0)
            else:
                val = st.text_input(f"{col} (kategori/free text)")
            input_data[col] = str(val) if val is not None else ""

    if st.button("Prediksi"):
        Xnew = pd.DataFrame([input_data], columns=features)
        # coerce numeric columns
        for col in features:
            if meta.get(col, {}).get("type") == "numeric":
                Xnew[col] = pd.to_numeric(Xnew[col], errors="coerce")
        try:
            pred = model.predict(Xnew)[0]
            st.success(f"Hasil prediksi ({target_col}): {pred}")
            if hasattr(model.named_steps["clf"], "predict_proba"):
                probs = model.predict_proba(Xnew)[0]
                classes = model.named_steps["clf"].classes_
                proba_dict = {str(c): float(p) for c, p in zip(classes, probs)}
                st.write("Probabilitas per kelas:")
                st.json(proba_dict)
        except Exception as e:
            st.error(f"Gagal prediksi: {e}")
