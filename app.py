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
st.set_page_config(page_title="Sistem Klasifikasi - Desa Cikembar", layout="wide")

# -------------------------
# State
# -------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = []
if "feature_meta" not in st.session_state:
    st.session_state.feature_meta = {}
if "target_col" not in st.session_state:
    st.session_state.target_col = None

# -------------------------
# Util: pilih target otomatis
# -------------------------
def auto_select_target(df):
    n = len(df)
    max_unique = max(50, int(0.5 * n))
    candidates = []
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        if 2 <= nunique <= max_unique:
            candidates.append((col, nunique, df[col].dtype))
    if candidates:
        candidates.sort(key=lambda x: (x[1], 0 if str(x[2]).startswith("object") else 1))
        return candidates[0][0]
    last = df.columns[-1]
    if df[last].nunique(dropna=True) < n and df[last].nunique(dropna=True) >= 2:
        return last
    return None

# -------------------------
# Util: deteksi fitur
# -------------------------
def detect_and_prepare_X(X):
    X2 = X.copy()
    num_cols, cat_cols = [], []
    feature_meta = {}
    for col in X2.columns:
        ser = X2[col]
        parsed = pd.to_numeric(ser, errors="coerce")
        frac_numeric = parsed.notna().mean()
        if frac_numeric >= 0.9:
            X2[col] = parsed
            num_cols.append(col)
            feature_meta[col] = {
                "type": "numeric",
                "mean": float(parsed.mean(skipna=True)) if parsed.notna().any() else 0.0,
            }
        else:
            X2[col] = ser.fillna("___NaN___").astype(str)
            cat_vals = pd.Series(X2[col].unique()).astype(str).tolist()
            feature_meta[col] = {"type": "categorical", "values": cat_vals[:200]}
            cat_cols.append(col)
    return X2, num_cols, cat_cols, feature_meta

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["1. Upload Dataset", "2. Training Model", "3. Prediksi Baru"])

# -------------------------
# Halaman 1: Upload & Preview
# -------------------------
if page == "1. Upload Dataset":
    st.title("📥 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()
        st.session_state.df = df

        st.subheader("Preview Data")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)

        buf = io.StringIO()
        df.info(buf=buf)
        st.text(buf.getvalue())
        st.write("Missing values per kolom:")
        st.write(df.isnull().sum())

# -------------------------
# Halaman 2: Training Model
# -------------------------
elif page == "2. Training Model":
    st.title("📊 Training Model Naïve Bayes")
    if st.session_state.df is None:
        st.warning("Upload dataset dulu di halaman 1.")
    else:
        df = st.session_state.df
        target = auto_select_target(df)
        if not target:
            st.error("Tidak bisa menemukan kolom target yang cocok. Pastikan dataset ada kolom label.")
            st.stop()

        features = [c for c in df.columns if c != target]
        X_raw, y = df[features], df[target]
        X, num_cols, cat_cols, feature_meta = detect_and_prepare_X(X_raw)

        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols))
        if cat_cols:
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe)
            ]), cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers)
        pipeline = Pipeline([("prep", preprocessor), ("clf", GaussianNB())])

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        except Exception:
            st.warning("Stratify gagal, gunakan split acak biasa.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        with st.spinner("Melatih model..."):
            pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Akurasi: {acc:.3f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # simpan
        st.session_state.model = pipeline
        st.session_state.features = features
        st.session_state.feature_meta = feature_meta
        st.session_state.target_col = target

# -------------------------
# Halaman 3: Prediksi Baru
# -------------------------
elif page == "3. Prediksi Baru":
    st.title("🔮 Prediksi Data Baru")
    if st.session_state.model is None:
        st.warning("Latih model dulu di halaman 2.")
    else:
        model = st.session_state.model
        features = st.session_state.features
        meta = st.session_state.feature_meta
        target_col = st.session_state.target_col

        st.write(f"Target: {target_col}")
        input_data = {}
        for col in features:
            m = meta.get(col, {})
            if m.get("type") == "numeric":
                val = st.number_input(col, value=float(m.get("mean", 0.0)))
            else:
                opts = m.get("values", [])
                if opts and len(opts) < 100:
                    val = st.selectbox(col, options=opts)
                else:
                    val = st.text_input(col)
            input_data[col] = val

        if st.button("Prediksi"):
            Xnew = pd.DataFrame([input_data], columns=features)
            for col in features:
                if meta.get(col, {}).get("type") == "numeric":
                    Xnew[col] = pd.to_numeric(Xnew[col], errors="coerce")
            pred = model.predict(Xnew)[0]
            st.success(f"Hasil prediksi: {pred}")
            if hasattr(model.named_steps["clf"], "predict_proba"):
                probs = model.predict_proba(Xnew)[0]
                st.write("Probabilitas kelas:", dict(zip(model.named_steps["clf"].classes_, probs)))
