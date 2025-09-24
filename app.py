import streamlit as st
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", page_icon="📑", layout="wide")

# Sidebar untuk navigasi
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "📊 Pelatihan Model", "🔮 Prediksi Data Baru"]
)

# ================================
# Halaman 1: Home
# ================================
if menu == "🏠 Home":
    st.title("📑 Penerapan Algoritma Naïve Bayes")
    st.subheader("Klasifikasi Penerima Bantuan Sosial di Desa Cikembar")

    st.markdown("""
    Selamat datang di sistem klasifikasi penerima bantuan sosial.  
    Gunakan menu di sidebar untuk:
    - 📊 **Pelatihan Model** → upload dataset, pilih label & fitur, latih model.
    - 🔮 **Prediksi Data Baru** → uji model dengan data input manual.
    """)

# ================================
# Halaman 2: Pelatihan Model
# ================================
elif menu == "📊 Pelatihan Model":
    st.title("📊 Pelatihan Model Naïve Bayes")

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel (.csv / .xls / .xlsx)",
        type=["csv", "xls", "xlsx"]
    )

    if uploaded_file:
        # Load dataset
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        st.subheader("🔍 Preview Dataset")
        st.dataframe(df.head())

        # Info dataset
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        # Pilih kolom target
        target_col = st.selectbox("🎯 Pilih kolom target (label)", df.columns)

        # Pilih fitur
        feature_cols = st.multiselect(
            "🧩 Pilih kolom fitur (predictors)",
            [c for c in df.columns if c != target_col]
        )

        if st.button("🚀 Latih Model"):
            if not feature_cols or not target_col:
                st.error("Pilih fitur dan target terlebih dahulu!")
                st.stop()

            X = df[feature_cols]
            y = df[target_col]

            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                st.warning("⚠️ Stratify gagal karena ada kelas dengan jumlah terlalu sedikit. Data dibagi tanpa stratify.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            # Pisahkan numerik & kategorikal
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
            cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
            ])

            model = Pipeline([
                ("preprocess", preprocessor),
                ("clf", GaussianNB())
            ])

            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            st.success(f"✅ Akurasi Model: {acc:.2f}")
            st.text("Laporan Klasifikasi:")
            st.text(classification_report(y_test, y_pred))

            # Simpan model di session
            st.session_state["model"] = model
            st.session_state["features"] = feature_cols

# ================================
# Halaman 3: Prediksi Data Baru
# ================================
elif menu == "🔮 Prediksi Data Baru":
    st.title("🔮 Prediksi Data Baru")

    if "model" not in st.session_state:
        st.warning("⚠️ Belum ada model. Latih dulu di halaman 📊 Pelatihan Model.")
        st.stop()

    model = st.session_state["model"]
    features = st.session_state["features"]

    st.subheader("📝 Input Data Penduduk")

    input_data = {}
    for col in features:
        val = st.text_input(f"Masukkan nilai untuk {col}")
        input_data[col] = val

    if st.button("🔍 Prediksi"):
        df_new = pd.DataFrame([input_data])
        try:
            pred = model.predict(df_new)[0]
            st.success(f"🎯 Hasil Prediksi: **{pred}**")
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
