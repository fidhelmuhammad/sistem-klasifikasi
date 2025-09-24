import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# Konfigurasi dasar
# ===============================
st.set_page_config(page_title="Naive Bayes Bantuan Sosial", layout="wide")

# State untuk simpan model dan fitur
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = []
if "target" not in st.session_state:
    st.session_state.target = None

# ===============================
# Sidebar Menu
# ===============================
menu = st.sidebar.radio(
    "Navigasi", 
    ["🏠 Beranda", "📊 Pelatihan Model", "🔮 Prediksi Baru"]
)

# ===============================
# Halaman 1: Beranda
# ===============================
if menu == "🏠 Beranda":
    st.title("📊 Penerapan Algoritma Naïve Bayes")
    st.subheader("Klasifikasi Penerima Bantuan Sosial di Desa Cikembar")

    st.markdown("""
    Selamat datang di sistem klasifikasi penerima bantuan sosial berbasis **Naïve Bayes**.  
    Sistem ini dapat membantu menentukan apakah warga **layak** atau **tidak layak** menerima bantuan.  

    **Fitur utama sistem:**
    1. Upload dataset (CSV/Excel)  
    2. Latih model Naïve Bayes dan lihat hasil evaluasi  
    3. Prediksi data baru dengan input manual  

    👉 Silakan gunakan menu di **sidebar** untuk navigasi.
    """)

# ===============================
# Halaman 2: Pelatihan Model
# ===============================
elif menu == "📊 Pelatihan Model":
    st.header("📊 Pelatihan Model Naïve Bayes")

    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Preview Dataset")
        st.dataframe(df.head())

        target = st.selectbox("Pilih kolom target (label)", df.columns)
        features = st.multiselect("Pilih kolom fitur (predictors)", [col for col in df.columns if col != target])

        if features and target:
            X = df[features]
            y = df[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Model Naive Bayes
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Simpan model ke session_state
            st.session_state.model = model
            st.session_state.features = features
            st.session_state.target = target

            # Evaluasi
            st.write("### Hasil Evaluasi Model")
            st.write("Akurasi:", round(accuracy_score(y_test, y_pred), 3))
            st.write("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("Classification Report")
            st.text(classification_report(y_test, y_pred))
    else:
        st.info("Silakan upload dataset terlebih dahulu.")

# ===============================
# Halaman 3: Prediksi Baru
# ===============================
elif menu == "🔮 Prediksi Baru":
    st.header("🔮 Prediksi Data Baru")

    if st.session_state.model is not None and st.session_state.features:
        input_data = []
        st.write("Masukkan data sesuai fitur berikut:")
        for col in st.session_state.features:
            val = st.number_input(f"{col}", value=0.0)
            input_data.append(val)

        if st.button("Prediksi"):
            pred = st.session_state.model.predict([input_data])
            st.success(f"Hasil Prediksi: **{pred[0]}**")
    else:
        st.warning("⚠️ Model belum dilatih. Silakan latih model di menu **Pelatihan Model** terlebih dahulu.")
