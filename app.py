import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Konfigurasi halaman utama
st.set_page_config(page_title="Naive Bayes Bantuan Sosial", layout="wide")

# Sidebar menu
menu = st.sidebar.radio("Navigasi", ["🏠 Beranda", "📊 Pelatihan Model", "🔮 Prediksi Baru"])

# Variabel global (untuk menyimpan model & fitur terpilih)
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = []
if "target" not in st.session_state:
    st.session_state.target = None


# ================= HALAMAN 1: BERANDA =================
if menu == "🏠 Beranda":
    st.title("📊 Penerapan Algoritma Naïve Bayes")
    st.subheader("Klasifikasi Penerima Bantuan Sosial di Desa Cikembar")

    st.markdown("""
    Sistem ini dibuat untuk membantu menentukan apakah warga **layak** atau **tidak layak** 
    menerima bantuan sosial di Desa Cikembar berdasarkan data yang tersedia.  
    
    🔹 **Algoritma yang digunakan:** Naïve Bayes  
    🔹 **Fitur utama:**  
    1. Upload dataset  
    2. Latih model & evaluasi performa  
    3. Prediksi data baru  

    Silakan gunakan menu di **sidebar** untuk mengakses halaman lain.
    """)


# ================= HALAMAN 2: PELATIHAN MODEL =================
elif menu == "📊 Pelatihan Model":
    st.header("📊 Pelatihan Model Naïve Bayes")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview Dataset")
        st.dataframe(df.head())

        target = st.selectbox("Pilih kolom target (label)", df.columns)
        features = st.multiselect("Pilih kolom fitur", [col for col in df.columns if col != target])

        if features and target:
            X = df[features]
            y = df[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
            st.write("Akurasi:", accuracy_score(y_test, y_pred))
            st.write("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("Classification Report")
            st.text(classification_report(y_test, y_pred))
    else:
        st.info("Silakan upload dataset terlebih dahulu.")


# ================= HALAMAN 3: PREDIKSI BARU =================
elif menu == "🔮 Prediksi Baru":
    st.header("🔮 Prediksi Data Baru")

    if st.session_state.model is not None and st.session_state.features:
        input_data = []
        for col in st.session_state.features:
            val = st.number_input(f"Masukkan nilai untuk {col}", value=0.0)
            input_data.append(val)

        if st.button("Prediksi"):
            pred = st.session_state.model.predict([input_data])
            st.success(f"Hasil Prediksi: **{pred[0]}**")
    else:
        st.warning("⚠️ Model belum dilatih. Silakan latih model di menu **Pelatihan Model** terlebih dahulu.")
