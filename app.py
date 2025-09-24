import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Penerima Bantuan Sosial", layout="wide")

# Fungsi untuk membuat dataset dummy (jika tidak ada file data)
@st.cache_data
def load_data():
    # Dataset dummy untuk simulasi data warga Desa Cikembar
    data = {
        'usia_kepala_keluarga': [45, 30, 55, 40, 60, 35, 50, 28, 65, 42, 38, 52, 47, 33, 58],
        'pendapatan_bulanan': [1500000, 800000, 2000000, 1200000, 500000, 900000, 1800000, 600000, 400000, 1100000, 700000, 2200000, 1300000, 850000, 450000],
        'jumlah_anggota_keluarga': [4, 6, 3, 5, 2, 7, 4, 8, 1, 5, 6, 3, 4, 7, 2],
        'memiliki_rumah': ['Tidak', 'Tidak', 'Ya', 'Tidak', 'Tidak', 'Tidak', 'Ya', 'Tidak', 'Tidak', 'Tidak', 'Tidak', 'Ya', 'Ya', 'Tidak', 'Tidak'],
        'berhak_bantuan': [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1]  # 1: Berhak, 0: Tidak Berhak
    }
    df = pd.DataFrame(data)
    return df

# Fungsi untuk melatih model
@st.cache_resource
def train_model():
    df = load_data()
    
    # Preprocessing
    le_rumah = LabelEncoder()
    df['memiliki_rumah_encoded'] = le_rumah.fit_transform(df['memiliki_rumah'])
    
    X = df[['usia_kepala_keluarga', 'pendapatan_bulanan', 'jumlah_anggota_keluarga', 'memiliki_rumah_encoded']]
    y = df['berhak_bantuan']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Simpan model
    with open('naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le_rumah, f)
    
    return model, accuracy, classification_report(y_test, y_pred, output_dict=True)

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    if os.path.exists('naive_bayes_model.pkl'):
        with open('naive_bayes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le_rumah = pickle.load(f)
        return model, le_rumah
    else:
        return None, None

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman:", ["Informasi Sistem", "Pelatihan Model", "Prediksi Hasil"])

# Halaman 1: Informasi Sistem (menggunakan teks persis dari permintaan)
if page == "Informasi Sistem":
    st.title("Klasifikasi Penerima Bantuan Sosial di Desa Cikembar menggunakan Algoritma Naive Bayes")
    
    st.header("Deskripsi Sistem")
    st.markdown("""
    Sistem ini dirancang untuk mengklasifikasikan warga Desa Cikembar yang berhak menerima bantuan sosial berdasarkan data demografis dan ekonomi mereka. Menggunakan algoritma Naive Bayes, sistem ini memproses fitur-fitur seperti usia kepala keluarga, pendapatan bulanan, jumlah anggota keluarga, dan status kepemilikan rumah untuk memprediksi apakah seorang warga layak mendapatkan bantuan atau tidak.

    Data yang digunakan bersifat simulasi untuk demonstrasi, tetapi dapat diganti dengan data real dari desa.
    """)
    
    st.header("Tujuan Sistem")
    st.markdown("""
    Memberikan rekomendasi akurat untuk distribusi bantuan sosial agar tepat sasaran.
    Mengoptimalkan proses seleksi penerima bantuan menggunakan machine learning.
    Membantu pemerintah desa dalam pengambilan keputusan berbasis data untuk mengurangi kemiskinan dan ketidakadilan sosial.
    """)
    
    st.header("Manfaat Sistem")
    st.markdown("""
    Efisiensi: Mengurangi waktu dan biaya manual dalam verifikasi penerima bantuan.
    Akurasi: Algoritma Naive Bayes memberikan prediksi probabilistik yang andal berdasarkan asumsi independensi fitur.
    Transparansi: Sistem dapat menampilkan alasan prediksi, sehingga proses lebih transparan.
    Skalabilitas: Dapat diintegrasikan dengan data real-time untuk pemantauan berkelanjutan di Desa Cikembar.
    """)
    
    # Placeholder image (opsional, bisa diganti dengan gambar real)
    st.image("https://via.placeholder.com/800x400?text=Desa+Cikembar")

# Halaman 2: Pelatihan Model
elif page == "Pelatihan Model":
    st.title("Pelatihan Model Naive Bayes")
    
    if st.button("Latih Model"):
        with st.spinner("Melatih model..."):
            model, accuracy, report = train_model()
            st.success("Model berhasil dilatih!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Akurasi Model", f"{accuracy:.2%}")
            with col2:
                st.metric("Jumlah Data Pelatihan", len(load_data()))
            
            st.subheader("Laporan Klasifikasi")
            st.write("Precision, Recall, dan F1-Score untuk kelas:")
            st.json(report)
            
            st.subheader("Dataset yang Digunakan")
            df = load_data()
            st.dataframe(df)
    
    else:
        model, le = load_model()
        if model is not None:
            st.info("Model sudah tersedia dari pelatihan sebelumnya.")
        else:
            st.warning("Belum ada model yang dilatih. Tekan tombol untuk melatih.")

# Halaman 3: Prediksi Hasil
elif page == "Prediksi Hasil":
    st.title("Prediksi Penerima Bantuan Sosial")
    
    model, le_rumah = load_model()
    if model is None:
        st.warning("Model belum dilatih. Silakan latih model terlebih dahulu di halaman Pelatihan Model.")
        st.stop()
    
    st.header("Input Data Warga")
    col1, col2 = st.columns(2)
    with col1:
        usia = st.number_input("Usia Kepala Keluarga", min_value=18, max_value=100, value=40)
        pendapatan = st.number_input("Pendapatan Bulanan (Rp)", min_value=0, max_value=5000000, value=1000000)
        jumlah_anggota = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=20, value=4)
    
    with col2:
        memiliki_rumah = st.selectbox("Memiliki Rumah Sendiri?", ["Ya", "Tidak"])
    
    if st.button("Prediksi"):
        # Preprocessing input
        rumah_encoded = le_rumah.transform([memiliki_rumah])[0]
        input_data = np.array([[usia, pendapatan, jumlah_anggota, rumah_encoded]])
        
        # Prediksi
        prediksi = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        
        st.header("Hasil Prediksi")
        if prediksi == 1:
            st.success("✅ **Berhak Menerima Bantuan Sosial**")
            st.write("Warga ini layak mendapatkan bantuan berdasarkan kriteria: usia rendah, pendapatan rendah, keluarga besar, dan/atau tidak memiliki rumah.")
        else:
            st.error("❌ **Tidak Berhak / Belum Layak**")
            st.write("Warga ini tidak memenuhi kriteria utama untuk bantuan saat ini. Saran: Periksa ulang data atau tunggu evaluasi lebih lanjut.")
        
        st.subheader("Probabilitas")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilitas Berhak", f"{prob[1]:.2%}")
        with col2:
            st.metric("Probabilitas Tidak Berhak", f"{prob[0]:.2%}")
        
        # Contoh prediksi batch (opsional)
        st.subheader("Contoh Prediksi untuk Beberapa Warga")
        df = load_data()
        df['memiliki_rumah_encoded'] = le_rumah.transform(df['memiliki_rumah'])
        sample_X = df[['usia_kepala_keluarga', 'pendapatan_bulanan', 'jumlah_anggota_keluarga', 'memiliki_rumah_encoded']]
        sample_pred = model.predict(sample_X)
        sample_prob = model.predict_proba(sample_X)
        sample_df = df.copy()
        sample_df['Prediksi'] = ['Berhak' if p == 1 else 'Tidak Berhak' for p in sample_pred]
        sample_df['Prob Berhak'] = [f"{prob[1]:.2%}" for prob in sample_prob]
        st.dataframe(sample_df[['usia_kepala_keluarga', 'pendapatan_bulanan', 'jumlah_anggota_keluarga', 'memiliki_rumah', 'Prediksi', 'Prob Berhak']])
