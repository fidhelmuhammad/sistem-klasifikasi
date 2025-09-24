import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime
import io  # Untuk handling download Excel

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

# Inisialisasi session state
if 'riwayat_prediksi' not in st.session_state:
    st.session_state.riwayat_prediksi = []

if 'model' not in st.session_state:
    st.session_state.model = None

if 'le_rumah' not in st.session_state:
    st.session_state.le_rumah = None

if 'le_target' not in st.session_state:
    st.session_state.le_target = None

if 'dataset' not in st.session_state:
    st.session_state.dataset = None

# Fungsi untuk membuat dataset dummy sesuai struktur baru
@st.cache_data
def load_dummy_data():
    data = {
        'Nama': ['Ahmad', 'Siti', 'Budi', 'Dewi', 'Eko', 'Fani', 'Gatot', 'Hani', 'Indra', 'Joko'],
        'Jenis Kelamin': ['Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Laki-laki'],
        'Desa': ['Cikembar', 'Cikembar', 'Cikembar', 'Cikembar', 'Cikembar', 'Cikembar', 'Cikembar', 'Cikembar', 'Cikembar', 'Cikembar'],
        'Alamat': ['Jl. Merdeka 1', 'Jl. Sudirman 2', 'Jl. Gatot Subroto 3', 'Jl. Thamrin 4', 'Jl. Sudirman 5', 'Jl. Merdeka 6', 'Jl. Gatot Subroto 7', 'Jl. Thamrin 8', 'Jl. Sudirman 9', 'Jl. Merdeka 10'],
        'RT': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'RW': [1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
        'Jumlah_Anggota_Keluarga': [4, 6, 3, 5, 2, 7, 4, 8, 1, 5],
        'Usia_Kepala_Keluarga': [45, 30, 55, 40, 60, 35, 50, 28, 65, 42],
        'Pendidikan_Kepala_Keluarga': ['SD', 'SMP', 'SMA', 'SD', 'SMP', 'SMA', 'SD', 'SMP', 'SMA', 'SD'],
        'Pekerjaan_Kepala_Keluarga': ['Petani', 'Ibu Rumah Tangga', 'Pegawai Negeri', 'Buruh', 'Petani', 'Wiraswasta', 'Buruh', 'Ibu Rumah Tangga', 'Pensiunan', 'Petani'],
        'Pendapatan_Bulanan': [1500000, 800000, 2000000, 1200000, 500000, 900000, 1800000, 600000, 400000, 1100000],
        'Pengeluaran_Bulanan': [1400000, 750000, 1900000, 1150000, 480000, 850000, 1700000, 580000, 380000, 1050000],
        'Kepemilikan_Rumah': ['Tidak', 'Tidak', 'Ya', 'Tidak', 'Tidak', 'Tidak', 'Ya', 'Tidak', 'Tidak', 'Tidak'],
        'Jenis_Rumah': ['Sewa', 'Sewa', 'Milik', 'Sewa', 'Sewa', 'Sewa', 'Milik', 'Sewa', 'Sewa', 'Sewa'],
        'Sumber_Air': ['PDAM', 'Sumur', 'PDAM', 'Sumur', 'Sungai', 'PDAM', 'Sumur', 'Sungai', 'PDAM', 'Sumur'],
        'Sumber_Listrik': ['PLN', 'PLN', 'PLN', 'Genset', 'PLN', 'PLN', 'Genset', 'PLN', 'PLN', 'Genset'],
        'Aset_Dimiliki': ['Sepeda Motor', 'Tidak Ada', 'Mobil, Sepeda Motor', 'Sepeda Motor', 'Tidak Ada', 'Sepeda Motor', 'Tidak Ada', 'Sepeda Motor', 'Tidak Ada', 'Sepeda Motor'],
        'BPJS': ['Ya', 'Tidak', 'Ya', 'Tidak', 'Tidak', 'Ya', 'Tidak', 'Tidak', 'Ya', 'Tidak'],
        'Status_Kesejahteraan': ['Tidak Layak', 'Layak', 'Tidak Layak', 'Layak', 'Layak', 'Layak', 'Tidak Layak', 'Layak', 'Layak', 'Layak']
    }
    return pd.DataFrame(data)

# Fungsi untuk melatih model (disesuaikan dengan kolom baru)
def train_model(data):
    df = data.copy()
    
    # Validasi kolom kategorikal (handle NaN)
    if 'Kepemilikan_Rumah' not in df.columns or len(df) == 0:
        st.error("Data tidak valid atau kolom 'Kepemilikan_Rumah' hilang.")
        return None, None, None, 0, {}
    unique_rumah = set(df['Kepemilikan_Rumah'].dropna().unique())
    if len(unique_rumah) > 2 or unique_rumah - {'Ya', 'Tidak'}:
        st.error("Kolom 'Kepemilikan_Rumah' harus hanya berisi 'Ya' atau 'Tidak'.")
        return None, None, None, 0, {}
    
    if 'Status_Kesejahteraan' not in df.columns or len(df) == 0:
        st.error("Data tidak valid atau kolom 'Status_Kesejahteraan' hilang.")
        return None, None, None, 0, {}
    unique_target = set(df['Status_Kesejahteraan'].dropna().unique())
    if len(unique_target) > 2 or unique_target - {'Layak', 'Tidak Layak'}:
        st.error("Kolom 'Status_Kesejahteraan' harus hanya berisi 'Layak' atau 'Tidak Layak'.")
        return None, None, None, 0, {}
    
    # Encoding untuk Kepemilikan_Rumah
    le_rumah = LabelEncoder()
    df['Kepemilikan_Rumah_encoded'] = le_rumah.fit_transform(df['Kepemilikan_Rumah'])
    
    # Encoding untuk target
    le_target = LabelEncoder()
    df['Status_Kesejahteraan_encoded'] = le_target.fit_transform(df['Status_Kesejahteraan'])
    
    X = df[['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan', 'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah_encoded']]
    y = df['Status_Kesejahteraan_encoded']
    
    if len(X) == 0 or len(y) == 0:
        st.error("Data fitur atau target kosong setelah encoding.")
        return None, None, None, 0, {}
    
    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
        st.warning("Dataset kecil (<5 baris), menggunakan full data untuk training tanpa split evaluasi.")
    
    # Train Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    y_test_labels = le_target.inverse_transform(y_test)
    y_pred_labels = le_target.inverse_transform(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    
    return model, le_rumah, le_target, accuracy, report

# Fungsi untuk melakukan prediksi
def predict_single(model, le_rumah, le_target, data):
    try:
        # Validasi input numerik
        if np.isnan(data['usia']) or np.isnan(data['pendapatan']) or np.isnan(data['jumlah_anggota']):
            raise ValueError("Input numerik (usia, pendapatan, jumlah anggota) tidak valid.")
        
        # Validasi input encoder
        if data['kepemilikan_rumah'] not in le_rumah.classes_:
            raise ValueError(f"Kepemilikan Rumah harus salah satu dari: {list(le_rumah.classes_)}")
        
        rumah_encoded = le_rumah.transform([data['kepemilikan_rumah']])[0]
        input_data = np.array([[data['usia'], data['pendapatan'], data['jumlah_anggota'], rumah_encoded]])
        
        prediksi_encoded = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        
        prediksi = le_target.inverse_transform([prediksi_encoded])[0]
        
        # Sesuaikan probabilitas berdasarkan classes_
        class_to_idx = {cls: idx for idx, cls in enumerate(le_target.classes_)}
        prob_layak = prob[class_to_idx['Layak']]
        prob_tidak = prob[class_to_idx['Tidak Layak']]
        
        return prediksi, [prob_layak, prob_tidak]
    except KeyError:
        st.error("❌ Label target tidak dikenali. Pastikan dataset memiliki 'Layak' dan 'Tidak Layak'.")
        return None, None
    except ValueError as e:
        st.error(f"❌ Error prediksi: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error tak terduga: {e}")
        return None, None

# Sidebar untuk navigasi
st.sidebar.title("Navigasi Sistem")
page = st.sidebar.selectbox("Pilih Halaman:", ["Dashboard Informasi", "Upload Dataset & Prediksi", "Riwayat Prediksi"])

# Halaman 1: Dashboard Informasi
if page == "Dashboard Informasi":
    st.title("🏠 Dashboard Informasi")
    st.subheader("Klasifikasi Penerima Bantuan Sosial di Desa Cikembar")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.dataset is not None:
            st.metric("Total Dataset", f"{len(st.session_state.dataset)} Warga", "Data Terupload")
        else:
            st.metric("Total Dataset", "10 Warga", "Data Dummy")
    
    with col2:
        if st.session_state.model:
            st.metric("Status Model", "Tersedia", "Siap Prediksi")
        else:
            st.metric("Status Model", "Belum Dilatih", "Upload Dataset")
    
    with col3:
        st.metric("Riwayat Prediksi", f"{len(st.session_state.riwayat_prediksi)}", "Hasil Tersimpan")
    
    st.markdown("---")
    
    # Informasi Sistem
    st.header("
