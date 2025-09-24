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

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

# Inisialisasi session state
if 'riwayat_prediksi' not in st.session_state:
    st.session_state.riwayat_prediksi = []

if 'model' not in st.session_state:
    st.session_state.model = None

if 'le_rumah' not in st.session_state:
    st.session_state.le_rumah = None

if 'dataset' not in st.session_state:
    st.session_state.dataset = None

# Fungsi untuk melatih model
def train_model(data):
    df = data.copy()
    
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
    
    return model, le_rumah, accuracy, classification_report(y_test, y_pred, output_dict=True)

# Fungsi untuk melakukan prediksi
def predict_single(model, le_rumah, data):
    rumah_encoded = le_rumah.transform([data['memiliki_rumah']])[0]
    input_data = np.array([[data['usia'], data['pendapatan'], data['jumlah_anggota'], rumah_encoded]])
    
    prediksi = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    
    return prediksi, prob

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
        st.metric("Total Dataset", "15 Warga", "Data Simulasi")
    
    with col2:
        if st.session_state.model:
            st.metric("Status Model", "Tersedia", "Siap Prediksi")
        else:
            st.metric("Status Model", "Belum Dilatih", "Upload Dataset")
    
    with col3:
        st.metric("Riwayat Prediksi", f"{len(st.session_state.riwayat_prediksi)}", "Hasil Tersimpan")
    
    st.markdown("---")
    
    # Informasi Sistem
    st.header("📋 Deskripsi Sistem")
    st.markdown("""
    Sistem ini dirancang untuk mengklasifikasikan warga Desa Cikembar yang berhak menerima bantuan sosial 
    berdasarkan data demografis dan ekonomi menggunakan algoritma Naive Bayes.
    """)
    
    # Fitur yang digunakan
    st.header("📊 Fitur yang Dianalisis")
    features = [
        "Usia Kepala Keluarga",
        "Pendapatan Bulanan",
        "Jumlah Anggota Keluarga", 
        "Status Kepemilikan Rumah"
    ]
    
    for i, feature in enumerate(features, 1):
        st.write(f"{i}. {feature}")
    
    # Status terkini
    st.header("📈 Status Terkini")
    if st.session_state.model:
        st.success("✅ Model sudah dilatih dan siap digunakan untuk prediksi")
    else:
        st.warning("⚠️ Silakan upload dataset dan latih model terlebih dahulu")

# Halaman 2: Upload Dataset dan Prediksi
elif page == "Upload Dataset & Prediksi":
    st.title("📁 Upload Dataset & Prediksi")
    
    tab1, tab2 = st.tabs(["Upload Dataset", "Prediksi Manual"])
    
    with tab1:
        st.header("Upload Dataset")
        
        uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.dataset = df
                
                st.success("✅ Dataset berhasil diupload!")
                st.dataframe(df.head())
                
                if st.button("🚀 Latih Model dengan Dataset Ini"):
                    with st.spinner("Melatih model Naive Bayes..."):
                        model, le_rumah, accuracy, report = train_model(df)
                        st.session_state.model = model
                        st.session_state.le_rumah = le_rumah
                        
                        st.success(f"✅ Model berhasil dilatih dengan akurasi: {accuracy:.2%}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Akurasi Model", f"{accuracy:.2%}")
                        with col2:
                            st.metric("Jumlah Data", len(df))
                        
            except Exception as e:
                st.error(f"❌ Error membaca file: {e}")
        else:
            st.info("📝 Silakan upload file CSV dengan format yang sesuai")
            st.markdown("""
            **Format CSV yang diharapkan:**
            - usia_kepala_keluarga (number)
            - pendapatan_bulanan (number) 
            - jumlah_anggota_keluarga (number)
            - memiliki_rumah (Ya/Tidak)
            - berhak_bantuan (0/1)
            """)
    
    with tab2:
        st.header("Prediksi Manual")
        
        if st.session_state.model is None:
            st.warning("⚠️ Silakan upload dataset dan latih model terlebih dahulu")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                usia = st.number_input("Usia Kepala Keluarga", min_value=18, max_value=100, value=40)
                pendapatan = st.number_input("Pendapatan Bulanan (Rp)", min_value=0, max_value=10000000, value=1500000)
            
            with col2:
                jumlah_anggota = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=20, value=4)
                memiliki_rumah = st.selectbox("Memiliki Rumah Sendiri?", ["Ya", "Tidak"])
            
            if st.button("🔮 Prediksi Kelayakan"):
                data_input = {
                    'usia': usia,
                    'pendapatan': pendapatan,
                    'jumlah_anggota': jumlah_anggota,
                    'memiliki_rumah': memiliki_rumah
                }
                
                prediksi, prob = predict_single(st.session_state.model, st.session_state.le_rumah, data_input)
                
                # Simpan ke riwayat
                riwayat = {
                    'tanggal': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'data': data_input,
                    'prediksi': prediksi,
                    'probabilitas': prob
                }
                st.session_state.riwayat_prediksi.append(riwayat)
                
                # Tampilkan hasil
                st.markdown("---")
                st.header("🎯 Hasil Prediksi")
                
                if prediksi == 1:
                    st.success("✅ **BERHAK MENERIMA BANTUAN SOSIAL**")
                    st.write("Warga ini layak mendapatkan bantuan berdasarkan kriteria yang dianalisis.")
                else:
                    st.error("❌ **TIDAK BERHAK / BELUM LAYAK**")
                    st.write("Warga ini tidak memenuhi kriteria untuk menerima bantuan saat ini.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilitas Berhak", f"{prob[1]:.2%}")
                with col2:
                    st.metric("Probabilitas Tidak Berhak", f"{prob[0]:.2%}")
                
                st.info("💡 Hasil prediksi telah disimpan di riwayat")

# Halaman 3: Riwayat Prediksi
elif page == "Riwayat Prediksi":
    st.title("📋 Riwayat Prediksi")
    
    if not st.session_state.riwayat_prediksi:
        st.info("📝 Belum ada riwayat prediksi. Silakan lakukan prediksi terlebih dahulu.")
    else:
        st.write(f"Total {len(st.session_state.riwayat_prediksi)} prediksi tersimpan")
        
        for i, riwayat in enumerate(reversed(st.session_state.riwayat_prediksi), 1):
            with st.expander(f"Prediksi #{i} - {riwayat['tanggal']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Data Input:**")
                    st.write(f"Usia: {riwayat['data']['usia']} tahun")
                    st.write(f"Pendapatan: Rp {riwayat['data']['pendapatan']:,}")
                    st.write(f"Jumlah Anggota: {riwayat['data']['jumlah_anggota']} orang")
                    st.write(f"Memiliki Rumah: {riwayat['data']['memiliki_rumah']}")
                
                with col2:
                    st.write("**Hasil Prediksi:**")
                    if riwayat['prediksi'] == 1:
                        st.success("BERHAK MENERIMA BANTUAN")
                    else:
                        st.error("TIDAK BERHAK")
                    
                    st.write(f"Probabilitas Berhak: {riwayat['probabilitas'][1]:.2%}")
                    st.write(f"Probabilitas Tidak: {riwayat['probabilitas'][0]:.2%}")
        
        # Tombol clear riwayat
        if st.button("🗑️ Hapus Semua Riwayat"):
            st.session_state.riwayat_prediksi = []
            st.success("Riwayat prediksi telah dihapus")
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Sistem Klasifikasi Bantuan Sosial - Desa Cikembar")
