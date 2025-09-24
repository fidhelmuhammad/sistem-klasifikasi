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
    
    # Pilih fitur utama (numerik dan kategorikal yang relevan)
    # Fitur: Usia, Pendapatan, Jumlah Anggota, Kepemilikan Rumah
    # Target: Status_Kesejahteraan (encode ke 0/1: Layak=0, Tidak Layak=1 berdasarkan LabelEncoder)
    
    # Encoding untuk Kepemilikan_Rumah
    le_rumah = LabelEncoder()
    df['Kepemilikan_Rumah_encoded'] = le_rumah.fit_transform(df['Kepemilikan_Rumah'])
    
    # Encoding untuk target (Layak=0, Tidak Layak=1 secara alfabetis)
    le_target = LabelEncoder()
    df['Status_Kesejahteraan_encoded'] = le_target.fit_transform(df['Status_Kesejahteraan'])
    
    X = df[['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan', 'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah_encoded']]
    y = df['Status_Kesejahteraan_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Evaluasi (decode target untuk report)
    y_pred = model.predict(X_test)
    y_test_labels = le_target.inverse_transform(y_test)
    y_pred_labels = le_target.inverse_transform(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    
    return model, le_rumah, le_target, accuracy, report

# Fungsi untuk melakukan prediksi
def predict_single(model, le_rumah, le_target, data):
    rumah_encoded = le_rumah.transform([data['kepemilikan_rumah']])[0]
    input_data = np.array([[data['usia'], data['pendapatan'], data['jumlah_anggota'], rumah_encoded]])
    
    prediksi_encoded = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]  # [prob_Layak (0), prob_Tidak Layak (1)]
    
    # Decode prediksi
    prediksi = le_target.inverse_transform([prediksi_encoded])[0]
    
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
    st.header("📋 Deskripsi Sistem")
    st.markdown("""
    Sistem ini dirancang untuk mengklasifikasikan warga Desa Cikembar yang berhak menerima bantuan sosial 
    berdasarkan data demografis dan ekonomi menggunakan algoritma Naive Bayes.
    """)
    
    # Fitur yang digunakan (disesuaikan)
    st.header("📊 Fitur yang Dianalisis")
    features = [
        "Usia Kepala Keluarga",
        "Pendapatan Bulanan",
        "Jumlah Anggota Keluarga", 
        "Kepemilikan Rumah"
    ]
    
    for i, feature in enumerate(features, 1):
        st.write(f"{i}. {feature}")
    
    st.write("**Target:** Status_Kesejahteraan (Layak / Tidak Layak)")
    
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
        
        # Dukung CSV dan Excel (.xlsx, .xls)
        uploaded_file = st.file_uploader("Pilih file dataset (CSV atau Excel)", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                # Baca file berdasarkan ekstensi
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                else:  # Excel
                    df = pd.read_excel(uploaded_file)
                
                # Validasi kolom dasar (sesuai struktur dataset baru)
                required_columns = ['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan', 'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah', 'Status_Kesejahteraan']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"❌ File harus memiliki kolom: {', '.join(required_columns)}. Kolom saat ini: {', '.join(df.columns)}")
                    st.info("Pastikan nama kolom persis seperti: Usia_Kepala_Keluarga, Pendapatan_Bulanan, dll.")
                else:
                    # Bersihkan data jika ada NaN atau invalid
                    df = df.dropna(subset=required_columns)
                    if len(df) == 0:
                        st.error("❌ Dataset kosong setelah pembersihan. Periksa data.")
                    else:
                        st.session_state.dataset = df
                        st.success("✅ Dataset berhasil diupload!")
                        st.dataframe(df.head())
                        
                        if st.button("🚀 Latih Model dengan Dataset Ini"):
                            with st.spinner("Melatih model Naive Bayes..."):
                                model, le_rumah, le_target, accuracy, report = train_model(df)
                                st.session_state.model = model
                                st.session_state.le_rumah = le_rumah
                                st.session_state.le_target = le_target
                                
                                st.success(f"✅ Model berhasil dilatih dengan akurasi: {accuracy:.2%}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Akurasi Model", f"{accuracy:.2%}")
                                with col2:
                                    st.metric("Jumlah Data", len(df))
                                
                                # Tampilkan laporan singkat
                                st.subheader("Laporan Klasifikasi")
                                st.json(report)
                                
            except Exception as e:
                st.error(f"❌ Error membaca file: {e}. Pastikan file tidak rusak dan format kolom benar.")
        else:
            st.info("📝 Silakan upload file CSV atau Excel dengan format yang sesuai")
            st.markdown("""
            **Format file yang diharapkan (kolom wajib):**
            - Usia_Kepala_Keluarga (number)
            - Pendapatan_Bulanan (number) 
            - Jumlah_Anggota_Keluarga (number)
            - Kepemilikan_Rumah (Ya/Tidak)
            - Status_Kesejahteraan (Layak/Tidak Layak)
            
            **Kolom opsional lain:** Nama, Jenis Kelamin, Desa, Alamat, RT, RW, Pendidikan_Kepala_Keluarga, dll.
            """)
            dummy_df = load_dummy_data()
            st.dataframe(dummy_df.head())
            st.download_button(
                label="📥 Download Contoh Dataset CSV",
                data=dummy_df.to_csv(index=False).encode('utf-8'),
                file_name='contoh_dataset_cikembar.csv',
                mime='text/csv'
            )
            
            # Download Excel menggunakan BytesIO
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                dummy_df.to_excel(writer, index=False, sheet_name='Data Warga')
            buffer.seek(0)
            st.download_button(
                label="📥 Download Contoh Dataset Excel",
                data=buffer.getvalue(),
                file_name='contoh_dataset_cikembar.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    
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
                kepemilikan_rumah = st.selectbox("Kepemilikan Rumah?", ["Ya", "Tidak"])
            
            if st.button("🔮 Prediksi Kelayakan"):
                data_input = {
                    'usia': usia,
                    'pendapatan': pendapatan,
                    'jumlah_anggota': jumlah_anggota,
                    'kepemilikan_rumah': kepemilikan_rumah
                }
                
                prediksi, prob = predict_single(st.session_state.model, st.session_state.le_rumah, st.session_state.le_target, data_input)
                
                # Simpan ke riwayat (simpan prediksi sebagai string)
                riwayat = {
                    'tanggal': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'data': data_input,
                    'prediksi': prediksi,
                    'probabilitas': prob  # [prob_Layak, prob_Tidak Layak]
                }
                st.session_state.riwayat_prediksi.append(riwayat)
                
                # Tampilkan hasil
                st.markdown("---")
                st.header("🎯 Hasil Prediksi")
                
                if prediksi == 'Layak':
                    st.success("✅ **LAYAK MENERIMA BANTUAN SOSIAL**")
                    st.write("Warga ini layak mendapatkan bantuan berdasarkan kriteria yang dianalisis.")
                else:
                    st.error("❌ **TIDAK LAYAK / BELUM BERHAK**")
                    st.write("Warga ini tidak memenuhi kriteria untuk menerima bantuan saat ini.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilitas Layak", f"{prob[0]:.2%}")
                with col2:
                    st.metric("Probabilitas Tidak Layak", f"{prob[1]:.2%}")
                
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
                    st.write(f"Kepemilikan Rumah: {ri
