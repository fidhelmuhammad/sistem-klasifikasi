import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
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

# Fungsi untuk membuat dataset dummy (hanya Ya/Tidak untuk Kepemilikan_Rumah)
@st.cache_data
def load_dummy_data():
    data = {
        'Nama': ['Ahmad', 'Siti', 'Budi', 'Dewi', 'Eko', 'Fani', 'Gatot', 'Hani', 'Indra', 'Joko'],
        'Jenis Kelamin': ['Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Laki-laki'],
        'Desa': ['Cikembar'] * 10,
        'Alamat': [f"Jl. Contoh {i}" for i in range(1, 11)],
        'RT': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'RW': [1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
        'Jumlah_Anggota_Keluarga': [4, 6, 3, 5, 2, 7, 4, 8, 1, 5],
        'Usia_Kepala_Keluarga': [45, 30, 55, 40, 60, 35, 50, 28, 65, 42],
        'Pendidikan_Kepala_Keluarga': ['SD', 'SMP', 'SMA', 'SD', 'SMP', 'SMA', 'SD', 'SMP', 'SMA', 'SD'],
        'Pekerjaan_Kepala_Keluarga': ['Petani', 'IRT', 'PNS', 'Buruh', 'Petani', 'Wiraswasta', 'Buruh', 'IRT', 'Pensiunan', 'Petani'],
        'Pendapatan_Bulanan': [1500000, 800000, 2000000, 1200000, 500000, 900000, 1800000, 600000, 400000, 1100000],
        'Pengeluaran_Bulanan': [1400000, 750000, 1900000, 1150000, 480000, 850000, 1700000, 580000, 380000, 1050000],
        'Kepemilikan_Rumah': ['Tidak', 'Tidak', 'Ya', 'Tidak', 'Tidak', 'Tidak', 'Ya', 'Tidak', 'Tidak', 'Tidak'],
        'Status_Kesejahteraan': ['Tidak Layak', 'Layak', 'Tidak Layak', 'Layak', 'Layak', 'Layak', 'Tidak Layak', 'Layak', 'Layak', 'Layak']
    }
    return pd.DataFrame(data)

# Fungsi untuk melatih model
def train_model(data):
    df = data.copy()

    # Encoding Kepemilikan_Rumah (Ya/Tidak)
    le_rumah = LabelEncoder()
    df['Kepemilikan_Rumah_encoded'] = le_rumah.fit_transform(df['Kepemilikan_Rumah'])

    # Encoding target
    le_target = LabelEncoder()
    df['Status_Kesejahteraan_encoded'] = le_target.fit_transform(df['Status_Kesejahteraan'])

    X = df[['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan', 'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah_encoded']]
    y = df['Status_Kesejahteraan_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_test_labels = le_target.inverse_transform(y_test)
    y_pred_labels = le_target.inverse_transform(y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)

    return model, le_rumah, le_target, accuracy, report

# Fungsi prediksi tunggal
def predict_single(model, le_rumah, le_target, data):
    rumah_encoded = le_rumah.transform([data['kepemilikan_rumah']])[0]
    input_data = np.array([[data['usia'], data['pendapatan'], data['jumlah_anggota'], rumah_encoded]])

    prediksi_encoded = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    prediksi = le_target.inverse_transform([prediksi_encoded])[0]
    return prediksi, prob

# Sidebar
st.sidebar.title("Navigasi Sistem")
page = st.sidebar.selectbox("Pilih Halaman:", ["Dashboard Informasi", "Upload Dataset & Prediksi", "Riwayat Prediksi"])

# Halaman 1: Dashboard
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
    st.header("📋 Deskripsi Sistem")
    st.markdown("""
    Sistem ini menggunakan algoritma Naive Bayes untuk mengklasifikasikan warga Desa Cikembar 
    apakah layak menerima bantuan sosial atau tidak.
    """)

# Halaman 2: Upload & Prediksi
elif page == "Upload Dataset & Prediksi":
    st.title("📁 Upload Dataset & Prediksi")

    tab1, tab2 = st.tabs(["Upload Dataset", "Prediksi Manual"])

    with tab1:
        uploaded_file = st.file_uploader("Pilih file dataset (CSV atau Excel)", type=['csv', 'xlsx', 'xls'])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                required_columns = ['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan', 'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah', 'Status_Kesejahteraan']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"❌ File harus memiliki kolom: {', '.join(required_columns)}")
                else:
                    df = df.dropna(subset=required_columns)
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
                            st.json(report)

            except Exception as e:
                st.error(f"❌ Error membaca file: {e}")
        else:
            st.info("📝 Silakan upload file CSV/Excel atau gunakan data dummy")
            dummy_df = load_dummy_data()
            st.dataframe(dummy_df.head())
            st.download_button(
                label="📥 Download Contoh Dataset CSV",
                data=dummy_df.to_csv(index=False).encode('utf-8'),
                file_name='contoh_dataset_cikembar.csv',
                mime='text/csv'
            )
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
        if st.session_state.model is None:
            st.warning("⚠️ Latih model terlebih dahulu")
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
                riwayat = {
                    'tanggal': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'data': data_input,
                    'prediksi': prediksi,
                    'probabilitas': prob
                }
                st.session_state.riwayat_prediksi.append(riwayat)

                st.header("🎯 Hasil Prediksi")
                if prediksi == 'Layak':
                    st.success("✅ **LAYAK MENERIMA BANTUAN SOSIAL**")
                else:
                    st.error("❌ **TIDAK LAYAK / BELUM BERHAK**")
                st.metric("Probabilitas Layak", f"{prob[0]:.2%}")
                st.metric("Probabilitas Tidak Layak", f"{prob[1]:.2%}")

# Halaman 3: Riwayat
elif page == "Riwayat Prediksi":
    st.title("📋 Riwayat Prediksi")

    if not st.session_state.riwayat_prediksi:
        st.info("📝 Belum ada riwayat prediksi.")
    else:
        for i, riwayat in enumerate(reversed(st.session_state.riwayat_prediksi), 1):
            with st.expander(f"Prediksi #{i} - {riwayat['tanggal']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Data Input:**")
                    st.write(f"Usia: {riwayat['data']['usia']} tahun")
                    st.write(f"Pendapatan: Rp {riwayat['data']['pendapatan']:,}")
                    st.write(f"Jumlah Anggota: {riwayat['data']['jumlah_anggota']} orang")
                    st.write(f"Kepemilikan Rumah: {riwayat['data']['kepemilikan_rumah']}")
                with col2:
                    st.write("**Hasil Prediksi:**")
                    st.write(f"Prediksi: {riwayat['prediksi']}")
                    st.write(f"Probabilitas Layak: {riwayat['probabilitas'][0]:.2%}")
                    st.write(f"Probabilitas Tidak Layak: {riwayat['probabilitas'][1]:.2%}")
