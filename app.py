import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# ==============================
# Konfigurasi halaman
# ==============================
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


# ==============================
# Fungsi dataset dummy
# ==============================
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


# ==============================
# Fungsi training model
# ==============================
def train_model(data):
    df = data.copy()

    if df is None or len(df) == 0:
        st.error("❌ Dataset kosong, tidak bisa melatih model.")
        return None, None, None, 0, {}

    if 'Kepemilikan_Rumah' not in df.columns or 'Status_Kesejahteraan' not in df.columns:
        st.error("❌ Dataset tidak sesuai format.")
        return None, None, None, 0, {}

    # Validasi nilai kategori
    unique_rumah = set(df['Kepemilikan_Rumah'].dropna().unique())
    if unique_rumah - {'Ya', 'Tidak'}:
        st.error("❌ Kolom 'Kepemilikan_Rumah' hanya boleh berisi 'Ya' atau 'Tidak'.")
        return None, None, None, 0, {}

    unique_target = set(df['Status_Kesejahteraan'].dropna().unique())
    if unique_target - {'Layak', 'Tidak Layak'}:
        st.error("❌ Kolom 'Status_Kesejahteraan' hanya boleh berisi 'Layak' atau 'Tidak Layak'.")
        return None, None, None, 0, {}

    # Encoding
    le_rumah = LabelEncoder()
    df['Kepemilikan_Rumah_encoded'] = le_rumah.fit_transform(df['Kepemilikan_Rumah'])

    le_target = LabelEncoder()
    df['Status_Kesejahteraan_encoded'] = le_target.fit_transform(df['Status_Kesejahteraan'])

    # Fitur & target
    X = df[['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan', 'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah_encoded']]
    y = df['Status_Kesejahteraan_encoded']

    # Train-test split
    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
        st.warning("⚠️ Dataset kecil (<5 baris), training tanpa evaluasi split.")

    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Evaluasi
    if len(y_test) > 0:
        y_pred = model.predict(X_test)
        try:
            y_test_labels = le_target.inverse_transform(y_test)
            y_pred_labels = le_target.inverse_transform(y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        except Exception:
            accuracy = 0
            report = {}
    else:
        accuracy = 0
        report = {}

    return model, le_rumah, le_target, accuracy, report


# ==============================
# Fungsi prediksi tunggal
# ==============================
def predict_single(model, le_rumah, le_target, data):
    try:
        # Validasi numerik
        usia = float(data['usia'])
        pendapatan = float(data['pendapatan'])
        jumlah_anggota = float(data['jumlah_anggota'])

        if pd.isna(usia) or pd.isna(pendapatan) or pd.isna(jumlah_anggota):
            raise ValueError("Input numerik tidak boleh kosong.")

        # Validasi kategori
        if data['kepemilikan_rumah'] not in le_rumah.classes_:
            raise ValueError(f"Kepemilikan Rumah harus salah satu dari: {list(le_rumah.classes_)}")

        rumah_encoded = le_rumah.transform([data['kepemilikan_rumah']])[0]
        input_data = np.array([[usia, pendapatan, jumlah_anggota, rumah_encoded]])

        prediksi_encoded = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        prediksi = le_target.inverse_transform([prediksi_encoded])[0]

        # Probabilitas aman
        class_to_idx = {cls: idx for idx, cls in enumerate(le_target.classes_)}
        prob_layak = prob[class_to_idx['Layak']] if 'Layak' in class_to_idx else 0
        prob_tidak = prob[class_to_idx['Tidak Layak']] if 'Tidak Layak' in class_to_idx else 0

        return prediksi, [prob_layak, prob_tidak]

    except ValueError as e:
        st.error(f"❌ Error prediksi: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error tak terduga: {e}")
        return None, None


# ==============================
# Sidebar Navigasi
# ==============================
st.sidebar.title("Navigasi Sistem")
page = st.sidebar.selectbox("Pilih Halaman:", ["Dashboard Informasi", "Upload Dataset & Prediksi", "Riwayat Prediksi"])

# ==============================
# Halaman Dashboard
# ==============================
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
    st.write("👈 Pilih menu di sidebar untuk melanjutkan.")


# ==============================
# Halaman Upload & Prediksi
# ==============================
elif page == "Upload Dataset & Prediksi":
    st.title("📂 Upload Dataset & Prediksi")

    # Upload file
    uploaded_file = st.file_uploader("Upload file Excel/CSV:", type=["xlsx", "csv"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith("csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.dataset = df
            st.success("✅ Dataset berhasil diupload.")
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            st.session_state.dataset = None
    else:
        st.info("Menggunakan dataset dummy.")
        st.session_state.dataset = load_dummy_data()

    # Training
    if st.button("Latih Model"):
        model, le_rumah, le_target, acc, report = train_model(st.session_state.dataset)
        if model:
            st.session_state.model = model
            st.session_state.le_rumah = le_rumah
            st.session_state.le_target = le_target
            st.success(f"✅ Model berhasil dilatih. Akurasi: {acc:.2f}")
            st.json(report)

    st.markdown("---")
    st.subheader("🔮 Prediksi Manual")

    if st.session_state.model:
        with st.form("form_prediksi"):
            usia = st.number_input("Usia Kepala Keluarga", min_value=18, max_value=100, value=40)
            pendapatan = st.number_input("Pendapatan Bulanan", min_value=0, step=100000, value=1000000)
            jumlah = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=20, value=4)
            kepemilikan = st.selectbox("Kepemilikan Rumah", ["Ya", "Tidak"])

            submitted = st.form_submit_button("Prediksi")
            if submitted:
                data_input = {
                    "usia": usia,
                    "pendapatan": pendapatan,
                    "jumlah_anggota": jumlah,
                    "kepemilikan_rumah": kepemilikan
                }
                hasil, prob = predict_single(st.session_state.model, st.session_state.le_rumah, st.session_state.le_target, data_input)

                if hasil:
                    st.success(f"Hasil Prediksi: **{hasil}**")
                    st.write(f"Probabilitas Layak: {prob[0]:.2f}, Tidak Layak: {prob[1]:.2f}")

                    # Simpan ke riwayat
                    st.session_state.riwayat_prediksi.append({
                        "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Input": data_input,
                        "Hasil": hasil,
                        "Probabilitas": prob
                    })


# ==============================
# Halaman Riwayat
# ==============================
elif page == "Riwayat Prediksi":
    st.title("📝 Riwayat Prediksi")
    if st.session_state.riwayat_prediksi:
        st.dataframe(pd.DataFrame(st.session_state.riwayat_prediksi))
    else:
        st.info("Belum ada riwayat prediksi.")
