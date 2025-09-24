import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import io

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


@st.cache_data
def load_dummy_data():
    """Contoh dataset sederhana hanya untuk struktur kolom"""
    data = {
        'Usia_Kepala_Keluarga': [30, 45],
        'Pendapatan_Bulanan': [1000000, 1500000],
        'Jumlah_Anggota_Keluarga': [4, 5],
        'Kepemilikan_Rumah': ['Ya', 'Tidak'],
        'Status_Kesejahteraan': ['Layak', 'Tidak Layak']
    }
    return pd.DataFrame(data)


def train_model(data):
    df = data.copy()

    # Validasi kolom
    if 'Kepemilikan_Rumah' not in df.columns or len(df) == 0:
        st.error("Data tidak valid atau kolom 'Kepemilikan_Rumah' hilang.")
        return None, None, None, 0, {}
    unique_rumah = set(df['Kepemilikan_Rumah'].dropna().unique())
    if unique_rumah - {'Ya', 'Tidak'}:
        st.error("Kolom 'Kepemilikan_Rumah' harus hanya berisi 'Ya' atau 'Tidak'.")
        return None, None, None, 0, {}

    if 'Status_Kesejahteraan' not in df.columns or len(df) == 0:
        st.error("Data tidak valid atau kolom 'Status_Kesejahteraan' hilang.")
        return None, None, None, 0, {}
    unique_target = set(df['Status_Kesejahteraan'].dropna().unique())
    if unique_target - {'Layak', 'Tidak Layak'}:
        st.error("Kolom 'Status_Kesejahteraan' harus hanya berisi 'Layak' atau 'Tidak Layak'.")
        return None, None, None, 0, {}

    # Encoding
    le_rumah = LabelEncoder()
    df['Kepemilikan_Rumah_encoded'] = le_rumah.fit_transform(df['Kepemilikan_Rumah'])

    le_target = LabelEncoder()
    df['Status_Kesejahteraan_encoded'] = le_target.fit_transform(df['Status_Kesejahteraan'])

    # Fitur
    X = df[['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan', 'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah_encoded']]
    y = df['Status_Kesejahteraan_encoded']

    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
        st.warning("Dataset kecil (<5 baris), menggunakan full data untuk training tanpa split evaluasi.")

    # Training
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test_labels = le_target.inverse_transform(y_test)
    y_pred_labels = le_target.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)

    return model, le_rumah, le_target, accuracy, report


def predict_single(model, le_rumah, le_target, data):
    try:
        if model is None or le_rumah is None or le_target is None:
            return None, None

        rumah_encoded = le_rumah.transform([data['kepemilikan_rumah']])[0]
        input_data = np.array([[data['usia'], data['pendapatan'], data['jumlah_anggota'], rumah_encoded]])

        prediksi_encoded = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        prediksi = le_target.inverse_transform([prediksi_encoded])[0]

        return prediksi, prob
    except Exception as e:
        st.error(f"Error prediksi: {e}")
        return None, None


# Sidebar
st.sidebar.title("Navigasi Sistem")
page = st.sidebar.selectbox("Pilih Halaman:", ["Dashboard Informasi", "Upload Dataset & Prediksi", "Riwayat Prediksi"])

# ==================== DASHBOARD ====================
if page == "Dashboard Informasi":
    st.title("🏠 Dashboard Informasi")
    st.subheader("Klasifikasi Penerima Bantuan Sosial di Desa Cikembar")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.dataset is not None:
            st.metric("Total Dataset", f"{len(st.session_state.dataset)} Warga", "Data Terupload")
        else:
            st.metric("Total Dataset", "2 Warga", "Data Dummy")
    with col2:
        if st.session_state.model:
            st.metric("Status Model", "Tersedia", "Siap Prediksi")
        else:
            st.metric("Status Model", "Belum Dilatih", "Upload Dataset")
    with col3:
        st.metric("Riwayat Prediksi", f"{len(st.session_state.riwayat_prediksi)}", "Hasil Tersimpan")

    st.markdown("---")
    st.header("📋 Deskripsi Sistem")
    st.markdown("Sistem ini dirancang untuk mengklasifikasikan warga Desa Cikembar yang berhak menerima bantuan sosial berdasarkan data demografis dan ekonomi menggunakan algoritma Naive Bayes.")

    st.header("📊 Fitur yang Dianalisis")
    features = ["Usia Kepala Keluarga", "Pendapatan Bulanan", "Jumlah Anggota Keluarga", "Kepemilikan Rumah"]
    for i, feature in enumerate(features, 1):
        st.write(f"{i}. {feature}")
    st.write("**Target:** Status_Kesejahteraan (Layak / Tidak Layak)")

    st.header("📈 Status Terkini")
    if st.session_state.model:
        st.success("✅ Model sudah dilatih dan siap digunakan untuk prediksi")
    else:
        st.warning("⚠️ Silakan upload dataset dan latih model terlebih dahulu")

# ==================== UPLOAD & PREDIKSI ====================
elif page == "Upload Dataset & Prediksi":
    st.title("📁 Upload Dataset & Prediksi")
    tab1, tab2 = st.tabs(["Upload Dataset", "Prediksi Manual"])

    # --- Upload Dataset ---
    with tab1:
        st.header("Upload Dataset")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Pilih file dataset (CSV atau Excel)", type=['csv', 'xlsx', 'xls'])
        with col2:
            dummy_df = load_dummy_data()
            csv = dummy_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Contoh Dataset CSV", csv, "contoh_dataset.csv", "text/csv")
            excel_buffer = io.BytesIO()
            dummy_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button("📥 Download Contoh Dataset Excel", excel_buffer.getvalue(), "contoh_dataset.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # Pastikan numeric
                numeric_cols = ['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan', 'Jumlah_Anggota_Keluarga']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                required_columns = ['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan',
                                    'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah',
                                    'Status_Kesejahteraan']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"File harus memiliki kolom: {', '.join(required_columns)}. Kolom saat ini: {', '.join(df.columns)}")
                    st.info("Pastikan nama kolom persis seperti contoh dataset.")
                else:
                    initial_len = len(df)
                    df = df.dropna(subset=required_columns)
                    if len(df) == 0:
                        st.error("Dataset kosong setelah pembersihan. Periksa data.")
                    elif len(df) < initial_len:
                        st.warning(f"{initial_len - len(df)} baris dihapus karena data kosong/invalid.")

                    if len(df) > 0:
                        st.session_state.dataset = df
                        st.success("✅ Dataset berhasil diupload!")
                        st.dataframe(df.head())

                        if st.button("🚀 Latih Model dengan Dataset Ini"):
                            with st.spinner("Melatih model Naive Bayes..."):
                                model, le_rumah, le_target, accuracy, report = train_model(df)
                                if model:
                                    st.session_state.model = model
                                    st.session_state.le_rumah = le_rumah
                                    st.session_state.le_target = le_target
                                    st.success(f"✅ Model berhasil dilatih! Akurasi: {accuracy:.2f}")

                                    st.subheader("📊 Classification Report")
                                    report_df = pd.DataFrame(report).transpose()
                                    st.dataframe(report_df)
                                else:
                                    st.error("❌ Gagal melatih model. Periksa dataset Anda.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {e}")

    # --- Prediksi Manual ---
    with tab2:
        st.header("Prediksi Manual")
        if st.session_state.model is None:
            st.warning("⚠️ Latih model terlebih dahulu sebelum prediksi.")
        else:
            usia = st.number_input("Usia Kepala Keluarga", min_value=18, max_value=100, value=40)
            pendapatan = st.number_input("Pendapatan Bulanan", min_value=0, value=1000000, step=100000)
            jumlah = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=20, value=4)
            kepemilikan = st.selectbox("Kepemilikan Rumah", st.session_state.le_rumah.classes_)

            if st.button("🔮 Prediksi Status"):
                data_input = {
                    "usia": usia,
                    "pendapatan": pendapatan,
                    "jumlah_anggota": jumlah,
                    "kepemilikan_rumah": kepemilikan
                }
                pred, prob = predict_single(
                    st.session_state.model,
                    st.session_state.le_rumah,
                    st.session_state.le_target,
                    data_input
                )

                if pred is not None:
                    st.success(f"Hasil Prediksi: **{pred}**")
                    st.write("Probabilitas per kelas:")
                    st.write(dict(zip(st.session_state.le_target.classes_, prob)))

                    st.session_state.riwayat_prediksi.append({
                        "usia": usia,
                        "pendapatan": pendapatan,
                        "jumlah": jumlah,
                        "rumah": kepemilikan,
                        "hasil": pred
                    })
                else:
                    st.error("Prediksi gagal. Pastikan model sudah benar.")

# ==================== RIWAYAT ====================
elif page == "Riwayat Prediksi":
    st.title("🕑 Riwayat Prediksi")
    if len(st.session_state.riwayat_prediksi) == 0:
        st.info("Belum ada riwayat prediksi.")
    else:
        df_history = pd.DataFrame(st.session_state.riwayat_prediksi)
        st.dataframe(df_history)
