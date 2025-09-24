import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import io

st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

# Session state
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

# Dummy data sederhana (hanya contoh kolom)
@st.cache_data
def load_dummy_data():
    data = {
        'Usia_Kepala_Keluarga': [30, 45],
        'Pendapatan_Bulanan': [1000000, 1500000],
        'Jumlah_Anggota_Keluarga': [4, 5],
        'Kepemilikan_Rumah': ['Ya', 'Tidak'],
        'Status_Kesejahteraan': ['Layak', 'Tidak Layak']
    }
    return pd.DataFrame(data)

# Training model
def train_model(data):
    df = data.copy()
    if 'Kepemilikan_Rumah' not in df.columns or 'Status_Kesejahteraan' not in df.columns:
        st.error("Dataset harus memiliki kolom 'Kepemilikan_Rumah' dan 'Status_Kesejahteraan'.")
        return None, None, None, 0, {}

    le_rumah = LabelEncoder()
    df['Kepemilikan_Rumah_encoded'] = le_rumah.fit_transform(df['Kepemilikan_Rumah'])

    le_target = LabelEncoder()
    df['Status_Kesejahteraan_encoded'] = le_target.fit_transform(df['Status_Kesejahteraan'])

    X = df[['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan', 'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah_encoded']]
    y = df['Status_Kesejahteraan_encoded']

    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
        st.warning("Dataset kecil (<5 baris), menggunakan semua data untuk training.")

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_test_labels = le_target.inverse_transform(y_test)
    y_pred_labels = le_target.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)

    return model, le_rumah, le_target, accuracy, report

# Prediksi single input
def predict_single(model, le_rumah, le_target, data):
    try:
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

# Dashboard
if page == "Dashboard Informasi":
    st.title("🏠 Dashboard Informasi")
    st.subheader("Klasifikasi Penerima Bantuan Sosial")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.dataset is not None:
            st.metric("Total Dataset", f"{len(st.session_state.dataset)} Warga", "Data Terupload")
        else:
            st.metric("Total Dataset", "2 Warga", "Data Contoh")
    with col2:
        if st.session_state.model:
            st.metric("Status Model", "Tersedia", "Siap Prediksi")
        else:
            st.metric("Status Model", "Belum Dilatih", "Upload Dataset")
    with col3:
        st.metric("Riwayat Prediksi", f"{len(st.session_state.riwayat_prediksi)}", "Hasil Tersimpan")

    st.markdown("---")
    st.header("📋 Deskripsi Sistem")
    st.markdown("Sistem ini menggunakan algoritma Naive Bayes untuk menentukan apakah warga layak menerima bantuan sosial berdasarkan data ekonomi dan demografi.")

    st.header("📊 Fitur yang Dianalisis")
    features = ["Usia Kepala Keluarga", "Pendapatan Bulanan", "Jumlah Anggota Keluarga", "Kepemilikan Rumah"]
    for i, f in enumerate(features, 1):
        st.write(f"{i}. {f}")
    st.write("**Target:** Status_Kesejahteraan (Layak / Tidak Layak)")

# Upload dataset & prediksi
elif page == "Upload Dataset & Prediksi":
    st.title("📁 Upload Dataset & Prediksi")
    tab1, tab2 = st.tabs(["Upload Dataset", "Prediksi Manual"])

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
                ext = uploaded_file.name.split('.')[-1].lower()
                if ext == 'csv':
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                required = ['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan', 'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah', 'Status_Kesejahteraan']
                if not all(c in df.columns for c in required):
                    st.error(f"Dataset harus punya kolom: {', '.join(required)}")
                else:
                    st.session_state.dataset = df
                    st.success("✅ Dataset berhasil diupload")
                    st.dataframe(df.head())

                    if st.button("🚀 Latih Model dengan Dataset Ini"):
                        with st.spinner("Melatih model..."):
                            model, le_rumah, le_target, acc, report = train_model(df)
                            if model:
                                st.session_state.model = model
                                st.session_state.le_rumah = le_rumah
                                st.session_state.le_target = le_target
                                st.success(f"✅ Model berhasil dilatih! Akurasi: {acc:.2f}")

                                st.subheader("📊 Classification Report")
                                st.dataframe(pd.DataFrame(report).transpose())
                            else:
                                st.error("❌ Gagal melatih model.")

            except Exception as e:
                st.error(f"Error upload: {e}")

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
                data_input = {"usia": usia, "pendapatan": pendapatan, "jumlah_anggota": jumlah, "kepemilikan_rumah": kepemilikan}
                pred, prob = predict_single(st.session_state.model, st.session_state.le_rumah, st.session_state.le_target, data_input)
                if pred:
                    st.success(f"Hasil Prediksi: **{pred}**")
                    st.write("Probabilitas:", prob)
                    st.session_state.riwayat_prediksi.append({"usia": usia, "pendapatan": pendapatan, "jumlah": jumlah, "rumah": kepemilikan, "hasil": pred})

# Riwayat prediksi
elif page == "Riwayat Prediksi":
    st.title("📜 Riwayat Prediksi")
    if len(st.session_state.riwayat_prediksi) == 0:
        st.info("Belum ada riwayat prediksi.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.riwayat_prediksi))
