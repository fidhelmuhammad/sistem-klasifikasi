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

# -------------------------------
# Fungsi training
# -------------------------------
def train_model(data):
    df = data.copy()

    # Normalisasi nilai kepemilikan rumah
    if 'Kepemilikan_Rumah' in df.columns:
        df['Kepemilikan_Rumah'] = df['Kepemilikan_Rumah'].astype(str).str.strip().str.lower()
        df['Kepemilikan_Rumah'] = df['Kepemilikan_Rumah'].replace({
            'ya': 'Ya', 'y': 'Ya', 'milik sendiri': 'Ya', 'punya sendiri': 'Ya',
            'tidak': 'Tidak', 't': 'Tidak', 'n': 'Tidak', 'kontrak': 'Tidak', 'sewa': 'Tidak'
        })

    # Validasi kolom
    required = ['Usia_Kepala_Keluarga','Pendapatan_Bulanan','Jumlah_Anggota_Keluarga',
                'Kepemilikan_Rumah','Status_Kesejahteraan']
    if not all(col in df.columns for col in required):
        st.error(f"Dataset harus punya kolom: {', '.join(required)}")
        return None, None, None, 0, {}

    # Label encoding
    le_rumah = LabelEncoder()
    df['Kepemilikan_Rumah_encoded'] = le_rumah.fit_transform(df['Kepemilikan_Rumah'])

    le_target = LabelEncoder()
    df['Status_Kesejahteraan_encoded'] = le_target.fit_transform(df['Status_Kesejahteraan'])

    X = df[['Usia_Kepala_Keluarga','Pendapatan_Bulanan','Jumlah_Anggota_Keluarga','Kepemilikan_Rumah_encoded']]
    y = df['Status_Kesejahteraan_encoded']

    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
        st.warning("Dataset terlalu kecil, training tanpa split test.")

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        le_target.inverse_transform(y_test),
        le_target.inverse_transform(y_pred),
        output_dict=True
    )

    return model, le_rumah, le_target, accuracy, report

# -------------------------------
# Fungsi prediksi single input
# -------------------------------
def predict_single(model, le_rumah, le_target, data):
    try:
        rumah_encoded = le_rumah.transform([data['kepemilikan_rumah']])[0]
        input_data = np.array([[data['usia'], data['pendapatan'], data['jumlah_anggota'], rumah_encoded]])
        pred_encoded = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        pred = le_target.inverse_transform([pred_encoded])[0]
        prob_dict = dict(zip(le_target.classes_, prob))
        return pred, prob_dict
    except Exception as e:
        st.error(f"Error prediksi: {e}")
        return None, None

# -------------------------------
# Navigasi
# -------------------------------
st.sidebar.title("Navigasi Sistem")
page = st.sidebar.selectbox("Pilih Halaman:", ["Dashboard Informasi", "Upload Dataset & Latih Model", "Prediksi Manual", "Riwayat Prediksi"])

# -------------------------------
# Dashboard
# -------------------------------
if page == "Dashboard Informasi":
    st.title("🏠 Dashboard Informasi")
    st.subheader("Klasifikasi Penerima Bantuan Sosial Desa Cikembar")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.dataset is not None:
            st.metric("Total Dataset", f"{len(st.session_state.dataset)} Warga")
        else:
            st.metric("Total Dataset", "Belum ada")
    with col2:
        if st.session_state.model:
            st.metric("Status Model", "✅ Siap Prediksi")
        else:
            st.metric("Status Model", "❌ Belum Dilatih")
    with col3:
        st.metric("Riwayat Prediksi", f"{len(st.session_state.riwayat_prediksi)}")

    st.markdown("---")
    st.write("Sistem ini menggunakan algoritma **Naive Bayes** untuk mengklasifikasikan warga apakah **Layak** atau **Tidak Layak** menerima bantuan sosial.")

# -------------------------------
# Upload & Training
# -------------------------------
elif page == "Upload Dataset & Latih Model":
    st.title("📁 Upload Dataset & Latih Model")

    uploaded_file = st.file_uploader("Upload file dataset (.csv / .xlsx)", type=["csv","xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.dataset = df
            st.success("✅ Dataset berhasil diupload!")
            st.dataframe(df.head())

            if st.button("🚀 Latih Model"):
                model, le_rumah, le_target, acc, report = train_model(df)
                if model:
                    st.session_state.model = model
                    st.session_state.le_rumah = le_rumah
                    st.session_state.le_target = le_target
                    st.success(f"Model berhasil dilatih. Akurasi: {acc:.2f}")
                    st.subheader("📊 Classification Report")
                    st.dataframe(pd.DataFrame(report).transpose())
        except Exception as e:
            st.error(f"Gagal membaca dataset: {e}")

# -------------------------------
# Prediksi Manual
# -------------------------------
elif page == "Prediksi Manual":
    st.title("🔮 Prediksi Manual")

    if st.session_state.model is None:
        st.warning("Latih model terlebih dahulu di menu Upload Dataset.")
    else:
        usia = st.number_input("Usia Kepala Keluarga", min_value=18, max_value=100, value=40)
        pendapatan = st.number_input("Pendapatan Bulanan", min_value=0, value=1000000, step=100000)
        jumlah = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=20, value=4)
        kepemilikan = st.selectbox("Kepemilikan Rumah", st.session_state.le_rumah.classes_)

        if st.button("Prediksi"):
            data_input = {
                "usia": usia,
                "pendapatan": pendapatan,
                "jumlah_anggota": jumlah,
                "kepemilikan_rumah": kepemilikan
            }
            pred, prob = predict_single(st.session_state.model, st.session_state.le_rumah, st.session_state.le_target, data_input)
            if pred:
                st.success(f"Hasil Prediksi: **{pred}**")
                st.write("Probabilitas:", prob)
                st.session_state.riwayat_prediksi.append({**data_input, "hasil": pred})

# -------------------------------
# Riwayat
# -------------------------------
elif page == "Riwayat Prediksi":
    st.title("📜 Riwayat Prediksi")
    if len(st.session_state.riwayat_prediksi) == 0:
        st.info("Belum ada prediksi yang disimpan.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.riwayat_prediksi))
