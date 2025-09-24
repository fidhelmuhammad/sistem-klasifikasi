import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import io

st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

# ================================
# Fungsi Alasan Prediksi
# ================================
def alasan_bansos_row(row):
    """Menentukan alasan mengapa seseorang Layak / Tidak Layak"""
    if row["Keterangan_Layak"] == "Layak":
        alasan = []
        if row["Pendapatan_Bulanan"] < 1500000:
            alasan.append(f"Pendapatan rendah (Rp {row['Pendapatan_Bulanan']:,})")
        if row["Jumlah_Anggota_Keluarga"] >= 5:
            alasan.append(f"Tanggungan keluarga besar ({row['Jumlah_Anggota_Keluarga']} orang)")
        if row["Kepemilikan_Rumah"] == "Tidak":
            alasan.append("Tidak memiliki rumah pribadi")
        if not alasan:
            alasan.append("Kondisi ekonomi terbatas")
        return ", ".join(alasan) + " → Layak menerima bansos."
    else:
        alasan = []
        if row["Pendapatan_Bulanan"] >= 1500000:
            alasan.append(f"Pendapatan cukup tinggi (Rp {row['Pendapatan_Bulanan']:,})")
        if row["Jumlah_Anggota_Keluarga"] < 5:
            alasan.append(f"Tanggungan keluarga kecil ({row['Jumlah_Anggota_Keluarga']} orang)")
        if row["Kepemilikan_Rumah"] == "Ya":
            alasan.append("Sudah memiliki rumah pribadi")
        if not alasan:
            alasan.append("Kondisi ekonomi memadai")
        return ", ".join(alasan) + " → Tidak Layak menerima bansos."

# ================================
# Fungsi Training Model
# ================================
def train_model(df):
    df = df.copy()

    # Encode kolom
    le_rumah = LabelEncoder()
    le_target = LabelEncoder()

    df["Kepemilikan_Rumah_encoded"] = le_rumah.fit_transform(df["Kepemilikan_Rumah"])
    df["Status_Kesejahteraan_encoded"] = le_target.fit_transform(df["Status_Kesejahteraan"])

    X = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan",
            "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah_encoded"]]
    y = df["Status_Kesejahteraan_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return model, le_rumah, le_target, acc

# ================================
# Sidebar
# ================================
st.sidebar.title("Navigasi Sistem")
page = st.sidebar.radio("Pilih Halaman:", ["Dashboard", "Upload & Prediksi", "Daftar Penerima Bansos"])

# ================================
# Dashboard
# ================================
if page == "Dashboard":
    st.title("🏠 Dashboard Informasi")
    st.subheader("Klasifikasi Penerima Bantuan Sosial Desa Cikembar")
    st.markdown("---")

    st.markdown("""
    Sistem ini menggunakan **Naive Bayes** untuk mengklasifikasikan warga desa apakah **Layak** atau **Tidak Layak**
    menerima bantuan sosial.  

    **Fitur yang digunakan:**
    - Usia Kepala Keluarga
    - Pendapatan Bulanan
    - Jumlah Anggota Keluarga
    - Kepemilikan Rumah
    """)

# ================================
# Upload & Prediksi
# ================================
elif page == "Upload & Prediksi":
    st.title("📁 Upload Dataset & Prediksi")

    uploaded_file = st.file_uploader("Upload dataset penduduk (Excel/CSV)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset berhasil diupload")
        st.dataframe(df.head())

        # Latih model
        model, le_rumah, le_target, acc = train_model(df)
        st.info(f"Model dilatih dengan akurasi: **{acc:.2f}**")

        # Prediksi otomatis
        df["Kepemilikan_Rumah_encoded"] = le_rumah.transform(df["Kepemilikan_Rumah"])
        X_all = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan",
                    "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah_encoded"]]
        y_pred = model.predict(X_all)
        df["Prediksi_Status"] = le_target.inverse_transform(y_pred)

        # Mapping Layak / Tidak Layak
        mapping = {
            "Miskin": "Layak",
            "Rentan Miskin": "Layak",
            "Sejahtera": "Tidak Layak",
            "Sangat Sejahtera": "Tidak Layak"
        }
        df["Keterangan_Layak"] = df["Prediksi_Status"].map(mapping)

        # Alasan Prediksi
        df["Keterangan_Alasan"] = df.apply(alasan_bansos_row, axis=1)

        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df[["Nama", "Prediksi_Status", "Keterangan_Layak", "Keterangan_Alasan"]])

        # Download hasil
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Hasil Prediksi", buffer,
                           file_name="hasil_prediksi_bansos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ================================
# Daftar Penerima Bansos
# ================================
elif page == "Daftar Penerima Bansos":
    st.title("📋 Daftar Penerima Bansos")

    uploaded_file = st.file_uploader("Upload dataset untuk lihat penerima bansos", type=["csv", "xlsx"], key="daftar")
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Latih & Prediksi
        model, le_rumah, le_target, acc = train_model(df)
        df["Kepemilikan_Rumah_encoded"] = le_rumah.transform(df["Kepemilikan_Rumah"])
        X_all = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan",
                    "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah_encoded"]]
        y_pred = model.predict(X_all)
        df["Prediksi_Status"] = le_target.inverse_transform(y_pred)

        mapping = {
            "Miskin": "Layak",
            "Rentan Miskin": "Layak",
            "Sejahtera": "Tidak Layak",
            "Sangat Sejahtera": "Tidak Layak"
        }
        df["Keterangan_Layak"] = df["Prediksi_Status"].map(mapping)
        df["Keterangan_Alasan"] = df.apply(alasan_bansos_row, axis=1)

        penerima = df[df["Keterangan_Layak"] == "Layak"]

        st.success(f"Total penerima bansos: **{len(penerima)} orang**")
        st.dataframe(penerima[["Nama", "Prediksi_Status", "Keterangan_Alasan"]])

        # Download penerima
        buffer = io.BytesIO()
        penerima.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Daftar Penerima", buffer,
                           file_name="daftar_penerima_bansos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
