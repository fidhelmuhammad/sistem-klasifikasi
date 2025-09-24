import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# === KONFIGURASI APLIKASI ===
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

MODEL_FILE = "model_bansos.pkl"
HISTORY_FILE = "riwayat.csv"

# ===== DASHBOARD =====
def halaman_dashboard():
    st.title("📊 Dashboard Sistem Klasifikasi Bantuan Sosial")
    st.write("""
    Sistem ini dibuat untuk membantu Desa Cikembar dalam menentukan siapa yang **berhak**
    dan siapa yang **belum layak** menerima **Bantuan Sosial** menggunakan algoritma
    **Naive Bayes**.
    
    ### Tujuan
    - Membantu perangkat desa dalam pengambilan keputusan.
    - Meningkatkan transparansi penyaluran bantuan.
    - Mengurangi kesalahan subjektif.

    ### Manfaat
    - Cepat dalam klasifikasi penerima.
    - Data dapat dikelola dan diupdate.
    - Riwayat prediksi tersimpan otomatis.
    """)

# ===== SIMPAN RIWAYAT =====
def simpan_riwayat(df):
    if os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)

# ===== PREDIKSI =====
def halaman_prediksi():
    st.title("🔮 Prediksi Penerima Bantuan Sosial")

    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ Model belum tersedia. Silakan latih model dulu secara lokal dan upload ke folder.")
        return

    model_data = joblib.load(MODEL_FILE)
    model = model_data["model"]
    encoders = model_data["encoders"]
    target_encoder = model_data["target"]

    pilihan = st.radio("Pilih metode input:", ["Input Manual", "Upload File"])

    if pilihan == "Input Manual":
        nama = st.text_input("Nama")
        pekerjaan = st.selectbox("Pekerjaan", ["Petani", "Buruh", "Pedagang", "Tidak Bekerja"])
        pendidikan = st.selectbox("Pendidikan", ["SD", "SMP", "SMA", "S1"])
        penghasilan = st.number_input("Penghasilan per Bulan (Rp)", min_value=0)
        tanggungan = st.number_input("Jumlah Tanggungan", min_value=0, step=1)

        if st.button("Prediksi"):
            data = pd.DataFrame([[pekerjaan, pendidikan, penghasilan, tanggungan]],
                                columns=["Pekerjaan", "Pendidikan", "Penghasilan", "Tanggungan"])
            for col in ["Pekerjaan", "Pendidikan"]:
                data[col] = encoders[col].transform(data[col])
            pred = model.predict(data)[0]
            hasil = target_encoder.inverse_transform([pred])[0]

            st.success(f"✅ {nama} diprediksi: **{hasil}**")

            df_hist = pd.DataFrame([[nama, pekerjaan, pendidikan, penghasilan, tanggungan, hasil]],
                                   columns=["Nama", "Pekerjaan", "Pendidikan", "Penghasilan", "Tanggungan", "Hasil"])
            simpan_riwayat(df_hist)

    else:
        file_data = st.file_uploader("📂 Upload file CSV/Excel", type=["csv", "xlsx", "xls"])
        if file_data is not None:
            try:
                if file_data.name.endswith(".csv"):
                    df_new = pd.read_csv(file_data)
                else:
                    df_new = pd.read_excel(file_data, engine="openpyxl")

                st.write("📑 Data yang diupload:")
                st.dataframe(df_new.head())

                for col in ["Pekerjaan", "Pendidikan"]:
                    if col in df_new.columns:
                        df_new[col] = encoders[col].transform(df_new[col])

                preds = model.predict(df_new.drop(columns=["Nama"]))
                df_new["Hasil Prediksi"] = target_encoder.inverse_transform(preds)

                st.success("✅ Prediksi selesai")
                st.dataframe(df_new)

                simpan_riwayat(df_new[["Nama", "Pekerjaan", "Pendidikan", "Penghasilan", "Tanggungan", "Hasil Prediksi"]])

            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

# ===== RIWAYAT PREDIKSI =====
def halaman_riwayat():
    st.title("📜 Riwayat Prediksi")
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.dataframe(df)
    else:
        st.info("Belum ada riwayat prediksi.")

# ===== MAIN MENU =====
menu = st.sidebar.radio("Navigasi", ["Dashboard", "Prediksi", "Riwayat Prediksi"])

if menu == "Dashboard":
    halaman_dashboard()
elif menu == "Prediksi":
    halaman_prediksi()
else:
    halaman_riwayat()
