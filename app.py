import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# === KONFIGURASI APLIKASI ===
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

MODEL_FILE = "model_bansos.pkl"
HISTORY_FILE = "riwayat.csv"

# ===== DASHBOARD =====
def halaman_dashboard():
    st.title("📊 Dashboard Sistem Klasifikasi Bantuan Sosial")
    st.write("""
    Sistem ini digunakan untuk menentukan siapa yang **berhak** dan siapa yang
    **belum layak** menerima bantuan sosial di Desa Cikembar menggunakan
    algoritma **Naive Bayes**.
    """)

    st.subheader("📂 Upload Dataset untuk Melatih Model")
    uploaded = st.file_uploader("Upload file CSV/Excel", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded, engine="openpyxl")

        st.write("📑 Data awal:")
        st.dataframe(df.head())

        target_col = st.selectbox("Pilih kolom target (label)", df.columns)

        if st.button("Latih Model Naive Bayes"):
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Encode kolom kategori
            encoders = {}
            for col in X.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le

            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = GaussianNB()
            model.fit(X_train, y_train)

            acc = accuracy_score(y_test, model.predict(X_test))
            st.success(f"✅ Model berhasil dilatih. Akurasi: {acc:.2f}")

            joblib.dump({"model": model, "encoders": encoders, "target": target_encoder}, MODEL_FILE)
            st.info("Model tersimpan ke file `model_bansos.pkl`")

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
        st.warning("⚠️ Model belum ada. Silakan latih model dulu di halaman Dashboard.")
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
        file_data = st.file_uploader("📂 Upload file CSV/Excel untuk prediksi", type=["csv", "xlsx", "xls"])
        if file_data is not None:
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

            simpan_riwayat(
                df_new[["Nama", "Pekerjaan", "Pendidikan", "Penghasilan", "Tanggungan", "Hasil Prediksi"]]
            )

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
