import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
import os
from datetime import datetime

# ===============================
# Fungsi bantu
# ===============================
def load_dataset(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        st.error("Format file harus CSV atau Excel (XLSX).")
        return None
    return df

def encode_features(df, target_col):
    le_dict = {}
    df_encoded = df.copy()

    for col in df.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    return X, y, le_dict

def train_model(df, target_col):
    X, y, le_dict = encode_features(df, target_col)

    if len(y.unique()) < 2:
        st.error("Target hanya memiliki 1 kelas. Harus ada minimal 2 kelas.")
        return None, None, None, None

    stratify_opt = y if y.nunique() > 1 and y.value_counts().min() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_opt
    )

    model = CategoricalNB()
    model.fit(X_train, y_train)

    return model, le_dict, X_test, y_test

def simpan_arsip(hasil):
    os.makedirs("arsip", exist_ok=True)
    path = f"arsip/arsip_prediksi.csv"

    df = pd.DataFrame(hasil)
    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

def load_arsip():
    if not os.path.exists("arsip"):
        return pd.DataFrame()
    arsip_files = [f for f in os.listdir("arsip") if f.endswith(".csv")]
    all_data = []
    for file in arsip_files:
        df = pd.read_csv(os.path.join("arsip", file))
        all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# ===============================
# Halaman
# ===============================
def home_page():
    st.title("📊 Sistem Klasifikasi Penerima Bantuan Sosial")
    st.write("""
    Sistem ini dibuat untuk membantu **menentukan penerima bantuan sosial** agar lebih tepat sasaran.
    
    ### Tujuan:
    - Memastikan bantuan diberikan hanya kepada masyarakat yang **layak** menerima.
    - Mengurangi potensi salah sasaran.
    - Memberikan transparansi jumlah penerima dan non-penerima.

    ### Cara Kerja:
    1. Upload dataset penerima bantuan (CSV atau Excel).
    2. Lakukan training model dengan metode **Naïve Bayes**.
    3. Lihat hasil prediksi semua data.
    4. Hasil prediksi disimpan ke arsip untuk monitoring.
    """)

def training_page():
    st.header("📂 Upload Data & Training Model")
    uploaded_file = st.file_uploader("Upload dataset (CSV/XLSX)", type=["csv", "xlsx"])

    if uploaded_file:
        df = load_dataset(uploaded_file)
        if df is not None:
            st.subheader("Data Awal (5 baris pertama)")
            st.dataframe(df.head())

            target_col = st.selectbox("Pilih kolom target (label)", df.columns)

            if st.button("🚀 Mulai Training"):
                model, le_dict, X_test, y_test = train_model(df, target_col)

                if model:
                    st.success("✅ Model berhasil dilatih!")
                    st.session_state["model"] = model
                    st.session_state["le_dict"] = le_dict
                    st.session_state["data"] = df
                    st.session_state["target_col"] = target_col

def prediction_page():
    st.header("🔍 Prediksi Penerima Bantuan")
    if "model" not in st.session_state:
        st.warning("⚠️ Silakan lakukan training model terlebih dahulu di halaman Upload & Training.")
        return

    df = st.session_state["data"]
    model = st.session_state["model"]
    le_dict = st.session_state["le_dict"]
    target_col = st.session_state["target_col"]

    if st.button("🔎 Tampilkan Prediksi"):
        df_encoded = df.copy()
        for col in df_encoded.columns:
            df_encoded[col] = le_dict[col].transform(df_encoded[col].astype(str))

        X_new = df_encoded.drop(columns=[target_col])
        y_pred = model.predict(X_new)

        df["Prediksi"] = le_dict[target_col].inverse_transform(y_pred)

        st.subheader("Hasil Prediksi Keseluruhan")
        for label in df["Prediksi"].unique():
            subset = df[df["Prediksi"] == label]
            st.write(f"🔹 Jumlah {label}: **{len(subset)} orang**")
            st.dataframe(subset)

        simpan_arsip(df)

def arsip_page():
    st.header("📑 Arsip Riwayat Prediksi")
    arsip_df = load_arsip()

    if arsip_df.empty:
        st.info("Belum ada arsip tersimpan.")
    else:
        st.dataframe(arsip_df)

# ===============================
# Main App
# ===============================
def main():
    st.sidebar.title("Navigasi")
    menu = st.sidebar.radio("Pilih Halaman", ["Home", "Upload & Training", "Prediksi", "Arsip Riwayat"])

    if menu == "Home":
        home_page()
    elif menu == "Upload & Training":
        training_page()
    elif menu == "Prediksi":
        prediction_page()
    elif menu == "Arsip Riwayat":
        arsip_page()

if __name__ == "__main__":
    main()
