import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import joblib

# ----------------------------
# Helper functions
# ----------------------------

def preprocess_data(df, target_col):
    """
    Encode semua kolom kategorikal termasuk target
    """
    le_dict = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y, le_dict

def train_model(df, target_col):
    """
    Latih model Naive Bayes
    """
    X, y, le_dict = preprocess_data(df, target_col)

    # cek apakah cukup data untuk stratify
    stratify_option = y if len(y.unique()) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_option
    )

    model = CategoricalNB()
    model.fit(X_train, y_train)

    return model, le_dict, X_test, y_test

def save_history(input_data, prediction, filename="history.csv"):
    """
    Simpan riwayat prediksi
    """
    new_row = input_data.copy()
    new_row["Hasil Prediksi"] = prediction

    if os.path.exists(filename):
        old = pd.read_csv(filename)
        updated = pd.concat([old, pd.DataFrame([new_row])], ignore_index=True)
        updated.to_csv(filename, index=False)
    else:
        pd.DataFrame([new_row]).to_csv(filename, index=False)

def load_history(filename="history.csv"):
    """
    Load riwayat prediksi
    """
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return pd.DataFrame()

# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(page_title="Klasifikasi Penerima Bantuan Sosial", layout="wide")

def home_page():
    st.title("🎯 Sistem Klasifikasi Penerima Bantuan Sosial")
    st.markdown(
        """
        ## Tentang Sistem
        Website ini dibuat untuk membantu pemerintah desa dalam 
        **menentukan siapa yang layak dan tidak layak menerima bantuan sosial**
        menggunakan **Metode Naïve Bayes**.  
        
        ### 🎯 Tujuan:
        - Menyediakan sistem yang transparan dan akurat.  
        - Memastikan bantuan diberikan kepada masyarakat yang paling membutuhkan.  
        - Menyimpan riwayat prediksi agar bisa dijadikan arsip.  
        """
    )

def upload_and_training_page():
    st.title("📂 Upload Dataset & Training Model")

    uploaded_file = st.file_uploader(
        "Upload dataset Anda (CSV atau Excel)", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")

            st.success("✅ Dataset berhasil diupload!")
            st.dataframe(df.head())

            target_col = st.selectbox("Pilih kolom target (label):", df.columns)

            if st.button("🚀 Mulai Training"):
                model, le_dict, X_test, y_test = train_model(df, target_col)

                # simpan model & encoder
                joblib.dump((model, le_dict, target_col), "model.pkl")
                st.session_state["model_ready"] = True

                st.success("✅ Training selesai! Model siap digunakan.")

                st.write("### Contoh Data Uji")
                st.dataframe(X_test.head())

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")

def prediction_page():
    st.title("🔮 Prediksi Penerima Bantuan Sosial")

    if not os.path.exists("model.pkl"):
        st.warning("⚠️ Model belum dilatih. Silakan upload data dan latih model terlebih dahulu.")
        return

    model, le_dict, target_col = joblib.load("model.pkl")

    # ambil semua fitur kecuali target
    features = [col for col in le_dict.keys() if col != target_col]
    input_data = {}

    for col in features:
        options = le_dict[col].classes_
        val = st.selectbox(f"Pilih {col}:", options)
        input_data[col] = val

    if st.button("🔎 Prediksi"):
        try:
            # encode input sesuai encoder
            X_new = []
            for col in features:
                le = le_dict[col]
                X_new.append(le.transform([input_data[col]])[0])
            X_new = pd.DataFrame([X_new], columns=features)

            pred = model.predict(X_new)[0]
            pred_label = le_dict[target_col].inverse_transform([pred])[0]

            st.success(f"📌 Hasil Prediksi: **{pred_label}**")

            # simpan riwayat
            save_history(input_data, pred_label)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

def history_page():
    st.title("📜 Arsip / Riwayat Prediksi")
    df = load_history()
    if df.empty:
        st.info("Belum ada riwayat prediksi.")
    else:
        st.dataframe(df)

# ----------------------------
# Navigation
# ----------------------------

menu = st.sidebar.radio(
    "Navigasi", ["Home", "Upload & Training", "Prediksi", "Arsip"]
)

def main():
    if menu == "Home":
        home_page()
    elif menu == "Upload & Training":
        upload_and_training_page()
    elif menu == "Prediksi":
        prediction_page()
    elif menu == "Arsip":
        history_page()

if __name__ == "__main__":
    main()
