import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import joblib

st.set_page_config(page_title="Klasifikasi Kesejahteraan", layout="wide")

# Sidebar navigasi
menu = st.sidebar.radio("Navigasi", ["Home", "Training", "Prediksi"])


# ==============================
# 1. HALAMAN HOME
# ==============================
if menu == "Home":
    st.title("📊 Sistem Klasifikasi Status Kesejahteraan Masyarakat")
    st.markdown("""
    Aplikasi ini dibuat untuk **menentukan status kesejahteraan masyarakat** 
    menggunakan metode **Naïve Bayes**.

    ### Navigasi
    - 📌 **Training** → latih model dengan dataset.  
    - 📌 **Prediksi** → prediksi status kesejahteraan untuk data baru.  
    """)

    try:
        df = pd.read_excel("dataset_penduduk_cikembar_enriched.xlsx")
        st.subheader("📂 Ringkasan Dataset (Default)")
        st.write(df.head())

        # Visualisasi distribusi target kalau ada
        if "Status_Kesejahteraan" in df.columns:
            st.subheader("Distribusi Status Kesejahteraan")
            fig, ax = plt.subplots()
            df["Status_Kesejahteraan"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)
    except Exception as e:
        st.warning("Dataset default tidak ditemukan. Silakan upload di menu Training.")


# ==============================
# 2. HALAMAN TRAINING
# ==============================
elif menu == "Training":
    st.title("📌 Training Model Naïve Bayes")

    uploaded_file = st.file_uploader("Upload dataset (Excel)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        try:
            df = pd.read_excel("dataset_penduduk_cikembar_enriched.xlsx")
        except:
            st.error("❌ Dataset tidak ditemukan. Silakan upload file Excel.")
            st.stop()

    st.write("📂 Preview Dataset", df.head())

    # Pilih target kolom
    target_col = st.selectbox("Pilih kolom target (status kesejahteraan)", df.columns)

    if st.button("🚀 Latih Model"):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Label Encoding
        le_dict = {}
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi sederhana
        labels = np.unique(y_test)
        target_names = le_target.inverse_transform(labels)
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        st.subheader("📋 Evaluasi Model")
        eval_data = []
        for i, label in enumerate(labels):
            total = cm[i].sum()
            benar = cm[i, i]
            salah = total - benar
            eval_data.append([target_names[i], total, benar, salah])

        eval_df = pd.DataFrame(
            eval_data,
            columns=["Kelas", "Total Data Uji", "Prediksi Benar", "Prediksi Salah"],
        )
        st.table(eval_df)

        # Simpan model
        joblib.dump((model, le_dict, le_target, target_col, list(X.columns)), "naive_bayes_model.pkl")
        st.success("✅ Model berhasil dilatih dan disimpan sebagai `naive_bayes_model.pkl`")


# ==============================
# 3. HALAMAN PREDIKSI
# ==============================
elif menu == "Prediksi":
    st.title("🔮 Prediksi Status Kesejahteraan")

    try:
        model, le_dict, le_target, target_col, feature_names = joblib.load("naive_bayes_model.pkl")
    except:
        st.error("❌ Model belum ada. Silakan latih model di halaman Training terlebih dahulu.")
        st.stop()

    st.info("Isi data berikut untuk memprediksi status kesejahteraan:")

    # Buat form input sesuai fitur yang dipakai saat training
    user_input = {}
    for col in feature_names:
        if col in le_dict:  # kolom kategorikal
            options = list(le_dict[col].classes_)
            pilihan = st.selectbox(f"{col}", options, key=col)
            user_input[col] = le_dict[col].transform([pilihan])[0]
        else:  # kolom numerik
            nilai = st.number_input(f"{col}", value=0, key=col)
            user_input[col] = nilai

    if st.button("Prediksi"):
        # Pastikan kolom sama persis dengan saat training
        X_new = pd.DataFrame([user_input], columns=feature_names)

        pred = model.predict(X_new)
        hasil = le_target.inverse_transform(pred)[0]
        st.success(f"📌 Hasil Prediksi: **{hasil}**")
