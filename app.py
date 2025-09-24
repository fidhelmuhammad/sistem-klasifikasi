# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ------------------------------
# Halaman Dashboard
# ------------------------------
def halaman_dashboard():
    st.title("📊 Sistem Klasifikasi Penerima Bantuan Sosial")
    st.subheader("Desa Cikembar")

    st.markdown("""
    ### Deskripsi  
    Sistem ini membantu menentukan siapa saja yang **berhak** menerima bantuan sosial di Desa Cikembar
    dengan algoritma **Naive Bayes**.

    ### Tujuan  
    - Membantu proses klasifikasi penerima bansos secara objektif.  
    - Mempercepat pengolahan data penerima bantuan.  
    - Mengurangi kesalahan subjektif dalam penentuan penerima.  

    ### Manfaat  
    - Transparansi dalam penyaluran bansos.  
    - Efisiensi waktu dan biaya.  
    - Meningkatkan keadilan dalam distribusi bantuan sosial.  
    """)

    st.info("Navigasi ada di sidebar: **Dashboard**, **Prediksi**, dan **Riwayat Prediksi**.")


# ------------------------------
# Halaman Prediksi
# ------------------------------
def halaman_prediksi():
    st.title("🔮 Prediksi Penerima Bantuan Sosial")

    uploaded_model = st.file_uploader("📂 Upload model (.joblib)", type=["joblib"])

    if uploaded_model is not None:
        data = joblib.load(uploaded_model)
        pipeline = data["pipeline"]
        classes = data["classes"]

        st.success("✅ Model berhasil dimuat")

        option = st.radio("Pilih metode input:", ["Input Manual", "Upload CSV/Excel"])

        if option == "Input Manual":
            umur = st.number_input("Umur", min_value=18, max_value=100, value=30)
            pekerjaan = st.selectbox("Pekerjaan", ["Petani", "Buruh", "Pedagang", "Tidak Bekerja"])
            penghasilan = st.number_input("Penghasilan (Rp)", min_value=0, value=1000000)

            input_df = pd.DataFrame({
                "umur": [umur],
                "pekerjaan": [pekerjaan],
                "penghasilan": [penghasilan]
            })

            if st.button("Prediksi"):
                pred = pipeline.predict(input_df)[0]
                hasil = classes[pred]

                st.success(f"Hasil Prediksi: **{hasil}**")

                # Simpan ke riwayat
                if "riwayat" not in st.session_state:
                    st.session_state.riwayat = []
                st.session_state.riwayat.append({"Input": input_df.to_dict(orient="records")[0], "Hasil": hasil})

        else:
            file_data = st.file_uploader("Upload file data (CSV/Excel)", type=["csv", "xlsx", "xls"])
            if file_data is not None:
                if file_data.name.endswith(".csv"):
                    df_new = pd.read_csv(file_data)
                else:
                    df_new = pd.read_excel(file_data)

                st.write("### Data yang diupload")
                st.dataframe(df_new.head())

                if st.button("Prediksi Batch"):
                    preds = pipeline.predict(df_new)
                    df_new["Prediksi"] = [classes[p] for p in preds]

                    st.write("### Hasil Prediksi")
                    st.dataframe(df_new)

                    # Simpan ke riwayat
                    if "riwayat" not in st.session_state:
                        st.session_state.riwayat = []
                    for row in df_new.to_dict(orient="records"):
                        st.session_state.riwayat.append({"Input": row, "Hasil": row["Prediksi"]})

                    # Download hasil
                    csv = df_new.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Hasil Prediksi",
                        data=csv,
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )


# ------------------------------
# Halaman Riwayat Prediksi
# ------------------------------
def halaman_riwayat():
    st.title("📝 Riwayat Prediksi")

    if "riwayat" not in st.session_state or len(st.session_state.riwayat) == 0:
        st.warning("Belum ada riwayat prediksi.")
    else:
        df_history = pd.DataFrame(st.session_state.riwayat)
        st.dataframe(df_history)


# ------------------------------
# Main App
# ------------------------------
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih Halaman", ["Dashboard", "Prediksi", "Riwayat Prediksi"])

if halaman == "Dashboard":
    halaman_dashboard()
elif halaman == "Prediksi":
    halaman_prediksi()
elif halaman == "Riwayat Prediksi":
    halaman_riwayat()
