# app.py
import streamlit as st
import pandas as pd
import joblib
import tempfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ------------------------------
# Halaman 1: Informasi Sistem
# ------------------------------
def halaman_informasi():
    st.title("📊 Sistem Klasifikasi Penerima Bantuan Sosial")
    st.subheader("Desa Cikembar")
    st.markdown("""
    ### Deskripsi  
    Sistem ini dirancang untuk membantu pemerintah Desa Cikembar dalam menentukan siapa saja yang **berhak** 
    menerima bantuan sosial dengan menggunakan algoritma **Naive Bayes**.

    ### Tujuan  
    - Membantu proses klasifikasi penerima bansos secara objektif.  
    - Mempercepat pengolahan data penerima bantuan.  
    - Mengurangi kesalahan subjektif dalam penentuan penerima.  

    ### Manfaat  
    - Transparansi dalam penyaluran bansos.  
    - Efisiensi waktu dan biaya.  
    - Meningkatkan keadilan dalam distribusi bantuan sosial.  
    """)

# ------------------------------
# Halaman 2: Pelatihan Model
# ------------------------------
def halaman_pelatihan():
    st.title("⚙️ Pelatihan Model Naive Bayes")

    uploaded_file = st.file_uploader("📂 Upload dataset (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Sample")
        st.dataframe(df.head())

        target_col = st.selectbox("Pilih kolom target (label)", df.columns)
        feature_cols = st.multiselect("Pilih kolom fitur", [c for c in df.columns if c != target_col])

        if st.button("Latih Model"):
            X = df[feature_cols]
            y = df[target_col]

            # Encode target
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Preprocessing pipeline
            numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
            categorical_features = X.select_dtypes(include=["object"]).columns

            numeric_transformer = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", GaussianNB())
            ])

            # Split & Train
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Evaluasi
            st.subheader("📈 Hasil Evaluasi")
            st.text(classification_report(y_test, y_pred, target_names=le.classes_))

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_, ax=ax)
            st.pyplot(fig)

            # Simpan model ke file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                joblib.dump({"pipeline": pipeline, "classes": le.classes_}, tmp.name)
                st.success("✅ Model berhasil dilatih dan disimpan")
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        label="💾 Download Model",
                        data=f,
                        file_name="model_bansos.joblib"
                    )

# ------------------------------
# Halaman 3: Prediksi
# ------------------------------
def halaman_prediksi():
    st.title("🔮 Prediksi Penerima Bantuan Sosial")

    uploaded_model = st.file_uploader("📂 Upload model (.joblib)", type="joblib")

    if uploaded_model:
        data = joblib.load(uploaded_model)
        pipeline = data["pipeline"]
        classes = data["classes"]

        st.success("✅ Model berhasil dimuat")

        option = st.radio("Pilih metode input:", ["Input Manual", "Upload CSV"])

        if option == "Input Manual":
            # Contoh input sederhana (ubah sesuai dataset Anda)
            st.subheader("Masukkan Data")
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

        else:
            st.subheader("Upload Data untuk Prediksi Batch")
            file_csv = st.file_uploader("Upload file CSV", type="csv")
            if file_csv:
                df_new = pd.read_csv(file_csv)
                st.write("### Data yang diupload")
                st.dataframe(df_new.head())

                if st.button("Prediksi Batch"):
                    preds = pipeline.predict(df_new)
                    df_new["Prediksi"] = [classes[p] for p in preds]
                    st.write("### Hasil Prediksi")
                    st.dataframe(df_new)

                    # Download hasil
                    csv = df_new.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Hasil Prediksi",
                        data=csv,
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )

# ------------------------------
# Main App
# ------------------------------
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih Halaman", ["Informasi", "Pelatihan Model", "Prediksi"])

if halaman == "Informasi":
    halaman_informasi()
elif halaman == "Pelatihan Model":
    halaman_pelatihan()
elif halaman == "Prediksi":
    halaman_prediksi()
