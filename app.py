import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import io

# =============================
# Konfigurasi halaman
# =============================
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

# =============================
# State awal
# =============================
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "model" not in st.session_state:
    st.session_state.model = None
if "le_target" not in st.session_state:
    st.session_state.le_target = None

# =============================
# Fungsi
# =============================
def train_model(df):
    """Latih model Naive Bayes dari dataset"""
    # Encode target
    le_target = LabelEncoder()
    df["Status_encoded"] = le_target.fit_transform(df["Status_Kesejahteraan"])

    # Fitur sederhana (bisa dikembangkan)
    X = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga"]]
    y = df["Status_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le_target.classes_)

    return model, le_target, acc, report


def klasifikasi_bansos(status):
    mapping = {
        "Miskin": "Layak",
        "Rentan Miskin": "Layak",
        "Sejahtera": "Tidak Layak",
        "Sangat Sejahtera": "Tidak Layak",
    }
    return mapping.get(status, "Tidak Diketahui")


def alasan_bansos(status):
    if status in ["Miskin", "Rentan Miskin"]:
        return "Kondisi ekonomi lemah, sehingga Layak mendapat bansos"
    elif status in ["Sejahtera", "Sangat Sejahtera"]:
        return "Kondisi ekonomi mencukupi, sehingga Tidak Layak mendapat bansos"
    else:
        return "Tidak diketahui"


# =============================
# Sidebar Navigasi
# =============================
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman", ["🏠 Dashboard", "📂 Upload & Prediksi", "✅ Daftar Penerima"]
)

# =============================
# Halaman Dashboard
# =============================
if page == "🏠 Dashboard":
    st.title("🏠 Dashboard Informasi")
    st.markdown("---")
    if st.session_state.dataset is not None:
        st.success(f"Dataset terupload dengan {len(st.session_state.dataset)} data")
    else:
        st.warning("Belum ada dataset yang diupload")

    if st.session_state.model:
        st.success("Model Naive Bayes sudah dilatih ✅")
    else:
        st.warning("Model belum dilatih ❌")

# =============================
# Halaman Upload & Prediksi
# =============================
elif page == "📂 Upload & Prediksi":
    st.title("📂 Upload Dataset & Prediksi Otomatis")

    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Pastikan kolom ada
            required = [
                "Nama",
                "Usia_Kepala_Keluarga",
                "Pendapatan_Bulanan",
                "Jumlah_Anggota_Keluarga",
                "Status_Kesejahteraan",
            ]
            if not all(col in df.columns for col in required):
                st.error(f"Dataset harus memiliki kolom: {required}")
            else:
                st.session_state.dataset = df

                st.write("### Contoh Data Awal")
                st.dataframe(df.head())

                if st.button("🚀 Latih Model"):
                    model, le_target, acc, report = train_model(df)
                    st.session_state.model = model
                    st.session_state.le_target = le_target

                    st.success(f"Model berhasil dilatih dengan akurasi: {acc:.2f}")
                    st.text("Classification Report:")
                    st.text(report)

                if st.session_state.model:
                    # Prediksi otomatis untuk semua data
                    X_all = df[
                        ["Usia_Kepala_Keluarga", "Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga"]
                    ]
                    pred_encoded = st.session_state.model.predict(X_all)
                    pred_labels = st.session_state.le_target.inverse_transform(pred_encoded)

                    df["Prediksi_Status"] = pred_labels
                    df["Keterangan_Layak"] = df["Prediksi_Status"].apply(klasifikasi_bansos)
                    df["Keterangan_Alasan"] = df["Prediksi_Status"].apply(alasan_bansos)

                    st.write("### Hasil Prediksi Otomatis")
                    st.dataframe(df[["Nama", "Prediksi_Status", "Keterangan_Layak", "Keterangan_Alasan"]])

                    # Simpan ke state
                    st.session_state.dataset = df

                    # Download hasil
                    output = io.BytesIO()
                    df.to_excel(output, index=False, engine="openpyxl")
                    output.seek(0)
                    st.download_button(
                        label="📥 Download Hasil Prediksi Excel",
                        data=output,
                        file_name="hasil_prediksi_bansos.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

        except Exception as e:
            st.error(f"Terjadi error saat membaca file: {e}")

# =============================
# Halaman Daftar Penerima
# =============================
elif page == "✅ Daftar Penerima":
    st.title("✅ Daftar Warga Layak Menerima Bansos")

    if st.session_state.dataset is None or "Keterangan_Layak" not in st.session_state.dataset.columns:
        st.warning("Silakan upload dataset dan lakukan prediksi terlebih dahulu.")
    else:
        penerima = st.session_state.dataset[
            st.session_state.dataset["Keterangan_Layak"] == "Layak"
        ]

        st.write("### Daftar Penerima Bansos")
        st.dataframe(penerima[["Nama", "Prediksi_Status", "Keterangan_Alasan"]])

        # Download penerima saja
        output = io.BytesIO()
        penerima.to_excel(output, index=False, engine="openpyxl")
        output.seek(0)
        st.download_button(
            label="📥 Download Daftar Penerima Saja",
            data=output,
            file_name="daftar_penerima_bansos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
