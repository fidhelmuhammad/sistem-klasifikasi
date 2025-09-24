import streamlit as st
import pandas as pd
import io

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

st.title("📊 Sistem Klasifikasi Penerima Bantuan Sosial")
st.write("Aplikasi ini akan otomatis menentukan siapa saja yang **Layak menerima bantuan sosial** berdasarkan status kesejahteraan.")

# Fungsi klasifikasi
def klasifikasi_bansos(status):
    mapping = {
        "Miskin": "Layak",
        "Rentan Miskin": "Layak",
        "Sejahtera": "Tidak Layak",
        "Sangat Sejahtera": "Tidak Layak"
    }
    return mapping.get(status, "Tidak Diketahui")

# Upload file
st.header("📂 Upload Dataset")
uploaded_file = st.file_uploader("Pilih file Excel atau CSV dataset penduduk", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        # Baca file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset berhasil diupload!")
        st.write("### Contoh Data Awal")
        st.dataframe(df.head())

        # Pastikan ada kolom status kesejahteraan
        if "Status_Kesejahteraan" not in df.columns:
            st.error("❌ Dataset tidak memiliki kolom 'Status_Kesejahteraan'.")
        else:
            # Tambahkan kolom klasifikasi
            df["Keterangan_Layak"] = df["Status_Kesejahteraan"].apply(klasifikasi_bansos)

            # Filter penerima bansos
            penerima_bansos = df[df["Keterangan_Layak"] == "Layak"]

            st.write("### ✅ Daftar Warga Layak Mendapatkan Bansos")
            if "Nama" in penerima_bansos.columns:
                st.dataframe(penerima_bansos[["Nama", "Status_Kesejahteraan", "Keterangan_Layak"]])
            else:
                st.dataframe(penerima_bansos[["Status_Kesejahteraan", "Keterangan_Layak"]])

            # Statistik jumlah
            st.write("### 📊 Statistik")
            stats = df["Keterangan_Layak"].value_counts()
            st.bar_chart(stats)

            st.metric("Total Warga", len(df))
            st.metric("Layak (Dapat Bansos)", len(penerima_bansos))
            st.metric("Tidak Layak (Tidak Dapat Bansos)", len(df) - len(penerima_bansos))

            # Download hasil lengkap
            st.write("### 💾 Download Hasil Klasifikasi")
            output = io.BytesIO()
            df.to_excel(output, index=False, engine="openpyxl")
            output.seek(0)
            st.download_button(
                label="📥 Download Hasil Klasifikasi Excel",
                data=output,
                file_name="hasil_klasifikasi_bansos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Terjadi error saat memproses file: {e}")
else:
    st.info("👆 Silakan upload file dataset untuk mulai klasifikasi.")
