import streamlit as st
import pandas as pd
import io

# ==============================
# Konfigurasi halaman
# ==============================
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

st.title("📊 Sistem Klasifikasi Penerima Bantuan Sosial")
st.write("Aplikasi ini mengklasifikasikan warga **Layak / Tidak Layak** menerima bantuan sosial berdasarkan status kesejahteraan.")

# ==============================
# Fungsi mapping bansos
# ==============================
def klasifikasi_bansos(status):
    mapping = {
        "Miskin": "Layak",
        "Rentan Miskin": "Layak",
        "Sejahtera": "Tidak Layak",
        "Sangat Sejahtera": "Tidak Layak"
    }
    return mapping.get(status, "Tidak Diketahui")

# ==============================
# Navigasi sidebar
# ==============================
menu = st.sidebar.radio("Pilih Menu", ["📂 Upload Dataset", "✍️ Prediksi Manual"])

# ==============================
# Halaman Upload Dataset
# ==============================
if menu == "📂 Upload Dataset":
    st.header("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Pilih file Excel atau CSV dataset penduduk", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        try:
            # Baca file sesuai format
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("✅ Dataset berhasil diupload!")
            st.write("### Contoh Data Awal")
            st.dataframe(df.head())

            # Pastikan ada kolom target
            if "Status_Kesejahteraan" not in df.columns:
                st.error("❌ Dataset tidak memiliki kolom 'Status_Kesejahteraan'.")
            else:
                # Tambahkan kolom klasifikasi
                df["Keterangan_Layak"] = df["Status_Kesejahteraan"].apply(klasifikasi_bansos)

                st.write("### Hasil Klasifikasi")
                if "Nama" in df.columns:
                    st.dataframe(df[["Nama", "Status_Kesejahteraan", "Keterangan_Layak"]])
                else:
                    st.dataframe(df[["Status_Kesejahteraan", "Keterangan_Layak"]])

                # Statistik ringkas
                st.write("### 📊 Statistik Hasil")
                stats = df["Keterangan_Layak"].value_counts()
                st.bar_chart(stats)

                # Download hasil
                st.write("### 💾 Download Hasil")
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

# ==============================
# Halaman Prediksi Manual
# ==============================
elif menu == "✍️ Prediksi Manual":
    st.header("✍️ Prediksi Manual")

    with st.form("prediksi_form"):
        nama = st.text_input("Nama Warga")
        usia = st.number_input("Usia Kepala Keluarga", min_value=0, max_value=120, value=30)
        pendapatan = st.number_input("Pendapatan Bulanan (Rp)", min_value=0, step=100000, value=1000000)
        anggota = st.number_input("Jumlah Anggota Keluarga", min_value=1, value=4)
        status = st.selectbox("Status Kesejahteraan", ["Miskin", "Rentan Miskin", "Sejahtera", "Sangat Sejahtera"])

        submitted = st.form_submit_button("Prediksi")

        if submitted:
            hasil = klasifikasi_bansos(status)
            st.subheader("📌 Hasil Prediksi")
            st.write(f"**Nama:** {nama}")
            st.write(f"**Usia Kepala Keluarga:** {usia} tahun")
            st.write(f"**Pendapatan Bulanan:** Rp {pendapatan:,}")
            st.write(f"**Jumlah Anggota Keluarga:** {anggota}")
            st.write(f"**Status Kesejahteraan:** {status}")
            st.success(f"➡️ **Keterangan: {hasil} menerima bantuan sosial**")
