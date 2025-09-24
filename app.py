import streamlit as st
import pandas as pd
import io

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

# Session state buat simpan dataset
if "dataset" not in st.session_state:
    st.session_state.dataset = None

# Navigasi Sidebar
st.sidebar.title("Navigasi Sistem")
page = st.sidebar.selectbox("Pilih Halaman:", 
                            ["🏠 Dashboard", "📂 Upload Dataset & Prediksi", "📋 Daftar Penerima Bansos"])

# =============================
# 1. DASHBOARD
# =============================
if page == "🏠 Dashboard":
    st.title("🏠 Dashboard Informasi")
    st.subheader("Klasifikasi Penerima Bantuan Sosial Desa Cikembar")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.dataset is not None:
            st.metric("Total Dataset", f"{len(st.session_state.dataset)} Warga")
        else:
            st.metric("Total Dataset", "Belum ada data")

    with col2:
        if st.session_state.dataset is not None:
            stats = st.session_state.dataset["Keterangan_Layak"].value_counts()
            st.metric("Penerima Bansos (Layak)", stats.get("Layak", 0))
        else:
            st.metric("Penerima Bansos (Layak)", "0")

    st.markdown("---")
    st.write("Sistem ini akan mengklasifikasikan warga berdasarkan **Status_Kesejahteraan** menjadi:")
    st.write("- **Miskin / Rentan Miskin → Layak (mendapat bansos)**")
    st.write("- **Sejahtera / Sangat Sejahtera → Tidak Layak (tidak mendapat bansos)**")

# =============================
# 2. UPLOAD & PREDIKSI
# =============================
elif page == "📂 Upload Dataset & Prediksi":
    st.title("📂 Upload Dataset & Prediksi Otomatis")

    uploaded_file = st.file_uploader("Pilih file Excel atau CSV dataset penduduk", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        try:
            # Baca dataset
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
                # Mapping Layak / Tidak Layak
                mapping = {
                    "Miskin": "Layak",
                    "Rentan Miskin": "Layak",
                    "Sejahtera": "Tidak Layak",
                    "Sangat Sejahtera": "Tidak Layak"
                }
                df["Keterangan_Layak"] = df["Status_Kesejahteraan"].map(mapping)

                # Simpan ke session state
                st.session_state.dataset = df

                st.write("### Hasil Klasifikasi Otomatis")
                st.dataframe(df[["Nama", "Status_Kesejahteraan", "Keterangan_Layak"]])

                # Grafik ringkas
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

# =============================
# 3. DAFTAR PENERIMA BANSOS
# =============================
elif page == "📋 Daftar Penerima Bansos":
    st.title("📋 Daftar Penerima Bansos (Layak)")

    if st.session_state.dataset is None:
        st.warning("⚠️ Belum ada dataset. Silakan upload dulu di menu **Upload Dataset & Prediksi**.")
    else:
        df = st.session_state.dataset
        penerima = df[df["Keterangan_Layak"] == "Layak"]

        if penerima.empty:
            st.info("Tidak ada warga yang terklasifikasi Layak (mendapat bansos).")
        else:
            st.success(f"✅ Ada {len(penerima)} warga yang Layak menerima bansos")
            st.dataframe(penerima[["Nama", "Status_Kesejahteraan", "Keterangan_Layak"]])

            # Download daftar penerima saja
            output = io.BytesIO()
            penerima.to_excel(output, index=False, engine="openpyxl")
            output.seek(0)
            st.download_button(
                label="📥 Download Daftar Penerima (Excel)",
                data=output,
                file_name="daftar_penerima_bansos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
