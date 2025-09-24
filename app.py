# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Naive Bayes Bantuan Sosial - Desa Cikembar", layout="wide")

# ---------- session state init ----------
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = []
if "target" not in st.session_state:
    st.session_state.target = None
if "feature_meta" not in st.session_state:
    st.session_state.feature_meta = {}

# ---------- sidebar ----------
menu = st.sidebar.radio("Navigasi", ["🏠 Beranda", "📊 Pelatihan Model", "🔮 Prediksi Baru"])

# ---------- BERANDA ----------
if menu == "🏠 Beranda":
    st.title("📊 Penerapan Algoritma Naïve Bayes")
    st.subheader("Klasifikasi Penerima Bantuan Sosial di Desa Cikembar")
    st.markdown("""
    Sistem ini melatih model Naïve Bayes untuk memprediksi **layak / tidak layak** penerima bantuan.
    - Upload dataset (CSV / Excel)
    - Pilih kolom target & fitur
    - Klik **Latih Model**
    - Cek hasil evaluasi lalu gunakan menu *Prediksi Baru*
    """)

# ---------- PELATIHAN MODEL ----------
elif menu == "📊 Pelatihan Model":
    st.header("📊 Pelatihan Model Naïve Bayes")
    uploaded_file = st.file_uploader("Upload file CSV atau Excel (.csv / .xls / .xlsx)", type=["csv", "xls", "xlsx"])

    if not uploaded_file:
        st.info("Silakan upload file dataset terlebih dahulu.")
    else:
        # baca file dengan handling error
        try:
            name = uploaded_file.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                # requires openpyxl
                df = pd.read_excel(uploaded_file, engine="openpyxl")
        except ImportError as e:
            st.error("Paket `openpyxl` belum terpasang. Tambahkan `openpyxl` ke requirements.txt dan re-deploy.")
            st.stop()
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        st.write("### Preview Dataset")
        st.dataframe(df.head())

        st.write("### Info singkat dataset")
        with st.expander("Tipe kolom & nilai kosong"):
            st.write(df.dtypes.astype(str))
            st.write("Jumlah nilai kosong per kolom:")
            st.write(df.isna().sum())

        # pilih target & fitur
        target = st.selectbox("Pilih kolom target (label)", options=df.columns)
        feature_options = [c for c in df.columns if c != target]
        features = st.multiselect("Pilih kolom fitur (predictors)", options=feature_options, default=feature_options)

        if not features:
            st.info("Pilih setidaknya 1 fitur untuk melatih model.")
        else:
            st.write(f"Fitur terpilih: {features}")
            # tombol latih model agar tidak auto re-run tiap change
            if st.button("Latih Model"):
                X = df[features].copy()
                y = df[target].copy()

                # deteksi tipe fitur
                numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                categorical_features = [c for c in X.columns if c not in numeric_features]

                transformers = []
                if numeric_features:
                    numeric_transformer = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ])
                    transformers.append(("num", numeric_transformer, numeric_features))

                if categorical_features:
                    categorical_transformer = Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
                    ])
                    transformers.append(("cat", categorical_transformer, categorical_features))

                if not transformers:
                    st.error("Tidak ada fitur numerik atau kategorikal yang valid untuk dilatih.")
                    st.stop()

                preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
                pipeline = Pipeline([("preprocessor", preprocessor), ("clf", GaussianNB())])

                # split data (jika memungkinkan stratify supaya distribusi target terjaga)
                try:
                    strat = y if y.nunique() > 1 else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=strat
                    )
                except Exception as e:
                    st.error(f"Gagal membagi data: {e}")
                    st.stop()

                # training
                try:
                    pipeline.fit(X_train, y_train)
                except Exception as e:
                    st.error(f"Gagal melatih model: {e}")
                    st.stop()

                # simpan model dan metadata fitur
                st.session_state.model = pipeline
                st.session_state.features = features
                feat_meta = {}
                for col in features:
                    if col in numeric_features:
                        feat_meta[col] = {
                            "type": "numeric",
                            "min": float(X[col].min(skipna=True)) if X[col].notna().any() else 0.0,
                            "max": float(X[col].max(skipna=True)) if X[col].notna().any() else 0.0,
                            "mean": float(X[col].mean(skipna=True)) if X[col].notna().any() else 0.0
                        }
                    else:
                        vals = X[col].dropna().unique().tolist()
                        # convert to string so selectbox tidak crash on mixed types
                        vals = [str(v) for v in vals]
                        feat_meta[col] = {"type": "categorical", "values": vals}
                st.session_state.feature_meta = feat_meta
                st.session_state.target = target

                # evaluasi
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Model berhasil dilatih — Akurasi (test set): {acc:.3f}")

                st.write("Confusion Matrix:")
                labels = pipeline.named_steps["clf"].classes_
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                fig, ax = plt.subplots()
                im = ax.imshow(cm, interpolation="nearest")
                ax.set_xticks(np.arange(len(labels)))
                ax.set_yticks(np.arange(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                # beri angka pada sel
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, cm[i, j], ha="center", va="center")
                st.pyplot(fig)

                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))

# ---------- PREDIKSI BARU ----------
elif menu == "🔮 Prediksi Baru":
    st.header("🔮 Prediksi Data Baru")
    if st.session_state.model is None:
        st.warning("Model belum dilatih. Silakan latih model pada menu 'Pelatihan Model' terlebih dahulu.")
    else:
        st.write("Masukkan nilai untuk tiap fitur (tipe input disesuaikan otomatis).")
        input_dict = {}
        for col, meta in st.session_state.feature_meta.items():
            if meta["type"] == "numeric":
                default = meta.get("mean", 0.0)
                val = st.number_input(col, value=float(default))
                input_dict[col] = val
            else:
                options = meta.get("values", [])
                if not options:
                    # fallback ke textbox
                    val = st.text_input(col, value="")
                else:
                    val = st.selectbox(col, options)
                input_dict[col] = val

        if st.button("Prediksi"):
            Xnew = pd.DataFrame([input_dict], columns=st.session_state.features)
            try:
                pred = st.session_state.model.predict(Xnew)[0]
                st.success(f"Hasil Prediksi: **{pred}**")
                # tunjukkan probabilitas kalau tersedia
                if hasattr(st.session_state.model.named_steps["clf"], "predict_proba"):
                    proba = st.session_state.model.predict_proba(Xnew)[0]
                    classes = st.session_state.model.named_steps["clf"].classes_
                    proba_dict = {str(c): float(p) for c, p in zip(classes, proba)}
                    st.write("Probabilitas per kelas:")
                    st.json(proba_dict)
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")
