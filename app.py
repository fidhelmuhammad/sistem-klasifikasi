import streamlit as st
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Konfigurasi Halaman
# -------------------------
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

st.title("📊 Penerapan Algoritma Naïve Bayes")
st.write("Aplikasi ini digunakan untuk klasifikasi penerima bantuan sosial di Desa Cikembar.")

# -------------------------
# Upload Dataset
# -------------------------
uploaded_file = st.file_uploader(
    "📂 Upload file CSV atau Excel (.csv / .xls / .xlsx)",
    type=["csv", "xls", "xlsx"]
)

if uploaded_file:
    # baca dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # -------------------------
    # Preview Dataset
    # -------------------------
    st.subheader("📄 Preview Dataset")
    st.dataframe(df.head())

    # Info dataset
    st.subheader("ℹ️ Info singkat dataset")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write("Jumlah data:", df.shape[0])
    st.write("Jumlah kolom:", df.shape[1])
    st.write("Jumlah missing value tiap kolom:")
    st.write(df.isnull().sum())

    # -------------------------
    # Pilih Target
    # -------------------------
    target_col = st.selectbox("🎯 Pilih kolom target (label)", df.columns)

    if target_col:
        # cek validitas target
        class_counts = df[target_col].value_counts()
        invalid_classes = class_counts[class_counts < 2]

        if len(invalid_classes) > 0:
            st.error(
                f"Kolom target **{target_col}** tidak valid karena ada kelas dengan jumlah < 2:\n\n{invalid_classes}"
            )
            st.stop()

        # -------------------------
        # Pilih Fitur
        # -------------------------
        feature_cols = st.multiselect(
            "📌 Pilih kolom fitur (predictors)",
            [col for col in df.columns if col != target_col]
        )

        if feature_cols:
            st.write("Fitur terpilih:", feature_cols)

            # siapkan X, y
            X = df[feature_cols]
            y = df[target_col]

            # identifikasi kolom kategorikal & numerik
            cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

            # -------------------------
            # Preprocessing
            # -------------------------
            try:
                onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.4
            except TypeError:
                onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.4

            cat_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", onehot)
            ])
            num_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            preprocessor = ColumnTransformer([
                ("cat", cat_transformer, cat_cols),
                ("num", num_transformer, num_cols)
            ])

            # -------------------------
            # Model Pipeline
            # -------------------------
            model = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", GaussianNB())
            ])

            # -------------------------
            # Split Data
            # -------------------------
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
            except ValueError:
                st.warning(
                    "⚠️ Stratify gagal karena ada kelas dengan jumlah terlalu sedikit. Data dibagi tanpa stratify."
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

            # -------------------------
            # Training
            # -------------------------
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # -------------------------
            # Evaluasi
            # -------------------------
            st.subheader("📊 Hasil Evaluasi Model")
            acc = accuracy_score(y_test, y_pred)
            st.write(f"**Akurasi:** {acc:.2f}")

            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            st.subheader("📌 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
