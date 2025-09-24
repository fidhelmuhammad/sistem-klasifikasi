# Streamlit app: Klasifikasi Penerima Bantuan Sosial
# Judul: Klasifikasi Penerima Bantuan Sosial di Desa Cikembar
# Menggunakan algoritma: Naive Bayes
# Tiga halaman: Informasi | Pelatihan Model | Prediksi

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib
import io
import tempfile

st.set_page_config(page_title='Klasifikasi Penerima Bantuan Sosial - Desa Cikembar', layout='wide')

# ---------- Helper functions ----------

def generate_sample_data(n=500, random_state=42):
    rng = np.random.default_rng(random_state)
    income = rng.normal(1500000, 800000, size=n).clip(200000, 8000000).astype(int)
    dependents = rng.integers(0, 7, size=n)
    house_condition = rng.choice(['Baik', 'Sedang', 'Buruk'], size=n, p=[0.3,0.5,0.2])
    has_id = rng.choice(['Ya','Tidak'], size=n, p=[0.95,0.05])
    land_owner = rng.choice(['Ya','Tidak'], size=n, p=[0.25,0.75])
    electricity = rng.choice(['Listrik PLN','Listrik Non-PLN','Tidak Ada'], size=n, p=[0.7,0.05,0.25])
    employment = rng.choice(['Pekerja Formal','Pekerja Informal','Tidak Bekerja'], size=n, p=[0.25,0.5,0.25])

    score = (
        (income < 1000000).astype(int)*2 +
        (dependents >= 3).astype(int)*1 +
        (house_condition == 'Buruk').astype(int)*2 +
        (has_id == 'Tidak').astype(int)*1 +
        (electricity == 'Tidak Ada').astype(int)*1 +
        (employment == 'Tidak Bekerja').astype(int)*1 -
        (land_owner == 'Ya').astype(int)*1
    )
    label = np.where(score >= 3, 'Layak', 'Tidak Layak')

    df = pd.DataFrame({
        'penghasilan_per_bulan': income,
        'jumlah_tanggungan': dependents,
        'kondisi_rumah': house_condition,
        'memiliki_ktp': has_id,
        'pemilik_tanah': land_owner,
        'sumber_listrik': electricity,
        'status_pekerjaan': employment,
        'label': label
    })
    return df


def get_default_columns():
    return {
        'numerical': ['penghasilan_per_bulan','jumlah_tanggungan'],
        'categorical': ['kondisi_rumah','memiliki_ktp','pemilik_tanah','sumber_listrik','status_pekerjaan']
    }

from sklearn.preprocessing import FunctionTransformer

def build_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('toarray', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x, validate=False)),
                          ('nb', GaussianNB())])
    return clf

page = st.sidebar.radio("Halaman", ['Informasi','Pelatihan Model','Prediksi'])

if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'trained' not in st.session_state:
    st.session_state['trained'] = False
if 'columns' not in st.session_state:
    st.session_state['columns'] = get_default_columns()

if page == 'Informasi':
    st.title('Klasifikasi Penerima Bantuan Sosial di Desa Cikembar')
    st.markdown("""
    Sistem ini bertujuan membantu pemerintah desa dalam mengklasifikasikan keluarga/rumah tangga mana yang berhak menerima bantuan sosial dan mana yang belum layak berdasarkan data demografis dan ekonomi.
    """)

elif page == 'Pelatihan Model':
    st.header('Pelatihan Model Naive Bayes')
    uploaded_file = st.file_uploader('Upload CSV dataset (opsional)', type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success('Dataset berhasil diupload.')
        except Exception as e:
            st.error(f'Gagal membaca file CSV: {e}')
            df = None
    else:
        df = None

    if df is None:
        st.warning('Menggunakan dataset contoh sintetis karena Anda belum mengupload file.')
        df = generate_sample_data(n=500)

    st.subheader('Pratinjau data')
    st.dataframe(df.head(10))

    if 'label' not in df.columns:
        st.error('Kolom "label" tidak ditemukan.')
    else:
        default_cols = get_default_columns()
        num_cols = st.multiselect('Pilih kolom numerik', options=[c for c in df.columns if df[c].dtype.kind in 'biufc'], default=default_cols['numerical'])
        cat_cols = st.multiselect('Pilih kolom kategorikal', options=[c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category'], default=default_cols['categorical'])

        st.session_state['columns'] = {'numerical': num_cols, 'categorical': cat_cols}

        test_size = st.slider('Proporsi data untuk test set', min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input('Random state', value=42, step=1)

        if st.button('Latih model Naive Bayes'):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_enc = le.fit_transform(df['label'])

            pipeline = build_pipeline(num_cols, cat_cols)
            X_train, X_test, y_train, y_test = train_test_split(df[num_cols + cat_cols], y_enc, test_size=test_size, random_state=int(random_state), stratify=y_enc if len(np.unique(y_enc))>1 else None)

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

            st.session_state['model'] = {'pipeline': pipeline, 'label_encoder': le}
            st.session_state['trained'] = True

            st.success(f'Akurasi: {acc:.3f}')
            st.text(report)

            # Simpan model ke file sementara agar bisa diunduh
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                joblib.dump({'pipeline': pipeline, 'label_encoder': le}, tmp.name)
                tmp.seek(0)
                with open(tmp.name, 'rb') as f:
                    st.download_button('Download model (joblib)', data=f, file_name='model_naive_bayes_cikembar.joblib')

elif page == 'Prediksi':
    st.header('Prediksi Penerima Bantuan Sosial')
    st.write("Gunakan model yang sudah dilatih atau upload model.")

st.sidebar.markdown('---')
st.sidebar.caption('Aplikasi contoh Naive Bayes untuk klasifikasi bantuan sosial.')
