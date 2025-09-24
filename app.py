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

st.set_page_config(page_title='Klasifikasi Penerima Bantuan Sosial - Desa Cikembar', layout='wide')

# ---------- Helper functions ----------

def generate_sample_data(n=500, random_state=42):
    """Buat dataset sintetis contoh jika user tidak mengupload data."""
    rng = np.random.default_rng(random_state)
    income = rng.normal(1500000, 800000, size=n).clip(200000, 8000000).astype(int)  # penghasilan per bulan
    dependents = rng.integers(0, 7, size=n)  # jumlah tanggungan
    house_condition = rng.choice(['Baik', 'Sedang', 'Buruk'], size=n, p=[0.3,0.5,0.2])
    has_id = rng.choice(['Ya','Tidak'], size=n, p=[0.95,0.05])
    land_owner = rng.choice(['Ya','Tidak'], size=n, p=[0.25,0.75])
    electricity = rng.choice(['Listrik PLN','Listrik Non-PLN','Tidak Ada'], size=n, p=[0.7,0.05,0.25])
    employment = rng.choice(['Pekerja Formal','Pekerja Informal','Tidak Bekerja'], size=n, p=[0.25,0.5,0.25])

    # rules of thumb untuk label (hanya contoh sederhana)
    score = (
        (income < 1000000).astype(int)*2 +
        (dependents >= 3).astype(int)*1 +
        (house_condition == 'Buruk').astype(int)*2 +
        (has_id == 'Tidak').astype(int)*1 +
        (electricity == 'Tidak Ada').astype(int)*1 +
        (employment == 'Tidak Bekerja').astype(int)*1 -
        (land_owner == 'Ya').astype(int)*1
    )
    # threshold sederhana: jika score >=3 => Layak
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
    """Kolom yang diharapkan oleh aplikasi ini."""
    return {
        'numerical': ['penghasilan_per_bulan','jumlah_tanggungan'],
        'categorical': ['kondisi_rumah','memiliki_ktp','pemilik_tanah','sumber_listrik','status_pekerjaan']
    }


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

    # Naive Bayes: menggunakan GaussianNB di pipeline (bisa diterapkan setelah preprocessing)
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('toarray', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x, validate=False)),
                          ('nb', GaussianNB())])
    return clf

# Fungsi kecil untuk mengubah sparse matrix ke array saat diperlukan
from sklearn.preprocessing import FunctionTransformer


# ---------- UI: Sidebar untuk pilihan halaman ----------
page = st.sidebar.radio("Halaman", ['Informasi','Pelatihan Model','Prediksi'])

# Simpan model di session_state agar tidak hilang saat interaksi
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'trained' not in st.session_state:
    st.session_state['trained'] = False
if 'columns' not in st.session_state:
    st.session_state['columns'] = get_default_columns()

# ---------- Halaman 1: Informasi ----------
if page == 'Informasi':
    st.title('Klasifikasi Penerima Bantuan Sosial di Desa Cikembar')
    st.markdown("""
    **Deskripsi sistem**

    Sistem ini bertujuan membantu pemerintah desa dalam mengklasifikasikan **keluarga/rumah tangga** mana yang **berhak** menerima bantuan sosial dan mana yang **belum layak** berdasarkan data demografis dan ekonomi sederhana.

    **Tujuan**
    - Menyediakan alat bantu keputusan yang cepat berbasis data.
    - Mengurangi subjektivitas dalam penentuan penerima.
    - Memprioritaskan keluarga yang paling membutuhkan.

    **Manfaat**
    - Mempercepat proses verifikasi calon penerima bantuan.
    - Meningkatkan transparansi penyaluran bantuan.
    - Membantu alokasi anggaran yang lebih tepat sasaran.

    **Catatan penting**: Model ini adalah alat bantu — keputusan akhir tetap berada di tangan petugas desa dan kebijakan lokal. Evaluasi lapangan dan verifikasi administrasi tetap diperlukan.
    """)

    st.header('Kolom data yang direkomendasikan')
    st.write('- penghasilan_per_bulan (angka, rupiah)')
    st.write('- jumlah_tanggungan (angka)')
    st.write('- kondisi_rumah (Baik/Sedang/Buruk)')
    st.write('- memiliki_ktp (Ya/Tidak)')
    st.write('- pemilik_tanah (Ya/Tidak)')
    st.write('- sumber_listrik (Listrik PLN/Listrik Non-PLN/Tidak Ada)')

    st.info('Anda dapat mengupload file CSV pada halaman Pelatihan Model atau Prediksi. Jika tidak, sistem akan menggunakan dataset contoh.')

# ---------- Halaman 2: Pelatihan Model ----------
elif page == 'Pelatihan Model':
    st.header('Pelatihan Model Naive Bayes')
    st.markdown('Upload dataset CSV atau gunakan dataset contoh. Pastikan terdapat kolom label ("label") berisi "Layak" atau "Tidak Layak".')

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

    # Pilih kolom target
    if 'label' not in df.columns:
        st.error('Kolom "label" tidak ditemukan di dataset. Tambahkan kolom "label" dengan nilai "Layak" atau "Tidak Layak".')
    else:
        # Izinkan user memilih kolom-kolom (tapi default ke yang kami rekomendasikan)
        default_cols = get_default_columns()
        num_cols = st.multiselect('Pilih kolom numerik', options=[c for c in df.columns if df[c].dtype.kind in 'biufc'], default=default_cols['numerical'])
        cat_cols = st.multiselect('Pilih kolom kategorikal', options=[c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category'], default=default_cols['categorical'])

        # Update columns di session
        st.session_state['columns'] = {'numerical': num_cols, 'categorical': cat_cols}

        test_size = st.slider('Proporsi data untuk test set', min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input('Random state (bilangan bulat)', value=42, step=1)

        if st.button('Latih model Naive Bayes'):
            with st.spinner('Melatih model...'):
                X = df[num_cols + cat_cols]
                y = df['label']

                # encode target
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_enc = le.fit_transform(y)

                pipeline = build_pipeline(num_cols, cat_cols)

                X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=int(random_state), stratify=y_enc if len(np.unique(y_enc))>1 else None)

                pipeline.fit(X_train, y_train)

                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

                st.session_state['model'] = {'pipeline': pipeline, 'label_encoder': le}
                st.session_state['trained'] = True

                st.success(f'Model selesai dilatih — Akurasi pada data test: {acc:.3f}')

                st.subheader('Confusion Matrix')
                fig, ax = plt.subplots()
                im = ax.imshow(cm, interpolation='nearest')
                ax.figure.colorbar(im, ax=ax)
                ax.set(xticks=np.arange(len(le.classes_)), yticks=np.arange(len(le.classes_)),
                       xticklabels=le.classes_, yticklabels=le.classes_,
                       ylabel='True label', xlabel='Predicted label')
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center')
                st.pyplot(fig)

                st.subheader('Laporan Klasifikasi')
                st.text(report)

                # Simpan model ke bytes untuk didownload
                buf = io.BytesIO()
                joblib.dump({'pipeline': pipeline, 'label_encoder': le}, buf)
                buf.seek(0)
                st.download_button('Download model (joblib)', data=buf, file_name='model_naive_bayes_cikembar.joblib')

        # Jika sudah dilatih sebelumnya, tampilkan ringkasan
        if st.session_state['trained'] and st.session_state['model'] is not None:
            st.info('Terdapat model yang sudah dilatih dalam sesi ini. Anda bisa langsung ke halaman Prediksi.')

# ---------- Halaman 3: Prediksi ----------
elif page == 'Prediksi':
    st.header('Prediksi Penerima Bantuan Sosial')

    if not st.session_state.get('trained', False) or st.session_state.get('model') is None:
        st.warning('Model belum dilatih. Silakan latih model di halaman "Pelatihan Model" terlebih dahulu, atau upload model joblib.')

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader('Input Manual')
        # Ambil kolom default dari session
        cols = st.session_state.get('columns', get_default_columns())
        # Jika tidak ada, pakai default
        num_cols = cols.get('numerical', get_default_columns()['numerical'])
        cat_cols = cols.get('categorical', get_default_columns()['categorical'])

        user_input = {}
        # numerical inputs
        for c in num_cols:
            if c == 'penghasilan_per_bulan':
                user_input[c] = st.number_input('Penghasilan per bulan (Rp)', min_value=0, value=1000000, step=50000)
            elif c == 'jumlah_tanggungan':
                user_input[c] = st.number_input('Jumlah tanggungan', min_value=0, value=2, step=1)
            else:
                user_input[c] = st.number_input(c, value=0)

        # categorical inputs (memberi pilihan dari contoh umum)
        for c in cat_cols:
            if c == 'kondisi_rumah':
                user_input[c] = st.selectbox('Kondisi rumah', options=['Baik','Sedang','Buruk'])
            elif c == 'memiliki_ktp':
                user_input[c] = st.selectbox('Memiliki KTP?', options=['Ya','Tidak'])
            elif c == 'pemilik_tanah':
                user_input[c] = st.selectbox('Pemilik tanah?', options=['Ya','Tidak'])
            elif c == 'sumber_listrik':
                user_input[c] = st.selectbox('Sumber Listrik', options=['Listrik PLN','Listrik Non-PLN','Tidak Ada'])
            elif c == 'status_pekerjaan':
                user_input[c] = st.selectbox('Status pekerjaan', options=['Pekerja Formal','Pekerja Informal','Tidak Bekerja'])
            else:
                user_input[c] = st.text_input(c)

        if st.button('Prediksi (kasus tunggal)'):
            if st.session_state.get('model') is None:
                st.error('Tidak ada model tersedia di sesi ini. Upload model joblib atau latih model di halaman Pelatihan Model.')
            else:
                model_dict = st.session_state['model']
                pipeline = model_dict['pipeline']
                le = model_dict['label_encoder']
                X_new = pd.DataFrame([user_input])
                try:
                    y_proba = pipeline.predict_proba(X_new)
                except Exception:
                    # be resilient: some naive bayes pipelines may not implement predict_proba properly if shape mismatch
                    y_pred = pipeline.predict(X_new)
                    y_proba = None
                y_pred = pipeline.predict(X_new)
                label_pred = le.inverse_transform(y_pred)

                st.markdown('**Hasil Prediksi**')
                st.write('Label prediksi:', label_pred[0])
                if y_proba is not None:
                    probs = {le.classes_[i]: float(y_proba[0][i]) for i in range(len(le.classes_))}
                    st.write('Probabilitas:', probs)

                st.success('Prediksi selesai. Ingat: ini hanya alat bantu keputusan.')

    with col2:
        st.subheader('Prediksi Batch (CSV)')
        uploaded_pred = st.file_uploader('Upload file CSV berisi data tanpa kolom label untuk diprediksi', type=['csv'], key='pred_upload')
        if uploaded_pred is not None:
            try:
                df_pred = pd.read_csv(uploaded_pred)
                st.write('Pratinjau data untuk diprediksi:')
                st.dataframe(df_pred.head())
            except Exception as e:
                st.error(f'Gagal membaca CSV: {e}')
                df_pred = None

            if df_pred is not None:
                if st.button('Jalankan Prediksi batch'):
                    if st.session_state.get('model') is None:
                        st.error('Tidak ada model tersedia di sesi ini. Upload model joblib atau latih model di halaman Pelatihan Model.')
                    else:
                        model_dict = st.session_state['model']
                        pipeline = model_dict['pipeline']
                        le = model_dict['label_encoder']
                        try:
                            preds = pipeline.predict(df_pred)
                            try:
                                probs = pipeline.predict_proba(df_pred)
                            except Exception:
                                probs = None
                            labels = le.inverse_transform(preds)
                            result = df_pred.copy()
                            result['prediksi_label'] = labels
                            if probs is not None:
                                # tambahkan kolom probabilitas untuk tiap kelas
                                for i, cls in enumerate(le.classes_):
                                    result[f'proba_{cls}'] = probs[:, i]
                            st.write('Hasil prediksi:')
                            st.dataframe(result.head(50))

                            # download
                            csv = result.to_csv(index=False).encode('utf-8')
                            st.download_button('Download hasil prediksi (CSV)', data=csv, file_name='hasil_prediksi.csv')
                        except Exception as e:
                            st.error(f'Kesalahan saat prediksi: {e}')

        st.markdown('---')
        st.subheader('Upload / Load model')
        uploaded_model = st.file_uploader('Upload file model (.joblib) jika ingin menggunakan model tersimpan', type=['joblib','pkl'], key='model_upload')
        if uploaded_model is not None:
            try:
                model_loaded = joblib.load(uploaded_model)
                # model_loaded diharapkan dict dengan keys 'pipeline' dan 'label_encoder'
                if isinstance(model_loaded, dict) and 'pipeline' in model_loaded and 'label_encoder' in model_loaded:
                    st.session_state['model'] = model_loaded
                    st.session_state['trained'] = True
                    st.success('Model berhasil dimuat ke sesi.')
                else:
                    # jika file langsung menyimpan pipeline
                    st.session_state['model'] = {'pipeline': model_loaded, 'label_encoder': None}
                    st.session_state['trained'] = True
                    st.warning('Model dimuat, tetapi file tidak memiliki label encoder. Pastikan label encoder ada agar prediksi berlabel berjalan. Anda bisa melatih ulang model di halaman Pelatihan Model untuk mendapatkan label encoder.')
            except Exception as e:
                st.error(f'Gagal memuat model: {e}')

# ---------- Footer / catatan ----------
st.sidebar.markdown('---')
st.sidebar.caption('Aplikasi contoh: Naive Bayes untuk membantu klasifikasi penerima bantuan. Keputusan akhir tetap pada petugas desa.')


# Jika file dijalankan langsung (bukan sebagai modul), tidak perlu apa-apa lagi — streamlit menjalankannya.
