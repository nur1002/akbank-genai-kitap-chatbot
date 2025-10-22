import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import sys
from dotenv import load_dotenv
import os

# --- CONFIG: Google Drive file IDs (replace with your own) ---
CSV_DRIVE_ID = "https://drive.google.com/uc?id=157Hr56uDKVgBvptwzy7SN9fxvI7jbz_X"  # mevcut CSV id'niz
NPY_DRIVE_ID = "https://drive.google.com/uc?id=1fakOKy2kERH1DoEmCjOl9GgvAxGFXGgP"    # buraya book_embeddings_FULL.npy dosya id'sini koyun
CSV_LOCAL = "clean_books_v2.csv"
NPY_LOCAL = "book_embeddings_FULL.npy"

# --- Helper: download from Google Drive using gdown (sağlam ve büyük dosyalar için uygun) ---
def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except Exception:
        st.info("Gerekli paketler eksik, gdown yüklenecek...")
        os.system(f"{sys.executable} -m pip install --upgrade gdown")

def download_from_gdrive(file_id: str, dest_path: str):
    if os.path.exists(dest_path):
        return
    ensure_gdown()
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        st.info(f"Dosya indiriliyor: {os.path.basename(dest_path)} (Drive id: {file_id})")
        gdown.download(url, dest_path, quiet=False)
    except Exception as e:
        st.error(f"Dosya indirilirken hata oluştu: {e}")
        raise

# --- VERİ YÜKLEME ---
@st.cache_resource
def load_data():
    try:
        # Drive'dan CSV ve NPY'yi indir (eğer yoksa)
        try:
            download_from_gdrive(CSV_DRIVE_ID, CSV_LOCAL)
        except Exception:
            st.warning("CSV indirme başarısız oldu; yerel dosyanın varlığını kontrol edin.")
        try:
            download_from_gdrive(NPY_DRIVE_ID, NPY_LOCAL)
        except Exception:
            st.warning("Embeddings (.npy) indirme başarısız oldu; yerel dosyanın varlığını kontrol edin.")

        # CSV okuma (encoding fallback)
        try:
            df = pd.read_csv(CSV_LOCAL, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_LOCAL, encoding="windows-1254")

        # embeddings yükleme
        if not os.path.exists(NPY_LOCAL):
            raise FileNotFoundError(f"HATA: '{NPY_LOCAL}' bulunamadı. Lütfen dosyayı doğru yere koyun veya Drive paylaşımını kontrol edin.")
        book_embeddings = np.load(NPY_LOCAL, allow_pickle=True)

        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return df, book_embeddings, model

    except FileNotFoundError as e:
        st.error(f"HATA: Gerekli veri dosyaları bulunamadı: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Veri yüklenirken beklenmeyen bir hata oluştu: {e}")
        return None, None, None

# --- RETRIEVAL FONKSİYONU ---
def get_hybrid_recommendations(query, model, embeddings, df, top_n=5):
    query = query.lower()
    if 'yabancı' in query:
        filtered_df = df[df['book_type'].str.lower() == 'dünya roman']
    elif 'türk' in query:
        filtered_df = df[df['book_type'].str.lower().isin(['türk romanı', 'türk klasikleri'])]
    else:
        potential_filters = pd.Series(False, index=df.index)
        applied_filter = False
        genre_map = {
            'roman': ['dünya roman', 'türk romanı'],
            'fantastik': ['fantastik'],
            'bilim kurgu': ['bilim kurgu'],
            'polisiye': ['polisiye'],
            'korku': ['korku gerilim'],
            'gerilim': ['korku gerilim'],
            'macera': ['macera']
        }
        for keyword, genres in genre_map.items():
            if keyword in query:
                potential_filters = potential_filters | df['book_type'].str.lower().isin(genres)
                applied_filter = True
        filtered_df = df[potential_filters] if applied_filter else df.copy()

    if len(filtered_df) == 0:
        filtered_df = df.copy()

    filtered_indices = filtered_df.index.to_numpy()
    filtered_embeddings = embeddings[filtered_indices]

    query_embedding = model.encode([query], show_progress_bar=False)
    similarities = cosine_similarity(query_embedding, filtered_embeddings).flatten()

    num_results = min(top_n, len(filtered_df))
    top_local_indices = np.argsort(similarities)[-num_results:][::-1]

    top_global_indices = [filtered_indices[i] for i in top_local_indices]

    return df.iloc[top_global_indices]

# --- GENERATION FONKSİYONU ---
def generate_recommendation_text(query, recommendations):
    context = ""
    for index, row in recommendations.iterrows():
        context += f"- Kitap Adı: {row['name']}, Yazar: {row['author']}, Tür: {row['book_type']}\n"
        context += f"  Açıklama: {str(row.get('explanation',''))[:300]}...\n\n"

    prompt = f"""
    Bir kitap öneri uzmanısın. Kullanıcının isteği ve benim bulduğum kitap önerileri aşağıdadır. 
    Bu bilgilere dayanarak, kullanıcıya akıcı ve sohbet havasında bir öneri metni yaz. 
    Neden bu kitapları sevebileceğini açıkla ve en az 2-3 kitaba ismen değin.
    
    Kullanıcının İsteği: "{query}"
    
    Bulunan Kitaplar ve Açıklamaları:
    {context}
    
    Lütfen sadece öneri metnini yaz, başka bir şey ekleme.
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Yapay zeka ile metin üretilirken bir hata oluştu: {e}"

# --- STREAMLIT ARAYÜZÜ ---
load_dotenv()
df, book_embeddings, model_st = load_data()

st.title('📚 RAG Destekli Kitap Öneri Motoru')
st.markdown("Yapay zeka, aramanıza en uygun kitapları bulur ve size özel bir öneri metni hazırlar.")

st.sidebar.header("API Anahtarı")
api_key = st.sidebar.text_input("Google Gemini API Anahtarınızı Girin", type="password")

if api_key:
    try:
        genai.configure(api_key=api_key)
        st.sidebar.success("API Anahtarı başarıyla ayarlandı!")
    except Exception as e:
        st.sidebar.error(f"API Anahtarını ayarlarken hata oluştu: {e}")

user_query = st.text_input('Ne tür bir kitap arıyorsunuz?', placeholder="Örn: Taht Oyunları gibi fantastik bir kitap")

if st.button('Kitap Öner'):
    if not api_key:
        st.warning("Lütfen sol taraftaki menüden Google Gemini API anahtarınızı girin.")
    elif df is not None and model_st is not None and user_query:
        with st.spinner('En uygun kitaplar bulunuyor... (Retrieval)'):
            recommendations = get_hybrid_recommendations(user_query, model_st, book_embeddings, df)

        with st.spinner('Yapay zeka size özel öneri metni hazırlıyor... (Generation)'):
            generated_text = generate_recommendation_text(user_query, recommendations)
            st.success("İşte size özel kitap önerileri!")
            st.markdown(generated_text)

            with st.expander("Yapay zekanın kullandığı kitapları gör"):
                st.dataframe(recommendations[['name', 'author', 'book_type']])
    else:
        st.warning("Lütfen bir kitap adı veya konu girin.")

