# akbank-genai-kitap-chatbot


📚 RAG Destekli Kitap Öneri Motoru
🎯 Projenin Amacı

Bu proje, kullanıcıların ilgi alanlarına ve tercihlerine göre kitap önerileri sunan RAG (Retrieval-Augmented Generation) tabanlı bir yapay zeka sistemidir.
Kullanıcıdan alınan doğal dildeki kitap isteği (“Taht Oyunları gibi fantastik bir kitap” gibi) önce vektör tabanlı arama ile en uygun kitap açıklamalarıyla eşleştirilir, ardından Gemini 2.0 Flash modeli bu sonuçları kullanarak kişiye özel bir öneri metni üretir.

📊 Veri Seti Hakkında

Proje, kitap açıklamaları, yazar bilgileri, tür bilgileri ve yayınevlerini içeren geniş bir veri setine dayanır.
Veri seti temizlenmiş ve metin bazlı analiz için uygun hale getirilmiştir.
Aşağıdaki sütunlar kullanılmıştır:

name → Kitap adı

author → Yazar adı

publisher → Yayınevi

book_type → Kitap türü

explanation → Kitabın kısa açıklaması

Veri seti büyük boyutlu olduğu için doğrudan GitHub’a yüklenmemiş, Google Drive üzerinde paylaşılmıştır.

📎 Veri Kaynağı Linkleri:

clean_books_v2.csv: Drive Linki

book_embeddings_FULL.npy: Drive Linki (örnek)

🧠 Kullanılan Yöntemler

1. Retrieval Aşaması (Bilgi Getirme)

Kullanıcı sorgusu SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2) modeliyle vektörel forma dönüştürülür.

Cosine Similarity metriğiyle en benzer kitap açıklamaları bulunur.

2. Generation Aşaması (Metin Üretimi)

Bulunan kitapların kısa açıklamaları Gemini modeline context olarak verilir.

Gemini 2.0 Flash modeli, bu bilgilerle kullanıcıya doğal, akıcı bir öneri metni üretir.

Bu yapı, klasik “kitap arama” sistemlerinden farklı olarak hem bilgi tabanlı hem de yaratıcı metin tabanlı bir öneri yaklaşımı sağlar.

⚙️ Kodun Çalışma Kılavuzu
1. Gerekli Kurulumlar
# Sanal ortam oluşturma
python -m venv venv
venv\Scripts\activate   # (Windows)
source venv/bin/activate   # (Mac/Linux)

# Bağımlılıkların kurulumu
pip install -r requirements.txt

2. Ortam Değişkeni (.env)

.env dosyası oluşturun ve içine API anahtarınızı yazın:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY

3. Uygulamayı Çalıştırma
streamlit run app.py

4. Önemli Not

Veri seti dosyaları (.csv ve .npy) GitHub’a yüklenmemeli, .gitignore dosyasına eklenmelidir.
Kod bu dosyaları doğrudan Google Drive’dan çeker.

🏗️ Çözüm Mimarisi

Teknolojiler:

Python

Streamlit

Sentence Transformers

scikit-learn

Google Gemini 2.0 Flash (Generative AI API)

Mimari Yapı:

Input (User Query) → Kullanıcıdan doğal dil girdisi alınır.

Encoder → Cümleler SentenceTransformer ile vektörleştirilir.

Retriever → Benzer kitaplar cosine similarity ile bulunur.

Generator (Gemini) → Bulunan kitaplara dayanarak metin üretir.

Frontend (Streamlit) → Kullanıcıya öneri metnini ve kitap listesini gösterir.

Basitleştirilmiş Akış Şeması:

User Query
    ↓
SentenceTransformer (Encoding)
    ↓
Cosine Similarity Search (Retrieval)
    ↓
Gemini 2.0 Flash (Generation)
    ↓
Streamlit UI (Display)

🌐 Web Arayüzü & Product Kılavuzu
🔗 Canlı Uygulama Linki

👉 Streamlit Deploy Linki

👀 Arayüz Özellikleri

Kullanıcıdan doğal dilde kitap isteği alır

Uygun kitapları bulup tablo olarak gösterir

Gemini modeliyle kişisel öneri metni üretir

Kitap türlerine göre otomatik filtreleme yapar (fantastik, roman, polisiye vb.)



📈 Sonuç

Bu proje, RAG (Retrieval-Augmented Generation) mimarisinin kitap öneri sistemlerinde nasıl uygulanabileceğini gösteren bir örnektir.
Kullanıcı odaklı, açıklayıcı ve üretken bir öneri sistemi geliştirilmiştir.
Model, kullanıcı ifadelerinden bağlam çıkararak hem bilgiye dayalı hem de kişisel öneriler sunar.

📎 Kaynaklar

Sentence Transformers

Streamlit Docs

Google Gemini API
