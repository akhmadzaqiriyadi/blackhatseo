# 🔍 BlackHat SEO Detector

Alat berbasis Machine Learning untuk mendeteksi praktik Black Hat SEO pada website, dengan fokus khusus pada domain pemerintah Indonesia (.go.id) dan pendidikan (.ac.id) yang mungkin telah dikompromikan dengan injeksi konten judi/spam.

## ✨ Fitur Utama

- **Deteksi Berbasis ML**: Model ensemble yang menggabungkan SVM (teks) + Random Forest (fitur)
- **Analisis Komprehensif**: Mendeteksi keyword stuffing, cloaking, hidden content, link mencurigakan
- **Whitelist Domain Terpercaya**: Mengurangi false positive untuk situs legitim (Shopee, Kompas, dll)
- **Override Domain Resmi**: Penanganan khusus untuk domain .go.id dan .ac.id
- **REST API**: API berbasis Flask untuk integrasi
- **CLI Tool**: Command-line interface untuk batch processing

## 🚀 Memulai

### Instalasi

```bash
# Clone repository
git clone https://github.com/akhmadzaqiriyadi/blackhatseo.git
cd blackhatseo

# Buat virtual environment
python3 -m venv venv
source venv/bin/activate  # Di Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Penggunaan Dasar

#### Command Line

```bash
# Prediksi single/multiple URL
python3 -m src.main predict --urls data/test_urls.txt

# Dapatkan penjelasan detail
python3 -m src.main explain --url https://example.com

# Training model dengan data kustom
python3 -m src.main train --train-urls data/train_data.csv
```

#### REST API

```bash
# Jalankan API server
python3 app.py

# Health check
curl http://localhost:5001/api/health

# Prediksi URL
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com"]}'

# Dapatkan penjelasan
curl -X POST http://localhost:5001/api/explain \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

## 📊 Contoh Hasil Pengujian

### Hasil Test Situs di Yogyakarta

| URL | Tipe | Status | Confidence |
|-----|------|--------|------------|
| jogjakota.go.id | Pemerintah | ✅ Bersih | 5.48% |
| bantulkab.go.id | Pemerintah | ✅ Bersih | 5.53% |
| ugm.ac.id | Universitas | ✅ Bersih | 4.08% |
| uny.ac.id | Universitas | ✅ Bersih | 4.96% |
| uad.ac.id | Universitas | ✅ Bersih | 14.64% |

### Deteksi Situs Terkompromisi

| URL | Tipe | Status | Confidence |
|-----|------|--------|------------|
| bkpsdm.purbalinggakab.go.id | Pemerintah | ⚠️ Black Hat | 65% |
| sinora.umpwr.ac.id | Universitas | ⚠️ Black Hat | 65% |
| sumbarprov.go.id | Pemerintah | ⚠️ Black Hat | 65% |

## 📁 Struktur Project

```
blackhatseo/
├── src/
│   ├── main.py          # Entry point CLI
│   ├── app.py           # Flask API
│   ├── detector.py      # Logika deteksi utama
│   ├── scraper.py       # Web scraping & analisis
│   ├── analyzer.py      # Analisis teks (TF-IDF, BERT)
│   ├── features.py      # Feature engineering
│   ├── builder.py       # Pembuatan dataset
│   ├── config.py        # Manajemen konfigurasi
│   └── utils.py         # Fungsi utilitas
├── data/
│   ├── train_data_balanced_corrected.csv  # Dataset training
│   ├── blackhat_queries.txt               # Kata kunci judi
│   ├── clean_queries.txt                  # Query situs bersih
│   └── test_urls.txt                      # URL untuk testing
├── models/
│   ├── model.pkl        # Model terlatih
│   └── vectorizer.pkl   # TF-IDF vectorizer
├── logs/                # Log aplikasi
├── config/              # File konfigurasi
├── app.py               # Entry point API
├── startup.sh           # Script startup
└── requirements.txt     # Dependencies
```

## 🎯 Fitur Deteksi

Model menganalisis:

1. **Analisis Kata Kunci**
   - Kata kunci judi (judi, slot, togel, casino, dll)
   - Kata kunci spam (viagra, obat kuat, crypto investment, dll)

2. **Analisis Meta Tag**
   - Title, description, keywords meta tags
   - Open Graph dan Twitter cards

3. **Analisis Konten**
   - Deteksi keyword stuffing (>3% density)
   - Hidden content (display:none, font-size:0)
   - Deteksi thin content (<300 kata)

4. **Analisis Link**
   - Link eksternal mencurigakan
   - Link ke domain judi yang dikenal

5. **Analisis Teknis**
   - Deteksi cloaking (konten berbeda untuk bot vs user)
   - Redirect mencurigakan
   - Pola JavaScript berbahaya

## ⚙️ Konfigurasi

Edit `config/config.json`:

```json
{
  "use_selenium": false,
  "use_bert": false,
  "verbose": false,
  "host": "0.0.0.0",
  "port": 5001
}
```

## 🔧 API Endpoints

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/api/health` | GET | Health check |
| `/api/predict` | POST | Prediksi URL |
| `/api/explain` | POST | Penjelasan detail |

### Contoh Request/Response

**POST /api/predict**
```json
// Request
{"urls": ["https://example.com", "https://test.go.id"]}

// Response
{
  "status": "success",
  "results": [
    {
      "url": "https://example.com",
      "prediction": 0,
      "probability": 0.05,
      "metrics": {...},
      "reasons": []
    }
  ]
}
```

## 📈 Performa Model

- **Data Training**: 1,607 URL berlabel
- **Akurasi Ensemble**: ~83%
- **Fitur**: TF-IDF teks + 14 engineered features

## 🤝 Kontribusi

1. Fork repository
2. Buat feature branch (`git checkout -b feature/fitur-baru`)
3. Commit perubahan (`git commit -m 'Tambah fitur baru'`)
4. Push ke branch (`git push origin feature/fitur-baru`)
5. Buka Pull Request

## 📝 Lisensi

MIT License - lihat file LICENSE untuk detail.

## 👤 Penulis

**Akhmad Zaqi Riyadi**
- GitHub: [@akhmadzaqiriyadi](https://github.com/akhmadzaqiriyadi)

## 🙏 Terima Kasih

- Institusi pemerintah dan pendidikan Indonesia untuk data pengujian
- Komunitas scikit-learn, Flask, BeautifulSoup
