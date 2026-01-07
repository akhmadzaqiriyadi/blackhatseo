# 🔍 BlackHat SEO Detector

An intelligent machine learning-based tool to detect Black Hat SEO practices on websites, with special focus on Indonesian government (.go.id) and educational (.ac.id) domains that may have been compromised with gambling/spam content injection.

## ✨ Features

- **ML-Powered Detection**: Ensemble model combining SVM (text) + Random Forest (features)
- **Comprehensive Analysis**: Detects keyword stuffing, cloaking, hidden content, suspicious links
- **Trusted Domain Whitelist**: Reduces false positives for known legitimate sites (Shopee, Kompas, etc.)
- **Official Domain Override**: Special handling for .go.id and .ac.id domains
- **REST API**: Flask-based API for integration
- **CLI Tool**: Command-line interface for batch processing

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/akhmadzaqiriyadi/blackhatseo.git
cd blackhatseo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Command Line

```bash
# Predict single/multiple URLs
python3 -m src.main predict --urls data/test_urls.txt

# Get detailed explanation
python3 -m src.main explain --url https://example.com

# Train model with custom data
python3 -m src.main train --train-urls data/train_data.csv
```

#### REST API

```bash
# Start API server
python3 app.py

# Health check
curl http://localhost:5001/api/health

# Predict URLs
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com"]}'

# Get explanation
curl -X POST http://localhost:5001/api/explain \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

## 📊 Example Results

### Test Results (Yogyakarta Sites)

| URL | Type | Status | Confidence |
|-----|------|--------|------------|
| jogjakota.go.id | Government | ✅ Clean | 5.48% |
| bantulkab.go.id | Government | ✅ Clean | 5.53% |
| ugm.ac.id | University | ✅ Clean | 4.08% |
| uny.ac.id | University | ✅ Clean | 4.96% |
| uad.ac.id | University | ✅ Clean | 14.64% |

### Compromised Sites Detection

| URL | Type | Status | Confidence |
|-----|------|--------|------------|
| bkpsdm.purbalinggakab.go.id | Government | ⚠️ Black Hat | 65% |
| sinora.umpwr.ac.id | University | ⚠️ Black Hat | 65% |
| sumbarprov.go.id | Government | ⚠️ Black Hat | 65% |

## 📁 Project Structure

```
blackhatseo/
├── src/
│   ├── main.py          # CLI entry point
│   ├── app.py           # Flask API (imported by root app.py)
│   ├── detector.py      # Main detection logic
│   ├── scraper.py       # Web scraping & analysis
│   ├── analyzer.py      # Text analysis (TF-IDF, BERT)
│   ├── features.py      # Feature engineering
│   ├── builder.py       # Dataset building
│   ├── config.py        # Configuration management
│   └── utils.py         # Utility functions
├── data/
│   ├── train_data_balanced_corrected.csv  # Training dataset
│   ├── blackhat_queries.txt               # Gambling keywords
│   ├── clean_queries.txt                  # Clean site queries
│   └── test_urls.txt                      # Test URLs
├── models/
│   ├── model.pkl        # Trained model
│   └── vectorizer.pkl   # TF-IDF vectorizer
├── logs/                # Application logs
├── config/              # Configuration files
├── app.py               # API entry point
├── startup.sh           # Startup script
└── requirements.txt     # Dependencies
```

## 🎯 Detection Features

The model analyzes:

1. **Keyword Analysis**
   - Gambling keywords (judi, slot, togel, casino, etc.)
   - Spam keywords (viagra, obat kuat, crypto investment, etc.)

2. **Meta Tag Analysis**
   - Title, description, keywords meta tags
   - Open Graph and Twitter cards

3. **Content Analysis**
   - Keyword stuffing detection (>3% density)
   - Hidden content (display:none, font-size:0)
   - Thin content detection (<300 words)

4. **Link Analysis**
   - Suspicious external links
   - Links to known gambling domains

5. **Technical Analysis**
   - Cloaking detection (different content for bots vs users)
   - Suspicious redirects
   - Malicious JavaScript patterns

## ⚙️ Configuration

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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/predict` | POST | Predict URLs |
| `/api/explain` | POST | Detailed explanation |

### Request/Response Examples

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

## 📈 Model Performance

- **Training Data**: 1,607 labeled URLs
- **Ensemble Accuracy**: ~83%
- **Features**: TF-IDF text + 14 engineered features

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

MIT License - see LICENSE file for details.

## 👤 Author

**Akhmad Zaqi Riyadi**
- GitHub: [@akhmadzaqiriyadi](https://github.com/akhmadzaqiriyadi)

## 🙏 Acknowledgments

- Indonesian government and educational institutions for test data
- scikit-learn, Flask, BeautifulSoup communities
