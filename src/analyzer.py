import pickle
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import Utils

# Optional torch import (not needed for inference without BERT)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Stopwords fallback
STOPWORDS = []
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    try:
        STOPWORDS = list(stopwords.words('indonesian'))
    except:
        STOPWORDS = list(stopwords.words('english'))
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Text preprocessing limited.")

# Optional BERT imports (only if torch is available)
BERT_AVAILABLE = False
BERT_MODEL = None
BERT_TOKENIZER = None
if TORCH_AVAILABLE:
    try:
        from transformers import AutoModel, AutoTokenizer
        BERT_MODEL_NAME = "indolem/indobert-base-uncased"
        BERT_TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        BERT_MODEL = AutoModel.from_pretrained(BERT_MODEL_NAME)
        BERT_AVAILABLE = True
    except:
        try:
            BERT_MODEL_NAME = "bert-base-multilingual-uncased"
            BERT_TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            BERT_MODEL = AutoModel.from_pretrained(BERT_MODEL_NAME)
            BERT_AVAILABLE = True
        except:
            logger.warning("Transformers not available. BERT embeddings disabled.")
else:
    logger.info("PyTorch not installed. BERT embeddings disabled (TF-IDF only).")

class TextAnalyzer:
    """Text analysis with TF-IDF and optional BERT embeddings."""
    
    def __init__(self, use_bert=False, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, min_df=2, max_df=0.85, ngram_range=(1, 2), stop_words=STOPWORDS if NLTK_AVAILABLE else None
        )
        self.use_bert = use_bert and BERT_AVAILABLE
        self.bert_tokenizer = BERT_TOKENIZER if BERT_AVAILABLE else None
        self.bert_model = BERT_MODEL if BERT_AVAILABLE else None
        self.vectorizer_fitted = False

    def preprocess_text(self, text):
        """Preprocess text."""
        if not text:
            return ""
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower())
                tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
                return ' '.join(tokens)
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}. Falling back to basic normalization.")
                return Utils.normalize_text(text)
        return Utils.normalize_text(text)

    def fit_vectorizer(self, text_samples):
        """Fit TF-IDF vectorizer."""
        processed_samples = [self.preprocess_text(text) for text in text_samples]
        if not any(processed_samples):
            raise ValueError("All text samples are empty after preprocessing.")
        self.vectorizer.fit(processed_samples)
        self.vectorizer_fitted = True
        return self

    def transform_text(self, text_samples):
        """Transform text to TF-IDF vectors."""
        processed_samples = [self.preprocess_text(text) for text in text_samples]
        return self.vectorizer.transform(processed_samples)

    def get_bert_embeddings(self, text_samples, max_length=512):
        """Get BERT embeddings."""
        if not self.use_bert:
            return None
        embeddings = []
        for text in text_samples:
            inputs = self.bert_tokenizer(
                self.preprocess_text(text)[:10000], return_tensors="pt", max_length=max_length, padding="max_length", truncation=True
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].numpy().flatten())
        return np.array(embeddings)

    def save_vectorizer(self, path):
        """Save vectorizer."""
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_vectorizer(self, path):
        """Load vectorizer."""
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            self.vectorizer_fitted = True
        return self
