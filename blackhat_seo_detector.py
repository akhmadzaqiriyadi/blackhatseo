#!/usr/bin/env python3

# Black Hat SEO Detection: Enhanced WebScraper and ML Models
# Improved version with dataset building, fixed stop_words, robust error handling, and imbalance handling
# Updated with enhanced meta tag analysis, cloaking detection, adjusted domain weighting, and more

import argparse
import json
import logging
import os
import pickle
import random
import re
import sys
import time
import concurrent.futures
from datetime import datetime
from urllib.parse import urlparse, quote

import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample
from tqdm import tqdm
import scipy.sparse
import difflib

# Optional dependencies
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("Selenium not available. Dynamic content scraping disabled.")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    try:
        STOPWORDS = list(stopwords.words('indonesian'))
    except:
        STOPWORDS = list(stopwords.words('english'))
    NLTK_AVAILABLE = True
except ImportError:
    STOPWORDS = []
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Text preprocessing limited.")

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
        BERT_AVAILABLE = False
        logging.warning("Transformers not available. BERT embeddings disabled.")

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('seo_detector.log')
    ]
)
logger = logging.getLogger(__name__)

# Enhanced keyword lists
JUDI_KEYWORDS = [
    'poker', 'casino', 'slot', 'togel', 'betting', 'judi', 'taruhan', 'bola',
    'sportsbook', 'bandar', 'jackpot', 'bonus', 'deposit', 'withdraw', 'gambling',
    'roulette', 'blackjack', 'baccarat', 'lottery', 'gaple', 'domino', 'toto',
    'vegas', 'parlay', '4d', '3d', '2d', 'bandarq', 'capsa', 'aduq', 'sakong',
    'ceme', 'domino99', 'agen bola', 'mesin slot', 'pragmatic play', 'habanero',
    'spadegaming', 'microgaming', 'bet88', 'situs judi', 'idn poker', 'live casino',
    'sabung ayam', 'tembak ikan', 'joker123', 'freebet', 'cashback', 'welcome bonus',
    'rupiahtoto', 'gacor', 'maxwin', 'rajaslot', 'bandarqq', 'raja toto',
    'situs judi online', 'jackpot online', 'slot online', 'agen bola online',
    'betting online', 'daftar slot', 'link alternatif', 'bo slot'
]

SPAM_KEYWORDS = [
    'viagra', 'cialis', 'pharmacy', 'pills', 'cheap', 'discount', 'free',
    'guaranteed', 'winner', 'obat', 'kuat', 'pembesar', 'make money', 'weight loss',
    'diet', 'earn money', 'bisnis online', 'passive income', 'pelangsing',
    'peninggi badan', 'crypto investment', 'profit harian', 'pinjaman instan',
    'kaya mendadak', 'dropship', 'affiliate marketing', 'obat herbal', 'cepat kaya',
    'link alternatif', 'situs resmi', 'daftar sekarang', 'bonus new member',
    'banting harga', 'gampang menang', 'terpercaya', 'terbaik', 'terlengkap',
    'resmi dan terpercaya', 'paling gacor', 'nomor 1', 'terbesar', 'modal receh'
]

class Utils:
    """Utility functions for text processing and URL handling."""
    
    @staticmethod
    def get_random_user_agent():
        """Return a random user agent."""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/93.0.4577.63 Safari/537.36 Edg/93.0.961.38',
        ]
        return random.choice(user_agents)

    @staticmethod
    def normalize_text(text):
        """Clean and normalize text."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def is_valid_url(url):
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    @staticmethod
    def get_domain_info(url):
        """Extract domain information."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            tld_parts = domain.split('.')
            tld = tld_parts[-1] if len(tld_parts) > 1 else ""
            return {
                'domain': domain,
                'tld': tld,
                'is_gov': domain.endswith('.go.id'),
                'is_edu': domain.endswith('.ac.id') or domain.endswith('.edu'),
                'is_org': domain.endswith('.org') or domain.endswith('.or.id'),
                'is_com': domain.endswith('.com') or domain.endswith('.co.id')
            }
        except:
            return {'domain': '', 'tld': '', 'is_gov': False, 'is_edu': False, 'is_org': False, 'is_com': False}

class WebScraper:
    """Web scraper for static and dynamic content with enhanced analysis."""
    
    def __init__(self, use_selenium=False, chrome_driver_path=None, headless=True, timeout=60):
        """Initialize scraper."""
        self.headers = {'User-Agent': Utils.get_random_user_agent()}
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.timeout = timeout
        self.driver = None
        if self.use_selenium:
            try:
                options = Options()
                if headless:
                    options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument(f'user-agent={self.headers["User-Agent"]}')
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
                self.driver.set_page_load_timeout(timeout)
                logger.info("Selenium WebDriver initialized.")
            except Exception as e:
                logger.warning(f"Selenium initialization failed: {e}. Falling back to requests.")
                self.use_selenium = False

    def __del__(self):
        """Clean up Selenium driver."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

    def get_page_content(self, url):
        """Fetch page content."""
        if not Utils.is_valid_url(url):
            logger.error(f"Invalid URL: {url}")
            return None
        if self.use_selenium:
            content = self._get_with_selenium(url)
            if content:
                return content
            logger.info(f"Selenium failed for {url}, falling back to requests.")
        return self._get_with_requests(url)

    def _get_with_selenium(self, url):
        """Fetch content using Selenium."""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Selenium error for {url}: {e}")
            return None

    def _get_with_requests(self, url):
        """Fetch content using requests with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': Utils.get_random_user_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive'
                }
                response = requests.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Request error for {url}: {e}")
                    return None
                time.sleep(random.uniform(2, 5))

    def detect_cloaking(self, url):
        """Detect cloaking by comparing content from different user agents."""
        try:
            headers_user = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36'}
            response_user = requests.get(url, headers=headers_user, timeout=self.timeout)
            headers_bot = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}
            response_bot = requests.get(url, headers=headers_bot, timeout=self.timeout)
            soup_user = BeautifulSoup(response_user.text, 'html.parser')
            soup_bot = BeautifulSoup(response_bot.text, 'html.parser')
            user_text = soup_user.get_text(separator=' ', strip=True)[:1000]
            bot_text = soup_bot.get_text(separator=' ', strip=True)[:1000]
            similarity = difflib.SequenceMatcher(None, user_text, bot_text).ratio()
            return {
                'cloaking_detected': similarity < 0.8,
                'similarity_score': similarity
            }
        except Exception as e:
            logger.warning(f"Error detecting cloaking for {url}: {e}")
            return {'cloaking_detected': False, 'similarity_score': 1.0}

    def check_redirect_chain(self, url):
        """Check redirect chain for suspicious URLs."""
        redirects = []
        try:
            response = requests.head(url, allow_redirects=True, timeout=self.timeout)
            if response.history:
                for resp in response.history:
                    redirects.append(resp.url)
                redirects.append(response.url)
            suspicious_redirects = any(
                any(keyword in redirect.lower() for keyword in JUDI_KEYWORDS)
                for redirect in redirects
            )
            return {
                'redirect_chain': redirects,
                'has_redirects': len(redirects) > 0,
                'suspicious_redirects': suspicious_redirects
            }
        except Exception as e:
            logger.warning(f"Error checking redirects for {url}: {e}")
            return {'redirect_chain': [], 'has_redirects': False, 'suspicious_redirects': False}

    def check_last_modified_date(self, url):
        """Check when the page was last modified."""
        try:
            response = requests.head(url, timeout=self.timeout)
            last_modified = response.headers.get('Last-Modified')
            if last_modified:
                last_mod_date = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S %Z')
                days_since_modified = (datetime.now() - last_mod_date).days
                return {'last_modified': last_modified, 'days_since_modified': days_since_modified}
            return {'last_modified': None, 'days_since_modified': None}
        except Exception as e:
            logger.warning(f"Error checking last modified date for {url}: {e}")
            return {'last_modified': None, 'days_since_modified': None}

    def detect_hidden_content(self, soup):
        """Detect hidden content that may contain spam."""
        hidden_elements = []
        for element in soup.find_all(style=True):
            style = element.get('style', '').lower()
            if ('display:none' in style or 'visibility:hidden' in style or 
                'height:0' in style or 'width:0' in style or 
                'font-size:0' in style or 'opacity:0' in style):
                text = element.get_text(strip=True)
                if text:
                    hidden_elements.append({
                        'element': element.name,
                        'text': text[:100],
                        'style': style
                    })
        suspicious_classes = ['hidden', 'hide', 'invisible', 'seo', 'crawl-only']
        for cls in suspicious_classes:
            for element in soup.find_all(class_=lambda x: x and cls in x.lower()):
                text = element.get_text(strip=True)
                if text:
                    hidden_elements.append({
                        'element': element.name,
                        'text': text[:100],
                        'class': element.get('class', '')
                    })
        hidden_judi_keywords = sum(
            keyword in ' '.join([item['text'].lower() for item in hidden_elements])
            for keyword in JUDI_KEYWORDS
        )
        return {
            'hidden_elements_count': len(hidden_elements),
            'hidden_elements': hidden_elements[:5],
            'hidden_judi_keywords': hidden_judi_keywords,
            'hidden_content_detected': len(hidden_elements) > 0
        }

    def detect_malicious_js(self, soup):
        """Detect suspicious JavaScript that may be used for redirects."""
        scripts = soup.find_all('script')
        suspicious_patterns = [
            'window.location', 'document.location', 'location.href', 
            'eval(', 'document.write(', 'setTimeout(', 'obfuscated',
            'unescape(', 'fromCharCode', 'createElement', '.src='
        ]
        suspicious_scripts = []
        for script in scripts:
            script_content = script.string if script.string else ''
            if any(pattern in script_content for pattern in suspicious_patterns):
                suspicious_scripts.append({
                    'content': script_content[:200] + '...' if len(script_content) > 200 else script_content,
                    'suspicious_pattern': [p for p in suspicious_patterns if p in script_content]
                })
        return {
            'script_count': len(scripts),
            'suspicious_script_count': len(suspicious_scripts),
            'suspicious_scripts': suspicious_scripts[:3]
        }

    def analyze_link_structure(self, soup, base_url):
        """Analyze link structure in detail."""
        links = soup.find_all('a', href=True)
        internal_links = []
        external_links = []
        suspicious_links = []
        domain = urlparse(base_url).netloc
        for link in links:
            href = link['href']
            link_text = link.get_text(strip=True)
            if href.startswith('/'):
                full_url = f"{urlparse(base_url).scheme}://{domain}{href}"
            elif not href.startswith(('http://', 'https://')):
                full_url = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
            else:
                full_url = href
            if domain in full_url:
                internal_links.append({'url': full_url, 'text': link_text})
            else:
                external_links.append({'url': full_url, 'text': link_text})
                if any(keyword in link_text.lower() for keyword in JUDI_KEYWORDS + SPAM_KEYWORDS):
                    suspicious_links.append({'url': full_url, 'text': link_text, 'reason': 'suspicious_text'})
                external_domain = urlparse(full_url).netloc
                if any(keyword in external_domain for keyword in ['slot', 'casino', 'poker', 'judi', 'bet', 'togel']):
                    suspicious_links.append({'url': full_url, 'text': link_text, 'reason': 'suspicious_domain'})
        return {
            'internal_link_count': len(internal_links),
            'external_link_count': len(external_links),
            'suspicious_link_count': len(suspicious_links),
            'suspicious_links': suspicious_links,
            'link_ratio': len(external_links) / max(len(internal_links), 1)
        }

    def _analyze_meta_tags(self, soup):
        """Analyze meta tags in detail."""
        meta_tags = {}
        title = soup.find('title').get_text() if soup.find('title') else ""
        description = ""
        keywords = ""
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            property = meta.get('property', '').lower()
            content = meta.get('content', '')
            if name:
                meta_tags[name] = content
                if name == 'description':
                    description = content
                elif name == 'keywords':
                    keywords = content
            elif property:
                meta_tags[property] = content
                if property in ['og:description', 'twitter:description']:
                    if not description:
                        description = content
        meta_content = (title + " " + description + " " + keywords).lower()
        judi_count = sum(keyword in meta_content for keyword in JUDI_KEYWORDS)
        spam_count = sum(keyword in meta_content for keyword in SPAM_KEYWORDS)
        headers = {f'h{i}': [tag.get_text(strip=True) for tag in soup.find_all(f'h{i}')] for i in range(1, 7)}
        return {
            'title': title,
            'description': description,
            'keywords': keywords,
            'meta_tags': meta_tags,
            'judi_count': judi_count,
            'spam_count': spam_count,
            'suspicious': judi_count > 0 or spam_count > 2,
            'headers': headers
        }

    def extract_text_from_url(self, url):
        """Extract and process text from URL with comprehensive analysis."""
        content = self.get_page_content(url)
        if not content:
            return None
        try:
            soup = BeautifulSoup(content, 'html.parser')
            for element in soup(['script', 'style', 'comment', 'noscript', 'iframe']):
                element.extract()
            meta_analysis = self._analyze_meta_tags(soup)
            link_analysis = self.analyze_link_structure(soup, url)
            hidden_content = self.detect_hidden_content(soup)
            js_analysis = self.detect_malicious_js(soup)
            cloaking = self.detect_cloaking(url)
            redirect_info = self.check_redirect_chain(url)
            modified_info = self.check_last_modified_date(url)
            text = soup.get_text(separator=' ', strip=True)
            text = Utils.normalize_text(text)
            domain_info = Utils.get_domain_info(url)
            return {
                'url': url,
                'domain': domain_info['domain'],
                'tld': domain_info['tld'],
                'is_gov_domain': domain_info['is_gov'],
                'is_edu_domain': domain_info['is_edu'],
                'is_org_domain': domain_info['is_org'],
                'title': meta_analysis['title'],
                'description': meta_analysis['description'],
                'full_text': text,
                'combined_text': f"{meta_analysis['title']} {meta_analysis['description']} {text}",
                'important_text': f"{meta_analysis['title']} {meta_analysis['description']} {' '.join([h for headers in meta_analysis['headers'].values() for h in headers])}",
                'extracted_at': datetime.now().isoformat(),
                'meta_tags': meta_analysis['meta_tags'],
                'meta_judi_count': meta_analysis['judi_count'],
                'meta_spam_count': meta_analysis['spam_count'],
                'meta_suspicious': meta_analysis['suspicious'],
                'links': link_analysis['suspicious_links'],
                'suspicious_link_count': link_analysis['suspicious_link_count'],
                'link_ratio': link_analysis['link_ratio'],
                'hidden_elements_count': hidden_content['hidden_elements_count'],
                'hidden_judi_keywords': hidden_content['hidden_judi_keywords'],
                'hidden_content_detected': hidden_content['hidden_content_detected'],
                'suspicious_script_count': js_analysis['suspicious_script_count'],
                'cloaking_detected': cloaking['cloaking_detected'],
                'similarity_score': cloaking['similarity_score'],
                'suspicious_redirects': redirect_info['suspicious_redirects'],
                'redirect_chain': redirect_info['redirect_chain'],
                'last_modified': modified_info['last_modified'],
                'days_since_modified': modified_info['days_since_modified']
            }
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None

    def check_seo_metrics(self, url_data):
        """Analyze SEO metrics with adjusted thresholds."""
        if not url_data:
            return {}
        metrics = {}
        combined_text = url_data.get('combined_text', '').lower()
        important_text = url_data.get('important_text', '').lower()
        title_text = url_data.get('title', '').lower()
        desc_text = url_data.get('description', '').lower()
        if combined_text:
            words = combined_text.split()
            word_count = len(words)
            metrics['word_count'] = word_count
            metrics['is_thin_content'] = word_count < 300
            word_freq = {word: words.count(word) for word in set(words) if len(word) > 3 and word not in STOPWORDS}
            if word_freq:
                top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                metrics['top_keywords'] = top_keywords
                metrics['top_keyword_density'] = top_keywords[0][1] / word_count if top_keywords and word_count > 0 else 0
                metrics['keyword_stuffing_detected'] = any(count / word_count > 0.03 for count in word_freq.values())
        metrics.update({
            'judi_keyword_count': sum(keyword in combined_text for keyword in JUDI_KEYWORDS),
            'spam_keyword_count': sum(keyword in combined_text for keyword in SPAM_KEYWORDS),
            'judi_in_title': sum(keyword in title_text for keyword in JUDI_KEYWORDS),
            'spam_in_title': sum(keyword in title_text for keyword in SPAM_KEYWORDS),
            'judi_in_desc': sum(keyword in desc_text for keyword in JUDI_KEYWORDS),
            'spam_in_desc': sum(keyword in desc_text for keyword in SPAM_KEYWORDS),
            'judi_in_important': sum(keyword in important_text for keyword in JUDI_KEYWORDS),
            'spam_in_important': sum(keyword in important_text for keyword in SPAM_KEYWORDS)
        })
        links = url_data.get('links', [])
        metrics.update({
            'suspicious_link_count': len(links),
            'suspicious_link_ratio': len(links) / len(links) if links else 0,
            'hidden_text_suspicion': (
                (metrics.get('judi_in_important', 0) + metrics.get('spam_in_important', 0)) /
                (metrics.get('judi_keyword_count', 1) + metrics.get('spam_keyword_count', 1) + 1)
            ) < 0.3
        })
        is_official_domain = (url_data.get('is_gov_domain', False) or 
                             url_data.get('is_edu_domain', False) or 
                             url_data.get('is_org_domain', False))
        judi_threshold = 1 if is_official_domain else 3
        spam_threshold = 2 if is_official_domain else 5
        is_black_hat = (
            metrics.get('judi_keyword_count', 0) >= judi_threshold or
            url_data.get('meta_judi_count', 0) > 0 or
            metrics.get('spam_keyword_count', 0) >= spam_threshold or
            metrics.get('keyword_stuffing_detected', False) or
            url_data.get('cloaking_detected', False) or
            metrics.get('hidden_text_suspicion', False)
        )
        metrics['lower_threshold_triggered'] = is_black_hat and is_official_domain
        return metrics

class TextAnalyzer:
    """Text analysis with TF-IDF and optional BERT embeddings."""
    
    def __init__(self, use_bert=False, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, min_df=2, max_df=0.85, ngram_range=(1, 2), stop_words=STOPWORDS if NLTK_AVAILABLE else None
        )
        self.use_bert = use_bert and BERT_AVAILABLE
        self.bert_tokenizer = BERT_TOKENIZER if BERT_AVAILABLE else None
        self.bert_model = BERT_MODEL if BERT_AVAILABLE else None

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
        return self

class FeatureEngineer:
    """Feature engineering for ML model."""
    
    def __init__(self, text_analyzer):
        self.text_analyzer = text_analyzer

    def transform(self, X):
        """Transform data into features with adjusted weights."""
        features = []
        for item in X:
            is_edu = int(item.get('is_edu_domain', False))
            judi_count = item.get('judi_keyword_count', 0)
            meta_judi_count = item.get('meta_judi_count', 0)
            edu_weight = 1.0 if is_edu and (judi_count == 0 and meta_judi_count == 0) else 0.2
            feature_dict = {
                'is_gov_domain': int(item.get('is_gov_domain', False)),
                'is_edu_domain': edu_weight * is_edu,
                'word_count': item.get('word_count', 0),
                'judi_keyword_count': judi_count,
                'meta_judi_count': meta_judi_count * 2,
                'spam_keyword_count': item.get('spam_keyword_count', 0),
                'meta_spam_count': item.get('meta_spam_count', 0) * 2,
                'judi_in_title': item.get('judi_in_title', 0) * 3,
                'spam_in_title': item.get('spam_in_title', 0) * 3,
                'suspicious_link_count': item.get('suspicious_link_count', 0),
                'keyword_stuffing_detected': int(item.get('keyword_stuffing_detected', False)),
                'hidden_text_suspicion': int(item.get('hidden_text_suspicion', False)),
                'is_thin_content': int(item.get('is_thin_content', False)),
                'meta_suspicious': int(item.get('meta_suspicious', False)) * 3
            }
            features.append(feature_dict)
        return pd.DataFrame(features)

class DatasetBuilder:
    """Class to assist in building a training dataset for Black Hat SEO detection."""
    
    def __init__(self, scraper, serpapi_key=None):
        """Initialize with a WebScraper instance and optional SerpApi key."""
        self.scraper = scraper
        self.search_api_key = serpapi_key
    
    def collect_urls_from_search(self, queries, max_results=10, use_api=True):
        """Collect URLs from search engine results using SerpApi or manual scraping."""
        urls = set()
        for query in queries:
            logger.info(f"Collecting URLs for query: {query}")
            if use_api and self.search_api_key:
                try:
                    params = {
                        'q': query,
                        'api_key': self.search_api_key,
                        'engine': 'google',
                        'hl': 'id',
                        'gl': 'id',
                        'num': max_results
                    }
                    response = requests.get('https://serpapi.com/search', params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    if 'error' in data:
                        logger.error(f"SerpApi error for query {query}: {data['error']}")
                        continue
                    organic_results = data.get('organic_results', [])
                    for result in organic_results[:max_results]:
                        url = result.get('link')
                        if url and Utils.is_valid_url(url):
                            urls.add(url)
                    logger.info(f"Collected {len(urls)} URLs for query: {query}")
                    time.sleep(random.uniform(1, 3))
                except Exception as e:
                    logger.error(f"SerpApi error for query {query}: {e}")
                    logger.info(f"Falling back to manual scraping for {query}")
                    urls.update(self._collect_urls_manually(query, max_results))
            else:
                urls.update(self._collect_urls_manually(query, max_results))
        return list(urls)[:max_results * len(queries)]

    def _collect_urls_manually(self, query, max_results):
        """Fallback method for manual Google scraping."""
        urls = set()
        try:
            headers = {'User-Agent': Utils.get_random_user_agent()}
            encoded_query = quote(query)
            response = requests.get(
                f"https://www.google.com/search?q={encoded_query}",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('http') and 'google.com' not in href:
                    urls.add(href)
            logger.info(f"Manually collected {len(urls)} URLs for query: {query}")
            time.sleep(random.uniform(1, 3))
        except Exception as e:
            logger.error(f"Manual scraping error for {query}: {e}")
        return list(urls)[:max_results]

    def collect_urls_from_file(self, file_path):
        """Read URLs from a text file."""
        try:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and Utils.is_valid_url(line.strip())]
            logger.info(f"Loaded {len(urls)} URLs from {file_path}")
            return urls
        except Exception as e:
            logger.error(f"Error reading URLs from {file_path}: {e}")
            return []

    def label_urls(self, urls, threshold=5):
        """Semi-automatically label URLs based on SEO metrics."""
        labeled_data = []
        for url in tqdm(urls, desc="Labeling URLs"):
            try:
                url_data = self.scraper.extract_text_from_url(url)
                if not url_data:
                    logger.warning(f"Skipping {url}: Failed to scrape")
                    continue
                metrics = self.scraper.check_seo_metrics(url_data)
                is_black_hat = (
                    metrics.get('judi_keyword_count', 0) > threshold or
                    metrics.get('spam_keyword_count', 0) > threshold or
                    metrics.get('keyword_stuffing_detected', False) or
                    metrics.get('hidden_text_suspicion', False)
                )
                label = 1 if is_black_hat else 0
                if metrics.get('is_gov_domain', False) or metrics.get('is_edu_domain', False):
                    label = 0
                labeled_data.append({'url': url, 'label': label})
                logger.debug(f"Labeled {url}: {'Black Hat' if label == 1 else 'Clean'}")
            except Exception as e:
                logger.error(f"Error labeling {url}: {e}")
        return labeled_data

    def save_dataset(self, data, output_path):
        """Save labeled data to CSV."""
        try:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            logger.info(f"Dataset saved to {output_path} with {len(df)} URLs")
        except Exception as e:
            logger.error(f"Error saving dataset to {output_path}: {e}")

    def build_dataset(self, clean_queries=None, blackhat_queries=None, url_files=None, output_path="train_data.csv", threshold=5):
        """Build a dataset by collecting and labeling URLs."""
        all_urls = set()
        if clean_queries:
            clean_urls = self.collect_urls_from_search(clean_queries, use_api=True)
            all_urls.update(clean_urls)
        if blackhat_queries:
            blackhat_urls = self.collect_urls_from_search(blackhat_queries, use_api=True)
            all_urls.update(blackhat_urls)
        if url_files:
            for file_path in url_files:
                file_urls = self.collect_urls_from_file(file_path)
                all_urls.update(file_urls)
        logger.info(f"Total URLs collected: {len(all_urls)}")
        labeled_data = self.label_urls(list(all_urls), threshold=threshold)
        if labeled_data:
            self.save_dataset(labeled_data, output_path)
        else:
            logger.error("No data collected. Dataset not created.")

class BlackHatSEODetector:
    """Black Hat SEO detector with ML models."""
    
    def __init__(self, model_path=None, vectorizer_path=None, use_selenium=False, use_bert=False):
        """Initialize detector."""
        self.web_scraper = WebScraper(use_selenium=use_selenium)
        self.text_analyzer = TextAnalyzer(use_bert=use_bert)
        self.feature_engineer = FeatureEngineer(self.text_analyzer)
        self.model = None
        self.vectorizer = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        if vectorizer_path and os.path.exists(vectorizer_path):
            self.text_analyzer.load_vectorizer(vectorizer_path)
            self.vectorizer = self.text_analyzer.vectorizer

    def _process_url(self, url):
        """Process a single URL."""
        if not Utils.is_valid_url(url):
            logger.error(f"Invalid URL: {url}")
            return None
        try:
            response = requests.head(url, headers={'User-Agent': Utils.get_random_user_agent()}, timeout=5, allow_redirects=True)
            if response.status_code >= 400:
                logger.warning(f"Skipping {url}: HTTP {response.status_code}")
                return None
        except requests.RequestException:
            logger.warning(f"Skipping {url}: Inaccessible")
            return None
        url_data = self.web_scraper.extract_text_from_url(url)
        if not url_data:
            return None
        seo_metrics = self.web_scraper.check_seo_metrics(url_data)
        return {**url_data, **seo_metrics}

    def _check_overriding(self, url_data):
        """Detect indicators that override edu/gov domain status."""
        is_official = url_data.get('is_edu_domain', False) or url_data.get('is_gov_domain', False)
        if not is_official:
            return False, "Not an official domain"
        override_signals = []
        if url_data.get('meta_judi_count', 0) > 0:
            override_signals.append(f"Meta tags contain {url_data['meta_judi_count']} gambling keywords")
        if url_data.get('judi_in_title', 0) > 0:
            override_signals.append(f"Title contains {url_data['judi_in_title']} gambling keywords")
        if url_data.get('cloaking_detected', False):
            override_signals.append(f"Cloaking detected (similarity: {url_data.get('similarity_score', 0):.2f})")
        if url_data.get('suspicious_link_count', 0) > 2:
            override_signals.append(f"High number of suspicious links: {url_data['suspicious_link_count']}")
        if url_data.get('suspicious_script_count', 0) > 0:
            override_signals.append(f"Suspicious JavaScript detected: {url_data['suspicious_script_count']} scripts")
        if url_data.get('hidden_judi_keywords', 0) > 0:
            override_signals.append(f"Hidden content contains {url_data['hidden_judi_keywords']} gambling keywords")
        should_override = len(override_signals) > 0
        reason = ", ".join(override_signals) if should_override else "No strong indicators to override official domain status"
        return should_override, reason

    def _prepare_dataset(self, urls, labels=None):
        """Prepare dataset."""
        data = []
        url_to_label = dict(zip(urls, labels)) if labels is not None else {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_url = {executor.submit(self._process_url, url): url for url in urls}
            for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(urls), desc="Scraping URLs"):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        if url in url_to_label:
                            result['label'] = url_to_label[url]
                        data.append(result)
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
        df = pd.DataFrame(data)
        if df.empty:
            logger.error("No data extracted from URLs.")
            return df
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"DataFrame shape: {df.shape}")
        return df

    def train_model(self, urls, labels, test_size=0.2, random_state=42):
        """Train ML model with oversampling."""
        logger.info(f"Training model on {len(urls)} URLs...")
        df = self._prepare_dataset(urls, labels)
        if df.empty:
            logger.error("No data extracted from URLs.")
            return None
        if 'label' not in df.columns:
            logger.error("No labels found in the dataset. Ensure the input CSV contains a 'label' column.")
            return None
        df = df.dropna(subset=['label', 'combined_text'])
        if df.empty:
            logger.error("No valid data after filtering. Cannot train model.")
            return None
        text_samples = df['combined_text'].fillna('').tolist()
        if not any(text_samples):
            logger.error("All text samples are empty. Cannot train model.")
            return None
        self.text_analyzer.fit_vectorizer(text_samples)
        self.vectorizer = self.text_analyzer.vectorizer
        X_text = self.text_analyzer.transform_text(text_samples)
        X_features = self.feature_engineer.transform(df.to_dict('records'))
        y = df['label'].values
        train_df = pd.DataFrame(X_features)
        train_df['label'] = y
        train_df['text_features'] = [X_text[i] for i in range(X_text.shape[0])]
        majority = train_df[train_df['label'] == 1]
        minority = train_df[train_df['label'] == 0]
        if len(minority) < len(majority) and len(minority) > 0:
            minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=random_state)
            train_df = pd.concat([majority, minority_upsampled])
            X_features_resampled = train_df.drop(['label', 'text_features'], axis=1).values
            X_text_resampled = scipy.sparse.vstack(train_df['text_features'])
            y_resampled = train_df['label'].values
        else:
            X_features_resampled, X_text_resampled, y_resampled = X_features, X_text, y
        X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
            X_text_resampled, X_features_resampled, y_resampled, test_size=test_size, random_state=random_state, stratify=y_resampled
        )
        text_model = SVC(probability=True, kernel='linear', C=1.0).fit(X_text_train, y_train)
        feature_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features_train, y_train)
        self.model = {'text_model': text_model, 'feature_model': feature_model}
        self._evaluate_model(X_text_test, X_features_test, y_test)
        return self.model

    def _evaluate_model(self, X_text_test, X_features_test, y_test):
        """Evaluate model."""
        text_proba = self.model['text_model'].predict_proba(X_text_test)[:, 1]
        feature_proba = self.model['feature_model'].predict_proba(X_features_test)[:, 1]
        ensemble_proba = (text_proba + feature_proba) / 2
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        logger.info("\nModel Evaluation:")
        logger.info(f"Text Model AUC: {roc_auc_score(y_test, text_proba):.4f}")
        logger.info(f"Feature Model AUC: {roc_auc_score(y_test, feature_proba):.4f}")
        logger.info(f"Ensemble Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
        logger.info(f"Ensemble AUC: {roc_auc_score(y_test, ensemble_proba):.4f}")
        logger.info("\nClassification Report:\n" + classification_report(y_test, ensemble_pred))

    def predict(self, urls):
        """Predict black hat SEO with enhanced logic."""
        if not self.model or not self.vectorizer:
            raise ValueError("Model or vectorizer not trained or loaded.")
        results = []
        for url in tqdm(urls, desc="Predicting"):
            try:
                url_data = self._process_url(url)
                if not url_data:
                    results.append({'url': url, 'error': 'Failed to process URL', 'prediction': None, 'probability': None, 'metrics': {}})
                    continue
                should_override, override_reason = self._check_overriding(url_data)
                X_text = self.text_analyzer.transform_text([url_data.get('combined_text', '')])
                X_features = self.feature_engineer.transform([url_data])
                text_proba = self.model['text_model'].predict_proba(X_text)[0, 1]
                feature_proba = self.model['feature_model'].predict_proba(X_features)[0, 1]
                ensemble_proba = (text_proba + feature_proba) / 2
                if should_override:
                    ensemble_proba = max(0.65, ensemble_proba)
                if url_data.get('meta_judi_count', 0) > 0:
                    ensemble_proba = max(ensemble_proba, 0.7)
                if url_data.get('cloaking_detected', False):
                    ensemble_proba = max(ensemble_proba, 0.75)
                prediction = 1 if ensemble_proba >= 0.5 else 0
                reasons = []
                if url_data.get('judi_keyword_count', 0) > 0:
                    reasons.append(f"Found {url_data['judi_keyword_count']} gambling-related keywords")
                if url_data.get('meta_judi_count', 0) > 0:
                    reasons.append(f"Found {url_data['meta_judi_count']} gambling keywords in meta tags")
                if url_data.get('judi_in_title', 0) > 0:
                    reasons.append(f"Found {url_data['judi_in_title']} gambling keywords in title")
                if url_data.get('spam_keyword_count', 0) > 0:
                    reasons.append(f"Found {url_data['spam_keyword_count']} spam-related keywords")
                if url_data.get('keyword_stuffing_detected', False):
                    reasons.append("Detected potential keyword stuffing")
                if url_data.get('suspicious_link_count', 0) > 0:
                    reasons.append(f"Found {url_data['suspicious_link_count']} suspicious links")
                if url_data.get('hidden_content_detected', False):
                    reasons.append("Detected hidden content")
                if url_data.get('cloaking_detected', False):
                    reasons.append(f"Detected possible cloaking (similarity: {url_data.get('similarity_score', 0):.2f})")
                if url_data.get('suspicious_redirects', False):
                    reasons.append("Detected suspicious redirects")
                if should_override:
                    reasons.append(f"Override applied for official domain: {override_reason}")
                results.append({
                    'url': url,
                    'prediction': prediction,
                    'probability': ensemble_proba,
                    'metrics': {
                        'judi_keyword_count': url_data.get('judi_keyword_count', 0),
                        'spam_keyword_count': url_data.get('spam_keyword_count', 0),
                        'meta_judi_count': url_data.get('meta_judi_count', 0),
                        'suspicious_link_count': url_data.get('suspicious_link_count', 0),
                        'keyword_stuffing_detected': url_data.get('keyword_stuffing_detected', False),
                        'hidden_content_detected': url_data.get('hidden_content_detected', False),
                        'cloaking_detected': url_data.get('cloaking_detected', False),
                        'suspicious_redirects': url_data.get('suspicious_redirects', False),
                        'is_edu_domain': url_data.get('is_edu_domain', False),
                        'is_gov_domain': url_data.get('is_gov_domain', False)
                    },
                    'reasons': reasons
                })
            except Exception as e:
                logger.error(f"Error predicting for {url}: {e}")
                results.append({'url': url, 'error': str(e), 'prediction': None, 'probability': None, 'metrics': {}})
        failed_urls = [r['url'] for r in results if 'error' in r]
        if failed_urls:
            logger.warning(f"Failed to process {len(failed_urls)} URLs: {failed_urls}")
        return results

    def save_model(self, model_path, vectorizer_path=None):
        """Save model and vectorizer."""
        if not self.model:
            raise ValueError("Model not trained.")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        if vectorizer_path and self.vectorizer:
            self.text_analyzer.save_vectorizer(vectorizer_path)
        logger.info(f"Model saved to {model_path}")
        if vectorizer_path:
            logger.info(f"Vectorizer saved to {vectorizer_path}")

    def load_model(self, model_path):
        """Load model."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        return self.model

    def explain_prediction(self, url):
        """Explain prediction for a URL."""
        if not self.model or not self.vectorizer:
            raise ValueError("Model or vectorizer not trained or loaded.")
        url_data = self._process_url(url)
        if not url_data:
            return {'url': url, 'error': 'Failed to process URL', 'explanation': 'Could not scrape or analyze URL'}
        prediction_result = self.predict([url])[0]
        explanation = {
            'url': url,
            'prediction': 'Black Hat SEO detected' if prediction_result['prediction'] == 1 else 'No Black Hat SEO detected',
            'confidence': f"{prediction_result['probability'] * 100:.2f}%",
            'reasons': prediction_result['reasons'],
            'top_keywords': url_data.get('top_keywords', [])[:5]
        }
        return explanation

def load_config(config_path):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        return {'use_selenium': False, 'use_bert': False, 'verbose': False}
    with open(config_path, 'r') as f:
        return json.load(f)

def test_in_jupyter():
    """Test detector in Jupyter."""
    logger.info("Running Jupyter test...")
    scraper = WebScraper(use_selenium=False)
    logger.info("Testing dataset building...")
    builder = DatasetBuilder(scraper)
    clean_queries = ["berita indonesia", "pendidikan indonesia"]
    blackhat_queries = ["judi online", "togel online"]
    builder.build_dataset(
        clean_queries=clean_queries,
        blackhat_queries=blackhat_queries,
        output_path="test_train_data.csv",
        threshold=5
    )
    detector = BlackHatSEODetector(model_path='model.pkl', vectorizer_path='vectorizer.pkl', use_selenium=False)
    urls = ['https://example.com']
    results = detector.predict(urls)
    for result in results:
        if 'error' in result:
            logger.error(f"{result['url']}: {result['error']}")
        else:
            prediction = "Black Hat SEO" if result['prediction'] == 1 else "Clean"
            logger.info(f"{result['url']}: {prediction} (confidence: {result['probability']*100:.2f}%)")
    explanation = detector.explain_prediction(urls[0])
    logger.info(f"\nExplanation for {explanation['url']}:")
    logger.info(f"Prediction: {explanation['prediction']}")
    logger.info(f"Confidence: {explanation['confidence']}")
    logger.info("Reasons:")
    for reason in explanation['reasons']:
        logger.info(f"- {reason}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Black Hat SEO Detection Tool')
    parser.add_argument('--config', help='Path to config JSON file', default='config.json')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--train-urls', required=True, help='CSV file with URLs and labels')
    train_parser.add_argument('--model-output', required=True, help='Path to save model')
    train_parser.add_argument('--vectorizer-output', help='Path to save vectorizer')
    predict_parser = subparsers.add_parser('predict', help='Predict black hat SEO')
    predict_parser.add_argument('--urls', required=True, help='Text file with URLs')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    predict_parser.add_argument('--vectorizer', required=True, help='Path to trained vectorizer')
    predict_parser.add_argument('--output', help='Path to save results as JSON')
    explain_parser = subparsers.add_parser('explain', help='Explain prediction')
    explain_parser.add_argument('--url', required=True, help='URL to explain')
    explain_parser.add_argument('--model', required=True, help='Path to trained model')
    explain_parser.add_argument('--vectorizer', required=True, help='Path to trained vectorizer')
    build_parser = subparsers.add_parser('build-dataset', help='Build training dataset')
    build_parser.add_argument('--clean-queries', help='File with clean search queries (one per line)')
    build_parser.add_argument('--blackhat-queries', help='File with black hat search queries (one per line)')
    build_parser.add_argument('--url-files', nargs='*', help='Text files with URLs')
    build_parser.add_argument('--output', default='train_data.csv', help='Output CSV file')
    build_parser.add_argument('--threshold', type=int, default=5, help='Keyword count threshold for labeling')
    build_parser.add_argument('--serpapi-key', help='SerpApi key for search', default='7f0a9dd1fa51513c901923823c413d94906c75b0f171cdce302706da08c97ec2')
    if 'ipykernel' in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
    config = load_config(args.config)
    if args.verbose or config.get('verbose', False):
        logger.setLevel(logging.DEBUG)
    scraper = WebScraper(use_selenium=config.get('use_selenium', False))
    if args.command == 'train':
        train_df = pd.read_csv(args.train_urls)
        if 'url' not in train_df.columns or 'label' not in train_df.columns:
            logger.error("Input CSV must contain 'url' and 'label' columns.")
            sys.exit(1)
        if len(train_df) < 10:
            logger.warning("Dataset is very small (<10 URLs). Training may be unreliable.")
        detector = BlackHatSEODetector(use_selenium=config.get('use_selenium', False), use_bert=config.get('use_bert', False))
        detector.train_model(train_df['url'].tolist(), train_df['label'].tolist())
        detector.save_model(args.model_output, args.vectorizer_output)
    elif args.command == 'predict':
        with open(args.urls, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        detector = BlackHatSEODetector(args.model, args.vectorizer, config.get('use_selenium', False), config.get('use_bert', False))
        results = detector.predict(urls)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        else:
            for result in results:
                if 'error' in result:
                    logger.error(f"{result['url']}: {result['error']}")
                else:
                    prediction = "Black Hat SEO" if result['prediction'] == 1 else "Clean"
                    logger.info(f"{result['url']}: {prediction} (confidence: {result['probability']*100:.2f}%)")
    elif args.command == 'explain':
        detector = BlackHatSEODetector(args.model, args.vectorizer, config.get('use_selenium', False), config.get('use_bert', False))
        explanation = detector.explain_prediction(args.url)
        logger.info(f"URL: {explanation['url']}")
        logger.info(f"Prediction: {explanation['prediction']}")
        logger.info(f"Confidence: {explanation['confidence']}")
        logger.info("Reasons:")
        for reason in explanation['reasons']:
            logger.info(f"- {reason}")
        if 'top_keywords' in explanation:
            logger.info("Top Keywords:")
            for keyword, count in explanation['top_keywords']:
                logger.info(f"- {keyword}: {count}")
    elif args.command == 'build-dataset':
        builder = DatasetBuilder(scraper, serpapi_key=args.serpapi_key)
        clean_queries = []
        blackhat_queries = []
        if args.clean_queries:
            with open(args.clean_queries, 'r') as f:
                clean_queries = [line.strip() for line in f if line.strip()]
        if args.blackhat_queries:
            with open(args.blackhat_queries, 'r') as f:
                blackhat_queries = [line.strip() for line in f if line.strip()]
        builder.build_dataset(
            clean_queries=clean_queries,
            blackhat_queries=blackhat_queries,
            url_files=args.url_files,
            output_path=args.output,
            threshold=args.threshold
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    if 'ipykernel' in sys.modules:
        test_in_jupyter()
    else:
        main()