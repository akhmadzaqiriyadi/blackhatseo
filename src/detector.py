import os
import pickle
import logging
import requests
import pandas as pd
import scipy.sparse
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from .utils import Utils
from .scraper import WebScraper
from .analyzer import TextAnalyzer
from .features import FeatureEngineer

logger = logging.getLogger(__name__)

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
        logger.info("\nClassification Report:\n" + classification_report(y_test, ensemble_pred, zero_division=0))

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
                
                # Check if this is a trusted domain - skip all hardcoded overrides
                is_trusted = Utils.is_trusted_domain(url)
                
                # Only apply hardcoded overrides for untrusted domains
                if not is_trusted:
                    if should_override:
                        ensemble_proba = max(0.65, ensemble_proba)
                    # Only boost for meta_judi if count is high (>2) and not a trusted domain
                    if url_data.get('meta_judi_count', 0) > 2:
                        ensemble_proba = max(ensemble_proba, 0.55)
                    # Only boost for cloaking if not a trusted domain (e-commerce/news often have different bot views)
                    if url_data.get('cloaking_detected', False):
                        ensemble_proba = max(ensemble_proba, 0.55)
                
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
