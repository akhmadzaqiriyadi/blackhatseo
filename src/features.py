import pandas as pd

class FeatureEngineer:
    """Feature engineering for ML model."""
    
    def __init__(self, text_analyzer):
        self.text_analyzer = text_analyzer

    def _safe_int(self, value):
        """Safely convert value to int, handling NaN and None."""
        try:
            if pd.isna(value):
                return 0
            return int(value)
        except (ValueError, TypeError):
            return 0

    def transform(self, X):
        """Transform data into features with adjusted weights."""
        features = []
        for item in X:
            is_edu = self._safe_int(item.get('is_edu_domain', False))
            judi_count = item.get('judi_keyword_count', 0)
            meta_judi_count = item.get('meta_judi_count', 0)
            # Penalize educational domains less if they don't have gambling keywords
            edu_weight = 1.0 if is_edu and (judi_count == 0 and meta_judi_count == 0) else 0.2
            
            feature_dict = {
                'is_gov_domain': self._safe_int(item.get('is_gov_domain', False)),
                'is_edu_domain': edu_weight * is_edu,
                'word_count': item.get('word_count', 0),
                'judi_keyword_count': judi_count,
                'meta_judi_count': meta_judi_count * 2,
                'spam_keyword_count': item.get('spam_keyword_count', 0),
                'meta_spam_count': item.get('meta_spam_count', 0) * 2,
                'judi_in_title': item.get('judi_in_title', 0) * 3,
                'spam_in_title': item.get('spam_in_title', 0) * 3,
                'suspicious_link_count': item.get('suspicious_link_count', 0),
                'keyword_stuffing_detected': self._safe_int(item.get('keyword_stuffing_detected', False)),
                'hidden_text_suspicion': self._safe_int(item.get('hidden_text_suspicion', False)),
                'is_thin_content': self._safe_int(item.get('is_thin_content', False)),
                'meta_suspicious': self._safe_int(item.get('meta_suspicious', False)) * 3
            }
            features.append(feature_dict)
        return pd.DataFrame(features)
