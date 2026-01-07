import random
import re
from urllib.parse import urlparse

class Utils:
    """Utility functions for text processing and URL handling."""
    
    @staticmethod
    def get_random_user_agent():
        """Return a random user agent."""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36 Edg/93.0.961.38',
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

    # Trusted domains - legitimate sites that should not be flagged as black hat
    TRUSTED_DOMAINS = [
        # E-commerce
        'shopee.co.id', 'shopee.com', 'tokopedia.com', 'bukalapak.com', 'lazada.co.id',
        'blibli.com', 'zalora.co.id', 'jd.id', 'bhinneka.com',
        # News & Media
        'kompas.com', 'detik.com', 'tribunnews.com', 'liputan6.com', 'cnnindonesia.com',
        'tempo.co', 'republika.co.id', 'sindonews.com', 'okezone.com', 'tirto.id',
        'kumparan.com', 'suara.com', 'viva.co.id', 'merdeka.com', 'idntimes.com',
        # Tech & Portals
        'google.com', 'google.co.id', 'facebook.com', 'instagram.com', 'twitter.com',
        'youtube.com', 'linkedin.com', 'microsoft.com', 'apple.com', 'amazon.com',
        # Government & Official
        'kemendikbud.go.id', 'kemdikbud.go.id', 'bps.go.id', 'bnpb.go.id',
        # Banking & Finance
        'bca.co.id', 'bni.co.id', 'mandiri.co.id', 'bri.co.id', 'ovo.id', 'gopay.co.id',
        'dana.id', 'linkaja.id',
    ]

    @staticmethod
    def is_trusted_domain(url):
        """Check if URL is from a trusted domain."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix for comparison
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain in Utils.TRUSTED_DOMAINS
        except:
            return False

