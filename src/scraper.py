import time
import random
import logging
import difflib
import requests
from urllib.parse import urlparse, quote
from datetime import datetime
from bs4 import BeautifulSoup

# Optional Selenium imports
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

if not SELENIUM_AVAILABLE:
    logging.warning("Selenium not available. Dynamic content scraping disabled.")

from .utils import Utils

# Constants for keywords (moved from global scope of original file)
# Note: Keywords are strictly gambling/spam related, avoiding generic terms
JUDI_KEYWORDS = [
    'poker', 'casino', 'togel', 'betting', 'judi', 'taruhan',
    'sportsbook', 'bandar', 'gambling',
    'roulette', 'blackjack', 'baccarat', 'lottery', 'gaple', 'toto',
    'vegas', 'parlay', '4d', '3d', '2d', 'bandarq', 'capsa', 'aduq', 'sakong',
    'ceme', 'domino99', 'agen bola', 'mesin slot', 'pragmatic play', 'habanero',
    'spadegaming', 'microgaming', 'bet88', 'situs judi', 'idn poker', 'live casino',
    'sabung ayam', 'tembak ikan', 'joker123', 'freebet',
    'rupiahtoto', 'slot gacor', 'maxwin', 'rajaslot', 'bandarqq', 'raja toto',
    'situs judi online', 'jackpot online', 'slot online', 'agen bola online',
    'betting online', 'daftar slot', 'bo slot'
]

SPAM_KEYWORDS = [
    'viagra', 'cialis', 'pharmacy', 'pills', 
    'obat kuat', 'pembesar', 'make money', 'weight loss',
    'earn money', 'passive income', 'pelangsing',
    'peninggi badan', 'crypto investment', 'profit harian', 'pinjaman instan',
    'kaya mendadak', 'obat herbal', 'cepat kaya',
    'banting harga', 'gampang menang', 'paling gacor', 'modal receh'
]




# Stopwords fallback
STOPWORDS = []
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        STOPWORDS = list(stopwords.words('indonesian'))
    except:
        STOPWORDS = list(stopwords.words('english'))
except ImportError:
    pass

logger = logging.getLogger(__name__)

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
        max_retries = 2  # Reduced retries
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': Utils.get_random_user_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive'
                }
                # Reduced timeout and verify=False to bypass certificate errors
                response = requests.get(url, headers=headers, timeout=10, verify=False)
                response.raise_for_status()
                return response.text
            except requests.exceptions.SSLError as e:
                logger.warning(f"SSL Error for {url}: {e}")
                return None # Fail fast on SSL error
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Request error for {url}: {e}")
                    return None
                state = random.getstate()
                time.sleep(random.uniform(1, 2)) # Reduced sleep


    def detect_cloaking(self, url):
        """Detect cloaking by comparing content from different user agents."""
        try:
            headers_user = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response_user = requests.get(url, headers=headers_user, timeout=self.timeout)
            headers_bot = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}
            response_bot = requests.get(url, headers=headers_bot, timeout=self.timeout)
            soup_user = BeautifulSoup(response_user.text, 'html.parser')
            soup_bot = BeautifulSoup(response_bot.text, 'html.parser')
            user_text = soup_user.get_text(separator=' ', strip=True)[:1000]
            bot_text = soup_bot.get_text(separator=' ', strip=True)[:1000]
            similarity = difflib.SequenceMatcher(None, user_text, bot_text).ratio()
            return {
                'cloaking_detected': similarity < 0.6,
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
