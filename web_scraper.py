import requests
import random
import time
import json
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, config_path: str = 'config.json'):
        """Initialize WebScraper with configuration from config.json."""
        self.config = self._load_config(config_path)
        self.timeout = self.config.get('timeout', 10)
        self.max_retries = self.config.get('max_retries', 3)
        self.use_selenium = self.config.get('use_selenium', False)
        self.verbose = self.config.get('verbose', False)
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'
        ]
        if self.use_selenium:
            self._setup_selenium()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {config_path}. Using defaults.")
            return {}

    def _setup_selenium(self):
        """Setup Selenium WebDriver with headless Chrome."""
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'user-agent={random.choice(self.user_agents)}')
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            logger.info("Selenium WebDriver initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {str(e)}")
            self.use_selenium = False

    def _get_with_requests(self, url: str) -> Optional[str]:
        """Fetch URL content using Requests with retries and random User-Agent."""
        for attempt in range(self.max_retries):
            try:
                headers = {
                    'User-Agent': random.choice(self.user_agents),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive'
                }
                response = requests.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch {url}: {str(e)}")
                    return None
                time.sleep(random.uniform(1, 3))

    def _get_with_selenium(self, url: str) -> Optional[str]:
        """Fetch URL content using Selenium."""
        try:
            self.driver.get(url)
            time.sleep(random.uniform(2, 5))  # Wait for JavaScript to load
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Selenium failed for {url}: {str(e)}")
            return None

    def scrape_url(self, url: str) -> Optional[Dict]:
        """Scrape content from a URL and return structured data."""
        logger.info(f"Scraping URL: {url}")
        html_content = None

        # Try Selenium if enabled, else use Requests
        if self.use_selenium:
            html_content = self._get_with_selenium(url)
        if not html_content:
            html_content = self._get_with_requests(url)

        if not html_content:
            logger.warning(f"Skipping {url}: Unable to fetch content")
            return None

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            parsed_url = urlparse(url)

            # Extract title
            title = soup.title.string.strip() if soup.title else ''

            # Extract description
            description = ''
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                description = meta_desc['content'].strip()

            # Extract full text
            full_text = ' '.join(p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'article']))
            full_text = ' '.join(full_text.split())  # Clean whitespace

            # Extract important text (headers and bold)
            important_text = ' '.join(
                tag.get_text(strip=True)
                for tag in soup.find_all(['h1', 'h2', 'h3', 'strong', 'b'])
            )

            # Extract links
            links = [
                {'href': a.get('href', ''), 'text': a.get_text(strip=True)}
                for a in soup.find_all('a', href=True)
            ]

            # Extract image alt texts
            img_alt_texts = [img.get('alt', '') for img in soup.find_all('img') if img.get('alt')]

            # Combine text for analysis
            combined_text = f"{title} {description} {full_text} {important_text} {' '.join(img_alt_texts)}"
            combined_text = ' '.join(combined_text.split())

            return {
                'url': url,
                'domain': parsed_url.netloc,
                'tld': parsed_url.netloc.split('.')[-1],
                'is_gov_domain': parsed_url.netloc.endswith('.gov') or parsed_url.netloc.endswith('.go.id'),
                'is_edu_domain': parsed_url.netloc.endswith('.edu') or parsed_url.netloc.endswith('.ac.id'),
                'is_org_domain': parsed_url.netloc.endswith('.org'),
                'title': title,
                'description': description,
                'full_text': full_text,
                'links': links,
                'headers': important_text,
                'img_alt_texts': img_alt_texts,
                'combined_text': combined_text,
                'important_text': important_text,
                'extracted_at': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
            }
        except Exception as e:
            logger.error(f"Failed to parse {url}: {str(e)}")
            return None

    def scrape_urls(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple URLs and return list of structured data."""
        results = []
        for url in urls:
            result = self.scrape_url(url)
            if result:
                results.append(result)
            time.sleep(random.uniform(0.5, 2))  # Avoid overwhelming servers
        return results

    def __del__(self):
        """Cleanup Selenium WebDriver if initialized."""
        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()
            logger.info("Selenium WebDriver closed.")

if __name__ == '__main__':
    # Example usage
    scraper = WebScraper(config_path='config.json')
    url = 'https://www.bbc.com/indonesia'
    result = scraper.scrape_url(url)
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))