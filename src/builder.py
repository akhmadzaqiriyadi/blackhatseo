import time
import random
import logging
import requests
import pandas as pd
from urllib.parse import quote
from tqdm import tqdm
from .utils import Utils

logger = logging.getLogger(__name__)

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
            # Need to import BeautifulSoup locally or pass it in if we want to avoid circular imports? 
            # Actually WebScraper is not imported here, but BeautifulSoup is needed.
            from bs4 import BeautifulSoup
            
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

    def label_urls(self, urls, output_path=None, threshold=5):
        """Semi-automatically label URLs based on SEO metrics."""
        labeled_data = []
        import os # Ensure os is imported
        
        # Check if file exists to determine header
        file_exists = os.path.exists(output_path) if output_path else False
        
        for i, url in enumerate(tqdm(urls, desc="Labeling URLs")):
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
                
                item = {'url': url, 'label': label}
                labeled_data.append(item)
                logger.debug(f"Labeled {url}: {'Black Hat' if label == 1 else 'Clean'}")

                # Save incrementally every 10 items or if it's the last one
                if output_path and (len(labeled_data) >= 10 or i == len(urls) - 1):
                    df = pd.DataFrame(labeled_data)
                    df.to_csv(output_path, mode='a', header=not file_exists, index=False)
                    file_exists = True # Header written
                    labeled_data = [] # Clear buffer

            except Exception as e:
                logger.error(f"Error labeling {url}: {e}")
        
        # Save remaining
        if output_path and labeled_data:
             df = pd.DataFrame(labeled_data)
             df.to_csv(output_path, mode='a', header=not file_exists, index=False)
        
        return [] # Return empty list as we saved to file directly

    def save_dataset(self, data, output_path):
        """Save labeled data to CSV."""
        # This method is now redundant for build_dataset but kept for compatibility
        try:
            if not data: return # Nothing to save
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
        
        # Clear the output file first to avoid appending to old runs if starting fresh
        # But maybe user wants to append? 
        # For now, let's assume if they run build-dataset, they might want a fresh start or we can rename.
        # But safely, let's just use what's passed. If user wants overwrite, they should delete file manually or we force it.
        # Given the "stop and resume" scenario, appending is better?
        # But if we stopped halfway, we might have partial data?
        # Actually, "build-dataset" usually implies creating from scratch. 
        # But to be safe, let's overwrite if it exists at the START of this function, then append in label_urls.
        # NOTE: If we want to support resume, we shouldn't overwrite.
        # The user said "stop and fix", implying a restart but maybe reusing data?
        # Let's overwrite for now to keep it clean, as we collected new URLs.
        import os
        if os.path.exists(output_path):
             try:
                 os.remove(output_path)
             except:
                 pass

        self.label_urls(list(all_urls), output_path=output_path, threshold=threshold)
        logger.info(f"Dataset built and saved to {output_path}")
