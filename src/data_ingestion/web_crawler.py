"""
Web crawler for MOSDAC portal content extraction.
"""
import requests
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Dict, List, Set, Optional, Generator
import json
from pathlib import Path
import hashlib
from dataclasses import dataclass
import re

from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

@dataclass
class CrawledPage:
    """Represents a crawled web page with metadata."""
    url: str
    title: str
    content: str
    links: List[str]
    metadata: Dict
    content_type: str
    timestamp: str
    hash: str

class MOSDACCrawler:
    """
    Intelligent web crawler for MOSDAC portal.
    Handles both static and dynamic content extraction.
    """
    
    def __init__(self):
        self.base_url = config.portal_base_url
        self.crawl_delay = config.get('portal.crawl_delay', 1)
        self.crawl_depth = config.get('portal.crawl_depth', 3)
        self.user_agent = config.get('portal.user_agent', 'MOSDAC-AI-Bot/1.0')
        self.respect_robots = config.get('portal.respect_robots_txt', True)
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Selenium setup for dynamic content
        self.driver = None
        self._setup_selenium()
        
        # Crawl state
        self.visited_urls: Set[str] = set()
        self.crawled_pages: List[CrawledPage] = []
        self.robots_parser = None
        
        # Initialize robots.txt parser
        if self.respect_robots:
            self._load_robots_txt()
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver for dynamic content."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'--user-agent={self.user_agent}')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Selenium: {e}")
            self.driver = None
    
    def _load_robots_txt(self):
        """Load and parse robots.txt file."""
        try:
            robots_url = urljoin(self.base_url, '/robots.txt')
            self.robots_parser = urllib.robotparser.RobotFileParser()
            self.robots_parser.set_url(robots_url)
            self.robots_parser.read()
            logger.info(f"Loaded robots.txt from {robots_url}")
        except Exception as e:
            logger.warning(f"Could not load robots.txt: {e}")
            self.robots_parser = None
    
    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        if not self.robots_parser:
            return True
        return self.robots_parser.can_fetch(self.user_agent, url)
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistent processing."""
        # Remove fragment identifier
        url = url.split('#')[0]
        # Remove common tracking parameters
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        # Remove tracking parameters
        tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'fbclid', 'gclid']
        for param in tracking_params:
            query_params.pop(param, None)
        
        # Reconstruct URL
        from urllib.parse import urlencode, urlunparse
        new_query = urlencode(query_params, doseq=True)
        normalized = urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, new_query, ''
        ))
        
        return normalized
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract metadata from page."""
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc,
        }
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            name = tag.get('name') or tag.get('property')
            content = tag.get('content')
            if name and content:
                metadata[f"meta_{name}"] = content
        
        # Page structure
        metadata['headings'] = {
            f'h{i}': [h.get_text().strip() for h in soup.find_all(f'h{i}')]
            for i in range(1, 7)
        }
        
        # Links
        links = soup.find_all('a', href=True)
        metadata['internal_links'] = len([l for l in links if self.base_url in l['href']])
        metadata['external_links'] = len([l for l in links if self.base_url not in l['href']])
        
        # Images
        images = soup.find_all('img')
        metadata['image_count'] = len(images)
        metadata['images_with_alt'] = len([img for img in images if img.get('alt')])
        
        # Tables
        tables = soup.find_all('table')
        metadata['table_count'] = len(tables)
        
        # Forms
        forms = soup.find_all('form')
        metadata['form_count'] = len(forms)
        
        return metadata
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page, removing navigation and boilerplate."""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Try to find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|body'))
        
        if main_content:
            content = main_content.get_text()
        else:
            content = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        content = ' '.join(chunk for chunk in chunks if chunk)
        
        return content
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all valid links from page."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Filter valid links
            if (full_url.startswith(self.base_url) and 
                not full_url.endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx')) and
                '#' not in full_url.split('/')[-1]):
                links.append(self._normalize_url(full_url))
        
        return list(set(links))  # Remove duplicates
    
    def _crawl_page_static(self, url: str) -> Optional[CrawledPage]:
        """Crawl a single page using requests."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract page data
            title = soup.find('title')
            title = title.get_text().strip() if title else url.split('/')[-1]
            
            content = self._extract_content(soup)
            links = self._extract_links(soup, url)
            metadata = self._extract_metadata(soup, url)
            
            # Create hash for deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            page = CrawledPage(
                url=url,
                title=title,
                content=content,
                links=links,
                metadata=metadata,
                content_type='html',
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                hash=content_hash
            )
            
            logger.info(f"Successfully crawled: {url}")
            return page
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return None
    
    def _crawl_page_dynamic(self, url: str) -> Optional[CrawledPage]:
        """Crawl a single page using Selenium for dynamic content."""
        if not self.driver:
            return self._crawl_page_static(url)
        
        try:
            self.driver.get(url)
            
            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Handle dynamic content loading
            time.sleep(2)
            
            # Get page source and parse
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Extract page data
            title = self.driver.title or url.split('/')[-1]
            content = self._extract_content(soup)
            links = self._extract_links(soup, url)
            metadata = self._extract_metadata(soup, url)
            
            # Create hash for deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            page = CrawledPage(
                url=url,
                title=title,
                content=content,
                links=links,
                metadata=metadata,
                content_type='html_dynamic',
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                hash=content_hash
            )
            
            logger.info(f"Successfully crawled (dynamic): {url}")
            return page
            
        except Exception as e:
            logger.error(f"Error crawling {url} with Selenium: {e}")
            return None
    
    def crawl_url(self, url: str, use_selenium: bool = False) -> Optional[CrawledPage]:
        """Crawl a single URL."""
        # Check robots.txt
        if not self._can_fetch(url):
            logger.warning(f"Robots.txt disallows crawling: {url}")
            return None
        
        # Check if already visited
        normalized_url = self._normalize_url(url)
        if normalized_url in self.visited_urls:
            return None
        
        self.visited_urls.add(normalized_url)
        
        # Rate limiting
        time.sleep(self.crawl_delay)
        
        # Crawl page
        if use_selenium:
            page = self._crawl_page_dynamic(url)
        else:
            page = self._crawl_page_static(url)
        
        if page:
            self.crawled_pages.append(page)
        
        return page
    
    def crawl_portal(self, start_urls: Optional[List[str]] = None) -> Generator[CrawledPage, None, None]:
        """
        Crawl the entire MOSDAC portal using BFS approach.
        
        Args:
            start_urls: Starting URLs. Uses configured sections if None.
            
        Yields:
            CrawledPage objects as they are processed
        """
        if start_urls is None:
            sections = config.get('portal.sections', ['/'])
            start_urls = [urljoin(self.base_url, section) for section in sections]
        
        # Initialize queue with starting URLs
        url_queue = [(url, 0) for url in start_urls]  # (url, depth)
        
        logger.info(f"Starting crawl with {len(start_urls)} URLs")
        
        while url_queue:
            current_url, depth = url_queue.pop(0)
            
            if depth > self.crawl_depth:
                continue
            
            # Determine if we need Selenium
            use_selenium = any(keyword in current_url.lower() 
                             for keyword in ['search', 'map', 'interactive', 'dynamic'])
            
            page = self.crawl_url(current_url, use_selenium=use_selenium)
            
            if page:
                yield page
                
                # Add new links to queue
                if depth < self.crawl_depth:
                    for link in page.links:
                        if link not in self.visited_urls:
                            url_queue.append((link, depth + 1))
        
        logger.info(f"Crawl completed. Processed {len(self.crawled_pages)} pages")
    
    def save_crawled_data(self, output_dir: str = "data/raw/crawled"):
        """Save crawled data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual pages
        for i, page in enumerate(self.crawled_pages):
            filename = f"page_{i:04d}_{hashlib.md5(page.url.encode()).hexdigest()[:8]}.json"
            file_path = output_path / filename
            
            page_data = {
                'url': page.url,
                'title': page.title,
                'content': page.content,
                'links': page.links,
                'metadata': page.metadata,
                'content_type': page.content_type,
                'timestamp': page.timestamp,
                'hash': page.hash
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(page_data, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary = {
            'total_pages': len(self.crawled_pages),
            'total_urls_visited': len(self.visited_urls),
            'crawl_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'base_url': self.base_url,
            'crawl_config': {
                'depth': self.crawl_depth,
                'delay': self.crawl_delay,
                'user_agent': self.user_agent
            }
        }
        
        with open(output_path / 'crawl_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved {len(self.crawled_pages)} crawled pages to {output_path}")
    
    def __del__(self):
        """Cleanup resources."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass