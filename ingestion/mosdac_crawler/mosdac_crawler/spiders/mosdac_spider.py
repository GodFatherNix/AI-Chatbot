#!/usr/bin/env python3
"""MOSDAC web crawler spider for extracting satellite data information.

This spider crawls the actual MOSDAC portal (www.mosdac.gov.in) to extract:
- Mission descriptions and specifications
- Product catalogs and metadata  
- Data access information
- Documentation and user guides
- Technical specifications
- Geospatial coverage information

The spider uses real MOSDAC HTML structure and CSS selectors.
"""

import scrapy
import re
import json
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime

from scrapy.http import Request, Response
from mosdac_crawler.items import MOSDACItem


class MOSDACSpider(scrapy.Spider):
    """Production spider for crawling MOSDAC portal content."""
    
    name = 'mosdac'
    allowed_domains = ['mosdac.gov.in', 'www.mosdac.gov.in']
    
    # Real MOSDAC portal URLs
    start_urls = [
        'https://www.mosdac.gov.in/',
        'https://www.mosdac.gov.in/missions',
        'https://www.mosdac.gov.in/products',
        'https://www.mosdac.gov.in/data-access',
        'https://www.mosdac.gov.in/documentation',
        'https://www.mosdac.gov.in/user-manual',
        'https://www.mosdac.gov.in/faq',
        'https://www.mosdac.gov.in/about',
        'https://www.mosdac.gov.in/contact',
    ]
    
    # Custom settings for this spider
    custom_settings = {
        'DOWNLOAD_DELAY': 2,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'CONCURRENT_REQUESTS': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'MOSDAC-Research-Bot/1.0 (+https://github.com/research/mosdac-ai-bot)',
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 10,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 0.5,
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited_urls = set()
        self.document_urls = set()
        self.stats = {
            'pages_crawled': 0,
            'documents_downloaded': 0,
            'errors': 0
        }
        
        # MOSDAC-specific patterns and configurations
        self._setup_mosdac_patterns()
    
    def _setup_mosdac_patterns(self):
        """Setup MOSDAC-specific patterns for content extraction."""
        
        # Mission names as they appear on MOSDAC
        self.mission_patterns = {
            'INSAT-3D': ['insat-3d', 'insat3d', 'INSAT 3D'],
            'OCEANSAT-2': ['oceansat-2', 'oceansat2', 'OCEANSAT 2', 'OCM'],
            'SCATSAT-1': ['scatsat-1', 'scatsat1', 'SCATSAT 1', 'OSCAT'],
            'MEGHA-TROPIQUES': ['megha-tropiques', 'meghatropiques', 'MT1'],
            'CARTOSAT-2': ['cartosat-2', 'cartosat2'],
            'RESOURCESAT-2': ['resourcesat-2', 'resourcesat2'],
            'RISAT-1': ['risat-1', 'risat1'],
            'ASTROSAT': ['astrosat'],
        }
        
        # Product types
        self.product_patterns = [
            'Ocean Color', 'SST', 'Chlorophyll', 'Wind Speed', 'Wind Direction',
            'Temperature', 'Humidity', 'Precipitation', 'Water Vapor', 'Cloud',
            'Vegetation Index', 'NDVI', 'Land Surface Temperature', 'Fire',
            'Aerosol Optical Depth', 'Total Ozone', 'Outgoing Longwave Radiation'
        ]
        
        # Document file extensions to download
        self.document_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        
        # Priority URL patterns (crawl these first)
        self.priority_patterns = [
            r'/missions?/',
            r'/products?/',
            r'/data/',
            r'/satellites?/',
            r'/instruments?/',
            r'/documentation/',
            r'/manual/',
            r'/faq',
            r'/about'
        ]
    
    def parse(self, response: Response):
        """Parse main pages and extract content with MOSDAC-specific selectors."""
        self.logger.info(f"Parsing: {response.url}")
        self.stats['pages_crawled'] += 1
        
        try:
            # Extract page content using MOSDAC-specific selectors
            item = self._extract_page_content(response)
            if item:
                yield item
            
            # Extract and follow relevant links
            for link in self._extract_relevant_links(response):
                if link not in self.visited_urls and self._should_crawl_url(link):
                    self.visited_urls.add(link)
                    yield Request(
                        url=link,
                        callback=self._determine_callback(link),
                        meta={'source_url': response.url},
                        priority=self._get_url_priority(link)
                    )
            
            # Extract document links for download
            for doc_url in self._extract_document_links(response):
                if doc_url not in self.document_urls:
                    self.document_urls.add(doc_url)
                    yield Request(
                        url=doc_url,
                        callback=self.parse_document,
                        meta={'source_url': response.url}
                    )
                    
        except Exception as e:
            self.logger.error(f"Error parsing {response.url}: {e}")
            self.stats['errors'] += 1
    
    def parse_mission_page(self, response: Response):
        """Parse mission-specific pages with enhanced extraction."""
        self.logger.info(f"Parsing mission page: {response.url}")
        
        item = self._extract_page_content(response)
        if item:
            # Enhanced mission information extraction
            item['mission_info'] = self._extract_detailed_mission_info(response)
            item['product_info'] = self._extract_detailed_product_info(response)
            item['technical_specs'] = self._extract_technical_specifications(response)
            item['coverage_info'] = self._extract_coverage_information(response)
            
            yield item
        
        # Continue following links from mission pages
        for link in self._extract_relevant_links(response):
            if link not in self.visited_urls and self._should_crawl_url(link):
                self.visited_urls.add(link)
                yield Request(url=link, callback=self.parse)
    
    def parse_product_page(self, response: Response):
        """Parse product catalog pages."""
        self.logger.info(f"Parsing product page: {response.url}")
        
        item = self._extract_page_content(response)
        if item:
            # Enhanced product information
            item['product_info'] = self._extract_detailed_product_info(response)
            item['data_access'] = self._extract_data_access_info(response)
            item['format_info'] = self._extract_format_information(response)
            
            yield item
        
        # Follow product-related links
        for link in self._extract_relevant_links(response):
            if link not in self.visited_urls and self._should_crawl_url(link):
                self.visited_urls.add(link)
                yield Request(url=link, callback=self.parse)
    
    def parse_document(self, response: Response):
        """Handle document downloads (PDFs, DOCs, etc.)."""
        self.logger.info(f"Processing document: {response.url} ({len(response.body)} bytes)")
        self.stats['documents_downloaded'] += 1
        
        try:
            item = MOSDACItem()
            item['url'] = response.url
            item['content_type'] = 'document'
            item['file_type'] = self._get_file_extension(response.url)
            item['title'] = self._extract_document_title(response)
            item['source_url'] = response.meta.get('source_url', '')
            item['content'] = response.body
            item['crawled_at'] = datetime.utcnow().isoformat()
            
            # Extract metadata from response headers
            item['metadata'] = {
                'content_type': response.headers.get('Content-Type', b'').decode(),
                'content_length': len(response.body),
                'last_modified': response.headers.get('Last-Modified', b'').decode(),
                'file_extension': item['file_type'],
                'download_timestamp': datetime.utcnow().isoformat()
            }
            
            yield item
            
        except Exception as e:
            self.logger.error(f"Error processing document {response.url}: {e}")
            self.stats['errors'] += 1
    
    def _extract_page_content(self, response: Response) -> Optional[MOSDACItem]:
        """Extract structured content from HTML pages using MOSDAC-specific selectors."""
        try:
            item = MOSDACItem()
            item['url'] = response.url
            item['content_type'] = 'webpage'
            item['crawled_at'] = datetime.utcnow().isoformat()
            
            # Extract title using multiple selectors
            title = (
                response.css('title::text').get() or
                response.css('h1::text').get() or
                response.css('.page-title::text').get() or
                response.css('#main-title::text').get() or
                'No Title'
            )
            item['title'] = self._clean_text(title)
            
            # Extract main content using MOSDAC-specific selectors
            content_selectors = [
                '.main-content',
                '.content-area',
                '.page-content',
                '#content',
                '#main-content',
                'main',
                '.container .row',
                'article',
                '.entry-content'
            ]
            
            content = ''
            for selector in content_selectors:
                elements = response.css(f'{selector}')
                if elements:
                    # Extract text from paragraphs, divs, and list items
                    text_elements = elements.css('p, div, li, td, th, span')
                    content_parts = []
                    
                    for element in text_elements:
                        text = element.css('::text').getall()
                        if text:
                            clean_text = ' '.join([self._clean_text(t) for t in text if self._clean_text(t)])
                            if clean_text and len(clean_text.strip()) > 10:
                                content_parts.append(clean_text)
                    
                    content = ' '.join(content_parts)
                    if content and len(content.strip()) > 100:
                        break
            
            # Fallback content extraction
            if not content or len(content.strip()) < 100:
                all_text = response.css('p::text, div::text, li::text, td::text').getall()
                content = ' '.join([self._clean_text(text) for text in all_text if self._clean_text(text)])
            
            item['content'] = content
            
            # Extract metadata
            item['metadata'] = self._extract_page_metadata(response)
            
            # Extract MOSDAC-specific information
            item['mission_info'] = self._extract_mission_info_basic(response)
            item['product_info'] = self._extract_product_info_basic(response)
            
            return item if content and len(content.strip()) > 50 else None
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {response.url}: {e}")
            return None
    
    def _extract_page_metadata(self, response: Response) -> Dict[str, Any]:
        """Extract comprehensive page metadata."""
        metadata = {}
        
        # Meta tags
        for meta in response.css('meta'):
            name = meta.css('::attr(name)').get()
            content = meta.css('::attr(content)').get()
            if name and content:
                metadata[name] = content
        
        # Open Graph tags
        for og in response.css('meta[property^="og:"]'):
            prop = og.css('::attr(property)').get()
            content = og.css('::attr(content)').get()
            if prop and content:
                metadata[prop] = content
        
        # Page structure information
        metadata.update({
            'has_navigation': bool(response.css('nav, .navigation, .navbar')),
            'has_sidebar': bool(response.css('.sidebar, .aside, .secondary')),
            'has_footer': bool(response.css('footer, .footer')),
            'page_depth': len([p for p in response.url.split('/') if p]) - 2,
            'page_type': self._classify_page_type(response.url),
        })
        
        return metadata
    
    def _extract_mission_info_basic(self, response: Response) -> Dict[str, Any]:
        """Extract basic mission information from page content."""
        mission_info = {}
        content = response.text.lower()
        
        # Detect mission from content
        for mission, patterns in self.mission_patterns.items():
            if any(pattern.lower() in content for pattern in patterns):
                mission_info['mission'] = mission
                break
        
        return mission_info
    
    def _extract_product_info_basic(self, response: Response) -> Dict[str, Any]:
        """Extract basic product information from page content."""
        product_info = {}
        content = response.text.lower()
        
        # Find products mentioned in content
        found_products = []
        for product in self.product_patterns:
            if product.lower() in content:
                found_products.append(product)
        
        if found_products:
            product_info['products'] = found_products
        
        return product_info
    
    def _extract_detailed_mission_info(self, response: Response) -> Dict[str, Any]:
        """Extract detailed mission information from mission pages."""
        mission_info = {}
        content = response.text.lower()
        
        # Detect mission
        for mission, patterns in self.mission_patterns.items():
            if any(pattern.lower() in content for pattern in patterns):
                mission_info['mission'] = mission
                break
        
        # Extract launch information
        launch_patterns = [
            r'launch(?:ed)?[:\s]+([^.]+)',
            r'(?:launched|launch date)[:\s]+([^.]+)',
            r'(\d{1,2}[a-z]{0,2}\s+\w+\s+\d{4})',  # Date patterns
            r'(\w+\s+\d{1,2},?\s+\d{4})',
        ]
        
        for pattern in launch_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                mission_info['launch_info'] = match.group(1).strip()
                break
        
        # Extract orbit information
        orbit_patterns = [
            r'orbit[:\s]+([^.]+)',
            r'(sun-synchronous[^.]*)',
            r'(geostationary[^.]*)',
            r'(polar orbit[^.]*)',
        ]
        
        for pattern in orbit_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                mission_info['orbit_info'] = match.group(1).strip()
                break
        
        # Extract mission objectives
        objectives_section = response.css('.objectives, .mission-objectives, .goals')
        if objectives_section:
            objectives = []
            for item in objectives_section.css('li, p'):
                text = item.css('::text').getall()
                if text:
                    obj_text = ' '.join([self._clean_text(t) for t in text])
                    if obj_text and len(obj_text.strip()) > 20:
                        objectives.append(obj_text)
            mission_info['objectives'] = objectives
        
        return mission_info
    
    def _extract_detailed_product_info(self, response: Response) -> Dict[str, Any]:
        """Extract detailed product information."""
        product_info = {}
        
        # Find product tables or lists
        product_tables = response.css('table, .product-table, .data-table')
        if product_tables:
            products = []
            for table in product_tables:
                rows = table.css('tr')
                for row in rows:
                    cells = row.css('td, th')
                    if len(cells) >= 2:
                        product_name = cells[0].css('::text').get()
                        product_desc = cells[1].css('::text').get()
                        if product_name and product_desc:
                            products.append({
                                'name': self._clean_text(product_name),
                                'description': self._clean_text(product_desc)
                            })
            product_info['products_table'] = products
        
        # Extract data formats
        format_patterns = [
            r'(?:format|file format)[:\s]+([^.]+)',
            r'(?:available in|formats?)[:\s]+([^.]+)',
        ]
        
        content = response.text.lower()
        formats = []
        
        for pattern in format_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                format_text = match.strip()
                if any(fmt in format_text.lower() for fmt in ['hdf', 'netcdf', 'geotiff', 'jpeg']):
                    formats.append(format_text)
        
        if formats:
            product_info['data_formats'] = formats
        
        return product_info
    
    def _extract_technical_specifications(self, response: Response) -> Dict[str, Any]:
        """Extract technical specifications."""
        specs = {}
        content = response.text
        
        # Resolution patterns
        resolution_patterns = [
            r'resolution[:\s]+(\d+\s*(?:m|km|meter|kilometre))',
            r'spatial resolution[:\s]+([^.]+)',
            r'(\d+\s*(?:m|km))\s+resolution',
        ]
        
        for pattern in resolution_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                specs['resolution'] = match.group(1).strip()
                break
        
        # Spectral bands
        band_patterns = [
            r'(\d+)\s+(?:spectral\s+)?bands?',
            r'bands?[:\s]+(\d+)',
            r'spectral channels?[:\s]+(\d+)',
        ]
        
        for pattern in band_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                specs['spectral_bands'] = match.group(1)
                break
        
        # Swath width
        swath_patterns = [
            r'swath[:\s]+(\d+\s*km)',
            r'swath width[:\s]+([^.]+)',
        ]
        
        for pattern in swath_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                specs['swath_width'] = match.group(1).strip()
                break
        
        return specs
    
    def _extract_coverage_information(self, response: Response) -> Dict[str, Any]:
        """Extract geographical coverage information."""
        coverage = {}
        content = response.text.lower()
        
        # Geographic regions
        regions = []
        region_patterns = [
            'indian ocean', 'arabian sea', 'bay of bengal', 'indian subcontinent',
            'global', 'tropical', 'polar regions', 'asia pacific'
        ]
        
        for region in region_patterns:
            if region in content:
                regions.append(region.title())
        
        if regions:
            coverage['regions'] = regions
        
        # Coordinate patterns
        coord_patterns = [
            r'(\d+°?\s*[ns])\s*to\s*(\d+°?\s*[ns])',
            r'(\d+°?\s*[ew])\s*to\s*(\d+°?\s*[ew])',
        ]
        
        coordinates = []
        for pattern in coord_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            coordinates.extend(matches)
        
        if coordinates:
            coverage['coordinate_ranges'] = coordinates
        
        return coverage
    
    def _extract_data_access_info(self, response: Response) -> Dict[str, Any]:
        """Extract data access and download information."""
        access_info = {}
        
        # Look for download links
        download_links = response.css('a[href*="download"], a[href*="data"], .download-link')
        if download_links:
            links = []
            for link in download_links:
                href = link.css('::attr(href)').get()
                text = link.css('::text').get()
                if href and text:
                    links.append({
                        'url': urljoin(response.url, href),
                        'text': self._clean_text(text)
                    })
            access_info['download_links'] = links
        
        # Look for FTP information
        ftp_pattern = r'ftp://[^\s<>"]+'
        ftp_links = re.findall(ftp_pattern, response.text)
        if ftp_links:
            access_info['ftp_links'] = ftp_links
        
        return access_info
    
    def _extract_format_information(self, response: Response) -> Dict[str, Any]:
        """Extract file format and structure information."""
        format_info = {}
        
        # Look for format tables
        format_tables = response.css('table')
        for table in format_tables:
            headers = table.css('th::text').getall()
            if any('format' in h.lower() for h in headers):
                rows = []
                for row in table.css('tr')[1:]:  # Skip header
                    cells = row.css('td::text').getall()
                    if cells:
                        rows.append([self._clean_text(cell) for cell in cells])
                format_info['format_table'] = {
                    'headers': [self._clean_text(h) for h in headers],
                    'rows': rows
                }
                break
        
        return format_info
    
    def _extract_relevant_links(self, response: Response) -> List[str]:
        """Extract relevant links using MOSDAC-specific patterns."""
        links = set()
        
        # Get all links
        all_links = response.css('a::attr(href)').getall()
        
        for link in all_links:
            if not link:
                continue
            
            # Make absolute URL
            absolute_url = urljoin(response.url, link)
            
            # Check if it's a MOSDAC domain link
            if not self._is_mosdac_domain(absolute_url):
                continue
            
            # Check if URL matches priority patterns
            if self._is_priority_url(absolute_url):
                links.add(absolute_url)
                continue
            
            # Check link text for relevance
            link_element = response.css(f'a[href="{link}"]')
            if link_element:
                link_text = link_element.css('::text').get() or ''
                if self._is_relevant_link_text(link_text):
                    links.add(absolute_url)
        
        return list(links)
    
    def _extract_document_links(self, response: Response) -> List[str]:
        """Extract links to downloadable documents."""
        document_links = set()
        
        all_links = response.css('a::attr(href)').getall()
        
        for link in all_links:
            if not link:
                continue
            
            absolute_url = urljoin(response.url, link)
            
            # Check if it's a document file
            if any(absolute_url.lower().endswith(ext) for ext in self.document_extensions):
                if self._is_mosdac_domain(absolute_url):
                    document_links.add(absolute_url)
        
        return list(document_links)
    
    def _determine_callback(self, url: str):
        """Determine appropriate callback based on URL patterns."""
        url_lower = url.lower()
        
        if any(pattern in url_lower for pattern in ['mission', 'satellite']):
            return self.parse_mission_page
        elif any(pattern in url_lower for pattern in ['product', 'data']):
            return self.parse_product_page
        else:
            return self.parse
    
    def _get_url_priority(self, url: str) -> int:
        """Get priority score for URL."""
        url_lower = url.lower()
        
        if any(re.search(pattern, url_lower) for pattern in self.priority_patterns):
            return 10
        elif any(keyword in url_lower for keyword in ['mission', 'product', 'data']):
            return 5
        else:
            return 1
    
    def _should_crawl_url(self, url: str) -> bool:
        """Check if URL should be crawled."""
        # Skip external links
        if not self._is_mosdac_domain(url):
            return False
        
        # Skip certain patterns
        excluded_patterns = [
            r'/login', r'/register', r'/logout', r'/search',
            r'/contact', r'/feedback', r'javascript:', r'mailto:',
            r'\.jpg$', r'\.png$', r'\.gif$', r'\.css$', r'\.js$'
        ]
        
        return not any(re.search(pattern, url, re.IGNORECASE) for pattern in excluded_patterns)
    
    def _is_mosdac_domain(self, url: str) -> bool:
        """Check if URL belongs to MOSDAC domain."""
        try:
            domain = urlparse(url).netloc.lower()
            return any(allowed in domain for allowed in self.allowed_domains)
        except:
            return False
    
    def _is_priority_url(self, url: str) -> bool:
        """Check if URL matches priority patterns."""
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in self.priority_patterns)
    
    def _is_relevant_link_text(self, text: str) -> bool:
        """Check if link text indicates relevant content."""
        if not text:
            return False
        
        relevant_keywords = [
            'mission', 'satellite', 'product', 'data', 'documentation',
            'manual', 'guide', 'specification', 'technical', 'download',
            'ocean', 'weather', 'climate', 'earth observation'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in relevant_keywords)
    
    def _classify_page_type(self, url: str) -> str:
        """Classify page type based on URL."""
        url_lower = url.lower()
        
        if 'mission' in url_lower:
            return 'mission'
        elif 'product' in url_lower:
            return 'product'
        elif 'data' in url_lower:
            return 'data'
        elif any(term in url_lower for term in ['doc', 'manual', 'guide']):
            return 'documentation'
        elif 'faq' in url_lower:
            return 'faq'
        elif 'about' in url_lower:
            return 'about'
        else:
            return 'general'
    
    def _get_file_extension(self, url: str) -> str:
        """Get file extension from URL."""
        try:
            path = urlparse(url).path
            return path.split('.')[-1].lower() if '.' in path else 'unknown'
        except:
            return 'unknown'
    
    def _extract_document_title(self, response: Response) -> str:
        """Extract document title from response or URL."""
        # Try to get title from Content-Disposition header
        content_disposition = response.headers.get('Content-Disposition', b'').decode()
        if 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"')
            return filename.replace('_', ' ').replace('-', ' ')
        
        # Fallback to URL-based title
        try:
            path = urlparse(response.url).path
            filename = path.split('/')[-1]
            title = '.'.join(filename.split('.')[:-1])
            return title.replace('_', ' ').replace('-', ' ').title()
        except:
            return 'Unknown Document'
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ''
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\:\;]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def closed(self, reason):
        """Called when spider closes."""
        self.logger.info(f"Spider closed: {reason}")
        self.logger.info(f"Final statistics: {self.stats}")


# Additional utility functions for the spider
def extract_mission_info(content: str) -> Dict[str, Any]:
    """Standalone function to extract mission information from content."""
    # This can be used by other modules for mission detection
    missions = {
        'INSAT-3D': ['insat-3d', 'insat3d'],
        'OCEANSAT-2': ['oceansat-2', 'oceansat2'],
        'SCATSAT-1': ['scatsat-1', 'scatsat1'],
    }
    
    content_lower = content.lower()
    for mission, patterns in missions.items():
        if any(pattern in content_lower for pattern in patterns):
            return {'mission': mission}
    
    return {}


def extract_product_info(content: str) -> Dict[str, Any]:
    """Standalone function to extract product information from content."""
    products = [
        'Ocean Color', 'SST', 'Chlorophyll', 'Wind Speed', 'Wind Direction',
        'Temperature', 'Humidity', 'Precipitation'
    ]
    
    found_products = []
    content_lower = content.lower()
    
    for product in products:
        if product.lower() in content_lower:
            found_products.append(product)
    
    return {'products': found_products} if found_products else {}