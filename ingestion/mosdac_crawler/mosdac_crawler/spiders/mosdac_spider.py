#!/usr/bin/env python3
"""MOSDAC web crawler spider for extracting satellite data information.

This spider crawls the MOSDAC portal to extract:
- Mission descriptions and specifications
- Product catalogs and metadata
- Data access information
- Documentation and user guides
"""

import scrapy
import re
from urllib.parse import urljoin, urlparse
from scrapy.http import Request
from mosdac_crawler.items import MOSDACItem


class MOSDACSpider(scrapy.Spider):
    """Spider for crawling MOSDAC portal content."""
    
    name = 'mosdac'
    allowed_domains = ['mosdac.gov.in']
    start_urls = [
        'https://www.mosdac.gov.in/',
        'https://www.mosdac.gov.in/missions',
        'https://www.mosdac.gov.in/products',
        'https://www.mosdac.gov.in/data-access',
        'https://www.mosdac.gov.in/documentation'
    ]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 2,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'CONCURRENT_REQUESTS': 4,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'MOSDAC-Research-Bot/1.0 (+https://github.com/research/mosdac-bot)'
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited_urls = set()
        self.document_urls = set()
        
    def parse(self, response):
        """Parse main pages and extract content."""
        self.logger.info(f"Parsing: {response.url}")
        
        # Extract page content
        item = self.extract_page_content(response)
        if item:
            yield item
        
        # Find and follow links to missions, products, and documentation
        for link in self.extract_relevant_links(response):
            if link not in self.visited_urls:
                self.visited_urls.add(link)
                yield Request(
                    url=link,
                    callback=self.parse_detail_page,
                    meta={'source_url': response.url}
                )
                
        # Extract document links (PDFs, DOCs)
        for doc_url in self.extract_document_links(response):
            if doc_url not in self.document_urls:
                self.document_urls.add(doc_url)
                yield Request(
                    url=doc_url,
                    callback=self.parse_document,
                    meta={'source_url': response.url}
                )
    
    def parse_detail_page(self, response):
        """Parse mission/product detail pages."""
        self.logger.info(f"Parsing detail page: {response.url}")
        
        item = self.extract_page_content(response)
        if item:
            item['source_url'] = response.meta.get('source_url')
            yield item
            
        # Follow additional links from detail pages
        for link in self.extract_relevant_links(response):
            if link not in self.visited_urls and self.is_mosdac_domain(link):
                self.visited_urls.add(link)
                yield Request(
                    url=link,
                    callback=self.parse_detail_page,
                    meta={'source_url': response.url}
                )
    
    def parse_document(self, response):
        """Handle document downloads (PDFs, DOCs)."""
        self.logger.info(f"Processing document: {response.url}")
        
        item = MOSDACItem()
        item['url'] = response.url
        item['content_type'] = 'document'
        item['file_type'] = self.get_file_extension(response.url)
        item['title'] = self.extract_document_title(response.url)
        item['source_url'] = response.meta.get('source_url')
        item['content'] = response.body  # Raw document content
        
        # Extract metadata from response headers
        content_type = response.headers.get('Content-Type', b'').decode()
        item['metadata'] = {
            'content_type': content_type,
            'content_length': len(response.body),
            'last_modified': response.headers.get('Last-Modified', b'').decode()
        }
        
        yield item
    
    def extract_page_content(self, response):
        """Extract structured content from HTML pages."""
        try:
            item = MOSDACItem()
            item['url'] = response.url
            item['content_type'] = 'webpage'
            
            # Extract title
            title = response.css('title::text').get()
            if not title:
                title = response.css('h1::text').get()
            item['title'] = self.clean_text(title) if title else 'No Title'
            
            # Extract main content
            content_selectors = [
                '.main-content',
                '.content',
                '#content',
                'main',
                '.container',
                'body'
            ]
            
            content = ''
            for selector in content_selectors:
                content_elements = response.css(f'{selector} p, {selector} div, {selector} span')
                if content_elements:
                    content = ' '.join([
                        self.clean_text(text) 
                        for text in content_elements.css('::text').getall()
                        if self.clean_text(text)
                    ])
                    break
            
            if not content:
                # Fallback: extract all text content
                all_text = response.css('p::text, div::text, span::text, td::text').getall()
                content = ' '.join([self.clean_text(text) for text in all_text if self.clean_text(text)])
            
            item['content'] = content
            
            # Extract metadata
            item['metadata'] = self.extract_metadata(response)
            
            # Extract mission/product information if present
            item['mission_info'] = self.extract_mission_info(response)
            item['product_info'] = self.extract_product_info(response)
            
            return item if content and len(content) > 50 else None
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {response.url}: {e}")
            return None
    
    def extract_metadata(self, response):
        """Extract page metadata."""
        metadata = {}
        
        # Meta tags
        for meta in response.css('meta'):
            name = meta.css('::attr(name)').get()
            content = meta.css('::attr(content)').get()
            if name and content:
                metadata[name] = content
                
        # Extract keywords and description
        keywords = response.css('meta[name="keywords"]::attr(content)').get()
        description = response.css('meta[name="description"]::attr(content)').get()
        
        if keywords:
            metadata['keywords'] = [kw.strip() for kw in keywords.split(',')]
        if description:
            metadata['description'] = description
            
        return metadata
    
    def extract_mission_info(self, response):
        """Extract mission-specific information."""
        mission_info = {}
        
        # Look for mission names in content
        missions = ['INSAT-3D', 'OCEANSAT-2', 'SCATSAT-1', 'MEGHA-TROPIQUES', 
                   'CARTOSAT', 'RESOURCESAT', 'RISAT', 'ASTROSAT']
        
        content_text = response.get().lower()
        for mission in missions:
            if mission.lower() in content_text:
                mission_info['mission'] = mission
                
                # Extract mission-specific details
                mission_info['specifications'] = self.extract_specifications(response, mission)
                break
                
        return mission_info
    
    def extract_product_info(self, response):
        """Extract product catalog information."""
        product_info = {}
        
        # Look for product types
        products = ['Ocean Color', 'SST', 'Chlorophyll', 'Wind Speed', 'Wind Direction',
                   'Temperature', 'Humidity', 'Precipitation', 'Water Vapor', 'Cloud']
        
        content_text = response.get().lower()
        found_products = []
        
        for product in products:
            if product.lower() in content_text:
                found_products.append(product)
                
        if found_products:
            product_info['products'] = found_products
            product_info['data_formats'] = self.extract_data_formats(response)
            product_info['access_methods'] = self.extract_access_methods(response)
            
        return product_info
    
    def extract_specifications(self, response, mission):
        """Extract technical specifications for missions."""
        specs = {}
        content = response.get().lower()
        
        # Look for resolution information
        resolution_patterns = [
            r'(\d+)\s*(?:m|meter|metre)',
            r'(\d+)\s*(?:km|kilometer|kilometre)',
            r'resolution.*?(\d+)',
        ]
        
        for pattern in resolution_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                specs['resolution'] = matches[0]
                break
                
        # Look for temporal coverage
        temporal_patterns = [
            r'(\d{4})\s*(?:to|-)?\s*(\d{4})',
            r'since\s*(\d{4})',
            r'from\s*(\d{4})'
        ]
        
        for pattern in temporal_patterns:
            matches = re.findall(pattern, content)
            if matches:
                if len(matches[0]) == 2:
                    specs['temporal_coverage'] = f"{matches[0][0]}-{matches[0][1]}"
                else:
                    specs['temporal_coverage'] = f"{matches[0]}-present"
                break
                
        return specs
    
    def extract_data_formats(self, response):
        """Extract supported data formats."""
        formats = []
        content = response.get().lower()
        
        format_keywords = ['hdf', 'netcdf', 'geotiff', 'jpeg', 'png', 'binary', 'ascii']
        
        for fmt in format_keywords:
            if fmt in content:
                formats.append(fmt.upper())
                
        return formats
    
    def extract_access_methods(self, response):
        """Extract data access methods."""
        methods = []
        content = response.get().lower()
        
        access_keywords = ['ftp', 'http', 'api', 'download', 'online', 'portal']
        
        for method in access_keywords:
            if method in content:
                methods.append(method.upper())
                
        return methods
    
    def extract_relevant_links(self, response):
        """Extract links relevant to MOSDAC content."""
        links = set()
        
        # Priority link patterns
        priority_patterns = [
            r'mission',
            r'product',
            r'data',
            r'satellite',
            r'documentation',
            r'user.*guide',
            r'technical',
            r'specification'
        ]
        
        all_links = response.css('a::attr(href)').getall()
        
        for link in all_links:
            if not link:
                continue
                
            absolute_url = urljoin(response.url, link)
            
            # Check if link matches priority patterns
            link_text = response.css(f'a[href="{link}"]::text').get() or ''
            
            is_priority = any(
                re.search(pattern, link.lower() + ' ' + link_text.lower(), re.IGNORECASE)
                for pattern in priority_patterns
            )
            
            if (is_priority and 
                self.is_mosdac_domain(absolute_url) and 
                not self.is_excluded_url(absolute_url)):
                links.add(absolute_url)
                
        return links
    
    def extract_document_links(self, response):
        """Extract links to documents (PDFs, DOCs, etc.)."""
        document_links = set()
        
        # Document file extensions
        doc_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        
        all_links = response.css('a::attr(href)').getall()
        
        for link in all_links:
            if not link:
                continue
                
            absolute_url = urljoin(response.url, link)
            
            # Check if link points to a document
            if any(absolute_url.lower().endswith(ext) for ext in doc_extensions):
                if self.is_mosdac_domain(absolute_url):
                    document_links.add(absolute_url)
                    
        return document_links
    
    def is_mosdac_domain(self, url):
        """Check if URL belongs to MOSDAC domain."""
        try:
            domain = urlparse(url).netloc.lower()
            return any(allowed in domain for allowed in self.allowed_domains)
        except:
            return False
    
    def is_excluded_url(self, url):
        """Check if URL should be excluded from crawling."""
        excluded_patterns = [
            r'login',
            r'register',
            r'logout',
            r'search',
            r'contact',
            r'feedback',
            r'javascript:',
            r'mailto:',
            r'#'
        ]
        
        return any(re.search(pattern, url.lower()) for pattern in excluded_patterns)
    
    def get_file_extension(self, url):
        """Get file extension from URL."""
        try:
            path = urlparse(url).path
            return path.split('.')[-1].lower() if '.' in path else 'unknown'
        except:
            return 'unknown'
    
    def extract_document_title(self, url):
        """Extract document title from URL."""
        try:
            path = urlparse(url).path
            filename = path.split('/')[-1]
            # Remove extension and clean up
            title = '.'.join(filename.split('.')[:-1])
            return title.replace('_', ' ').replace('-', ' ').title()
        except:
            return 'Unknown Document'
    
    def clean_text(self, text):
        """Clean and normalize text content."""
        if not text:
            return ''
            
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


# Example usage for testing
if __name__ == "__main__":
    # This would be run via scrapy crawl mosdac
    print("MOSDAC Spider ready for crawling")
    print("Run with: scrapy crawl mosdac -o mosdac_data.json")