#!/usr/bin/env python3
"""Scrapy middlewares for MOSDAC crawler."""

import logging
import random
import time
from typing import Optional
from urllib.parse import urljoin

from scrapy import signals
from scrapy.http import HtmlResponse, Request
from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.exceptions import IgnoreRequest, NotConfigured


class MosdacCrawlerSpiderMiddleware:
    """Spider middleware for MOSDAC-specific processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware instance from crawler."""
        middleware = cls()
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware

    def process_spider_input(self, response, spider):
        """Process response received by spider."""
        # Log successful responses
        if response.status == 200:
            self.logger.debug(f"Successfully crawled: {response.url}")
        
        return None

    def process_spider_output(self, response, result, spider):
        """Process spider output (items and requests)."""
        for item_or_request in result:
            if isinstance(item_or_request, Request):
                # Add custom headers for MOSDAC-specific requests
                item_or_request.headers.setdefault('Referer', response.url)
                
                # Add priority for certain URL patterns
                url = item_or_request.url.lower()
                if any(pattern in url for pattern in ['mission', 'product', 'data']):
                    item_or_request.priority = 10
                elif any(pattern in url for pattern in ['documentation', 'manual']):
                    item_or_request.priority = 5
                    
            yield item_or_request

    def process_spider_exception(self, response, exception, spider):
        """Handle spider exceptions."""
        self.logger.error(f"Spider exception for {response.url}: {exception}")
        return None

    def spider_opened(self, spider):
        """Called when spider is opened."""
        self.logger.info(f"Spider opened: {spider.name}")


class MosdacCrawlerDownloaderMiddleware:
    """Downloader middleware for MOSDAC-specific request handling."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware instance from crawler."""
        middleware = cls()
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware

    def process_request(self, request, spider):
        """Process outgoing requests."""
        self.request_count += 1
        
        # Add custom headers for MOSDAC
        request.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        request.headers['Accept-Language'] = 'en-US,en;q=0.5'
        request.headers['Cache-Control'] = 'no-cache'
        
        # Log every 10th request
        if self.request_count % 10 == 0:
            self.logger.info(f"Processed {self.request_count} requests")
        
        return None

    def process_response(self, request, response, spider):
        """Process response from server."""
        # Handle different response status codes
        if response.status == 200:
            # Check if content is actually HTML for webpage requests
            content_type = response.headers.get('Content-Type', b'').decode().lower()
            
            if 'text/html' in content_type:
                # Validate HTML content
                if len(response.text.strip()) < 100:
                    self.logger.warning(f"Suspiciously short HTML content from {response.url}")
                    
            elif any(ext in response.url.lower() for ext in ['.pdf', '.doc', '.xls']):
                # Document download - check file size
                content_length = len(response.body)
                if content_length < 1024:  # Less than 1KB
                    self.logger.warning(f"Suspiciously small document from {response.url}")
                else:
                    self.logger.info(f"Downloaded document: {response.url} ({content_length} bytes)")
        
        elif response.status in [301, 302, 303, 307, 308]:
            # Handle redirects
            self.logger.info(f"Redirect {response.status} from {response.url}")
            
        elif response.status == 404:
            self.logger.warning(f"Page not found: {response.url}")
            
        elif response.status >= 500:
            self.logger.error(f"Server error {response.status} for {response.url}")
        
        return response

    def process_exception(self, request, exception, spider):
        """Handle request exceptions."""
        self.logger.error(f"Request exception for {request.url}: {exception}")
        return None

    def spider_opened(self, spider):
        """Called when spider is opened."""
        self.logger.info(f"Downloader middleware activated for: {spider.name}")


class PoliteDelayMiddleware:
    """Add additional politeness delays for specific domains."""
    
    def __init__(self, delay_range=(1, 3)):
        self.delay_range = delay_range
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware from crawler settings."""
        delay_range = crawler.settings.getlist('POLITE_DELAY_RANGE', [1, 3])
        return cls(delay_range=tuple(map(int, delay_range)))
    
    def process_request(self, request, spider):
        """Add random delay before request."""
        # Only delay for MOSDAC domain
        if 'mosdac.gov.in' in request.url:
            delay = random.uniform(*self.delay_range)
            self.logger.debug(f"Adding {delay:.2f}s delay before {request.url}")
            time.sleep(delay)
        
        return None


class MOSDACRetryMiddleware(RetryMiddleware):
    """Custom retry middleware for MOSDAC-specific retry logic."""
    
    def __init__(self, settings):
        super().__init__(settings)
        self.logger = logging.getLogger(__name__)
    
    def retry(self, request, reason, spider):
        """Custom retry logic with exponential backoff."""
        retries = request.meta.get('retry_times', 0) + 1
        
        if retries <= self.max_retry_times:
            # Exponential backoff: 1s, 2s, 4s, 8s...
            delay = 2 ** (retries - 1)
            self.logger.info(f"Retrying {request.url} (attempt {retries}/{self.max_retry_times}) after {delay}s delay. Reason: {reason}")
            
            time.sleep(delay)
            
            new_request = request.copy()
            new_request.meta['retry_times'] = retries
            new_request.dont_filter = True
            
            return new_request
        else:
            self.logger.error(f"Giving up on {request.url} after {retries} retries. Reason: {reason}")
            return None


class ContentTypeFilterMiddleware:
    """Filter requests based on content type to avoid non-text content."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # File extensions to allow for document downloads
        self.allowed_doc_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'}
    
    def process_response(self, request, response, spider):
        """Filter responses based on content type."""
        content_type = response.headers.get('Content-Type', b'').decode().lower()
        
        # Allow HTML content
        if 'text/html' in content_type:
            return response
        
        # Allow XML content (sometimes used for sitemaps)
        if 'xml' in content_type:
            return response
        
        # Allow specific document types
        url = request.url.lower()
        if any(url.endswith(ext) for ext in self.allowed_doc_extensions):
            return response
        
        # Block everything else (images, videos, etc.)
        if any(ctype in content_type for ctype in ['image/', 'video/', 'audio/', 'application/octet-stream']):
            self.logger.debug(f"Filtering out {content_type} content from {request.url}")
            raise IgnoreRequest(f"Filtered content type: {content_type}")
        
        return response


class DuplicateRequestsMiddleware:
    """Enhanced duplicate request filtering."""
    
    def __init__(self):
        self.seen_requests = set()
        self.logger = logging.getLogger(__name__)
    
    def process_request(self, request, spider):
        """Check for duplicate requests."""
        # Create a fingerprint of the request
        fingerprint = self._get_request_fingerprint(request)
        
        if fingerprint in self.seen_requests:
            self.logger.debug(f"Dropping duplicate request: {request.url}")
            raise IgnoreRequest("Duplicate request")
        
        self.seen_requests.add(fingerprint)
        return None
    
    def _get_request_fingerprint(self, request):
        """Generate fingerprint for request."""
        # Use URL and method for fingerprint
        return f"{request.method}:{request.url}"