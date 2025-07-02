#!/usr/bin/env python3
"""Item pipelines for MOSDAC crawler.

Pipelines process items after they are scraped by spiders.
"""

import json
import hashlib
import logging
from typing import Dict, Any, Set
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class ValidationPipeline:
    """Validate scraped items before processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_item(self, item, spider):
        """Validate item has required fields."""
        adapter = ItemAdapter(item)
        
        # Required fields
        required_fields = ['url', 'title', 'content']
        
        for field in required_fields:
            if not adapter.get(field):
                self.logger.warning(f"Missing required field '{field}' in item from {adapter.get('url', 'unknown')}")
                raise DropItem(f"Missing required field: {field}")
        
        # Validate content length
        content = adapter.get('content', '')
        if len(content.strip()) < 50:
            self.logger.warning(f"Content too short ({len(content)} chars) for {adapter.get('url')}")
            raise DropItem("Content too short")
        
        # Validate URL format
        url = adapter.get('url', '')
        if not url.startswith(('http://', 'https://')):
            self.logger.warning(f"Invalid URL format: {url}")
            raise DropItem("Invalid URL format")
        
        self.logger.debug(f"Validated item: {adapter.get('url')}")
        return item


class DeduplicationPipeline:
    """Remove duplicate items based on URL and content hash."""
    
    def __init__(self):
        self.seen_urls: Set[str] = set()
        self.seen_content_hashes: Set[str] = set()
        self.logger = logging.getLogger(__name__)
    
    def process_item(self, item, spider):
        """Check for duplicates and drop if found."""
        adapter = ItemAdapter(item)
        
        url = adapter.get('url', '')
        content = adapter.get('content', '')
        
        # Check URL duplication
        if url in self.seen_urls:
            self.logger.info(f"Dropping duplicate URL: {url}")
            raise DropItem(f"Duplicate URL: {url}")
        
        # Check content duplication
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.seen_content_hashes:
            self.logger.info(f"Dropping duplicate content for: {url}")
            raise DropItem(f"Duplicate content: {url}")
        
        # Mark as seen
        self.seen_urls.add(url)
        self.seen_content_hashes.add(content_hash)
        
        # Add content hash to metadata
        if 'metadata' not in adapter:
            adapter['metadata'] = {}
        adapter['metadata']['content_hash'] = content_hash
        
        self.logger.debug(f"Added unique item: {url}")
        return item


class JsonWriterPipeline:
    """Write items to JSON file with proper formatting."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.items_count = 0
    
    def open_spider(self, spider):
        """Initialize when spider starts."""
        self.logger.info("Started JSON writer pipeline")
    
    def close_spider(self, spider):
        """Cleanup when spider closes."""
        self.logger.info(f"JSON writer pipeline processed {self.items_count} items")
    
    def process_item(self, item, spider):
        """Process item and increment counter."""
        self.items_count += 1
        
        # Add processing timestamp
        adapter = ItemAdapter(item)
        if 'metadata' not in adapter:
            adapter['metadata'] = {}
        
        from datetime import datetime
        adapter['metadata']['processed_at'] = datetime.utcnow().isoformat()
        adapter['metadata']['spider_name'] = spider.name
        
        return item


class ContentEnrichmentPipeline:
    """Enrich items with additional metadata and processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_item(self, item, spider):
        """Enrich item with additional metadata."""
        adapter = ItemAdapter(item)
        
        # Calculate content statistics
        content = adapter.get('content', '')
        word_count = len(content.split())
        char_count = len(content)
        
        # Add content statistics
        if 'metadata' not in adapter:
            adapter['metadata'] = {}
        
        adapter['metadata'].update({
            'word_count': word_count,
            'char_count': char_count,
            'estimated_reading_time': max(1, word_count // 200),  # Minutes
        })
        
        # Classify content type based on URL patterns
        url = adapter.get('url', '').lower()
        if 'mission' in url:
            adapter['metadata']['content_category'] = 'mission'
        elif 'product' in url:
            adapter['metadata']['content_category'] = 'product'
        elif 'data' in url:
            adapter['metadata']['content_category'] = 'data'
        elif 'documentation' in url or 'manual' in url:
            adapter['metadata']['content_category'] = 'documentation'
        else:
            adapter['metadata']['content_category'] = 'general'
        
        # Extract language (basic detection)
        if content:
            # Simple heuristic: if mostly ASCII, likely English
            ascii_ratio = sum(1 for c in content if ord(c) < 128) / len(content)
            adapter['metadata']['language'] = 'en' if ascii_ratio > 0.8 else 'mixed'
        
        self.logger.debug(f"Enriched item: {adapter.get('url')} ({word_count} words)")
        return item


class FileDownloadPipeline:
    """Handle file downloads for documents."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.download_count = 0
    
    def process_item(self, item, spider):
        """Process document downloads."""
        adapter = ItemAdapter(item)
        
        content_type = adapter.get('content_type', '')
        if content_type == 'document':
            self.download_count += 1
            
            # Add download metadata
            if 'metadata' not in adapter:
                adapter['metadata'] = {}
            
            adapter['metadata']['download_order'] = self.download_count
            adapter['metadata']['is_document'] = True
            
            file_type = adapter.get('file_type', 'unknown')
            self.logger.info(f"Processed document download #{self.download_count}: {file_type} from {adapter.get('url')}")
        
        return item