#!/usr/bin/env python3
"""Data items for MOSDAC crawler.

Defines the structure for collected data from MOSDAC portal including
web pages, documents, and extracted metadata.
"""

import scrapy
from scrapy import Field
from datetime import datetime


class MOSDACItem(scrapy.Item):
    """Main item for MOSDAC content."""
    
    # Basic metadata
    url = Field()
    title = Field()
    content = Field()
    content_type = Field()  # 'webpage', 'document'
    crawled_at = Field()
    
    # Document-specific fields
    file_type = Field()  # pdf, doc, docx, etc.
    file_size = Field()
    
    # Relationship fields
    source_url = Field()  # Parent page that linked to this content
    
    # Extracted metadata
    metadata = Field()  # Dict with additional metadata
    
    # Mission and product information
    mission_info = Field()  # Mission-specific details
    product_info = Field()  # Product catalog information
    
    # Processing status
    processed = Field()  # Boolean flag for processing pipeline
    embedding_created = Field()  # Boolean flag for embedding generation
    kg_extracted = Field()  # Boolean flag for KG entity extraction

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default values
        self.setdefault('crawled_at', datetime.utcnow().isoformat())
        self.setdefault('processed', False)
        self.setdefault('embedding_created', False)
        self.setdefault('kg_extracted', False)
        self.setdefault('metadata', {})
        self.setdefault('mission_info', {})
        self.setdefault('product_info', {})


class PageItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    html = scrapy.Field()
    crawled_at = scrapy.Field()


class FileItem(scrapy.Item):
    file_url = scrapy.Field()
    source_page = scrapy.Field()
    crawled_at = scrapy.Field()