#!/usr/bin/env python3
"""Scrapy settings for MOSDAC crawler project.

For complete settings documentation see:
https://docs.scrapy.org/en/latest/topics/settings.html
"""

import os
from datetime import datetime

# Scrapy settings for mosdac_crawler project

BOT_NAME = "mosdac_crawler"

SPIDER_MODULES = ["mosdac_crawler.spiders"]
NEWSPIDER_MODULE = "mosdac_crawler.spiders"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure delays for requests (be respectful to MOSDAC servers)
DOWNLOAD_DELAY = 3
RANDOMIZE_DOWNLOAD_DELAY = True
DOWNLOAD_DELAY_SPREAD = 0.5  # 1.5 to 4.5 seconds delay

# The download delay setting will honor only one of:
CONCURRENT_REQUESTS = 2
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en",
    "User-Agent": "MOSDAC-Research-Bot/1.0 (+https://github.com/research/mosdac-bot)",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# Enable or disable spider middlewares
SPIDER_MIDDLEWARES = {
    'mosdac_crawler.middlewares.MosdacCrawlerSpiderMiddleware': 543,
}

# Enable or disable downloader middlewares
DOWNLOADER_MIDDLEWARES = {
    'mosdac_crawler.middlewares.MosdacCrawlerDownloaderMiddleware': 543,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 550,
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
}

# Enable or disable extensions
EXTENSIONS = {
    'scrapy.extensions.telnet.TelnetConsole': None,
    'scrapy.extensions.logstats.LogStats': 300,
}

# Configure item pipelines
ITEM_PIPELINES = {
    'mosdac_crawler.pipelines.ValidationPipeline': 300,
    'mosdac_crawler.pipelines.DeduplicationPipeline': 400,
    'mosdac_crawler.pipelines.JsonWriterPipeline': 800,
}

# Enable AutoThrottle extension and configure it
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 60
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
AUTOTHROTTLE_DEBUG = False  # Enable to see throttling stats

# Enable and configure HTTP caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600  # 1 hour
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [503, 504, 505, 500, 408, 429]

# Retry settings
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Download timeout
DOWNLOAD_TIMEOUT = 180

# Enable memory usage extension
MEMUSAGE_ENABLED = True
MEMUSAGE_LIMIT_MB = 512
MEMUSAGE_WARNING_MB = 256

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/mosdac_crawler.log'

# Custom settings for different environments

# Production settings
if os.getenv('SCRAPY_ENV') == 'production':
    DOWNLOAD_DELAY = 5
    CONCURRENT_REQUESTS = 1
    CONCURRENT_REQUESTS_PER_DOMAIN = 1
    LOG_LEVEL = 'WARNING'
    
# Development settings
elif os.getenv('SCRAPY_ENV') == 'development':
    DOWNLOAD_DELAY = 1
    CONCURRENT_REQUESTS = 4
    ROBOTSTXT_OBEY = False  # For testing
    LOG_LEVEL = 'DEBUG'

# Feed export configuration
# Items will be exported to `data/mosdac_crawl_<timestamp>.jl`
_FEED_DIRECTORY = os.path.join(os.getcwd(), "data")
os.makedirs(_FEED_DIRECTORY, exist_ok=True)
_CURRENT_TIME = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

FEEDS = {
    'output/mosdac_data_%(time)s.json': {
        'format': 'json',
        'encoding': 'utf8',
        'store_empty': False,
        'fields': None,
        'indent': 2,
    },
    'output/mosdac_data_%(time)s.jsonlines': {
        'format': 'jsonlines',
        'encoding': 'utf8',
        'store_empty': False,
    }
}

# Enable FilesPipeline for document downloads
ITEM_PIPELINES = {
    # "scrapy.pipelines.files.FilesPipeline": 1,
}

# (Optional) configure the files store if you enable FilesPipeline
FILES_STORE = os.path.join(os.getcwd(), "downloaded_files")