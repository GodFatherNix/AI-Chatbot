import os
from datetime import datetime

# Scrapy settings for mosdac_crawler project

BOT_NAME = "mosdac_crawler"

SPIDER_MODULES = ["mosdac_crawler.spiders"]
NEWSPIDER_MODULE = "mosdac_crawler.spiders"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 8

# Configure a delay for requests for the same website (default: 0)
DOWNLOAD_DELAY = 1.0  # politeness, 1 second

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Default request headers
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en",
    "User-Agent": (
        "Mozilla/5.0 (compatible; MOSDACBot/1.0; +https://www.mosdac.gov.in)"
    ),
}

# Enable AutoThrottle extension (disabled by default)
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 10.0

# Feed export configuration
# Items will be exported to `data/mosdac_crawl_<timestamp>.jl`
_FEED_DIRECTORY = os.path.join(os.getcwd(), "data")
os.makedirs(_FEED_DIRECTORY, exist_ok=True)
_CURRENT_TIME = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

FEEDS = {
    os.path.join(_FEED_DIRECTORY, f"mosdac_crawl_{_CURRENT_TIME}.jl"): {
        "format": "jsonlines",
        "encoding": "utf8",
        "store_empty": False,
        "indent": None,
    }
}

# Enable FilesPipeline for document downloads
ITEM_PIPELINES = {
    # "scrapy.pipelines.files.FilesPipeline": 1,
}

# (Optional) configure the files store if you enable FilesPipeline
FILES_STORE = os.path.join(os.getcwd(), "downloaded_files")