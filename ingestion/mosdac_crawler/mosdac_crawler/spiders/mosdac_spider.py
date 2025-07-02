import re
import scrapy
from urllib.parse import urljoin, urlparse
from datetime import datetime

from mosdac_crawler.items import PageItem, FileItem


class MosdacSpider(scrapy.Spider):
    """Crawl MOSDAC (https://www.mosdac.gov.in) for public HTML pages and downloadable documents."""

    name = "mosdac"
    allowed_domains = ["mosdac.gov.in"]

    # Entry points – FAQ page plus sitemap index pages could be added later
    start_urls = [
        "https://www.mosdac.gov.in/faq",
        "https://www.mosdac.gov.in/product_details",
        "https://www.mosdac.gov.in/satellite-data",
    ]

    # File extensions to treat as downloadable docs
    FILE_EXT_PATTERN = re.compile(r"\.(?:pdf|docx?|xlsx?|zip)$", re.IGNORECASE)

    def parse(self, response: scrapy.http.Response, **kwargs):
        url = response.url
        self.logger.debug("Parsing page %s", url)

        # 1. Yield the page itself
        page_item = PageItem()
        page_item["url"] = url
        page_item["title"] = response.css("title::text").get(default="").strip()
        page_item["html"] = response.text
        page_item["crawled_at"] = datetime.utcnow().isoformat()
        yield page_item

        # 2. Extract and follow links
        for href in response.css("a::attr(href)").getall():
            if not href or href.startswith("javascript:"):
                continue

            abs_url = urljoin(url, href)
            if not self._is_same_domain(abs_url):
                # Skip external links
                continue

            if self.FILE_EXT_PATTERN.search(abs_url):
                # Document link → emit FileItem
                yield FileItem(
                    file_url=abs_url,
                    source_page=url,
                    crawled_at=datetime.utcnow().isoformat(),
                )
            else:
                # Schedule crawling of another HTML page
                yield scrapy.Request(abs_url, callback=self.parse, priority=10)

    @staticmethod
    def _is_same_domain(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.netloc.endswith("mosdac.gov.in")