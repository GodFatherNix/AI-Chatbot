import scrapy
from datetime import datetime


class PageItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    html = scrapy.Field()
    crawled_at = scrapy.Field()


class FileItem(scrapy.Item):
    file_url = scrapy.Field()
    source_page = scrapy.Field()
    crawled_at = scrapy.Field()