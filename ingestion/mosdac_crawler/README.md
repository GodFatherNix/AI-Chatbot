# MOSDAC Crawler

This Scrapy project collects publicly available content from the [MOSDAC](https://www.mosdac.gov.in) portal and outputs it as JSON-Lines for downstream indexing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt  # relative to ingestion/mosdac_crawler
```

## Run

```bash
scrapy crawl mosdac
```

The crawl will produce a file under `data/` named `mosdac_crawl_<timestamp>.jl` containing two item types:

* `PageItem` – raw HTML pages with metadata.
* `FileItem` – downloadable document links (PDF, DOCX, XLSX, ZIP) and their source page.

## Extending

* Add additional `start_urls` or implement a sitemap parser in `MosdacSpider` to increase coverage.
* Enable Scrapy's `FilesPipeline` in `settings.py` to automatically download linked documents.