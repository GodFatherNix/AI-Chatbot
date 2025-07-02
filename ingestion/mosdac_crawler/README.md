# MOSDAC Web Crawler

A production-ready web crawler for extracting satellite data information from the MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) portal at www.mosdac.gov.in.

## Features

- **Production-Ready**: Comprehensive error handling, rate limiting, and politeness settings
- **Domain-Specific Extraction**: Specialized parsers for MOSDAC's HTML structure and content
- **Content Type Recognition**: Automatic detection of missions, products, documentation, and data access information
- **Document Download**: Handles PDF, DOC, Excel, and PowerPoint document downloads
- **Metadata Enrichment**: Extracts technical specifications, coverage information, and relationships
- **Configurable Settings**: Environment-specific configurations for development and production
- **Comprehensive Logging**: Detailed logging with file and console output
- **Rate Limiting**: Respectful crawling with configurable delays and throttling

## Architecture

```
mosdac_crawler/
├── scrapy.cfg              # Scrapy project configuration
├── mosdac_crawler/
│   ├── __init__.py         # Package initialization
│   ├── settings.py         # Scrapy settings and configurations
│   ├── items.py           # Data structure definitions
│   ├── pipelines.py       # Data processing pipelines
│   ├── middlewares.py     # Request/response middlewares
│   └── spiders/
│       └── mosdac_spider.py # Main spider implementation
├── logs/                  # Log files directory
├── output/               # Crawled data output
├── data/                 # Additional data storage
├── run_spider.py         # Standalone spider runner
├── test_spider.py        # Test suite
└── README.md             # This documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Install Dependencies**:
   ```bash
   pip install scrapy itemadapter twisted
   ```

2. **Create Required Directories**:
   ```bash
   mkdir -p logs output data
   ```

3. **Verify Installation**:
   ```bash
   python test_spider.py
   ```

## Usage

### Basic Usage

1. **Standard Crawl**:
   ```bash
   scrapy crawl mosdac -o output/mosdac_data.json
   ```

2. **Using the Standalone Runner**:
   ```bash
   python run_spider.py --env production --output output/production_crawl.json
   ```

3. **Development Mode** (faster, less polite):
   ```bash
   python run_spider.py --env development --pages 10 --log-level DEBUG
   ```

### Advanced Usage

#### Custom Configuration

```bash
# Limited crawl for testing
python run_spider.py \
  --env test \
  --pages 5 \
  --timeout 300 \
  --output output/test_TIMESTAMP.json \
  --log-level INFO

# Production crawl with custom domains
python run_spider.py \
  --env production \
  --allowed-domains mosdac.gov.in www.mosdac.gov.in \
  --start-urls https://www.mosdac.gov.in/missions \
  --output output/missions_only.json

# Long-running production crawl
python run_spider.py \
  --env production \
  --timeout 3600 \
  --output output/full_crawl_TIMESTAMP.json \
  --log-level WARNING
```

#### Environment Variables

```bash
# Set environment for different behavior
export SCRAPY_ENV=production   # Uses stricter settings
export SCRAPY_ENV=development  # Uses relaxed settings for testing
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--env` | Environment mode (production/development/test) | production |
| `--output`, `-o` | Output file path | output/mosdac_data_TIMESTAMP.json |
| `--pages`, `-p` | Maximum pages to crawl | No limit |
| `--timeout`, `-t` | Maximum crawl time (seconds) | No limit |
| `--log-level` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--start-urls` | Custom start URLs | Default MOSDAC URLs |
| `--allowed-domains` | Allowed domains for crawling | mosdac.gov.in |

## Spider Configuration

### Content Extraction

The spider uses domain-specific selectors and patterns:

#### Mission Detection
- **INSAT-3D**: Weather and atmospheric monitoring
- **OCEANSAT-2**: Ocean color and SST monitoring  
- **SCATSAT-1**: Wind speed and direction measurement
- **MEGHA-TROPIQUES**: Tropical weather monitoring
- **CARTOSAT-2**: High-resolution imaging
- **RESOURCESAT-2**: Natural resource monitoring

#### Product Types
- Ocean Color, Chlorophyll, SST
- Wind Speed/Direction, Precipitation
- Temperature, Humidity, Water Vapor
- Vegetation Index, NDVI, Land Surface Temperature
- Aerosol Optical Depth, Total Ozone

#### Technical Specifications
- Spatial resolution extraction
- Spectral band information
- Swath width and coverage
- Temporal coverage periods
- Data formats (HDF, NetCDF, GeoTIFF)

### Rate Limiting

#### Production Settings
- Download delay: 5 seconds
- Concurrent requests: 1
- Domain-specific delays
- AutoThrottle enabled
- Robots.txt compliance

#### Development Settings  
- Download delay: 1 second
- Concurrent requests: 4
- Robots.txt ignored (for testing)
- Debug logging enabled

## Output Format

### Webpage Items
```json
{
  "url": "https://www.mosdac.gov.in/missions/oceansat2",
  "title": "OCEANSAT-2 Mission Details",
  "content": "OCEANSAT-2 is an Indian satellite...",
  "content_type": "webpage",
  "crawled_at": "2024-01-15T10:30:00Z",
  "mission_info": {
    "mission": "OCEANSAT-2",
    "launch_info": "September 23, 2009",
    "orbit_info": "Sun-synchronous polar orbit",
    "objectives": ["Ocean color monitoring", "SST measurement"]
  },
  "product_info": {
    "products_table": [
      {"name": "Ocean Color", "description": "Chlorophyll-a concentration"},
      {"name": "SST", "description": "Sea Surface Temperature"}
    ],
    "data_formats": ["HDF", "NetCDF"]
  },
  "technical_specs": {
    "resolution": "360m",
    "spectral_bands": "8",
    "swath_width": "1420 km"
  },
  "coverage_info": {
    "regions": ["Indian Ocean", "Global"],
    "coordinate_ranges": [["40°N", "40°S"], ["30°E", "120°E"]]
  },
  "metadata": {
    "word_count": 1247,
    "char_count": 7823,
    "estimated_reading_time": 6,
    "content_category": "mission",
    "language": "en",
    "page_type": "mission",
    "processed_at": "2024-01-15T10:30:15Z"
  }
}
```

### Document Items
```json
{
  "url": "https://www.mosdac.gov.in/docs/oceansat2_manual.pdf",
  "title": "OCEANSAT-2 User Manual",
  "content": "<binary_content>",
  "content_type": "document",
  "file_type": "pdf",
  "source_url": "https://www.mosdac.gov.in/missions/oceansat2",
  "crawled_at": "2024-01-15T10:35:00Z",
  "metadata": {
    "content_type": "application/pdf",
    "content_length": 2048576,
    "file_extension": "pdf",
    "download_timestamp": "2024-01-15T10:35:00Z",
    "is_document": true
  }
}
```

## Monitoring and Troubleshooting

### Log Files

Logs are stored in the `logs/` directory with timestamps:
- `logs/mosdac_spider_20240115_103000.log`
- `logs/test_spider_20240115_103000.log`

### Common Issues

#### 1. Connection Timeouts
```
Solution: Increase DOWNLOAD_TIMEOUT in settings
export SCRAPY_ENV=development  # Uses more relaxed timeouts
```

#### 2. Rate Limiting / 429 Errors
```
Solution: Increase download delays
python run_spider.py --env production  # Uses more conservative settings
```

#### 3. Memory Usage
```
Solution: Enable memory monitoring in settings.py
MEMUSAGE_ENABLED = True
MEMUSAGE_LIMIT_MB = 512
```

#### 4. Robots.txt Blocking
```
Solution: Check MOSDAC's robots.txt
# For testing only:
python run_spider.py --env development  # Ignores robots.txt
```

### Performance Tuning

#### For Large Crawls
```python
# In settings.py or command line
CONCURRENT_REQUESTS = 1           # Very conservative
DOWNLOAD_DELAY = 5               # Respectful timing
AUTOTHROTTLE_ENABLED = True      # Automatic adjustment
HTTPCACHE_ENABLED = True         # Cache responses
```

#### For Development/Testing
```python
CONCURRENT_REQUESTS = 4
DOWNLOAD_DELAY = 1
ROBOTSTXT_OBEY = False
CLOSESPIDER_PAGECOUNT = 10       # Limit pages
```

## Integration

### With Document Processing Pipeline
```python
from mosdac_crawler.spiders.mosdac_spider import MOSDACSpider

# Use extracted content in document processor
spider = MOSDACSpider()
content = spider._clean_text(raw_text)
mission_info = spider._extract_mission_info(response)
```

### With Knowledge Graph Builder
```python
# Import utility functions
from mosdac_crawler.spiders.mosdac_spider import extract_mission_info, extract_product_info

mission_data = extract_mission_info(content)
product_data = extract_product_info(content)
```

## Testing

### Run Test Suite
```bash
python test_spider.py
```

The test suite validates:
- Spider class structure and methods
- Content extraction functionality
- URL validation and filtering
- Text cleaning and normalization
- Limited crawl with timeout

### Manual Testing
```bash
# Test with single URL
scrapy parse --spider=mosdac https://www.mosdac.gov.in/

# Test with custom settings
scrapy crawl mosdac -s DOWNLOAD_DELAY=0.5 -s CLOSESPIDER_PAGECOUNT=3
```

## Deployment

### Production Deployment
```bash
# 1. Install in production environment
pip install scrapy itemadapter twisted

# 2. Set production environment
export SCRAPY_ENV=production

# 3. Run with production settings
python run_spider.py \
  --env production \
  --output /data/mosdac/crawl_$(date +%Y%m%d).json \
  --log-level WARNING

# 4. Set up cron job for regular crawling
0 2 * * 0 /path/to/python /path/to/run_spider.py --env production --output /data/weekly_crawl.json
```

### Docker Deployment
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run_spider.py", "--env", "production"]
```

## Contributing

1. Follow PEP 8 coding standards
2. Add comprehensive logging
3. Include error handling for all network operations
4. Test with limited crawls before full deployment
5. Respect robots.txt and rate limits
6. Document any new extraction patterns

## License

This project is developed for research and educational purposes. Please respect MOSDAC's terms of service and robots.txt when crawling.

---

**Note**: This crawler is designed specifically for the MOSDAC portal structure as of 2024. Website structure changes may require updates to the selectors and extraction logic.