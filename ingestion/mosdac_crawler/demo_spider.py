#!/usr/bin/env python3
"""Demonstration of MOSDAC spider capabilities with sample content."""

import json
import logging
from datetime import datetime
from io import StringIO
from scrapy.http import HtmlResponse, Request
from mosdac_crawler.spiders.mosdac_spider import MOSDACSpider


def setup_demo_logging():
    """Setup logging for demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def create_sample_mission_page():
    """Create sample HTML content resembling MOSDAC mission page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCEANSAT-2 Mission - MOSDAC</title>
        <meta name="description" content="OCEANSAT-2 satellite mission for ocean color monitoring">
        <meta name="keywords" content="oceansat-2, ocean color, sst, satellite">
    </head>
    <body>
        <div class="main-content">
            <h1>OCEANSAT-2 Mission</h1>
            
            <div class="mission-overview">
                <p>OCEANSAT-2 is an Indian satellite designed for ocean color monitoring and 
                sea surface temperature (SST) measurements. The satellite was launched on 
                September 23, 2009 from Satish Dhawan Space Centre.</p>
                
                <p>The satellite operates in a sun-synchronous polar orbit at an altitude 
                of 720 km. It carries the Ocean Color Monitor (OCM) instrument with 
                8 spectral bands and a spatial resolution of 360m.</p>
            </div>
            
            <div class="mission-objectives">
                <h3>Mission Objectives</h3>
                <ul>
                    <li>Monitor ocean color for chlorophyll-a concentration mapping</li>
                    <li>Measure sea surface temperature with high accuracy</li>
                    <li>Study coastal zone dynamics and water quality</li>
                    <li>Support fisheries and marine ecosystem research</li>
                </ul>
            </div>
            
            <div class="technical-specs">
                <h3>Technical Specifications</h3>
                <table class="data-table">
                    <tr><td>Spatial Resolution</td><td>360m (Ocean Color), 1.4km (SST)</td></tr>
                    <tr><td>Spectral Bands</td><td>8 bands (OCM), 2 bands (TMI)</td></tr>
                    <tr><td>Swath Width</td><td>1420 km</td></tr>
                    <tr><td>Repeat Cycle</td><td>2 days</td></tr>
                </table>
            </div>
            
            <div class="coverage-info">
                <h3>Coverage Information</h3>
                <p>OCEANSAT-2 provides global coverage with emphasis on the Indian Ocean 
                region. The coverage extends from 40°N to 40°S and 30°E to 120°E for 
                regular monitoring.</p>
            </div>
            
            <div class="product-info">
                <h3>Data Products</h3>
                <table class="product-table">
                    <tr><th>Product</th><th>Description</th><th>Format</th></tr>
                    <tr><td>Ocean Color</td><td>Chlorophyll-a concentration</td><td>HDF, NetCDF</td></tr>
                    <tr><td>SST</td><td>Sea Surface Temperature</td><td>HDF, GeoTIFF</td></tr>
                    <tr><td>TSM</td><td>Total Suspended Matter</td><td>HDF</td></tr>
                    <tr><td>CDOM</td><td>Colored Dissolved Organic Matter</td><td>NetCDF</td></tr>
                </table>
            </div>
            
            <div class="download-links">
                <h3>Data Access</h3>
                <p>Data products are available in HDF and NetCDF formats through:</p>
                <ul>
                    <li><a href="/data/oceansat2/download">Direct Download Portal</a></li>
                    <li><a href="ftp://mosdac.gov.in/oceansat2/">FTP Access</a></li>
                    <li><a href="/docs/oceansat2_manual.pdf">User Manual (PDF)</a></li>
                    <li><a href="/docs/oceansat2_quickstart.doc">Quick Start Guide (DOC)</a></li>
                </ul>
            </div>
        </div>
        
        <nav class="navigation">
            <a href="/missions/insat3d">INSAT-3D Mission</a>
            <a href="/missions/scatsat1">SCATSAT-1 Mission</a>
            <a href="/products">Data Products</a>
            <a href="/documentation">Documentation</a>
        </nav>
    </body>
    </html>
    """
    return html_content


def create_sample_product_page():
    """Create sample HTML content for product catalog."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ocean Color Products - MOSDAC</title>
    </head>
    <body>
        <div class="content-area">
            <h1>Ocean Color Data Products</h1>
            
            <p>MOSDAC provides various ocean color products derived from OCEANSAT-2 
            and other satellite missions. These products support marine research, 
            fisheries, and coastal zone management.</p>
            
            <div class="format-info">
                <h3>Available Formats</h3>
                <p>Data is available in HDF, NetCDF, and GeoTIFF formats. 
                All products include comprehensive metadata and quality flags.</p>
                
                <table>
                    <tr><th>Format</th><th>Description</th><th>Applications</th></tr>
                    <tr><td>HDF</td><td>Hierarchical Data Format</td><td>Scientific analysis</td></tr>
                    <tr><td>NetCDF</td><td>Network Common Data Form</td><td>Climate modeling</td></tr>
                    <tr><td>GeoTIFF</td><td>Georeferenced TIFF</td><td>GIS applications</td></tr>
                </table>
            </div>
            
            <div class="data-access">
                <h3>Data Access Methods</h3>
                <ul>
                    <li>Online portal with search functionality</li>
                    <li>FTP server for bulk downloads</li>
                    <li>Web services and APIs</li>
                    <li>Data request system for historical data</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


def demo_content_extraction(logger):
    """Demonstrate content extraction capabilities."""
    logger.info("="*60)
    logger.info("DEMONSTRATING MOSDAC SPIDER CONTENT EXTRACTION")
    logger.info("="*60)
    
    # Create spider instance
    spider = MOSDACSpider()
    logger.info(f"✓ Created spider: {spider.name}")
    logger.info(f"  Allowed domains: {spider.allowed_domains}")
    logger.info(f"  Start URLs: {len(spider.start_urls)} URLs")
    
    # Test 1: Mission page extraction
    logger.info("\n--- TEST 1: Mission Page Extraction ---")
    
    mission_html = create_sample_mission_page()
    mission_response = HtmlResponse(
        url='https://www.mosdac.gov.in/missions/oceansat2',
        body=mission_html.encode('utf-8'),
        encoding='utf-8'
    )
    
    # Extract content using spider methods
    mission_item = spider._extract_page_content(mission_response)
    
    if mission_item:
        logger.info(f"✓ Extracted mission page content:")
        logger.info(f"  Title: {mission_item['title']}")
        logger.info(f"  Content length: {len(mission_item['content'])} characters")
        logger.info(f"  Mission detected: {mission_item.get('mission_info', {}).get('mission', 'None')}")
        
        # Test detailed extraction
        detailed_mission = spider._extract_detailed_mission_info(mission_response)
        detailed_product = spider._extract_detailed_product_info(mission_response)
        tech_specs = spider._extract_technical_specifications(mission_response)
        coverage = spider._extract_coverage_information(mission_response)
        
        logger.info(f"  Launch info: {detailed_mission.get('launch_info', 'Not found')}")
        logger.info(f"  Orbit info: {detailed_mission.get('orbit_info', 'Not found')}")
        logger.info(f"  Products found: {len(detailed_product.get('products_table', []))}")
        logger.info(f"  Resolution: {tech_specs.get('resolution', 'Not found')}")
        logger.info(f"  Spectral bands: {tech_specs.get('spectral_bands', 'Not found')}")
        logger.info(f"  Regions: {coverage.get('regions', [])}")
    else:
        logger.error("✗ Failed to extract mission page content")
    
    # Test 2: Product page extraction
    logger.info("\n--- TEST 2: Product Page Extraction ---")
    
    product_html = create_sample_product_page()
    product_response = HtmlResponse(
        url='https://www.mosdac.gov.in/products/ocean-color',
        body=product_html.encode('utf-8'),
        encoding='utf-8'
    )
    
    product_item = spider._extract_page_content(product_response)
    
    if product_item:
        logger.info(f"✓ Extracted product page content:")
        logger.info(f"  Title: {product_item['title']}")
        logger.info(f"  Content length: {len(product_item['content'])} characters")
        
        # Test format and access info extraction
        format_info = spider._extract_format_information(product_response)
        access_info = spider._extract_data_access_info(product_response)
        
        logger.info(f"  Format table found: {'Yes' if format_info.get('format_table') else 'No'}")
        logger.info(f"  Download links: {len(access_info.get('download_links', []))}")
        logger.info(f"  FTP links: {len(access_info.get('ftp_links', []))}")
    else:
        logger.error("✗ Failed to extract product page content")
    
    # Test 3: Link extraction
    logger.info("\n--- TEST 3: Link Extraction ---")
    
    relevant_links = spider._extract_relevant_links(mission_response)
    document_links = spider._extract_document_links(mission_response)
    
    logger.info(f"✓ Link extraction results:")
    logger.info(f"  Relevant links found: {len(relevant_links)}")
    for link in relevant_links:
        logger.info(f"    - {link}")
    
    logger.info(f"  Document links found: {len(document_links)}")
    for doc_link in document_links:
        logger.info(f"    - {doc_link}")
    
    # Test 4: Utility functions
    logger.info("\n--- TEST 4: Utility Functions ---")
    
    # Test text cleaning
    dirty_text = "  OCEANSAT-2   satellite   provides   ocean   data  "
    clean_text = spider._clean_text(dirty_text)
    logger.info(f"✓ Text cleaning:")
    logger.info(f"  Input: '{dirty_text}'")
    logger.info(f"  Output: '{clean_text}'")
    
    # Test domain validation
    test_urls = [
        "https://www.mosdac.gov.in/missions",
        "https://mosdac.gov.in/products",
        "https://example.com/test",
        "javascript:void(0)"
    ]
    
    logger.info(f"✓ Domain validation:")
    for url in test_urls:
        is_valid = spider._is_mosdac_domain(url)
        should_crawl = spider._should_crawl_url(url)
        logger.info(f"  {url}: domain={is_valid}, crawl={should_crawl}")
    
    # Test priority scoring
    logger.info(f"✓ URL priority scoring:")
    test_priority_urls = [
        "https://www.mosdac.gov.in/missions/oceansat2",
        "https://www.mosdac.gov.in/products/ocean-color",
        "https://www.mosdac.gov.in/documentation/manual",
        "https://www.mosdac.gov.in/contact"
    ]
    
    for url in test_priority_urls:
        priority = spider._get_url_priority(url)
        callback = spider._determine_callback(url).__name__
        logger.info(f"  {url}: priority={priority}, callback={callback}")


def demo_json_output(logger):
    """Demonstrate JSON output format."""
    logger.info("\n--- SAMPLE JSON OUTPUT ---")
    
    # Create sample item data
    sample_item = {
        "url": "https://www.mosdac.gov.in/missions/oceansat2",
        "title": "OCEANSAT-2 Mission - MOSDAC",
        "content": "OCEANSAT-2 is an Indian satellite designed for ocean color monitoring...",
        "content_type": "webpage",
        "crawled_at": datetime.utcnow().isoformat(),
        "mission_info": {
            "mission": "OCEANSAT-2",
            "launch_info": "September 23, 2009",
            "orbit_info": "Sun-synchronous polar orbit",
            "objectives": [
                "Monitor ocean color for chlorophyll-a concentration mapping",
                "Measure sea surface temperature with high accuracy"
            ]
        },
        "product_info": {
            "products_table": [
                {"name": "Ocean Color", "description": "Chlorophyll-a concentration"},
                {"name": "SST", "description": "Sea Surface Temperature"}
            ],
            "data_formats": ["HDF", "NetCDF", "GeoTIFF"]
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
            "word_count": 847,
            "char_count": 5432,
            "estimated_reading_time": 4,
            "content_category": "mission",
            "language": "en",
            "page_type": "mission"
        }
    }
    
    # Pretty print JSON
    json_output = json.dumps(sample_item, indent=2, ensure_ascii=False)
    logger.info("Sample extracted item JSON:")
    print(json_output)


def main():
    """Run the complete demonstration."""
    logger = setup_demo_logging()
    
    logger.info("MOSDAC Spider Demonstration")
    logger.info("This demo shows the spider's content extraction capabilities")
    logger.info("using sample HTML that mimics real MOSDAC portal structure.")
    
    try:
        # Run content extraction demo
        demo_content_extraction(logger)
        
        # Show JSON output format
        demo_json_output(logger)
        
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("The spider is ready for actual crawling with:")
        logger.info("  python run_spider.py --env development --pages 5")
        logger.info("  python test_spider.py")
        logger.info("  scrapy crawl mosdac -o output/test.json")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())