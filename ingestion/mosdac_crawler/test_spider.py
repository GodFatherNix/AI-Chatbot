#!/usr/bin/env python3
"""Test script for MOSDAC spider to validate functionality."""

import os
import sys
import json
import logging
from datetime import datetime
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from twisted.internet import reactor, defer
from scrapy.utils.project import get_project_settings

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mosdac_crawler.spiders.mosdac_spider import MOSDACSpider


def setup_test_environment():
    """Setup test environment and logging."""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Setup logging
    configure_logging({
        'LOG_LEVEL': 'INFO',
        'LOG_FILE': f'logs/test_spider_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    })
    
    logger = logging.getLogger(__name__)
    logger.info("Test environment setup complete")
    return logger


@defer.inlineCallbacks
def run_test_crawl():
    """Run a limited test crawl of MOSDAC."""
    logger = setup_test_environment()
    
    # Test settings - limited crawl for testing
    settings = get_project_settings()
    settings.update({
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS': 2,
        'ROBOTSTXT_OBEY': True,
        'LOG_LEVEL': 'INFO',
        'CLOSESPIDER_PAGECOUNT': 5,  # Limit to 5 pages for testing
        'CLOSESPIDER_TIMEOUT': 120,  # 2 minutes timeout
        'FEEDS': {
            f'output/test_crawl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json': {
                'format': 'json',
                'encoding': 'utf8',
                'store_empty': False,
                'indent': 2,
            }
        }
    })
    
    logger.info("Starting test crawl...")
    logger.info("This will crawl maximum 5 pages with 2-minute timeout")
    
    try:
        runner = CrawlerRunner(settings)
        
        # Custom start URLs for testing (fewer URLs)
        test_start_urls = [
            'https://www.mosdac.gov.in/',
            'https://www.mosdac.gov.in/missions'
        ]
        
        yield runner.crawl(
            MOSDACSpider,
            start_urls=test_start_urls
        )
        
        logger.info("Test crawl completed successfully!")
        
    except Exception as e:
        logger.error(f"Test crawl failed: {e}", exc_info=True)
        raise
    finally:
        reactor.stop()


def validate_spider_class():
    """Validate spider class and its methods."""
    logger = logging.getLogger(__name__)
    
    logger.info("Validating spider class...")
    
    # Test spider instantiation
    try:
        spider = MOSDACSpider()
        
        # Check essential attributes
        assert hasattr(spider, 'name'), "Spider missing 'name' attribute"
        assert hasattr(spider, 'allowed_domains'), "Spider missing 'allowed_domains' attribute"
        assert hasattr(spider, 'start_urls'), "Spider missing 'start_urls' attribute"
        
        assert spider.name == 'mosdac', f"Expected spider name 'mosdac', got '{spider.name}'"
        assert len(spider.start_urls) > 0, "Spider has no start URLs"
        assert len(spider.allowed_domains) > 0, "Spider has no allowed domains"
        
        # Check essential methods
        essential_methods = [
            'parse', 'parse_mission_page', 'parse_product_page', 'parse_document',
            '_extract_page_content', '_extract_relevant_links', '_extract_document_links'
        ]
        
        for method_name in essential_methods:
            assert hasattr(spider, method_name), f"Spider missing method '{method_name}'"
            method = getattr(spider, method_name)
            assert callable(method), f"'{method_name}' is not callable"
        
        # Test pattern matching
        assert hasattr(spider, 'mission_patterns'), "Spider missing mission patterns"
        assert hasattr(spider, 'product_patterns'), "Spider missing product patterns"
        assert hasattr(spider, 'document_extensions'), "Spider missing document extensions"
        
        logger.info("✓ Spider class validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Spider class validation failed: {e}")
        return False


def test_content_extraction():
    """Test content extraction methods with sample data."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing content extraction methods...")
    
    try:
        spider = MOSDACSpider()
        
        # Test mission detection
        test_content = """
        OCEANSAT-2 is an Indian satellite for ocean color monitoring.
        It carries the Ocean Color Monitor (OCM) instrument.
        The satellite provides Sea Surface Temperature (SST) data.
        """
        
        # Test text cleaning
        clean_text = spider._clean_text("  Test   content  with  extra   spaces  ")
        assert clean_text == "Test content with extra spaces", f"Text cleaning failed: '{clean_text}'"
        
        # Test URL validation
        assert spider._is_mosdac_domain("https://www.mosdac.gov.in/test"), "MOSDAC domain validation failed"
        assert not spider._is_mosdac_domain("https://example.com"), "External domain incorrectly validated"
        
        # Test file extension extraction
        assert spider._get_file_extension("https://test.com/file.pdf") == "pdf", "File extension extraction failed"
        
        logger.info("✓ Content extraction tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Content extraction tests failed: {e}")
        return False


def main():
    """Run all spider tests."""
    logger = setup_test_environment()
    
    logger.info("="*60)
    logger.info("MOSDAC Spider Test Suite")
    logger.info("="*60)
    
    # Run validation tests
    tests_passed = 0
    total_tests = 2
    
    if validate_spider_class():
        tests_passed += 1
    
    if test_content_extraction():
        tests_passed += 1
    
    logger.info(f"\nTest Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("All tests passed! Running limited crawl test...")
        
        # Run actual crawl test
        try:
            reactor.callWhenRunning(run_test_crawl)
            reactor.run()
        except Exception as e:
            logger.error(f"Crawl test failed: {e}")
            sys.exit(1)
    else:
        logger.error("Some tests failed. Skipping crawl test.")
        sys.exit(1)


if __name__ == '__main__':
    main()