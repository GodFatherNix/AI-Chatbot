#!/usr/bin/env python3
"""Standalone runner for MOSDAC spider with configuration options."""

import os
import sys
import argparse
import logging
from datetime import datetime
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


def setup_logging(log_level='INFO'):
    """Setup logging configuration."""
    log_format = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup file handler
    log_filename = f"logs/mosdac_spider_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_filename


def get_spider_settings(args):
    """Get spider settings based on command line arguments."""
    settings = get_project_settings()
    
    # Override settings based on environment
    if args.env == 'production':
        settings.update({
            'DOWNLOAD_DELAY': 5,
            'CONCURRENT_REQUESTS': 1,
            'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
            'LOG_LEVEL': 'WARNING',
            'ROBOTSTXT_OBEY': True,
        })
    elif args.env == 'development':
        settings.update({
            'DOWNLOAD_DELAY': 1,
            'CONCURRENT_REQUESTS': 4,
            'ROBOTSTXT_OBEY': False,
            'LOG_LEVEL': 'DEBUG',
        })
    
    # Custom output settings
    if args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = args.output.replace('TIMESTAMP', timestamp)
        settings.update({
            'FEEDS': {
                output_file: {
                    'format': 'json',
                    'encoding': 'utf8',
                    'store_empty': False,
                    'indent': 2,
                }
            }
        })
    
    # Custom limits
    if args.pages:
        settings.update({
            'CLOSESPIDER_PAGECOUNT': args.pages
        })
    
    if args.timeout:
        settings.update({
            'CLOSESPIDER_TIMEOUT': args.timeout
        })
    
    return settings


def main():
    """Main function to run the MOSDAC spider."""
    parser = argparse.ArgumentParser(
        description='MOSDAC Web Crawler - Extract satellite data information',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--env', 
        choices=['production', 'development', 'test'],
        default='production',
        help='Environment to run in (affects politeness settings)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='output/mosdac_data_TIMESTAMP.json',
        help='Output file path (use TIMESTAMP for auto timestamp)'
    )
    
    parser.add_argument(
        '--pages', '-p',
        type=int,
        help='Maximum number of pages to crawl'
    )
    
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        help='Maximum time to run (seconds)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--start-urls',
        nargs='+',
        help='Custom start URLs (overrides default ones)'
    )
    
    parser.add_argument(
        '--allowed-domains',
        nargs='+',
        default=['mosdac.gov.in', 'www.mosdac.gov.in'],
        help='Allowed domains for crawling'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("MOSDAC Web Crawler Starting")
    logger.info("="*60)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {args.log_level}")
    
    if args.pages:
        logger.info(f"Page limit: {args.pages}")
    if args.timeout:
        logger.info(f"Timeout: {args.timeout} seconds")
    
    try:
        # Get crawler settings
        settings = get_spider_settings(args)
        
        # Create crawler process
        process = CrawlerProcess(settings)
        
        # Spider arguments
        spider_kwargs = {}
        if args.start_urls:
            spider_kwargs['start_urls'] = args.start_urls
        if args.allowed_domains:
            spider_kwargs['allowed_domains'] = args.allowed_domains
        
        # Add spider to process
        process.crawl('mosdac', **spider_kwargs)
        
        # Start crawling
        logger.info("Starting crawler...")
        process.start()
        
        logger.info("Crawler finished successfully!")
        
    except KeyboardInterrupt:
        logger.info("Crawler interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Crawler failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()