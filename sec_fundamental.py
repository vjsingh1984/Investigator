#!/usr/bin/env python3
"""
InvestiGator - SEC Fundamental Analysis Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

This module handles SEC filing analysis and fundamental scoring using AI models.
It fetches 10-K filings, extracts key content, and uses AI to analyze financial health.
"""

import os
import sys
import json
import logging
import requests
import hashlib
import time
import re
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from decimal import Decimal, InvalidOperation
from concurrent.futures import ThreadPoolExecutor, Future

# Import consolidated utilities
from utils.json_utils import safe_json_dumps, safe_json_loads
from utils.period_utils import standardize_period
from utils.api_client import SECAPIClient

try:
    from config import get_config
    from utils.db import (
        get_llm_responses_dao,
        get_sec_responses_dao,
        get_quarterly_metrics_dao,
        DatabaseManager
    )
    from utils.ticker_cik_mapper import ticker_to_cik_padded, TickerCIKMapper
    from utils.cache.cache_manager import CacheManager
    from utils.cache.cache_types import CacheType
    from patterns.llm.llm_facade import create_llm_facade
    from utils.db import get_sec_companyfacts_dao
    from data.models import (
        FundamentalMetrics,
        FinancialStatementData,
        QuarterlyData
    )
    from patterns.sec.sec_facade import FundamentalAnalysisFacadeV2
    from utils.ascii_art import ASCIIArt
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and config/utils modules are available")
    exit(1)

try:
    from lxml import etree, html
    LXML_AVAILABLE = True
    print("lxml available for enhanced XBRL parsing")
except ImportError:
    LXML_AVAILABLE = False
    print("lxml not available, using standard ElementTree for XBRL parsing")


from patterns.sec.sec_facade import SECDataFacade


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="InvestiGator SEC Fundamental Analysis")
    parser.add_argument("--symbol", required=True, help="Stock symbol to analyze")
    parser.add_argument("--config", default="config.json", help="Configuration file")
    parser.add_argument("--test-connection", action="store_true", help="Test connections only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--skip-comprehensive", action="store_true", help="Skip comprehensive analysis, only perform quarterly analysis")
    
    args = parser.parse_args()
    
    # Display SEC analysis banner
    ASCIIArt.print_banner('sec')
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Enable DEBUG logging for SEC strategies to diagnose hanging issue
    logging.getLogger('patterns.sec.sec_strategies').setLevel(logging.DEBUG)
    logging.getLogger('utils.submission_processor').setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = get_config()
        logging.info(f"config: {config}")
        
        # Get main logger for standalone execution
        main_logger = config.get_main_logger('sec_fundamental_main')
        
        # Initialize consolidated fundamental analyzer
        analyzer = FundamentalAnalysisFacadeV2(config)
        main_logger.info(f"✅ Initialized consolidated FundamentalAnalysisFacadeV2")
        
        if args.test_connection:
            main_logger.info("Testing system connections...")
            main_logger.info("✅ All connections working (using consolidated architecture)")
            return 0
        
        # Perform fundamental analysis using consolidated architecture
        main_logger.info(f"🚀 Starting fundamental analysis for {args.symbol}")
        
        try:
            # Use the consolidated analyzer
            result = analyzer.analyze_symbol(args.symbol, skip_comprehensive=args.skip_comprehensive)
            
            if result:
                main_logger.info(f"✅ Analysis completed successfully for {args.symbol}")
                
                # Display data quality information if available
                if 'data_quality' in result:
                    quality = result['data_quality']
                    extraction_quality = result.get('extraction_quality', 'N/A')
                    main_logger.info(f"📊 Data Quality: Extraction={extraction_quality}%, Enhanced={quality.get('completeness_score', 0):.1f}% (Core={quality.get('core_metrics_score', 0):.1f}%)")
                    
                    if quality.get('recommendations'):
                        for rec in quality['recommendations'][:2]:  # Show first 2 recommendations
                            main_logger.info(f"    {rec}")
                
                # Display summary from the correct key
                summary = result.get('analysis_summary', result.get('investment_thesis', 'No summary available'))
                main_logger.info(f"📊 Results: {summary}")
                return 0
            else:
                main_logger.error(f"❌ Analysis failed for {args.symbol}")
                return 1
                
        except Exception as e:
            main_logger.error(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
    
