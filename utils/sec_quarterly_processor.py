#!/usr/bin/env python3
"""
InvestiGator - SEC Quarterly Data Processor Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

SEC Quarterly Data Processor Module
Handles extraction and processing of quarterly financial data from SEC EDGAR APIs
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import gzip
import time

from config import get_config
from utils.cache.cache_manager import get_cache_manager
from utils.cache.cache_types import CacheType
from utils.ticker_cik_mapper import ticker_to_cik
from utils.submission_processor import SubmissionProcessor, Filing
from utils.cache.submission_cache_handler import SubmissionCacheHandler
from utils.sec_frame_api import SECFrameAPI
from utils.db import safe_json_dumps

logger = logging.getLogger(__name__)

class QuarterlyData:
    """Represents quarterly financial data for a company"""
    
    def __init__(self, symbol: str, fiscal_year: int, fiscal_period: str, 
                 form_type: str, filing_date: str, accession_number: str, cik: str):
        self.symbol = symbol
        self.fiscal_year = fiscal_year
        self.fiscal_period = fiscal_period
        self.form_type = form_type
        self.filing_date = filing_date
        self.accession_number = accession_number
        self.cik = cik
        self.financial_data: Dict = {}
        self.metadata: Dict = {}
        
    def add_category_data(self, category: str, data: Dict):
        """Add financial data for a category"""
        self.financial_data[category] = data
        
    def get_period_key(self) -> str:
        """Get standardized period key"""
        return f"{self.fiscal_year}-{self.fiscal_period}"
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        from datetime import date, datetime
        
        # Handle date serialization
        filing_date_str = self.filing_date
        if isinstance(self.filing_date, (date, datetime)):
            filing_date_str = self.filing_date.isoformat()
        
        return {
            'symbol': self.symbol,
            'fiscal_year': self.fiscal_year,
            'fiscal_period': self.fiscal_period,
            'form_type': self.form_type,
            'filing_date': filing_date_str,
            'accession_number': self.accession_number,
            'cik': self.cik,
            'period_key': self.get_period_key(),
            'financial_data': self.financial_data,
            'metadata': self.metadata
        }

class SECQuarterlyProcessor:
    """
    Processes quarterly financial data from SEC EDGAR APIs.
    
    This class handles:
    1. Fetching submissions data
    2. Extracting quarterly periods from company facts
    3. Processing XBRL concepts and tags
    4. Consolidating financial data
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.cache_manager = get_cache_manager()
        self.submission_processor = SubmissionProcessor()
        self.submission_cache = SubmissionCacheHandler()
        self.frame_api = SECFrameAPI()
        
        # Logging setup
        self.main_logger = self.config.get_main_logger('sec_quarterly_processor')
        
    def get_recent_quarterly_data(self, ticker: str) -> List[QuarterlyData]:
        """
        Get recent quarterly data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of QuarterlyData objects
        """
        try:
            # Get CIK for ticker
            cik = ticker_to_cik(ticker)
            if not cik:
                self.main_logger.error(f"No CIK found for ticker {ticker}")
                return []
                
            max_periods = self.config.sec.max_periods_to_analyze
            
            # Check if we should skip submissions and use company facts directly
            if not self.config.sec.require_submissions:
                self.main_logger.info(f"Skipping submissions lookup, using company facts directly for {ticker}")
                return self._get_quarterly_data_from_facts(ticker, cik, max_periods)
                
            # Check for cached submissions
            if self._check_submissions_store(ticker, cik):
                return self._extract_recent_periods(ticker, cik, max_periods)
            
            # Fetch and store submissions
            if self._fetch_and_store_submissions(ticker, cik):
                return self._extract_recent_periods(ticker, cik, max_periods)
            else:
                # Fallback to company facts if submissions unavailable
                self.main_logger.warning(f"Submissions unavailable for {ticker}, falling back to company facts")
                return self._get_quarterly_data_from_facts(ticker, cik, max_periods)
                
        except Exception as e:
            self.main_logger.error(f"Error getting quarterly data for {ticker}: {e}")
            return []
    
    def _check_submissions_store(self, ticker: str, cik: str) -> bool:
        """Check if submissions are available in store"""
        try:
            # Use existing database DAO to check for submissions
            from utils.db import get_all_submission_store_dao
            dao = get_all_submission_store_dao()
            
            # Try to get at least one submission
            submissions = dao.get_recent_earnings_submissions(ticker, limit=1)
            return len(submissions) > 0
            
        except Exception as e:
            self.main_logger.error(f"Error checking submissions store: {e}")
            return False
    
    def _fetch_and_store_submissions(self, ticker: str, cik: str) -> bool:
        """Fetch submissions from SEC and store them"""
        try:
            # This would use the SEC EDGAR API to fetch submissions
            # For now, return False to fallback to company facts
            return False
        except Exception as e:
            self.main_logger.error(f"Error fetching submissions for {ticker}: {e}")
            return False
    
    def _extract_recent_periods(self, ticker: str, cik: str, max_periods: int) -> List[QuarterlyData]:
        """Extract recent periods from cached submissions"""
        try:
            # Use existing database DAO to get recent earnings submissions
            from utils.db import get_all_submission_store_dao
            dao = get_all_submission_store_dao()
            
            # Get recent earnings submissions
            submissions = dao.get_recent_earnings_submissions(ticker, limit=max_periods)
            
            if not submissions:
                self.main_logger.warning(f"No submissions found for {ticker}")
                return []
            
            # Convert to QuarterlyData objects
            quarterly_data = []
            for submission in submissions:
                qd = QuarterlyData(
                    symbol=ticker,
                    fiscal_year=submission.get('fiscal_year', 0) or 2024,  # Default if missing
                    fiscal_period=submission.get('fiscal_period', 'Q1') or 'Q1',  # Default if missing
                    form_type=submission.get('form_type', '10-Q'),
                    filing_date=submission.get('filing_date', ''),
                    accession_number=submission.get('accession_number', ''),
                    cik=cik
                )
                quarterly_data.append(qd)
            
            symbol_logger = self.config.get_symbol_logger(ticker, 'sec_quarterly_processor')
            symbol_logger.info(f"📋 Retrieved {len(quarterly_data)} recent earnings submissions for {ticker} from database")
            
            return quarterly_data
            
        except Exception as e:
            self.main_logger.error(f"Error extracting recent periods: {e}")
            return []
    
    def _get_quarterly_data_from_facts(self, ticker: str, cik: str, max_periods: int) -> List[QuarterlyData]:
        """Extract quarterly data directly from company facts API"""
        try:
            # Get company facts
            facts = self._get_company_facts(ticker, cik)
            if not facts:
                return []
                
            # Extract periods from facts
            periods = self._extract_periods_from_facts(facts, max_periods)
            
            # Convert to QuarterlyData objects
            quarterly_data = []
            for period in periods:
                qd = QuarterlyData(
                    symbol=ticker,
                    fiscal_year=period.get('fiscal_year', 0),
                    fiscal_period=period.get('fiscal_period', ''),
                    form_type=period.get('form_type', '10-Q'),
                    filing_date=period.get('filing_date', ''),
                    accession_number=period.get('accession_number', ''),
                    cik=cik
                )
                quarterly_data.append(qd)
                
            return quarterly_data
            
        except Exception as e:
            self.main_logger.error(f"Error getting quarterly data from facts: {e}")
            return []
    
    def _get_company_facts(self, ticker: str, cik: str) -> Optional[Dict]:
        """Get company facts from cache or SEC API"""
        try:
            # Use existing company facts DAO
            from utils.db import get_all_companyfacts_store_dao
            facts_dao = get_all_companyfacts_store_dao()
            
            # Get company facts from database cache
            facts_result = facts_dao.get_company_facts(ticker)
            
            if facts_result and 'companyfacts' in facts_result:
                symbol_logger = self.config.get_symbol_logger(ticker, 'sec_quarterly_processor')
                symbol_logger.info(f"💾 Using cached Company Facts for {ticker}")
                return facts_result['companyfacts']
            
            self.main_logger.warning(f"No company facts found for {ticker}")
            return None
            
        except Exception as e:
            self.main_logger.error(f"Error getting company facts: {e}")
            return None
    
    def _extract_periods_from_facts(self, facts: Dict, max_periods: int) -> List[Dict]:
        """Extract recent periods from company facts data"""
        try:
            periods = []
            
            # Look for revenue data as a proxy for available periods
            revenue_concepts = [
                'us-gaap:Revenues',
                'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
                'us-gaap:SalesRevenueNet'
            ]
            
            facts_data = facts.get('facts', {}).get('us-gaap', {})
            
            for concept in revenue_concepts:
                if concept in facts_data:
                    units = facts_data[concept].get('units', {})
                    
                    # Look for USD units
                    if 'USD' in units:
                        for entry in units['USD']:
                            fy = entry.get('fy')
                            fp = entry.get('fp')
                            
                            if fy and fp:
                                period = {
                                    'fiscal_year': fy,
                                    'fiscal_period': fp,
                                    'form_type': entry.get('form', '10-Q'),
                                    'filing_date': entry.get('filed', ''),
                                    'accession_number': entry.get('accn', '')
                                }
                                periods.append(period)
                    
                    break  # Use first available revenue concept
            
            # Sort by fiscal year and period, take most recent
            periods.sort(key=lambda x: (x['fiscal_year'], x['fiscal_period']), reverse=True)
            return periods[:max_periods]
            
        except Exception as e:
            self.main_logger.error(f"Error extracting periods from facts: {e}")
            return []
    
    def populate_quarterly_data(self, quarterly_data: List[QuarterlyData]) -> List[QuarterlyData]:
        """
        Populate quarterly data objects with detailed financial information.
        
        Args:
            quarterly_data: List of QuarterlyData objects to populate
            
        Returns:
            List of populated QuarterlyData objects
        """
        for qd in quarterly_data:
            try:
                symbol_logger = self.config.get_symbol_logger(qd.symbol, 'sec_quarterly_processor')
                symbol_logger.info(f"📊 Fetching detailed financial data for {qd.symbol} (CIK: {qd.cik}), FY{qd.fiscal_year} {qd.fiscal_period}")
                
                # Get financial data for this period
                financial_data = self._fetch_period_financial_data(qd)
                qd.financial_data = financial_data
                
                # Save consolidated data
                self._save_consolidated_data(qd)
                
            except Exception as e:
                self.main_logger.error(f"Error populating quarterly data for {qd.symbol}: {e}")
                
        return quarterly_data
    
    def _fetch_period_financial_data(self, qd: QuarterlyData) -> Dict:
        """Fetch detailed financial data for a specific period"""
        try:
            # Get all financial categories from config
            frame_concepts = self.config.sec.frame_api_details
            
            financial_data = {}
            missing_categories = []
            
            # Check for existing cached data first
            for category in frame_concepts.keys():
                cached_data = self._get_cached_category_data(qd, category)
                if cached_data:
                    financial_data[category] = cached_data
                else:
                    missing_categories.append(category)
            
            # Fetch missing categories
            if missing_categories:
                symbol_logger = self.config.get_symbol_logger(qd.symbol, 'sec_quarterly_processor')
                symbol_logger.info(f"📡 Fetching {len(missing_categories)} missing categories from Company Facts")
                
                # Use company facts to get data
                facts_data = self._get_company_facts(qd.symbol, qd.cik)
                if facts_data:
                    for category in missing_categories:
                        category_data = self._extract_category_from_facts(
                            facts_data, category, qd.fiscal_year, qd.fiscal_period
                        )
                        if category_data:
                            financial_data[category] = category_data
                            self._cache_category_data(qd, category, category_data)
            
            return financial_data
            
        except Exception as e:
            self.main_logger.error(f"Error fetching period financial data: {e}")
            return {}
    
    def _get_cached_category_data(self, qd: QuarterlyData, category: str) -> Optional[Dict]:
        """Get cached data for a specific category and period"""
        try:
            cache_key = f"{category}_{qd.get_period_key()}"
            return self.cache_manager.get(
                CacheType.SEC_RESPONSE,
                (qd.symbol, cache_key)
            )
        except Exception as e:
            self.main_logger.error(f"Error getting cached category data: {e}")
            return None
    
    def _cache_category_data(self, qd: QuarterlyData, category: str, data: Dict):
        """Cache data for a specific category and period"""
        try:
            cache_key = f"{category}_{qd.get_period_key()}"
            metadata = {
                'category': category,
                'period': qd.get_period_key(),
                'form_type': qd.form_type,
                'api_url': 'company_facts'
            }
            
            self.cache_manager.set(
                CacheType.SEC_RESPONSE,
                (qd.symbol, cache_key),
                {'data': data, 'metadata': metadata}
            )
        except Exception as e:
            self.main_logger.error(f"Error caching category data: {e}")
    
    def _extract_category_from_facts(self, facts: Dict, category: str, 
                                   fiscal_year: int, fiscal_period: str) -> Optional[Dict]:
        """Extract specific category data from company facts"""
        try:
            # Get concepts for this category
            frame_concepts = self.config.sec.frame_api_details
            if category not in frame_concepts:
                return None
                
            concepts = frame_concepts[category]
            category_data = {
                'concepts': {},
                'metadata': {
                    'category': category,
                    'fiscal_year': fiscal_year,
                    'fiscal_period': fiscal_period,
                    'source': 'company_facts'
                }
            }
            
            facts_data = facts.get('facts', {}).get('us-gaap', {})
            
            # Extract data for each concept in this category
            for concept_name, xbrl_tags in concepts.items():
                concept_data = self._extract_concept_data(
                    facts_data, concept_name, xbrl_tags, fiscal_year, fiscal_period
                )
                category_data['concepts'][concept_name] = concept_data
            
            return category_data
            
        except Exception as e:
            self.main_logger.error(f"Error extracting category from facts: {e}")
            return None
    
    def _extract_concept_data(self, facts_data: Dict, concept_name: str, 
                            xbrl_tags: List[str], fiscal_year: int, fiscal_period: str) -> Dict:
        """Extract data for a specific concept from facts"""
        try:
            for tag in xbrl_tags:
                if tag in facts_data:
                    units = facts_data[tag].get('units', {})
                    
                    # Look for USD units
                    if 'USD' in units:
                        for entry in units['USD']:
                            if (entry.get('fy') == fiscal_year and 
                                entry.get('fp') == fiscal_period):
                                return {
                                    'value': entry.get('val'),
                                    'concept': tag,
                                    'unit': 'USD',
                                    'form': entry.get('form'),
                                    'filed': entry.get('filed'),
                                    'accn': entry.get('accn')
                                }
            
            # If no data found, return missing indicator
            return {
                'value': '',
                'concept': xbrl_tags[0] if xbrl_tags else '',
                'unit': 'USD',
                'missing': True
            }
            
        except Exception as e:
            self.main_logger.error(f"Error extracting concept data: {e}")
            return {'value': '', 'missing': True, 'error': str(e)}
    
    def _save_consolidated_data(self, qd: QuarterlyData):
        """Save consolidated quarterly data to cache"""
        try:
            # Create cache directory
            cache_dir = self.config.get_symbol_cache_path(qd.symbol, 'sec')
            
            # Save as JSON file
            filename = f"{qd.get_period_key()}.json"
            filepath = cache_dir / filename
            
            with open(filepath, 'w') as f:
                # Use safe JSON dumping with date handling
                json_str = safe_json_dumps(qd.to_dict(), indent=2, default=str)
                f.write(json_str)
                
            symbol_logger = self.config.get_symbol_logger(qd.symbol, 'sec_quarterly_processor')
            symbol_logger.info(f"📄 Consolidated {len(qd.financial_data)} categories into {filepath}")
            
        except Exception as e:
            self.main_logger.error(f"Error saving consolidated data: {e}")

def get_quarterly_processor() -> SECQuarterlyProcessor:
    """Get SEC quarterly processor instance"""
    return SECQuarterlyProcessor()