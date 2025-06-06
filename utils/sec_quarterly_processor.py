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
# Now using cache_manager interface for all cache operations
from utils.sec_frame_api import SECFrameAPI
from utils.db import safe_json_dumps
from data.models import QuarterlyData, FinancialStatementData

logger = logging.getLogger(__name__)

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
        # Using cache_manager interface for all cache operations
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
        """Check if submissions are available using cache manager interface"""
        try:
            # Use cache manager's exists method to check for submission data
            cache_key = {
                'symbol': ticker,
                'cik': cik
            }
            
            # Check if submission data exists in any cache layer (disk -> RDBMS)
            return self.cache_manager.exists(CacheType.SUBMISSION_DATA, cache_key)
            
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
        """Extract recent periods using cache manager interface"""
        try:
            # Use cache manager's get method to retrieve submission data
            cache_key = {
                'symbol': ticker,
                'cik': cik
            }
            
            # Get submission data via cache manager (handles disk -> RDBMS priority)
            submission_result = self.cache_manager.get(CacheType.SUBMISSION_DATA, cache_key)
            
            if not submission_result:
                self.main_logger.warning(f"No submissions found for {ticker}")
                return []
            
            # Extract submissions_data from the cached result
            submissions_json = submission_result.get('submissions_data', {})
            if not submissions_json:
                self.main_logger.warning(f"No submission data in cache for {ticker}")
                return []
            
            # Process the submission JSON to get recent earnings filings
            recent_filings = self.submission_processor.get_recent_earnings_filings(
                submissions_json, limit=max_periods
            )
            
            if not recent_filings:
                self.main_logger.warning(f"No recent filings found for {ticker}")
                return []
            
            # Convert Filing objects to QuarterlyData objects
            quarterly_data = []
            for filing in recent_filings:
                # Create empty FinancialStatementData
                financial_data = FinancialStatementData(
                    symbol=ticker,
                    period=f"{filing.fiscal_year or 2024}-{filing.fiscal_period or 'Q1'}"
                )
                
                qd = QuarterlyData(
                    symbol=ticker,
                    cik=cik,
                    fiscal_year=filing.fiscal_year or 2024,
                    fiscal_period=filing.fiscal_period or 'Q1',
                    form_type=filing.form_type,
                    financial_data=financial_data
                )
                # Set optional fields from Filing object
                qd.filing_date = filing.filing_date
                qd.accession_number = filing.accession_number
                qd.report_date = filing.report_date
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
                # Create empty FinancialStatementData
                financial_data = FinancialStatementData(
                    symbol=ticker,
                    period=f"{period.get('fiscal_year', 0)}-{period.get('fiscal_period', '')}"
                )
                
                qd = QuarterlyData(
                    symbol=ticker,
                    cik=cik,
                    fiscal_year=period.get('fiscal_year', 0),
                    fiscal_period=period.get('fiscal_period', ''),
                    form_type=period.get('form_type', '10-Q'),
                    financial_data=financial_data
                )
                # Set optional fields
                qd.filing_date = period.get('filing_date', '')
                qd.accession_number = period.get('accession_number', '')
                quarterly_data.append(qd)
                
            return quarterly_data
            
        except Exception as e:
            self.main_logger.error(f"Error getting quarterly data from facts: {e}")
            return []
    
    def _get_company_facts(self, ticker: str, cik: str) -> Optional[Dict]:
        """Get company facts from cache or SEC API"""
        try:
            # Use existing company facts DAO
            from utils.db import get_sec_companyfacts_dao
            facts_dao = get_sec_companyfacts_dao()
            
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
            frame_concepts = self.config.sec.frame_api_concepts
            if category not in frame_concepts:
                return None
                
            concepts = frame_concepts[category]
            category_data = {
                'concepts': {},
                'calculated_metrics': {},  # For derived calculations like EPS
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
            
            # Calculate derived metrics for comprehensive analysis
            if category == 'income_statement':
                category_data['calculated_metrics'] = self._calculate_income_statement_metrics(
                    category_data['concepts'], facts_data, fiscal_year, fiscal_period
                )
            elif category == 'balance_sheet':
                category_data['calculated_metrics'] = self._calculate_balance_sheet_metrics(
                    category_data['concepts'], facts_data, fiscal_year, fiscal_period
                )
            
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
    
    def _calculate_income_statement_metrics(self, concepts: Dict, facts_data: Dict, 
                                          fiscal_year: int, fiscal_period: str) -> Dict:
        """Calculate comprehensive income statement metrics including EPS"""
        try:
            calculated = {}
            
            # Get key values
            revenue = self._get_concept_value(concepts, 'revenues')
            net_income = self._get_concept_value(concepts, 'net_income')
            gross_profit = self._get_concept_value(concepts, 'gross_profit')
            operating_income = self._get_concept_value(concepts, 'operating_income')
            
            # Get shares outstanding data
            shares_basic = self._extract_shares_data(facts_data, 'basic', fiscal_year, fiscal_period)
            shares_diluted = self._extract_shares_data(facts_data, 'diluted', fiscal_year, fiscal_period)
            
            # Calculate EPS if we have net income and shares
            if net_income and shares_basic:
                calculated['eps_basic'] = {
                    'value': round(net_income / shares_basic, 2),
                    'calculation': f"{net_income} / {shares_basic}",
                    'components': {
                        'net_income': net_income,
                        'shares_basic': shares_basic
                    }
                }
            
            if net_income and shares_diluted:
                calculated['eps_diluted'] = {
                    'value': round(net_income / shares_diluted, 2),
                    'calculation': f"{net_income} / {shares_diluted}",
                    'components': {
                        'net_income': net_income,
                        'shares_diluted': shares_diluted
                    }
                }
            
            # Calculate margins
            if revenue:
                if gross_profit:
                    calculated['gross_margin'] = {
                        'value': round((gross_profit / revenue) * 100, 2),
                        'calculation': f"({gross_profit} / {revenue}) * 100",
                        'unit': 'percentage'
                    }
                
                if operating_income:
                    calculated['operating_margin'] = {
                        'value': round((operating_income / revenue) * 100, 2),
                        'calculation': f"({operating_income} / {revenue}) * 100",
                        'unit': 'percentage'
                    }
                
                if net_income:
                    calculated['net_margin'] = {
                        'value': round((net_income / revenue) * 100, 2),
                        'calculation': f"({net_income} / {revenue}) * 100",
                        'unit': 'percentage'
                    }
            
            # Add shares outstanding for transparency
            if shares_basic:
                calculated['shares_outstanding_basic'] = {
                    'value': shares_basic,
                    'unit': 'shares',
                    'source': 'xbrl_extraction'
                }
            
            if shares_diluted:
                calculated['shares_outstanding_diluted'] = {
                    'value': shares_diluted,
                    'unit': 'shares',
                    'source': 'xbrl_extraction'
                }
            
            return calculated
            
        except Exception as e:
            self.main_logger.error(f"Error calculating income statement metrics: {e}")
            return {}
    
    def _calculate_balance_sheet_metrics(self, concepts: Dict, facts_data: Dict,
                                       fiscal_year: int, fiscal_period: str) -> Dict:
        """Calculate comprehensive balance sheet metrics"""
        try:
            calculated = {}
            
            # Get key values
            total_assets = self._get_concept_value(concepts, 'total_assets')
            current_assets = self._get_concept_value(concepts, 'current_assets')
            current_liabilities = self._get_concept_value(concepts, 'current_liabilities')
            total_liabilities = self._get_concept_value(concepts, 'total_liabilities')
            shareholders_equity = self._get_concept_value(concepts, 'shareholders_equity')
            cash = self._get_concept_value(concepts, 'cash_and_equivalents')
            
            # Calculate ratios
            if current_assets and current_liabilities:
                calculated['current_ratio'] = {
                    'value': round(current_assets / current_liabilities, 2),
                    'calculation': f"{current_assets} / {current_liabilities}",
                    'components': {
                        'current_assets': current_assets,
                        'current_liabilities': current_liabilities
                    }
                }
            
            if total_liabilities and shareholders_equity:
                calculated['debt_to_equity'] = {
                    'value': round(total_liabilities / shareholders_equity, 2),
                    'calculation': f"{total_liabilities} / {shareholders_equity}",
                    'components': {
                        'total_liabilities': total_liabilities,
                        'shareholders_equity': shareholders_equity
                    }
                }
            
            if current_assets and current_liabilities:
                working_capital = current_assets - current_liabilities
                calculated['working_capital'] = {
                    'value': working_capital,
                    'calculation': f"{current_assets} - {current_liabilities}",
                    'unit': 'USD'
                }
            
            # Book value per share calculation
            shares_outstanding = self._extract_shares_data(facts_data, 'outstanding', fiscal_year, fiscal_period)
            if shareholders_equity and shares_outstanding:
                calculated['book_value_per_share'] = {
                    'value': round(shareholders_equity / shares_outstanding, 2),
                    'calculation': f"{shareholders_equity} / {shares_outstanding}",
                    'components': {
                        'shareholders_equity': shareholders_equity,
                        'shares_outstanding': shares_outstanding
                    }
                }
            
            return calculated
            
        except Exception as e:
            self.main_logger.error(f"Error calculating balance sheet metrics: {e}")
            return {}
    
    def _get_concept_value(self, concepts: Dict, concept_name: str) -> Optional[float]:
        """Get numeric value from concept data"""
        try:
            concept_data = concepts.get(concept_name, {})
            if concept_data and not concept_data.get('missing', False):
                value = concept_data.get('value')
                if value and str(value).replace('-', '').replace('.', '').isdigit():
                    return float(value)
            return None
        except:
            return None
    
    def _extract_shares_data(self, facts_data: Dict, share_type: str, 
                           fiscal_year: int, fiscal_period: str) -> Optional[float]:
        """Extract shares outstanding data from facts"""
        try:
            # Define share concept tags based on type
            share_concepts = {
                'basic': ['WeightedAverageNumberOfSharesOutstandingBasic', 'SharesBasic'],
                'diluted': ['WeightedAverageNumberOfDilutedSharesOutstanding', 'SharesDiluted'],
                'outstanding': ['CommonStockSharesOutstanding', 'SharesOutstanding', 'SharesOut']
            }
            
            tags = share_concepts.get(share_type, [])
            
            for tag in tags:
                if tag in facts_data:
                    units = facts_data[tag].get('units', {})
                    
                    # Look for shares units
                    for unit_type in ['shares', 'USD/shares', 'pure']:
                        if unit_type in units:
                            for entry in units[unit_type]:
                                if (entry.get('fy') == fiscal_year and 
                                    entry.get('fp') == fiscal_period):
                                    value = entry.get('val')
                                    if value and str(value).replace('-', '').replace('.', '').isdigit():
                                        return float(value)
            
            return None
            
        except Exception as e:
            self.main_logger.error(f"Error extracting shares data: {e}")
            return None
    
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