#!/usr/bin/env python3
"""
InvestiGator - SEC Data Fetching Strategies
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

SEC Data Fetching Strategies
Strategy pattern implementations for different SEC data fetching approaches
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from utils.api_client import SECAPIClient
from utils.cache.cache_manager import get_cache_manager
from utils.cache.cache_types import CacheType
from utils.ticker_cik_mapper import TickerCIKMapper
from data.models import QuarterlyData, FinancialStatementData
from config import get_config

logger = logging.getLogger(__name__)


class ISECDataFetchStrategy(ABC):
    """Interface for SEC data fetching strategies"""
    
    @abstractmethod
    def fetch_quarterly_data(self, symbol: str, cik: str, max_periods: int) -> List[QuarterlyData]:
        """Fetch quarterly data using this strategy"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy"""
        pass
    
    @abstractmethod
    def supports_incremental_fetch(self) -> bool:
        """Whether this strategy supports incremental fetching"""
        pass


class CompanyFactsStrategy(ISECDataFetchStrategy):
    """Fetch data using SEC Company Facts API"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.api_client = SECAPIClient(user_agent=self.config.sec.user_agent, config=self.config)
        self.cache_manager = get_cache_manager()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def fetch_quarterly_data(self, symbol: str, cik: str, max_periods: int) -> List[QuarterlyData]:
        """Fetch quarterly data from company facts"""
        try:
            # Check cache first - include both symbol and cik for file cache compatibility
            cache_key = {'symbol': symbol, 'cik': cik}
            self.logger.info(f"🔍 CACHE GET: About to call cache_manager.get for {symbol}")
            cached_data = self.cache_manager.get(CacheType.COMPANY_FACTS, cache_key)
            self.logger.info(f"🔍 CACHE RESULT: {symbol} -> cached_data type: {type(cached_data)}, is_truthy: {bool(cached_data)}")
            
            if cached_data:
                self.logger.info(f"Using cached company facts for {symbol}")
                facts = cached_data.get('companyfacts')
            else:
                # Fetch from API using provided CIK
                self.logger.info(f"Fetching company facts for {symbol} (CIK: {cik}) from SEC API")
                facts = self.api_client.get_company_facts(cik)
                
                # Cache the results
                cache_value = {
                    'companyfacts': facts,
                    'metadata': {
                        'fetched_at': datetime.now().isoformat(),
                        'cik': cik,
                        'entity_name': facts.get('entityName', '')
                    }
                }
                self.cache_manager.set(CacheType.COMPANY_FACTS, cache_key, cache_value)
            
            # Extract quarterly data from facts
            self.logger.info(f"🔍 ABOUT TO CALL _extract_quarterly_data for {symbol}")
            return self._extract_quarterly_data(facts, symbol, cik, max_periods)
            
        except Exception as e:
            self.logger.error(f"Error fetching company facts for {symbol}: {e}")
            return []
    
    def _extract_quarterly_data(self, facts: Dict, symbol: str, cik: str, max_periods: int) -> List[QuarterlyData]:
        """Extract quarterly data from company facts"""
        self.logger.info(f"🔍 EXTRACT_QUARTERLY_DATA CALLED: {symbol}, max_periods: {max_periods}")
        self.logger.info(f"🔍 Facts data type: {type(facts)}, facts keys: {list(facts.keys()) if isinstance(facts, dict) else 'NOT_DICT'}")
        
        # First, collect all unique filing periods
        us_gaap = facts.get('facts', {}).get('us-gaap', {})
        self.logger.info(f"Found {len(us_gaap)} us-gaap concepts for {symbol}")
        
        # Collect all unique periods from 10-K and 10-Q filings
        period_data = {}  # key: (fy, fp, form, accn, filed) -> values
        
        # Key financial metrics to extract
        key_metrics = {
            # Income Statement
            'Revenues': 'revenue',
            'RevenueFromContractWithCustomerExcludingAssessedTax': 'revenue',
            'SalesRevenueNet': 'revenue',
            'CostOfRevenue': 'cost_of_revenue',
            'GrossProfit': 'gross_profit',
            'OperatingExpenses': 'operating_expenses',
            'OperatingIncomeLoss': 'operating_income',
            'NetIncomeLoss': 'net_income',
            'EarningsPerShareBasic': 'eps_basic',
            'EarningsPerShareDiluted': 'eps_diluted',
            
            # Balance Sheet
            'Assets': 'total_assets',
            'AssetsCurrent': 'current_assets',
            'Liabilities': 'total_liabilities',
            'LiabilitiesCurrent': 'current_liabilities',
            'StockholdersEquity': 'stockholders_equity',
            'CashAndCashEquivalentsAtCarryingValue': 'cash',
            'AccountsReceivableNetCurrent': 'accounts_receivable',
            'Inventory': 'inventory',
            'PropertyPlantAndEquipmentNet': 'ppe_net',
            'LongTermDebt': 'long_term_debt',
            
            # Cash Flow
            'NetCashProvidedByUsedInOperatingActivities': 'operating_cash_flow',
            'NetCashProvidedByUsedInInvestingActivities': 'investing_cash_flow',
            'NetCashProvidedByUsedInFinancingActivities': 'financing_cash_flow',
            
            # Other Metrics
            'ResearchAndDevelopmentExpense': 'rd_expense',
            'GeneralAndAdministrativeExpense': 'ga_expense',
            'SellingAndMarketingExpense': 'sm_expense',
            'CommonStockSharesOutstanding': 'shares_outstanding'
        }
        
        # Extract data for each concept
        for concept, metric_name in key_metrics.items():
            if concept in us_gaap:
                units = us_gaap[concept].get('units', {})
                
                # Handle USD values
                if 'USD' in units:
                    for entry in units['USD']:
                        if entry.get('form') in ['10-K', '10-Q']:
                            key = (
                                entry.get('fy', 0),
                                entry.get('fp', ''),
                                entry.get('form', ''),
                                entry.get('accn', ''),
                                entry.get('filed', '')
                            )
                            if key not in period_data:
                                period_data[key] = {
                                    'fiscal_year': entry.get('fy', 0),
                                    'fiscal_period': entry.get('fp', ''),
                                    'form_type': entry.get('form', ''),
                                    'accession_number': entry.get('accn', ''),
                                    'filing_date': entry.get('filed', ''),
                                    'end_date': entry.get('end', ''),
                                    'metrics': {}
                                }
                            period_data[key]['metrics'][metric_name] = entry.get('val', 0)
                
                # Handle shares (non-monetary)
                elif 'shares' in units:
                    for entry in units['shares']:
                        if entry.get('form') in ['10-K', '10-Q']:
                            key = (
                                entry.get('fy', 0),
                                entry.get('fp', ''),
                                entry.get('form', ''),
                                entry.get('accn', ''),
                                entry.get('filed', '')
                            )
                            if key not in period_data:
                                period_data[key] = {
                                    'fiscal_year': entry.get('fy', 0),
                                    'fiscal_period': entry.get('fp', ''),
                                    'form_type': entry.get('form', ''),
                                    'accession_number': entry.get('accn', ''),
                                    'filing_date': entry.get('filed', ''),
                                    'end_date': entry.get('end', ''),
                                    'metrics': {}
                                }
                            period_data[key]['metrics'][metric_name] = entry.get('val', 0)
        
        # Sort periods by fiscal year and period, most recent first
        sorted_periods = sorted(period_data.items(), 
                               key=lambda x: (x[1]['fiscal_year'], x[1]['fiscal_period']), 
                               reverse=True)
        
        # Ensure we get a mix of quarterly (10-Q) and annual (10-K) filings
        # Prioritize: most recent quarters + at least one annual filing
        recent_periods = []
        quarterly_count = 0
        annual_count = 0
        target_annual = max(1, max_periods // 4)  # At least 1 annual filing, more for larger requests
        target_quarterly = max_periods - target_annual
        
        # First pass: get the most recent filings, prioritizing a good mix
        for period_key, period_info in sorted_periods:
            form_type = period_info['form_type']
            
            if form_type == '10-K' and annual_count < target_annual:
                recent_periods.append((period_key, period_info))
                annual_count += 1
            elif form_type == '10-Q' and quarterly_count < target_quarterly:
                recent_periods.append((period_key, period_info))
                quarterly_count += 1
            
            if len(recent_periods) >= max_periods:
                break
        
        # Second pass: fill remaining slots if we haven't reached max_periods
        if len(recent_periods) < max_periods:
            for period_key, period_info in sorted_periods:
                if (period_key, period_info) not in recent_periods:
                    recent_periods.append((period_key, period_info))
                    if len(recent_periods) >= max_periods:
                        break
        
        self.logger.info(f"Selected {len(recent_periods)} periods: {quarterly_count} quarterly (10-Q), {annual_count} annual (10-K) from {len(period_data)} total periods")
        
        # Convert to QuarterlyData objects for only the recent periods
        quarterly_data = []
        for period_key, period_info in recent_periods:
            metrics = period_info['metrics']
            
            # Create financial statement data with actual values
            financial_data = FinancialStatementData(
                symbol=symbol,
                cik=cik,
                fiscal_year=period_info['fiscal_year'],
                fiscal_period=period_info['fiscal_period'],
                form_type=period_info['form_type'],
                filing_date=period_info['filing_date'],
                income_statement={
                    'revenue': metrics.get('revenue', 0),
                    'cost_of_revenue': metrics.get('cost_of_revenue', 0),
                    'gross_profit': metrics.get('gross_profit', 0),
                    'operating_expenses': metrics.get('operating_expenses', 0),
                    'operating_income': metrics.get('operating_income', 0),
                    'net_income': metrics.get('net_income', 0),
                    'eps_basic': metrics.get('eps_basic', 0),
                    'eps_diluted': metrics.get('eps_diluted', 0),
                    'rd_expense': metrics.get('rd_expense', 0),
                    'ga_expense': metrics.get('ga_expense', 0),
                    'sm_expense': metrics.get('sm_expense', 0)
                },
                balance_sheet={
                    'total_assets': metrics.get('total_assets', 0),
                    'current_assets': metrics.get('current_assets', 0),
                    'cash': metrics.get('cash', 0),
                    'accounts_receivable': metrics.get('accounts_receivable', 0),
                    'inventory': metrics.get('inventory', 0),
                    'ppe_net': metrics.get('ppe_net', 0),
                    'total_liabilities': metrics.get('total_liabilities', 0),
                    'current_liabilities': metrics.get('current_liabilities', 0),
                    'long_term_debt': metrics.get('long_term_debt', 0),
                    'stockholders_equity': metrics.get('stockholders_equity', 0),
                    'shares_outstanding': metrics.get('shares_outstanding', 0)
                },
                cash_flow_statement={
                    'operating_cash_flow': metrics.get('operating_cash_flow', 0),
                    'investing_cash_flow': metrics.get('investing_cash_flow', 0),
                    'financing_cash_flow': metrics.get('financing_cash_flow', 0)
                }
            )
            
            # Calculate data quality score based on available metrics
            total_expected = len(key_metrics)
            total_available = len([v for v in metrics.values() if v != 0])
            financial_data.data_quality_score = total_available / total_expected if total_expected > 0 else 0
            
            qd = QuarterlyData(
                symbol=symbol,
                cik=cik,
                fiscal_year=period_info['fiscal_year'],
                fiscal_period=period_info['fiscal_period'],
                form_type=period_info['form_type'],
                filing_date=period_info['filing_date'],
                accession_number=period_info['accession_number'],
                financial_data=financial_data
            )
            quarterly_data.append(qd)
            
            self.logger.info(f"Extracted {len(metrics)} metrics for {symbol} {period_info['fiscal_year']}-{period_info['fiscal_period']} (quality: {financial_data.data_quality_score:.2%})")
        
        self.logger.info(f"Returning {len(quarterly_data)} quarterly periods for {symbol} (limited from {len(period_data)} total periods)")
        
        return quarterly_data
    
    def get_strategy_name(self) -> str:
        return "CompanyFactsStrategy"
    
    def supports_incremental_fetch(self) -> bool:
        return False  # Company facts returns all data at once


class SubmissionsStrategy(ISECDataFetchStrategy):
    """Fetch data using SEC Submissions API"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.api_client = SECAPIClient(user_agent=self.config.sec.user_agent, config=self.config)
        self.cache_manager = get_cache_manager()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def fetch_quarterly_data(self, symbol: str, cik: str, max_periods: int) -> List[QuarterlyData]:
        """Fetch quarterly data from submissions"""
        try:
            # Try to get cached submissions using cache manager interface
            cache_key = {
                'symbol': symbol,
                'cik': cik
            }
            
            self.logger.debug(f"Attempting to get cached submission data for {symbol}")
            cached_result = self.cache_manager.get(CacheType.SUBMISSION_DATA, cache_key)
            self.logger.debug(f"Cache result obtained: {cached_result is not None}")
            
            if cached_result:
                self.logger.debug(f"Processing cached submission data for {symbol}")
                # Extract parsed data from cache result
                submissions_data = cached_result.get('submissions_data', {})
                self.logger.debug(f"Submissions data extracted: {submissions_data is not None}, size: {len(str(submissions_data)) if submissions_data else 0}")
                
                if submissions_data:
                    # Parse the cached data
                    self.logger.debug(f"Starting to parse cached submissions data for {symbol}")
                    from utils.submission_processor import get_submission_processor
                    processor = get_submission_processor()
                    self.logger.debug(f"Got submission processor, calling parse_submissions")
                    parsed_data = processor.parse_submissions(submissions_data)
                    self.logger.debug(f"Successfully parsed cached submissions data for {symbol}")
                else:
                    self.logger.debug(f"No submissions data in cache for {symbol}")
                    cached_result = None
            
            if not cached_result:
                # Fetch from API
                self.logger.info(f"Fetching submissions for {symbol} from SEC API")
                submissions_data = self.api_client.get_submissions(cik)
                
                # Store in cache using cache manager interface
                cache_value = {
                    'symbol': symbol,
                    'cik': cik,
                    'submissions_data': submissions_data,
                    'company_name': submissions_data.get('name', ''),
                    'cached_at': time.time()
                }
                self.cache_manager.set(CacheType.SUBMISSION_DATA, cache_key, cache_value)
                
                # Parse the data
                from utils.submission_processor import get_submission_processor
                processor = get_submission_processor()
                parsed_data = processor.parse_submissions(submissions_data)
            
            # Get recent earnings filings
            self.logger.debug(f"Getting recent earnings filings for {symbol}, max_periods: {max_periods}")
            from utils.submission_processor import get_submission_processor
            processor = get_submission_processor()
            self.logger.debug(f"Calling get_recent_earnings_filings for {symbol}")
            recent_filings = processor.get_recent_earnings_filings(parsed_data, limit=max_periods)
            self.logger.debug(f"Got {len(recent_filings)} recent filings for {symbol}")
            
            # Convert to QuarterlyData
            self.logger.debug(f"Converting {len(recent_filings)} filings to QuarterlyData for {symbol}")
            quarterly_data = []
            for filing in recent_filings:
                qd = QuarterlyData(
                    symbol=symbol,
                    cik=cik,
                    fiscal_year=filing.fiscal_year,
                    fiscal_period=filing.fiscal_period,
                    form_type=filing.form_type,
                    filing_date=filing.filing_date,
                    accession_number=filing.accession_number,
                    financial_data=FinancialStatementData(
                        symbol=symbol,
                        cik=cik,
                        fiscal_year=filing.fiscal_year,
                        fiscal_period=filing.fiscal_period,
                        form_type=filing.form_type,
                        filing_date=filing.filing_date,
                        accession_number=filing.accession_number
                    )
                )
                quarterly_data.append(qd)
            
            return quarterly_data
            
        except Exception as e:
            self.logger.error(f"Error fetching submissions for {symbol}: {e}")
            return []
    
    def get_strategy_name(self) -> str:
        return "SubmissionsStrategy"
    
    def supports_incremental_fetch(self) -> bool:
        return True  # Can fetch specific periods


class CachedDataStrategy(ISECDataFetchStrategy):
    """Fetch data from cache layers only"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.cache_manager = get_cache_manager()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def fetch_quarterly_data(self, symbol: str, cik: str, max_periods: int) -> List[QuarterlyData]:
        """Fetch quarterly data from cache only"""
        try:
            # Check database for existing quarterly data
            from utils.db import get_quarterly_metrics_dao
            dao = get_quarterly_metrics_dao()
            
            # Get recent quarters from database
            quarterly_data = []
            
            # Try to get from various cache sources
            # 1. Check quarterly metrics table
            # 2. Check SEC response cache
            # 3. Check file cache
            
            self.logger.info(f"Fetched {len(quarterly_data)} quarters from cache for {symbol}")
            return quarterly_data
            
        except Exception as e:
            self.logger.error(f"Error fetching cached data for {symbol}: {e}")
            return []
    
    def get_strategy_name(self) -> str:
        return "CachedDataStrategy"
    
    def supports_incremental_fetch(self) -> bool:
        return True


class HybridFetchStrategy(ISECDataFetchStrategy):
    """Hybrid strategy that tries multiple approaches"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        # TEMPORARY FIX: Skip SubmissionsStrategy due to hanging issue
        self.strategies = [
            CachedDataStrategy(config),
            # SubmissionsStrategy(config),  # COMMENTED OUT - CAUSES HANGING
            CompanyFactsStrategy(config)
        ]
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def fetch_quarterly_data(self, symbol: str, cik: str, max_periods: int) -> List[QuarterlyData]:
        """Try multiple strategies to fetch data"""
        all_data = []
        
        for strategy in self.strategies:
            try:
                self.logger.info(f"Trying {strategy.get_strategy_name()} for {symbol}")
                data = strategy.fetch_quarterly_data(symbol, cik, max_periods)
                
                if data:
                    all_data.extend(data)
                    
                    # If we have enough data, stop
                    if len(all_data) >= max_periods:
                        break
                        
            except Exception as e:
                self.logger.warning(f"Strategy {strategy.get_strategy_name()} failed: {e}")
                continue
        
        # Deduplicate and sort
        unique_data = self._deduplicate_data(all_data)
        unique_data.sort(key=lambda x: (x.fiscal_year, x.fiscal_period), reverse=True)
        
        return unique_data[:max_periods]
    
    def _deduplicate_data(self, data: List[QuarterlyData]) -> List[QuarterlyData]:
        """Remove duplicate quarterly data"""
        seen = set()
        unique = []
        
        for item in data:
            key = (item.symbol, item.fiscal_year, item.fiscal_period)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        
        return unique
    
    def get_strategy_name(self) -> str:
        return "HybridFetchStrategy"
    
    def supports_incremental_fetch(self) -> bool:
        return True