#!/usr/bin/env python3
"""
InvestiGator - SEC Data Fetching Strategies
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

SEC Data Fetching Strategies
Strategy pattern implementations for different SEC data fetching approaches
"""

import logging
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
            # Check cache first
            cache_key = {'symbol': symbol}
            cached_data = self.cache_manager.get(CacheType.COMPANY_FACTS, cache_key)
            
            if cached_data:
                self.logger.info(f"Using cached company facts for {symbol}")
                facts = cached_data.get('companyfacts')
            else:
                # Fetch from API
                self.logger.info(f"Fetching company facts for {symbol} from SEC API")
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
            return self._extract_quarterly_data(facts, symbol, cik, max_periods)
            
        except Exception as e:
            self.logger.error(f"Error fetching company facts for {symbol}: {e}")
            return []
    
    def _extract_quarterly_data(self, facts: Dict, symbol: str, cik: str, max_periods: int) -> List[QuarterlyData]:
        """Extract quarterly data from company facts"""
        quarterly_data = []
        
        # Extract available periods from revenue data
        revenue_concepts = [
            'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax',
            'SalesRevenueNet'
        ]
        
        us_gaap = facts.get('facts', {}).get('us-gaap', {})
        
        for concept in revenue_concepts:
            if concept in us_gaap:
                units = us_gaap[concept].get('units', {})
                if 'USD' in units:
                    for entry in units['USD']:
                        if entry.get('form') in ['10-K', '10-Q']:
                            qd = QuarterlyData(
                                symbol=symbol,
                                cik=cik,
                                fiscal_year=entry.get('fy', 0),
                                fiscal_period=entry.get('fp', ''),
                                form_type=entry.get('form', ''),
                                filing_date=entry.get('filed', ''),
                                accession_number=entry.get('accn', ''),
                                financial_data=FinancialStatementData(
                                    symbol=symbol,
                                    cik=cik,
                                    fiscal_year=entry.get('fy', 0),
                                    fiscal_period=entry.get('fp', '')
                                )
                            )
                            quarterly_data.append(qd)
                    break
        
        # Sort and limit
        quarterly_data.sort(key=lambda x: (x.fiscal_year, x.fiscal_period), reverse=True)
        return quarterly_data[:max_periods]
    
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
            # Try to get cached submissions first
            from utils.cache.submission_cache_handler import get_submission_cache_handler
            handler = get_submission_cache_handler(self.config)
            
            parsed_data = handler.get_submission(symbol, cik)
            if not parsed_data:
                # Fetch from API
                self.logger.info(f"Fetching submissions for {symbol} from SEC API")
                submissions_data = self.api_client.get_submissions(cik)
                
                # Store in cache
                handler.save_submission(symbol, cik, submissions_data)
                
                # Parse the data
                from utils.submission_processor import get_submission_processor
                processor = get_submission_processor()
                parsed_data = processor.parse_submissions(submissions_data)
            
            # Get recent earnings filings
            from utils.submission_processor import get_submission_processor
            processor = get_submission_processor()
            recent_filings = processor.get_recent_earnings_filings(parsed_data, limit=max_periods)
            
            # Convert to QuarterlyData
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
        self.strategies = [
            CachedDataStrategy(config),
            SubmissionsStrategy(config),
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