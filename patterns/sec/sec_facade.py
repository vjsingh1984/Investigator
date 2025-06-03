#!/usr/bin/env python3
"""
InvestiGator - SEC Data Facade Pattern
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

SEC Data Facade Pattern
Provides simplified interface for SEC data operations using design patterns
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from patterns.sec.sec_strategies import (
    ISECDataFetchStrategy, CompanyFactsStrategy, SubmissionsStrategy, 
    CachedDataStrategy, HybridFetchStrategy
)
from utils.cache import get_cache_facade
from patterns.sec.sec_adapters import (
    SECToInternalAdapter, InternalToLLMAdapter, 
    FilingContentAdapter, CompanyFactsToDetailedAdapter
)
from patterns.core.interfaces import DataSourceType, QuarterlyMetrics
from data.models import QuarterlyData, FinancialStatementData
from utils.ticker_cik_mapper import TickerCIKMapper
from utils.api_client import SECAPIClient
from config import get_config

logger = logging.getLogger(__name__)


class SECDataFacade:
    """
    Simplified interface for SEC data operations.
    Replaces the monolithic SECDataFetcher class.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.ticker_mapper = TickerCIKMapper(data_dir=str(self.config.sec.cache_dir))
        self.cache_facade = get_cache_facade(config)
        
        # Initialize strategies
        self.strategies = {
            'company_facts': CompanyFactsStrategy(config),
            'submissions': SubmissionsStrategy(config),
            'cached': CachedDataStrategy(config),
            'hybrid': HybridFetchStrategy(config)
        }
        
        # Initialize adapters
        self.sec_adapter = SECToInternalAdapter(config)
        self.llm_adapter = InternalToLLMAdapter(config)
        self.filing_adapter = FilingContentAdapter(config)
        self.detailed_adapter = CompanyFactsToDetailedAdapter(config)
        
        # Default strategy
        self.default_strategy = 'hybrid'
    
    def get_recent_quarterly_data(self, symbol: str, max_periods: int = 4,
                                 strategy: Optional[str] = None) -> List[QuarterlyData]:
        """
        Get recent quarterly data for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            max_periods: Maximum number of periods to fetch
            strategy: Strategy to use (company_facts, submissions, cached, hybrid)
            
        Returns:
            List of QuarterlyData objects
        """
        try:
            # Get CIK for symbol
            cik = self.ticker_mapper.get_cik_padded(symbol)
            if not cik:
                self.logger.error(f"Could not find CIK for {symbol}")
                return []
            
            # Select strategy
            fetch_strategy = self.strategies.get(strategy or self.default_strategy)
            if not fetch_strategy:
                self.logger.warning(f"Unknown strategy {strategy}, using default")
                fetch_strategy = self.strategies[self.default_strategy]
            
            # Fetch data using strategy
            self.logger.info(f"Fetching quarterly data for {symbol} using {fetch_strategy.get_strategy_name()}")
            quarterly_data = fetch_strategy.fetch_quarterly_data(symbol, cik, max_periods)
            
            # Populate financial data if needed
            for qd in quarterly_data:
                if not qd.financial_data or not hasattr(qd.financial_data, 'income_statement'):
                    self._populate_financial_data(qd)
            
            return quarterly_data
            
        except Exception as e:
            self.logger.error(f"Error getting quarterly data for {symbol}: {e}")
            return []
    
    def get_company_facts(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company facts with caching"""
        try:
            cik = self.ticker_mapper.get_cik_padded(symbol)
            if not cik:
                return None
            
            return self.cache_facade.get_company_facts(symbol, cik)
            
        except Exception as e:
            self.logger.error(f"Error getting company facts for {symbol}: {e}")
            return None
    
    def get_latest_filing(self, symbol: str, form_type: str = "10-K") -> Optional[Dict[str, Any]]:
        """Get latest SEC filing for a symbol"""
        try:
            cik = self.ticker_mapper.get_cik_padded(symbol)
            if not cik:
                return None
            
            # Get submissions
            submissions_data = self.cache_facade.get_submissions(symbol, cik)
            if not submissions_data:
                return None
            
            # Find latest filing of requested type
            submissions = submissions_data.get('submissions', {})
            if isinstance(submissions, list):
                # Already processed format
                for sub in submissions:
                    if sub.get('form_type') == form_type:
                        return sub
            else:
                # Raw SEC format
                recent_filings = submissions.get('filings', {}).get('recent', {})
                form_types = recent_filings.get('form', [])
                
                for i, form in enumerate(form_types):
                    if form == form_type:
                        return {
                            'form_type': form,
                            'filing_date': recent_filings.get('filingDate', [])[i],
                            'accession_number': recent_filings.get('accessionNumber', [])[i]
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest {form_type} for {symbol}: {e}")
            return None
    
    def get_filing_content(self, filing_url: str, max_length: int = 100000) -> str:
        """Get and clean filing content"""
        try:
            api_client = SECAPIClient(self.config.sec.user_agent, self.config)
            
            # Fetch filing content
            response = api_client.session.get(filing_url, timeout=60)
            response.raise_for_status()
            
            # Clean and adapt content
            cleaned_content = self.filing_adapter.adapt(response.text)
            
            # Truncate if needed
            if len(cleaned_content) > max_length:
                cleaned_content = cleaned_content[:max_length]
            
            return cleaned_content
            
        except Exception as e:
            self.logger.error(f"Error getting filing content: {e}")
            return ""
    
    def format_for_llm(self, quarterly_data: List[QuarterlyData]) -> str:
        """Format quarterly data for LLM consumption"""
        return self.llm_adapter.adapt(quarterly_data)
    
    def get_detailed_categories(self, symbol: str, fiscal_year: int, 
                               fiscal_period: str) -> Dict[str, Any]:
        """Get detailed financial categories for a specific period"""
        try:
            # Get company facts
            facts_data = self.get_company_facts(symbol)
            if not facts_data or 'companyfacts' not in facts_data:
                return {}
            
            # Convert to detailed categories
            detailed = self.detailed_adapter.adapt(facts_data['companyfacts'])
            
            # Update metadata for specific period
            if 'company_metadata' in detailed:
                detailed['company_metadata']['fiscal_year'] = fiscal_year
                detailed['company_metadata']['fiscal_period'] = fiscal_period
            
            return detailed
            
        except Exception as e:
            self.logger.error(f"Error getting detailed categories: {e}")
            return {}
    
    def _populate_financial_data(self, quarterly_data: QuarterlyData) -> None:
        """Populate financial data for a quarterly period"""
        try:
            # Get detailed categories
            detailed = self.get_detailed_categories(
                quarterly_data.symbol,
                quarterly_data.fiscal_year,
                quarterly_data.fiscal_period
            )
            
            if not detailed:
                return
            
            # Map to financial statement data
            if not quarterly_data.financial_data:
                quarterly_data.financial_data = FinancialStatementData(
                    symbol=quarterly_data.symbol,
                    cik=quarterly_data.cik,
                    fiscal_year=quarterly_data.fiscal_year,
                    fiscal_period=quarterly_data.fiscal_period
                )
            
            # Populate income statement
            income_categories = [k for k in detailed.keys() if k.startswith('income_statement_')]
            quarterly_data.financial_data.income_statement = {
                cat: detailed[cat] for cat in income_categories
            }
            
            # Populate balance sheet
            balance_categories = [k for k in detailed.keys() if k.startswith('balance_sheet_')]
            quarterly_data.financial_data.balance_sheet = {
                cat: detailed[cat] for cat in balance_categories
            }
            
            # Populate cash flow
            cashflow_categories = [k for k in detailed.keys() if k.startswith('cash_flow_')]
            quarterly_data.financial_data.cash_flow_statement = {
                cat: detailed[cat] for cat in cashflow_categories
            }
            
            # Store comprehensive data
            quarterly_data.financial_data.comprehensive_data = detailed
            
        except Exception as e:
            self.logger.error(f"Error populating financial data: {e}")


class FundamentalAnalysisFacadeV2:
    """
    Enhanced facade that uses the new SEC pattern-based architecture.
    Replaces the monolithic FundamentalAnalyzer class.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.sec_facade = SECDataFacade(config)
        self.cache_facade = get_cache_facade(config)
        
        # Use existing aggregator and LLM interface
        from utils.financial_data_aggregator import FinancialDataAggregator
        from patterns.llm.llm_facade import create_llm_facade
        
        self.data_aggregator = FinancialDataAggregator(config)
        self.ollama = create_llm_facade(config)
        
        # Observer pattern removed - using direct logging instead
    
    def analyze_symbol(self, symbol: str, **options) -> Dict[str, Any]:
        """
        Perform fundamental analysis for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            **options: Additional options (max_periods, strategy, etc.)
            
        Returns:
            Analysis results
        """
        try:
            # Start analysis
            self.logger.info(f"Starting fundamental analysis for {symbol}")
            
            # Get quarterly data
            max_periods = options.get('max_periods', 4)
            strategy = options.get('strategy', 'hybrid')
            
            quarterly_data = self.sec_facade.get_recent_quarterly_data(
                symbol, max_periods, strategy
            )
            
            if not quarterly_data:
                return self._create_error_result(symbol, "No quarterly data available")
            
            self.logger.info(f"Data fetched for {symbol}")
            
            # Aggregate data
            aggregated = self.data_aggregator.aggregate_quarterly_data(quarterly_data)
            
            self.logger.info(f"Data aggregated for {symbol}")
            
            # Perform LLM analysis
            llm_prompt = self.sec_facade.format_for_llm(quarterly_data)
            
            model_name = self.config.ollama.models.get(
                'fundamental_analysis', 
                'llama3.1:8b-instruct-q8_0'
            )
            
            # Convert quarterly data to dictionaries for JSON serialization
            quarterly_data_dicts = []
            for qdata in quarterly_data:
                if hasattr(qdata, 'to_dict'):
                    quarterly_data_dicts.append(qdata.to_dict())
                elif isinstance(qdata, dict):
                    quarterly_data_dicts.append(qdata)
                else:
                    # Fallback to basic dict conversion
                    quarterly_data_dicts.append({
                        'symbol': getattr(qdata, 'symbol', symbol),
                        'fiscal_year': getattr(qdata, 'fiscal_year', 'Unknown'),
                        'fiscal_period': getattr(qdata, 'fiscal_period', 'Unknown'),
                        'form_type': getattr(qdata, 'form_type', 'Unknown')
                    })
            
            # Use pattern-based fundamental analysis
            analysis_result = self.ollama.analyze_fundamental(
                symbol=symbol,
                quarterly_data=quarterly_data_dicts,
                filing_data=aggregated
            )
            
            # Cache the LLM analysis result
            try:
                cache_success = self.cache_facade.cache_llm_response(
                    symbol=symbol,
                    llm_type='sec',
                    response_data=analysis_result,
                    form_type='10-K',
                    period='latest'
                )
                if cache_success:
                    self.logger.debug(f"Cached LLM fundamental analysis for {symbol}")
                else:
                    self.logger.warning(f"Failed to cache LLM analysis for {symbol}")
            except Exception as cache_error:
                self.logger.warning(f"Error caching LLM analysis for {symbol}: {cache_error}")
            
            # Extract response for processing
            response = analysis_result.get('analysis_summary', '')
            
            self.logger.info(f"Analysis complete for {symbol}")
            
            # Parse and return results
            result = self._parse_analysis_response(response, symbol, aggregated)
            
            self.logger.info(f"Fundamental analysis completed successfully for {symbol}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return self._create_error_result(symbol, str(e))
    
    # Observer pattern removed - using direct logging for progress tracking
    
    def _create_analysis_prompt(self, symbol: str, aggregated: Dict, 
                               llm_data: str) -> str:
        """Create prompt for LLM analysis"""
        return f"""
Analyze the fundamental financial health of {symbol} based on the following data:

{llm_data}

Data Quality: {aggregated.get('data_quality', {}).get('completeness_score', 0):.1f}%

Please provide:
1. Financial Health Score (0-10)
2. Business Quality Score (0-10)
3. Growth Prospects Score (0-10)
4. Overall Score (0-10)
5. Key Insights (3-5 bullet points)
6. Key Risks (3-5 bullet points)
7. Confidence Level (HIGH/MEDIUM/LOW)

Format as JSON.
"""
    
    def _parse_analysis_response(self, response: str, symbol: str, 
                                aggregated: Dict) -> Dict[str, Any]:
        """Parse LLM response"""
        try:
            import json
            
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
            else:
                result = {}
            
            # Ensure all fields are present
            return {
                'symbol': symbol,
                'financial_health_score': result.get('financial_health_score', 5.0),
                'business_quality_score': result.get('business_quality_score', 5.0),
                'growth_prospects_score': result.get('growth_prospects_score', 5.0),
                'overall_score': result.get('overall_score', 5.0),
                'key_insights': result.get('key_insights', []),
                'key_risks': result.get('key_risks', []),
                'confidence_level': result.get('confidence_level', 'MEDIUM'),
                'analysis_summary': result.get('analysis_summary', ''),
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'data_quality': aggregated.get('data_quality', {}),
                'quarters_analyzed': aggregated.get('quarters_analyzed', 0),
                'metadata': {
                    'architecture': 'pattern-based',
                    'version': '2.0'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            return self._create_error_result(symbol, "Failed to parse analysis")
    
    def _create_error_result(self, symbol: str, error: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'symbol': symbol,
            'financial_health_score': 5.0,
            'business_quality_score': 5.0,
            'growth_prospects_score': 5.0,
            'overall_score': 5.0,
            'key_insights': [f'Analysis error: {error}'],
            'key_risks': ['Unable to complete analysis'],
            'confidence_level': 'LOW',
            'analysis_summary': error,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'data_quality': {'completeness_score': 0},
            'quarters_analyzed': 0,
            'metadata': {'error': True}
        }