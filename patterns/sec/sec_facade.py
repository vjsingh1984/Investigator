#!/usr/bin/env python3
"""
InvestiGator - SEC Data Facade Pattern
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

SEC Data Facade Pattern
Provides simplified interface for SEC data operations using design patterns
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from patterns.sec.sec_strategies import (
    ISECDataFetchStrategy, CompanyFactsStrategy, SubmissionsStrategy, 
    CachedDataStrategy, HybridFetchStrategy
)
from utils.cache.cache_manager import get_cache_manager
from utils.cache.cache_types import CacheType
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
        self.cache_manager = get_cache_manager()
        
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
    
    def get_recent_quarterly_data(self, symbol: str, max_periods: int = 8,
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
            cik = self.ticker_mapper.resolve_cik(symbol)
            if not cik:
                return None
            
            cache_key = {'symbol': symbol, 'cik': cik}
            return self.cache_manager.get(CacheType.COMPANY_FACTS, cache_key)
            
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
            cache_key = {'symbol': symbol, 'cik': cik}
            submissions_data = self.cache_manager.get(CacheType.SUBMISSION_DATA, cache_key)
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
        self.cache_manager = get_cache_manager()
        
        # Use existing aggregator and LLM facade with cache management
        from utils.financial_data_aggregator import FinancialDataAggregator
        from patterns.llm.llm_facade import create_llm_facade
        
        self.data_aggregator = FinancialDataAggregator(config)
        # Pass cache_manager to enable caching in LLM facade
        self.ollama = create_llm_facade(config, self.cache_manager)
        
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
            max_periods = options.get('max_periods', 8)
            strategy = options.get('strategy', 'hybrid')
            
            quarterly_data = self.sec_facade.get_recent_quarterly_data(
                symbol, max_periods, strategy
            )
            
            if not quarterly_data:
                return self._create_error_result(symbol, "No quarterly data available")
            
            self.logger.info(f"Data fetched for {symbol}")
            
            # Aggregate data
            aggregated = self.data_aggregator.aggregate_quarterly_data(quarterly_data)
            
            # Calculate average extraction-level quality for comparison
            extraction_qualities = [qd.financial_data.data_quality_score * 100 
                                  for qd in quarterly_data 
                                  if hasattr(qd.financial_data, 'data_quality_score') and qd.financial_data.data_quality_score]
            avg_extraction_quality = sum(extraction_qualities) / len(extraction_qualities) if extraction_qualities else 0
            
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
            
            # Analyze each quarter separately first
            quarterly_analyses = []
            for i, qdata in enumerate(quarterly_data_dicts):
                self.logger.info(f"Analyzing quarter {i+1}/{len(quarterly_data_dicts)}: {qdata.get('fiscal_year')}-{qdata.get('fiscal_period')}")
                
                # Analyze individual quarter
                quarter_result = self._analyze_single_quarter(symbol, qdata)
                if quarter_result and not quarter_result.get('error'):
                    quarterly_analyses.append(quarter_result)
                else:
                    self.logger.warning(f"Failed to analyze quarter {qdata.get('fiscal_period')}")
            
            # Create comprehensive analysis based on all quarters (conditional on skip_comprehensive)
            if not options.get('skip_comprehensive', False):
                self.logger.info(f"Creating comprehensive fundamental analysis from {len(quarterly_analyses)} quarters")
                analysis_result = self._create_comprehensive_fundamental_analysis(
                    symbol=symbol,
                    quarterly_analyses=quarterly_analyses,
                    aggregated_data=aggregated
                )
            else:
                self.logger.info(f"Skipping comprehensive analysis (--skip-comprehensive flag set)")
                # Return just the quarterly analyses without comprehensive synthesis
                analysis_result = {
                    'symbol': symbol,
                    'quarterly_analyses': quarterly_analyses,
                    'quarters_analyzed': len(quarterly_analyses),
                    'analysis_summary': f'Quarterly analysis completed for {len(quarterly_analyses)} quarters',
                    'skip_comprehensive': True,
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            
            # Cache the comprehensive analysis result (only if comprehensive analysis was performed)
            if not options.get('skip_comprehensive', False):
                try:
                    # Save comprehensive analysis with expected structure
                    fiscal_year = datetime.now().year
                    cache_response_data = {
                        'prompt': '',  # Prompt is generated inside _create_comprehensive_fundamental_analysis
                        'response': analysis_result,     # The parsed analysis result
                        'model_info': {
                            'model': self.config.ollama.models.get('fundamental_analysis', 'llama3.1:8b-instruct-q8_0'),
                            'temperature': 0.3,
                            'top_p': 0.9
                        }
                    }
                    # Cache LLM response using cache manager directly
                    cache_key = {
                        'symbol': symbol,
                        'form_type': 'COMPREHENSIVE',
                        'period': f"{fiscal_year}-FY",
                        'llm_type': 'sec'
                    }
                    cache_success = self.cache_manager.set(CacheType.LLM_RESPONSE, cache_key, cache_response_data)
                    if cache_success:
                        self.logger.debug(f"Cached LLM fundamental analysis for {symbol}")
                    else:
                        self.logger.warning(f"Failed to cache LLM analysis for {symbol}")
                except Exception as cache_error:
                    self.logger.warning(f"Error caching LLM analysis for {symbol}: {cache_error}")
            else:
                self.logger.info(f"Skipping comprehensive analysis cache (not performed)")
            
            # Extract response for processing
            # The comprehensive analysis should return the structured data directly
            if isinstance(analysis_result, dict) and 'analysis_summary' in analysis_result:
                # Use the structured analysis result directly instead of trying to parse text
                result = {
                    'symbol': symbol,
                    'financial_health_score': analysis_result.get('financial_health_score', 5.0),
                    'business_quality_score': analysis_result.get('business_quality_score', 5.0),
                    'growth_prospects_score': analysis_result.get('growth_prospects_score', 5.0),
                    'overall_score': analysis_result.get('overall_score', 5.0),
                    'key_insights': analysis_result.get('key_insights', []),
                    'key_risks': analysis_result.get('key_risks', []),
                    'confidence_level': analysis_result.get('confidence_level', 'MEDIUM'),
                    'analysis_summary': analysis_result.get('analysis_summary', ''),
                    'investment_thesis': analysis_result.get('investment_thesis', ''),
                    'trend_analysis': analysis_result.get('trend_analysis', {}),
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'data_quality': aggregated.get('data_quality', {}),
                    'extraction_quality': avg_extraction_quality,
                    'quarters_analyzed': aggregated.get('quarters_analyzed', 0),
                    'quarterly_analyses': analysis_result.get('quarterly_analyses', []),
                    'metadata': {
                        'architecture': 'pattern-based',
                        'version': '2.0'
                    }
                }
            else:
                # Fallback to text parsing if needed
                response_text = analysis_result.get('analysis_summary', '') if isinstance(analysis_result, dict) else str(analysis_result)
                result = self._parse_analysis_response(response_text, symbol, aggregated)
            
            self.logger.info(f"Analysis complete for {symbol}")
            
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
            from utils.json_utils import extract_json_from_text
            
            # Extract JSON from response using robust parser
            try:
                result = extract_json_from_text(response)
            except ValueError:
                self.logger.warning(f"Failed to extract JSON from response for {symbol}")
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
        """Raise error instead of returning default values"""
        error_msg = f"SEC analysis failed for {symbol}: {error}"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _analyze_single_quarter(self, symbol: str, quarter_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single quarter's financial data"""
        try:
            from utils.prompt_manager import get_prompt_manager
            prompt_manager = get_prompt_manager()
            
            # Render quarterly analysis prompt (removed dynamic analysis_date for better caching)
            quarterly_prompt = prompt_manager.render_template(
                'quarterly_fundamental_analysis.j2',
                symbol=symbol,
                analysis_date='STATIC_DATE_FOR_CACHE',  # Use static date for consistent cache keys
                fiscal_year=quarter_data.get('fiscal_year', 'Unknown'),
                fiscal_period=quarter_data.get('fiscal_period', 'Unknown'),
                form_type=quarter_data.get('form_type', '10-Q'),
                quarterly_data=json.dumps(quarter_data, indent=2)
            )
            
            # Get model for quarterly analysis
            model_name = self.config.ollama.models.get(
                'quarterly_analysis', 
                'llama3.1:8b-instruct-q8_0'
            )
            
            # Submit to LLM facade with proper metadata for cache key generation
            from patterns.llm.llm_interfaces import LLMTaskType
            task_data = {
                'symbol': symbol,
                'quarter_data': quarter_data,
                'prompt': quarterly_prompt,
                # Explicitly add fiscal metadata for cache key generation
                'fiscal_year': quarter_data.get('fiscal_year', 'Unknown'),
                'fiscal_period': quarter_data.get('fiscal_period', 'Unknown'),
                'form_type': quarter_data.get('form_type', '10-Q')
            }
            
            # Use LLM facade's orchestration - it handles cache and queue management
            response = self.ollama.generate_response(
                task_type=LLMTaskType.QUARTERLY_SUMMARY,
                data=task_data
            )
            
            # LLM facade handles caching automatically - no manual caching needed
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error analyzing quarter {quarter_data.get('fiscal_period')}: {e}")
            return {'error': str(e)}
    
    def _create_comprehensive_fundamental_analysis(self, symbol: str, 
                                                 quarterly_analyses: List[Dict],
                                                 aggregated_data: Dict) -> Dict[str, Any]:
        """Create comprehensive analysis from quarterly analyses"""
        try:
            from utils.prompt_manager import get_prompt_manager
            prompt_manager = get_prompt_manager()
            
            # Prepare summary of quarterly analyses
            quarters_summary = []
            for qa in quarterly_analyses:
                if 'quarterly_summary' in qa:
                    quarters_summary.append(qa['quarterly_summary'])
                elif 'analysis_summary' in qa:
                    quarters_summary.append({
                        'period': qa.get('period', 'Unknown'),
                        'score': qa.get('overall_score', 0),
                        'summary': qa.get('analysis_summary', '')
                    })
            
            # Create comprehensive prompt with trend analysis
            comprehensive_prompt = f"""
You are a senior equity research analyst creating a comprehensive fundamental analysis based on multiple quarterly reports.

COMPANY: {symbol}
ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d')}

QUARTERLY ANALYSES SUMMARY:
{json.dumps(quarters_summary, indent=2)}

AGGREGATED FINANCIAL DATA:
{json.dumps(aggregated_data, indent=2)}

Create a comprehensive fundamental analysis that:
1. Identifies multi-quarter trends in revenue, earnings, and margins
2. Evaluates the consistency and quality of financial performance
3. Assesses balance sheet evolution and capital allocation
4. Highlights strategic developments and competitive positioning
5. Projects future performance based on historical trends
6. Assesses data quality based on completeness, consistency, and reliability across quarters

DATA QUALITY ASSESSMENT INSTRUCTIONS:
- Analyze the consistency of financial statement data across quarters
- Evaluate the completeness of key financial metrics (revenue, earnings, balance sheet items)
- Assess the reliability of reported numbers (check for unusual patterns, restatements, or gaps)
- Consider the quality of disclosure and footnote information
- Rate the overall data quality based on XBRL completeness and SEC filing consistency

IMPORTANT: Respond ONLY with valid JSON, no explanatory text before or after.
Provide analysis in the following exact JSON format:
{{
    "financial_health_score": float (1-10),
    "business_quality_score": float (1-10),
    "growth_prospects_score": float (1-10),
    "data_quality_score": {{
        "score": float (1-10),
        "explanation": "Assessment of financial data quality, completeness, and reliability across quarters"
    }},
    "overall_score": float (1-10),
    "trend_analysis": {{
        "revenue_trend": "accelerating|stable|decelerating",
        "margin_trend": "expanding|stable|contracting",
        "cash_flow_trend": "improving|stable|deteriorating"
    }},
    "key_insights": ["list of 5-7 key insights from trend analysis"],
    "key_risks": ["list of 3-5 key risks identified"],
    "investment_thesis": "comprehensive investment thesis based on trends",
    "confidence_level": "HIGH|MEDIUM|LOW",
    "analysis_summary": "executive summary of findings"
}}
"""
            
            # Submit comprehensive analysis to LLM
            model_name = self.config.ollama.models.get(
                'fundamental_analysis', 
                'llama3.1:8b-instruct-q8_0'
            )
            
            # Use queue-based processing for comprehensive analysis
            from patterns.llm.llm_interfaces import LLMTaskType
            comprehensive_task_data = {
                'symbol': symbol,
                'quarterly_analyses': quarterly_analyses,
                'aggregated_data': aggregated_data,
                'prompt': comprehensive_prompt
            }
            
            response = self.ollama.generate_response(
                task_type=LLMTaskType.COMPREHENSIVE_ANALYSIS,
                data=comprehensive_task_data
            )
            
            # Extract and validate response based on response structure
            from utils.json_utils import extract_json_from_text
            
            try:
                # The queue-based generate_response returns structured data directly
                if isinstance(response, dict):
                    # If response has structured data directly (from queue processing)
                    if 'financial_health_score' in response:
                        result = response.copy()
                    elif 'response' in response:
                        # If response is wrapped (legacy query_ollama format)
                        result = extract_json_from_text(response.get('response', ''))
                    else:
                        # Try to extract from the response content
                        response_content = response.get('content', '') or str(response)
                        result = extract_json_from_text(response_content)
                else:
                    # String response
                    result = extract_json_from_text(str(response))
                
                result['quarterly_analyses'] = quarterly_analyses
                result['quarters_analyzed'] = len(quarterly_analyses)
                return result
                
            except Exception as e:
                self.logger.warning(f"Failed to parse comprehensive analysis response: {e}")
                # Fallback structure
                return {
                    'financial_health_score': 5.0,
                    'business_quality_score': 5.0,
                    'growth_prospects_score': 5.0,
                    'overall_score': 5.0,
                    'analysis_summary': str(response),
                    'quarterly_analyses': quarterly_analyses,
                    'quarters_analyzed': len(quarterly_analyses),
                    'confidence_level': 'LOW'
                }
                
        except Exception as e:
            self.logger.error(f"Error creating comprehensive analysis: {e}")
            return {
                'error': str(e),
                'quarterly_analyses': quarterly_analyses
            }
