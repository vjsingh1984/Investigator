#!/usr/bin/env python3
"""
InvestiGator - Analysis Synthesis Module (Refactored)
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

This module synthesizes fundamental and technical analysis to generate final investment
recommendations. Report generation and charting are delegated to separate modules.
"""

import logging
import time
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

from config import get_config
from utils.db import DatabaseManager, get_llm_responses_dao
from utils.cache import CacheManager, CacheType, get_cache_manager
from patterns.llm.llm_facade import create_llm_facade
from patterns.analysis.peer_comparison import get_peer_comparison_analyzer
from utils.chart_generator import ChartGenerator
from utils.report_generator import PDFReportGenerator, ReportConfig
from utils.weekly_report_generator import WeeklyReportGenerator
from utils.ascii_art import ASCIIArt
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class InvestmentRecommendation:
    """Data class for investment recommendations"""
    symbol: str
    overall_score: float
    fundamental_score: float
    technical_score: float
    income_score: float  # Income statement score
    cashflow_score: float  # Cash flow statement score
    balance_score: float  # Balance sheet score
    growth_score: float  # Growth prospects score
    value_score: float  # Value investment score
    business_quality_score: float  # Business quality score from SEC analysis
    recommendation: str  # BUY, HOLD, SELL
    confidence: str  # HIGH, MEDIUM, LOW
    price_target: Optional[float]
    current_price: Optional[float]
    investment_thesis: str
    time_horizon: str  # SHORT-TERM, MEDIUM-TERM, LONG-TERM
    position_size: str  # LARGE, MODERATE, SMALL, AVOID
    key_catalysts: List[str]
    key_risks: List[str]
    key_insights: List[str]
    entry_strategy: str
    exit_strategy: str
    stop_loss: Optional[float]
    analysis_timestamp: datetime
    data_quality_score: float
    analysis_thinking: Optional[str] = None
    synthesis_details: Optional[str] = None


class InvestmentSynthesizer:
    """Synthesizes fundamental and technical analysis into actionable recommendations"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize synthesizer with configuration"""
        self.config = get_config()
        self.db_manager = DatabaseManager()
        
        # Initialize interfaces
        self.ollama = create_llm_facade(self.config, cache_manager=None)
        # Import CacheManager here to avoid circular imports
        from utils.cache.cache_manager import CacheManager
        self.cache_manager = CacheManager(self.config)  # Use cache manager with config
        
        # Initialize generators
        self.chart_generator = ChartGenerator(self.config.reports_dir / "charts")
        self.report_generator = PDFReportGenerator(
            self.config.reports_dir / "synthesis",
            ReportConfig(
                title="Investment Analysis Report",
                subtitle="Comprehensive Stock Analysis",
                include_charts=True
            )
        )
        self.weekly_report_generator = WeeklyReportGenerator(
            self.config.reports_dir / "weekly"
        )
        
        # Initialize DAOs
        self.llm_dao = get_llm_responses_dao()
        
        # Initialize loggers
        self.main_logger = self.config.get_main_logger('synthesizer')
        
        # Response processing handled by LLM facade
        
        # Cache directories
        self.llm_cache_dir = self.config.data_dir / "llm_cache"
        self.llm_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.main_logger.info("Investment synthesizer initialized")
    
    
    def _get_latest_fiscal_period(self, fundamental_data=None, technical_data=None):
        """
        Determine the latest fiscal period from available data.
        
        Args:
            fundamental_data: SEC fundamental data (optional)
            technical_data: Technical analysis data (optional)
            
        Returns:
            Tuple of (fiscal_year, fiscal_period)
        """
        try:
            # For now, use current year and determine quarter based on current date
            from datetime import datetime
            current_date = datetime.now()
            current_year = current_date.year
            
            # Determine fiscal quarter based on current month
            month = current_date.month
            if month <= 3:
                fiscal_period = "Q4"  # Q4 of previous year
                fiscal_year = current_year - 1
            elif month <= 6:
                fiscal_period = "Q1"
                fiscal_year = current_year
            elif month <= 9:
                fiscal_period = "Q2"
                fiscal_year = current_year
            else:
                fiscal_period = "Q3"
                fiscal_year = current_year
            
            # TODO: In the future, extract this from actual SEC filing data
            # if fundamental_data and 'fiscal_year' in fundamental_data:
            #     fiscal_year = fundamental_data['fiscal_year']
            #     fiscal_period = fundamental_data.get('fiscal_period', fiscal_period)
            
            return fiscal_year, fiscal_period
            
        except Exception as e:
            self.main_logger.warning(f"Could not determine fiscal period: {e}, using defaults")
            return datetime.now().year, "FY"
    
    def synthesize_analysis(self, symbol: str, synthesis_mode: str = 'comprehensive') -> InvestmentRecommendation:
        """
        Synthesize fundamental and technical analysis for a symbol
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            InvestmentRecommendation object
        """
        try:
            # Get symbol-specific logger
            symbol_logger = self.config.get_symbol_logger(symbol, 'synthesizer')
            
            self.main_logger.info(f"Starting synthesis for {symbol}")
            symbol_logger.info("Starting investment synthesis")
            
            # Fetch all LLM responses for the symbol
            symbol_logger.info("Fetching LLM analysis responses")
            llm_responses = self._fetch_llm_responses(symbol)
            
            # Fetch latest market data
            symbol_logger.info("Fetching latest market data")
            latest_data = self._fetch_latest_data(symbol)
            
            # Calculate base scores
            symbol_logger.info("Calculating analysis scores")
            fundamental_score = self._calculate_fundamental_score(llm_responses)
            technical_score = self._calculate_technical_score(llm_responses)
            
            # Calculate weighted overall score
            overall_score = self._calculate_weighted_score(fundamental_score, technical_score)
            
            # Get current price from latest data (database) first, then try LLM response
            current_price = latest_data.get('current_price', 0)
            if current_price == 0:
                current_price = latest_data.get('technical', {}).get('current_price', 0)
            
            # If still no price, try to extract from technical LLM response
            if current_price == 0 and llm_responses.get('technical'):
                tech_content = llm_responses['technical'].get('content', '')
                if isinstance(tech_content, dict):
                    tech_content = json.dumps(tech_content)
                elif not isinstance(tech_content, str):
                    tech_content = str(tech_content)
                import re
                price_match = re.search(r'"current_price":\s*([\d.]+)', tech_content)
                if price_match:
                    current_price = float(price_match.group(1))
            
            # NEW APPROACH: Extract everything from existing LLM responses
            symbol_logger.info("=== NEW DIRECT EXTRACTION APPROACH ===")
            
            # Extract comprehensive SEC analysis data
            sec_data = self._extract_sec_comprehensive_data(llm_responses)
            symbol_logger.info(f"SEC data extracted: business_quality={sec_data.get('business_quality_score', 0):.1f}, financial_health={sec_data.get('financial_health_score', 0):.1f}")
            
            # Extract technical indicators
            tech_indicators = self._extract_technical_indicators(llm_responses)
            symbol_logger.info(f"Technical indicators extracted: trend={tech_indicators.get('trend_direction', 'N/A')}, support_levels={len(tech_indicators.get('support_levels', []))}")
            
            # Assess data quality from SEC analysis first, fallback to calculated
            if sec_data.get('data_quality_score', 0) > 0:
                data_quality = sec_data['data_quality_score']
                symbol_logger.info(f"Using SEC data quality score: {data_quality:.2f}")
            else:
                data_quality = self._assess_data_quality(llm_responses, latest_data)
                symbol_logger.info(f"Using calculated data quality score: {data_quality:.2f}")
            
            # Check if we can use direct extraction optimization
            use_direct_extraction = sec_data and tech_indicators
            
            if use_direct_extraction:
                symbol_logger.info("OPTIMIZATION: Using direct LLM extraction - skipping traditional synthesis")
                # We'll create the recommendation directly later, for now just set a flag
                direct_extraction_data = {
                    'sec_data': sec_data,
                    'tech_indicators': tech_indicators,
                    'current_price': current_price,
                    'overall_score': overall_score
                }
                # Set placeholder values for direct extraction
                synthesis_prompt = "Direct extraction from LLM analysis responses - no synthesis prompt needed"
                synthesis_response = ""  # Will be set later
                synthesis_metadata = {"source": "direct_extraction", "model": "comprehensive_analysis"}
            
            # Traditional synthesis path (or minimal fallback)
            symbol_logger.info("Preparing traditional synthesis path")
            
            # Use prompt manager for synthesis
            from utils.prompt_manager import get_prompt_manager
            prompt_manager = get_prompt_manager()
            
            # Prepare data for synthesis prompt
            from utils.synthesis_helpers import format_fundamental_data_for_synthesis, format_technical_data_for_synthesis, get_performance_data
            
            fundamental_data_str = format_fundamental_data_for_synthesis(llm_responses.get('fundamental', {}))
            technical_data_str = format_technical_data_for_synthesis(llm_responses.get('technical', {}))
            
            # Get peer comparison data
            symbol_logger.info("Fetching peer comparison data")
            peer_analyzer = get_peer_comparison_analyzer()
            peer_comparison = peer_analyzer.get_peer_comparison(symbol)
            
            # Generate synthesis using LLM (but only if not using direct extraction)
            model_name = self.config.ollama.models.get('synthesis', 'llama3.1:8b-instruct-q8_0')
            
            if not use_direct_extraction:
                # Only run traditional synthesis if direct extraction failed
                # Generate synthesis using LLM
                model_name = self.config.ollama.models.get('synthesis', 'llama3.1:8b-instruct-q8_0')
                
                # Choose synthesis approach based on mode
                if synthesis_mode == 'quarterly':
                    symbol_logger.info(f"Using quarterly synthesis mode (last N quarters + technical analysis)")
                    synthesis_prompt = self._create_quarterly_synthesis_prompt(
                        symbol, llm_responses, latest_data, prompt_manager
                    )
                else:
                    # Determine if we should use peer-enhanced synthesis
                    use_peer_synthesis = (
                        peer_comparison and 
                        peer_comparison.get('company_ratios') and 
                        peer_comparison.get('peer_statistics') and
                        len(peer_comparison.get('peer_statistics', {})) > 5  # At least 5 metrics
                    )
                    
                    if use_peer_synthesis:
                        symbol_logger.info(f"Using peer-enhanced synthesis with {peer_comparison.get('peers_analyzed', 0)} peers")
                        # Use peer-enhanced prompt
                        synthesis_prompt = prompt_manager.render_investment_synthesis_peer_prompt(
                            symbol=symbol,
                            analysis_date=datetime.now().strftime('%Y-%m-%d'),
                            current_price=latest_data.get('current_price', 0.0),
                            sector=peer_comparison.get('peer_group', {}).get('sector', 'N/A'),
                            industry=peer_comparison.get('peer_group', {}).get('industry', 'N/A'),
                            fundamental_data=fundamental_data_str,
                            technical_data=technical_data_str,
                            latest_market_data=str(latest_data),
                            peer_list=peer_comparison.get('peer_group', {}).get('peers', [])[:10],
                            company_ratios=peer_comparison.get('company_ratios', {}),
                            peer_statistics=peer_comparison.get('peer_statistics', {}),
                            relative_position=peer_comparison.get('relative_position', {})
                        )
                    else:
                        symbol_logger.info("Using comprehensive synthesis mode")
                        
                        # Debug: Log available fundamental keys
                        fundamental_keys = list(llm_responses.get('fundamental', {}).keys())
                        symbol_logger.info(f"Available fundamental keys: {fundamental_keys}")
                        
                        # Extract comprehensive analysis and quarterly data
                        comprehensive_analysis = ""
                        quarterly_analyses = []
                        
                        # Get comprehensive analysis if available
                        if 'comprehensive' in llm_responses.get('fundamental', {}):
                            comp_data = llm_responses['fundamental']['comprehensive']
                            content = comp_data.get('content', comp_data)
                            symbol_logger.info(f"Found comprehensive analysis, content type: {type(content)}")
                            if isinstance(content, dict):
                                comprehensive_analysis = json.dumps(content, indent=2)
                                symbol_logger.info(f"Comprehensive analysis length: {len(comprehensive_analysis)}")
                            else:
                                comprehensive_analysis = str(content)
                        else:
                            symbol_logger.warning("No comprehensive analysis found in fundamental responses")
                        
                        # Get quarterly analyses
                        for key, resp in llm_responses.get('fundamental', {}).items():
                            if key != 'comprehensive':
                                qa = {
                                    'form_type': resp.get('form_type', 'Unknown'),
                                    'period': resp.get('period', 'Unknown'),
                                    'content': resp.get('content', {})
                                }
                                quarterly_analyses.append(qa)
                        
                        # Sort quarterly analyses by period
                        quarterly_analyses.sort(key=lambda x: x['period'])
                        
                        # Use comprehensive synthesis template
                        synthesis_prompt = prompt_manager.render_investment_synthesis_comprehensive_prompt(
                            symbol=symbol,
                            analysis_date=datetime.now().strftime('%Y-%m-%d'),
                            current_price=latest_data.get('current_price', 0.0),
                            comprehensive_analysis=comprehensive_analysis,
                            quarterly_analyses=quarterly_analyses,
                            quarterly_count=len(quarterly_analyses),
                            technical_analysis=technical_data_str,
                            market_data=latest_data
                        )
            
            # Enhanced system prompt for institutional-grade analysis
            system_prompt = """You are a senior portfolio manager and CFA charterholder with 25+ years of institutional investment experience. You excel at:

• Synthesizing complex multi-source financial analyses into actionable investment decisions
• Risk-adjusted portfolio construction for $2B+ institutional mandates
• Quantitative valuation analysis across market cycles and economic regimes
• Technical analysis integration with fundamental research for optimal timing
• ESG integration and fiduciary standard investment processes

Your responses must be precise, quantitative, and suitable for institutional investment committees. Focus on risk-adjusted returns, position sizing discipline, and clear execution frameworks. Provide specific price targets, stop-losses, and measurable investment criteria."""
            
            # Check cache first for synthesis response
            # Determine fiscal period for synthesis cache lookup
            fiscal_year, fiscal_period = self._get_latest_fiscal_period()
            
            # Use synthesis mode-specific cache key
            llm_type = f'synthesis_{synthesis_mode}'  # synthesis_comprehensive or synthesis_quarterly
            cache_key = {
                'symbol': symbol,
                'form_type': 'SYNTHESIS',  # Use consistent intelligent default
                'period': f"{fiscal_year}-{fiscal_period}",
                'fiscal_year': fiscal_year,  # Separate key for file pattern
                'fiscal_period': fiscal_period,  # Separate key for file pattern
                'llm_type': llm_type
            }
            
            cached_response = self.cache_manager.get(CacheType.LLM_RESPONSE, cache_key)
            
            if cached_response:
                symbol_logger.info(f"Using cached synthesis response for {symbol}")
                self.main_logger.info(f"Cache HIT for synthesis: {symbol}")
                
                # Use cached response directly (already processed by LLM facade)
                synthesis_response = cached_response.get('response', {})
                processing_time_ms = cached_response.get('metadata', {}).get('processing_time_ms', 0)
            else:
                symbol_logger.info(f"No cached synthesis found, generating with {model_name}")
                self.main_logger.info(f"Cache MISS for synthesis: {symbol}, generating with {model_name} (32K context)")
                
                start_time = time.time()
                # Use queue-based processing for synthesis
                from patterns.llm.llm_interfaces import LLMTaskType
                synthesis_task_data = {
                    'symbol': symbol,
                    'fundamental_data': fundamental_data_str,
                    'technical_data': technical_data_str,
                    'latest_data': latest_data,
                    'prompt': synthesis_prompt
                }
                
                synthesis_result = self.ollama.generate_response(
                    task_type=LLMTaskType.SYNTHESIS,
                    data=synthesis_task_data
                )
                
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # Use direct response from LLM facade (already processed)
                synthesis_response = synthesis_result
                
                # Save synthesis LLM response through cache manager
                self._save_synthesis_llm_response(symbol, synthesis_prompt, synthesis_response, processing_time_ms, synthesis_mode)
            
            symbol_logger.info(f"Synthesis response generated in {processing_time_ms}ms")
            
            # Debug: Check what the synthesis response contains
            self.main_logger.info(f"Synthesis response type: {type(synthesis_response)}")
            self.main_logger.info(f"Synthesis response length: {len(str(synthesis_response))}")
            self.main_logger.info(f"Synthesis response preview: {str(synthesis_response)}")
            
            # Parse JSON synthesis response with metadata
            synthesis_metadata = {
                'model': model_name,
                'processing_time_ms': processing_time_ms,
                'symbol': symbol,
                'analysis_type': 'investment_synthesis'
            }
            
            # DEBUG: Inspect incoming synthesis response data
            symbol_logger.info("=== SYNTHESIS RESPONSE DEBUG START ===")
            symbol_logger.info(f"synthesis_response type: {type(synthesis_response)}")
            symbol_logger.info(f"synthesis_response keys (if dict): {list(synthesis_response.keys()) if isinstance(synthesis_response, dict) else 'N/A'}")
            
            if isinstance(synthesis_response, dict):
                symbol_logger.info(f"synthesis_response content preview: {str(synthesis_response)[:300]}...")
                # Check for common response formats
                if 'content' in synthesis_response:
                    symbol_logger.info(f"Found 'content' key, type: {type(synthesis_response['content'])}")
                    symbol_logger.info(f"Content preview: {str(synthesis_response['content'])[:200]}...")
                if 'response' in synthesis_response:
                    symbol_logger.info(f"Found 'response' key, type: {type(synthesis_response['response'])}")
                if 'overall_score' in synthesis_response:
                    symbol_logger.info("Response appears to already be parsed JSON")
            else:
                symbol_logger.info(f"synthesis_response preview: {str(synthesis_response)[:300]}...")
            
            symbol_logger.info(f"synthesis_metadata: {synthesis_metadata}")
            symbol_logger.info("=== SYNTHESIS RESPONSE DEBUG END ===")

            # Robust JSON validation with fallback handling
            try:
                symbol_logger.info("BEFORE validation: Calling prompt_manager.validate_json_response")
                validated_response = prompt_manager.validate_json_response(synthesis_response, synthesis_metadata)
                symbol_logger.info(f"AFTER validation: validated_response type: {type(validated_response)}")
                symbol_logger.info(f"AFTER validation: validated_response keys: {list(validated_response.keys()) if isinstance(validated_response, dict) else 'N/A'}")
                
                # Check if validation failed
                if validated_response.get('error'):
                    symbol_logger.warning(f"JSON validation failed: {validated_response['error']}")
                    symbol_logger.warning(f"Validation error details: {validated_response.get('details', 'No details')}")
                    # Try to extract any partial JSON or create fallback
                    ai_recommendation = self._create_fallback_recommendation(synthesis_response, symbol, overall_score)
                else:
                    ai_recommendation = validated_response['value']
                    symbol_logger.info(f"Validation successful, ai_recommendation type: {type(ai_recommendation)}")
                    
            except Exception as e:
                import traceback
                symbol_logger.error(f"EXCEPTION in JSON validation: {str(e)}")
                symbol_logger.error(f"Exception type: {type(e).__name__}")
                symbol_logger.error(f"Exception traceback: {traceback.format_exc()}")
                symbol_logger.error(f"Raw LLM response (first 500 chars): {str(synthesis_response)[:500]}")
                
                # Create a fallback recommendation based on computed scores
                ai_recommendation = self._create_fallback_recommendation(synthesis_response, symbol, overall_score)
                symbol_logger.info("Created fallback recommendation due to JSON parsing failure")
            
            # OPTIMIZATION: Use direct extraction if available
            if use_direct_extraction and 'direct_extraction_data' in locals():
                symbol_logger.info("OPTIMIZATION: Overriding synthesis with direct extraction recommendation")
                ai_recommendation = self._create_recommendation_from_llm_data(
                    symbol,
                    direct_extraction_data['sec_data'],
                    direct_extraction_data['tech_indicators'],
                    direct_extraction_data['current_price'],
                    direct_extraction_data['overall_score']
                )
                synthesis_response = json.dumps(ai_recommendation)
                synthesis_metadata = {"source": "direct_extraction", "model": "comprehensive_analysis"}
                symbol_logger.info("Direct extraction recommendation created successfully")
            
            # Handle different response types and capture additional insights
            additional_insights = []
            additional_risks = []
            
            # DEBUG: Inspect final ai_recommendation structure
            symbol_logger.info("=== AI_RECOMMENDATION DEBUG START ===")
            symbol_logger.info(f"ai_recommendation type: {type(ai_recommendation)}")
            symbol_logger.info(f"ai_recommendation keys: {list(ai_recommendation.keys()) if isinstance(ai_recommendation, dict) else 'N/A'}")
            symbol_logger.info(f"ai_recommendation preview: {str(ai_recommendation)[:300]}...")
            
            thinking_content = ai_recommendation.get('thinking','')
            symbol_logger.info(f"thinking_content length: {len(thinking_content)}")
            symbol_logger.info(f"thinking_content preview: {thinking_content[:100]}..." if thinking_content else "No thinking content")
            
            additional_details = ai_recommendation.get('details', '')
            symbol_logger.info(f"additional_details length: {len(additional_details)}")
            symbol_logger.info(f"additional_details preview: {additional_details[:100]}..." if additional_details else "No additional details")
            
            # Check for standard synthesis fields
            standard_fields = ['overall_score', 'investment_thesis', 'recommendation', 'confidence_level', 'position_size', 'time_horizon', 'risk_reward_ratio', 'key_catalysts', 'downside_risks']
            for field in standard_fields:
                if field in ai_recommendation:
                    field_value = ai_recommendation[field]
                    symbol_logger.info(f"Found {field}: {type(field_value)} = {str(field_value)[:100]}...")
                else:
                    symbol_logger.warning(f"Missing standard field: {field}")
            
            symbol_logger.info("=== AI_RECOMMENDATION DEBUG END ===")
            
            # Create extensible structure for additional insights and evolution
            extensible_insights = self._create_extensible_insights_structure(
                ai_recommendation, thinking_content, additional_details, symbol
            )
            symbol_logger.info(f"Created extensible insights structure with {len(extensible_insights)} sections")
            
            if additional_details:
                symbol_logger.info(f"Mixed response detected - capturing both JSON and additional text details")
                symbol_logger.info(f"Additional details captured: {len(additional_details)} characters")

            if thinking_content:
                symbol_logger.info(f"Captured {len(thinking_content)} chars of thinking/reasoning content")
            
            # Extract insights and risks from additional text details
            if additional_details:
                additional_insights, additional_risks = self._extract_insights_from_text(additional_details)
                symbol_logger.info(f"Extracted {len(additional_insights)} insights and {len(additional_risks)} risks from text details")
                 
            # Also extract from thinking content if present
            if thinking_content:
                think_insights, think_risks = self._extract_insights_from_text(thinking_content)
                additional_insights.extend(think_insights)
                additional_risks.extend(think_risks)
            
            # Extract scores from parsed response or use defaults
            # Use LLM-provided scores if available
            overall_score = ai_recommendation.get('overall_score', overall_score)
            fundamental_score = ai_recommendation.get('fundamental_score', fundamental_score)
            technical_score = ai_recommendation.get('technical_score', technical_score)
            income_score = ai_recommendation.get('income_statement_score', 
                self._extract_income_score(llm_responses, ai_recommendation))
            cashflow_score = ai_recommendation.get('cash_flow_score',
                self._extract_cashflow_score(llm_responses, ai_recommendation))
            balance_score = ai_recommendation.get('balance_sheet_score',
                    self._extract_balance_score(llm_responses, ai_recommendation))
            growth_score = ai_recommendation.get('growth_score',
                    self._extract_growth_score(llm_responses, ai_recommendation))
            value_score = ai_recommendation.get('value_score',
                    self._extract_value_score(llm_responses, ai_recommendation))
            business_quality_score = ai_recommendation.get('business_quality_score',
                    self._extract_business_quality_score(llm_responses, ai_recommendation))

            # Determine final recommendation with risk management
            final_recommendation = self._determine_final_recommendation(overall_score, ai_recommendation, data_quality)
            
            # Calculate price targets and risk levels
            price_target = self._calculate_price_target(symbol, llm_responses, ai_recommendation, current_price)
            stop_loss = self._calculate_stop_loss(current_price, final_recommendation, overall_score)
            
            # Clean symbol of any quotes that might have been added
            clean_symbol = symbol.strip('"\'')
            symbol_logger.info(f"Retrieved ai_recommendation from propmpt response.")
            
            # Create comprehensive recommendation
            recommendation = InvestmentRecommendation(
                symbol=clean_symbol,
                overall_score=overall_score,
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                income_score=income_score,
                cashflow_score=cashflow_score,
                balance_score=balance_score,
                growth_score=growth_score,
                value_score=value_score,
                business_quality_score=business_quality_score,
                recommendation=final_recommendation.get('recommendation', 'HOLD'),
                confidence=final_recommendation.get('confidence', 'LOW'),
                price_target=price_target,
                current_price=current_price,
                investment_thesis=ai_recommendation.get('executive_summary', {}).get('investment_thesis', 
                    ai_recommendation.get('investment_thesis', 'Analysis based on available data')),
                time_horizon=ai_recommendation.get('investment_recommendation', {}).get('time_horizon',
                    ai_recommendation.get('time_horizon', 'MEDIUM-TERM')),
                position_size=self._extract_position_size(ai_recommendation),
                key_catalysts=self._extract_catalysts(ai_recommendation),
                key_risks=self._extract_comprehensive_risks(llm_responses, ai_recommendation, additional_risks),
                key_insights=self._extract_comprehensive_insights(llm_responses, ai_recommendation, additional_insights),
                entry_strategy=ai_recommendation.get('entry_strategy', ''),
                exit_strategy=ai_recommendation.get('exit_strategy', ''),
                stop_loss=stop_loss,
                analysis_timestamp=datetime.utcnow(),
                data_quality_score=data_quality,
                analysis_thinking=ai_recommendation.get('analysis_thinking', thinking_content),
                synthesis_details=ai_recommendation.get('synthesis_details', additional_details)
            )
            
            # Attach extensible insights for enhanced reporting and future evolution
            recommendation.extensible_insights = extensible_insights
            symbol_logger.info(f"Attached extensible insights structure with {len(extensible_insights)} sections")
            
            # Save synthesis results
            symbol_logger.info("Saving synthesis results to database")
            self._save_synthesis_results(symbol, recommendation)
            
            symbol_logger.info(f"Investment synthesis completed: {recommendation.recommendation} ({recommendation.overall_score:.1f}/10)")
            self.main_logger.info(f"✅ Synthesis completed for {symbol}: {recommendation.recommendation} ({recommendation.overall_score:.1f}/10)")
            return recommendation
            
        except Exception as e:
            if 'symbol_logger' in locals():
                symbol_logger.error(f"Investment synthesis failed: {str(e)}")
            self.main_logger.error(f"Error synthesizing analysis for {symbol}: {e}")
            import traceback
            self.main_logger.error(f"Traceback: {traceback.format_exc()}")
            # Raise the exception instead of returning a default recommendation
            raise RuntimeError(f"Investment synthesis failed for {symbol}: {str(e)}")
    
    def generate_report(self, recommendations: List[InvestmentRecommendation], 
                       report_type: str = "synthesis") -> str:
        """
        Generate PDF report from recommendations
        
        Args:
            recommendations: List of investment recommendations
            report_type: Type of report to generate
            
        Returns:
            Path to generated report
        """
        try:
            # Convert recommendations to dict format for report generator
            rec_dicts = []
            for rec in recommendations:
                rec_dict = {
                    'symbol': rec.symbol,
                    'overall_score': rec.overall_score,
                    'fundamental_score': rec.fundamental_score,
                    'technical_score': rec.technical_score,
                    'income_score': rec.income_score,
                    'cashflow_score': rec.cashflow_score,
                    'balance_score': rec.balance_score,
                    'growth_score': rec.growth_score,
                    'value_score': rec.value_score,
                    'business_quality_score': rec.business_quality_score,
                    'data_quality_score': rec.data_quality_score,
                    'recommendation': rec.recommendation,
                    'confidence': rec.confidence,
                    'price_target': rec.price_target,
                    'current_price': rec.current_price,
                    'investment_thesis': rec.investment_thesis,
                    'time_horizon': rec.time_horizon,
                    'position_size': rec.position_size,
                    'key_catalysts': rec.key_catalysts,
                    'key_risks': rec.key_risks,
                    'key_insights': rec.key_insights,
                    'entry_strategy': rec.entry_strategy,
                    'exit_strategy': rec.exit_strategy,
                    'stop_loss': rec.stop_loss,
                    'data_quality_score': rec.data_quality_score
                }
                rec_dicts.append(rec_dict)
            
            # Generate charts
            chart_paths = []
            
            # Generate technical charts for each symbol
            for rec in recommendations:
                # Load price data if available
                price_data_path = Path(self.config.data_dir) / "price_cache" / f"{rec.symbol}.parquet"
                if price_data_path.exists():
                    import pandas as pd
                    price_data = pd.read_parquet(price_data_path)
                    tech_chart = self.chart_generator.generate_technical_chart(rec.symbol, price_data)
                    if tech_chart:
                        chart_paths.append(tech_chart)
            
            # Generate 3D fundamental plot for both single and multi-symbol reports
            # (Single symbol will show position in 3D space relative to ideal scores)
            fundamental_3d = self.chart_generator.generate_3d_fundamental_plot(rec_dicts)
            if fundamental_3d:
                chart_paths.append(fundamental_3d)
            
            # Generate 2D technical vs fundamental plot  
            tech_fund_2d = self.chart_generator.generate_2d_technical_fundamental_plot(rec_dicts)
            if tech_fund_2d:
                chart_paths.append(tech_fund_2d)
            
            # Generate growth vs value plot
            growth_value_plot = self.chart_generator.generate_growth_value_plot(rec_dicts)
            if growth_value_plot:
                chart_paths.append(growth_value_plot)
            
            # Generate report based on type
            if report_type == "weekly":
                report_path = self.weekly_report_generator.generate_weekly_report(
                    portfolio_data=rec_dicts,
                    market_summary=self._get_market_summary(),
                    performance_data=self._calculate_portfolio_performance(rec_dicts)
                )
            else:
                report_path = self.report_generator.generate_report(
                    recommendations=rec_dicts,
                    report_type=report_type,
                    include_charts=chart_paths
                )
            
            self.main_logger.info(f"📊 Generated {report_type} report: {report_path}")
            return report_path
            
        except Exception as e:
            self.main_logger.error(f"Error generating report: {e}")
            raise
    
    def _fetch_llm_responses(self, symbol: str) -> Dict[str, Dict]:
        """Fetch ALL LLM responses for comprehensive synthesis including 8 quarterly + comprehensive + technical"""
        self.main_logger.info(f"Fetching LLM responses for {symbol}")
        
        try:
            llm_responses = {
                'fundamental': {},
                'technical': None
            }
            
            # 1. FETCH COMPREHENSIVE FUNDAMENTAL ANALYSIS
            fiscal_year, fiscal_period = self._get_latest_fiscal_period()
            comp_cache_key = {
                'symbol': symbol,
                'form_type': 'COMPREHENSIVE',
                'period': f"{fiscal_year}-FY",
                'llm_type': 'sec'
            }
            
            comp_resp = self.cache_manager.get(CacheType.LLM_RESPONSE, comp_cache_key)
            if comp_resp:
                response_data = comp_resp.get('response', {}) if isinstance(comp_resp.get('response'), dict) else comp_resp.get('response', {})
                llm_responses['fundamental']['comprehensive'] = {
                    'content': response_data,
                    'metadata': comp_resp.get('metadata', {}),
                    'form_type': 'COMPREHENSIVE',
                    'period': f"{fiscal_year}-FY"
                }
                self.main_logger.info(f"✅ Fetched comprehensive fundamental analysis for {symbol}")
            
            # 2. FETCH ALL INDIVIDUAL QUARTERLY ANALYSES (for quarter-by-quarter trends)
            sec_responses = self.llm_dao.get_llm_responses_by_symbol(symbol, llm_type='sec')
            quarterly_count = 0
            
            for resp in sec_responses:
                form_type = resp['form_type']
                period = resp['period']
                
                # Skip comprehensive (already fetched above)
                if form_type == 'COMPREHENSIVE':
                    continue
                
                # Parse period for cache key
                if '-' in period:
                    period_parts = period.split('-')
                    period_fiscal_year = int(period_parts[0])
                    period_fiscal_period = period_parts[1]
                else:
                    period_fiscal_year = fiscal_year
                    period_fiscal_period = 'FY'
                
                cache_key = {
                    'symbol': symbol,
                    'form_type': form_type,
                    'period': period,
                    'llm_type': 'sec'
                }
                
                cached_resp = self.cache_manager.get(CacheType.LLM_RESPONSE, cache_key)
                
                if cached_resp:
                    response_data = cached_resp.get('response', {}) if isinstance(cached_resp.get('response'), dict) else cached_resp.get('response', {})
                    
                    key = f"{form_type}_{period}"
                    llm_responses['fundamental'][key] = {
                        'content': response_data,
                        'metadata': cached_resp.get('metadata', {}),
                        'form_type': form_type,
                        'period': period
                    }
                    quarterly_count += 1
            
            self.main_logger.info(f"✅ Fetched {quarterly_count} individual quarterly analyses for {symbol}")
            
            # 3. FETCH TECHNICAL ANALYSIS
            # Try multiple cache key formats for technical analysis
            technical_cache_keys = [
                {'symbol': symbol, 'llm_type': 'ta'},
                {'symbol': symbol, 'llm_type': 'technical'},
                {'symbol': symbol, 'llm_type': 'technical_analysis'},
                {'symbol': symbol, 'form_type': 'TECHNICAL', 'llm_type': 'ta'},
                {'symbol': symbol, 'period': f"{fiscal_year}-Q1", 'llm_type': 'ta'}
            ]
            
            for tech_key in technical_cache_keys:
                tech_resp = self.cache_manager.get(CacheType.LLM_RESPONSE, tech_key)
                if tech_resp:
                    tech_content = tech_resp.get('response', {}) if isinstance(tech_resp.get('response'), dict) else tech_resp.get('response', '')
                    llm_responses['technical'] = {
                        'content': tech_content,
                        'metadata': tech_resp.get('metadata', {}),
                        'cache_key_used': tech_key
                    }
                    self.main_logger.info(f"✅ Fetched technical analysis for {symbol} using key: {tech_key}")
                    break
            
            # 4. FALLBACK: Check file-based technical analysis cache
            if not llm_responses['technical']:
                try:
                    tech_file_path = f"data/llm_cache/{symbol}/response_technical_indicators.txt"
                    if Path(tech_file_path).exists():
                        with open(tech_file_path, 'r') as f:
                            tech_content = f.read()
                        llm_responses['technical'] = {
                            'content': tech_content,
                            'metadata': {'source': 'file_fallback'},
                            'cache_key_used': 'file_fallback'
                        }
                        self.main_logger.info(f"✅ Fetched technical analysis from file for {symbol}")
                except Exception as e:
                    self.main_logger.warning(f"Failed to fetch technical analysis from file: {e}")
            
            if not llm_responses['technical']:
                self.main_logger.warning(f"❌ No technical analysis found for {symbol}")
                llm_responses['technical'] = None
            
            # SUMMARY LOG
            fundamental_count = len(llm_responses['fundamental'])
            technical_count = 1 if llm_responses['technical'] else 0
            self.main_logger.info(f"📊 SYNTHESIS DATA SUMMARY: Retrieved {fundamental_count} fundamental analyses "
                                f"(including {'✅ comprehensive + ' if 'comprehensive' in llm_responses['fundamental'] else '❌ no comprehensive, '}"
                                f"{fundamental_count-1 if 'comprehensive' in llm_responses['fundamental'] else fundamental_count} quarterly) "
                                f"and {technical_count} technical analysis for {symbol}")
            
            return llm_responses
            
        except Exception as e:
            self.main_logger.error(f"Error fetching LLM responses: {e}")
            return {'fundamental': {}, 'technical': None}
    
    def _fetch_latest_data(self, symbol: str) -> Dict:
        """Fetch latest fundamental and technical data from parquet files"""
        self.main_logger.info(f"Fetching latest data for {symbol}")
        
        try:
            # Fetch latest technical data from parquet file (preferred) or CSV fallback
            technical_data = {}
            
            # First try parquet format (compressed, efficient)
            parquet_data_path = Path(self.config.data_dir) / "technical_cache" / symbol / f"technical_data_{symbol}.parquet"
            csv_data_path = Path(self.config.data_dir) / "technical_cache" / symbol / f"technical_data_{symbol}.csv"
            
            self.main_logger.info(f"Looking for technical data - Parquet: {parquet_data_path}, CSV: {csv_data_path}")
            
            # Try parquet first, then CSV fallback
            technical_data_path = None
            if parquet_data_path.exists():
                technical_data_path = parquet_data_path
                read_func = lambda path: pd.read_parquet(path)
                file_type = "parquet"
            elif csv_data_path.exists():
                technical_data_path = csv_data_path
                read_func = lambda path: pd.read_csv(path)
                file_type = "CSV"
            
            if technical_data_path:
                import pandas as pd
                try:
                    # Read data file with technical analysis data
                    df = read_func(technical_data_path)
                    self.main_logger.info(f"Successfully read {file_type} file with {len(df)} rows from {technical_data_path}")
                    
                    if not df.empty:
                        # Get the latest row (most recent data)
                        latest_row = df.iloc[-1]
                        
                        # Extract technical data from the latest row
                        # Handle volume which may be comma-formatted string (CSV) or proper numeric (parquet)
                        volume_raw = latest_row.get('Volume', 0)
                        if isinstance(volume_raw, str):
                            volume = int(volume_raw.replace(',', ''))
                        else:
                            volume = int(volume_raw)
                        
                        technical_data = {
                            'current_price': float(latest_row.get('Close', 0)),
                            'price_change_1d': float(latest_row.get('Price_Change_1D', 0)),  # Use the correct column name
                            'price_change_1w': float(latest_row.get('price_change_1w', 0)),
                            'price_change_1m': float(latest_row.get('price_change_1m', 0)),
                            'rsi': float(latest_row.get('RSI_14', 50)),
                            'macd': float(latest_row.get('MACD', 0)),
                            'sma_20': float(latest_row.get('SMA_20', 0)),
                            'sma_50': float(latest_row.get('SMA_50', 0)),
                            'sma_200': float(latest_row.get('SMA_200', 0)),
                            'volume': volume,
                            'analysis_date': str(latest_row.get('Date', 'Unknown'))
                        }
                        
                        self.main_logger.info(f"Loaded technical data from CSV: current_price=${technical_data['current_price']:.2f}")
                    else:
                        self.main_logger.warning(f"Empty CSV file for {symbol}")
                        
                except Exception as e:
                    self.main_logger.error(f"Error reading CSV file for {symbol}: {e}")
            else:
                self.main_logger.warning(f"CSV file not found for {symbol}: {technical_data_path}")
            
            return {
                'fundamental': {},
                'technical': technical_data,
                'current_price': technical_data.get('current_price', 0)
            }
            
        except Exception as e:
            self.main_logger.error(f"Error fetching latest data for {symbol}: {e}")
            # Fail immediately as requested - no fallbacks that give wrong answers
            raise RuntimeError(f"Failed to fetch latest data for {symbol}: {e}")
    
    def _calculate_fundamental_score(self, llm_responses: Dict) -> float:
        """Calculate fundamental score from LLM responses"""
        fundamental_responses = llm_responses.get('fundamental', {})
        if not fundamental_responses:
            return 0.0  # Clear fallback - no data available
        
        # First try to get from comprehensive analysis
        if 'comprehensive' in fundamental_responses:
            comp_resp = fundamental_responses['comprehensive']
            content = comp_resp.get('content', comp_resp)
            
            # Handle structured response
            if isinstance(content, dict):
                # Try financial_health_score first, then overall_score
                if 'financial_health_score' in content:
                    return float(content['financial_health_score'])
                elif 'overall_score' in content:
                    return float(content['overall_score'])
            
            # Handle string response
            elif isinstance(content, str):
                import re
                # Try to extract from JSON string
                try:
                    import json
                    parsed = json.loads(content)
                    if 'financial_health_score' in parsed:
                        return float(parsed['financial_health_score'])
                    elif 'overall_score' in parsed:
                        return float(parsed['overall_score'])
                except:
                    # Fall back to regex
                    score_match = re.search(r'(?:Financial Health|Overall|Score)[:\s]*(\d+(?:\.\d+)?)/10', content)
                    if score_match:
                        return float(score_match.group(1))
        
        # If no comprehensive, try averaging quarterly scores
        scores = []
        for key, response in fundamental_responses.items():
            if key == 'comprehensive':
                continue
            content = response.get('content', '')
            if isinstance(content, dict) and 'financial_health_score' in content:
                scores.append(float(content['financial_health_score']))
            elif isinstance(content, str):
                import re
                score_match = re.search(r'(?:Financial Health|Overall|Score)[:\s]*(\d+(?:\.\d+)?)/10', content)
                if score_match:
                    scores.append(float(score_match.group(1)))
        
        return sum(scores) / len(scores) if scores else 0.0  # Clear fallback - no scores found
    
    def _calculate_technical_score(self, llm_responses: Dict) -> float:
        """Calculate technical score from structured JSON LLM response"""
        technical_response = llm_responses.get('technical')
        if not technical_response:
            return 0.0  # Clear fallback - no technical data
        
        content = technical_response.get('content', '')
        
        # First try to parse as structured JSON (new format)
        if isinstance(content, dict):
            # Check for new structured format
            if 'technical_score' in content:
                score_data = content['technical_score']
                if isinstance(score_data, dict):
                    return float(score_data.get('score', 0.0))
                return float(score_data)
        elif isinstance(content, str):
            # Handle file format with headers - extract JSON part
            json_content = content
            if "=== AI RESPONSE ===" in content:
                json_start = content.find("=== AI RESPONSE ===") + len("=== AI RESPONSE ===")
                json_content = content[json_start:].strip()
            
            try:
                # Try to parse JSON from string
                parsed = json.loads(json_content)
                if 'technical_score' in parsed:
                    score_data = parsed['technical_score']
                    if isinstance(score_data, dict):
                        return float(score_data.get('score', 0.0))
                    return float(score_data)
            except json.JSONDecodeError:
                pass
            
            # Fall back to regex for legacy format
            import re
            score_match = re.search(r'(?:TECHNICAL[_\s]SCORE|technical_score)[:\s]*(\d+(?:\.\d+)?)', json_content, re.IGNORECASE)
            if score_match:
                return float(score_match.group(1))
        
        return 0.0  # Clear fallback - no score found in response
    
    def _extract_technical_indicators(self, llm_responses: Dict) -> Dict:
        """Extract technical indicators from structured technical analysis JSON response"""
        technical_response = llm_responses.get('technical')
        if not technical_response:
            return {}
        
        content = technical_response.get('content', '')
        indicators = {}
        
        # First try to parse as structured JSON
        if isinstance(content, dict):
            # New structured format with comprehensive technical data
            indicators = {
                'technical_score': content.get('technical_score', {}).get('score', 0.0),
                'trend_direction': content.get('trend_analysis', {}).get('primary_trend', 'NEUTRAL'),
                'trend_strength': content.get('trend_analysis', {}).get('trend_strength', 'WEAK'),
                'support_levels': [
                    content.get('support_resistance', {}).get('immediate_support', 0.0),
                    content.get('support_resistance', {}).get('major_support', 0.0)
                ],
                'resistance_levels': [
                    content.get('support_resistance', {}).get('immediate_resistance', 0.0),
                    content.get('support_resistance', {}).get('major_resistance', 0.0)
                ],
                'fibonacci_levels': content.get('support_resistance', {}).get('fibonacci_levels', {}),
                'momentum_signals': self._extract_momentum_signals(content),
                'risk_factors': content.get('risk_factors', []),
                'key_insights': content.get('key_insights', []),
                'catalysts': content.get('catalysts', []),
                'time_horizon': content.get('recommendation', {}).get('time_horizon', 'MEDIUM'),
                'recommendation': content.get('recommendation', {}).get('technical_rating', 'HOLD'),
                'confidence': content.get('recommendation', {}).get('confidence', 'MEDIUM'),
                'position_sizing': content.get('recommendation', {}).get('position_sizing', 'MODERATE'),
                'entry_strategy': content.get('entry_exit_strategy', {}),
                'volume_analysis': content.get('volume_analysis', {}),
                'volatility_analysis': content.get('volatility_analysis', {}),
                'sector_relative_strength': content.get('sector_relative_strength', {})
            }
        elif isinstance(content, str):
            try:
                # Handle file format with headers - extract JSON part
                json_content = content
                if "=== AI RESPONSE ===" in content:
                    json_start = content.find("=== AI RESPONSE ===") + len("=== AI RESPONSE ===")
                    json_content = content[json_start:].strip()
                
                # Try to parse JSON from string
                parsed = json.loads(json_content)
                indicators = {
                    'technical_score': parsed.get('technical_score', 0.0),
                    'trend_direction': parsed.get('trend_direction', 'NEUTRAL'),
                    'trend_strength': parsed.get('trend_strength', 'WEAK'),
                    'support_levels': parsed.get('support_levels', []),
                    'resistance_levels': parsed.get('resistance_levels', []),
                    'fibonacci_levels': parsed.get('support_resistance', {}).get('fibonacci_levels', {}),
                    'momentum_signals': parsed.get('momentum_signals', []),
                    'risk_factors': parsed.get('risk_factors', []),
                    'key_insights': parsed.get('key_insights', []),
                    'catalysts': parsed.get('catalysts', []),
                    'time_horizon': parsed.get('time_horizon', 'MEDIUM'),
                    'recommendation': parsed.get('recommendation', 'HOLD'),
                    'confidence': parsed.get('confidence', 'MEDIUM'),
                    'position_sizing': 'MODERATE',  # Use default since not in our format
                    'entry_strategy': {},  # Use default since not in our format
                    'volume_analysis': {},  # Use default since not in our format
                    'volatility_analysis': {},  # Use default since not in our format
                    'sector_relative_strength': {}  # Use default since not in our format
                }
            except json.JSONDecodeError:
                # Fall back to legacy format extraction
                indicators = self._extract_legacy_technical_indicators(content)
        
        # Filter out zero values from support/resistance
        indicators['support_levels'] = [s for s in indicators.get('support_levels', []) if s > 0]
        indicators['resistance_levels'] = [r for r in indicators.get('resistance_levels', []) if r > 0]
        
        return indicators
    
    def _extract_momentum_signals(self, content: Dict) -> List[str]:
        """Extract momentum signals from technical analysis response"""
        signals = []
        
        momentum = content.get('momentum_analysis', {})
        if momentum:
            # RSI signals
            rsi = momentum.get('rsi_14', 0)
            rsi_assessment = momentum.get('rsi_assessment', '')
            if rsi and rsi_assessment:
                signals.append(f"RSI ({rsi:.1f}) indicates {rsi_assessment.lower()} conditions")
            
            # MACD signals
            macd = momentum.get('macd', {})
            if macd.get('signal'):
                signals.append(f"MACD shows {macd['signal'].lower()} momentum")
            
            # Stochastic signals
            stoch = momentum.get('stochastic', {})
            if stoch.get('signal'):
                signals.append(f"Stochastic indicates {stoch['signal'].lower()} conditions")
        
        # Volume signals
        volume = content.get('volume_analysis', {})
        if volume.get('volume_trend'):
            signals.append(f"Volume trend is {volume['volume_trend'].lower()}")
        
        return signals
    
    def _extract_legacy_technical_indicators(self, content: str) -> Dict:
        """Extract technical indicators from legacy format response"""
        import re
        indicators = {}
        
        support_match = re.search(r'support_levels[:\s]*\[([^\]]+)\]', content, re.IGNORECASE)
        resistance_match = re.search(r'resistance_levels[:\s]*\[([^\]]+)\]', content, re.IGNORECASE)
        trend_match = re.search(r'trend_direction[:\s]*["\']?([A-Z]+)["\']?', content, re.IGNORECASE)
        
        if support_match:
            try:
                indicators['support_levels'] = [float(x.strip()) for x in support_match.group(1).split(',')]
            except:
                indicators['support_levels'] = []
                
        if resistance_match:
            try:
                indicators['resistance_levels'] = [float(x.strip()) for x in resistance_match.group(1).split(',')]
            except:
                indicators['resistance_levels'] = []
                
        if trend_match:
            indicators['trend_direction'] = trend_match.group(1).upper()
        
        return indicators
    
    def _extract_sec_comprehensive_data(self, llm_responses: Dict) -> Dict:
        """Extract all valuable data from SEC comprehensive analysis"""
        fundamental_responses = llm_responses.get('fundamental', {})
        if 'comprehensive' not in fundamental_responses:
            return {}
        
        comp_resp = fundamental_responses['comprehensive']
        content = comp_resp.get('content', comp_resp)
        
        # Handle structured response
        if isinstance(content, dict):
            return {
                'financial_health_score': content.get('financial_health_score', 0.0),
                'business_quality_score': content.get('business_quality_score', 0.0),
                'growth_prospects_score': content.get('growth_prospects_score', 0.0),
                'data_quality_score': content.get('data_quality_score', {}).get('score', 0.0) if isinstance(content.get('data_quality_score'), dict) else content.get('data_quality_score', 0.0),
                'overall_score': content.get('overall_score', 0.0),
                'investment_thesis': content.get('investment_thesis', ''),
                'key_insights': content.get('key_insights', []),
                'key_risks': content.get('key_risks', []),
                'trend_analysis': content.get('trend_analysis', {}),
                'confidence_level': content.get('confidence_level', 'MEDIUM')
            }
        
        # Handle string response (legacy format)
        elif isinstance(content, str):
            try:
                parsed = json.loads(content)
                return {
                    'financial_health_score': parsed.get('financial_health_score', 0.0),
                    'business_quality_score': parsed.get('business_quality_score', 0.0),
                    'growth_prospects_score': parsed.get('growth_prospects_score', 0.0),
                    'data_quality_score': parsed.get('data_quality_score', {}).get('score', 0.0) if isinstance(parsed.get('data_quality_score'), dict) else parsed.get('data_quality_score', 0.0),
                    'overall_score': parsed.get('overall_score', 0.0),
                    'investment_thesis': parsed.get('investment_thesis', ''),
                    'key_insights': parsed.get('key_insights', []),
                    'key_risks': parsed.get('key_risks', []),
                    'trend_analysis': parsed.get('trend_analysis', {}),
                    'confidence_level': parsed.get('confidence_level', 'MEDIUM')
                }
            except:
                return {}
        
        return {}
    
    def _create_recommendation_from_llm_data(self, symbol: str, sec_data: Dict, tech_indicators: Dict, current_price: float, overall_score: float) -> Dict:
        """Create investment recommendation by combining SEC comprehensive and technical analysis data"""
        
        # Extract key scores and data (use fundamental_score instead of financial_health_score)
        business_quality = sec_data.get('business_quality_score', 0.0)
        fundamental_score = sec_data.get('financial_health_score', 0.0)  # This becomes our fundamental score
        growth_score = sec_data.get('growth_prospects_score', 0.0)  # Use growth_score not growth_prospects
        data_quality = sec_data.get('data_quality_score', 0.0)
        sec_confidence = sec_data.get('confidence_level', 'MEDIUM')
        
        # Technical data
        tech_trend = tech_indicators.get('trend_direction', 'NEUTRAL')
        tech_recommendation = tech_indicators.get('recommendation', 'HOLD')
        support_levels = tech_indicators.get('support_levels', [])
        resistance_levels = tech_indicators.get('resistance_levels', [])
        tech_risks = tech_indicators.get('risk_factors', [])
        
        # Combine recommendations - prioritize fundamental for long-term view
        if fundamental_score >= 8.0 and business_quality >= 8.0:
            if tech_trend in ['BULLISH', 'NEUTRAL']:
                final_recommendation = 'BUY'
                confidence = 'HIGH' if tech_trend == 'BULLISH' else 'MEDIUM'
            else:  # BEARISH
                final_recommendation = 'HOLD'  # Strong fundamentals but poor technicals
                confidence = 'MEDIUM'
        elif fundamental_score >= 6.0 and business_quality >= 6.0:
            if tech_trend == 'BULLISH':
                final_recommendation = 'BUY'
                confidence = 'MEDIUM'
            elif tech_trend == 'BEARISH':
                final_recommendation = 'HOLD'
                confidence = 'LOW'
            else:
                final_recommendation = 'HOLD'
                confidence = 'MEDIUM'
        else:  # Weak fundamentals
            if tech_trend == 'BEARISH':
                final_recommendation = 'SELL'
                confidence = 'MEDIUM'
            else:
                final_recommendation = 'HOLD'
                confidence = 'LOW'
        
        # Adjust confidence based on data quality
        if data_quality < 5.0:
            confidence = 'LOW'
        elif data_quality >= 8.0 and confidence == 'MEDIUM':
            confidence = 'HIGH'
        
        # Create combined investment thesis
        sec_thesis = sec_data.get('investment_thesis', '')
        if sec_thesis and tech_indicators:
            investment_thesis = f"{sec_thesis} Technical analysis shows {tech_trend.lower()} trend with {tech_recommendation.lower()} recommendation."
        elif sec_thesis:
            investment_thesis = sec_thesis
        else:
            investment_thesis = f"Based on fundamental score of {fundamental_score:.1f} and business quality of {business_quality:.1f}, with {tech_trend.lower()} technical trend."
        
        # Combine key insights and risks
        sec_insights = sec_data.get('key_insights', [])
        sec_risks = sec_data.get('key_risks', [])
        
        # Add technical insights
        tech_insights = []
        if support_levels:
            tech_insights.append(f"Key support levels at ${', $'.join([f'{s:.2f}' for s in support_levels[:3]])}")
        if resistance_levels:
            tech_insights.append(f"Key resistance levels at ${', $'.join([f'{r:.2f}' for r in resistance_levels[:3]])}")
        
        all_insights = sec_insights + tech_insights
        all_risks = sec_risks + tech_risks
        
        # Calculate position sizing based on combined analysis
        if final_recommendation == 'BUY':
            if confidence == 'HIGH' and business_quality >= 9.0:
                position_size = 'LARGE'
            elif confidence in ['HIGH', 'MEDIUM']:
                position_size = 'MODERATE' 
            else:
                position_size = 'SMALL'
        elif final_recommendation == 'SELL':
            position_size = 'AVOID'
        else:  # HOLD
            position_size = 'SMALL'
        
        # Time horizon based on fundamental strength
        if business_quality >= 8.0 and fundamental_score >= 8.0:
            time_horizon = 'LONG-TERM'
        elif business_quality >= 6.0:
            time_horizon = 'MEDIUM-TERM'
        else:
            time_horizon = 'SHORT-TERM'
        
        # Calculate price targets using support/resistance
        price_target = None
        stop_loss = None
        if resistance_levels and final_recommendation == 'BUY':
            price_target = max(resistance_levels)
        if support_levels and final_recommendation in ['BUY', 'HOLD']:
            stop_loss = min(support_levels) * 0.95  # 5% below support
        
        return {
            'overall_score': overall_score,
            'fundamental_score': fundamental_score,
            'technical_score': tech_indicators.get('technical_score', 0.0),
            'business_quality_score': business_quality,
            'growth_score': growth_score,
            'data_quality_score': data_quality,
            'investment_recommendation': {
                'recommendation': final_recommendation,
                'confidence': confidence
            },
            'investment_thesis': investment_thesis,
            'position_size': position_size,
            'time_horizon': time_horizon,
            'price_target': price_target,
            'stop_loss': stop_loss,
            'key_catalysts': all_insights[:5],  # Top 5 insights as catalysts
            'downside_risks': all_risks[:5],   # Top 5 risks
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'trend_direction': tech_trend,
            'momentum_signals': tech_indicators.get('momentum_signals', []),
            'confidence_level': confidence,
            'source': 'direct_llm_extraction'
        }
    
    def _calculate_weighted_score(self, fundamental_score: float, technical_score: float) -> float:
        """Calculate weighted overall score"""
        if fundamental_score is None or technical_score is None:
            return 5.0
        
        fund_weight = self.config.analysis.fundamental_weight
        tech_weight = self.config.analysis.technical_weight
        
        # Adjust weights for extreme scores
        if fundamental_score >= 8.5 or fundamental_score <= 2.5:
            fund_weight *= 1.2
        
        if technical_score >= 8.5 or technical_score <= 2.5:
            tech_weight *= 1.1
        
        total_weight = fund_weight + tech_weight
        
        if total_weight == 0:
            return 0.0  # Clear fallback - no weights
        
        norm_fund_weight = fund_weight / total_weight
        norm_tech_weight = tech_weight / total_weight
        
        overall_score = (fundamental_score * norm_fund_weight + technical_score * norm_tech_weight)
        
        return round(overall_score, 1)
    
    def _assess_data_quality(self, llm_responses: Dict, latest_data: Dict) -> float:
        """Assess overall data quality and completeness, prioritizing SEC comprehensive analysis
        
        Returns:
            float: Data quality score on 1-10 scale
        """
        # First, try to get data quality score from SEC comprehensive analysis
        comprehensive_analysis = llm_responses.get('fundamental', {}).get('comprehensive', {})
        if isinstance(comprehensive_analysis, dict):
            # Direct score from comprehensive analysis
            if 'data_quality_score' in comprehensive_analysis:
                score_data = comprehensive_analysis['data_quality_score']
                if isinstance(score_data, dict):
                    return float(score_data.get('score', 0.0))  # Already in 1-10 scale
                return float(score_data)  # Already in 1-10 scale
            
            # Extract from response content if it's nested
            content = comprehensive_analysis.get('content', {})
            if isinstance(content, dict) and 'data_quality_score' in content:
                score_data = content['data_quality_score']
                if isinstance(score_data, dict):
                    return float(score_data.get('score', 0.0))  # Already in 1-10 scale
                return float(score_data)  # Already in 1-10 scale
        
        # Fallback to traditional data quality assessment (convert to 1-10 scale)
        quality_score = 0.0
        
        # Check fundamental data availability (max 4 points)
        if llm_responses.get('fundamental'):
            quality_score += 4.0
            if len(llm_responses['fundamental']) >= 3:  # Multiple quarters
                quality_score += 1.0
        
        # Check technical data availability (max 3 points)
        if llm_responses.get('technical'):
            quality_score += 3.0
        
        # Check data freshness (max 2 points)
        if latest_data.get('technical', {}).get('current_price'):
            quality_score += 1.0
        
        if latest_data.get('fundamental'):
            quality_score += 1.0
        
        return min(quality_score, 10.0)  # Cap at 10.0
    
    def _parse_synthesis_response(self, response: str) -> Dict:
        """Parse the synthesis LLM response"""
        import re
        
        result = {
            'recommendation': 'HOLD',
            'confidence': 'MEDIUM',
            'investment_thesis': '',
            'key_catalysts': [],
            'key_risks': [],
            'price_targets': {},
            'position_size': 'MODERATE',
            'time_horizon': 'MEDIUM-TERM',
            'entry_strategy': '',
            'exit_strategy': ''
        }
        
        try:
            # Extract final recommendation
            rec_match = re.search(r'FINAL RECOMMENDATION[:\s]*\*?\*?\s*\[?([A-Z\s]+)\]?', response, re.IGNORECASE)
            if rec_match:
                rec_text = rec_match.group(1).strip().upper()
                if 'STRONG BUY' in rec_text:
                    result['recommendation'] = 'STRONG BUY'
                elif 'STRONG SELL' in rec_text:
                    result['recommendation'] = 'STRONG SELL'
                elif 'BUY' in rec_text:
                    result['recommendation'] = 'BUY'
                elif 'SELL' in rec_text:
                    result['recommendation'] = 'SELL'
                else:
                    result['recommendation'] = 'HOLD'
            
            # Extract confidence level
            conf_match = re.search(r'CONFIDENCE LEVEL[:\s]*\*?\*?\s*\[?([A-Z]+)\]?', response, re.IGNORECASE)
            if conf_match:
                result['confidence'] = conf_match.group(1).strip().upper()
            
            # Extract investment thesis
            thesis_match = re.search(r'INVESTMENT THESIS[:\s]*\*?\*?(.*?)(?=\*\*[A-Z]|\n\n)', response, re.IGNORECASE | re.DOTALL)
            if thesis_match:
                result['investment_thesis'] = thesis_match.group(1).strip()
            
            # Extract catalysts
            catalysts_match = re.search(r'KEY CATALYSTS[:\s]*\*?\*?(.*?)(?=\*\*[A-Z]|\n\n)', response, re.IGNORECASE | re.DOTALL)
            if catalysts_match:
                catalysts_text = catalysts_match.group(1)
                result['key_catalysts'] = [cat.strip() for cat in re.findall(r'[•\-]\s*(.+)', catalysts_text)]
            
            # Extract risks
            risks_match = re.search(r'RISK ASSESSMENT[:\s]*\*?\*?(.*?)(?=\*\*[A-Z]|\n\n)', response, re.IGNORECASE | re.DOTALL)
            if risks_match:
                risks_text = risks_match.group(1)
                result['key_risks'] = [risk.strip() for risk in re.findall(r'[•\-]\s*(.+)', risks_text)]
            
            # Extract price targets
            target_match = re.search(r'12-month.*?Target[:\s]*\$?([\d.]+)', response, re.IGNORECASE)
            if target_match:
                result['price_targets']['12_month'] = float(target_match.group(1))
            
            # Extract position size
            pos_match = re.search(r'POSITION SIZING[:\s]*\*?\*?\s*\[?([A-Z\s\/%]+)\]?', response, re.IGNORECASE)
            if pos_match:
                pos_text = pos_match.group(1).strip().upper()
                if 'LARGE' in pos_text or 'CONCENTRATED' in pos_text:
                    result['position_size'] = 'LARGE'
                elif 'SMALL' in pos_text or 'STARTER' in pos_text:
                    result['position_size'] = 'SMALL'
                else:
                    result['position_size'] = 'MODERATE'
            
            # Extract time horizon
            horizon_match = re.search(r'TIME HORIZON[:\s]*\*?\*?\s*\[?([A-Z\s\-]+)\]?', response, re.IGNORECASE)
            if horizon_match:
                result['time_horizon'] = horizon_match.group(1).strip().upper()
            
        except Exception as e:
            self.main_logger.warning(f"Error parsing synthesis response: {e}")
        
        return result
    
    def _extract_income_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """Extract income statement score from responses"""
        # First check AI recommendation
        if 'income_statement_score' in ai_recommendation:
            return float(ai_recommendation['income_statement_score'])
        
        # Check comprehensive analysis for income statement analysis
        comp_analysis = llm_responses.get('fundamental', {}).get('comprehensive', {})
        content = comp_analysis.get('content', comp_analysis) if isinstance(comp_analysis, dict) else {}
        
        if isinstance(content, dict):
            # Look for income statement analysis section
            income_analysis = content.get('income_statement_analysis', {})
            if income_analysis:
                # Try to extract a score from profitability metrics
                profitability = income_analysis.get('profitability_analysis', {})
                margins = [
                    profitability.get('gross_margin', 0),
                    profitability.get('operating_margin', 0),
                    profitability.get('net_margin', 0)
                ]
                # Convert margins to score (assuming good margins are >15%)
                avg_margin = sum(m for m in margins if m > 0) / len([m for m in margins if m > 0]) if any(m > 0 for m in margins) else 0
                if avg_margin > 0:
                    return min(10.0, max(1.0, avg_margin * 100 / 3))  # Scale to 1-10
        
        # Fallback to fundamental score with adjustment
        base_fundamental = self._calculate_fundamental_score(llm_responses)
        return base_fundamental * 0.9 if base_fundamental > 0 else 0.0
    
    def _extract_cashflow_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """Extract cash flow score from responses"""
        base_fundamental = self._calculate_fundamental_score(llm_responses)
        
        # Look for cash flow keywords
        cashflow_keywords = ['cash flow', 'cash', 'liquidity', 'fcf', 'working capital', 'operating cash']
        cashflow_score_adjustments = []
        
        for resp in llm_responses.get('fundamental', {}).values():
            content = resp.get('content', '')
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            content = content.lower()
            cashflow_mentions = sum(1 for keyword in cashflow_keywords if keyword in content)
            if cashflow_mentions > 3:
                cashflow_score_adjustments.append(0.5)
            elif cashflow_mentions > 0:
                cashflow_score_adjustments.append(0.0)
            else:
                cashflow_score_adjustments.append(-0.5)
        
        adjustment = sum(cashflow_score_adjustments) / len(cashflow_score_adjustments) if cashflow_score_adjustments else 0
        return max(0.0, min(10.0, base_fundamental + adjustment)) if base_fundamental > 0 else 0.0
    
    def _extract_balance_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """Extract balance sheet score from responses"""
        base_fundamental = self._calculate_fundamental_score(llm_responses)
        
        # Look for balance sheet keywords
        balance_keywords = ['asset', 'liability', 'equity', 'debt', 'balance sheet', 'leverage', 'solvency']
        balance_score_adjustments = []
        
        for resp in llm_responses.get('fundamental', {}).values():
            content = resp.get('content', '')
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            content = content.lower()
            balance_mentions = sum(1 for keyword in balance_keywords if keyword in content)
            if balance_mentions > 3:
                balance_score_adjustments.append(0.5)
            elif balance_mentions > 0:
                balance_score_adjustments.append(0.0)
            else:
                balance_score_adjustments.append(-0.5)
        
        adjustment = sum(balance_score_adjustments) / len(balance_score_adjustments) if balance_score_adjustments else 0
        return max(0.0, min(10.0, base_fundamental + adjustment)) if base_fundamental > 0 else 0.0
    
    def _extract_growth_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """Extract growth prospects score from responses"""
        # First check if growth score is in the comprehensive fundamental analysis
        if 'comprehensive' in llm_responses.get('fundamental', {}):
            comp_content = llm_responses['fundamental']['comprehensive'].get('content', {})
            if isinstance(comp_content, dict) and 'growth_prospects_score' in comp_content:
                return float(comp_content['growth_prospects_score'])
        
        # Check AI recommendation for growth assessment
        if 'fundamental_assessment' in ai_recommendation:
            fund_assess = ai_recommendation['fundamental_assessment']
            if 'growth_prospects' in fund_assess:
                # Extract numeric score if available
                growth_data = fund_assess['growth_prospects']
                if isinstance(growth_data, dict) and 'score' in growth_data:
                    return float(growth_data['score'])
        
        # Fallback: analyze growth keywords
        base_fundamental = self._calculate_fundamental_score(llm_responses)
        growth_keywords = ['growth', 'expansion', 'increase', 'momentum', 'acceleration', 'scaling']
        growth_score_adjustments = []
        
        for resp in llm_responses.get('fundamental', {}).values():
            content = resp.get('content', '')
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            content = content.lower()
            growth_mentions = sum(1 for keyword in growth_keywords if keyword in content)
            if growth_mentions > 5:
                growth_score_adjustments.append(1.0)
            elif growth_mentions > 2:
                growth_score_adjustments.append(0.5)
            else:
                growth_score_adjustments.append(0.0)
        
        adjustment = sum(growth_score_adjustments) / len(growth_score_adjustments) if growth_score_adjustments else 0
        return max(0.0, min(10.0, base_fundamental + adjustment)) if base_fundamental > 0 else 0.0
    
    def _extract_value_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """Extract value investment score from responses"""
        # Check for valuation metrics in AI recommendation
        if 'fundamental_assessment' in ai_recommendation:
            fund_assess = ai_recommendation['fundamental_assessment']
            if 'valuation' in fund_assess:
                val_data = fund_assess['valuation']
                if isinstance(val_data, dict) and 'score' in val_data:
                    return float(val_data['score'])
        
        # Look for value indicators
        base_fundamental = self._calculate_fundamental_score(llm_responses)
        value_keywords = ['undervalued', 'discount', 'cheap', 'value', 'pe ratio', 'price to book', 'dividend yield']
        negative_value_keywords = ['overvalued', 'expensive', 'premium', 'overpriced']
        value_score_adjustments = []
        
        for resp in llm_responses.get('fundamental', {}).values():
            content = resp.get('content', '')
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            content = content.lower()
            
            value_mentions = sum(1 for keyword in value_keywords if keyword in content)
            negative_mentions = sum(1 for keyword in negative_value_keywords if keyword in content)
            
            net_value_signal = value_mentions - negative_mentions
            if net_value_signal > 3:
                value_score_adjustments.append(1.0)
            elif net_value_signal > 0:
                value_score_adjustments.append(0.5)
            elif net_value_signal < -3:
                value_score_adjustments.append(-1.0)
            else:
                value_score_adjustments.append(0.0)
        
        adjustment = sum(value_score_adjustments) / len(value_score_adjustments) if value_score_adjustments else 0
        return max(0.0, min(10.0, base_fundamental + adjustment)) if base_fundamental > 0 else 0.0
    
    def _extract_business_quality_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """
        Extract business quality score from SEC comprehensive analysis.
        
        This method works backwards from the comprehensive SEC analysis which aggregates
        quarterly data to assess business quality based on:
        - Core business concepts and tags identified across quarters
        - Revenue quality and consistency patterns  
        - Operational efficiency metrics
        - Competitive positioning indicators
        - Management effectiveness signals
        """
        # First, try to get the business_quality_score directly from SEC comprehensive analysis
        comprehensive_analysis = llm_responses.get('fundamental', {}).get('comprehensive', {})
        if isinstance(comprehensive_analysis, dict):
            # Direct score from comprehensive analysis
            if 'business_quality_score' in comprehensive_analysis:
                score_data = comprehensive_analysis['business_quality_score']
                if isinstance(score_data, dict):
                    return float(score_data.get('score', 5.0))
                return float(score_data)
            
            # Extract from response content if it's nested
            content = comprehensive_analysis.get('content', {})
            if isinstance(content, dict) and 'business_quality_score' in content:
                score_data = content['business_quality_score']
                if isinstance(score_data, dict):
                    return float(score_data.get('score', 5.0))
                return float(score_data)
        
        # If comprehensive analysis is available as string/JSON, parse it
        if isinstance(comprehensive_analysis, str):
            try:
                import json
                parsed = json.loads(comprehensive_analysis)
                if 'business_quality_score' in parsed:
                    return float(parsed['business_quality_score'])
            except:
                pass
        
        # Fallback: Calculate from quarterly analyses patterns
        quarterly_analyses = llm_responses.get('fundamental', {})
        quality_indicators = []
        
        for period_key, analysis in quarterly_analyses.items():
            if period_key == 'comprehensive':  # Skip the comprehensive entry
                continue
                
            content = analysis.get('content', '')
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            
            # Analyze quarterly data for business quality indicators
            quality_score = self._analyze_quarterly_business_quality(content, period_key)
            if quality_score > 0:
                quality_indicators.append(quality_score)
        
        # Calculate average business quality from quarterly analyses
        if quality_indicators:
            avg_quality = sum(quality_indicators) / len(quality_indicators)
            
            # Apply weighting based on data consistency and trends
            consistency_bonus = self._calculate_consistency_bonus(quality_indicators)
            final_score = min(10.0, max(1.0, avg_quality + consistency_bonus))
            
            return final_score
        
        # Ultimate fallback: Return 0 to indicate no business quality score available
        return 0.0
    
    def _analyze_quarterly_business_quality(self, content: str, period: str) -> float:
        """Analyze individual quarterly content for business quality indicators"""
        content_lower = content.lower()
        quality_score = 5.0  # Base score
        
        # Revenue quality indicators
        revenue_quality_keywords = [
            'recurring revenue', 'subscription', 'diversified revenue', 'stable revenue',
            'revenue growth', 'market share', 'competitive advantage', 'moat'
        ]
        
        # Operational excellence indicators  
        operational_keywords = [
            'margin expansion', 'efficiency', 'productivity', 'automation',
            'cost control', 'operating leverage', 'scalability'
        ]
        
        # Innovation and competitive position
        innovation_keywords = [
            'innovation', 'r&d', 'research and development', 'patent', 'technology',
            'differentiation', 'competitive position', 'market leadership'
        ]
        
        # Management effectiveness
        management_keywords = [
            'capital allocation', 'strategic initiative', 'execution', 'guidance',
            'shareholder value', 'dividend', 'buyback', 'investment'
        ]
        
        # Calculate weighted scores for each category
        categories = [
            (revenue_quality_keywords, 1.5),  # Revenue quality most important
            (operational_keywords, 1.2),     # Operational efficiency
            (innovation_keywords, 1.0),      # Innovation capacity  
            (management_keywords, 0.8)       # Management effectiveness
        ]
        
        total_weight = 0
        weighted_score = 0
        
        for keywords, weight in categories:
            category_score = 0
            for keyword in keywords:
                if keyword in content_lower:
                    category_score += 1
            
            # Normalize category score to 0-10 scale
            normalized_score = min(10.0, (category_score / len(keywords)) * 10)
            weighted_score += normalized_score * weight
            total_weight += weight
        
        # Calculate final weighted average
        if total_weight > 0:
            quality_score = weighted_score / total_weight
        
        return max(1.0, min(10.0, quality_score))
    
    def _calculate_consistency_bonus(self, quality_indicators: List[float]) -> float:
        """Calculate bonus for consistent business quality across quarters"""
        if len(quality_indicators) < 2:
            return 0.0
        
        # Calculate standard deviation
        mean_quality = sum(quality_indicators) / len(quality_indicators)
        variance = sum((x - mean_quality) ** 2 for x in quality_indicators) / len(quality_indicators)
        std_dev = variance ** 0.5
        
        # Lower standard deviation = more consistent = higher bonus
        # Scale: 0-1 point bonus based on consistency
        max_bonus = 1.0
        consistency_bonus = max(0.0, max_bonus - (std_dev / 2.0))
        
        return consistency_bonus
    
    def _determine_final_recommendation(self, overall_score: float, ai_recommendation: Dict, data_quality: float) -> Dict:
        """Determine final recommendation with risk management"""
        # Try to get recommendation from structured response first
        if 'investment_recommendation' in ai_recommendation:
            inv_rec = ai_recommendation['investment_recommendation']
            base_recommendation = inv_rec.get('recommendation', 'HOLD')
            confidence = inv_rec.get('confidence_level', 'MEDIUM')
        else:
            # Handle case where recommendation might be a dict due to JSON parsing errors
            rec_data = ai_recommendation.get('recommendation', 'HOLD')
            if isinstance(rec_data, dict):
                base_recommendation = rec_data.get('rating', 'HOLD')
                confidence = rec_data.get('confidence', 'LOW')
            else:
                base_recommendation = rec_data if isinstance(rec_data, str) else 'HOLD'
                confidence = ai_recommendation.get('confidence', 'MEDIUM')
        
        # Adjust for data quality
        if data_quality < 0.5:
            confidence = 'LOW'
            if base_recommendation in ['STRONG BUY', 'STRONG SELL']:
                base_recommendation = base_recommendation.replace('STRONG ', '')
        
        # Adjust based on score thresholds
        if overall_score >= 8.0 and base_recommendation not in ['BUY', 'STRONG BUY']:
            base_recommendation = 'BUY'
        elif overall_score <= 3.0 and base_recommendation not in ['SELL', 'STRONG SELL']:
            base_recommendation = 'SELL'
        elif 4.0 <= overall_score <= 6.0 and base_recommendation in ['STRONG BUY', 'STRONG SELL']:
            base_recommendation = 'HOLD'
        
        return {
            'recommendation': base_recommendation,
            'confidence': confidence
        }
    
    def _calculate_price_target(self, symbol: str, llm_responses: Dict, ai_recommendation: Dict, current_price: float) -> float:
        """Calculate sophisticated price target"""
        # Try to extract from structured AI recommendation first
        if 'investment_recommendation' in ai_recommendation:
            target_data = ai_recommendation['investment_recommendation'].get('target_price', {})
            if target_data.get('12_month_target'):
                return target_data['12_month_target']
        
        # Try legacy format
        ai_targets = ai_recommendation.get('price_targets', {})
        if ai_targets.get('12_month'):
            return ai_targets['12_month']
        
        # Use current price passed in, fallback to reasonable default if price is 0
        if current_price <= 0:
            # Log warning and use a placeholder
            self.main_logger.warning(f"No current price available for {symbol}, using placeholder for target calculation")
            current_price = 100  # Fallback only as last resort
        
        # Extract overall score from different possible locations
        overall_score = 5.0  # Default
        if 'composite_scores' in ai_recommendation:
            overall_score = ai_recommendation['composite_scores'].get('overall_score', 5.0)
        elif 'overall_score' in ai_recommendation:
            overall_score = ai_recommendation.get('overall_score', 5.0)
        
        # Expected return mapping based on score
        if overall_score >= 8.0:
            expected_return = 0.15  # 15% (more conservative for institutional)
        elif overall_score >= 6.5:
            expected_return = 0.10  # 10%
        elif overall_score >= 5.0:
            expected_return = 0.05  # 5%
        else:
            expected_return = -0.05  # -5%
        
        price_target = round(current_price * (1 + expected_return), 2)
        self.main_logger.info(f"Calculated price target for {symbol}: ${price_target:.2f} (current: ${current_price:.2f}, score: {overall_score:.1f})")
        
        return price_target
    
    def _calculate_stop_loss(self, current_price: float, recommendation: Dict, overall_score: float) -> float:
        """Calculate stop loss based on risk management"""
        if not current_price or current_price <= 0:
            return 0
        
        # Base stop loss percentage on score and recommendation
        rec_type = recommendation.get('recommendation', 'HOLD')
        
        if 'STRONG BUY' in rec_type:
            stop_loss_pct = 0.12  # 12% stop loss
        elif 'BUY' in rec_type:
            stop_loss_pct = 0.10  # 10% stop loss
        elif 'HOLD' in rec_type:
            stop_loss_pct = 0.08  # 8% stop loss
        else:  # SELL
            stop_loss_pct = 0.05  # 5% stop loss
        
        # Adjust for overall score
        if overall_score < 4.0:
            stop_loss_pct *= 0.5  # Tighter stop for low conviction
        
        return round(current_price * (1 - stop_loss_pct), 2)
    
    def _extract_position_size(self, ai_recommendation: Dict) -> str:
        """Extract position size recommendation"""
        if 'investment_recommendation' in ai_recommendation:
            pos_sizing = ai_recommendation['investment_recommendation'].get('position_sizing', {})
            weight = pos_sizing.get('recommended_weight', 0.0)
            if weight >= 0.05:
                return 'LARGE'
            elif weight >= 0.03:
                return 'MODERATE'
            elif weight > 0:
                return 'SMALL'
        return ai_recommendation.get('position_size', 'MODERATE')
    
    def _extract_catalysts(self, ai_recommendation: Dict) -> List[str]:
        """Extract key catalysts from recommendation"""
        catalysts = []
        
        # Try structured format first
        if 'key_catalysts' in ai_recommendation:
            cat_data = ai_recommendation['key_catalysts']
            if isinstance(cat_data, list):
                for cat in cat_data[:3]:
                    if isinstance(cat, dict):
                        catalysts.append(cat.get('catalyst', ''))
                    elif isinstance(cat, str):
                        catalysts.append(cat)
        
        # Fallback to simple list
        return catalysts or ai_recommendation.get('catalysts', [])
    
    def _extract_insights_from_text(self, text_details: str) -> tuple[List[str], List[str]]:
        """Extract insights and risks from additional text details beyond JSON"""
        import re
        
        insights = []
        risks = []
        
        if not text_details:
            return insights, risks
        
        # Clean and normalize text
        text = text_details.strip()
        
        # Extract insights using various patterns
        insight_patterns = [
            r'(?:key\s+)?insights?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)',
            r'(?:important\s+)?findings?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)',
            r'(?:notable\s+)?observations?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)',
            r'(?:investment\s+)?highlights?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)',
        ]
        
        for pattern in insight_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Extract bullet points or numbered items
                bullet_items = re.findall(r'[•\-\*]\s*(.+)', match)
                numbered_items = re.findall(r'\d+\.\s*(.+)', match)
                
                # Add bullet items
                for item in bullet_items:
                    clean_item = item.strip()
                    if len(clean_item) > 10 and clean_item not in insights:
                        insights.append(clean_item[:200])  # Limit length
                
                # Add numbered items
                for item in numbered_items:
                    clean_item = item.strip()
                    if len(clean_item) > 10 and clean_item not in insights:
                        insights.append(clean_item[:200])  # Limit length
        
        # Extract risks using various patterns
        risk_patterns = [
            r'(?:key\s+)?risks?[:\s]+(.*?)(?=\n\n|\nkey\s+insights?|\n[A-Z]|$)',
            r'(?:risk\s+)?factors?[:\s]+(.*?)(?=\n\n|\nkey\s+insights?|\n[A-Z]|$)',
            r'(?:potential\s+)?concerns?[:\s]+(.*?)(?=\n\n|\nkey\s+insights?|\n[A-Z]|$)',
            r'(?:investment\s+)?risks?[:\s]+(.*?)(?=\n\n|\nkey\s+insights?|\n[A-Z]|$)',
            r'downside[:\s]+(.*?)(?=\n\n|\nkey\s+insights?|\n[A-Z]|$)',
        ]
        
        for pattern in risk_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Extract bullet points or numbered items
                bullet_items = re.findall(r'[•\-\*]\s*(.+)', match)
                numbered_items = re.findall(r'\d+\.\s*(.+)', match)
                
                # Add bullet items
                for item in bullet_items:
                    clean_item = item.strip()
                    if len(clean_item) > 10 and clean_item not in risks:
                        risks.append(clean_item[:200])  # Limit length
                
                # Add numbered items
                for item in numbered_items:
                    clean_item = item.strip()
                    if len(clean_item) > 10 and clean_item not in risks:
                        risks.append(clean_item[:200])  # Limit length
        
        # If no structured patterns found, try to extract from general text
        if not insights and not risks:
            # Split text into sentences and look for insight/risk indicators
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                # Check for insight indicators
                insight_indicators = ['strength', 'opportunity', 'advantage', 'positive', 'growth', 'improve']
                if any(indicator in sentence.lower() for indicator in insight_indicators):
                    if len(insights) < 3:  # Limit to avoid noise
                        insights.append(sentence[:200])
                
                # Check for risk indicators
                risk_indicators = ['risk', 'concern', 'challenge', 'threat', 'weakness', 'decline', 'pressure']
                if any(indicator in sentence.lower() for indicator in risk_indicators):
                    if len(risks) < 3:  # Limit to avoid noise
                        risks.append(sentence[:200])
        
        # Limit total items
        insights = insights[:5]  # Max 5 insights
        risks = risks[:5]  # Max 5 risks
        
        return insights, risks
    
    def _extract_comprehensive_risks(self, llm_responses: Dict, ai_recommendation: Dict, additional_risks: List[str] = None) -> List[str]:
        """Extract and prioritize comprehensive risk factors"""
        import re
        risks = []
        
        # From AI synthesis
        if isinstance(ai_recommendation, dict):
            ai_risks = ai_recommendation.get('key_risks', [])
            if isinstance(ai_risks, list):
                risks.extend(ai_risks[:3])
        
        # Add additional risks from text details if available
        if additional_risks:
            risks.extend(additional_risks)
        
        # Extract from fundamental responses
        for resp in llm_responses.get('fundamental', {}).values():
            content = resp.get('content', '')
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            risk_section = re.search(r'risk[s]?[:\s]*(.*?)(?=\n\n|\d+\.)', content, re.IGNORECASE | re.DOTALL)
            if risk_section:
                risk_items = re.findall(r'[•\-]\s*(.+)', risk_section.group(1))
                risks.extend(risk_items[:2])
        
        # Deduplicate and limit
        unique_risks = []
        seen = set()
        for risk in risks:
            risk_lower = risk.lower().strip()
            if risk_lower not in seen and len(risk_lower) > 10:
                seen.add(risk_lower)
                unique_risks.append(risk)
        
        return unique_risks[:8] if unique_risks else ["Limited risk data available"]
    
    def _extract_comprehensive_insights(self, llm_responses: Dict, ai_recommendation: Dict, additional_insights: List[str] = None) -> List[str]:
        """Extract and prioritize comprehensive insights"""
        import re
        insights = []
        
        # Add additional insights from text details if available
        if additional_insights:
            insights.extend(additional_insights)
        
        # From AI synthesis catalysts
        if isinstance(ai_recommendation, dict):
            catalysts = ai_recommendation.get('key_catalysts', [])
            if isinstance(catalysts, list):
                insights.extend([f"Catalyst: {cat}" for cat in catalysts[:2]])
        
        # Extract key findings from responses
        for resp in llm_responses.get('fundamental', {}).values():
            content = resp.get('content', '')
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            insights_section = re.search(r'key\s+(?:insight|finding)[s]?[:\s]*(.*?)(?=\n\n|\d+\.)', content, re.IGNORECASE | re.DOTALL)
            if insights_section:
                insight_items = re.findall(r'[•\-]\s*(.+)', insights_section.group(1))
                insights.extend(insight_items[:2])
        
        # Technical insights
        tech_resp = llm_responses.get('technical')
        if tech_resp:
            content = tech_resp.get('content', '')
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            tech_insights = re.findall(r'KEY INSIGHTS[:\s]*\*?\*?(.*?)(?=\*\*[A-Z]|\n\n)', content, re.IGNORECASE | re.DOTALL)
            if tech_insights:
                tech_items = re.findall(r'[•\-]\s*(.+)', tech_insights[0])
                insights.extend([f"Technical: {item}" for item in tech_items[:2]])
        
        # Deduplicate and limit
        unique_insights = []
        seen = set()
        for insight in insights:
            insight_lower = insight.lower().strip()
            if insight_lower not in seen and len(insight_lower) > 10:
                seen.add(insight_lower)
                unique_insights.append(insight)
        
        return unique_insights[:8] if unique_insights else ["Analysis insights pending"]
    
    def _save_synthesis_llm_response(self, symbol: str, prompt: str, response: str, 
                                   processing_time_ms: int, synthesis_mode: str = 'comprehensive'):
        """Save synthesis LLM response to database and disk"""
        try:
            # Prepare data for DAO
            response_obj = {
                'type': 'text',
                'content': response
            }
            
            metadata = {
                'processing_time_ms': processing_time_ms,
                'response_length': len(response),
                'timestamp': datetime.now().isoformat(),
                'synthesis_type': 'full',
                'model': self.config.ollama.models.get('synthesis', 'llama3.1:8b-instruct-q8_0')
            }
            
            # Store full prompt directly 
            prompt_data = prompt
            
            model_info = {
                'model': metadata['model'],
                'temperature': 0.3,
                'top_p': 0.9,
                'num_ctx': 32768,
                'num_predict': 4096
            }
            
            # Determine fiscal period from available data
            fiscal_year, fiscal_period = self._get_latest_fiscal_period()
            
            # Save to cache using cache manager with synthesis mode-specific llm_type
            # Use intelligent defaults: SYNTHESIS as form_type for synthesis analysis
            llm_type = f'synthesis_{synthesis_mode}'  # synthesis_comprehensive or synthesis_quarterly
            cache_key = {
                'symbol': symbol,
                'form_type': 'SYNTHESIS',  # Intelligent default for synthesis analysis
                'period': f"{fiscal_year}-{fiscal_period}",
                'fiscal_year': fiscal_year,  # Separate key for file pattern
                'fiscal_period': fiscal_period,  # Separate key for file pattern
                'llm_type': llm_type
            }
            cache_value = {
                'prompt': prompt_data,
                'model_info': model_info,
                'response': response_obj,
                'metadata': metadata
            }
            
            success = self.cache_manager.set(CacheType.LLM_RESPONSE, cache_key, cache_value)
            
            if success:
                self.main_logger.info(f"💾 Stored synthesis LLM response for {symbol}")
            else:
                self.main_logger.error(f"Failed to store synthesis LLM response for {symbol}")
            
            # Also save the prompt and response as separate text files for visibility
            symbol_cache_dir = self.llm_cache_dir / symbol
            symbol_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Use mode-specific filenames to avoid overlap
            mode_suffix = '_comprehensive' if synthesis_mode == 'comprehensive' else '_quarterly'
            
            # Save prompt
            prompt_file = symbol_cache_dir / f"prompt_synthesis{mode_suffix}.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            # Save response
            response_file = symbol_cache_dir / f"response_synthesis{mode_suffix}.txt"
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(response)
                
            self.main_logger.info(f"💾 Saved synthesis prompt and response to {symbol_cache_dir} (mode: {synthesis_mode})")
            
        except Exception as e:
            self.main_logger.error(f"Error saving synthesis LLM response: {e}")
    
    def _save_synthesis_results(self, symbol: str, recommendation: InvestmentRecommendation):
        """Save synthesis results to database"""
        try:
            synthesis_data = {
                'symbol': symbol,
                'overall_score': recommendation.overall_score,
                'fundamental_score': recommendation.fundamental_score,
                'technical_score': recommendation.technical_score,
                'income_score': recommendation.income_score,
                'cashflow_score': recommendation.cashflow_score,
                'balance_score': recommendation.balance_score,
                'recommendation': recommendation.recommendation,
                'confidence': recommendation.confidence,
                'price_target': recommendation.price_target,
                'current_price': recommendation.current_price,
                'investment_thesis': recommendation.investment_thesis,
                'time_horizon': recommendation.time_horizon,
                'position_size': recommendation.position_size,
                'key_catalysts': recommendation.key_catalysts,
                'key_risks': recommendation.key_risks,
                'key_insights': recommendation.key_insights,
                'entry_strategy': recommendation.entry_strategy,
                'exit_strategy': recommendation.exit_strategy,
                'stop_loss': recommendation.stop_loss,
                'data_quality_score': recommendation.data_quality_score,
                'analysis_timestamp': recommendation.analysis_timestamp.isoformat() if hasattr(recommendation.analysis_timestamp, 'isoformat') else str(recommendation.analysis_timestamp)
            }
            
            # Store in synthesis_results table if it exists
            # For now, log the results
            self.main_logger.info(f"Synthesis results for {symbol}: {recommendation.recommendation} (Score: {recommendation.overall_score})")
            
        except Exception as e:
            self.main_logger.error(f"Failed to save synthesis results for {symbol}: {e}")
    
    
    def _get_market_summary(self) -> Dict:
        """Get market summary data for weekly reports"""
        # Placeholder - would fetch real market data
        return {
            'sp500': '4,500.00',
            'sp500_week_change': '+1.2%',
            'sp500_ytd_change': '+15.3%',
            'nasdaq': '14,000.00',
            'nasdaq_week_change': '+2.1%',
            'nasdaq_ytd_change': '+22.5%',
            'dow': '35,000.00',
            'dow_week_change': '+0.8%',
            'dow_ytd_change': '+8.2%',
            'commentary': 'Markets showed resilience this week despite mixed economic data.'
        }
    
    def _calculate_portfolio_performance(self, recommendations: List[Dict]) -> Dict:
        """Calculate portfolio performance metrics"""
        if not recommendations:
            return {}
        
        # Calculate aggregate metrics
        avg_score = sum(r['overall_score'] for r in recommendations) / len(recommendations)
        buy_count = sum(1 for r in recommendations if 'BUY' in r['recommendation'])
        win_rate = buy_count / len(recommendations) * 100 if recommendations else 0
        
        # Find best/worst performers (placeholder logic)
        sorted_by_score = sorted(recommendations, key=lambda x: x['overall_score'], reverse=True)
        
        return {
            'week_return': '+2.3%',  # Placeholder
            'month_return': '+5.1%',  # Placeholder
            'ytd_return': '+18.7%',  # Placeholder
            'win_rate': f"{win_rate:.1f}%",
            'best_performer': sorted_by_score[0]['symbol'] if sorted_by_score else 'N/A',
            'worst_performer': sorted_by_score[-1]['symbol'] if sorted_by_score else 'N/A'
        }

    def _create_synthesis_prompt(self, symbol: str, llm_responses: Dict, latest_data: Dict) -> str:
        """Create comprehensive synthesis prompt using Jinja2 template with all quarterly analyses, comprehensive analysis, and technical data"""
        
        # 1. ORGANIZE FUNDAMENTAL DATA BY TYPE
        comprehensive_analysis = ""
        quarterly_analyses = []
        financial_metrics_by_quarter = []
        
        # Extract comprehensive analysis
        if 'comprehensive' in llm_responses.get('fundamental', {}):
            comp_data = llm_responses['fundamental']['comprehensive']
            if comp_data and comp_data.get('content'):
                content = comp_data['content']
                if isinstance(content, dict):
                    comprehensive_analysis = json.dumps(content, indent=2)
                else:
                    comprehensive_analysis = str(content)[:10000]
        
        # Extract and sort quarterly analyses chronologically
        for key, resp in llm_responses.get('fundamental', {}).items():
            if key != 'comprehensive' and resp and resp.get('content'):
                period = resp.get('period', 'Unknown')
                form_type = resp.get('form_type', 'Unknown')
                content = resp.get('content', {})
                
                if isinstance(content, dict):
                    content_str = json.dumps(content, indent=2)[:3000]  # Limit for readability
                else:
                    content_str = str(content)[:3000]
                
                quarterly_analyses.append({
                    'period': period,
                    'form_type': form_type,
                    'content': content_str,
                    'raw_data': content if isinstance(content, dict) else {}
                })
                
                # Extract key financial metrics for trend analysis
                if isinstance(content, dict):
                    metrics = self._extract_financial_metrics_from_quarter(content, period)
                    if metrics:
                        financial_metrics_by_quarter.append(metrics)
        
        # Sort quarterly analyses chronologically (newest first)
        quarterly_analyses.sort(key=lambda x: x['period'], reverse=True)
        
        # 2. CREATE FINANCIAL TRENDS AND RATIOS
        financial_trends = self._create_financial_trends_analysis(financial_metrics_by_quarter)
        
        # 3. EXTRACT TECHNICAL ANALYSIS
        technical_analysis = ""
        technical_signals = {}
        
        if llm_responses.get('technical') and llm_responses['technical'].get('content'):
            technical_content = llm_responses['technical']['content']
            if isinstance(technical_content, dict):
                technical_analysis = json.dumps(technical_content, indent=2)
                technical_signals = technical_content
            else:
                technical_analysis = str(technical_content)[:6000]
                # Try to extract signals from text
                technical_signals = self._extract_technical_signals_from_text(technical_analysis)
        
        # 4. GET CURRENT MARKET DATA
        current_price = latest_data.get('technical', {}).get('current_price', 0)
        market_data = latest_data.get('technical', {})
        
        # 5. USE JINJA2 TEMPLATE TO CREATE SYNTHESIS PROMPT
        from utils.prompt_manager import get_prompt_manager
        prompt_manager = get_prompt_manager()
        
        template_data = {
            'symbol': symbol,
            'current_price': current_price,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'comprehensive_analysis': comprehensive_analysis,
            'quarterly_analyses': quarterly_analyses[:8],  # Limit to 8 most recent
            'quarterly_count': len(quarterly_analyses),
            'financial_trends': financial_trends,
            'technical_analysis': technical_analysis,
            'technical_signals': technical_signals,
            'market_data': market_data
        }
        
        try:
            # Use appropriate template based on synthesis mode
            template_name = 'investment_synthesis_comprehensive_mode.j2'
            prompt = prompt_manager.render_template(template_name, **template_data)
            self.main_logger.info(f"✅ Generated comprehensive synthesis prompt using Jinja2 template: {len(prompt)} chars, "
                                f"{len(quarterly_analyses)} quarters, "
                                f"{'✅' if comprehensive_analysis else '❌'} comprehensive, "
                                f"{'✅' if technical_analysis else '❌'} technical")
            return prompt
            
        except Exception as e:
            self.main_logger.error(f"❌ Failed to render Jinja2 comprehensive synthesis template: {e}")
            # Fallback to simple template
            return f"""Investment synthesis for {symbol} at ${current_price}:
Comprehensive: {'✅' if comprehensive_analysis else '❌'}
Quarterly: {len(quarterly_analyses)} quarters  
Technical: {'✅' if technical_analysis else '❌'}
Respond with detailed JSON investment analysis."""

    def _create_quarterly_synthesis_prompt(self, symbol: str, llm_responses: Dict, latest_data: Dict, prompt_manager) -> str:
        """Create quarterly synthesis prompt using last N quarters + technical analysis (no comprehensive analysis)"""
        
        # Get symbol-specific logger
        symbol_logger = self.config.get_symbol_logger(symbol, 'synthesizer')
        
        # 1. EXTRACT ALL QUARTERLY ANALYSES (NO COMPREHENSIVE)
        quarterly_analyses = []
        
        # Extract and sort quarterly analyses chronologically
        for key, resp in llm_responses.get('fundamental', {}).items():
            if key != 'comprehensive' and resp and resp.get('content'):
                period = resp.get('period', 'Unknown')
                form_type = resp.get('form_type', 'Unknown')
                content = resp.get('content', {})
                
                if isinstance(content, dict):
                    content_str = json.dumps(content, indent=2)[:4000]  # More space since no comprehensive
                else:
                    content_str = str(content)[:4000]
                
                quarterly_analyses.append({
                    'period': period,
                    'form_type': form_type,
                    'content': content_str,
                    'raw_data': content if isinstance(content, dict) else {}
                })
        
        # Sort quarterly analyses by period (most recent first)
        quarterly_analyses.sort(key=lambda x: x['period'], reverse=True)
        quarterly_count = len(quarterly_analyses)
        
        symbol_logger.info(f"Quarterly synthesis: Using {quarterly_count} quarterly analyses for {symbol}")
        
        # 2. EXTRACT TECHNICAL ANALYSIS
        technical_analysis = ""
        technical_signals = {}
        
        if llm_responses.get('technical'):
            tech_content = llm_responses['technical'].get('content', '')
            if isinstance(tech_content, dict):
                technical_analysis = json.dumps(tech_content, indent=2)[:3000]
                technical_signals = tech_content
            else:
                technical_analysis = str(tech_content)[:3000]
        
        # 3. PREPARE FINANCIAL TRENDS FROM QUARTERLY DATA
        financial_trends = self._extract_quarterly_trends(quarterly_analyses)
        
        # 4. USE QUARTERLY-SPECIFIC SYNTHESIS TEMPLATE
        quarterly_synthesis_prompt = prompt_manager.render_template('investment_synthesis_quarterly_mode.j2',
            symbol=symbol,
            analysis_date=datetime.now().strftime('%Y-%m-%d'),
            current_price=latest_data.get('current_price', 0.0),
            comprehensive_analysis="",  # No comprehensive analysis in quarterly mode
            quarterly_analyses=quarterly_analyses,
            quarterly_count=quarterly_count,
            financial_trends=financial_trends,
            technical_analysis=technical_analysis,
            technical_signals=technical_signals,
            market_data=latest_data
        )
        
        symbol_logger.info(f"Generated quarterly synthesis prompt: {len(quarterly_synthesis_prompt)} characters")
        return quarterly_synthesis_prompt

    def _extract_quarterly_trends(self, quarterly_analyses: List[Dict]) -> str:
        """Extract and summarize trends across quarters for quarterly synthesis"""
        if not quarterly_analyses:
            return "No quarterly data available for trend analysis"
        
        trends = []
        trends.append(f"Quarterly Analysis Summary ({len(quarterly_analyses)} quarters):")
        
        # Add trends based on available quarterly data
        for i, qa in enumerate(quarterly_analyses[:8]):  # Use last 8 quarters max
            period = qa.get('period', f'Q{i+1}')
            form_type = qa.get('form_type', 'Unknown')
            trends.append(f"- {period} ({form_type}): Key financial metrics and performance indicators")
        
        if len(quarterly_analyses) > 8:
            trends.append(f"... and {len(quarterly_analyses) - 8} additional quarters")
        
        return "\n".join(trends)
    
    def _extract_financial_metrics_from_quarter(self, quarter_data: Dict, period: str) -> Optional[Dict]:
        """Extract key financial metrics from a quarterly analysis"""
        try:
            metrics = {'period': period}
            
            # Try to extract common financial metrics from the quarter data
            if isinstance(quarter_data, dict):
                # Look for revenue metrics
                for key in ['revenue', 'total_revenue', 'revenues', 'sales']:
                    if key in quarter_data:
                        metrics['revenue'] = quarter_data[key]
                        break
                
                # Look for profit metrics  
                for key in ['net_income', 'net_profit', 'earnings', 'profit']:
                    if key in quarter_data:
                        metrics['net_income'] = quarter_data[key]
                        break
                
                # Look for margin metrics
                for key in ['gross_margin', 'operating_margin', 'profit_margin']:
                    if key in quarter_data:
                        metrics[key] = quarter_data[key]
                
                # Look for other key metrics
                for key in ['eps', 'operating_cash_flow', 'free_cash_flow', 'total_assets', 'total_debt']:
                    if key in quarter_data:
                        metrics[key] = quarter_data[key]
            
            return metrics if len(metrics) > 1 else None
            
        except Exception as e:
            self.main_logger.warning(f"Error extracting metrics from quarter {period}: {e}")
            return None
    
    def _create_financial_trends_analysis(self, metrics_by_quarter: List[Dict]) -> str:
        """Create financial trends analysis from quarterly metrics"""
        try:
            if not metrics_by_quarter:
                return "[NO QUARTERLY METRICS AVAILABLE FOR TREND ANALYSIS]"
            
            trends = []
            trends.append(f"📊 FINANCIAL TRENDS ANALYSIS ({len(metrics_by_quarter)} quarters):")
            
            # Sort by period for chronological analysis
            sorted_metrics = sorted(metrics_by_quarter, key=lambda x: x.get('period', ''))
            
            # Revenue trend
            revenues = [m.get('revenue', 0) for m in sorted_metrics if m.get('revenue')]
            if len(revenues) >= 2:
                revenue_growth = ((revenues[-1] - revenues[0]) / revenues[0] * 100) if revenues[0] > 0 else 0
                trends.append(f"📈 Revenue Trend: {revenue_growth:+.1f}% over {len(revenues)} quarters")
            
            # Margin trends
            margins = [m.get('profit_margin', 0) for m in sorted_metrics if m.get('profit_margin')]
            if len(margins) >= 2:
                margin_change = margins[-1] - margins[0]
                trends.append(f"💰 Margin Trend: {margin_change:+.1f}pp change in profit margin")
            
            # Add quarterly breakdown
            trends.append("\n📋 Quarterly Progression:")
            for i, metrics in enumerate(sorted_metrics[-4:]):  # Last 4 quarters
                period = metrics.get('period', f'Q{i+1}')
                revenue = metrics.get('revenue', 0)
                margin = metrics.get('profit_margin', 0)
                trends.append(f"  {period}: Revenue ${revenue:,.0f}M, Margin {margin:.1f}%")
            
            return '\n'.join(trends)
            
        except Exception as e:
            self.main_logger.warning(f"Error creating trends analysis: {e}")
            return "[ERROR CREATING TRENDS ANALYSIS]"
    
    def _extract_technical_signals_from_text(self, technical_text: str) -> Dict:
        """Extract technical signals from text analysis"""
        try:
            import re
            signals = {}
            
            # Extract RSI
            rsi_match = re.search(r'RSI[^:]*:\s*([\d.]+)', technical_text, re.IGNORECASE)
            if rsi_match:
                signals['rsi'] = float(rsi_match.group(1))
            
            # Extract MACD
            macd_match = re.search(r'MACD[^:]*:\s*([-\d.]+)', technical_text, re.IGNORECASE)
            if macd_match:
                signals['macd'] = float(macd_match.group(1))
            
            # Extract trend
            trend_match = re.search(r'trend[^:]*:\s*([A-Za-z]+)', technical_text, re.IGNORECASE)
            if trend_match:
                signals['trend'] = trend_match.group(1).upper()
            
            # Extract support/resistance
            support_match = re.search(r'support[^:]*:\s*\$?([\d.]+)', technical_text, re.IGNORECASE)
            if support_match:
                signals['support'] = float(support_match.group(1))
                
            resistance_match = re.search(r'resistance[^:]*:\s*\$?([\d.]+)', technical_text, re.IGNORECASE)
            if resistance_match:
                signals['resistance'] = float(resistance_match.group(1))
            
            return signals
            
        except Exception as e:
            self.main_logger.warning(f"Error extracting technical signals: {e}")
            return {}
        
        # Create sector/industry context for better analysis
        sector_context = self._get_sector_context(symbol)
        market_environment = self._get_market_environment_context()
        
        prompt = f"""You are a senior portfolio manager and CFA charterholder with 25+ years of experience synthesizing complex financial analyses for institutional investment decisions. You have deep expertise in:

• Multi-timeframe fundamental analysis across economic cycles
• Advanced technical analysis and market microstructure
• Risk-adjusted portfolio construction and position sizing
• Quantitative valuation models and relative value analysis
• Macro-economic impact assessment on sector rotation
• ESG integration and sustainable investing frameworks

Your task is to synthesize comprehensive analyses for {symbol} into an actionable investment recommendation suitable for a $2B+ institutional portfolio.

=== COMPANY PROFILE & CONTEXT ===
Symbol: {symbol}
Sector Context: {sector_context}
Current Market Environment: {market_environment}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

=== COMPREHENSIVE FUNDAMENTAL ANALYSIS ===
{chr(10).join(fundamental_summaries) if fundamental_summaries else "[INSUFFICIENT FUNDAMENTAL DATA - HIGH RISK FLAG]"}

=== QUARTERLY PERFORMANCE EVOLUTION ===
{''.join([f"{chr(10)}{q['period']} ({q['form']}): {q['content'][:800]}{chr(10)}" for q in quarterly_trends]) if quarterly_trends else "[LIMITED QUARTERLY DATA AVAILABLE]"}

=== COMPREHENSIVE TECHNICAL ANALYSIS ===
{technical_analysis if technical_analysis else "[NO TECHNICAL ANALYSIS AVAILABLE - TIMING UNCERTAINTY]"}

=== EXTRACTED TECHNICAL SIGNALS ===
Key Technical Signals: {', '.join(technical_signals[:10]) if technical_signals else 'None extracted'}

=== CURRENT FINANCIAL SNAPSHOT ===
Revenue (TTM): ${fund_data.get('revenue', 0):,.0f}
Net Income (TTM): ${fund_data.get('net_income', 0):,.0f}
Earnings Per Share: ${fund_data.get('eps', 0):.2f}
Total Assets: ${fund_data.get('total_assets', 0):,.0f}
Total Liabilities: ${fund_data.get('total_liabilities', 0):,.0f}
Shareholders' Equity: ${fund_data.get('shareholders_equity', 0):,.0f}
Operating Cash Flow: ${fund_data.get('operating_cash_flow', 0):,.0f}
Free Cash Flow: ${fund_data.get('free_cash_flow', 0):,.0f}

=== VALUATION & QUALITY METRICS ===
P/E Ratio: {fund_data.get('pe_ratio', 'N/A')} | Industry Avg: {fund_data.get('industry_pe', 'N/A')}
P/B Ratio: {fund_data.get('pb_ratio', 'N/A')} | PEG Ratio: {fund_data.get('peg_ratio', 'N/A')}
EV/EBITDA: {fund_data.get('ev_ebitda', 'N/A')} | P/S Ratio: {fund_data.get('ps_ratio', 'N/A')}
Debt/Equity: {fund_data.get('debt_to_equity', 'N/A')} | Interest Coverage: {fund_data.get('interest_coverage', 'N/A')}
Current Ratio: {fund_data.get('current_ratio', 'N/A')} | Quick Ratio: {fund_data.get('quick_ratio', 'N/A')}
ROE: {fund_data.get('roe', 'N/A')}% | ROA: {fund_data.get('roa', 'N/A')}% | ROIC: {fund_data.get('roic', 'N/A')}%
Gross Margin: {fund_data.get('gross_margin', 'N/A')}% | Operating Margin: {fund_data.get('operating_margin', 'N/A')}%
Net Margin: {fund_data.get('profit_margin', 'N/A')}% | FCF Margin: {fund_data.get('fcf_margin', 'N/A')}%

=== REAL-TIME MARKET DATA ===
Current Price: ${tech_data.get('current_price', 0):.2f}
Intraday Range: ${tech_data.get('day_low', 0):.2f} - ${tech_data.get('day_high', 0):.2f}
Price Performance:
• 1 Day: {tech_data.get('price_change_1d', 0):+.2f}% | Volume vs Avg: {tech_data.get('volume_ratio', 1):.1f}x
• 1 Week: {tech_data.get('price_change_1w', 0):+.2f}% | Relative Strength: {tech_data.get('relative_strength', 'N/A')}
• 1 Month: {tech_data.get('price_change_1m', 0):+.2f}% | Beta: {tech_data.get('beta', 'N/A')}
• 3 Months: {tech_data.get('price_change_3m', 0):+.2f}% | 52W High: ${tech_data.get('week_52_high', 0):.2f}
• YTD: {tech_data.get('price_change_ytd', 0):+.2f}% | 52W Low: ${tech_data.get('week_52_low', 0):.2f}

=== TECHNICAL STRUCTURE ANALYSIS ===
Moving Averages Alignment:
• SMA 20: ${tech_data.get('sma_20', 0):.2f} | Position vs Price: {self._calculate_ma_position(tech_data.get('current_price', 0), tech_data.get('sma_20', 0))}
• SMA 50: ${tech_data.get('sma_50', 0):.2f} | Golden/Death Cross: {self._check_ma_cross(tech_data.get('sma_50', 0), tech_data.get('sma_200', 0))}
• SMA 200: ${tech_data.get('sma_200', 0):.2f} | Trend Strength: {self._assess_trend_strength(tech_data)}

Momentum & Oscillators:
• RSI (14): {tech_data.get('rsi', 0):.1f} | Stochastic: {tech_data.get('stochastic', 'N/A')}
• MACD: {tech_data.get('macd', 0):.4f} | Williams %R: {tech_data.get('williams_r', 'N/A')}
• MACD Signal: {tech_data.get('macd_signal', 0):.4f} | Money Flow Index: {tech_data.get('mfi', 'N/A')}
• MACD Histogram: {tech_data.get('macd_histogram', 0):.4f} | Rate of Change: {tech_data.get('roc', 'N/A')}%

Volatility & Risk Metrics:
• Bollinger Upper: ${tech_data.get('bollinger_upper', 0):.2f} | ATR (14): ${tech_data.get('atr', 0):.2f}
• Bollinger Lower: ${tech_data.get('bollinger_lower', 0):.2f} | Volatility (20D): {tech_data.get('volatility_20d', 'N/A')}%
• BB Position: {self._calculate_bb_position(tech_data)} | VIX Correlation: {tech_data.get('vix_correlation', 'N/A')}

Volume & Liquidity Analysis:
• Current Volume: {tech_data.get('volume', 0):,} | Volume Trend: {self._assess_volume_trend(tech_data)}
• 20-Day Avg Volume: {tech_data.get('avg_volume_20', 0):,} | On-Balance Volume: {tech_data.get('obv', 'N/A')}
• Volume-Price Relationship: {self._assess_volume_price_relationship(tech_data)}

=== INSTITUTIONAL SYNTHESIS REQUIREMENTS ===

Provide a comprehensive institutional-grade investment synthesis addressing:

**I. EXECUTIVE SUMMARY & RECOMMENDATION**
1. **FINAL RECOMMENDATION**: [STRONG BUY/BUY/HOLD/SELL/STRONG SELL]
   • Weight fundamental quality (60%) vs technical timing (40%)
   • Consider risk-adjusted returns and Sharpe ratio implications
   • Account for correlation with existing portfolio holdings

2. **INVESTMENT SCORE**: [1.0-10.0] (single decimal precision)
   • Fundamental Quality Score (1-10): ___
   • Technical Timing Score (1-10): ___
   • Risk-Adjusted Score (1-10): ___
   • **Overall Composite Score**: ___

3. **CONFIDENCE LEVEL**: [HIGH/MEDIUM/LOW] + Percentage (e.g., "HIGH - 85%")
   • Data quality assessment (completeness, recency, reliability)
   • Analysis convergence between fundamental and technical views
   • Market regime appropriateness

**II. STRATEGIC INVESTMENT THESIS** (3-4 detailed paragraphs)
• **Value Creation Narrative**: Synthesize core business strengths, competitive positioning, and growth catalysts from fundamental analysis
• **Market Timing & Entry Strategy**: Integrate technical signals, momentum factors, and optimal entry/exit levels
• **Risk-Return Profile**: Quantify expected returns, downside protection, and correlation benefits for portfolio construction
• **Investment Horizon Alignment**: Match thesis durability with recommended holding period

**III. MULTI-TIMEFRAME CATALYSTS** (5-7 detailed points)
• **Near-term (0-3 months)**: Technical breakouts, earnings events, product launches
• **Medium-term (3-12 months)**: Fundamental inflection points, market share gains, operational improvements
• **Long-term (1-3 years)**: Strategic positioning, industry transformation, ESG factors
• **Macro catalysts**: Economic cycle positioning, interest rate sensitivity, currency impacts

**IV. COMPREHENSIVE RISK ASSESSMENT** (6-8 detailed points)
• **Business/Fundamental Risks**: Competition, regulation, execution, cyclicality
• **Technical/Market Risks**: Support breaks, momentum reversals, correlation shifts
• **Macro/Systematic Risks**: Economic slowdown, geopolitical events, sector rotation
• **Liquidity/Operational Risks**: Position sizing constraints, execution challenges
• **ESG/Governance Risks**: Environmental liabilities, social license, board effectiveness

**V. PRECISION PRICE TARGETS & SCENARIOS**
• **12-month Base Case Target**: $XXX.XX (probability: XX%)
  - Methodology: DCF, comparable multiples, technical projections
• **Bull Case Target**: $XXX.XX (probability: XX%) - Key assumptions
• **Bear Case Target**: $XXX.XX (probability: XX%) - Risk scenario triggers
• **Technical Levels**: Support: $XXX.XX | Resistance: $XXX.XX | Stop-loss: $XXX.XX

**VI. INSTITUTIONAL PORTFOLIO IMPLEMENTATION**
• **Position Sizing**: [1-5% / STARTER | 5-10% / MODERATE | 10-15% / LARGE | 15%+ / CONCENTRATED]
  - Risk budgeting: Expected volatility, Value-at-Risk, correlation impact
  - Liquidity requirements: Average daily volume, market impact analysis

• **Time Horizon**: [SHORT (0-6M) | MEDIUM (6M-2Y) | LONG (2Y+)]
  - Investment committee review schedule
  - Rebalancing triggers and thresholds

• **Implementation Strategy**:
  - Entry: Optimal execution algorithm (TWAP, VWAP, Implementation Shortfall)
  - Hedging: Currency, sector, market beta considerations
  - Monitoring: Key performance indicators and risk metrics

**VII. ACTIONABLE EXECUTION PLAN**
• **Phase 1 - Entry Strategy**:
  - Primary entry zone: $XXX.XX - $XXX.XX
  - Secondary accumulation level: $XXX.XX
  - Maximum allocation timeframe: X weeks
  - Market condition dependencies

• **Phase 2 - Active Management**:
  - Profit-taking levels: 25% at $XXX, 50% at $XXX
  - Stop-loss discipline: Hard stop at $XXX (XX% below entry)
  - Rebalancing triggers: Fundamental deterioration, technical breakdown

• **Phase 3 - Exit Strategy**:
  - Target achievement: Systematic profit-taking plan
  - Thesis invalidation: Clear exit criteria and timeline
  - Tax optimization: Long-term capital gains considerations

**VIII. MONITORING & REVIEW FRAMEWORK**
• **Weekly Monitoring**: Technical levels, volume patterns, relative performance
• **Monthly Review**: Fundamental metrics, earnings revisions, competitive dynamics
• **Quarterly Assessment**: Strategic progress, thesis validation, position sizing optimization
• **Annual Strategy Review**: Long-term positioning, portfolio construction efficiency

Provide specific, quantitative analysis with clear reasoning for each recommendation. Focus on actionable insights that justify the investment decision for a fiduciary standard institutional portfolio management context."""
        
        return prompt
    
    def _get_sector_context(self, symbol: str) -> str:
        """Get sector and industry context for the symbol"""
        try:
            # Load sector mapping from external file
            sector_mapping_file = Path(self.config.data_dir) / "sector_mapping.json"
            
            if sector_mapping_file.exists():
                with open(sector_mapping_file, 'r') as f:
                    sector_data = json.load(f)
                
                mappings = sector_data.get('sector_mappings', {})
                default = sector_data.get('default_mapping', {})
                
                if symbol in mappings:
                    mapping = mappings[symbol]
                    return f"{mapping['sector']} - {mapping['industry']}"
                else:
                    return f"{default['sector']} - {default['industry']}"
            else:
                self.main_logger.warning(f"Sector mapping file not found: {sector_mapping_file}")
                return "Unknown Sector - Requires Research"
                
        except Exception as e:
            self.main_logger.error(f"Error loading sector mapping: {e}")
            return "Unknown Sector - Requires Research"
    
    def _get_market_environment_context(self) -> str:
        """Get current market environment context"""
        # This could be enhanced to fetch real market data
        return "Mixed signals with elevated volatility, Fed policy uncertainty, and sector rotation dynamics"
    
    def _calculate_ma_position(self, current_price: float, ma_price: float) -> str:
        """Calculate moving average position relative to current price"""
        if not current_price or not ma_price:
            return "N/A"
        
        if current_price > ma_price * 1.02:
            return "Strong Above"
        elif current_price > ma_price:
            return "Above"
        elif current_price < ma_price * 0.98:
            return "Strong Below"
        else:
            return "Below"
    
    def _check_ma_cross(self, sma_50: float, sma_200: float) -> str:
        """Check for golden/death cross pattern"""
        if not sma_50 or not sma_200:
            return "N/A"
        
        if sma_50 > sma_200 * 1.01:
            return "Golden Cross"
        elif sma_50 < sma_200 * 0.99:
            return "Death Cross"
        else:
            return "Neutral"
    
    def _assess_trend_strength(self, tech_data: Dict) -> str:
        """Assess overall trend strength from technical data"""
        try:
            rsi = tech_data.get('rsi', 50)
            price_change_1m = tech_data.get('price_change_1m', 0)
            
            if rsi > 60 and price_change_1m > 5:
                return "Strong Bullish"
            elif rsi > 50 and price_change_1m > 0:
                return "Bullish"
            elif rsi < 40 and price_change_1m < -5:
                return "Strong Bearish"
            elif rsi < 50 and price_change_1m < 0:
                return "Bearish"
            else:
                return "Neutral"
        except:
            return "N/A"
    
    def _calculate_bb_position(self, tech_data: Dict) -> str:
        """Calculate Bollinger Band position"""
        try:
            current_price = tech_data.get('current_price', 0)
            bb_upper = tech_data.get('bollinger_upper', 0)
            bb_lower = tech_data.get('bollinger_lower', 0)
            
            if not all([current_price, bb_upper, bb_lower]):
                return "N/A"
            
            bb_range = bb_upper - bb_lower
            position = (current_price - bb_lower) / bb_range
            
            if position > 0.8:
                return "Upper Band"
            elif position > 0.6:
                return "Above Middle"
            elif position > 0.4:
                return "Middle Range"
            elif position > 0.2:
                return "Below Middle"
            else:
                return "Lower Band"
        except:
            return "N/A"
    
    def _assess_volume_trend(self, tech_data: Dict) -> str:
        """Assess volume trend"""
        try:
            volume_ratio = tech_data.get('volume_ratio', 1)
            
            if volume_ratio > 2.0:
                return "Very High"
            elif volume_ratio > 1.5:
                return "High"
            elif volume_ratio > 0.8:
                return "Normal"
            elif volume_ratio > 0.5:
                return "Low"
            else:
                return "Very Low"
        except:
            return "N/A"
    
    def _assess_volume_price_relationship(self, tech_data: Dict) -> str:
        """Assess volume-price relationship"""
        try:
            price_change_1d = tech_data.get('price_change_1d', 0)
            volume_ratio = tech_data.get('volume_ratio', 1)
            
            if price_change_1d > 0 and volume_ratio > 1.2:
                return "Bullish Confirmation"
            elif price_change_1d < 0 and volume_ratio > 1.2:
                return "Bearish Confirmation"
            elif abs(price_change_1d) > 2 and volume_ratio < 0.8:
                return "Divergence Warning"
            else:
                return "Neutral"
        except:
            return "N/A"
    
    def _create_fallback_recommendation(self, raw_response: Any, symbol: str, overall_score: float) -> Dict[str, Any]:
        """
        Create a fallback recommendation when JSON parsing fails
        
        Args:
            raw_response: The raw LLM response that failed to parse
            symbol: Stock symbol
            overall_score: Computed overall score
            
        Returns:
            Dict containing fallback recommendation structure
        """
        try:
            # Convert response to string for text parsing
            response_text = str(raw_response) if raw_response else ""
            
            # Try to extract any partial information using regex
            import re
            
            # Extract recommendation if present
            recommendation = "HOLD"  # Safe default
            rec_patterns = [
                r'recommendation["\']?\s*:\s*["\']?(STRONG_BUY|STRONG_SELL|BUY|SELL|HOLD)["\']?',
                r'FINAL\s+RECOMMENDATION[:\s]*\*?\*?\s*\[?([A-Z\s]+)\]?',
                r'"recommendation":\s*"([^"]+)"'
            ]
            
            for pattern in rec_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    rec_text = match.group(1).strip().upper()
                    if any(valid in rec_text for valid in ['BUY', 'SELL', 'HOLD']):
                        recommendation = rec_text
                        break
            
            # Extract confidence if present
            confidence = "LOW"  # Conservative default for failed parsing
            conf_patterns = [
                r'confidence["\']?\s*:\s*["\']?(HIGH|MEDIUM|LOW)["\']?',
                r'"confidence_level":\s*"([^"]+)"'
            ]
            
            for pattern in conf_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    confidence = match.group(1).strip().upper()
                    break
            
            # Extract investment thesis if present
            thesis = f"Analysis completed for {symbol} with computed overall score of {overall_score:.1f}/10."
            thesis_patterns = [
                r'investment_thesis["\']?\s*:\s*["\']([^"\']+)["\']',
                r'thesis["\']?\s*:\s*["\']([^"\']+)["\']',
                r'INVESTMENT\s+THESIS[:\s]*([^{}\[\]]+?)(?=\*\*|##|\n\n|$)'
            ]
            
            for pattern in thesis_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    extracted_thesis = match.group(1).strip()
                    if len(extracted_thesis) > 20:  # Only use if substantial
                        thesis = extracted_thesis[:500]  # Limit length
                        break
            
            # Create fallback structure that matches expected format
            fallback_recommendation = {
                'overall_score': overall_score,
                'fundamental_score': overall_score,  # Use computed score as fallback
                'technical_score': overall_score,    # Use computed score as fallback
                'investment_recommendation': {
                    'recommendation': recommendation,
                    'confidence_level': confidence
                },
                'executive_summary': {
                    'investment_thesis': thesis
                },
                'key_catalysts': [
                    f"Technical and fundamental analysis for {symbol}",
                    "Market position assessment",
                    "Financial performance review"
                ],
                'key_risks': [
                    "JSON parsing failure indicates potential data quality issues",
                    "LLM response formatting problems",
                    "Analysis may be incomplete due to parsing errors"
                ],
                'position_size': 'SMALL',  # Conservative due to parsing failure
                'time_horizon': 'MEDIUM-TERM',
                'entry_strategy': f"Conservative approach recommended due to analysis parsing issues",
                'exit_strategy': f"Monitor for improved data quality and re-analyze",
                'details': f"Fallback recommendation created due to JSON parsing failure. Raw response length: {len(response_text)} characters.",
                '_fallback_created': True,  # Flag to indicate this is a fallback
                '_parsing_error': True      # Flag to indicate parsing issues
            }
            
            self.main_logger.info(f"Created fallback recommendation for {symbol}: {recommendation} (confidence: {confidence})")
            
            return fallback_recommendation
            
        except Exception as e:
            self.main_logger.error(f"Error creating fallback recommendation: {e}")
            # Last resort fallback
            return {
                'overall_score': 5.0,  # Neutral score
                'fundamental_score': 5.0,
                'technical_score': 5.0,
                'investment_recommendation': {
                    'recommendation': 'HOLD',
                    'confidence_level': 'LOW'
                },
                'executive_summary': {
                    'investment_thesis': f"Unable to complete analysis for {symbol} due to processing errors."
                },
                'key_catalysts': ["Analysis pending"],
                'key_risks': ["Analysis incomplete", "Data processing errors"],
                'position_size': 'AVOID',
                'time_horizon': 'UNKNOWN',
                'entry_strategy': 'Wait for successful analysis',
                'exit_strategy': 'Not applicable',
                'details': 'Emergency fallback due to complete parsing failure',
                '_fallback_created': True,
                '_parsing_error': True,
                '_emergency_fallback': True
            }
    
    def _create_extensible_insights_structure(self, ai_recommendation: Dict, thinking_content: str, 
                                             additional_details: str, symbol: str) -> Dict[str, Any]:
        """
        Create an extensible structure for capturing additional insights that can evolve with 
        prompt and response changes. This structure is designed to be included in the final 
        PDF report so reporting and synthesizing modules can inspect this key to cover fields 
        graciously.
        
        Args:
            ai_recommendation: The main AI recommendation dict
            thinking_content: LLM thinking/reasoning content
            additional_details: Any additional text details from response
            symbol: Stock symbol being analyzed
            
        Returns:
            Dict containing extensible insights structure
        """
        try:
            extensible_insights = {
                # Core metadata about the insights capture
                '_insights_version': '1.0',
                '_symbol': symbol,
                '_timestamp': datetime.now().isoformat(),
                '_capture_method': 'llm_response_analysis',
                
                # Structured thinking and reasoning capture
                'reasoning_insights': {
                    'thinking_content': thinking_content if thinking_content else '',
                    'thinking_length': len(thinking_content) if thinking_content else 0,
                    'has_structured_reasoning': bool(thinking_content and len(thinking_content) > 100),
                    'reasoning_themes': self._extract_reasoning_themes(thinking_content) if thinking_content else [],
                    'decision_process': self._extract_decision_process(thinking_content) if thinking_content else {}
                },
                
                # Additional content and markdown capture
                'content_insights': {
                    'additional_details': additional_details if additional_details else '',
                    'details_length': len(additional_details) if additional_details else 0,
                    'has_markdown_content': self._detect_markdown_content(additional_details) if additional_details else False,
                    'extracted_bullet_points': self._extract_bullet_points(additional_details) if additional_details else [],
                    'extracted_numbers': self._extract_numerical_insights(additional_details) if additional_details else []
                },
                
                # Response structure analysis
                'response_structure': {
                    'field_completeness': self._analyze_field_completeness(ai_recommendation),
                    'response_type': type(ai_recommendation).__name__,
                    'custom_fields': self._identify_custom_fields(ai_recommendation),
                    'processing_metadata': ai_recommendation.get('processing_metadata', {}),
                    'contains_fallback_flags': self._check_fallback_flags(ai_recommendation)
                },
                
                # Future evolution placeholders (for prompt/response evolution)
                'evolution_capture': {
                    'sentiment_analysis': {},  # Future: sentiment from thinking
                    'confidence_indicators': {},  # Future: confidence markers in text
                    'methodology_insights': {},  # Future: LLM methodology preferences
                    'risk_assessment_depth': {},  # Future: detailed risk analysis
                    'peer_comparison_insights': {},  # Future: peer analysis reasoning
                    'market_timing_insights': {},  # Future: timing-specific insights
                    'scenario_analysis': {}  # Future: what-if scenario reasoning
                },
                
                # Report integration guidance
                'report_integration': {
                    'should_include_thinking': bool(thinking_content and len(thinking_content) > 200),
                    'should_include_details': bool(additional_details and len(additional_details) > 50),
                    'recommended_report_sections': self._recommend_report_sections(ai_recommendation, thinking_content, additional_details),
                    'priority_insights': self._extract_priority_insights(thinking_content, additional_details),
                    'visualization_suggestions': self._suggest_visualizations(ai_recommendation)
                }
            }
            
            return extensible_insights
            
        except Exception as e:
            self.main_logger.error(f"Error creating extensible insights structure: {e}")
            # Return minimal structure on error
            return {
                '_insights_version': '1.0',
                '_symbol': symbol,
                '_timestamp': datetime.now().isoformat(),
                '_error': str(e),
                'reasoning_insights': {},
                'content_insights': {},
                'response_structure': {},
                'evolution_capture': {},
                'report_integration': {'should_include_thinking': False, 'should_include_details': False}
            }
    
    def _extract_reasoning_themes(self, thinking_content: str) -> List[str]:
        """Extract key reasoning themes from thinking content"""
        if not thinking_content:
            return []
        
        themes = []
        # Look for common reasoning patterns
        patterns = [
            r'fundamental[s]?\s+(?:analysis|factors|strengths)',
            r'technical\s+(?:analysis|indicators|patterns)',
            r'risk[s]?\s+(?:assessment|factors|considerations)',
            r'market\s+(?:position|environment|conditions)',
            r'valuation\s+(?:methods|approaches|metrics)',
            r'growth\s+(?:prospects|potential|drivers)',
            r'competitive\s+(?:advantage|position|landscape)'
        ]
        
        for pattern in patterns:
            if re.search(pattern, thinking_content, re.IGNORECASE):
                # Extract the matched theme
                match = re.search(pattern, thinking_content, re.IGNORECASE)
                if match:
                    themes.append(match.group(0).lower())
        
        return list(set(themes))  # Remove duplicates
    
    def _extract_decision_process(self, thinking_content: str) -> Dict[str, Any]:
        """Extract decision-making process from thinking content"""
        if not thinking_content:
            return {}
        
        process = {
            'has_structured_approach': bool(re.search(r'first|then|next|finally', thinking_content, re.IGNORECASE)),
            'considers_alternatives': bool(re.search(r'but|however|alternatively|on the other hand', thinking_content, re.IGNORECASE)),
            'weighs_factors': bool(re.search(r'weight|balance|consider|factor', thinking_content, re.IGNORECASE)),
            'mentions_uncertainty': bool(re.search(r'uncertain|unclear|maybe|might|could', thinking_content, re.IGNORECASE)),
            'shows_confidence': bool(re.search(r'confident|certain|sure|clear', thinking_content, re.IGNORECASE))
        }
        
        return process
    
    def _detect_markdown_content(self, content: str) -> bool:
        """Detect if content contains markdown formatting"""
        if not content:
            return False
        
        markdown_patterns = [r'\*\*.*?\*\*', r'\*.*?\*', r'#+ ', r'- ', r'\d+\. ', r'```']
        return any(re.search(pattern, content) for pattern in markdown_patterns)
    
    def _extract_bullet_points(self, content: str) -> List[str]:
        """Extract bullet points from content"""
        if not content:
            return []
        
        # Look for various bullet point patterns
        patterns = [r'- (.+)', r'• (.+)', r'\* (.+)', r'\d+\. (.+)']
        bullets = []
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            bullets.extend(matches)
        
        return [bullet.strip() for bullet in bullets if len(bullet.strip()) > 5]
    
    def _extract_numerical_insights(self, content: str) -> List[Dict[str, Any]]:
        """Extract numerical insights from content"""
        if not content:
            return []
        
        numbers = []
        # Look for percentages, ratios, monetary amounts
        patterns = [
            (r'(\d+(?:\.\d+)?%)', 'percentage'),
            (r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', 'monetary'),
            (r'(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)', 'ratio'),
            (r'(\d+(?:\.\d+)?x)', 'multiple')
        ]
        
        for pattern, num_type in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    numbers.append({'type': num_type, 'value': match, 'context': 'content_extraction'})
                else:
                    numbers.append({'type': num_type, 'value': match, 'context': 'content_extraction'})
        
        return numbers[:10]  # Limit to first 10 to avoid clutter
    
    def _analyze_field_completeness(self, ai_recommendation: Dict) -> Dict[str, Any]:
        """Analyze completeness of standard fields in AI recommendation"""
        standard_fields = [
            'overall_score', 'investment_thesis', 'recommendation', 'confidence_level',
            'position_size', 'time_horizon', 'risk_reward_ratio', 'key_catalysts', 'downside_risks'
        ]
        
        completeness = {
            'total_standard_fields': len(standard_fields),
            'present_fields': [],
            'missing_fields': [],
            'completeness_ratio': 0.0
        }
        
        for field in standard_fields:
            if field in ai_recommendation and ai_recommendation[field]:
                completeness['present_fields'].append(field)
            else:
                completeness['missing_fields'].append(field)
        
        completeness['completeness_ratio'] = len(completeness['present_fields']) / len(standard_fields)
        
        return completeness
    
    def _identify_custom_fields(self, ai_recommendation: Dict) -> List[str]:
        """Identify custom/non-standard fields in the response"""
        standard_fields = {
            'overall_score', 'investment_thesis', 'recommendation', 'confidence_level',
            'position_size', 'time_horizon', 'risk_reward_ratio', 'key_catalysts', 'downside_risks',
            'thinking', 'details', 'processing_metadata', '_fallback_created', '_parsing_error'
        }
        
        custom_fields = []
        for key in ai_recommendation.keys():
            if key not in standard_fields:
                custom_fields.append(key)
        
        return custom_fields
    
    def _check_fallback_flags(self, ai_recommendation: Dict) -> Dict[str, bool]:
        """Check for various fallback and error flags"""
        return {
            'is_fallback': ai_recommendation.get('_fallback_created', False),
            'has_parsing_error': ai_recommendation.get('_parsing_error', False),
            'is_emergency_fallback': ai_recommendation.get('_emergency_fallback', False),
            'has_processing_metadata': bool(ai_recommendation.get('processing_metadata'))
        }
    
    def _recommend_report_sections(self, ai_recommendation: Dict, thinking_content: str, additional_details: str) -> List[str]:
        """Recommend which report sections should be included based on available content"""
        sections = ['executive_summary', 'recommendation']  # Always include these
        
        if thinking_content and len(thinking_content) > 200:
            sections.append('reasoning_analysis')
        
        if additional_details and len(additional_details) > 100:
            sections.append('additional_insights')
        
        if ai_recommendation.get('key_catalysts') and len(ai_recommendation['key_catalysts']) > 2:
            sections.append('catalyst_analysis')
        
        if ai_recommendation.get('downside_risks') and len(ai_recommendation['downside_risks']) > 2:
            sections.append('risk_assessment')
        
        if ai_recommendation.get('processing_metadata'):
            sections.append('methodology_notes')
        
        return sections
    
    def _extract_priority_insights(self, thinking_content: str, additional_details: str) -> List[str]:
        """Extract the most important insights for highlighting in reports"""
        insights = []
        
        if thinking_content:
            # Look for key insights in thinking
            key_phrases = re.findall(r'(?:key|important|crucial|critical|significant).{1,100}', thinking_content, re.IGNORECASE)
            insights.extend([phrase.strip() for phrase in key_phrases[:3]])
        
        if additional_details:
            # Look for highlighted items in details
            highlights = re.findall(r'(?:\*\*|##).{1,100}', additional_details)
            insights.extend([highlight.strip() for highlight in highlights[:2]])
        
        return insights
    
    def _suggest_visualizations(self, ai_recommendation: Dict) -> List[str]:
        """Suggest visualizations based on the content of the recommendation"""
        suggestions = []
        
        if ai_recommendation.get('overall_score'):
            suggestions.append('score_gauge_chart')
        
        if ai_recommendation.get('key_catalysts') and ai_recommendation.get('downside_risks'):
            suggestions.append('risk_catalyst_matrix')
        
        if ai_recommendation.get('time_horizon'):
            suggestions.append('timeline_chart')
        
        if ai_recommendation.get('processing_metadata', {}).get('tokens'):
            suggestions.append('processing_metrics_chart')
        
        return suggestions


def main():
    """Main entry point for standalone synthesis"""
    import argparse
    
    # Display synthesis banner
    ASCIIArt.print_banner('synthesis')
    
    # Get main logger for the standalone execution
    config = get_config()
    main_logger = config.get_main_logger('synthesizer_main')
    
    parser = argparse.ArgumentParser(description='Investment Synthesizer')
    parser.add_argument('--symbol', help='Stock symbol to analyze (required unless --weekly)')
    parser.add_argument('--symbols', nargs='*', help='Multiple stock symbols for batch analysis')
    parser.add_argument('--config', default='config.json', help='Config file path')
    parser.add_argument('--report', action='store_true', help='Generate PDF report')
    parser.add_argument('--weekly', action='store_true', help='Generate weekly report')
    parser.add_argument('--send-email', action='store_true', help='Send report via email')
    parser.add_argument('--synthesis-mode', choices=['comprehensive', 'quarterly'], default='comprehensive', 
                        help='Synthesis approach: comprehensive (default) or quarterly')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.weekly and not args.symbol and not args.symbols:
        parser.error("Either --symbol, --symbols, or --weekly is required")
    
    synthesizer = InvestmentSynthesizer()
    
    if args.weekly:
        # Generate weekly report for all tracked stocks
        try:
            tracked_stocks = config.stocks_to_track
            print(f"📊 Generating weekly report for {len(tracked_stocks)} stocks...")
            
            recommendations = []
            for symbol in tracked_stocks:
                try:
                    print(f"  Synthesizing analysis for {symbol}...")
                    recommendation = synthesizer.synthesize_analysis(symbol)
                    recommendations.append(recommendation)
                    print(f"  ✅ {symbol}: {recommendation.recommendation} ({recommendation.overall_score:.1f}/10)")
                except Exception as e:
                    print(f"  ❌ {symbol}: Failed to synthesize - {e}")
                    main_logger.warning(f"Failed to synthesize {symbol}: {e}")
            
            if recommendations:
                report_path = synthesizer.generate_report(recommendations, "weekly")
                print(f"\n📊 Weekly report generated: {report_path}")
                
                if args.send_email:
                    # TODO: Implement email sending
                    print("📧 Email sending not yet implemented")
            else:
                print("❌ No successful recommendations to include in weekly report")
                return 1
                
        except Exception as e:
            print(f"❌ Weekly report generation failed: {e}")
            main_logger.error(f"Weekly report generation failed: {e}")
            return 1
    
    elif args.symbols:
        # Batch analysis for multiple symbols
        recommendations = []
        for symbol in args.symbols:
            try:
                print(f"Synthesizing analysis for {symbol}...")
                recommendation = synthesizer.synthesize_analysis(symbol, args.synthesis_mode)
                recommendations.append(recommendation)
                print(f"✅ {symbol}: {recommendation.recommendation} ({recommendation.overall_score:.1f}/10)")
            except Exception as e:
                print(f"❌ {symbol}: Failed to synthesize - {e}")
        
        if args.report and recommendations:
            report_path = synthesizer.generate_report(recommendations, "batch")
            print(f"\n📊 Batch report generated: {report_path}")
    
    else:
        # Single symbol analysis
        symbol = args.symbol.upper()
        try:
            recommendation = synthesizer.synthesize_analysis(symbol, args.synthesis_mode)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Investment Recommendation for {symbol}")
            print(f"{'='*60}")
            print(f"Overall Score: {recommendation.overall_score:.1f}/10")
            print(f"├─ Fundamental: {recommendation.fundamental_score:.1f}/10")
            print(f"│  ├─ Income Statement: {recommendation.income_score:.1f}/10")
            print(f"│  ├─ Cash Flow: {recommendation.cashflow_score:.1f}/10")
            print(f"│  ├─ Balance Sheet: {recommendation.balance_score:.1f}/10")
            print(f"│  ├─ Growth Score: {recommendation.growth_score:.1f}/10")
            print(f"│  ├─ Value Score: {recommendation.value_score:.1f}/10")
            print(f"│  ├─ Business Quality: {recommendation.business_quality_score:.1f}/10")
            print(f"│  └─ Data Quality: {recommendation.data_quality_score:.1f}/10")
            print(f"└─ Technical: {recommendation.technical_score:.1f}/10")
            print(f"\nRecommendation: {recommendation.recommendation}")
            print(f"Confidence: {recommendation.confidence}")
            print(f"Time Horizon: {recommendation.time_horizon}")
            print(f"Position Size: {recommendation.position_size}")
            
            if recommendation.price_target:
                print(f"\nPrice Target: ${recommendation.price_target:.2f}")
                print(f"Current Price: ${recommendation.current_price:.2f}")
                upside = ((recommendation.price_target / recommendation.current_price - 1) * 100) if recommendation.current_price > 0 else 0
                print(f"Upside Potential: {upside:+.1f}%")
            
            # Generate report if requested
            if args.report:
                report_path = synthesizer.generate_report([recommendation], "synthesis")
                print(f"\n📊 Report generated: {report_path}")
                
        except Exception as e:
            print(f"❌ Analysis failed for {symbol}: {e}")
            main_logger.error(f"Analysis failed for {symbol}: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
