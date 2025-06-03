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
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from config import get_config
from utils.db import DatabaseManager, get_llm_response_store_dao
from utils.cache import CacheManager, CacheType, get_cache_manager
from patterns.llm.llm_facade import create_llm_facade
from utils.chart_generator import ChartGenerator
from utils.report_generator import PDFReportGenerator, ReportConfig
from utils.weekly_report_generator import WeeklyReportGenerator
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
        self.llm_dao = get_llm_response_store_dao()
        
        # Initialize loggers
        self.main_logger = self.config.get_main_logger('synthesizer')
        
        # Cache directories
        self.llm_cache_dir = self.config.data_dir / "llm_cache"
        self.llm_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.main_logger.info("Investment synthesizer initialized")
    
    def synthesize_analysis(self, symbol: str) -> InvestmentRecommendation:
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
            
            # Get current price from technical analysis response
            current_price = 0
            if llm_responses.get('technical'):
                tech_content = llm_responses['technical'].get('content', '')
                if isinstance(tech_content, dict):
                    tech_content = json.dumps(tech_content)
                elif not isinstance(tech_content, str):
                    tech_content = str(tech_content)
                import re
                price_match = re.search(r'"current_price":\s*([\d.]+)', tech_content)
                if price_match:
                    current_price = float(price_match.group(1))
            if current_price == 0:
                current_price = latest_data.get('technical', {}).get('current_price', 0)
            
            # Assess data quality
            data_quality = self._assess_data_quality(llm_responses, latest_data)
            symbol_logger.info(f"Data quality score: {data_quality:.2f}")
            
            # Use prompt manager for synthesis prompt with JSON response
            from utils.prompt_manager import get_prompt_manager
            prompt_manager = get_prompt_manager()
            
            # Prepare data for synthesis prompt
            from utils.synthesis_helpers import format_fundamental_data_for_synthesis, format_technical_data_for_synthesis, get_performance_data
            
            fundamental_data_str = format_fundamental_data_for_synthesis(llm_responses.get('fundamental', {}))
            technical_data_str = format_technical_data_for_synthesis(llm_responses.get('technical', {}))
            
            synthesis_prompt = prompt_manager.render_investment_synthesis_prompt(
                symbol=symbol,
                analysis_date=datetime.now().strftime('%Y-%m-%d'),
                current_price=latest_data.get('current_price', 0.0),
                sector_context=self._get_sector_context(symbol),
                market_environment=self._get_market_environment_context(),
                fundamental_data=fundamental_data_str,
                technical_data=technical_data_str,
                latest_market_data=str(latest_data),
                performance_data=get_performance_data(symbol)
            )
            
            # Generate synthesis using LLM
            model_name = self.config.ollama.models['report_generation']
            
            # Enhanced system prompt for institutional-grade analysis
            system_prompt = """You are a senior portfolio manager and CFA charterholder with 25+ years of institutional investment experience. You excel at:

• Synthesizing complex multi-source financial analyses into actionable investment decisions
• Risk-adjusted portfolio construction for $2B+ institutional mandates
• Quantitative valuation analysis across market cycles and economic regimes
• Technical analysis integration with fundamental research for optimal timing
• ESG integration and fiduciary standard investment processes

Your responses must be precise, quantitative, and suitable for institutional investment committees. Focus on risk-adjusted returns, position sizing discipline, and clear execution frameworks. Provide specific price targets, stop-losses, and measurable investment criteria."""
            
            # Check cache first for synthesis response
            cache_key = {
                'symbol': symbol,
                'form_type': 'N/A',
                'period': 'N/A',
                'llm_type': 'full'
            }
            
            cached_response = self.cache_manager.get(CacheType.LLM_RESPONSE, cache_key)
            
            if cached_response:
                symbol_logger.info(f"Using cached synthesis response for {symbol}")
                self.main_logger.info(f"Cache HIT for synthesis: {symbol}")
                
                # Extract response from cache
                response_data = cached_response.get('response', {})
                if isinstance(response_data, str):
                    import json
                    try:
                        response_data = json.loads(response_data)
                    except:
                        pass
                
                synthesis_response = response_data.get('content', '') if isinstance(response_data, dict) else str(response_data)
                processing_time_ms = cached_response.get('metadata', {}).get('processing_time_ms', 0)
            else:
                symbol_logger.info(f"No cached synthesis found, generating with {model_name}")
                self.main_logger.info(f"Cache MISS for synthesis: {symbol}, generating with {model_name} (32K context)")
                
                start_time = time.time()
                # Use LLM facade for synthesis
                synthesis_data = {
                    'symbol': symbol,
                    'fundamental_analysis': fundamental_data,
                    'technical_analysis': technical_data
                }
                
                synthesis_result = self.ollama.synthesize_analysis(
                    symbol=symbol,
                    fundamental_result=fundamental_data,
                    technical_result=technical_data
                )
                
                # Extract the synthesis response
                synthesis_response = synthesis_result.get('investment_thesis', '')
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # Save synthesis LLM response through cache manager
                self._save_synthesis_llm_response(symbol, synthesis_prompt, synthesis_response, processing_time_ms)
            
            symbol_logger.info(f"Synthesis response generated in {processing_time_ms}ms")
            
            # Parse JSON synthesis response
            ai_recommendation = prompt_manager.validate_json_response(synthesis_response)
            
            # If we got an error response, try to extract what we can
            if 'error' in ai_recommendation:
                symbol_logger.warning("JSON parsing failed, using fallback values")
                # Set reasonable defaults
                ai_recommendation = {
                    'executive_summary': {'investment_thesis': 'Analysis based on available data'},
                    'composite_scores': {
                        'overall_score': overall_score,
                        'fundamental_score': fundamental_score,
                        'technical_score': technical_score,
                        'income_statement_score': 8.0,
                        'balance_sheet_score': 8.0,
                        'cash_flow_score': 7.0
                    },
                    'investment_recommendation': {
                        'recommendation': 'HOLD' if 4 <= overall_score <= 6 else ('BUY' if overall_score > 6 else 'SELL'),
                        'confidence_level': 'MEDIUM',
                        'time_horizon': 'MEDIUM_TERM',
                        'position_sizing': {'recommended_weight': 0.03}
                    }
                }
            
            # Extract scores from parsed response or use defaults
            if 'composite_scores' in ai_recommendation:
                # Use LLM-provided scores if available
                overall_score = ai_recommendation['composite_scores'].get('overall_score', overall_score)
                fundamental_score = ai_recommendation['composite_scores'].get('fundamental_score', fundamental_score)
                technical_score = ai_recommendation['composite_scores'].get('technical_score', technical_score)
                income_score = ai_recommendation['composite_scores'].get('income_statement_score', 
                    self._extract_income_score(llm_responses, ai_recommendation))
                cashflow_score = ai_recommendation['composite_scores'].get('cash_flow_score',
                    self._extract_cashflow_score(llm_responses, ai_recommendation))
                balance_score = ai_recommendation['composite_scores'].get('balance_sheet_score',
                    self._extract_balance_score(llm_responses, ai_recommendation))
            else:
                # Extract detailed fundamental scores
                income_score = self._extract_income_score(llm_responses, ai_recommendation)
                cashflow_score = self._extract_cashflow_score(llm_responses, ai_recommendation)
                balance_score = self._extract_balance_score(llm_responses, ai_recommendation)
            
            # Determine final recommendation with risk management
            final_recommendation = self._determine_final_recommendation(overall_score, ai_recommendation, data_quality)
            
            # Calculate price targets and risk levels
            price_target = self._calculate_price_target(symbol, llm_responses, ai_recommendation)
            stop_loss = self._calculate_stop_loss(current_price, final_recommendation, overall_score)
            
            # Create comprehensive recommendation
            recommendation = InvestmentRecommendation(
                symbol=symbol,
                overall_score=overall_score,
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                income_score=income_score,
                cashflow_score=cashflow_score,
                balance_score=balance_score,
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
                key_risks=self._extract_comprehensive_risks(llm_responses, ai_recommendation),
                key_insights=self._extract_comprehensive_insights(llm_responses, ai_recommendation),
                entry_strategy=ai_recommendation.get('entry_strategy', ''),
                exit_strategy=ai_recommendation.get('exit_strategy', ''),
                stop_loss=stop_loss,
                analysis_timestamp=datetime.utcnow(),
                data_quality_score=data_quality
            )
            
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
            return self._create_default_recommendation(symbol, f"Synthesis error: {str(e)}")
    
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
            
            # Generate 3D fundamental plot
            if len(rec_dicts) > 1:
                fundamental_3d = self.chart_generator.generate_3d_fundamental_plot(rec_dicts)
                if fundamental_3d:
                    chart_paths.append(fundamental_3d)
                
                # Generate 2D technical vs fundamental plot
                tech_fund_2d = self.chart_generator.generate_2d_technical_fundamental_plot(rec_dicts)
                if tech_fund_2d:
                    chart_paths.append(tech_fund_2d)
            
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
        """Fetch all LLM responses for a symbol from cache/database"""
        self.main_logger.info(f"Fetching LLM responses for {symbol}")
        
        try:
            llm_responses = {
                'fundamental': {},
                'technical': None
            }
            
            # Use cache manager to fetch SEC fundamental responses
            # First, get a list of available periods from the database
            sec_responses = self.llm_dao.get_llm_responses_by_symbol(symbol, llm_type='sec')
            
            for resp in sec_responses:
                form_type = resp['form_type']
                period = resp['period']
                
                # Use cache manager to fetch the full response
                cache_key = {
                    'symbol': symbol,
                    'form_type': form_type,
                    'period': period,
                    'llm_type': 'sec'
                }
                
                cached_resp = self.cache_manager.get(CacheType.LLM_RESPONSE, cache_key)
                
                if cached_resp:
                    response_data = cached_resp.get('response', {}) if isinstance(cached_resp.get('response'), dict) else {}
                    metadata = cached_resp.get('metadata', {}) if isinstance(cached_resp.get('metadata'), dict) else {}
                    
                    key = f"{form_type}_{period}"
                    llm_responses['fundamental'][key] = {
                        'content': response_data.get('content', ''),
                        'metadata': metadata,
                        'form_type': form_type,
                        'period': period
                    }
            
            # Use cache manager to fetch technical analysis response
            ta_cache_key = {
                'symbol': symbol,
                'form_type': 'N/A',
                'period': 'N/A',
                'llm_type': 'ta'
            }
            
            ta_result = self.cache_manager.get(CacheType.LLM_RESPONSE, ta_cache_key)
            
            if ta_result:
                response_data = ta_result.get('response', {}) if isinstance(ta_result.get('response'), dict) else {}
                metadata = ta_result.get('metadata', {}) if isinstance(ta_result.get('metadata'), dict) else {}
                llm_responses['technical'] = {
                    'content': response_data.get('content', ''),
                    'metadata': metadata
                }
            
            self.main_logger.info(f"Retrieved {len(llm_responses['fundamental'])} fundamental and "
                      f"{'1' if llm_responses['technical'] else '0'} technical LLM responses")
            
            return llm_responses
            
        except Exception as e:
            self.main_logger.error(f"Error fetching LLM responses: {e}")
            return {'fundamental': {}, 'technical': None}
    
    def _fetch_latest_data(self, symbol: str) -> Dict:
        """Fetch latest fundamental and technical data"""
        self.main_logger.info(f"Fetching latest data for {symbol}")
        
        # For now, return empty data structure since we're working with LLM responses
        # This can be enhanced to fetch from quarterly_metrics and technical data tables
        return {
            'fundamental': {},
            'technical': {}
        }
    
    def _calculate_fundamental_score(self, llm_responses: Dict) -> float:
        """Calculate fundamental score from LLM responses"""
        fundamental_responses = llm_responses.get('fundamental', {})
        if not fundamental_responses:
            return 5.0
        
        scores = []
        for key, response in fundamental_responses.items():
            content = response.get('content', '')
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            # Extract score from LLM response
            import re
            score_match = re.search(r'(?:Financial Health|Overall|Score)[:\s]*(\d+(?:\.\d+)?)/10', content)
            if score_match:
                scores.append(float(score_match.group(1)))
        
        return sum(scores) / len(scores) if scores else 5.0
    
    def _calculate_technical_score(self, llm_responses: Dict) -> float:
        """Calculate technical score from LLM response"""
        technical_response = llm_responses.get('technical')
        if not technical_response:
            return 5.0
        
        content = technical_response.get('content', '')
        if isinstance(content, dict):
            content = json.dumps(content)
        elif not isinstance(content, str):
            content = str(content)
        import re
        score_match = re.search(r'TECHNICAL SCORE[:\s]*(\d+(?:\.\d+)?)', content)
        if score_match:
            return float(score_match.group(1))
        
        return 5.0
    
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
            return 5.0
        
        norm_fund_weight = fund_weight / total_weight
        norm_tech_weight = tech_weight / total_weight
        
        overall_score = (fundamental_score * norm_fund_weight + technical_score * norm_tech_weight)
        
        return round(overall_score, 1)
    
    def _assess_data_quality(self, llm_responses: Dict, latest_data: Dict) -> float:
        """Assess overall data quality and completeness"""
        quality_score = 0.0
        max_score = 1.0
        
        # Check fundamental data availability
        if llm_responses.get('fundamental'):
            quality_score += 0.4
            if len(llm_responses['fundamental']) >= 3:  # Multiple quarters
                quality_score += 0.1
        
        # Check technical data availability
        if llm_responses.get('technical'):
            quality_score += 0.3
        
        # Check data freshness
        if latest_data.get('technical', {}).get('current_price'):
            quality_score += 0.1
        
        if latest_data.get('fundamental'):
            quality_score += 0.1
        
        return min(quality_score, max_score)
    
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
        base_fundamental = self._calculate_fundamental_score(llm_responses)
        
        # Look for income-related keywords in responses
        income_keywords = ['revenue', 'income', 'earnings', 'profit', 'margin', 'sales']
        income_score_adjustments = []
        
        for resp in llm_responses.get('fundamental', {}).values():
            content = resp.get('content', '')
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            content = content.lower()
            income_mentions = sum(1 for keyword in income_keywords if keyword in content)
            if income_mentions > 3:
                income_score_adjustments.append(0.5)
            elif income_mentions > 0:
                income_score_adjustments.append(0.0)
            else:
                income_score_adjustments.append(-0.5)
        
        adjustment = sum(income_score_adjustments) / len(income_score_adjustments) if income_score_adjustments else 0
        return max(1.0, min(10.0, base_fundamental + adjustment))
    
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
        return max(1.0, min(10.0, base_fundamental + adjustment))
    
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
        return max(1.0, min(10.0, base_fundamental + adjustment))
    
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
    
    def _calculate_price_target(self, symbol: str, llm_responses: Dict, ai_recommendation: Dict) -> float:
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
        
        # Default to current price + expected return based on score
        current_price = self._fetch_latest_data(symbol).get('technical', {}).get('current_price', 100)
        overall_score = ai_recommendation.get('composite_scores', {}).get('overall_score', 5.0)
        
        # Expected return mapping
        if overall_score >= 8.0:
            expected_return = 0.25  # 25%
        elif overall_score >= 6.5:
            expected_return = 0.15  # 15%
        elif overall_score >= 5.0:
            expected_return = 0.08  # 8%
        else:
            expected_return = -0.10  # -10%
        
        return round(current_price * (1 + expected_return), 2)
    
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
    
    def _extract_comprehensive_risks(self, llm_responses: Dict, ai_recommendation: Dict) -> List[str]:
        """Extract and prioritize comprehensive risk factors"""
        import re
        risks = []
        
        # From AI synthesis
        if isinstance(ai_recommendation, dict):
            ai_risks = ai_recommendation.get('key_risks', [])
            if isinstance(ai_risks, list):
                risks.extend(ai_risks[:3])
        
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
        
        return unique_risks[:6] if unique_risks else ["Limited risk data available"]
    
    def _extract_comprehensive_insights(self, llm_responses: Dict, ai_recommendation: Dict) -> List[str]:
        """Extract and prioritize comprehensive insights"""
        import re
        insights = []
        
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
        
        return unique_insights[:6] if unique_insights else ["Analysis insights pending"]
    
    def _save_synthesis_llm_response(self, symbol: str, prompt: str, response: str, 
                                   processing_time_ms: int):
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
                'model': self.config.ollama.models['report_generation']
            }
            
            prompt_context = {
                'prompt': prompt,
                'prompt_length': len(prompt),
                'included_fundamental': True,
                'included_technical': True,
                'included_latest_data': True
            }
            
            model_info = {
                'model': metadata['model'],
                'temperature': 0.3,
                'top_p': 0.9,
                'num_ctx': 32768,
                'num_predict': 4096
            }
            
            # Save to cache using cache manager
            cache_key = {
                'symbol': symbol,
                'form_type': 'N/A',
                'period': 'N/A',
                'llm_type': 'full'
            }
            cache_value = {
                'prompt_context': prompt_context,
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
            
            # Save prompt
            prompt_file = symbol_cache_dir / "prompt_synthesis.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            # Save response
            response_file = symbol_cache_dir / "response_synthesis.txt"
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(response)
                
            self.main_logger.info(f"💾 Saved synthesis prompt and response to {symbol_cache_dir}")
            
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
    
    def _create_default_recommendation(self, symbol: str, error_msg: str) -> InvestmentRecommendation:
        """Create default recommendation when synthesis fails"""
        return InvestmentRecommendation(
            symbol=symbol,
            overall_score=5.0,
            fundamental_score=5.0,
            technical_score=5.0,
            income_score=5.0,
            cashflow_score=5.0,
            balance_score=5.0,
            recommendation='HOLD',
            confidence='LOW',
            price_target=None,
            current_price=None,
            investment_thesis=f"Analysis incomplete: {error_msg}",
            time_horizon='MEDIUM-TERM',
            position_size='AVOID',
            key_catalysts=[],
            key_risks=[error_msg],
            key_insights=[],
            entry_strategy='',
            exit_strategy='',
            stop_loss=None,
            analysis_timestamp=datetime.utcnow(),
            data_quality_score=0.1
        )
    
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
        """Create comprehensive 32K context synthesis prompt optimized for 64GB Mac with q4/q8 quantization"""
        
        # Extract full fundamental responses (increased from 2000 to 8000 chars for 32K context)
        fundamental_summaries = []
        quarterly_trends = []
        key_metrics_evolution = []
        
        for key, resp in llm_responses.get('fundamental', {}).items():
            if resp and resp.get('content'):
                content = resp['content'][:8000]  # Increased for full context
                form_type = resp.get('form_type', 'Unknown')
                period = resp.get('period', 'Unknown')
                
                fundamental_summaries.append(f"{chr(10)}=== {form_type} {period} COMPREHENSIVE ANALYSIS ==={chr(10)}{content}")
                
                # Extract quarterly performance metrics for trend analysis
                if 'Q' in period:
                    quarterly_trends.append({
                        'period': period,
                        'form': form_type,
                        'content': content[:1500]
                    })
        
        # Extract full technical response (increased from 2000 to 6000 chars)
        technical_analysis = ""
        technical_signals = []
        risk_factors = []
        
        if llm_responses.get('technical') and llm_responses['technical'].get('content'):
            technical_content = llm_responses['technical']['content']
            if isinstance(technical_content, dict):
                technical_content = json.dumps(technical_content)
            elif not isinstance(technical_content, str):
                technical_content = str(technical_content)
            technical_content = technical_content[:6000]
            technical_analysis = technical_content
            
            # Extract specific technical signals for synthesis
            import re
            signal_patterns = [
                r'TECHNICAL SCORE:\s*(\d+)',
                r'Primary Trend:\s*([A-Za-z]+)',
                r'RSI[^:]*:\s*([\d.]+)',
                r'Support[^:]*:\s*([\d.$,\s]+)',
                r'Resistance[^:]*:\s*([\d.$,\s]+)'
            ]
            
            for pattern in signal_patterns:
                matches = re.findall(pattern, technical_content, re.IGNORECASE)
                if matches:
                    technical_signals.extend(matches)
        
        # Format comprehensive data for maximum context utilization
        fund_data = latest_data.get('fundamental', {})
        tech_data = latest_data.get('technical', {})
        
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


def main():
    """Main entry point for standalone synthesis"""
    import argparse
    
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
                recommendation = synthesizer.synthesize_analysis(symbol)
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
            recommendation = synthesizer.synthesize_analysis(symbol)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Investment Recommendation for {symbol}")
            print(f"{'='*60}")
            print(f"Overall Score: {recommendation.overall_score:.1f}/10")
            print(f"├─ Fundamental: {recommendation.fundamental_score:.1f}/10")
            print(f"│  ├─ Income Statement: {recommendation.income_score:.1f}/10")
            print(f"│  ├─ Cash Flow: {recommendation.cashflow_score:.1f}/10")
            print(f"│  └─ Balance Sheet: {recommendation.balance_score:.1f}/10")
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