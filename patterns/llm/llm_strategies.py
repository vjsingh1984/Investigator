#!/usr/bin/env python3
"""
InvestiGator - LLM Strategy Pattern Implementations
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

LLM Strategy Pattern Implementations
Different strategies for LLM analysis, processing, and caching
"""

import logging
import json
import hashlib
from typing import Dict, List, Any
from datetime import datetime
import uuid

from .llm_interfaces import (
    ILLMStrategy, ILLMCacheStrategy, LLMRequest, LLMResponse, 
    LLMTaskType, LLMPriority
)
from utils.llm_response_processor import get_llm_response_processor

logger = logging.getLogger(__name__)

# ============================================================================
# LLM Analysis Strategies
# ============================================================================

class ComprehensiveLLMStrategy(ILLMStrategy):
    """Comprehensive LLM analysis strategy with detailed prompts"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.response_processor = get_llm_response_processor()
    
    def get_strategy_name(self) -> str:
        return "comprehensive"
    
    def get_model_for_task(self, task_type: LLMTaskType) -> str:
        """Get appropriate model for each task type"""
        model_mapping = {
            LLMTaskType.FUNDAMENTAL_ANALYSIS: self.config.ollama.models.get(
                'fundamental_analysis', 'llama3.1:8b-instruct-q8_0'
            ),
            LLMTaskType.TECHNICAL_ANALYSIS: self.config.ollama.models.get(
                'technical_analysis', 'qwen3-30b-40k-financial:latest'
            ),
            LLMTaskType.SYNTHESIS: self.config.ollama.models.get(
                'synthesis', 'llama-3.3-70b-instruct-q4_k_m-128K-custom:latest'
            ),
            LLMTaskType.QUARTERLY_SUMMARY: self.config.ollama.models.get(
                'quarterly_analysis', 'llama3.1:8b-instruct-q8_0'
            ),
            LLMTaskType.COMPREHENSIVE_ANALYSIS: self.config.ollama.models.get(
                'comprehensive_analysis', 'llama-3.3-70b-instruct-q4_k_m-128K-custom:latest'
            ),
            LLMTaskType.RISK_ASSESSMENT: self.config.ollama.models.get(
                'risk_assessment', 'llama3.1:8b-instruct-q8_0'
            )
        }
        
        return model_mapping.get(task_type, 'llama3.1:8b-instruct-q8_0')
    
    def prepare_request(self, task_type: LLMTaskType, data: Dict[str, Any]) -> LLMRequest:
        """Prepare detailed LLM request based on task type"""
        symbol = data.get('symbol', 'UNKNOWN')
        model = self.get_model_for_task(task_type)
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        if task_type == LLMTaskType.FUNDAMENTAL_ANALYSIS:
            return self._prepare_fundamental_request(symbol, data, model, request_id)
        elif task_type == LLMTaskType.TECHNICAL_ANALYSIS:
            return self._prepare_technical_request(symbol, data, model, request_id)
        elif task_type == LLMTaskType.SYNTHESIS:
            return self._prepare_synthesis_request(symbol, data, model, request_id)
        elif task_type == LLMTaskType.QUARTERLY_SUMMARY:
            return self._prepare_quarterly_request(symbol, data, model, request_id)
        elif task_type == LLMTaskType.COMPREHENSIVE_ANALYSIS:
            return self._prepare_comprehensive_request(symbol, data, model, request_id)
        elif task_type == LLMTaskType.RISK_ASSESSMENT:
            return self._prepare_risk_request(symbol, data, model, request_id)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _prepare_fundamental_request(self, symbol: str, data: Dict[str, Any], 
                                   model: str, request_id: str) -> LLMRequest:
        """Prepare fundamental analysis request"""
        quarterly_data = data.get('quarterly_data', [])
        filing_data = data.get('filing_data', {})
        
        system_prompt = """You are an expert financial analyst specializing in fundamental analysis of public companies. 
        Analyze the provided financial data and provide comprehensive insights on financial health, business quality, 
        and growth prospects. Be specific, quantitative, and actionable in your analysis."""
        
        prompt = f"""
        Analyze {symbol} based on the following financial data:
        
        QUARTERLY FINANCIAL DATA:
        {json.dumps(quarterly_data, indent=2)}
        
        FILING INFORMATION:
        {json.dumps(filing_data, indent=2)}
        
        Please provide a comprehensive fundamental analysis including:
        1. Financial Health Assessment (Score 1-10)
        2. Business Quality Evaluation (Score 1-10)
        3. Growth Prospects Analysis (Score 1-10)
        4. Key Investment Insights (3-5 bullet points)
        5. Primary Risks and Concerns (3-5 bullet points)
        6. Overall Investment Recommendation
        
        Format your response as structured JSON with the following schema:
        {{
            "financial_health_score": float,
            "business_quality_score": float,
            "growth_prospects_score": float,
            "overall_score": float,
            "key_insights": [list of strings],
            "key_risks": [list of strings],
            "recommendation": "BUY|HOLD|SELL",
            "confidence_level": "HIGH|MEDIUM|LOW",
            "analysis_summary": "detailed summary text"
        }}
        """
        
        return LLMRequest(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            priority=LLMPriority.HIGH.value,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            metadata={
                'task_type': LLMTaskType.FUNDAMENTAL_ANALYSIS.value,
                'symbol': symbol,
                'strategy': self.get_strategy_name()
            }
        )
    
    def _prepare_technical_request(self, symbol: str, data: Dict[str, Any], 
                                 model: str, request_id: str) -> LLMRequest:
        """Prepare technical analysis request"""
        price_data = data.get('price_data', {})
        indicators = data.get('indicators', {})
        
        system_prompt = """You are an expert technical analyst specializing in stock chart analysis and market indicators. 
        Analyze the provided price and indicator data to assess technical momentum, trend strength, and potential entry/exit points."""
        
        prompt = f"""
        Perform technical analysis for {symbol} based on:
        
        PRICE DATA:
        {json.dumps(price_data, indent=2)}
        
        TECHNICAL INDICATORS:
        {json.dumps(indicators, indent=2)}
        
        Provide technical analysis including:
        1. Trend Analysis (direction and strength)
        2. Momentum Assessment (RSI, MACD interpretation)
        3. Support/Resistance Levels
        4. Technical Score (1-10)
        5. Trading Recommendation
        6. Key Technical Risks
        
        Format as JSON:
        {{
            "technical_score": float,
            "trend_direction": "BULLISH|BEARISH|NEUTRAL",
            "trend_strength": "STRONG|MODERATE|WEAK",
            "momentum_signals": [list of strings],
            "support_levels": [list of floats],
            "resistance_levels": [list of floats],
            "recommendation": "BUY|HOLD|SELL",
            "time_horizon": "SHORT|MEDIUM|LONG",
            "risk_factors": [list of strings]
        }}
        """
        
        # Get num_predict from config for technical analysis
        num_predict = self.config.ollama.num_predict.get('technical_analysis', 4096)
        
        return LLMRequest(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            num_predict=num_predict,
            priority=LLMPriority.NORMAL.value,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            metadata={
                'task_type': LLMTaskType.TECHNICAL_ANALYSIS.value,
                'symbol': symbol,
                'strategy': self.get_strategy_name()
            }
        )
    
    def _prepare_synthesis_request(self, symbol: str, data: Dict[str, Any], 
                                 model: str, request_id: str) -> LLMRequest:
        """Prepare synthesis request combining multiple analyses"""
        fundamental_result = data.get('fundamental_analysis', {})
        technical_result = data.get('technical_analysis', {})
        
        system_prompt = """You are a senior investment analyst who synthesizes fundamental and technical analysis 
        to create comprehensive investment recommendations. Combine the provided analyses into a unified recommendation."""
        
        prompt = f"""
        Create a comprehensive investment recommendation for {symbol} by synthesizing:
        
        FUNDAMENTAL ANALYSIS:
        {json.dumps(fundamental_result, indent=2)}
        
        TECHNICAL ANALYSIS:
        {json.dumps(technical_result, indent=2)}
        
        Provide synthesis including:
        1. Combined Investment Score (1-10)
        2. Primary Investment Thesis
        3. Unified Recommendation
        4. Risk/Reward Assessment
        5. Position Sizing Guidance
        6. Time Horizon Recommendation
        
        Format as JSON:
        {{
            "overall_score": float,
            "investment_thesis": "detailed thesis",
            "recommendation": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
            "confidence_level": "HIGH|MEDIUM|LOW",
            "position_size": "LARGE|MODERATE|SMALL|AVOID",
            "time_horizon": "SHORT|MEDIUM|LONG",
            "risk_reward_ratio": float,
            "key_catalysts": [list of strings],
            "downside_risks": [list of strings]
        }}
        """
        
        # Get num_predict from config for synthesis
        num_predict = self.config.ollama.num_predict.get('synthesis', 4096)
        
        return LLMRequest(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            num_predict=num_predict,
            priority=LLMPriority.HIGH.value,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            metadata={
                'task_type': LLMTaskType.SYNTHESIS.value,
                'symbol': symbol,
                'strategy': self.get_strategy_name()
            }
        )
    
    def _prepare_quarterly_request(self, symbol: str, data: Dict[str, Any], 
                                 model: str, request_id: str) -> LLMRequest:
        """Prepare quarterly summary request"""
        quarter_data = data.get('quarter_data', {})
        
        system_prompt = """You are a financial analyst creating concise quarterly performance summaries. 
        Focus on key metrics, performance drivers, and notable changes from previous quarters."""
        
        prompt = f"""
        Summarize quarterly performance for {symbol}:
        
        QUARTER DATA:
        {json.dumps(quarter_data, indent=2)}
        
        Provide quarterly summary with:
        1. Key Performance Metrics
        2. Revenue and Profitability Trends
        3. Notable Changes from Prior Quarter
        4. Management Guidance Impact
        5. Quarterly Score (1-10)
        
        Format as JSON:
        {{
            "quarterly_score": float,
            "revenue_performance": "string",
            "profitability_trends": "string",
            "key_changes": [list of strings],
            "performance_drivers": [list of strings],
            "concerns": [list of strings],
            "outlook": "POSITIVE|NEUTRAL|NEGATIVE"
        }}
        """
        
        # Extract fiscal period information - try multiple sources
        fiscal_year = 'UNKNOWN'
        fiscal_period = 'UNKNOWN'
        
        # First try to get from top-level data (added by SEC facade)
        if 'fiscal_year' in data:
            fiscal_year = str(data['fiscal_year'])
        if 'fiscal_period' in data:
            fiscal_period = str(data['fiscal_period'])
            
        # Fallback to quarter_data
        if fiscal_year == 'UNKNOWN' or fiscal_period == 'UNKNOWN':
            fiscal_year = quarter_data.get('fiscal_year', fiscal_year)
            fiscal_period = quarter_data.get('fiscal_period', fiscal_period)
        
        # Get num_predict from config for quarterly analysis
        num_predict = self.config.ollama.num_predict.get('quarterly_analysis', 2048)
        
        return LLMRequest(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            num_predict=num_predict,
            priority=LLMPriority.NORMAL.value,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            metadata={
                'task_type': LLMTaskType.QUARTERLY_SUMMARY.value,
                'symbol': symbol,
                'fiscal_year': fiscal_year,
                'fiscal_period': fiscal_period,
                'strategy': self.get_strategy_name()
            }
        )
    
    def _prepare_comprehensive_request(self, symbol: str, data: Dict[str, Any], 
                                     model: str, request_id: str) -> LLMRequest:
        """Prepare comprehensive analysis request"""
        quarterly_analyses = data.get('quarterly_analyses', [])
        aggregated_data = data.get('aggregated_data', {})
        prompt = data.get('prompt', '')
        
        # Use the custom prompt provided by the caller
        system_prompt = """You are an expert equity research analyst specializing in fundamental analysis and financial trends. Always respond with properly formatted JSON only, no additional text or markdown."""
        
        # Get num_predict from config for comprehensive analysis
        num_predict = self.config.ollama.num_predict.get('comprehensive_analysis', 6144)
        
        # Extract fiscal period information - try to get the most recent quarter
        fiscal_year = 'UNKNOWN'
        fiscal_period = 'UNKNOWN'
        
        # First try to extract from aggregated_data
        if aggregated_data and 'periods' in aggregated_data:
            periods = aggregated_data['periods']
            if periods:
                # Use the most recent period
                latest_period = periods[0] if isinstance(periods, list) else periods
                if isinstance(latest_period, dict):
                    fiscal_year = latest_period.get('fiscal_year', 'UNKNOWN')
                    fiscal_period = latest_period.get('fiscal_period', 'UNKNOWN')
        
        # Fallback: try to extract from quarterly_analyses
        if fiscal_year == 'UNKNOWN' and quarterly_analyses:
            # Look at the first quarterly analysis for fiscal period info
            first_quarter = quarterly_analyses[0] if quarterly_analyses else {}
            if 'fiscal_year' in first_quarter:
                fiscal_year = first_quarter['fiscal_year']
            if 'fiscal_period' in first_quarter:
                fiscal_period = first_quarter['fiscal_period']
        
        # Final fallback: try to extract from the prompt itself
        if fiscal_year == 'UNKNOWN' or fiscal_period == 'UNKNOWN':
            import re
            period_match = re.search(r'\b(\d{4})-([QF][1-4Y]?)\b', prompt)
            if period_match:
                fiscal_year = period_match.group(1)
                fiscal_period = period_match.group(2)
        
        return LLMRequest(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            num_predict=num_predict,
            priority=LLMPriority.HIGH.value,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            metadata={
                'task_type': LLMTaskType.COMPREHENSIVE_ANALYSIS.value,
                'symbol': symbol,
                'fiscal_year': fiscal_year,
                'fiscal_period': fiscal_period,
                'strategy': self.get_strategy_name()
            }
        )
    
    def _prepare_risk_request(self, symbol: str, data: Dict[str, Any], 
                            model: str, request_id: str) -> LLMRequest:
        """Prepare risk assessment request using J2 template"""
        # Import prompt manager locally to avoid circular imports
        from utils.prompt_manager import get_prompt_manager
        
        # Get current price if available
        current_price = None
        if 'price_data' in data:
            price_data = data['price_data']
            if isinstance(price_data, dict) and 'Close' in price_data:
                # Get the most recent close price
                close_prices = price_data['Close']
                if isinstance(close_prices, list) and close_prices:
                    current_price = close_prices[-1]
                elif isinstance(close_prices, (int, float)):
                    current_price = close_prices
        
        # Prepare template variables for risk assessment
        template_vars = {
            'symbol': symbol,
            'analysis_date': datetime.utcnow().strftime('%Y-%m-%d'),
            'current_price': current_price,
            'fundamental_analysis': json.dumps(data.get('fundamental_analysis', {}), indent=2),
            'technical_analysis': json.dumps(data.get('technical_analysis', {}), indent=2),
            'market_data': json.dumps(data.get('market_data', {}), indent=2),
            'historical_data': json.dumps(data.get('historical_data', {}), indent=2)
        }
        
        # Render template
        prompt_manager = get_prompt_manager()
        try:
            prompt = prompt_manager.render_risk_assessment_prompt(**template_vars)
        except Exception as e:
            # Fallback to basic prompt if template fails
            self.logger.warning(f"Failed to render risk assessment template: {e}, using fallback")
            prompt = f"""
            Assess investment risks for {symbol}:
            
            ALL AVAILABLE DATA:
            {json.dumps(data, indent=2)}
            
            Identify and analyze:
            1. Fundamental Risks (business, financial)
            2. Technical Risks (chart patterns, momentum)  
            3. Market Risks (sector, macro factors)
            4. Company-Specific Risks
            5. Overall Risk Score (1-10, where 10 is highest risk)
            
            Format as JSON:
            {{
                "overall_risk_score": float,
                "fundamental_risks": [list of strings],
                "technical_risks": [list of strings],
                "market_risks": [list of strings],
                "company_risks": [list of strings],
                "risk_mitigation": [list of strings],
                "maximum_position_size": "percentage recommendation"
            }}
            """
        
        # Get num_predict from config for risk assessment
        num_predict = self.config.ollama.num_predict.get('risk_assessment', 2048)
        
        return LLMRequest(
            model=model,
            prompt=prompt,
            system_prompt="You are a senior risk management specialist and portfolio manager with expertise in systematic risk assessment across fundamental, technical, market, and company-specific factors.",
            temperature=0.3,
            num_predict=num_predict,
            priority=LLMPriority.NORMAL.value,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            metadata={
                'task_type': LLMTaskType.RISK_ASSESSMENT.value,
                'symbol': symbol,
                'strategy': self.get_strategy_name()
            }
        )
    
    def process_response(self, response: LLMResponse, task_type: LLMTaskType) -> Dict[str, Any]:
        """Process LLM response into structured data using common processor"""
        try:
            # Handle error responses
            if response.error:
                return {'error': response.error}
            
            # Use common processor to handle response content
            processed_content, metadata = self.response_processor.process_response(response.content, from_cache=False)
            
            # Try to parse as JSON
            result = self.response_processor.extract_json_from_text(processed_content)
            
            if result is None:
                # If JSON extraction failed, return text response
                self.logger.warning(f"Failed to extract JSON from response for {task_type}")
                return {
                    'error': 'Failed to parse JSON response',
                    'raw_response': processed_content[:500] + '...' if len(processed_content) > 500 else processed_content,
                    'thinking': metadata.get('thinking_content', '')
                }
            
            # Add any remaining text as detail (safely handled)
            if processed_content != json.dumps(result, ensure_ascii=False):
                # Calculate text without JSON
                import re
                json_str = json.dumps(result, ensure_ascii=False)
                # Remove the JSON part from content to get remaining details
                remaining_text = processed_content.replace(json_str, '').strip()
                if remaining_text:
                    # If remaining text looks like JSON, don't add it to avoid duplication
                    remaining_clean = remaining_text.strip()
                    if remaining_clean.startswith('{') and remaining_clean.endswith('}'):
                        # Try to parse it - if it's valid JSON and similar to our result, skip it
                        try:
                            remaining_json = json.loads(remaining_clean)
                            # If it's the same data, skip adding to detail to avoid duplication
                            if (remaining_json.get('overall_score') == result.get('overall_score') and
                                remaining_json.get('recommendation') == result.get('recommendation')):
                                self.logger.debug("Skipping duplicate JSON content in detail field")
                                remaining_text = ""
                        except json.JSONDecodeError:
                            # If it's not valid JSON, keep it as text detail
                            pass
                    
                    if remaining_text.strip():
                        # Clean up the remaining text by removing embedded JSON blocks and excessive escaping
                        cleaned_detail = self._clean_detail_content(remaining_text.strip())
                        if cleaned_detail:
                            result['detail'] = cleaned_detail
            
            # Add thinking content if present (properly escaped)
            if metadata.get('thinking_content'):
                result['thinking'] = self.response_processor.escape_for_json(metadata['thinking_content'])
            
            # Add processing metadata
            result['processing_metadata'] = {
                'task_type': task_type.value,
                'model_used': response.model,
                'processing_time_ms': response.processing_time_ms,
                'tokens_used': response.tokens_used,
                'request_id': response.request_id,
                'timestamp': response.timestamp.isoformat() if response.timestamp else None
            }
            
            # Add token details if available in metadata
            if response.metadata and 'tokens' in response.metadata:
                result['processing_metadata']['tokens'] = response.metadata['tokens']
            
            # Add timing details if available
            if response.metadata and 'timings' in response.metadata:
                result['processing_metadata']['timings'] = response.metadata['timings']
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing LLM response: {e}")
            return {'error': f'Response processing failed: {str(e)}'}
    
    def _clean_detail_content(self, detail_text: str) -> str:
        """
        Clean the detail content by removing embedded JSON blocks and fixing escaping.
        
        Args:
            detail_text: Raw detail text that may contain JSON blocks and escaped content
            
        Returns:
            Cleaned markdown content suitable for PDF display
        """
        if not detail_text:
            return ""
        
        # Step 1: Remove embedded JSON blocks (markdown code blocks)
        import re
        # Remove ```json...``` blocks
        detail_text = re.sub(r'```json\s*\n(.*?)\n```', '', detail_text, flags=re.DOTALL)
        # Remove ``` blocks without language specification that contain JSON
        detail_text = re.sub(r'```\s*\n(\{.*?\})\s*\n```', '', detail_text, flags=re.DOTALL)
        
        # Step 2: Remove standalone JSON objects (not in code blocks)
        # Look for standalone JSON objects that start with { and end with }
        lines = detail_text.split('\n')
        cleaned_lines = []
        in_json_block = False
        brace_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Detect start of JSON block
            if stripped.startswith('{') and not in_json_block:
                in_json_block = True
                brace_count = stripped.count('{') - stripped.count('}')
                if brace_count <= 0:
                    in_json_block = False  # Single line JSON
                continue
            
            # Skip lines that are part of JSON block
            if in_json_block:
                brace_count += stripped.count('{') - stripped.count('}')
                if brace_count <= 0:
                    in_json_block = False
                continue
            
            # Keep non-JSON lines
            cleaned_lines.append(line)
        
        detail_text = '\n'.join(cleaned_lines)
        
        # Step 3: Fix escaping issues
        # Convert escaped newlines to actual newlines for better readability
        detail_text = detail_text.replace('\\n', '\n')
        detail_text = detail_text.replace('\\t', '\t')
        
        # Step 4: Remove common LLM response artifacts
        # Remove lines that just explain the JSON structure
        lines = detail_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip lines that are just explaining the JSON format
            if any(phrase in stripped.lower() for phrase in [
                'here is the', 'here\'s the', 'the above json', 'json format',
                'breakdown of each section', 'please note that this',
                'this recommendation is based on', 'should not be considered investment advice'
            ]):
                continue
            
            # Skip empty lines at the beginning and end
            if stripped or cleaned_lines:  # Keep internal empty lines
                cleaned_lines.append(line)
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        detail_text = '\n'.join(cleaned_lines).strip()
        
        # Step 5: Return only substantial content
        # If the remaining content is too short or just boilerplate, return empty
        if len(detail_text) < 50 or detail_text.lower().startswith('the above'):
            return ""
        
        return detail_text

# ============================================================================
# Quick Analysis Strategy
# ============================================================================

class QuickLLMStrategy(ILLMStrategy):
    """Quick analysis strategy with simplified prompts"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.response_processor = get_llm_response_processor()
    
    def get_strategy_name(self) -> str:
        return "quick"
    
    def get_model_for_task(self, task_type: LLMTaskType) -> str:
        """Use faster, smaller models for quick analysis"""
        return self.config.ollama.models.get('quick_analysis', 'llama3.1:8b-instruct-q8_0')
    
    def prepare_request(self, task_type: LLMTaskType, data: Dict[str, Any]) -> LLMRequest:
        """Prepare simplified request for quick analysis"""
        symbol = data.get('symbol', 'UNKNOWN')
        model = self.get_model_for_task(task_type)
        request_id = str(uuid.uuid4())
        
        # Simplified prompts for quick analysis
        if task_type in [LLMTaskType.FUNDAMENTAL_ANALYSIS, LLMTaskType.QUARTERLY_SUMMARY]:
            prompt = f"Quickly analyze {symbol} financials. Provide: score (1-10), recommendation (BUY/HOLD/SELL), 2-3 key points. Data: {json.dumps(data, indent=2)[:1000]}..."
        elif task_type == LLMTaskType.TECHNICAL_ANALYSIS:
            prompt = f"Quick technical analysis for {symbol}. Provide: trend, score (1-10), recommendation. Data: {json.dumps(data, indent=2)[:1000]}..."
        else:
            prompt = f"Quick analysis for {symbol}. Provide brief assessment and recommendation. Data: {json.dumps(data, indent=2)[:1000]}..."
        
        return LLMRequest(
            model=model,
            prompt=prompt,
            system_prompt="Provide quick, concise financial analysis.",
            temperature=0.1,
            num_predict=200,  # Limit response length
            priority=LLMPriority.NORMAL.value,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            metadata={
                'task_type': task_type.value,
                'symbol': symbol,
                'strategy': self.get_strategy_name()
            }
        )
    
    def process_response(self, response: LLMResponse, task_type: LLMTaskType) -> Dict[str, Any]:
        """Process quick analysis response"""
        if response.error:
            return {'error': response.error}
        
        # For quick analysis, return raw text with minimal processing
        return {
            'quick_analysis': response.content,
            'model_used': response.model,
            'processing_time_ms': response.processing_time_ms,
            'task_type': task_type.value,
            'strategy': 'quick'
        }

# ============================================================================
# LLM Cache Strategies
# ============================================================================


class LLMCacheStrategy(ILLMCacheStrategy):
    """Optimized caching strategy for maximum performance and cache hit rates"""
    
    def __init__(self, config):
        self.config = config
    
    def get_cache_key(self, request: LLMRequest) -> str:
        """DEPRECATED: This method is no longer used. Cache keys are now generated as dictionaries directly in LLMCacheHandler._generate_cache_key_dict()"""
        # This method is kept for interface compatibility but should not be used
        # All cache operations now use dictionary-based keys for consistency
        return "deprecated_hash_key"
    
    def should_cache(self, request: LLMRequest, response: LLMResponse) -> bool:
        """Cache almost everything for maximum performance"""
        return not response.error and len(response.content) > 10
    
    def get_ttl(self, task_type: LLMTaskType) -> int:
        """TTL values aligned with cache manager specifications"""
        ttl_mapping = {
            LLMTaskType.QUARTERLY_SUMMARY: 86400 * 30,      # 30 days - fundamental analysis
            LLMTaskType.COMPREHENSIVE_ANALYSIS: 86400 * 30, # 30 days - fundamental analysis
            LLMTaskType.FUNDAMENTAL_ANALYSIS: 86400 * 30,   # 30 days - fundamental analysis
            LLMTaskType.TECHNICAL_ANALYSIS: 86400 * 7,      # 7 days - technical analysis
            LLMTaskType.SYNTHESIS: 86400 * 7,               # 7 days - synthesis
            LLMTaskType.RISK_ASSESSMENT: 86400 * 30         # 30 days - part of fundamental
        }
        
        return ttl_mapping.get(task_type, 86400 * 7)  # Default 7 days
    
    def is_cacheable_task(self, task_type: LLMTaskType) -> bool:
        """Cache all task types"""
        return True
