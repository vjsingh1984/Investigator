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

logger = logging.getLogger(__name__)

# ============================================================================
# LLM Analysis Strategies
# ============================================================================

class ComprehensiveLLMStrategy(ILLMStrategy):
    """Comprehensive LLM analysis strategy with detailed prompts"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def get_strategy_name(self) -> str:
        return "comprehensive"
    
    def get_model_for_task(self, task_type: LLMTaskType) -> str:
        """Get appropriate model for each task type"""
        model_mapping = {
            LLMTaskType.FUNDAMENTAL_ANALYSIS: self.config.ollama.models.get(
                'fundamental_analysis', 'phi4-reasoning:plus'
            ),
            LLMTaskType.TECHNICAL_ANALYSIS: self.config.ollama.models.get(
                'technical_analysis', 'qwen2.5:32b-instruct-q4_K_M'
            ),
            LLMTaskType.SYNTHESIS: self.config.ollama.models.get(
                'synthesizer', 'llama3.1:8b-instruct-q8_0'
            ),
            LLMTaskType.QUARTERLY_SUMMARY: self.config.ollama.models.get(
                'fundamental_analysis', 'phi4-reasoning:plus'
            ),
            LLMTaskType.RISK_ASSESSMENT: self.config.ollama.models.get(
                'fundamental_analysis', 'phi4-reasoning:plus'
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
                'task_type': task_type.value,
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
        
        return LLMRequest(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            priority=LLMPriority.NORMAL.value,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            metadata={
                'task_type': task_type.value,
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
        
        return LLMRequest(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            priority=LLMPriority.HIGH.value,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            metadata={
                'task_type': task_type.value,
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
        
        return LLMRequest(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            priority=LLMPriority.NORMAL.value,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            metadata={
                'task_type': task_type.value,
                'symbol': symbol,
                'strategy': self.get_strategy_name()
            }
        )
    
    def _prepare_risk_request(self, symbol: str, data: Dict[str, Any], 
                            model: str, request_id: str) -> LLMRequest:
        """Prepare risk assessment request"""
        all_data = data
        
        system_prompt = """You are a risk management specialist analyzing investment risks across fundamental, 
        technical, and market factors. Identify and quantify key risks that could impact investment performance."""
        
        prompt = f"""
        Assess investment risks for {symbol}:
        
        ALL AVAILABLE DATA:
        {json.dumps(all_data, indent=2)}
        
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
        
        return LLMRequest(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
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
        """Process LLM response into structured data"""
        try:
            # Parse JSON response
            if response.error:
                return {'error': response.error}
            
            # Clean the response content
            content = response.content.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            result = json.loads(content)
            
            # Add metadata
            result['processing_metadata'] = {
                'task_type': task_type.value,
                'model_used': response.model,
                'processing_time_ms': response.processing_time_ms,
                'tokens_used': response.tokens_used,
                'request_id': response.request_id,
                'timestamp': response.timestamp.isoformat() if response.timestamp else None
            }
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {
                'error': f'JSON parsing failed: {str(e)}',
                'raw_response': response.content[:500] + '...' if len(response.content) > 500 else response.content
            }
        except Exception as e:
            self.logger.error(f"Error processing LLM response: {e}")
            return {'error': f'Response processing failed: {str(e)}'}

# ============================================================================
# Quick Analysis Strategy
# ============================================================================

class QuickLLMStrategy(ILLMStrategy):
    """Quick analysis strategy with simplified prompts"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
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

class StandardLLMCacheStrategy(ILLMCacheStrategy):
    """Standard caching strategy for LLM responses"""
    
    def __init__(self, config):
        self.config = config
    
    def get_cache_key(self, request: LLMRequest) -> str:
        """Generate deterministic cache key"""
        # Create hash from prompt, model, and key parameters
        key_data = {
            'model': request.model,
            'prompt': request.prompt,
            'system_prompt': request.system_prompt,
            'temperature': request.temperature,
            'top_p': request.top_p
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def should_cache(self, request: LLMRequest, response: LLMResponse) -> bool:
        """Determine if response should be cached"""
        # Don't cache errors
        if response.error:
            return False
        
        # Don't cache very short responses (likely errors)
        if len(response.content) < 50:
            return False
        
        # Cache based on task type
        task_type = request.metadata.get('task_type') if request.metadata else None
        return self.is_cacheable_task(LLMTaskType(task_type)) if task_type else True
    
    def get_ttl(self, task_type: LLMTaskType) -> int:
        """Get cache TTL in seconds based on task type"""
        ttl_mapping = {
            LLMTaskType.FUNDAMENTAL_ANALYSIS: 86400 * 7,  # 1 week
            LLMTaskType.TECHNICAL_ANALYSIS: 86400,  # 1 day
            LLMTaskType.SYNTHESIS: 86400 * 3,  # 3 days
            LLMTaskType.QUARTERLY_SUMMARY: 86400 * 30,  # 1 month
            LLMTaskType.RISK_ASSESSMENT: 86400 * 7  # 1 week
        }
        
        return ttl_mapping.get(task_type, 86400)  # Default 1 day
    
    def is_cacheable_task(self, task_type: LLMTaskType) -> bool:
        """Check if task type should be cached"""
        # Cache all analysis types by default
        return True

class AggressiveLLMCacheStrategy(ILLMCacheStrategy):
    """Aggressive caching strategy for maximum performance"""
    
    def __init__(self, config):
        self.config = config
    
    def get_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key focusing on symbol and task type"""
        # Less specific caching for better hit rates
        symbol = request.metadata.get('symbol', 'UNKNOWN') if request.metadata else 'UNKNOWN'
        task_type = request.metadata.get('task_type', 'unknown') if request.metadata else 'unknown'
        
        key_data = {
            'symbol': symbol,
            'task_type': task_type,
            'model': request.model,
            'prompt_hash': hashlib.md5(request.prompt.encode()).hexdigest()[:16]
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def should_cache(self, request: LLMRequest, response: LLMResponse) -> bool:
        """Cache almost everything for maximum performance"""
        return not response.error and len(response.content) > 10
    
    def get_ttl(self, task_type: LLMTaskType) -> int:
        """Longer TTLs for aggressive caching"""
        return 86400 * 14  # 2 weeks for everything
    
    def is_cacheable_task(self, task_type: LLMTaskType) -> bool:
        """Cache all task types"""
        return True