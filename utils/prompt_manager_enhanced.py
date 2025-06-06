#!/usr/bin/env python3
"""
InvestiGator - Enhanced Prompt Manager with Context-Aware Optimization
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Enhanced Prompt Manager that intelligently manages prompt content
based on model context limits and prioritizes high-value information
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ModelContextInfo:
    """Model context information"""
    model_name: str
    context_size: int  # in tokens
    avg_chars_per_token: float = 4.0  # Conservative estimate
    reserved_output_tokens: int = 4096  # Reserve for output
    
    @property
    def available_context_chars(self) -> int:
        """Calculate available characters for input"""
        available_tokens = self.context_size - self.reserved_output_tokens
        return int(available_tokens * self.avg_chars_per_token)


class EnhancedPromptManager:
    """Enhanced prompt manager with context-aware content selection"""
    
    # Default model context sizes (conservative estimates)
    DEFAULT_MODEL_CONTEXTS = {
        'llama3.1:8b': ModelContextInfo('llama3.1:8b', 8192),
        'llama3.1:8b-instruct-q8_0': ModelContextInfo('llama3.1:8b-instruct-q8_0', 8192),
        'llama3.1:70b': ModelContextInfo('llama3.1:70b', 32768),
        'llama3.1:70b-instruct-q8_0': ModelContextInfo('llama3.1:70b-instruct-q8_0', 32768),
        'mistral:7b': ModelContextInfo('mistral:7b', 8192),
        'mixtral:8x7b': ModelContextInfo('mixtral:8x7b', 32768),
        'qwen2.5:32b': ModelContextInfo('qwen2.5:32b', 32768),
        'qwen2.5:72b': ModelContextInfo('qwen2.5:72b', 32768),
    }
    
    def __init__(self, base_prompt_manager, config=None):
        """
        Initialize enhanced prompt manager
        
        Args:
            base_prompt_manager: Base PromptManager instance
            config: Configuration object with model info
        """
        self.base_manager = base_prompt_manager
        self.config = config
        self.model_contexts = self.DEFAULT_MODEL_CONTEXTS.copy()
        
        # Update with config model info if available
        if config and hasattr(config, 'ollama') and hasattr(config.ollama, 'model_info_cache'):
            for model, info in config.ollama.model_info_cache.items():
                if 'num_ctx' in info:
                    self.model_contexts[model] = ModelContextInfo(
                        model_name=model,
                        context_size=info['num_ctx'],
                        reserved_output_tokens=info.get('num_predict', 4096)
                    )
    
    def get_model_context(self, model_name: str) -> ModelContextInfo:
        """Get context info for a model"""
        # Try exact match first
        if model_name in self.model_contexts:
            return self.model_contexts[model_name]
        
        # Try base model name (without quantization)
        base_model = model_name.split(':')[0]
        for known_model, context_info in self.model_contexts.items():
            if known_model.startswith(base_model):
                return context_info
        
        # Default to conservative 8k context
        logger.warning(f"Unknown model {model_name}, using conservative 8k context")
        return ModelContextInfo(model_name, 8192)
    
    def render_technical_analysis_prompt_optimized(self, 
                                                  symbol: str,
                                                  model_name: str,
                                                  csv_data: str,
                                                  indicators_summary: str,
                                                  **kwargs) -> str:
        """
        Render technical analysis prompt optimized for model context
        
        Prioritizes:
        1. Most recent price data
        2. Key technical indicators
        3. Support/resistance levels
        4. Volume patterns
        """
        context_info = self.get_model_context(model_name)
        available_chars = context_info.available_context_chars
        
        logger.info(f"Optimizing TA prompt for {model_name} with {available_chars:,} chars available")
        
        # Calculate space for each section
        base_prompt_size = 2000  # Instructions and structure
        indicators_size = min(len(indicators_summary), available_chars // 4)
        remaining_for_data = available_chars - base_prompt_size - indicators_size
        
        # Optimize CSV data
        optimized_csv = self._optimize_csv_data(csv_data, remaining_for_data, symbol)
        
        # Use base manager to render with optimized data
        return self.base_manager.render_technical_analysis_prompt(
            symbol=symbol,
            csv_data=optimized_csv,
            indicators_summary=indicators_summary[:indicators_size],
            **kwargs
        )
    
    def render_synthesis_prompt_optimized(self,
                                        model_name: str,
                                        symbol: str,
                                        fundamental_data: str,
                                        technical_data: str,
                                        **kwargs) -> str:
        """
        Render synthesis prompt optimized for model context
        
        Prioritizes:
        1. Latest quarterly results
        2. Key financial metrics and trends
        3. Technical signals and price levels
        4. Risk factors
        """
        context_info = self.get_model_context(model_name)
        available_chars = context_info.available_context_chars
        
        logger.info(f"Optimizing synthesis prompt for {model_name} with {available_chars:,} chars available")
        
        # Calculate space allocation (60% fundamental, 30% technical, 10% other)
        base_prompt_size = 3000  # Instructions and structure
        fundamental_size = int((available_chars - base_prompt_size) * 0.6)
        technical_size = int((available_chars - base_prompt_size) * 0.3)
        
        # Optimize fundamental data
        optimized_fundamental = self._optimize_fundamental_data(
            fundamental_data, fundamental_size, symbol
        )
        
        # Optimize technical data
        optimized_technical = self._optimize_technical_summary(
            technical_data, technical_size, symbol
        )
        
        # Use base manager to render with optimized data
        return self.base_manager.render_investment_synthesis_prompt(
            symbol=symbol,
            fundamental_data=optimized_fundamental,
            technical_data=optimized_technical,
            **kwargs
        )
    
    def render_comprehensive_fundamental_prompt_optimized(self,
                                                        model_name: str,
                                                        symbol: str,
                                                        quarterly_analyses: List[Dict],
                                                        aggregated_data: Dict,
                                                        **kwargs) -> str:
        """
        Render comprehensive fundamental prompt optimized for model context
        
        Prioritizes:
        1. Multi-quarter trends
        2. Key financial ratios evolution
        3. Revenue and earnings growth
        4. Balance sheet strength
        """
        context_info = self.get_model_context(model_name)
        available_chars = context_info.available_context_chars
        
        logger.info(f"Optimizing comprehensive fundamental prompt for {model_name} with {available_chars:,} chars available")
        
        # Optimize quarterly analyses
        optimized_analyses = self._optimize_quarterly_analyses(
            quarterly_analyses, available_chars // 2, symbol
        )
        
        # Optimize aggregated data
        optimized_aggregated = self._optimize_aggregated_data(
            aggregated_data, available_chars // 4, symbol
        )
        
        # Create optimized prompt
        prompt = f"""Comprehensive Fundamental Analysis for {symbol}

QUARTERLY TREND ANALYSIS:
{optimized_analyses}

AGGREGATED FINANCIAL DATA:
{optimized_aggregated}

Based on the quarterly trends and aggregated data, provide a comprehensive fundamental analysis with:
1. Multi-quarter trend assessment
2. Financial health evaluation
3. Growth trajectory analysis
4. Investment recommendation

Respond with properly formatted JSON only."""
        
        return prompt
    
    def _optimize_csv_data(self, csv_data: str, max_chars: int, symbol: str) -> str:
        """Optimize CSV data to fit context limits"""
        if len(csv_data) <= max_chars:
            return csv_data
        
        try:
            # Parse CSV data
            lines = csv_data.strip().split('\n')
            if len(lines) < 2:
                return csv_data
            
            header = lines[0]
            data_lines = lines[1:]
            
            # Prioritize recent data
            # Calculate how many lines we can fit
            chars_per_line = len(data_lines[0]) if data_lines else 100
            max_lines = max(10, (max_chars - len(header)) // chars_per_line)
            
            # Take most recent data (usually at the end)
            selected_lines = data_lines[-max_lines:] if len(data_lines) > max_lines else data_lines
            
            # Add summary of omitted data if truncated
            omitted_count = len(data_lines) - len(selected_lines)
            if omitted_count > 0:
                summary = f"# Note: Showing {len(selected_lines)} most recent days out of {len(data_lines)} total"
                return f"{summary}\n{header}\n" + '\n'.join(selected_lines)
            
            return header + '\n' + '\n'.join(selected_lines)
            
        except Exception as e:
            logger.error(f"Error optimizing CSV data: {e}")
            # Fallback to simple truncation
            return csv_data[:max_chars]
    
    def _optimize_fundamental_data(self, fundamental_data: str, max_chars: int, symbol: str) -> str:
        """Optimize fundamental data to fit context limits"""
        if len(fundamental_data) <= max_chars:
            return fundamental_data
        
        # Priority sections to extract
        priority_keywords = [
            "COMPREHENSIVE FUNDAMENTAL ANALYSIS",
            "QUARTERLY ANALYSES:",
            "MULTI-QUARTER TRENDS:",
            "OVERALL FUNDAMENTAL ASSESSMENT:",
            "Financial Health Score:",
            "Business Quality Score:",
            "Growth Prospects Score:",
            "Key Insights:",
            "Investment Thesis:",
            "Revenue",
            "Net Income",
            "EPS",
            "Cash Flow",
            "Debt",
            "Margin"
        ]
        
        # Extract priority sections
        extracted_sections = []
        remaining_chars = max_chars
        
        for keyword in priority_keywords:
            if keyword in fundamental_data and remaining_chars > 0:
                # Find section containing keyword
                start_idx = fundamental_data.find(keyword)
                # Find next section (marked by similar patterns)
                end_idx = len(fundamental_data)
                for next_keyword in priority_keywords:
                    if next_keyword != keyword:
                        next_idx = fundamental_data.find(next_keyword, start_idx + 1)
                        if next_idx > start_idx and next_idx < end_idx:
                            end_idx = next_idx
                
                section = fundamental_data[start_idx:end_idx].strip()
                if len(section) <= remaining_chars:
                    extracted_sections.append(section)
                    remaining_chars -= len(section)
                else:
                    # Truncate section to fit
                    extracted_sections.append(section[:remaining_chars])
                    break
        
        if extracted_sections:
            return "\n\n".join(extracted_sections)
        
        # Fallback to simple truncation
        return fundamental_data[:max_chars]
    
    def _optimize_technical_summary(self, technical_data: str, max_chars: int, symbol: str) -> str:
        """Optimize technical data summary to fit context limits"""
        if len(technical_data) <= max_chars:
            return technical_data
        
        # Priority sections for technical analysis
        priority_keywords = [
            "TECHNICAL ANALYSIS SUMMARY",
            "KEY TECHNICAL HIGHLIGHTS:",
            "Technical Score:",
            "Primary Trend:",
            "Trend Strength:",
            "Momentum Signals:",
            "Support/Resistance Levels:",
            "Immediate Support:",
            "Major Support:",
            "Immediate Resistance:",
            "Major Resistance:",
            "RSI",
            "MACD Signal:",
            "Technical Recommendation:",
            "Entry Points:",
            "Current Price:",
            "52W High:",
            "52W Low:"
        ]
        
        # Extract priority sections
        extracted_sections = []
        remaining_chars = max_chars
        
        for keyword in priority_keywords:
            if keyword in technical_data and remaining_chars > 0:
                # Extract line or section containing keyword
                lines = technical_data.split('\n')
                for line in lines:
                    if keyword in line and len(line) <= remaining_chars:
                        extracted_sections.append(line.strip())
                        remaining_chars -= len(line)
        
        if extracted_sections:
            return "\n".join(extracted_sections)
        
        # Fallback to simple truncation
        return technical_data[:max_chars]
    
    def _optimize_quarterly_analyses(self, quarterly_analyses: List[Dict], max_chars: int, symbol: str) -> str:
        """Optimize quarterly analyses for context limits"""
        if not quarterly_analyses:
            return "No quarterly analyses available"
        
        # Sort by period to get most recent
        sorted_analyses = sorted(quarterly_analyses, 
                               key=lambda x: x.get('quarterly_summary', {}).get('fiscal_period', ''),
                               reverse=True)
        
        result = []
        remaining_chars = max_chars
        
        for analysis in sorted_analyses:
            if remaining_chars <= 0:
                break
            
            summary = analysis.get('quarterly_summary', {})
            period = summary.get('fiscal_period', 'Unknown')
            
            # Extract key metrics
            section = f"\n{period}:\n"
            section += f"- Revenue: ${summary.get('revenue', 0):,.0f}\n"
            section += f"- Net Income: ${summary.get('net_income', 0):,.0f}\n"
            section += f"- EPS: ${summary.get('eps', 0):.2f}\n"
            section += f"- Operating Margin: {summary.get('operating_margin', 0):.1f}%\n"
            section += f"- Financial Health: {summary.get('financial_health_score', 0)}/10\n"
            
            if len(section) <= remaining_chars:
                result.append(section)
                remaining_chars -= len(section)
        
        return ''.join(result) if result else "Quarterly data exceeds context limit"
    
    def _optimize_aggregated_data(self, aggregated_data: Dict, max_chars: int, symbol: str) -> str:
        """Optimize aggregated financial data for context limits"""
        if not aggregated_data:
            return "No aggregated data available"
        
        # Priority metrics
        priority_metrics = [
            'revenue_growth_yoy',
            'earnings_growth_yoy',
            'operating_margin_trend',
            'debt_to_equity_current',
            'free_cash_flow_growth',
            'return_on_equity',
            'current_ratio',
            'quarters_analyzed'
        ]
        
        result = []
        for metric in priority_metrics:
            if metric in aggregated_data:
                result.append(f"{metric}: {aggregated_data[metric]}")
        
        output = '\n'.join(result)
        if len(output) > max_chars:
            return output[:max_chars]
        
        return output


def get_enhanced_prompt_manager(base_prompt_manager, config=None) -> EnhancedPromptManager:
    """Factory function to create enhanced prompt manager"""
    return EnhancedPromptManager(base_prompt_manager, config)