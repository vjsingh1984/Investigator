#!/usr/bin/env python3
"""
InvestiGator - Prompt Manager
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Prompt Manager for Jinja2 Templates
Handles loading and rendering of LLM prompt templates with proper JSON response formatting
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Environment = None
    FileSystemLoader = None
    Template = None

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages Jinja2 prompt templates for LLM requests"""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize prompt manager
        
        Args:
            templates_dir: Directory containing Jinja2 templates (defaults to prompts/)
        """
        if not JINJA2_AVAILABLE:
            logger.warning("Jinja2 not available. Install with: pip install jinja2")
            self.env = None
            return
            
        if templates_dir is None:
            # Default to prompts directory in project root
            project_root = Path(__file__).parent.parent
            templates_dir = project_root / "prompts"
        
        self.templates_dir = Path(templates_dir)
        
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            self.env = None
            return
            
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        logger.info(f"Prompt manager initialized with templates from: {self.templates_dir}")
    
    def render_sec_fundamental_prompt(self, **kwargs) -> str:
        """
        Render SEC fundamental analysis prompt
        
        Args:
            ticker: Stock ticker symbol
            period_key: Period identifier in standardized format (e.g., "2024-Q3", "2024-FY")
            form_type: SEC form type (e.g., "10-Q", "10-K")
            filing_date: Filing date string (when SEC received the filing)
            fiscal_year: Fiscal year (calendar year)
            fiscal_period: Fiscal period (e.g., "Q1", "Q2", "Q3", "Q4", "FY")
            data_section: Formatted financial data section
            
        Returns:
            Rendered prompt string with calendar year context and filing date details
        """
        # Ensure period_key is in standardized format
        if 'period_key' in kwargs and 'fiscal_year' in kwargs and 'fiscal_period' in kwargs:
            period_key = kwargs['period_key']
            fiscal_year = kwargs['fiscal_year']
            fiscal_period = kwargs['fiscal_period']
            
            # Standardize period if not already in YYYY-QX format
            if not period_key.startswith(str(fiscal_year)):
                # Import locally to avoid circular imports
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                try:
                    from sec_fundamental import standardize_period
                    kwargs['period_key'] = standardize_period(fiscal_year, fiscal_period)
                except ImportError:
                    # Fallback if import fails
                    if fiscal_period == 'FY':
                        kwargs['period_key'] = f"{fiscal_year}-FY"
                    elif fiscal_period.startswith('Q'):
                        kwargs['period_key'] = f"{fiscal_year}-{fiscal_period}"
                    else:
                        kwargs['period_key'] = f"{fiscal_year}-{fiscal_period}"
        
        return self._render_template("sec_fundamental_analysis.j2", **kwargs)
    
    def render_technical_analysis_prompt(self, **kwargs) -> str:
        """
        Render technical analysis prompt
        
        Args:
            symbol: Stock symbol
            analysis_date: Analysis date string
            data_points: Number of data points
            current_price: Current stock price
            csv_data: CSV formatted price/volume data
            indicators_summary: Technical indicators summary
            stock_info: Stock information dict
            
        Returns:
            Rendered prompt string with JSON response format
        """
        return self._render_template("technical_analysis.j2", **kwargs)
    
    def render_investment_synthesis_prompt(self, **kwargs) -> str:
        """
        Render investment synthesis prompt
        
        Args:
            symbol: Stock symbol
            analysis_date: Analysis date string
            current_price: Current stock price
            sector_context: Sector classification
            market_environment: Market environment description
            fundamental_data: Fundamental analysis data
            technical_data: Technical analysis data
            latest_market_data: Latest market data
            performance_data: Historical performance data
            
        Returns:
            Rendered prompt string with JSON response format
        """
        return self._render_template("investment_synthesis.j2", **kwargs)
    
    def _render_template(self, template_name: str, **kwargs) -> str:
        """
        Render a Jinja2 template with given parameters
        
        Args:
            template_name: Name of template file
            **kwargs: Template variables
            
        Returns:
            Rendered template string
        """
        if not self.env:
            # Fallback to basic string formatting if Jinja2 not available
            logger.warning(f"Jinja2 not available, using fallback for {template_name}")
            return self._fallback_render(template_name, **kwargs)
        
        try:
            template = self.env.get_template(template_name)
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            return self._fallback_render(template_name, **kwargs)
    
    def _fallback_render(self, template_name: str, **kwargs) -> str:
        """
        Fallback rendering when Jinja2 is not available
        
        Args:
            template_name: Template name (used for type detection)
            **kwargs: Template variables
            
        Returns:
            Basic prompt string
        """
        if "fundamental" in template_name:
            return self._create_basic_fundamental_prompt(**kwargs)
        elif "technical" in template_name:
            return self._create_basic_technical_prompt(**kwargs)
        elif "synthesis" in template_name:
            return self._create_basic_synthesis_prompt(**kwargs)
        else:
            return f"Analysis for {kwargs.get('symbol', 'UNKNOWN')} - Template: {template_name}"
    
    def _create_basic_fundamental_prompt(self, **kwargs) -> str:
        """Create basic fundamental analysis prompt without Jinja2"""
        ticker = kwargs.get('ticker', 'UNKNOWN')
        period_key = kwargs.get('period_key', 'UNKNOWN')
        data_section = kwargs.get('data_section', 'No data available')
        
        return f"""Analyze the fundamental data for {ticker} for period {period_key}.

{data_section}

Please provide your analysis in JSON format with the following structure:
{{
  "financial_health_score": {{"score": 0.0, "explanation": "..."}},
  "business_quality_score": {{"score": 0.0, "explanation": "..."}},
  "growth_prospects_score": {{"score": 0.0, "explanation": "..."}},
  "recommendation": {{"rating": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW"}}
}}

Respond with valid JSON only."""
    
    def _create_basic_technical_prompt(self, **kwargs) -> str:
        """Create basic technical analysis prompt without Jinja2"""
        symbol = kwargs.get('symbol', 'UNKNOWN')
        current_price = kwargs.get('current_price', 0.0)
        
        return f"""Analyze the technical data for {symbol} at current price ${current_price}.

Please provide your analysis in JSON format with the following structure:
{{
  "technical_score": {{"score": 0.0, "explanation": "..."}},
  "trend_analysis": "...",
  "recommendation": {{"rating": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW"}}
}}

Respond with valid JSON only."""
    
    def _create_basic_synthesis_prompt(self, **kwargs) -> str:
        """Create basic synthesis prompt without Jinja2"""
        symbol = kwargs.get('symbol', 'UNKNOWN')
        
        return f"""Synthesize the fundamental and technical analysis for {symbol}.

Please provide your synthesis in JSON format with the following structure:
{{
  "overall_score": 0.0,
  "investment_recommendation": {{"recommendation": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW"}},
  "investment_thesis": "..."
}}

Respond with valid JSON only."""
    
    def validate_json_response(self, response: str) -> Dict[str, Any]:
        """
        Validate and parse JSON response from LLM with robust error handling
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed JSON dict or error dict with fallback structure
        """
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned_response = response.strip()
            
            # Extract JSON from markdown code block
            if '```json' in cleaned_response and '```' in cleaned_response:
                start_idx = cleaned_response.find('```json') + 7
                end_idx = cleaned_response.find('```', start_idx)
                if end_idx > start_idx:
                    cleaned_response = cleaned_response[start_idx:end_idx].strip()
            elif cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
                if '```' in cleaned_response:
                    cleaned_response = cleaned_response[:cleaned_response.find('```')]
            elif cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Additional cleaning for common LLM issues
            cleaned_response = self._fix_common_json_issues(cleaned_response)
            
            # Parse JSON
            parsed_json = json.loads(cleaned_response)
            
            logger.debug("Successfully parsed JSON response")
            return parsed_json
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            logger.debug(f"Problematic JSON around error position: {response[max(0, e.pos-50):e.pos+50]}")
            
            # Try to salvage partial JSON by finding complete structure
            salvaged_json = self._attempt_json_salvage(cleaned_response if 'cleaned_response' in locals() else response)
            if salvaged_json:
                logger.warning("Successfully salvaged partial JSON response")
                return salvaged_json
            
            return {
                "error": "Invalid JSON response",
                "raw_response": response[:1000] + "..." if len(response) > 1000 else response,
                "parse_error": str(e),
                "error_position": getattr(e, 'pos', 'unknown'),
                # Provide fallback structure for fundamental analysis
                "analysis_summary": {
                    "error": "JSON parsing failed",
                    "overall_assessment": "Unable to parse LLM response due to malformed JSON"
                },
                "financial_health_score": {"score": 0.0, "explanation": "Analysis failed due to parsing error"},
                "recommendation": {"rating": "HOLD", "confidence": "LOW", "reason": "Unable to analyze due to JSON error"}
            }
        except Exception as e:
            logger.error(f"Error validating JSON response: {e}")
            return {
                "error": "Response validation failed",
                "raw_response": response[:1000] + "..." if len(response) > 1000 else response,
                "validation_error": str(e),
                # Provide fallback structure
                "analysis_summary": {
                    "error": "Validation failed",
                    "overall_assessment": "Unable to validate LLM response"
                },
                "financial_health_score": {"score": 0.0, "explanation": "Analysis failed due to validation error"},
                "recommendation": {"rating": "HOLD", "confidence": "LOW", "reason": "Unable to analyze due to validation error"}
            }
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues from LLM responses"""
        import re
        
        # Fix common escape issues
        json_str = json_str.replace('\\"', '"')  # Fix over-escaped quotes
        json_str = json_str.replace('\\n', '\\\\n')  # Fix newlines in strings
        json_str = json_str.replace('\\t', '\\\\t')  # Fix tabs in strings
        
        # Fix unterminated strings by finding the last complete quote pair
        # This is a simple heuristic - look for pattern of quote followed by comma/brace
        lines = json_str.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Check for unterminated strings (quotes without proper closing)
            if '"' in line and line.count('"') % 2 != 0:
                # Try to fix by adding closing quote before comma/brace
                if line.rstrip().endswith(',') or line.rstrip().endswith('}') or line.rstrip().endswith(']'):
                    # Find the last quote and ensure it's properly closed
                    last_quote_pos = line.rfind('"')
                    if last_quote_pos > 0:
                        before_quote = line[:last_quote_pos]
                        after_quote = line[last_quote_pos+1:]
                        # Add closing quote if missing
                        if not after_quote.startswith('"'):
                            line = before_quote + '""' + after_quote
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _attempt_json_salvage(self, json_str: str) -> Optional[Dict[str, Any]]:
        """Attempt to salvage a partial JSON response by finding complete structures"""
        try:
            import re
            # First, try to extract partial JSON data from the response
            # Remove comments like // ... (other sections follow)
            json_str = re.sub(r'//.*?\n', '', json_str)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            # Try to find the last complete JSON object
            brace_count = 0
            last_complete_pos = 0
            
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_complete_pos = i + 1
            
            if last_complete_pos > 0:
                # Try parsing the complete portion
                complete_json = json_str[:last_complete_pos]
                parsed = json.loads(complete_json)
                
                # If we got partial data, fill in missing sections with defaults
                if 'executive_summary' in parsed and 'composite_scores' in parsed:
                    # This is a synthesis response, fill in missing sections
                    if 'fundamental_assessment' not in parsed:
                        parsed['fundamental_assessment'] = {
                            'financial_health': {'score': parsed.get('composite_scores', {}).get('fundamental_score', 5.0)},
                            'business_quality': {'score': 7.0},
                            'growth_prospects': {'score': 6.0},
                            'valuation': {'fair_value_estimate': 0.0}
                        }
                    if 'technical_assessment' not in parsed:
                        parsed['technical_assessment'] = {
                            'trend_analysis': {'primary_trend': 'NEUTRAL'},
                            'entry_timing': {'optimal_entry_zone': {'lower_bound': 0.0, 'upper_bound': 0.0}},
                            'risk_management': {'stop_loss_level': 0.0}
                        }
                    if 'investment_recommendation' not in parsed:
                        parsed['investment_recommendation'] = {
                            'recommendation': 'HOLD',
                            'confidence_level': 'MEDIUM',
                            'time_horizon': 'MEDIUM_TERM',
                            'position_sizing': {'recommended_weight': 0.03}
                        }
                    if 'risk_analysis' not in parsed:
                        parsed['risk_analysis'] = {
                            'overall_risk_score': 5.0,
                            'fundamental_risks': [],
                            'technical_risks': [],
                            'market_risks': []
                        }
                    
                return parsed
            
            return None
        except Exception as e:
            logger.debug(f"JSON salvage attempt failed: {e}")
            return None

# Global instance for easy access
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """Get global prompt manager instance"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager