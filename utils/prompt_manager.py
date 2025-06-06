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
from datetime import datetime

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
        # Initialize response processor for JSON escaping
        try:
            from .llm_response_processor import get_llm_response_processor
            self.response_processor = get_llm_response_processor()
        except ImportError:
            logger.warning("Could not import llm_response_processor, JSON escaping may not work properly")
            self.response_processor = None
        
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
        
        # Add custom filters for context optimization
        def truncate_smart(text, length=800, end='...'):
            """Smart truncate that preserves JSON structure and key information"""
            if not text or len(text) <= length:
                return text
            
            # For JSON content, try to preserve structure
            text_str = str(text)
            if length >= len(text_str):
                return text_str
                
            # Try to cut at sentence boundary within limit
            truncated = text_str[:length]
            last_sentence = truncated.rfind('.')
            last_newline = truncated.rfind('\n')
            
            # Use the later of sentence or newline to preserve readability
            cut_point = max(last_sentence, last_newline)
            if cut_point > length * 0.8:  # Only if we don't lose more than 20%
                return text_str[:cut_point + 1] + end
            else:
                return text_str[:length] + end
        
        # Add the filter to Jinja2
        self.env.filters['truncate'] = truncate_smart
        
        logger.info(f"Prompt manager initialized with templates from: {self.templates_dir}")
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        Render any template by name
        
        Args:
            template_name: Name of the template file (e.g., 'quarterly_fundamental_analysis.j2')
            **kwargs: Variables to pass to the template
            
        Returns:
            Rendered template string
        """
        return self._render_template(template_name, **kwargs)
    
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
    
    def render_investment_synthesis_peer_prompt(self, **kwargs) -> str:
        """
        Render investment synthesis prompt with peer comparison
        
        Args:
            symbol: Stock symbol
            analysis_date: Analysis date string
            current_price: Current stock price
            sector: Sector name
            industry: Industry name
            fundamental_data: Fundamental analysis data
            technical_data: Technical analysis data
            latest_market_data: Latest market data
            peer_list: List of peer symbols
            company_ratios: Company financial ratios
            peer_statistics: Peer group statistics
            relative_position: Company's position relative to peers
            
        Returns:
            Rendered prompt string with JSON response format
        """
        return self._render_template("investment_synthesis_peer.j2", **kwargs)
    
    def render_investment_synthesis_comprehensive_prompt(self, **kwargs) -> str:
        """
        Render comprehensive investment synthesis prompt using all quarterly analyses + technical
        
        Args:
            symbol: Stock symbol
            analysis_date: Analysis date string
            current_price: Current stock price
            comprehensive_analysis: Comprehensive fundamental analysis (empty for quarterly mode)
            quarterly_analyses: List of quarterly analysis results
            quarterly_count: Number of quarterly analyses
            financial_trends: Financial trends summary
            technical_analysis: Technical analysis data
            technical_signals: Technical signals dictionary
            market_data: Current market data
            
        Returns:
            Rendered prompt string with JSON response format
        """
        return self._render_template("investment_synthesis_comprehensive.j2", **kwargs)
    
    def render_risk_assessment_prompt(self, **kwargs) -> str:
        """
        Render risk assessment prompt
        
        Args:
            symbol: Stock symbol
            analysis_date: Analysis date string
            current_price: Current stock price
            fundamental_analysis: Fundamental analysis data (JSON string)
            technical_analysis: Technical analysis data (JSON string)
            market_data: Market data (JSON string)
            historical_data: Historical performance data (JSON string)
            
        Returns:
            Rendered prompt string with JSON response format
        """
        return self._render_template("risk_assessment.j2", **kwargs)
    
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
        elif "synthesis_peer" in template_name:
            return self._create_basic_synthesis_peer_prompt(**kwargs)
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
    
    def _create_basic_synthesis_peer_prompt(self, **kwargs) -> str:
        """Create basic synthesis prompt with peer comparison without Jinja2"""
        symbol = kwargs.get('symbol', 'UNKNOWN')
        sector = kwargs.get('sector', 'N/A')
        industry = kwargs.get('industry', 'N/A')
        
        return f"""Synthesize the fundamental and technical analysis for {symbol} with peer comparison.
Sector: {sector}, Industry: {industry}

Please provide your synthesis in JSON format with the following structure:
{{
  "overall_score": 0.0,
  "peer_relative_score": 0.0,
  "price_targets": {{
    "short_term_3_months": {{"target": 0.0, "confidence": "HIGH|MEDIUM|LOW"}},
    "medium_term_12_months": {{"target": 0.0, "confidence": "HIGH|MEDIUM|LOW"}},
    "long_term_3_years": {{"target": 0.0, "confidence": "HIGH|MEDIUM|LOW"}}
  }},
  "investment_recommendation": {{"recommendation": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW"}},
  "investment_thesis": "..."
}}

Respond with valid JSON only."""
    
    def validate_json_response(self, response: Dict, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced JSON response validation with standardized response format
        
        Args:
            response: Raw LLM response string
            metadata: Optional metadata from LLM (tokens, timing, etc.)
            
        Returns:
            Standardized response dict with type, value, and metadata
        """
        # Initialize response metadata
        response_metadata = {
            'timestamp': datetime.now().isoformat(),
            'original_length': len(str(response)),
            'processing_method': 'unknown'
        }
        
        # Add LLM metadata if provided
        if metadata:
            response_metadata.update(metadata)
        
        try:
            # Extract content from dict response structure
            if isinstance(response, dict):
                content = response.get('content', '')
                
                # Check if content is already parsed JSON (dict)
                if isinstance(content, dict):
                    logger.debug("Content is already parsed JSON dict, using directly")
                    return {
                        'type': 'json',
                        'value': content,
                        'metadata': response_metadata
                    }
                else:
                    cleaned_response = str(content)
            else:
                cleaned_response = str(response)
            
            original_response = response
            
            # Check if response is empty
            if not cleaned_response:
                logger.error("Empty response received from LLM")
                logger.debug(f"Full response content: '{response}'")
                logger.debug(f"Response metadata: {response_metadata}")
                return {
                    'type': 'text',
                    'value': response,
                    'metadata': response_metadata,
                    'error': 'Empty response from LLM'
                }
            
            # Try to parse as JSON with preprocessing for common issues
            parsed_json = self._robust_json_parse(cleaned_response)
            
            if parsed_json is not None:
                result = {
                    'type': 'json',
                    'value': parsed_json,
                    'metadata': response_metadata
                }
                response_metadata['processing_method'] = 'json_parsed'
                return result
            else:
                # If JSON parsing failed, return error
                return {
                    'type': 'text',
                    'value': response,
                    'metadata': response_metadata,
                    'error': 'JSON parsing failed after preprocessing attempts'
                }
            
        except Exception as e:
            logger.error(f"Unexpected error in response validation: {e}")
            logger.error(f"FULL RESPONSE CONTENT FOR DEBUGGING:")
            logger.error(f"{'='*80}")
            logger.error(f"{response}")
            logger.error(f"{'='*80}")
            logger.error(f"Response metadata: {response_metadata}")
            
            return {
                'type': 'text',
                'value': response,
                'metadata': response_metadata,
                'error': f"Response validation failed: {str(e)}"
            }
    
    def _robust_json_parse(self, content: str) -> Optional[Dict]:
        """
        Attempt to parse JSON with various preprocessing strategies
        
        Args:
            content: Raw content to parse
            
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        if not content:
            return None
        
        # Strategy 1: Try direct parsing first
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")
        
        # Strategy 2: Clean and try common JSON fixes
        try:
            cleaned = self._preprocess_json_content(content)
            if cleaned != content:
                logger.debug("Attempting to parse preprocessed JSON")
                return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.debug(f"Preprocessed JSON parsing failed: {e}")
        
        # Strategy 3: Extract JSON from markdown or mixed content
        try:
            # Look for JSON in code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                return json.loads(json_content)
            
            # Look for JSON-like content (starts with { and ends with })
            brace_match = re.search(r'\{.*\}', content, re.DOTALL)
            if brace_match:
                json_content = brace_match.group(0)
                # Try to parse this extracted JSON
                return json.loads(json_content)
                
        except json.JSONDecodeError as e:
            logger.debug(f"Extracted JSON parsing failed: {e}")
        
        # Strategy 4: Try partial JSON extraction for common patterns
        try:
            partial_json = self._extract_partial_json_data(content)
            if partial_json:
                return partial_json
        except Exception as e:
            logger.debug(f"Partial JSON extraction failed: {e}")
        
        logger.error(f"All JSON parsing strategies failed for content (first 200 chars): {content[:200]}...")
        return None
    
    def _preprocess_json_content(self, content: str) -> str:
        """
        Preprocess content to fix common JSON formatting issues
        
        Args:
            content: Raw content to preprocess
            
        Returns:
            Preprocessed content
        """
        import re
        
        # Fix common escape sequence issues - simplified approach
        # Handle embedded quotes in JSON string values
        # Look for patterns like: "key": "value with "quotes" and fix them
        
        # First, try to fix the most common case: missing comma between fields
        content = re.sub(r'(\w+": *"[^"]*") +("[\w_]+":)', r'\1, \2', content)
        content = re.sub(r'(\w+": *\d+(?:\.\d+)?) +("[\w_]+":)', r'\1, \2', content)
        
        # Fix embedded quotes by replacing them with escaped quotes
        # Pattern for JSON string values containing unescaped quotes
        def escape_embedded_quotes(text):
            """Escape quotes that are embedded within JSON string values"""
            # Find JSON string patterns and fix embedded quotes
            pattern = r'"([^"]+)": *"([^"]*"[^"]*)"'
            
            def fix_match(match):
                key = match.group(1)
                value = match.group(2)
                # Remove the trailing quote and escape internal quotes
                fixed_value = value[:-1].replace('"', '\\"')
                return f'"{key}": "{fixed_value}"'
            
            return re.sub(pattern, fix_match, text)
        
        content = escape_embedded_quotes(content)
        
        # Fix unescaped newlines and tabs in JSON strings
        content = re.sub(r'(": "[^"]*?)\\n([^"]*?")', r'\1\\\\n\2', content)
        content = re.sub(r'(": "[^"]*?)\\t([^"]*?")', r'\1\\\\t\2', content)
        
        # Fix missing commas before closing braces/brackets (common issue)
        content = re.sub(r'(["\d\]\}])\s*\n\s*(["\{\[])', r'\1,\n\2', content)
        
        # Remove trailing commas before closing braces/brackets
        content = re.sub(r',(\s*[\]\}])', r'\1', content)
        
        # Strip any leading/trailing whitespace and non-JSON content
        content = content.strip()
        
        # If content doesn't start with { but contains JSON, try to extract it
        if not content.startswith('{') and '{' in content:
            start_idx = content.find('{')
            # Find the matching closing brace
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                content = content[start_idx:end_idx]
        
        return content
    
    def _extract_partial_json_data(self, content: str) -> Optional[Dict]:
        """
        Extract key data even if JSON is malformed
        
        Args:
            content: Content to extract from
            
        Returns:
            Dict with extracted data or None
        """
        import re
        
        extracted = {}
        
        # Try to extract key fields using regex patterns
        patterns = {
            'overall_score': r'"overall_score":\s*([\d.]+)',
            'recommendation': r'"recommendation":\s*"([^"]+)"',
            'confidence_level': r'"confidence_level":\s*"([^"]+)"',
            'investment_thesis': r'"investment_thesis":\s*"([^"]*)"'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1)
                if key == 'overall_score':
                    try:
                        extracted[key] = float(value)
                    except ValueError:
                        extracted[key] = 5.0  # Default
                else:
                    extracted[key] = value
        
        # Only return if we extracted meaningful data
        if len(extracted) >= 2:  # At least 2 fields extracted
            # Fill in missing required fields with defaults
            if 'overall_score' not in extracted:
                extracted['overall_score'] = 5.0
            if 'recommendation' not in extracted:
                extracted['recommendation'] = 'HOLD'
            if 'confidence_level' not in extracted:
                extracted['confidence_level'] = 'LOW'
            if 'investment_thesis' not in extracted:
                extracted['investment_thesis'] = 'Analysis incomplete due to parsing issues'
            
            return extracted
        
        return None
    

# Global instance for easy access
_prompt_manager = None
_enhanced_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """Get global prompt manager instance"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager

def get_enhanced_prompt_manager(config=None) -> 'EnhancedPromptManager':
    """Get enhanced prompt manager with context optimization"""
    global _enhanced_prompt_manager
    if _enhanced_prompt_manager is None:
        from .prompt_manager_enhanced import EnhancedPromptManager
        base_manager = get_prompt_manager()
        _enhanced_prompt_manager = EnhancedPromptManager(base_manager, config)
    return _enhanced_prompt_manager
