#!/usr/bin/env python3
"""
InvestiGator - Synthesis Helper Functions
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Helper functions for synthesis prompt preparation
Enhanced to include full content without truncation
"""

import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def format_fundamental_data_for_synthesis(fundamental_responses: Dict[str, Dict]) -> str:
    """Format fundamental analysis responses for synthesis prompt - includes full content"""
    try:
        if not fundamental_responses:
            return "No fundamental analysis data available."
        
        formatted_sections = []
        
        # Sort by period to ensure chronological order
        sorted_items = sorted(fundamental_responses.items(), key=lambda x: x[0])
        
        for period_key, response_data in sorted_items:
            content = response_data.get('content', '')
            form_type = response_data.get('form_type', 'Unknown')
            period = response_data.get('period', 'Unknown')
            
            # Try to parse as JSON first, fallback to text
            try:
                if isinstance(content, str):
                    parsed_content = json.loads(content)
                else:
                    parsed_content = content
                
                # Extract ALL data from structured response
                section = f"\n{'='*60}\n{period_key} ({form_type}) - Period: {period}\n{'='*60}"
                
                # Include complete JSON structure for comprehensive analysis
                section += "\n\nFULL ANALYSIS DATA:\n"
                section += json.dumps(parsed_content, indent=2)
                
                # Also extract key highlights for quick reference
                section += "\n\nKEY HIGHLIGHTS:\n"
                
                if 'financial_health_score' in parsed_content:
                    health_data = parsed_content['financial_health_score']
                    section += f"- Financial Health Score: {health_data.get('score', 'N/A')}/10\n"
                    if 'rationale' in health_data:
                        section += f"  Rationale: {health_data.get('rationale', 'N/A')}\n"
                
                if 'business_quality_score' in parsed_content:
                    quality_data = parsed_content['business_quality_score']
                    section += f"- Business Quality Score: {quality_data.get('score', 'N/A')}/10\n"
                    if 'rationale' in quality_data:
                        section += f"  Rationale: {quality_data.get('rationale', 'N/A')}\n"
                
                if 'growth_prospects_score' in parsed_content:
                    growth_data = parsed_content['growth_prospects_score']
                    section += f"- Growth Prospects Score: {growth_data.get('score', 'N/A')}/10\n"
                    if 'rationale' in growth_data:
                        section += f"  Rationale: {growth_data.get('rationale', 'N/A')}\n"
                
                # Include all insights
                if 'key_insights' in parsed_content:
                    insights = parsed_content.get('key_insights', [])
                    section += f"\n- Key Insights ({len(insights)} total):\n"
                    for i, insight in enumerate(insights, 1):
                        section += f"  {i}. {insight}\n"
                
                # Include all risks
                if 'key_risks' in parsed_content:
                    risks = parsed_content.get('key_risks', [])
                    section += f"\n- Key Risks ({len(risks)} total):\n"
                    for i, risk in enumerate(risks, 1):
                        section += f"  {i}. {risk}\n"
                
                formatted_sections.append(section)
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                # Fallback to text - include full content
                section = f"\n{'='*60}\n{period_key} ({form_type}) - Period: {period}\n{'='*60}\n"
                section += "FULL TEXT ANALYSIS:\n"
                section += content + "\n"  # Full content, no truncation
                logger.debug(f"Using text fallback for {period_key}: {str(e)}")
                formatted_sections.append(section)
        
        return "\n".join(formatted_sections)
        
    except Exception as e:
        logger.error(f"Error formatting fundamental data: {e}")
        return "Error formatting fundamental analysis data."

def format_technical_data_for_synthesis(technical_response: Dict[str, Any]) -> str:
    """Format technical analysis response for synthesis prompt - includes full content"""
    try:
        if not technical_response:
            return "No technical analysis data available."
        
        content = technical_response.get('content', '')
        
        # Try to parse as JSON first, fallback to text
        try:
            if isinstance(content, str):
                parsed_content = json.loads(content)
            else:
                parsed_content = content
            
            # Extract ALL data from structured response
            section = f"\n{'='*60}\nTECHNICAL ANALYSIS SUMMARY\n{'='*60}"
            
            # Include complete JSON structure for comprehensive analysis
            section += "\n\nFULL TECHNICAL ANALYSIS DATA:\n"
            section += json.dumps(parsed_content, indent=2)
            
            # Also extract key highlights for quick reference
            section += "\n\nKEY TECHNICAL HIGHLIGHTS:\n"
            
            if 'technical_score' in parsed_content:
                tech_score = parsed_content['technical_score']
                section += f"- Technical Score: {tech_score.get('score', 'N/A')}/10\n"
                if 'rationale' in tech_score:
                    section += f"  Rationale: {tech_score.get('rationale', 'N/A')}\n"
            
            if 'trend_analysis' in parsed_content:
                trend_data = parsed_content['trend_analysis']
                section += f"\n- Trend Analysis:\n"
                section += f"  Primary Trend: {trend_data.get('primary_trend', 'N/A')}\n"
                section += f"  Trend Strength: {trend_data.get('trend_strength', 'N/A')}\n"
                if 'momentum_signals' in trend_data:
                    section += f"  Momentum Signals: {trend_data.get('momentum_signals', 'N/A')}\n"
            
            if 'support_resistance' in parsed_content:
                sr_data = parsed_content['support_resistance']
                section += f"\n- Support/Resistance Levels:\n"
                section += f"  Immediate Support: ${sr_data.get('immediate_support', 'N/A')}\n"
                section += f"  Major Support: ${sr_data.get('major_support', 'N/A')}\n"
                section += f"  Immediate Resistance: ${sr_data.get('immediate_resistance', 'N/A')}\n"
                section += f"  Major Resistance: ${sr_data.get('major_resistance', 'N/A')}\n"
            
            if 'momentum_analysis' in parsed_content:
                momentum_data = parsed_content['momentum_analysis']
                section += f"\n- Momentum Indicators:\n"
                section += f"  RSI(14): {momentum_data.get('rsi_14', 'N/A')}\n"
                section += f"  RSI Assessment: {momentum_data.get('rsi_assessment', 'N/A')}\n"
                section += f"  MACD Signal: {momentum_data.get('macd_signal', 'N/A')}\n"
            
            if 'recommendation' in parsed_content:
                rec_data = parsed_content['recommendation']
                section += f"\n- Technical Recommendation:\n"
                section += f"  Rating: {rec_data.get('technical_rating', 'N/A')}\n"
                section += f"  Confidence: {rec_data.get('confidence', 'N/A')}\n"
                section += f"  Time Horizon: {rec_data.get('time_horizon', 'N/A')}\n"
                if 'entry_points' in rec_data:
                    section += f"  Entry Points: {rec_data.get('entry_points', 'N/A')}\n"
            
            # Include key patterns
            if 'chart_patterns' in parsed_content:
                patterns = parsed_content.get('chart_patterns', [])
                section += f"\n- Chart Patterns ({len(patterns)} identified):\n"
                for i, pattern in enumerate(patterns, 1):
                    section += f"  {i}. {pattern}\n"
            
            return section
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback to text - include full content
            section = f"\n{'='*60}\nTECHNICAL ANALYSIS SUMMARY\n{'='*60}\n"
            section += "FULL TEXT TECHNICAL ANALYSIS:\n"
            section += content  # Full content, no truncation
            logger.debug(f"Using text fallback for technical data: {str(e)}")
            return section
        
    except Exception as e:
        logger.error(f"Error formatting technical data: {e}")
        return "Error formatting technical analysis data."

def get_performance_data(symbol: str) -> str:
    """Get historical performance data with detailed metrics"""
    # This is a placeholder - in production, this would fetch actual performance data
    return f"""
HISTORICAL PERFORMANCE METRICS FOR {symbol}:
- 1 Week Return: Data to be fetched
- 1 Month Return: Data to be fetched
- 3 Month Return: Data to be fetched
- 6 Month Return: Data to be fetched
- YTD Return: Data to be fetched
- 1 Year Return: Data to be fetched
- 3 Year CAGR: Data to be fetched
- 5 Year CAGR: Data to be fetched
- Volatility (30-day): Data to be fetched
- Beta vs S&P 500: Data to be fetched
- Sharpe Ratio: Data to be fetched
- Maximum Drawdown: Data to be fetched
"""