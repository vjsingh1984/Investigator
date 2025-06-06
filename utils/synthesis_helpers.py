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
        
        # Check if we have a comprehensive analysis with quarterly breakdowns
        if 'comprehensive' in fundamental_responses:
            comp_data = fundamental_responses['comprehensive']
            content = comp_data.get('content', comp_data)
            
            # Format comprehensive analysis
            section = f"\n{'='*60}\nCOMPREHENSIVE FUNDAMENTAL ANALYSIS\n{'='*60}\n"
            
            if isinstance(content, dict):
                # Extract quarterly analyses if present
                if 'quarterly_analyses' in content:
                    section += "\n\nQUARTERLY ANALYSES:\n"
                    for qa in content['quarterly_analyses']:
                        period = qa.get('quarterly_summary', {}).get('fiscal_period', 'Unknown')
                        section += f"\n--- {period} ---\n"
                        section += json.dumps(qa, indent=2)
                
                # Add trend analysis
                if 'trend_analysis' in content:
                    section += "\n\nMULTI-QUARTER TRENDS:\n"
                    section += json.dumps(content['trend_analysis'], indent=2)
                
                # Add overall scores and insights
                section += "\n\nOVERALL FUNDAMENTAL ASSESSMENT:\n"
                section += f"Financial Health Score: {content.get('financial_health_score', 'N/A')}/10\n"
                section += f"Business Quality Score: {content.get('business_quality_score', 'N/A')}/10\n"
                section += f"Growth Prospects Score: {content.get('growth_prospects_score', 'N/A')}/10\n"
                section += f"Overall Score: {content.get('overall_score', 'N/A')}/10\n"
                
                if 'key_insights' in content:
                    section += "\nKey Insights:\n"
                    for insight in content['key_insights']:
                        section += f"• {insight}\n"
                
                if 'investment_thesis' in content:
                    section += f"\nInvestment Thesis: {content['investment_thesis']}\n"
            else:
                section += str(content)
            
            formatted_sections.append(section)
        else:
            # Fallback to legacy format
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
                # Handle both flat (new) and nested (legacy) structures
                if isinstance(tech_score, (int, float)):
                    section += f"- Technical Score: {tech_score}/10\n"
                elif isinstance(tech_score, dict):
                    section += f"- Technical Score: {tech_score.get('score')}/10\n"
                    if 'rationale' in tech_score:
                        section += f"  Rationale: {tech_score.get('rationale')}\n"
            
            # Handle both nested (legacy) and flat (new) trend data structures
            if 'trend_analysis' in parsed_content:
                trend_data = parsed_content['trend_analysis']
                section += f"\n- Trend Analysis:\n"
                section += f"  Primary Trend: {trend_data.get('primary_trend')}\n"
                section += f"  Trend Strength: {trend_data.get('trend_strength')}\n"
                if 'momentum_signals' in trend_data:
                    section += f"  Momentum Signals: {trend_data.get('momentum_signals')}\n"
            elif 'trend_direction' in parsed_content:
                # New flat structure
                section += f"\n- Trend Analysis:\n"
                section += f"  Trend Direction: {parsed_content.get('trend_direction')}\n"
                section += f"  Trend Strength: {parsed_content.get('trend_strength')}\n"
                if 'momentum_signals' in parsed_content:
                    signals = parsed_content.get('momentum_signals', [])
                    section += f"  Momentum Signals: {', '.join(signals) if isinstance(signals, list) else signals}\n"
            
            # Handle both nested (legacy) and flat (new) support/resistance structures
            if 'support_resistance' in parsed_content:
                sr_data = parsed_content['support_resistance']
                section += f"\n- Support/Resistance Levels:\n"
                section += f"  Immediate Support: ${sr_data.get('immediate_support')}\n"
                section += f"  Major Support: ${sr_data.get('major_support')}\n"
                section += f"  Immediate Resistance: ${sr_data.get('immediate_resistance')}\n"
                section += f"  Major Resistance: ${sr_data.get('major_resistance')}\n"
            elif 'support_levels' in parsed_content or 'resistance_levels' in parsed_content:
                # New flat structure
                section += f"\n- Support/Resistance Levels:\n"
                support_levels = parsed_content.get('support_levels', [])
                resistance_levels = parsed_content.get('resistance_levels', [])
                
                if support_levels and isinstance(support_levels, list) and len(support_levels) > 0:
                    section += f"  Primary Support: ${support_levels[0]}\n"
                    if len(support_levels) > 1:
                        section += f"  Secondary Support: ${support_levels[1]}\n"
                
                if resistance_levels and isinstance(resistance_levels, list) and len(resistance_levels) > 0:
                    section += f"  Primary Resistance: ${resistance_levels[0]}\n"
                    if len(resistance_levels) > 1:
                        section += f"  Secondary Resistance: ${resistance_levels[1]}\n"
            
            if 'momentum_analysis' in parsed_content:
                momentum_data = parsed_content['momentum_analysis']
                section += f"\n- Momentum Indicators:\n"
                section += f"  RSI(14): {momentum_data.get('rsi_14')}\n"
                section += f"  RSI Assessment: {momentum_data.get('rsi_assessment')}\n"
                section += f"  MACD Signal: {momentum_data.get('macd_signal')}\n"
            
            # Handle both nested (legacy) and flat (new) recommendation structures  
            if 'recommendation' in parsed_content:
                rec_data = parsed_content['recommendation']
                if isinstance(rec_data, dict):
                    section += f"\n- Technical Recommendation:\n"
                    section += f"  Rating: {rec_data.get('technical_rating')}\n"
                    section += f"  Confidence: {rec_data.get('confidence')}\n"
                    section += f"  Time Horizon: {rec_data.get('time_horizon')}\n"
                    if 'entry_points' in rec_data:
                        section += f"  Entry Points: {rec_data.get('entry_points')}\n"
                else:
                    # Flat structure
                    section += f"\n- Technical Recommendation: {rec_data}\n"
                    if 'time_horizon' in parsed_content:
                        section += f"  Time Horizon: {parsed_content.get('time_horizon')}\n"
            
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