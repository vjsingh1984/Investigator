#!/usr/bin/env python3
"""
InvestiGator - Chart Generation Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Chart Generation Module for InvestiGator
Handles technical and fundamental analysis chart creation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Try to import talib for technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available - some technical indicators may be limited")

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Handles chart generation for technical and fundamental analysis"""
    
    def __init__(self, charts_dir: Path):
        """Initialize chart generator"""
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                logger.warning("Seaborn style not available, using default")
    
    def generate_technical_chart(self, symbol: str, price_data: pd.DataFrame) -> str:
        """
        Generate comprehensive technical analysis chart with Fibonacci levels
        
        Args:
            symbol: Stock symbol
            price_data: DataFrame with OHLCV and indicator data
            
        Returns:
            Path to generated chart
        """
        try:
            # Normalize column names to handle both uppercase and lowercase
            price_data = self._normalize_column_names(price_data)
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), 
                                               gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Main price chart with candlesticks/line
            ax1.plot(price_data.index, price_data['close'], 'b-', linewidth=1.5, label='Close')
            
            # Add moving averages if available
            if 'sma_20' in price_data.columns:
                ax1.plot(price_data.index, price_data['sma_20'], 'g-', alpha=0.7, label='SMA 20')
            if 'sma_50' in price_data.columns:
                ax1.plot(price_data.index, price_data['sma_50'], 'r-', alpha=0.7, label='SMA 50')
            if 'sma_200' in price_data.columns:
                ax1.plot(price_data.index, price_data['sma_200'], 'm-', alpha=0.7, label='SMA 200')
            
            # Add Bollinger Bands if available
            if all(col in price_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                ax1.fill_between(price_data.index, price_data['bb_upper'], price_data['bb_lower'],
                               alpha=0.1, color='gray', label='Bollinger Bands')
                ax1.plot(price_data.index, price_data['bb_middle'], 'k--', alpha=0.5, linewidth=0.8)
            
            # Add support/resistance levels
            self._add_support_resistance_levels(ax1, price_data)
            
            # Add Fibonacci retracement levels
            self._add_fibonacci_levels(ax1, price_data)
            
            # Volume subplot
            colors = ['g' if price_data['close'].iloc[i] >= price_data['open'].iloc[i] else 'r' 
                     for i in range(len(price_data))]
            ax2.bar(price_data.index, price_data['volume'], color=colors, alpha=0.7)
            
            # Add OBV line if available
            if 'obv' in price_data.columns:
                ax2_twin = ax2.twinx()
                ax2_twin.plot(price_data.index, price_data['obv'], 'b-', linewidth=1, label='OBV')
                ax2_twin.set_ylabel('OBV')
                ax2_twin.yaxis.label.set_color('blue')
                ax2_twin.tick_params(axis='y', colors='blue')
            
            # MACD subplot
            if all(col in price_data.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
                ax3.plot(price_data.index, price_data['macd'], 'b-', label='MACD')
                ax3.plot(price_data.index, price_data['macd_signal'], 'r-', label='Signal')
                ax3.bar(price_data.index, price_data['macd_histogram'], alpha=0.3, label='Histogram')
                ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax3.legend(loc='upper left', fontsize=8)
            
            # Formatting
            ax1.set_title(f'{symbol} Technical Analysis', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Price ($)')
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.3)
            
            ax3.set_ylabel('MACD')
            ax3.set_xlabel('Date')
            ax3.grid(True, alpha=0.3)
            
            # Format x-axis dates
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.charts_dir / f"{symbol}_technical_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"📊 Generated technical chart: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Error generating technical chart: {e}")
            return ""
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to lowercase for consistent access
        
        Args:
            df: DataFrame with potentially mixed case column names
            
        Returns:
            DataFrame with normalized lowercase column names
        """
        # Create a copy to avoid modifying the original
        normalized_df = df.copy()
        
        # Create mapping for common column variations
        column_mapping = {}
        for col in df.columns:
            lower_col = col.lower()
            # Handle common variations
            if lower_col in ['close', 'high', 'low', 'open', 'volume']:
                column_mapping[col] = lower_col
            elif lower_col.startswith('sma_'):
                column_mapping[col] = lower_col
            elif lower_col.startswith('ema_'):
                column_mapping[col] = lower_col
            elif lower_col.startswith('bb_'):
                column_mapping[col] = lower_col
            elif lower_col in ['macd', 'macd_signal', 'macd_histogram']:
                column_mapping[col] = lower_col
            elif lower_col == 'obv':
                column_mapping[col] = 'obv'
            else:
                column_mapping[col] = col
        
        # Rename columns
        normalized_df = normalized_df.rename(columns=column_mapping)
        
        return normalized_df
    
    def _add_support_resistance_levels(self, ax, price_data: pd.DataFrame):
        """Add support and resistance levels to chart"""
        try:
            # Calculate support/resistance from recent highs/lows
            recent_data = price_data.tail(60)  # Last 60 periods
            
            # Find local maxima/minima
            if 'high' in recent_data.columns and 'low' in recent_data.columns:
                highs = recent_data['high'].rolling(window=5, center=True).max() == recent_data['high']
                lows = recent_data['low'].rolling(window=5, center=True).min() == recent_data['low']
                
                # Plot resistance levels (red)
                resistance_levels = recent_data[highs]['high'].unique()[-3:]  # Top 3 levels
                for level in resistance_levels:
                    ax.axhline(y=level, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    ax.text(price_data.index[-1], level, f'R: ${level:.2f}', 
                           verticalalignment='bottom', horizontalalignment='right', 
                           color='red', fontsize=8)
                
                # Plot support levels (green)
                support_levels = recent_data[lows]['low'].unique()[:3]  # Bottom 3 levels
                for level in support_levels:
                    ax.axhline(y=level, color='green', linestyle='--', alpha=0.5, linewidth=1)
                    ax.text(price_data.index[-1], level, f'S: ${level:.2f}', 
                           verticalalignment='top', horizontalalignment='right', 
                           color='green', fontsize=8)
                
        except Exception as e:
            logger.warning(f"Could not add support/resistance levels: {e}")
    
    def _add_fibonacci_levels(self, ax, price_data: pd.DataFrame):
        """Add Fibonacci retracement levels to chart"""
        try:
            # Calculate Fibonacci levels from recent high/low
            if 'high' in price_data.columns and 'low' in price_data.columns:
                recent_high = price_data['high'].tail(120).max()
                recent_low = price_data['low'].tail(120).min()
                diff = recent_high - recent_low
            
            # Fibonacci ratios
            fib_levels = {
                '0.0%': recent_high,
                '23.6%': recent_high - diff * 0.236,
                '38.2%': recent_high - diff * 0.382,
                '50.0%': recent_high - diff * 0.500,
                '61.8%': recent_high - diff * 0.618,
                '78.6%': recent_high - diff * 0.786,
                '100.0%': recent_low
            }
            
            # Colors for Fibonacci levels
            colors = ['#FF0000', '#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#00CED1', '#0000FF']
            
            # Plot Fibonacci levels
            for i, (level, price) in enumerate(fib_levels.items()):
                ax.axhline(y=price, color=colors[i], linestyle=':', alpha=0.6, 
                          linewidth=1, label=f'Fib {level}')
            
        except Exception as e:
            logger.warning(f"Could not add Fibonacci levels: {e}")
    
    def generate_3d_fundamental_plot(self, recommendations: List[Dict]) -> str:
        """Generate 3D plot showing income/cashflow/balance sheet dimensions"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract data for plotting
            symbols = []
            income_scores = []
            cashflow_scores = []
            balance_scores = []
            overall_scores = []
            colors = []
            
            for rec in recommendations:
                symbol = rec.get('symbol', 'UNK')
                symbols.append(symbol)
                
                # Extract fundamental scores from recommendation
                income_score = rec.get('income_score', 5.0)
                cashflow_score = rec.get('cashflow_score', 5.0)
                balance_score = rec.get('balance_score', 5.0)
                overall_score = rec.get('overall_score', 5.0)
                
                income_scores.append(income_score)
                cashflow_scores.append(cashflow_score)
                balance_scores.append(balance_score)
                overall_scores.append(overall_score)
                
                # Color based on recommendation
                recommendation = rec.get('recommendation', 'HOLD')
                if 'BUY' in recommendation.upper():
                    colors.append('green')
                elif 'SELL' in recommendation.upper():
                    colors.append('red')
                else:
                    colors.append('blue')
            
            # Create 3D scatter plot with size based on overall score
            scatter = ax.scatter(income_scores, cashflow_scores, balance_scores,
                               s=[score * 20 for score in overall_scores],  # Size by overall score
                               c=colors, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add labels for each point
            for i, symbol in enumerate(symbols):
                ax.text(income_scores[i], cashflow_scores[i], balance_scores[i], 
                       f'  {symbol}', fontsize=8)
            
            # Set labels and title
            ax.set_xlabel('Income Statement Score')
            ax.set_ylabel('Cash Flow Score')
            ax.set_zlabel('Balance Sheet Score')
            ax.set_title('3D Fundamental Analysis\nSize = Overall Score, Color = Recommendation')
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                          markersize=10, label='BUY'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                          markersize=10, label='HOLD'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=10, label='SELL')
            ]
            ax.legend(handles=legend_elements, loc='upper left')
            
            # Save plot
            plot_path = self.charts_dir / "3d_fundamental_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"📊 Generated 3D fundamental plot: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error generating 3D fundamental plot: {e}")
            return ""
    
    def generate_2d_technical_fundamental_plot(self, recommendations: List[Dict]) -> str:
        """Generate 2D plot showing fundamental vs technical scores with data quality as size"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Extract data for plotting
            symbols = []
            fundamental_scores = []
            technical_scores = []
            data_quality_scores = []
            colors = []
            
            for rec in recommendations:
                symbol = rec.get('symbol', 'UNK')
                symbols.append(symbol)
                
                # Get scores
                fundamental = rec.get('fundamental_score', 5.0)
                technical = rec.get('technical_score', 5.0)
                data_quality = rec.get('data_quality_score', 0.5)
                
                fundamental_scores.append(fundamental)
                technical_scores.append(technical)
                data_quality_scores.append(data_quality * 100)  # Scale for visibility
                
                # Color based on overall score
                overall = rec.get('overall_score', 5.0)
                if overall >= 7:
                    colors.append('green')
                elif overall >= 4:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            # Create scatter plot
            scatter = ax.scatter(fundamental_scores, technical_scores,
                               s=data_quality_scores,  # Size by data quality
                               c=colors, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add labels for each point
            for i, symbol in enumerate(symbols):
                ax.annotate(symbol, (fundamental_scores[i], technical_scores[i]),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Add diagonal reference line
            ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='Equal Weight')
            
            # Set labels and title
            ax.set_xlabel('Fundamental Score', fontsize=12)
            ax.set_ylabel('Technical Score', fontsize=12)
            ax.set_title('Technical vs Fundamental Analysis\nSize = Data Quality, Color = Overall Score', fontsize=14)
            
            # Set axis limits
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                          markersize=10, label='High Score (≥7)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                          markersize=10, label='Medium Score (4-7)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=10, label='Low Score (<4)')
            ]
            ax.legend(handles=legend_elements, loc='upper left')
            
            # Save plot
            plot_path = self.charts_dir / "2d_technical_fundamental_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"📊 Generated 2D technical/fundamental plot: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error generating 2D plot: {e}")
            return ""
    
    def _extract_income_score(self, rec: Dict) -> float:
        """Extract income statement score from recommendation"""
        # If explicitly provided, use it
        if 'income_score' in rec:
            return rec['income_score']
        
        # Try to extract from full analysis
        fundamental_score = rec.get('fundamental_score', 5.0)
        
        # Look for income-related keywords in analysis
        full_analysis = rec.get('full_analysis', {})
        synthesis = full_analysis.get('synthesis', {})
        insights = synthesis.get('key_insights', [])
        
        income_keywords = ['revenue', 'income', 'earnings', 'profit', 'margin']
        income_mentions = sum(1 for insight in insights 
                            if any(keyword in str(insight).lower() for keyword in income_keywords))
        
        # Adjust score based on mentions
        if income_mentions > 2:
            return min(fundamental_score + 0.5, 10.0)
        elif income_mentions > 0:
            return fundamental_score
        else:
            return max(fundamental_score - 0.5, 1.0)
    
    def _extract_cashflow_score(self, rec: Dict) -> float:
        """Extract cash flow score from recommendation"""
        # If explicitly provided, use it
        if 'cashflow_score' in rec:
            return rec['cashflow_score']
        
        # Try to extract from full analysis
        fundamental_score = rec.get('fundamental_score', 5.0)
        
        # Look for cash flow keywords
        full_analysis = rec.get('full_analysis', {})
        synthesis = full_analysis.get('synthesis', {})
        insights = synthesis.get('key_insights', [])
        
        cashflow_keywords = ['cash flow', 'cash', 'liquidity', 'fcf', 'working capital']
        cashflow_mentions = sum(1 for insight in insights 
                              if any(keyword in str(insight).lower() for keyword in cashflow_keywords))
        
        # Adjust score based on mentions
        if cashflow_mentions > 2:
            return min(fundamental_score + 0.5, 10.0)
        elif cashflow_mentions > 0:
            return fundamental_score
        else:
            return max(fundamental_score - 0.5, 1.0)
    
    def _extract_balance_score(self, rec: Dict) -> float:
        """Extract balance sheet score from recommendation"""
        # If explicitly provided, use it
        if 'balance_score' in rec:
            return rec['balance_score']
        
        # Try to extract from full analysis
        fundamental_score = rec.get('fundamental_score', 5.0)
        
        # Look for balance sheet keywords
        full_analysis = rec.get('full_analysis', {})
        synthesis = full_analysis.get('synthesis', {})
        insights = synthesis.get('key_insights', [])
        
        balance_keywords = ['asset', 'liability', 'equity', 'debt', 'balance sheet', 'leverage']
        balance_mentions = sum(1 for insight in insights 
                             if any(keyword in str(insight).lower() for keyword in balance_keywords))
        
        # Adjust score based on mentions
        if balance_mentions > 2:
            return min(fundamental_score + 0.5, 10.0)
        elif balance_mentions > 0:
            return fundamental_score
        else:
            return max(fundamental_score - 0.5, 1.0)