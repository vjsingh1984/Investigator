#!/usr/bin/env python3
"""
InvestiGator - Yahoo Technical Analysis Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Comprehensive Yahoo Technical Analysis Module
Handles comprehensive technical analysis with all major indicators and volume-based analysis
"""

import logging
import requests
import time
import json
import csv
import io
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from pathlib import Path
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("talib not available - using pandas for technical indicators")

from utils.technical_indicators import get_technical_calculator

from config import get_config
from utils.cache import CacheType
from data.models import TechnicalAnalysisData
from utils.cache.cache_manager import CacheManager
from patterns.llm.llm_facade import create_llm_facade
from utils.ascii_art import ASCIIArt

logger = logging.getLogger(__name__)


def safe_float_convert(value, default=0.0):
    """Safely convert value to float with default fallback"""
    try:
        if pd.isna(value) or value is None:
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default


class MarketDataFetcher:
    """Enhanced market data fetcher with configurable periods"""
    
    def __init__(self, config):
        self.config = config
        self.min_volume = config.analysis.min_volume
        self.default_days = getattr(config.analysis, 'technical_lookback_days', 365)
    
    def get_stock_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """Fetch comprehensive stock data with single API call"""
        try:
            if days is None:
                days = self.default_days
            
            logger.info(f"Fetching {days} days of market data for {symbol}")
            
            # Single API call for maximum data
            stock = yf.Ticker(symbol)
            df = stock.history(period=f"{days}d", interval="1d")
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Validate data quality
            if len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}: only {len(df)} days")
            
            # Check volume requirement
            avg_volume = df['Volume'].mean()
            if avg_volume < self.min_volume:
                logger.warning(f"Low volume for {symbol}: {avg_volume:,.0f} < {self.min_volume:,.0f}")
            
            logger.info(f"Successfully fetched {len(df)} days of data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get additional stock information"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'beta': info.get('beta'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'dividend_yield': info.get('dividendYield'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares')
            }
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return {}

class ComprehensiveTechnicalAnalyzer:
    """Comprehensive technical analyzer with all major indicators"""
    
    def __init__(self, config):
        self.config = config
        self.data_fetcher = MarketDataFetcher(config)
        self.ollama = create_llm_facade(config)  # Use pattern-based LLM facade
        self.cache_manager = CacheManager(config)  # Use cache manager with config
        # Note: Technical indicators use parquet-only storage, no database
        
        # Create price cache directory
        self.price_cache_dir = Path(config.data_dir) / "price_cache"
        self.price_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create technical cache directory  
        self.technical_cache_dir = Path(config.data_dir) / "technical_cache"
        self.technical_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create LLM cache directory
        self.llm_cache_dir = Path(config.data_dir) / "llm_cache"
        self.llm_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_stock(self, symbol: str, days: int = 365) -> Dict:
        """Perform comprehensive technical analysis"""
        # Get symbol-specific logger
        symbol_logger = self.config.get_symbol_logger(symbol, 'yahoo_technical')
        
        logger.info(f"Starting comprehensive technical analysis for {symbol}")
        symbol_logger.info(f"Starting comprehensive technical analysis - {days} days lookback")
        
        try:
            # Check cache first for recent technical data
            cache_key = (symbol, 'technical_data', f'{days}d')
            cached_data = self.cache_manager.get(CacheType.TECHNICAL_DATA, cache_key)
            
            if cached_data:
                # Check if cached data is recent enough (less than 24 hours old)
                cache_info = cached_data.get('cache_info', {})
                cached_at = cache_info.get('cached_at')
                
                if cached_at:
                    cache_time = datetime.fromisoformat(cached_at)
                    age = datetime.utcnow() - cache_time
                    
                    # If data is less than 24 hours old and market is closed, use cached data
                    if age < timedelta(hours=24):
                        symbol_logger.info(f"Using cached technical data (age: {age})")
                        logger.info(f"Using cached technical data for {symbol} (age: {age})")
                        
                        # Extract dataframe and perform analysis
                        if 'dataframe' in cached_data:
                            df = cached_data['dataframe']
                        else:
                            # Reconstruct dataframe from data
                            df = pd.DataFrame(cached_data['data'])
                            df.index = pd.to_datetime(df.index)
                        
                        # Use cached enhanced data with pre-calculated indicators  
                        # Extract indicators and generate CSV from cached enhanced data
                        calculator = get_technical_calculator()
                        recent_data = calculator.extract_recent_data_for_llm(df, days=30)
                        csv_data = self._generate_csv_for_llm_from_enhanced(recent_data)
                        indicators = self._create_indicators_from_enhanced_df(df, symbol)
                        
                        if indicators:
                            # Perform AI analysis
                            stock_info = self.data_fetcher.get_stock_info(symbol)
                            ai_analysis = self._perform_comprehensive_ai_analysis(symbol, csv_data, indicators, stock_info)
                            
                            # Calculate technical score
                            technical_score = self._calculate_comprehensive_score(indicators)
                            
                            # Return cached analysis result
                            analysis_result = {
                                'symbol': symbol,
                                'technical_score': technical_score,
                                'indicators': indicators.__dict__,
                                'stock_info': stock_info,
                                'ai_analysis': ai_analysis,
                                'analysis_timestamp': datetime.utcnow().isoformat(),
                                'data_points': len(df),
                                'csv_data_sample': csv_data[:1000],
                                'cache_used': True,
                                'cache_age': str(age)
                            }
                            
                            # Save to database
                            symbol_logger.info("Saving analysis results to database (from cache)")
                            self._save_analysis_to_db(symbol, analysis_result)
                            
                            return analysis_result
            
            # If no cache or cache is stale, fetch fresh data
            symbol_logger.info("Fetching fresh market data from Yahoo Finance")
            df = self.data_fetcher.get_stock_data(symbol, days)
            if df.empty:
                raise RuntimeError(f"No market data available for {symbol}")
            
            # Calculate all technical indicators using centralized calculator
            symbol_logger.info(f"Calculating comprehensive technical indicators for {symbol}")
            calculator = get_technical_calculator()
            enhanced_df = calculator.calculate_all_indicators(df, symbol)
            
            # Create TechnicalAnalysisData object from enhanced DataFrame
            indicators = self._create_indicators_from_enhanced_df(enhanced_df, symbol)
            if not indicators:
                raise RuntimeError(f"Failed to create indicators object for {symbol}")
            
            # Save enhanced data with all indicators to parquet FIRST
            symbol_logger.info(f"Caching enhanced data with indicators to parquet for {symbol}")
            self._save_to_parquet(symbol, enhanced_df, days)
            
            # Get stock info
            stock_info = self.data_fetcher.get_stock_info(symbol)
            
            # Generate CSV data for LLM using last 30 days from enhanced data
            calculator = get_technical_calculator()
            recent_data = calculator.extract_recent_data_for_llm(enhanced_df, days=30)
            csv_data = self._generate_csv_for_llm_from_enhanced(recent_data)
            
            # Perform AI analysis
            ai_analysis = self._perform_comprehensive_ai_analysis(symbol, csv_data, indicators, stock_info)
            
            # Calculate technical score
            technical_score = self._calculate_comprehensive_score(indicators)
            
            # Combine results
            analysis_result = {
                'symbol': symbol,
                'technical_score': technical_score,
                'indicators': indicators.__dict__,
                'stock_info': stock_info,
                'ai_analysis': ai_analysis,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'data_points': len(df),
                'csv_data_sample': csv_data[:1000]  # First 1000 chars for verification
            }
            
            # Save to database
            symbol_logger.info("Saving analysis results to database")
            self._save_analysis_to_db(symbol, analysis_result)
            self._save_csv_to_cache(symbol, csv_data)
            
            # Save LLM response to database using DAO
            if 'prompt' in ai_analysis and 'response' in ai_analysis:
                symbol_logger.info("Saving LLM response to database")
                self._save_llm_response_to_db(
                    symbol=symbol,
                    prompt=ai_analysis['prompt'],
                    system_prompt=ai_analysis.get('system_prompt', ''),
                    response=ai_analysis['response'],
                    processing_time_ms=ai_analysis.get('processing_time_ms', 0)
                )
            
            symbol_logger.info(f"Technical analysis completed - Score: {technical_score}/10, Data points: {len(df)}")
            logger.info(f"Completed comprehensive technical analysis for {symbol} - Score: {technical_score}/10")
            return analysis_result
            
        except Exception as e:
            symbol_logger.error(f"Technical analysis failed: {str(e)}")
            logger.error(f"Error in comprehensive technical analysis for {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Technical analysis failed for {symbol}: {str(e)}")
    
    def _create_indicators_from_enhanced_df(self, enhanced_df: pd.DataFrame, symbol: str) -> TechnicalAnalysisData:
        """
        Create TechnicalAnalysisData object from enhanced DataFrame with pre-calculated indicators
        Extracts the latest values from all calculated indicators
        """
        try:
            if enhanced_df.empty:
                return None
            
            # Get latest values
            latest = enhanced_df.iloc[-1]
            
            # Create moving averages dict
            moving_averages = {}
            for period in [5, 10, 12, 20, 26, 50, 100, 200]:
                if f'SMA_{period}' in enhanced_df.columns:
                    moving_averages[f'sma_{period}'] = float(latest[f'SMA_{period}']) if not pd.isna(latest[f'SMA_{period}']) else 0.0
                if f'EMA_{period}' in enhanced_df.columns:
                    moving_averages[f'ema_{period}'] = float(latest[f'EMA_{period}']) if not pd.isna(latest[f'EMA_{period}']) else 0.0
            
            # Create momentum indicators dict
            momentum_indicators = {}
            for period in [9, 14, 21]:
                if f'RSI_{period}' in enhanced_df.columns:
                    momentum_indicators[f'rsi_{period}'] = float(latest[f'RSI_{period}']) if not pd.isna(latest[f'RSI_{period}']) else 0.0
            
            momentum_indicators.update({
                'macd': float(latest['MACD']) if 'MACD' in enhanced_df.columns and not pd.isna(latest['MACD']) else 0.0,
                'macd_signal': float(latest['MACD_Signal']) if 'MACD_Signal' in enhanced_df.columns and not pd.isna(latest['MACD_Signal']) else 0.0,
                'macd_histogram': float(latest['MACD_Histogram']) if 'MACD_Histogram' in enhanced_df.columns and not pd.isna(latest['MACD_Histogram']) else 0.0,
                'stoch_k': float(latest['Stoch_K']) if 'Stoch_K' in enhanced_df.columns and not pd.isna(latest['Stoch_K']) else 0.0,
                'stoch_d': float(latest['Stoch_D']) if 'Stoch_D' in enhanced_df.columns and not pd.isna(latest['Stoch_D']) else 0.0,
                'williams_r': float(latest['Williams_R']) if 'Williams_R' in enhanced_df.columns and not pd.isna(latest['Williams_R']) else 0.0,
                'mfi_14': float(latest['MFI_14']) if 'MFI_14' in enhanced_df.columns and not pd.isna(latest['MFI_14']) else 0.0,
            })
            
            # Create volatility indicators dict
            volatility_indicators = {
                'bb_upper': float(latest['BB_Upper']) if 'BB_Upper' in enhanced_df.columns and not pd.isna(latest['BB_Upper']) else 0.0,
                'bb_middle': float(latest['BB_Middle']) if 'BB_Middle' in enhanced_df.columns and not pd.isna(latest['BB_Middle']) else 0.0,
                'bb_lower': float(latest['BB_Lower']) if 'BB_Lower' in enhanced_df.columns and not pd.isna(latest['BB_Lower']) else 0.0,
                'bb_width': float(latest['BB_Width']) if 'BB_Width' in enhanced_df.columns and not pd.isna(latest['BB_Width']) else 0.0,
                'atr_14': float(latest['ATR_14']) if 'ATR_14' in enhanced_df.columns and not pd.isna(latest['ATR_14']) else 0.0,
                'volatility_20': float(latest['Volatility_20']) if 'Volatility_20' in enhanced_df.columns and not pd.isna(latest['Volatility_20']) else 0.0,
                # Add Fibonacci levels
                'fib_0': float(latest['Fib_0']) if 'Fib_0' in enhanced_df.columns and not pd.isna(latest['Fib_0']) else 0.0,
                'fib_23_6': float(latest['Fib_23_6']) if 'Fib_23_6' in enhanced_df.columns and not pd.isna(latest['Fib_23_6']) else 0.0,
                'fib_38_2': float(latest['Fib_38_2']) if 'Fib_38_2' in enhanced_df.columns and not pd.isna(latest['Fib_38_2']) else 0.0,
                'fib_50_0': float(latest['Fib_50_0']) if 'Fib_50_0' in enhanced_df.columns and not pd.isna(latest['Fib_50_0']) else 0.0,
                'fib_61_8': float(latest['Fib_61_8']) if 'Fib_61_8' in enhanced_df.columns and not pd.isna(latest['Fib_61_8']) else 0.0,
                'fib_78_6': float(latest['Fib_78_6']) if 'Fib_78_6' in enhanced_df.columns and not pd.isna(latest['Fib_78_6']) else 0.0,
                'fib_100': float(latest['Fib_100']) if 'Fib_100' in enhanced_df.columns and not pd.isna(latest['Fib_100']) else 0.0,
                # Add pivot points
                'pivot_point': float(latest['Pivot_Point']) if 'Pivot_Point' in enhanced_df.columns and not pd.isna(latest['Pivot_Point']) else 0.0,
                'pivot_r1': float(latest['Pivot_R1']) if 'Pivot_R1' in enhanced_df.columns and not pd.isna(latest['Pivot_R1']) else 0.0,
                'pivot_r2': float(latest['Pivot_R2']) if 'Pivot_R2' in enhanced_df.columns and not pd.isna(latest['Pivot_R2']) else 0.0,
                'pivot_s1': float(latest['Pivot_S1']) if 'Pivot_S1' in enhanced_df.columns and not pd.isna(latest['Pivot_S1']) else 0.0,
                'pivot_s2': float(latest['Pivot_S2']) if 'Pivot_S2' in enhanced_df.columns and not pd.isna(latest['Pivot_S2']) else 0.0,
            }
            
            # Create volume indicators dict
            volume_indicators = {
                'volume': float(latest['Volume']) if not pd.isna(latest['Volume']) else 0.0,
                'volume_sma_20': float(latest['Volume_SMA_20']) if 'Volume_SMA_20' in enhanced_df.columns and not pd.isna(latest['Volume_SMA_20']) else 0.0,
                'volume_ratio': float(latest['Volume_Ratio']) if 'Volume_Ratio' in enhanced_df.columns and not pd.isna(latest['Volume_Ratio']) else 0.0,
                'obv': float(latest['OBV']) if 'OBV' in enhanced_df.columns and not pd.isna(latest['OBV']) else 0.0,
                'vpt': float(latest['VPT']) if 'VPT' in enhanced_df.columns and not pd.isna(latest['VPT']) else 0.0,
                'ad': float(latest['AD']) if 'AD' in enhanced_df.columns and not pd.isna(latest['AD']) else 0.0,
                'vwap': float(latest['VWAP']) if 'VWAP' in enhanced_df.columns and not pd.isna(latest['VWAP']) else 0.0,
            }
            
            # Extract support and resistance levels
            support_levels = []
            resistance_levels = []
            
            if 'Support_1' in enhanced_df.columns:
                support_levels.append(float(latest['Support_1']) if not pd.isna(latest['Support_1']) else 0.0)
            if 'Support_2' in enhanced_df.columns:
                support_levels.append(float(latest['Support_2']) if not pd.isna(latest['Support_2']) else 0.0)
            if 'Resistance_1' in enhanced_df.columns:
                resistance_levels.append(float(latest['Resistance_1']) if not pd.isna(latest['Resistance_1']) else 0.0)
            if 'Resistance_2' in enhanced_df.columns:
                resistance_levels.append(float(latest['Resistance_2']) if not pd.isna(latest['Resistance_2']) else 0.0)
            
            # Create TechnicalAnalysisData object
            indicators = TechnicalAnalysisData(
                symbol=symbol,
                period="365d",
                analysis_date=datetime.now(),
                current_price=float(latest['Close']) if not pd.isna(latest['Close']) else 0.0,
                price_change=float(latest['Price_Change_1D']) if 'Price_Change_1D' in enhanced_df.columns and not pd.isna(latest['Price_Change_1D']) else 0.0,
                moving_averages=moving_averages,
                momentum_indicators=momentum_indicators,
                volatility_indicators=volatility_indicators,
                volume_indicators=volume_indicators,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                metadata={'data_points': len(enhanced_df), 'calculation_method': 'centralized_calculator'}
            )
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error creating indicators from enhanced DataFrame for {symbol}: {e}")
            return None
    
    def _generate_csv_for_llm_from_enhanced(self, recent_df: pd.DataFrame) -> str:
        """
        Generate CSV data for LLM from enhanced DataFrame with pre-calculated indicators
        Uses the last 30 days of data with all indicators already calculated on full dataset
        """
        try:
            # Create CSV content
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            
            # Write header - all indicators are already calculated
            header = [
                'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI_14',
                'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR_14',
                'Volume_Ratio', 'Price_Change_1D', 'Pivot_Point',
                'Fib_23_6', 'Fib_38_2', 'Fib_50_0', 'Fib_61_8', 'Fib_78_6',
                'Support_1', 'Resistance_1'
            ]
            writer.writerow(header)
            
            # Write data rows using pre-calculated indicators
            for _, row in recent_df.iterrows():
                data_row = [
                    row.name.strftime('%Y-%m-%d'),
                    f"{safe_float_convert(row['Open']):.2f}",
                    f"{safe_float_convert(row['High']):.2f}",
                    f"{safe_float_convert(row['Low']):.2f}",
                    f"{safe_float_convert(row['Close']):.2f}",
                    f"{safe_float_convert(row['Volume']):,.0f}",
                    f"{safe_float_convert(row.get('SMA_20', 0)):.2f}",
                    f"{safe_float_convert(row.get('SMA_50', 0)):.2f}",
                    f"{safe_float_convert(row.get('SMA_200', 0)):.2f}",
                    f"{safe_float_convert(row.get('EMA_12', 0)):.2f}",
                    f"{safe_float_convert(row.get('EMA_26', 0)):.2f}",
                    f"{safe_float_convert(row.get('RSI_14', 0)):.1f}",
                    f"{safe_float_convert(row.get('MACD', 0)):.3f}",
                    f"{safe_float_convert(row.get('MACD_Signal', 0)):.3f}",
                    f"{safe_float_convert(row.get('BB_Upper', 0)):.2f}",
                    f"{safe_float_convert(row.get('BB_Lower', 0)):.2f}",
                    f"{safe_float_convert(row.get('ATR_14', 0)):.2f}",
                    f"{safe_float_convert(row.get('Volume_Ratio', 0)):.2f}",
                    f"{safe_float_convert(row.get('Price_Change_1D', 0)):.2f}",
                    f"{safe_float_convert(row.get('Pivot_Point', 0)):.2f}",
                    f"{safe_float_convert(row.get('Fib_23_6', 0)):.2f}",
                    f"{safe_float_convert(row.get('Fib_38_2', 0)):.2f}",
                    f"{safe_float_convert(row.get('Fib_50_0', 0)):.2f}",
                    f"{safe_float_convert(row.get('Fib_61_8', 0)):.2f}",
                    f"{safe_float_convert(row.get('Fib_78_6', 0)):.2f}",
                    f"{safe_float_convert(row.get('Support_1', 0)):.2f}",
                    f"{safe_float_convert(row.get('Resistance_1', 0)):.2f}"
                ]
                writer.writerow(data_row)
            
            return csv_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating CSV from enhanced DataFrame: {e}")
            return "Date,Open,High,Low,Close,Volume\nError generating CSV data"
    
    def _perform_comprehensive_ai_analysis(self, symbol: str, csv_data: str, indicators: TechnicalAnalysisData, stock_info: Dict) -> Dict:
        """Perform comprehensive AI-powered technical analysis"""
        # Get symbol-specific logger
        symbol_logger = self.config.get_symbol_logger(symbol, 'yahoo_technical')
        
        # Use prompt manager for enhanced prompting with JSON response
        from utils.prompt_manager import get_prompt_manager
        prompt_manager = get_prompt_manager()
        
        # Prepare indicators summary
        # Get values from dictionaries with defaults
        ma = indicators.moving_averages
        mom = indicators.momentum_indicators
        vol = indicators.volatility_indicators
        volume = indicators.volume_indicators
        
        indicators_summary = f"""Current Price: ${indicators.current_price or 0:.2f}

Moving Averages:
- SMA 20: ${ma.get('sma_20', 0):.2f} | SMA 50: ${ma.get('sma_50', 0):.2f} | SMA 200: ${ma.get('sma_200', 0):.2f}
- EMA 12: ${ma.get('ema_12', 0):.2f} | EMA 26: ${ma.get('ema_26', 0):.2f} | EMA 50: ${ma.get('ema_50', 0):.2f}

Momentum Indicators:
- RSI (14): {mom.get('rsi_14', 0):.1f} | Stochastic %K: {mom.get('stoch_k', 0):.1f} | Williams %R: {mom.get('williams_r', 0):.1f}
- ROC (10): {mom.get('roc_10', 0):.2f}% | ROC (20): {mom.get('roc_20', 0):.2f}%

MACD Analysis:
- MACD: {mom.get('macd', 0):.4f} | Signal: {mom.get('macd_signal', 0):.4f} | Histogram: {mom.get('macd_histogram', 0):.4f}

Bollinger Bands:
- Upper: ${vol.get('bb_upper', 0):.2f} | Lower: ${vol.get('bb_lower', 0):.2f} | Position: {vol.get('bb_position', 0):.2f}
- Width: {vol.get('bb_width', 0):.4f}

Volume Analysis:
- Current Volume: {volume.get('volume', 0):,.0f} | 20-Day Avg: {volume.get('volume_sma_20', 0):,.0f}
- Volume Ratio: {volume.get('volume_ratio', 0):.2f}
- OBV: {volume.get('obv', 0):,.0f} | Money Flow Index: {mom.get('mfi_14', 0):.1f}

Support & Resistance:
- Traditional Support: ${indicators.support_levels[0] if indicators.support_levels else 0:.2f} / ${indicators.support_levels[1] if len(indicators.support_levels) > 1 else 0:.2f}
- Traditional Resistance: ${indicators.resistance_levels[0] if indicators.resistance_levels else 0:.2f} / ${indicators.resistance_levels[1] if len(indicators.resistance_levels) > 1 else 0:.2f}
- Volume Support: ${volume.get('VOLUME_SUPPORT', 0):.2f} | Volume Resistance: ${volume.get('VOLUME_RESISTANCE', 0):.2f}
- VWAP: ${volume.get('vwap', 0):.2f} | Pivot Point: ${vol.get('pivot_point', 0):.2f}

Fibonacci Levels:
- 23.6%: ${vol.get('fib_23_6', 0):.2f} | 38.2%: ${vol.get('fib_38_2', 0):.2f} | 50.0%: ${vol.get('fib_50_0', 0):.2f}
- 61.8%: ${vol.get('fib_61_8', 0):.2f} | 78.6%: ${vol.get('fib_78_6', 0):.2f}
- 52W High: ${vol.get('fib_0', 0):.2f} | 52W Low: ${vol.get('fib_100', 0):.2f}

Volatility:
- ATR (14): ${vol.get('atr_14', 0):.2f} | 20-Day Volatility: {vol.get('volatility_20', 0):.1f}%

Price Performance:
- 1D: {indicators.price_change_percent or 0:.2f}% | 1W: {ma.get('PRICE_CHANGE_1W', 0):.2f}% | 1M: {ma.get('PRICE_CHANGE_1M', 0):.2f}%
- 3M: {indicators.metadata.get('price_changes', {}).get('price_change_3m', 0):.2f}% | 6M: {indicators.metadata.get('price_changes', {}).get('price_change_6m', 0):.2f}% | 1Y: {indicators.metadata.get('price_changes', {}).get('price_change_1y', 0):.2f}%"""
        
        # Use standard prompt manager
        analysis_prompt = prompt_manager.render_technical_analysis_prompt(
            symbol=symbol,
            analysis_date=datetime.now().strftime('%Y-%m-%d'),
            data_points=indicators.metadata.get('data_points', 0),
            current_price=indicators.current_price,
            csv_data=csv_data,
            indicators_summary=indicators_summary,
            stock_info=stock_info
        )
        
        # Get system prompt for technical analysis
        system_prompt = "You are a senior quantitative analyst and technical analysis expert with 20+ years of experience in institutional trading and portfolio management."
        
        try:
            # Track processing time
            start_time = time.time()
            
            # Get AI analysis
            # Use LLM facade for technical analysis
            technical_result = self.ollama.analyze_technical(
                symbol=symbol,
                price_data={
                    'csv_data': csv_data,
                    'stock_info': stock_info
                },
                indicators=indicators_summary
            )
            
            # Extract response for compatibility with existing code
            # The facade returns structured data, but we need the raw response content
            if 'error' in technical_result:
                # If there's an error, use the raw_response if available
                ai_response = technical_result.get('raw_response', '')
            else:
                # For successful responses, the facade returns parsed JSON
                # Convert it back to JSON string for compatibility with existing parsing logic
                import json
                ai_response = json.dumps(technical_result, ensure_ascii=False)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Parse JSON response with metadata
            technical_metadata = {
                'model': 'llama3.1:8b-instruct-q8_0',  # Default technical analysis model
                'processing_time_ms': processing_time_ms,
                'symbol': symbol,
                'analysis_type': 'technical_analysis'
            }
            
            # LLM facade handles all response processing internally
            # technical_result should be a properly processed dict
            if isinstance(technical_result, dict) and 'error' not in technical_result:
                analysis_result = technical_result
                symbol_logger.info(f"Successfully received processed technical analysis response")
            else:
                error_msg = f"LLM facade returned error for technical analysis of {symbol}: {technical_result.get('error', 'Unknown error')}"
                symbol_logger.error(error_msg)
                analysis_result = technical_result  # Return error response as-is
            
            # Save prompts to cache
            self._save_technical_prompts_to_cache(symbol, analysis_prompt, system_prompt, ai_response)
            
            # Note: LLM facade already handles response processing and caching internally
            # No need to manually save LLM responses here
            
            symbol_logger.info(f"AI technical analysis completed in {processing_time_ms}ms")
            logger.info(f"✅ Completed AI technical analysis for {symbol} in {processing_time_ms}ms")
            return analysis_result
            
        except Exception as e:
            symbol_logger.error(f"AI analysis failed: {str(e)}")
            logger.error(f"Error in AI analysis: {e}")
            return {
                'technical_score': 5.0,
                'trend_analysis': 'Analysis unavailable due to error',
                'error': str(e)
            }
    
    def _parse_technical_analysis(self, response: str) -> Dict:
        """Parse the AI technical analysis response"""
        try:
            result = {
                'technical_score': 5.0,
                'trend_analysis': '',
                'momentum_assessment': '',
                'volume_analysis': '',
                'support_resistance_analysis': '',
                'volatility_risk_assessment': '',
                'entry_exit_strategy': '',
                'key_insights': [],
                'risk_factors': [],
                'investment_recommendation': ''
            }
            
            # Extract technical score
            score_match = re.search(r'\*\*TECHNICAL SCORE:\s*\[?(\d+(?:\.\d+)?)\]?\*\*', response, re.IGNORECASE)
            if score_match:
                result['technical_score'] = float(score_match.group(1))
            
            # Extract sections
            sections = {
                'trend_analysis': r'\*\*TREND ANALYSIS:\*\*(.*?)(?=\*\*[A-Z]|\Z)',
                'momentum_assessment': r'\*\*MOMENTUM ASSESSMENT:\*\*(.*?)(?=\*\*[A-Z]|\Z)',
                'volume_analysis': r'\*\*VOLUME ANALYSIS:\*\*(.*?)(?=\*\*[A-Z]|\Z)',
                'support_resistance_analysis': r'\*\*SUPPORT & RESISTANCE ANALYSIS:\*\*(.*?)(?=\*\*[A-Z]|\Z)',
                'volatility_risk_assessment': r'\*\*VOLATILITY & RISK ASSESSMENT:\*\*(.*?)(?=\*\*[A-Z]|\Z)',
                'entry_exit_strategy': r'\*\*ENTRY & EXIT STRATEGY:\*\*(.*?)(?=\*\*[A-Z]|\Z)',
                'investment_recommendation': r'\*\*INVESTMENT RECOMMENDATION:\*\*(.*?)(?=\*\*[A-Z]|\Z)'
            }
            
            for key, pattern in sections.items():
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    result[key] = match.group(1).strip()
            
            # Extract insights and risks as lists
            insights_match = re.search(r'\*\*KEY INSIGHTS:\*\*(.*?)(?=\*\*[A-Z]|\Z)', response, re.DOTALL | re.IGNORECASE)
            if insights_match:
                insights_text = insights_match.group(1).strip()
                result['key_insights'] = [line.strip('- ').strip() for line in insights_text.split('\n') if line.strip().startswith('-')]
            
            risks_match = re.search(r'\*\*RISK FACTORS:\*\*(.*?)(?=\*\*[A-Z]|\Z)', response, re.DOTALL | re.IGNORECASE)
            if risks_match:
                risks_text = risks_match.group(1).strip()
                result['risk_factors'] = [line.strip('- ').strip() for line in risks_text.split('\n') if line.strip().startswith('-')]
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing technical analysis: {e}")
            return {
                'technical_score': 5.0,
                'trend_analysis': 'Error parsing analysis',
                'error': str(e)
            }
    
    def _calculate_comprehensive_score(self, indicators: TechnicalAnalysisData) -> float:
        """Calculate comprehensive technical score based on all indicators"""
        try:
            score = 5.0  # Start with neutral
            
            # Get values from dictionaries
            ma = indicators.moving_averages
            mom = indicators.momentum_indicators
            vol = indicators.volatility_indicators
            volume = indicators.volume_indicators
            
            # Moving Average Score (25% weight)
            ma_score = 5.0
            current = indicators.current_price or 0
            
            sma_20 = ma.get('SMA_20', 0)
            sma_50 = ma.get('SMA_50', 0)
            sma_200 = ma.get('SMA_200', 0)
            ema_20 = ma.get('EMA_20', 0)
            
            # Short-term MA comparison
            if current > sma_20 > sma_50:
                ma_score += 1.5  # Bullish alignment
            elif current < sma_20 < sma_50:
                ma_score -= 1.5  # Bearish alignment
            
            # Long-term trend
            if current > sma_200:
                ma_score += 1.0  # Above long-term trend
            else:
                ma_score -= 1.0  # Below long-term trend
            
            # EMA vs SMA strength
            if ema_20 > sma_20:
                ma_score += 0.5  # Recent momentum positive
            else:
                ma_score -= 0.5  # Recent momentum negative
            
            # Momentum Score (25% weight)
            momentum_score = 5.0
            
            rsi_14 = mom.get('RSI_14', 50)  # Default to neutral
            macd = mom.get('MACD', 0)
            macd_signal = mom.get('MACD_SIGNAL', 0)
            macd_histogram = mom.get('MACD_HISTOGRAM', 0)
            stoch_k = mom.get('STOCH_K', 50)
            
            # RSI analysis
            if 40 <= rsi_14 <= 60:
                momentum_score += 1.0  # Neutral zone
            elif 60 < rsi_14 <= 70:
                momentum_score += 1.5  # Bullish but not overbought
            elif 30 <= rsi_14 < 40:
                momentum_score -= 1.5  # Bearish but not oversold
            elif rsi_14 > 80:
                momentum_score -= 1.0  # Overbought
            elif rsi_14 < 20:
                momentum_score += 0.5  # Oversold reversal potential
            
            # MACD analysis
            if macd > macd_signal and macd_histogram > 0:
                momentum_score += 1.0  # Bullish MACD
            elif macd < macd_signal and macd_histogram < 0:
                momentum_score -= 1.0  # Bearish MACD
            
            # Stochastic analysis
            if 20 <= stoch_k <= 80:
                momentum_score += 0.5  # Not extreme
            
            # Volume Score (20% weight)
            volume_score = 5.0
            
            volume_ratio = volume.get('VOLUME_RATIO', 1.0)
            money_flow_index = mom.get('MFI', 50)  # MFI is in momentum indicators
            
            # Volume ratio analysis
            if volume_ratio > 1.5:
                volume_score += 1.5  # High volume
            elif volume_ratio < 0.5:
                volume_score -= 1.0  # Low volume
            
            # Money Flow Index
            if 40 <= money_flow_index <= 60:
                volume_score += 1.0  # Balanced
            elif money_flow_index > 80:
                volume_score -= 0.5  # Overbought
            elif money_flow_index < 20:
                volume_score += 0.5  # Oversold
            
            # Support/Resistance Score (15% weight)
            sr_score = 5.0
            
            # Position relative to support/resistance
            resistance_level_1 = indicators.resistance_levels[0] if indicators.resistance_levels else current
            support_level_1 = indicators.support_levels[0] if indicators.support_levels else current
            
            if current > resistance_level_1:
                sr_score += 1.0  # Above resistance
            elif current < support_level_1:
                sr_score -= 1.0  # Below support
            
            # Bollinger Band position
            bb_position = vol.get('BB_POSITION', 0.5)
            if 0.3 <= bb_position <= 0.7:
                sr_score += 0.5  # Middle range
            elif bb_position > 0.8:
                sr_score -= 0.5  # Near upper band
            elif bb_position < 0.2:
                sr_score += 0.5  # Near lower band (reversal potential)
            
            # Volatility Score (15% weight)
            volatility_score = 5.0
            
            # ATR relative to price
            atr_14 = vol.get('ATR_14', 0)
            atr_ratio = atr_14 / current if current > 0 else 0
            if 0.01 <= atr_ratio <= 0.03:
                volatility_score += 1.0  # Normal volatility
            elif atr_ratio > 0.05:
                volatility_score -= 1.0  # High volatility
            
            # Bollinger Band width
            bb_width = vol.get('BB_WIDTH', 0.2)
            if bb_width < 0.1:
                volatility_score += 0.5  # Low volatility (potential breakout)
            elif bb_width > 0.3:
                volatility_score -= 0.5  # High volatility
            
            # Weighted final score
            final_score = (
                ma_score * 0.25 +
                momentum_score * 0.25 +
                volume_score * 0.20 +
                sr_score * 0.15 +
                volatility_score * 0.15
            )
            
            # Ensure score is within bounds
            final_score = max(1.0, min(10.0, final_score))
            
            return round(final_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive score: {e}")
            return 5.0
    
    def _save_technical_prompts_to_cache(self, symbol: str, prompt: str, system_prompt: str, response: str):
        """Save technical analysis prompts to cache"""
        try:
            # Use symbol-specific LLM cache directory (matching sec_fundamental.py pattern)
            cache_dir = self.config.get_symbol_cache_path(symbol, 'llm')
            
            # Save prompt
            prompt_file = cache_dir / f"prompt_technical_indicators.txt"
            with open(prompt_file, 'w') as f:
                f.write(f"=== TECHNICAL ANALYSIS PROMPT FOR {symbol} ===\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== SYSTEM PROMPT ===\n")
                f.write(system_prompt)
                f.write("\n\n=== USER PROMPT ===\n")
                f.write(prompt)
            
            # Save response
            response_file = cache_dir / f"response_technical_indicators.txt"
            with open(response_file, 'w') as f:
                f.write(f"=== TECHNICAL ANALYSIS RESPONSE FOR {symbol} ===\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== AI RESPONSE ===\n")
                f.write(response)
            
            logger.info(f"📝 Saved technical analysis prompts to data/llm_cache/{symbol}/ for {symbol}")
            
        except Exception as e:
            logger.warning(f"Error saving technical prompts to cache: {e}")
    
    def _get_latest_fiscal_period(self):
        """
        Determine the latest fiscal period for cache key generation.
        Returns intelligent defaults for technical analysis.
        """
        try:
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
            
            return fiscal_year, fiscal_period
            
        except Exception as e:
            logger.warning(f"Could not determine fiscal period: {e}, using defaults")
            return datetime.now().year, "FY"
    
    def _save_llm_response_to_db(self, symbol: str, prompt: str, system_prompt: str, response: str, processing_time_ms: int, response_metadata: Dict = None):
        """Save LLM response using cache manager for technical analysis"""
        try:
            # Prepare prompt data (store full prompt directly)
            prompt_data = prompt
            
            # Prepare model info
            model_info = {
                'model': self.config.ollama.models.get('technical_analysis', 'qwen2.5:32b-instruct-q4_K_M'),
                'temperature': 0.1,
                'top_p': 0.9,
                'num_ctx': 4096
            }
            
            # Try to detect if response is JSON
            response_type = 'text'
            response_content = response
            
            try:
                # Attempt to parse as JSON
                parsed_json = json.loads(response)
                response_type = 'json'
                response_content = parsed_json
            except:
                # Keep as text
                pass
            
            # Prepare response object
            response_obj = {
                'type': response_type,
                'content': response_content
            }
            
            # Prepare metadata
            metadata = {
                'processing_time_ms': processing_time_ms,
                'response_length': len(response),
                'timestamp': datetime.now().isoformat(),
                'llm_type': 'ta'
            }
            
            # Extract technical score from response
            if response_type == 'text':
                parsed_data = self._parse_technical_analysis(response)
                if parsed_data:
                    metadata['extracted_scores'] = {
                        'technical_score': parsed_data.get('technical_score', 5.0)
                    }
                    metadata['summary'] = parsed_data.get('trend_analysis', '')
            
            # Determine fiscal period for cache key
            fiscal_year, fiscal_period = self._get_latest_fiscal_period()
            
            # Save using cache manager (will handle both disk and RDBMS storage)
            # Use intelligent defaults: TECHNICAL as form_type for technical analysis
            cache_key = {
                'symbol': symbol,
                'form_type': 'TECHNICAL',  # Intelligent default for technical analysis
                'period': f"{fiscal_year}-{fiscal_period}",
                'fiscal_year': fiscal_year,  # Separate key for file pattern
                'fiscal_period': fiscal_period,  # Separate key for file pattern
                'llm_type': 'ta'
            }
            cache_value = {
                'prompt': prompt_data,
                'model_info': model_info,
                'response': response_obj,
                'metadata': metadata
            }
            
            # Store with negative priority for LLM responses (audit-only, no lookup)
            success = self.cache_manager.set(CacheType.LLM_RESPONSE, cache_key, cache_value)
            
            if success:
                logger.info(f"💾 Stored technical analysis LLM response for {symbol} (type: {response_type})")
            else:
                logger.error(f"Failed to store technical analysis LLM response for {symbol}")
                
        except Exception as e:
            logger.error(f"Error saving LLM response to database: {e}")
            logger.error(f"Response preview: {response[:500]}...")
    
    def _save_csv_to_cache(self, symbol: str, csv_data: str):
        """Save CSV data to cache for debugging"""
        try:
            # Use symbol-specific cache directory
            cache_dir = self.config.get_symbol_cache_path(symbol, 'technical')
            
            csv_file = cache_dir / f"technical_data_{symbol}.csv"
            with open(csv_file, 'w') as f:
                f.write(csv_data)
            
            logger.info(f"📊 Saved technical CSV data to cache for {symbol}")
            
        except Exception as e:
            logger.warning(f"Error saving CSV to cache: {e}")
    
    def _save_analysis_to_db(self, symbol: str, analysis_result: Dict):
        """Technical analysis results are saved to parquet files only (no database)"""
        try:
            # Technical indicators use parquet-only storage as configured centrally
            logger.info(f"💾 Technical analysis for {symbol} saved to parquet storage only")
        except Exception as e:
            logger.warning(f"Note: Technical analysis uses parquet-only storage: {e}")
    
    def _create_default_analysis(self, symbol: str, error_message: str) -> Dict:
        """Create default analysis when errors occur"""
        return {
            'symbol': symbol,
            'technical_score': 5.0,
            'indicators': {},
            'stock_info': {},
            'ai_analysis': {
                'technical_score': 5.0,
                'trend_analysis': f'Analysis unavailable: {error_message}',
                'error': error_message
            },
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'data_points': 0,
            'error': error_message
        }
    
    def _save_to_parquet(self, symbol: str, df: pd.DataFrame, days: int = 365):
        """Save enhanced price data with indicators to cache using pandas DataFrame with parquet.gz compression"""
        try:
            # Use cache manager exclusively for pandas DataFrame storage with parquet.gz compression
            cache_key = (symbol, 'technical_data', f'{days}d')
            
            # Ensure index is datetime for proper parquet storage
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Prepare comprehensive metadata for technical indicators
            technical_indicator_columns = [col for col in df.columns if any(x in col for x in [
                'SMA_', 'EMA_', 'RSI_', 'MACD', 'BB_', 'Fib_', 'ATR_', 'Pivot_', 
                'Stoch_', 'Williams_', 'MFI_', 'Volume_', 'OBV', 'VPT', 'AD', 'VWAP',
                'Support_', 'Resistance_', 'Price_Change_', 'Volatility_'
            ])]
            
            # Prepare cache value with pandas DataFrame and comprehensive metadata
            cache_value = {
                'dataframe': df,  # Store the complete enhanced DataFrame with all calculated indicators
                'metadata': {
                    'symbol': symbol,
                    'days': days,
                    'start_date': str(df.index.min()),
                    'end_date': str(df.index.max()),
                    'records': len(df),
                    'total_columns': len(df.columns),
                    'price_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
                    'technical_indicator_columns': technical_indicator_columns,
                    'technical_indicators_count': len(technical_indicator_columns),
                    'indicators_included': True,
                    'fibonacci_included': any('Fib_' in col for col in df.columns),
                    'pivot_points_included': any('Pivot_' in col for col in df.columns),
                    'volume_indicators_included': any(x in str(df.columns) for x in ['OBV', 'VPT', 'AD', 'VWAP', 'Volume_Ratio']),
                    'calculation_method': 'centralized_technical_calculator',
                    'data_source': 'yahoo_finance',
                    'cache_timestamp': datetime.utcnow().isoformat()
                }
            }
            
            # Save to cache using ParquetCacheStorageHandler (pandas → parquet.gz conversion)
            success = self.cache_manager.set(CacheType.TECHNICAL_DATA, cache_key, cache_value)
            
            if success:
                logger.info(f"💾 Cached {len(df)} records with {len(technical_indicator_columns)} technical indicators (pandas → parquet.gz)")
                
                # Verify cache retrieval to confirm pandas DataFrame flow
                try:
                    cached_info = self.cache_manager.get(CacheType.TECHNICAL_DATA, cache_key)
                    if cached_info and 'dataframe' in cached_info:
                        cached_df = cached_info['dataframe']
                        cache_metadata = cached_info.get('cache_info', {})
                        logger.info(f"📊 Cache verification: {len(cached_df)} records, {cache_metadata.get('compression', 'unknown')} compression, {len(cached_df.columns)} columns")
                        logger.debug(f"🔍 Sample columns: {list(cached_df.columns)[:10]}...")
                    else:
                        logger.warning("Cache verification failed: no dataframe in cached data")
                except Exception as verify_error:
                    logger.warning(f"Cache verification failed: {verify_error}")
                    
            else:
                logger.error(f"Failed to save technical data to cache for {symbol}")
                # Fallback: save legacy parquet file
                try:
                    parquet_file = self.price_cache_dir / f"{symbol}.parquet"
                    df.to_parquet(parquet_file, compression='gzip', index=True)
                    logger.info(f"📄 Fallback: saved legacy parquet file with gzip compression: {parquet_file}")
                except Exception as fallback_error:
                    logger.error(f"Fallback parquet save also failed: {fallback_error}")
            
        except Exception as e:
            logger.error(f"Error saving technical data to cache: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    @staticmethod
    def load_price_data(symbol: str, config=None) -> Optional[pd.DataFrame]:
        """Load price data from parquet file"""
        try:
            if config is None:
                config = get_config()
            
            price_cache_dir = Path(config.data_dir) / "price_cache"
            parquet_file = price_cache_dir / f"{symbol}.parquet"
            
            if not parquet_file.exists():
                logger.warning(f"No parquet file found for {symbol}")
                return None
            
            df = pd.read_parquet(parquet_file)
            logger.info(f"📈 Loaded {len(df)} rows of price data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading parquet data for {symbol}: {e}")
            return None

# Main execution function
def analyze_symbol(symbol: str, days: int = 365) -> Dict:
    """Main function to analyze a symbol"""
    config = get_config()
    analyzer = ComprehensiveTechnicalAnalyzer(config)
    return analyzer.analyze_stock(symbol, days)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Technical Analysis")
    parser.add_argument("--symbol", required=True, help="Stock symbol to analyze")
    parser.add_argument("--days", type=int, default=365, help="Number of days of data to fetch")
    parser.add_argument("--test-data", action="store_true", help="Test data fetching")
    
    args = parser.parse_args()
    
    # Display technical analysis banner
    ASCIIArt.print_banner('technical')
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        result = analyze_symbol(args.symbol, args.days)
        print(f"\n=== TECHNICAL ANALYSIS RESULTS FOR {args.symbol} ===")
        print(f"Current Price: ${result['indicators']['current_price']:.2f}")
        print(f"Technical Score: {result['technical_score']}/10")
        print(f"Data Points: {result['data_points']}")
        print(f"Analysis Timestamp: {result['analysis_timestamp']}")
        
        if 'ai_analysis' in result:
            ai = result['ai_analysis']
            print(f"\nAI Technical Score: {ai.get('technical_score', 'N/A')}")
            if 'investment_recommendation' in ai:
                print(f"Investment Recommendation: {ai['investment_recommendation']}")
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()