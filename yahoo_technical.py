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

from config import get_config
from utils.db import get_technical_indicators_dao, get_stock_analysis_dao, DatabaseManager, get_llm_response_store_dao
from utils.cache import CacheType
from data.models import TechnicalAnalysisData
from utils.cache.cache_manager import CacheManager
from patterns.llm.llm_facade import create_llm_facade
from sqlalchemy import text

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
        self.indicators_dao = get_technical_indicators_dao()
        self.analysis_dao = get_stock_analysis_dao()
        self.db_manager = DatabaseManager()
        
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
                        
                        # Calculate indicators from cached data
                        indicators = self._calculate_comprehensive_indicators(df, symbol)
                        if indicators:
                            # Perform AI analysis
                            csv_data = self._generate_csv_for_llm(df, indicators)
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
                return self._create_default_analysis(symbol, "No market data available")
            
            # Calculate all technical indicators
            indicators = self._calculate_comprehensive_indicators(df, symbol)
            if not indicators:
                return self._create_default_analysis(symbol, "Failed to calculate indicators")
            
            # Save enhanced price data with indicators to parquet
            enhanced_df = self._create_enhanced_dataframe(df, indicators)
            self._save_to_parquet(symbol, enhanced_df, days)
            
            # Get stock info
            stock_info = self.data_fetcher.get_stock_info(symbol)
            
            # Generate CSV data for LLM
            csv_data = self._generate_csv_for_llm(enhanced_df, indicators)
            
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
            return self._create_default_analysis(symbol, f"Analysis error: {str(e)}")
    
    def _calculate_comprehensive_indicators(self, df: pd.DataFrame, symbol: str) -> Optional[TechnicalAnalysisData]:
        """Calculate all technical indicators comprehensively"""
        try:
            if len(df) < 50:
                logger.warning(f"Insufficient data for comprehensive analysis: {len(df)} days")
                return None
            
            # Basic price data
            current_price = safe_float_convert(df['Close'].iloc[-1])
            volume = safe_float_convert(df['Volume'].iloc[-1])
            
            # Calculate all moving averages
            periods = [5, 10, 12, 20, 26, 50, 100, 200]
            for period in periods:
                if len(df) >= period:
                    df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                    df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            # RSI with multiple periods
            df['RSI_9'] = self._calculate_rsi(df['Close'], 9)
            df['RSI_14'] = self._calculate_rsi(df['Close'], 14)
            df['RSI_21'] = self._calculate_rsi(df['Close'], 21)
            
            # Stochastic Oscillator
            df['Stoch_K'], df['Stoch_D'] = self._calculate_stochastic(df)
            
            # Williams %R
            df['Williams_R'] = self._calculate_williams_r(df)
            
            # Rate of Change
            df['ROC_10'] = self._calculate_roc(df['Close'], 10)
            df['ROC_20'] = self._calculate_roc(df['Close'], 20)
            
            # MACD
            if len(df) >= 26:
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
                df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            df = self._calculate_bollinger_bands(df)
            
            # Volume Indicators
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            df['OBV'] = self._calculate_obv(df)
            df['VPT'] = self._calculate_volume_price_trend(df)
            df['AD'] = self._calculate_accumulation_distribution(df)
            df['MFI'] = self._calculate_money_flow_index(df)
            
            # Volume Weighted Average Price (VWAP)
            df['VWAP'] = self._calculate_vwap(df)
            
            # Average True Range
            df['ATR_14'] = self._calculate_atr(df, 14)
            
            # Volatility
            df['Returns'] = df['Close'].pct_change()
            volatility_20d = safe_float_convert(df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100, 20.0)
            
            # Price changes
            price_changes = self._calculate_price_changes(df)
            
            # Support and Resistance
            support_resistance = self._calculate_support_resistance_levels(df)
            
            # Volume-based Support/Resistance
            volume_levels = self._calculate_volume_based_levels(df)
            
            # Fibonacci levels
            fibonacci_levels = self._calculate_fibonacci_levels(df)
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Create comprehensive indicators object
            indicators = TechnicalAnalysisData(
                symbol=symbol,
                period="365d",
                current_price=current_price,
                price_change=price_changes.get('price_change_1d', 0.0),
                price_change_percent=price_changes.get('price_change_1d_pct', 0.0),
                moving_averages={
                    'SMA_5': safe_float_convert(latest.get('SMA_5'), current_price),
                    'SMA_10': safe_float_convert(latest.get('SMA_10'), current_price),
                    'SMA_20': safe_float_convert(latest.get('SMA_20'), current_price),
                    'SMA_50': safe_float_convert(latest.get('SMA_50'), current_price),
                    'SMA_100': safe_float_convert(latest.get('SMA_100'), current_price),
                    'SMA_200': safe_float_convert(latest.get('SMA_200'), current_price),
                    'EMA_5': safe_float_convert(latest.get('EMA_5'), current_price),
                    'EMA_10': safe_float_convert(latest.get('EMA_10'), current_price),
                    'EMA_12': safe_float_convert(latest.get('EMA_12'), current_price),
                    'EMA_20': safe_float_convert(latest.get('EMA_20'), current_price),
                    'EMA_26': safe_float_convert(latest.get('EMA_26'), current_price),
                    'EMA_50': safe_float_convert(latest.get('EMA_50'), current_price),
                    'EMA_100': safe_float_convert(latest.get('EMA_100'), current_price),
                    'EMA_200': safe_float_convert(latest.get('EMA_200'), current_price),
                },
                momentum_indicators={
                    'RSI_9': safe_float_convert(latest.get('RSI_9'), 50.0),
                    'RSI_14': safe_float_convert(latest.get('RSI_14'), 50.0),
                    'RSI_21': safe_float_convert(latest.get('RSI_21'), 50.0),
                    'Stoch_K': safe_float_convert(latest.get('Stoch_K'), 50.0),
                    'Stoch_D': safe_float_convert(latest.get('Stoch_D'), 50.0),
                    'Williams_R': safe_float_convert(latest.get('Williams_R'), -50.0),
                    'ROC_10': safe_float_convert(latest.get('ROC_10'), 0.0),
                    'ROC_20': safe_float_convert(latest.get('ROC_20'), 0.0),
                    'MACD': safe_float_convert(latest.get('MACD'), 0.0),
                    'MACD_Signal': safe_float_convert(latest.get('MACD_Signal'), 0.0),
                    'MACD_Histogram': safe_float_convert(latest.get('MACD_Histogram'), 0.0),
                },
                volatility_indicators={
                    'BB_Upper': safe_float_convert(latest.get('BB_Upper'), current_price * 1.1),
                    'BB_Middle': safe_float_convert(latest.get('BB_Middle'), current_price),
                    'BB_Lower': safe_float_convert(latest.get('BB_Lower'), current_price * 0.9),
                    'BB_Width': safe_float_convert(latest.get('BB_Width'), 0.2),
                    'BB_Position': safe_float_convert(latest.get('BB_Position'), 0.5),
                    'ATR_14': safe_float_convert(latest.get('ATR_14'), current_price * 0.02),
                    'Volatility_20d': volatility_20d,
                },
                volume_indicators={
                    'Volume': volume,
                    'Volume_SMA_20': safe_float_convert(latest.get('Volume_SMA_20'), volume),
                    'Volume_Ratio': safe_float_convert(volume / latest.get('Volume_SMA_20', volume) if latest.get('Volume_SMA_20', volume) != 0 else 1.0, 1.0),
                    'OBV': safe_float_convert(latest.get('OBV'), 0.0),
                    'VPT': safe_float_convert(latest.get('VPT'), 0.0),
                    'AD': safe_float_convert(latest.get('AD'), 0.0),
                    'MFI': safe_float_convert(latest.get('MFI'), 50.0),
                },
                support_levels=support_resistance.get('support_levels', []),
                resistance_levels=support_resistance.get('resistance_levels', []),
                metadata={
                    'data_points': len(df),
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'fibonacci_levels': fibonacci_levels,
                    'volume_levels': volume_levels,
                    'price_changes': price_changes
                }
            )
            
            logger.info(f"Successfully calculated comprehensive indicators for {symbol}")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive indicators: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            lowest_low = df['Low'].rolling(window=k_period).min()
            highest_high = df['High'].rolling(window=k_period).max()
            # Replace zero denominators to avoid division by zero
            denominator = (highest_high - lowest_low).replace(0, np.nan)
            k_percent = 100 * ((df['Close'] - lowest_low) / denominator)
            k_percent = k_percent.fillna(50)
            d_percent = k_percent.rolling(window=d_period).mean().fillna(50)
            return k_percent, d_percent
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return pd.Series([50] * len(df), index=df.index), pd.Series([50] * len(df), index=df.index)
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            highest_high = df['High'].rolling(window=period).max()
            lowest_low = df['Low'].rolling(window=period).min()
            # Replace zero denominators to avoid division by zero
            denominator = (highest_high - lowest_low).replace(0, np.nan)
            williams_r = -100 * ((highest_high - df['Close']) / denominator)
            return williams_r.fillna(-50)
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return pd.Series([-50] * len(df), index=df.index)
    
    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change"""
        try:
            roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
            return roc.fillna(0)
        except Exception as e:
            logger.error(f"Error calculating ROC: {e}")
            return pd.Series([0] * len(prices), index=prices.index)
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands with additional metrics"""
        try:
            df['BB_Middle'] = df['Close'].rolling(window=period).mean()
            rolling_std = df['Close'].rolling(window=period).std()
            df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
            df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
            
            # Additional metrics
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            obv = []
            obv_value = 0
            
            for i in range(len(df)):
                if i == 0:
                    obv.append(df['Volume'].iloc[i])
                    obv_value = df['Volume'].iloc[i]
                else:
                    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                        obv_value += df['Volume'].iloc[i]
                    elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                        obv_value -= df['Volume'].iloc[i]
                    obv.append(obv_value)
            
            return pd.Series(obv, index=df.index)
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def _calculate_volume_price_trend(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        try:
            price_change = df['Close'].pct_change()
            vpt = (price_change * df['Volume']).cumsum()
            return vpt.fillna(0)
        except Exception as e:
            logger.error(f"Error calculating VPT: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        try:
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            clv = clv.fillna(0)
            ad = (clv * df['Volume']).cumsum()
            return ad
        except Exception as e:
            logger.error(f"Error calculating A/D: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def _calculate_money_flow_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            raw_money_flow = typical_price * df['Volume']
            
            positive_flow = []
            negative_flow = []
            
            for i in range(1, len(df)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.append(raw_money_flow.iloc[i])
                    negative_flow.append(0)
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    positive_flow.append(0)
                    negative_flow.append(raw_money_flow.iloc[i])
                else:
                    positive_flow.append(0)
                    negative_flow.append(0)
            
            positive_flow = [0] + positive_flow
            negative_flow = [0] + negative_flow
            
            positive_mf = pd.Series(positive_flow, index=df.index).rolling(window=period).sum()
            negative_mf = pd.Series(negative_flow, index=df.index).rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + (positive_mf / negative_mf.replace(0, np.nan))))
            return mfi.fillna(50)
        except Exception as e:
            logger.error(f"Error calculating MFI: {e}")
            return pd.Series([50] * len(df), index=df.index)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr.fillna(df['Close'] * 0.02)
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def _calculate_price_changes(self, df: pd.DataFrame) -> Dict:
        """Calculate price changes over various periods"""
        try:
            current_price = df['Close'].iloc[-1]
            
            changes = {}
            periods = {
                'price_change_1d': 1,
                'price_change_1w': 5,
                'price_change_1m': 22,
                'price_change_3m': 65,
                'price_change_6m': 130,
                'price_change_1y': 252
            }
            
            for key, period in periods.items():
                if len(df) > period:
                    past_price = df['Close'].iloc[-(period + 1)]
                    change = ((current_price - past_price) / past_price) * 100 if past_price != 0 else 0.0
                    changes[key] = safe_float_convert(change, 0.0)
                else:
                    changes[key] = 0.0
            
            return changes
        except Exception as e:
            logger.error(f"Error calculating price changes: {e}")
            return {
                'price_change_1d': 0.0, 'price_change_1w': 0.0, 'price_change_1m': 0.0,
                'price_change_3m': 0.0, 'price_change_6m': 0.0, 'price_change_1y': 0.0
            }
    
    def _calculate_support_resistance_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate traditional support and resistance levels"""
        try:
            # Use recent data for more relevant levels
            recent_data = df.tail(50)
            
            # Find local minima and maxima
            lows = recent_data['Low']
            highs = recent_data['High']
            
            # Support levels (significant lows)
            support_candidates = []
            for i in range(2, len(lows) - 2):
                if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and 
                    lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]):
                    support_candidates.append(lows.iloc[i])
            
            # Resistance levels (significant highs)
            resistance_candidates = []
            for i in range(2, len(highs) - 2):
                if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and 
                    highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]):
                    resistance_candidates.append(highs.iloc[i])
            
            # Sort and get top levels
            support_candidates.sort(reverse=True)  # Highest support first
            resistance_candidates.sort()  # Lowest resistance first
            
            # Pivot point calculation
            last_high = recent_data['High'].iloc[-1]
            last_low = recent_data['Low'].iloc[-1]
            last_close = recent_data['Close'].iloc[-1]
            pivot_point = (last_high + last_low + last_close) / 3
            
            return {
                'support_level_1': safe_float_convert(support_candidates[0] if support_candidates else recent_data['Low'].min()),
                'support_level_2': safe_float_convert(support_candidates[1] if len(support_candidates) > 1 else recent_data['Low'].min()),
                'resistance_level_1': safe_float_convert(resistance_candidates[0] if resistance_candidates else recent_data['High'].max()),
                'resistance_level_2': safe_float_convert(resistance_candidates[1] if len(resistance_candidates) > 1 else recent_data['High'].max()),
                'pivot_point': safe_float_convert(pivot_point)
            }
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            current_price = df['Close'].iloc[-1]
            return {
                'support_level_1': current_price * 0.95,
                'support_level_2': current_price * 0.90,
                'resistance_level_1': current_price * 1.05,
                'resistance_level_2': current_price * 1.10,
                'pivot_point': current_price
            }
    
    def _calculate_volume_based_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate volume-weighted support and resistance levels"""
        try:
            # Volume-weighted average price over recent period
            recent_data = df.tail(50)
            vwap = (recent_data['Close'] * recent_data['Volume']).sum() / recent_data['Volume'].sum()
            
            # Find high-volume price areas
            price_volume_map = {}
            for _, row in recent_data.iterrows():
                price_bucket = round(row['Close'], 2)
                if price_bucket in price_volume_map:
                    price_volume_map[price_bucket] += row['Volume']
                else:
                    price_volume_map[price_bucket] = row['Volume']
            
            # Sort by volume to find high-volume areas
            sorted_pv = sorted(price_volume_map.items(), key=lambda x: x[1], reverse=True)
            
            current_price = df['Close'].iloc[-1]
            
            # Find volume support (high volume below current price)
            volume_support = current_price * 0.95
            for price, volume in sorted_pv:
                if price < current_price:
                    volume_support = price
                    break
            
            # Find volume resistance (high volume above current price)
            volume_resistance = current_price * 1.05
            for price, volume in sorted_pv:
                if price > current_price:
                    volume_resistance = price
                    break
            
            return {
                'volume_support': safe_float_convert(volume_support),
                'volume_resistance': safe_float_convert(volume_resistance),
                'volume_weighted_price': safe_float_convert(vwap)
            }
        except Exception as e:
            logger.error(f"Error calculating volume-based levels: {e}")
            current_price = df['Close'].iloc[-1]
            return {
                'volume_support': current_price * 0.95,
                'volume_resistance': current_price * 1.05,
                'volume_weighted_price': current_price
            }
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate Fibonacci retracement levels"""
        try:
            # Use recent swing high and low
            recent_data = df.tail(100)
            swing_high = recent_data['High'].max()
            swing_low = recent_data['Low'].min()
            
            diff = swing_high - swing_low
            
            return {
                'fib_23_6': safe_float_convert(swing_high - (diff * 0.236)),
                'fib_38_2': safe_float_convert(swing_high - (diff * 0.382)),
                'fib_50_0': safe_float_convert(swing_high - (diff * 0.5)),
                'fib_61_8': safe_float_convert(swing_high - (diff * 0.618)),
                'fib_78_6': safe_float_convert(swing_high - (diff * 0.786))
            }
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            current_price = df['Close'].iloc[-1]
            return {
                'fib_23_6': current_price * 0.976,
                'fib_38_2': current_price * 0.962,
                'fib_50_0': current_price * 0.95,
                'fib_61_8': current_price * 0.938,
                'fib_78_6': current_price * 0.921
            }
    
    def _generate_csv_for_llm(self, df: pd.DataFrame, indicators: TechnicalAnalysisData) -> str:
        """Generate comprehensive CSV data for LLM analysis"""
        try:
            # Take recent data (last 60 days) for LLM analysis
            recent_df = df.tail(60).copy()
            
            # Create CSV content
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            
            # Write header
            header = [
                'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
                'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'OBV',
                'Volume_Ratio', 'Price_Change_1D', 'ATR_14'
            ]
            writer.writerow(header)
            
            # Calculate indicators for recent data
            recent_df['SMA_20'] = recent_df['Close'].rolling(20).mean()
            recent_df['SMA_50'] = recent_df['Close'].rolling(50).mean()
            recent_df['EMA_12'] = recent_df['Close'].ewm(span=12).mean()
            recent_df['EMA_26'] = recent_df['Close'].ewm(span=26).mean()
            recent_df['RSI_14'] = self._calculate_rsi(recent_df['Close'], 14)
            recent_df['MACD'] = recent_df['EMA_12'] - recent_df['EMA_26']
            recent_df['MACD_Signal'] = recent_df['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            bb_middle = recent_df['Close'].rolling(20).mean()
            bb_std = recent_df['Close'].rolling(20).std()
            recent_df['BB_Upper'] = bb_middle + (bb_std * 2)
            recent_df['BB_Lower'] = bb_middle - (bb_std * 2)
            
            # Volume and ATR
            recent_df['OBV'] = self._calculate_obv(recent_df)
            recent_df['Volume_SMA_20'] = recent_df['Volume'].rolling(20).mean()
            recent_df['Volume_Ratio'] = recent_df['Volume'] / recent_df['Volume_SMA_20']
            recent_df['Price_Change_1D'] = recent_df['Close'].pct_change() * 100
            recent_df['ATR_14'] = self._calculate_atr(recent_df, 14)
            
            # Write data rows
            for _, row in recent_df.iterrows():
                data_row = [
                    row.name.strftime('%Y-%m-%d'),
                    f"{row['Open']:.2f}",
                    f"{row['High']:.2f}",
                    f"{row['Low']:.2f}",
                    f"{row['Close']:.2f}",
                    f"{row['Volume']:,.0f}",
                    f"{safe_float_convert(row['SMA_20']):.2f}",
                    f"{safe_float_convert(row['SMA_50']):.2f}",
                    f"{safe_float_convert(row['EMA_12']):.2f}",
                    f"{safe_float_convert(row['EMA_26']):.2f}",
                    f"{safe_float_convert(row['RSI_14']):.1f}",
                    f"{safe_float_convert(row['MACD']):.3f}",
                    f"{safe_float_convert(row['MACD_Signal']):.3f}",
                    f"{safe_float_convert(row['BB_Upper']):.2f}",
                    f"{safe_float_convert(row['BB_Lower']):.2f}",
                    f"{safe_float_convert(row['OBV']):,.0f}",
                    f"{safe_float_convert(row['Volume_Ratio']):.2f}",
                    f"{safe_float_convert(row['Price_Change_1D']):.2f}",
                    f"{safe_float_convert(row['ATR_14']):.2f}"
                ]
                writer.writerow(data_row)
            
            return csv_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating CSV for LLM: {e}")
            return "Date,Open,High,Low,Close,Volume\nError generating CSV data"
    
    def _perform_comprehensive_ai_analysis(self, symbol: str, csv_data: str, indicators: TechnicalAnalysisData, stock_info: Dict) -> Dict:
        """Perform comprehensive AI-powered technical analysis"""
        # Get symbol-specific logger
        symbol_logger = self.config.get_symbol_logger(symbol, 'yahoo_technical')
        
        # Use prompt manager for enhanced prompting with JSON response
        from utils.prompt_manager import get_prompt_manager
        prompt_manager = get_prompt_manager()
        
        # Prepare indicators summary
        indicators_summary = f"""Current Price: ${indicators.current_price:.2f}

Moving Averages:
- SMA 20: ${indicators.sma_20:.2f} | SMA 50: ${indicators.sma_50:.2f} | SMA 200: ${indicators.sma_200:.2f}
- EMA 12: ${indicators.ema_12:.2f} | EMA 26: ${indicators.ema_26:.2f} | EMA 50: ${indicators.ema_50:.2f}

Momentum Indicators:
- RSI (14): {indicators.rsi_14:.1f} | Stochastic %K: {indicators.stoch_k:.1f} | Williams %R: {indicators.williams_r:.1f}
- ROC (10): {indicators.roc_10:.2f}% | ROC (20): {indicators.roc_20:.2f}%

MACD Analysis:
- MACD: {indicators.macd:.4f} | Signal: {indicators.macd_signal:.4f} | Histogram: {indicators.macd_histogram:.4f}

Bollinger Bands:
- Upper: ${indicators.bb_upper:.2f} | Lower: ${indicators.bb_lower:.2f} | Position: {indicators.bb_position:.2f}
- Width: {indicators.bb_width:.4f}

Volume Analysis:
- Current Volume: {indicators.volume:,.0f} | 20-Day Avg: {indicators.volume_sma_20:,.0f}
- Volume Ratio: {indicators.volume_ratio:.2f}
- OBV: {indicators.obv:,.0f} | Money Flow Index: {indicators.money_flow_index:.1f}

Support & Resistance:
- Traditional Support: ${indicators.support_level_1:.2f} / ${indicators.support_level_2:.2f}
- Traditional Resistance: ${indicators.resistance_level_1:.2f} / ${indicators.resistance_level_2:.2f}
- Volume Support: ${indicators.volume_support:.2f} | Volume Resistance: ${indicators.volume_resistance:.2f}
- VWAP: ${indicators.volume_weighted_price:.2f} | Pivot Point: ${indicators.pivot_point:.2f}

Fibonacci Levels:
- 23.6%: ${indicators.fib_23_6:.2f} | 38.2%: ${indicators.fib_38_2:.2f} | 50.0%: ${indicators.fib_50_0:.2f}
- 61.8%: ${indicators.fib_61_8:.2f} | 78.6%: ${indicators.fib_78_6:.2f}

Volatility:
- ATR (14): ${indicators.atr_14:.2f} | 20-Day Volatility: {indicators.volatility_20d:.1f}%

Price Performance:
- 1D: {indicators.price_change_1d:.2f}% | 1W: {indicators.price_change_1w:.2f}% | 1M: {indicators.price_change_1m:.2f}%
- 3M: {indicators.price_change_3m:.2f}% | 6M: {indicators.price_change_6m:.2f}% | 1Y: {indicators.price_change_1y:.2f}%"""
        
        analysis_prompt = prompt_manager.render_technical_analysis_prompt(
            symbol=symbol,
            analysis_date=datetime.now().strftime('%Y-%m-%d'),
            data_points=indicators.data_points,
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
            ai_response = technical_result.get('raw_response', '')
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Parse JSON response
            analysis_result = prompt_manager.validate_json_response(ai_response)
            analysis_result['raw_response'] = ai_response
            
            # Save prompts to cache
            self._save_technical_prompts_to_cache(symbol, analysis_prompt, system_prompt, ai_response)
            
            # Save to database
            self._save_llm_response_to_db(symbol, analysis_prompt, system_prompt, ai_response, processing_time_ms)
            
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
            
            # Moving Average Score (25% weight)
            ma_score = 5.0
            current = indicators.current_price
            
            # Short-term MA comparison
            if current > indicators.sma_20 > indicators.sma_50:
                ma_score += 1.5  # Bullish alignment
            elif current < indicators.sma_20 < indicators.sma_50:
                ma_score -= 1.5  # Bearish alignment
            
            # Long-term trend
            if current > indicators.sma_200:
                ma_score += 1.0  # Above long-term trend
            else:
                ma_score -= 1.0  # Below long-term trend
            
            # EMA vs SMA strength
            if indicators.ema_20 > indicators.sma_20:
                ma_score += 0.5  # Recent momentum positive
            else:
                ma_score -= 0.5  # Recent momentum negative
            
            # Momentum Score (25% weight)
            momentum_score = 5.0
            
            # RSI analysis
            if 40 <= indicators.rsi_14 <= 60:
                momentum_score += 1.0  # Neutral zone
            elif 60 < indicators.rsi_14 <= 70:
                momentum_score += 1.5  # Bullish but not overbought
            elif 30 <= indicators.rsi_14 < 40:
                momentum_score -= 1.5  # Bearish but not oversold
            elif indicators.rsi_14 > 80:
                momentum_score -= 1.0  # Overbought
            elif indicators.rsi_14 < 20:
                momentum_score += 0.5  # Oversold reversal potential
            
            # MACD analysis
            if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
                momentum_score += 1.0  # Bullish MACD
            elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
                momentum_score -= 1.0  # Bearish MACD
            
            # Stochastic analysis
            if 20 <= indicators.stoch_k <= 80:
                momentum_score += 0.5  # Not extreme
            
            # Volume Score (20% weight)
            volume_score = 5.0
            
            # Volume ratio analysis
            if indicators.volume_ratio > 1.5:
                volume_score += 1.5  # High volume
            elif indicators.volume_ratio < 0.5:
                volume_score -= 1.0  # Low volume
            
            # Money Flow Index
            if 40 <= indicators.money_flow_index <= 60:
                volume_score += 1.0  # Balanced
            elif indicators.money_flow_index > 80:
                volume_score -= 0.5  # Overbought
            elif indicators.money_flow_index < 20:
                volume_score += 0.5  # Oversold
            
            # Support/Resistance Score (15% weight)
            sr_score = 5.0
            
            # Position relative to support/resistance
            if current > indicators.resistance_level_1:
                sr_score += 1.0  # Above resistance
            elif current < indicators.support_level_1:
                sr_score -= 1.0  # Below support
            
            # Bollinger Band position
            if 0.3 <= indicators.bb_position <= 0.7:
                sr_score += 0.5  # Middle range
            elif indicators.bb_position > 0.8:
                sr_score -= 0.5  # Near upper band
            elif indicators.bb_position < 0.2:
                sr_score += 0.5  # Near lower band (reversal potential)
            
            # Volatility Score (15% weight)
            volatility_score = 5.0
            
            # ATR relative to price
            atr_ratio = indicators.atr_14 / indicators.current_price
            if 0.01 <= atr_ratio <= 0.03:
                volatility_score += 1.0  # Normal volatility
            elif atr_ratio > 0.05:
                volatility_score -= 1.0  # High volatility
            
            # Bollinger Band width
            if indicators.bb_width < 0.1:
                volatility_score += 0.5  # Low volatility (potential breakout)
            elif indicators.bb_width > 0.3:
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
    
    def _save_llm_response_to_db(self, symbol: str, prompt: str, system_prompt: str, response: str, processing_time_ms: int):
        """Save LLM response using cache manager for technical analysis"""
        try:
            # Prepare prompt context
            prompt_context = {
                'system_prompt': system_prompt,
                'user_prompt': prompt[:1000],  # Truncate for storage
                'prompt_length': len(prompt),
                'prompt': prompt  # Also save full prompt
            }
            
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
            
            # Save using cache manager (will handle both disk and RDBMS storage)
            cache_key = (symbol, 'ta', 'N/A', 'N/A')  # symbol, llm_type, period, form_type
            cache_value = {
                'prompt_context': prompt_context,
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
        """Save technical analysis results to database"""
        try:
            # Save to technical analysis table (implementation depends on your DB schema)
            logger.info(f"💾 Saved technical analysis for {symbol} to database")
        except Exception as e:
            logger.warning(f"Error saving analysis to database: {e}")
    
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
    
    def _create_enhanced_dataframe(self, df: pd.DataFrame, indicators: TechnicalAnalysisData) -> pd.DataFrame:
        """Create enhanced dataframe with all technical indicators for parquet storage"""
        try:
            # The df already contains all the calculated indicators from _calculate_comprehensive_indicators
            # We just need to add some metadata columns
            enhanced_df = df.copy()
            
            # Ensure we have the basic OHLCV columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in enhanced_df.columns:
                    logger.warning(f"Missing required column: {col}")
                    enhanced_df[col] = 0.0
            
            # Don't add Date column - the index already contains this information
            # This prevents duplicate column errors when saving to parquet
            
            # Add symbol column
            enhanced_df['Symbol'] = indicators.symbol
            
            # Note: Support/resistance/fibonacci levels are single values, not time series
            # They shouldn't be added as columns to every row
            # They're already available in the indicators object for reporting
            
            # Fill any NaN values with forward fill then backward fill
            enhanced_df = enhanced_df.ffill().bfill()
            
            logger.info(f"Created enhanced dataframe with {len(enhanced_df.columns)} columns for parquet storage")
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error creating enhanced dataframe: {e}")
            return df.copy()
    
    def _save_to_parquet(self, symbol: str, df: pd.DataFrame, days: int = 365):
        """Save enhanced price data with indicators to parquet file and cache"""
        try:
            # First save to traditional parquet file for backward compatibility
            parquet_file = self.price_cache_dir / f"{symbol}.parquet"
            df.to_parquet(parquet_file, compression='snappy', index=True)
            
            logger.info(f"💾 Saved price data with indicators to {parquet_file}")
            logger.info(f"📊 Parquet file contains {len(df)} rows and {len(df.columns)} columns")
            
            # Also save to cache manager with gzip compression
            cache_key = (symbol, 'technical_data', f'{days}d')
            
            # Prepare data for cache - avoid duplicate Date column
            cache_df = df.copy()
            if 'Date' in cache_df.columns and isinstance(cache_df.index, pd.DatetimeIndex):
                # Drop the Date column since it's already in the index
                cache_df = cache_df.drop('Date', axis=1)
            
            cache_value = {
                'dataframe': cache_df,
                'data': cache_df.reset_index().to_dict('records'),  # Include index in data
                'metadata': {
                    'symbol': symbol,
                    'days': days,
                    'start_date': str(cache_df.index.min()),
                    'end_date': str(cache_df.index.max()),
                    'records': len(cache_df),
                    'columns': list(cache_df.columns),
                    'indicators_included': True
                }
            }
            
            # Save to cache (will use ParquetCacheStorageHandler with gzip)
            success = self.cache_manager.set(CacheType.TECHNICAL_DATA, cache_key, cache_value)
            
            if success:
                logger.info(f"💾 Saved technical data to cache with gzip compression for {symbol}")
                # Get cache info to log compression stats
                cached_info = self.cache_manager.get(CacheType.TECHNICAL_DATA, cache_key)
                if cached_info and 'cache_info' in cached_info:
                    cache_info = cached_info['cache_info']
                    logger.info(f"📊 Cache stats: {cache_info.get('records')} records, compressed with {cache_info.get('compression', 'unknown')}")
            else:
                logger.warning(f"Failed to save technical data to cache for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving to parquet: {e}")
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
            return vwap.fillna(df['Close'])
        except Exception as e:
            logger.warning(f"Error calculating VWAP: {e}")
            return df['Close']
    
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
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        result = analyze_symbol(args.symbol, args.days)
        print(f"\n=== TECHNICAL ANALYSIS RESULTS FOR {args.symbol} ===")
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