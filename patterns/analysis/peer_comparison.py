#!/usr/bin/env python3
"""
Peer Group Comparison Module for InvestiGator
Calculates financial ratios and performs peer group analysis
"""

import json
import logging
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

# Removed cache facade - using cache manager directly
from utils.cache.cache_manager import get_cache_manager
from utils.peer_metrics_dao import get_peer_metrics_dao
from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class FinancialRatios:
    """Financial ratios for a company"""
    symbol: str
    # Profitability Ratios
    eps: Optional[float] = None
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    profit_margin: Optional[float] = None
    
    # Leverage Ratios
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    interest_coverage: Optional[float] = None
    
    # Valuation Ratios
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    
    # Growth Metrics
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    revenue_growth_3yr: Optional[float] = None
    earnings_growth_3yr: Optional[float] = None
    
    # Market Metrics
    beta: Optional[float] = None
    correlation_sp500: Optional[float] = None
    volatility_30d: Optional[float] = None
    volatility_90d: Optional[float] = None
    
    # Investment Style
    growth_score: Optional[float] = None
    value_score: Optional[float] = None
    quality_score: Optional[float] = None
    
    # Additional Context
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    shares_outstanding: Optional[float] = None
    last_updated: Optional[str] = None


@dataclass
class PeerGroupStats:
    """Statistical summary of peer group metrics"""
    metric_name: str
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    percentile_25: Optional[float] = None
    percentile_75: Optional[float] = None
    count: int = 0


class PeerGroupComparison:
    """Handles peer group comparison and financial ratio analysis"""
    
    def __init__(self):
        self.config = get_config()
        self.cache_manager = get_cache_manager()
        self.peer_metrics_dao = get_peer_metrics_dao()
        self._load_peer_groups()
        
    def _load_peer_groups(self):
        """Load peer group definitions from file"""
        try:
            with open('data/russell_1000_peer_groups.json', 'r') as f:
                self.peer_groups = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load peer groups: {e}")
            self.peer_groups = {}
    
    def get_peer_group(self, symbol: str) -> Dict[str, Any]:
        """Get peer group for a given symbol"""
        symbol = symbol.upper()
        
        # Search through peer_groups
        peer_groups = self.peer_groups.get('peer_groups', {})
        
        for sector_name, industries in peer_groups.items():
            for industry_name, industry_data in industries.items():
                # Check in large_cap
                large_cap = industry_data.get('large_cap', [])
                if symbol in large_cap:
                    all_peers = large_cap + industry_data.get('mid_cap', [])
                    return {
                        'sector': sector_name.replace('_', ' ').title(),
                        'industry': industry_name.replace('_', ' ').title(),
                        'peers': all_peers,
                        'large_cap': large_cap,
                        'mid_cap': industry_data.get('mid_cap', [])
                    }
                
                # Check in mid_cap
                mid_cap = industry_data.get('mid_cap', [])
                if symbol in mid_cap:
                    all_peers = large_cap + mid_cap
                    return {
                        'sector': sector_name.replace('_', ' ').title(),
                        'industry': industry_name.replace('_', ' ').title(),
                        'peers': all_peers,
                        'large_cap': large_cap,
                        'mid_cap': mid_cap
                    }
        
        return {'sector': None, 'industry': None, 'peers': []}
    
    def _generate_peer_group_id(self, peer_info: Dict[str, Any]) -> str:
        """Generate a unique peer group ID from peer info"""
        sector = peer_info.get('sector', 'UNKNOWN')
        industry = peer_info.get('industry', 'UNKNOWN')
        # Clean and format the ID
        sector_clean = sector.upper().replace(' ', '_').replace('&', 'AND')
        industry_clean = industry.upper().replace(' ', '_').replace('&', 'AND')
        return f"{sector_clean}_{industry_clean}"
    
    def calculate_financial_ratios(self, symbol: str, force_refresh: bool = False) -> Optional[FinancialRatios]:
        """Calculate comprehensive financial ratios for a company"""
        # Get peer group info for this symbol
        peer_info = self.get_peer_group(symbol)
        peer_group_id = self._generate_peer_group_id(peer_info)
        
        # Check DAO cache first
        if not force_refresh:
            cached_data = self.peer_metrics_dao.get_peer_metrics(
                peer_group_id, symbol, 'financial_ratios'
            )
            if cached_data:
                logger.info(f"Using cached financial ratios for {symbol}")
                metrics_data = cached_data.get('metrics_data', {})
                if 'ratios' in metrics_data:
                    return FinancialRatios(**metrics_data['ratios'])
        
        logger.info(f"Calculating financial ratios for {symbol}")
        
        try:
            # Get financial data from SEC
            sec_data = self._get_sec_financial_data(symbol)
            if not sec_data:
                logger.warning(f"No SEC data available for {symbol}")
                return None
                
            # Get market data from Yahoo Finance
            market_data = self._get_market_data(symbol)
            
            # Calculate ratios
            ratios = FinancialRatios(symbol=symbol)
            
            # Profitability ratios
            ratios.eps = sec_data.get('eps')
            if market_data and ratios.eps and ratios.eps > 0:
                ratios.pe_ratio = market_data.get('price', 0) / ratios.eps
            
            ratios.roe = self._safe_divide(sec_data.get('net_income'), sec_data.get('shareholders_equity'))
            ratios.roa = self._safe_divide(sec_data.get('net_income'), sec_data.get('total_assets'))
            ratios.profit_margin = self._safe_divide(sec_data.get('net_income'), sec_data.get('revenue'))
            
            # Leverage ratios
            ratios.debt_to_equity = self._safe_divide(sec_data.get('total_debt'), sec_data.get('shareholders_equity'))
            ratios.debt_to_assets = self._safe_divide(sec_data.get('total_debt'), sec_data.get('total_assets'))
            ratios.interest_coverage = self._safe_divide(sec_data.get('ebit'), sec_data.get('interest_expense'))
            
            # Valuation ratios
            if market_data:
                ratios.price_to_book = self._safe_divide(
                    market_data.get('market_cap'),
                    sec_data.get('shareholders_equity')
                )
                ratios.price_to_sales = self._safe_divide(
                    market_data.get('market_cap'),
                    sec_data.get('revenue')
                )
                ratios.ev_to_ebitda = self._safe_divide(
                    market_data.get('enterprise_value'),
                    sec_data.get('ebitda')
                )
                
                # Market metrics
                ratios.beta = market_data.get('beta')
                ratios.market_cap = market_data.get('market_cap')
                ratios.enterprise_value = market_data.get('enterprise_value')
            
            # Growth metrics
            ratios.revenue_growth_yoy = sec_data.get('revenue_growth_yoy')
            ratios.earnings_growth_yoy = sec_data.get('earnings_growth_yoy')
            ratios.revenue_growth_3yr = sec_data.get('revenue_growth_3yr')
            ratios.earnings_growth_3yr = sec_data.get('earnings_growth_3yr')
            
            # Calculate investment style scores
            ratios.growth_score = self._calculate_growth_score(ratios)
            ratios.value_score = self._calculate_value_score(ratios)
            ratios.quality_score = self._calculate_quality_score(ratios)
            
            # Calculate volatility and correlation
            if market_data and 'price_history' in market_data:
                ratios.volatility_30d = self._calculate_volatility(market_data['price_history'], 30)
                ratios.volatility_90d = self._calculate_volatility(market_data['price_history'], 90)
                ratios.correlation_sp500 = self._calculate_sp500_correlation(symbol)
            
            ratios.last_updated = datetime.now().isoformat()
            
            # Save to database using DAO
            metrics_data = {
                'ratios': asdict(ratios),
                'timestamp': datetime.now().isoformat()
            }
            
            self.peer_metrics_dao.save_peer_metrics(
                peer_group_id=peer_group_id,
                symbol=symbol,
                metric_type='financial_ratios',
                metrics_data=metrics_data,
                sector=peer_info.get('sector'),
                industry=peer_info.get('industry'),
                peer_symbols=peer_info.get('peers', [])[:10]  # Store up to 10 peers
            )
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating ratios for {symbol}: {e}")
            return None
    
    def _get_sec_financial_data(self, symbol: str) -> Dict[str, Any]:
        """Get financial data from SEC filings"""
        try:
            # For now, return mock data - in production this would query the database
            # or use the financial data aggregator when available
            quarterly_data = []
            if not quarterly_data:
                return {}
            
            latest_quarter = quarterly_data[0]
            year_ago_quarter = quarterly_data[-1] if len(quarterly_data) >= 4 else None
            
            # Extract key financial metrics
            financial_data = {
                'revenue': latest_quarter.get('revenues'),
                'net_income': latest_quarter.get('net_income'),
                'total_assets': latest_quarter.get('total_assets'),
                'total_debt': latest_quarter.get('long_term_debt', 0) + latest_quarter.get('short_term_debt', 0),
                'shareholders_equity': latest_quarter.get('stockholders_equity'),
                'eps': latest_quarter.get('earnings_per_share'),
                'ebit': latest_quarter.get('operating_income'),
                'ebitda': latest_quarter.get('ebitda'),
                'interest_expense': latest_quarter.get('interest_expense'),
                'shares_outstanding': latest_quarter.get('shares_outstanding')
            }
            
            # Calculate growth metrics
            if year_ago_quarter:
                if year_ago_quarter.get('revenues') and latest_quarter.get('revenues'):
                    financial_data['revenue_growth_yoy'] = (
                        (latest_quarter['revenues'] - year_ago_quarter['revenues']) / 
                        year_ago_quarter['revenues']
                    )
                
                if year_ago_quarter.get('net_income') and latest_quarter.get('net_income'):
                    financial_data['earnings_growth_yoy'] = (
                        (latest_quarter['net_income'] - year_ago_quarter['net_income']) / 
                        abs(year_ago_quarter['net_income'])
                    )
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error getting SEC data for {symbol}: {e}")
            return {}
    
    def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get price history for volatility calculation
            history = ticker.history(period="3mo")
            
            market_data = {
                'price': info.get('currentPrice') or info.get('previousClose'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'beta': info.get('beta'),
                'price_history': history['Close'].to_dict() if not history.empty else {}
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}
    
    def _calculate_volatility(self, price_history: Dict, days: int) -> Optional[float]:
        """Calculate historical volatility"""
        try:
            if not price_history:
                return None
                
            # Convert to series and calculate returns
            prices = pd.Series(price_history).sort_index()
            prices = prices.last(f'{days}D')
            
            if len(prices) < 2:
                return None
                
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return None
    
    def _calculate_sp500_correlation(self, symbol: str) -> Optional[float]:
        """Calculate correlation with S&P 500"""
        try:
            # Get stock and SPY data
            stock = yf.Ticker(symbol)
            spy = yf.Ticker("SPY")
            
            # Get 1 year of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            stock_data = stock.history(start=start_date, end=end_date)
            spy_data = spy.history(start=start_date, end=end_date)
            
            if stock_data.empty or spy_data.empty:
                return None
            
            # Calculate daily returns
            stock_returns = stock_data['Close'].pct_change().dropna()
            spy_returns = spy_data['Close'].pct_change().dropna()
            
            # Align the data
            aligned_data = pd.DataFrame({
                'stock': stock_returns,
                'spy': spy_returns
            }).dropna()
            
            if len(aligned_data) < 20:
                return None
            
            correlation = aligned_data['stock'].corr(aligned_data['spy'])
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating S&P 500 correlation for {symbol}: {e}")
            return None
    
    def _calculate_growth_score(self, ratios: FinancialRatios) -> Optional[float]:
        """Calculate growth score (0-10)"""
        scores = []
        
        # Revenue growth
        if ratios.revenue_growth_yoy is not None:
            scores.append(min(10, max(0, ratios.revenue_growth_yoy * 50)))
        
        # Earnings growth
        if ratios.earnings_growth_yoy is not None:
            scores.append(min(10, max(0, ratios.earnings_growth_yoy * 40)))
        
        # PEG ratio (lower is better for growth at reasonable price)
        if ratios.peg_ratio is not None and ratios.peg_ratio > 0:
            peg_score = 10 - min(10, ratios.peg_ratio * 2)
            scores.append(peg_score)
        
        return statistics.mean(scores) if scores else None
    
    def _calculate_value_score(self, ratios: FinancialRatios) -> Optional[float]:
        """Calculate value score (0-10)"""
        scores = []
        
        # P/E ratio (lower is better)
        if ratios.pe_ratio is not None and ratios.pe_ratio > 0:
            pe_score = 10 - min(10, ratios.pe_ratio / 3)
            scores.append(pe_score)
        
        # P/B ratio (lower is better)
        if ratios.price_to_book is not None and ratios.price_to_book > 0:
            pb_score = 10 - min(10, ratios.price_to_book / 0.5)
            scores.append(pb_score)
        
        # EV/EBITDA (lower is better)
        if ratios.ev_to_ebitda is not None and ratios.ev_to_ebitda > 0:
            ev_score = 10 - min(10, ratios.ev_to_ebitda / 2)
            scores.append(ev_score)
        
        return statistics.mean(scores) if scores else None
    
    def _calculate_quality_score(self, ratios: FinancialRatios) -> Optional[float]:
        """Calculate quality score (0-10)"""
        scores = []
        
        # ROE
        if ratios.roe is not None:
            scores.append(min(10, max(0, ratios.roe * 50)))
        
        # Profit margin
        if ratios.profit_margin is not None:
            scores.append(min(10, max(0, ratios.profit_margin * 40)))
        
        # Debt to equity (lower is better)
        if ratios.debt_to_equity is not None:
            de_score = 10 - min(10, ratios.debt_to_equity * 2)
            scores.append(de_score)
        
        # Interest coverage (higher is better)
        if ratios.interest_coverage is not None and ratios.interest_coverage > 0:
            ic_score = min(10, ratios.interest_coverage)
            scores.append(ic_score)
        
        return statistics.mean(scores) if scores else None
    
    def _safe_divide(self, numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """Safely divide two numbers"""
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator
    
    def get_peer_comparison(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get comprehensive peer comparison analysis"""
        logger.info(f"Getting peer comparison for {symbol}")
        
        # Get peer group
        peer_info = self.get_peer_group(symbol)
        if not peer_info['peers']:
            logger.warning(f"No peer group found for {symbol}")
            return {
                'symbol': symbol,
                'peer_group': None,
                'company_ratios': None,
                'peer_statistics': None,
                'relative_position': None
            }
        
        # Calculate ratios for target company
        company_ratios = self.calculate_financial_ratios(symbol, force_refresh)
        if not company_ratios:
            logger.warning(f"Could not calculate ratios for {symbol}")
            return {
                'symbol': symbol,
                'peer_group': peer_info,
                'company_ratios': None,
                'peer_statistics': None,
                'relative_position': None
            }
        
        # Calculate ratios for peers (limit to top peers to avoid overload)
        peer_ratios = []
        peers_to_analyze = peer_info['peers'][:10]  # Limit to 10 peers
        
        for peer_symbol in peers_to_analyze:
            if peer_symbol == symbol:
                continue
            peer_ratio = self.calculate_financial_ratios(peer_symbol, force_refresh)
            if peer_ratio:
                peer_ratios.append(peer_ratio)
        
        # Calculate peer group statistics
        peer_statistics = self._calculate_peer_statistics(peer_ratios)
        
        # Calculate relative position
        relative_position = self._calculate_relative_position(company_ratios, peer_ratios)
        
        return {
            'symbol': symbol,
            'peer_group': peer_info,
            'company_ratios': asdict(company_ratios),
            'peer_statistics': peer_statistics,
            'relative_position': relative_position,
            'peers_analyzed': len(peer_ratios),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_peer_statistics(self, peer_ratios: List[FinancialRatios]) -> Dict[str, PeerGroupStats]:
        """Calculate statistical summaries for peer group metrics"""
        if not peer_ratios:
            return {}
        
        # Define metrics to analyze
        metrics = [
            'pe_ratio', 'peg_ratio', 'roe', 'roa', 'profit_margin',
            'debt_to_equity', 'debt_to_assets', 'price_to_book',
            'price_to_sales', 'ev_to_ebitda', 'revenue_growth_yoy',
            'earnings_growth_yoy', 'beta', 'growth_score', 'value_score',
            'quality_score'
        ]
        
        statistics_dict = {}
        
        for metric in metrics:
            values = [getattr(r, metric) for r in peer_ratios if getattr(r, metric) is not None]
            
            if values:
                stats = PeerGroupStats(metric_name=metric)
                stats.count = len(values)
                stats.mean = statistics.mean(values)
                stats.median = statistics.median(values)
                stats.std_dev = statistics.stdev(values) if len(values) > 1 else 0
                stats.min_val = min(values)
                stats.max_val = max(values)
                stats.percentile_25 = np.percentile(values, 25)
                stats.percentile_75 = np.percentile(values, 75)
                
                statistics_dict[metric] = asdict(stats)
        
        return statistics_dict
    
    def _calculate_relative_position(self, company_ratios: FinancialRatios, 
                                   peer_ratios: List[FinancialRatios]) -> Dict[str, Any]:
        """Calculate company's relative position vs peers"""
        if not peer_ratios:
            return {}
        
        relative_position = {}
        
        # Valuation metrics (lower percentile = more attractive)
        valuation_metrics = ['pe_ratio', 'price_to_book', 'price_to_sales', 'ev_to_ebitda']
        
        # Performance metrics (higher percentile = better)
        performance_metrics = ['roe', 'roa', 'profit_margin', 'revenue_growth_yoy', 
                             'earnings_growth_yoy', 'growth_score', 'quality_score']
        
        # Risk metrics (context dependent)
        risk_metrics = ['debt_to_equity', 'debt_to_assets', 'beta', 'volatility_90d']
        
        for metric in valuation_metrics + performance_metrics + risk_metrics:
            company_value = getattr(company_ratios, metric)
            if company_value is not None:
                peer_values = [getattr(r, metric) for r in peer_ratios 
                             if getattr(r, metric) is not None]
                
                if peer_values:
                    all_values = peer_values + [company_value]
                    percentile = stats.percentileofscore(all_values, company_value)
                    
                    # Determine if higher or lower is better
                    if metric in valuation_metrics:
                        attractiveness = 100 - percentile  # Lower is better
                    elif metric in performance_metrics:
                        attractiveness = percentile  # Higher is better
                    else:
                        attractiveness = 50 + (50 - abs(percentile - 50))  # Middle is better
                    
                    relative_position[metric] = {
                        'value': company_value,
                        'percentile': percentile,
                        'attractiveness': attractiveness,
                        'peer_median': statistics.median(peer_values),
                        'interpretation': self._interpret_percentile(metric, percentile)
                    }
        
        # Overall scores
        valuation_scores = [relative_position[m]['attractiveness'] 
                          for m in valuation_metrics if m in relative_position]
        performance_scores = [relative_position[m]['attractiveness'] 
                            for m in performance_metrics if m in relative_position]
        
        relative_position['overall_valuation_score'] = (
            statistics.mean(valuation_scores) if valuation_scores else None
        )
        relative_position['overall_performance_score'] = (
            statistics.mean(performance_scores) if performance_scores else None
        )
        
        return relative_position
    
    def _interpret_percentile(self, metric: str, percentile: float) -> str:
        """Interpret percentile position for a metric"""
        if percentile >= 80:
            position = "top quintile"
        elif percentile >= 60:
            position = "second quintile"
        elif percentile >= 40:
            position = "middle quintile"
        elif percentile >= 20:
            position = "fourth quintile"
        else:
            position = "bottom quintile"
        
        # Metric-specific interpretations
        if metric in ['pe_ratio', 'price_to_book', 'price_to_sales', 'ev_to_ebitda']:
            if percentile < 30:
                return f"{position} (attractive valuation)"
            elif percentile > 70:
                return f"{position} (expensive valuation)"
            else:
                return f"{position} (fair valuation)"
        
        elif metric in ['roe', 'roa', 'profit_margin']:
            if percentile > 70:
                return f"{position} (strong profitability)"
            elif percentile < 30:
                return f"{position} (weak profitability)"
            else:
                return f"{position} (moderate profitability)"
        
        elif metric in ['debt_to_equity', 'debt_to_assets']:
            if percentile < 30:
                return f"{position} (low leverage)"
            elif percentile > 70:
                return f"{position} (high leverage)"
            else:
                return f"{position} (moderate leverage)"
        
        else:
            return position
    


def get_peer_comparison_analyzer() -> PeerGroupComparison:
    """Factory function to get peer comparison analyzer instance"""
    return PeerGroupComparison()