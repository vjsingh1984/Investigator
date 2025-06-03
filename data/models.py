#!/usr/bin/env python3
"""
InvestiGator - Data Models Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Consolidated Data Models for InvestiGator
Unified models to eliminate duplication across the codebase
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime, date


@dataclass
class FinancialStatementData:
    """Comprehensive financial statement data from SEC Frame API"""
    
    # Company identifiers
    symbol: str
    cik: str
    fiscal_year: int
    fiscal_period: str
    
    # Raw financial statement data
    income_statement: Dict[str, Any] = field(default_factory=dict)
    balance_sheet: Dict[str, Any] = field(default_factory=dict)  
    cash_flow_statement: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced category-based data
    quarterly_data: Dict[str, Any] = field(default_factory=dict)
    comprehensive_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    form_type: Optional[str] = None
    filing_date: Optional[date] = None
    period_end_date: Optional[date] = None
    accession_number: Optional[str] = None
    data_quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_period_key(self) -> str:
        """Get standardized period key"""
        return f"{self.fiscal_year}-{self.fiscal_period}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'cik': self.cik,
            'fiscal_year': self.fiscal_year,
            'fiscal_period': self.fiscal_period,
            'period_key': self.get_period_key(),
            'form_type': self.form_type,
            'filing_date': self.filing_date.isoformat() if self.filing_date and hasattr(self.filing_date, 'isoformat') else str(self.filing_date) if self.filing_date else None,
            'income_statement': self.income_statement,
            'balance_sheet': self.balance_sheet,
            'cash_flow_statement': self.cash_flow_statement,
            'quarterly_data': self.quarterly_data,
            'comprehensive_data': self.comprehensive_data,
            'data_quality_score': self.data_quality_score,
            'metadata': self.metadata
        }


@dataclass
class QuarterlyData:
    """Unified quarterly data model with AI analysis capabilities"""
    
    # Core identifiers
    symbol: str
    cik: str
    fiscal_year: int
    fiscal_period: str
    form_type: str
    
    # Financial data (comprehensive)
    financial_data: FinancialStatementData
    
    # AI Analysis results
    ai_summary: Optional[str] = None
    ai_scores: Dict[str, float] = field(default_factory=dict)
    ai_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Processing status
    processing_status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
    
    # Filing metadata
    filing_date: Optional[date] = None
    accession_number: Optional[str] = None
    period_end_date: Optional[date] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def period_key(self) -> str:
        """Get standardized period key"""
        return f"{self.fiscal_year}-{self.fiscal_period}"
    
    @property
    def ticker(self) -> str:
        """Alias for symbol (for backwards compatibility)"""
        return self.symbol
    
    def add_category_data(self, category: str, data: Dict[str, Any]) -> None:
        """Add financial data for a category"""
        if not hasattr(self.financial_data, 'quarterly_data'):
            self.financial_data.quarterly_data = {}
        self.financial_data.quarterly_data[category] = data
    
    def set_ai_analysis(self, summary: str, scores: Dict[str, float], 
                       full_analysis: Dict[str, Any] = None) -> None:
        """Set AI analysis results"""
        self.ai_summary = summary
        self.ai_scores = scores
        if full_analysis:
            self.ai_analysis = full_analysis
        self.processing_status = "completed"
    
    def set_processing_error(self, error: str) -> None:
        """Set processing error"""
        self.error_message = error
        self.processing_status = "failed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'cik': self.cik,
            'fiscal_year': self.fiscal_year,
            'fiscal_period': self.fiscal_period,
            'period_key': self.period_key,
            'form_type': self.form_type,
            'filing_date': self.filing_date.isoformat() if self.filing_date and hasattr(self.filing_date, 'isoformat') else str(self.filing_date) if self.filing_date else None,
            'accession_number': self.accession_number,
            'financial_data': self.financial_data.to_dict() if self.financial_data else None,
            'ai_summary': self.ai_summary,
            'ai_scores': self.ai_scores,
            'ai_analysis': self.ai_analysis,
            'processing_status': self.processing_status,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class FundamentalMetrics:
    """Comprehensive fundamental analysis metrics and scores"""
    
    # Core identifiers
    symbol: str
    period: str
    analysis_date: datetime = field(default_factory=datetime.now)
    
    # AI Analysis scores (0-10 scale)
    financial_health_score: float = 0.0
    business_quality_score: float = 0.0
    growth_prospects_score: float = 0.0
    overall_score: float = 0.0
    
    # Key insights
    key_insights: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    investment_thesis: str = ""
    
    # Detailed analysis components
    profitability_metrics: Dict[str, float] = field(default_factory=dict)
    liquidity_metrics: Dict[str, float] = field(default_factory=dict)
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    growth_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Financial ratios
    profit_margin: Optional[float] = None
    ROE: Optional[float] = None
    ROA: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    
    # Metadata
    model_used: Optional[str] = None
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_recommendation(self) -> str:
        """Get investment recommendation based on scores"""
        if self.overall_score >= 8.0:
            return "Strong Buy"
        elif self.overall_score >= 7.0:
            return "Buy" 
        elif self.overall_score >= 6.0:
            return "Hold"
        elif self.overall_score >= 4.0:
            return "Weak Hold"
        else:
            return "Sell"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'period': self.period,
            'analysis_date': self.analysis_date.isoformat(),
            'financial_health_score': self.financial_health_score,
            'business_quality_score': self.business_quality_score,
            'growth_prospects_score': self.growth_prospects_score,
            'overall_score': self.overall_score,
            'recommendation': self.get_recommendation(),
            'key_insights': self.key_insights,
            'key_risks': self.key_risks,
            'investment_thesis': self.investment_thesis,
            'profitability_metrics': self.profitability_metrics,
            'liquidity_metrics': self.liquidity_metrics,
            'efficiency_metrics': self.efficiency_metrics,
            'growth_metrics': self.growth_metrics,
            'profit_margin': self.profit_margin,
            'ROE': self.ROE,
            'ROA': self.ROA,
            'debt_to_equity': self.debt_to_equity,
            'current_ratio': self.current_ratio,
            'quick_ratio': self.quick_ratio,
            'model_used': self.model_used,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }


@dataclass
class Filing:
    """Represents a single SEC filing"""
    
    symbol: str
    form_type: str
    filing_date: date
    accession_number: str
    cik: str
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    primary_document: Optional[str] = None
    is_amended: bool = False
    amendment_number: Optional[int] = None
    filing_url: Optional[str] = None
    
    def get_filing_key(self) -> str:
        """Get unique filing identifier"""
        return f"{self.form_type}_{self.fiscal_year}_{self.fiscal_period}"
    
    def get_period_key(self) -> str:
        """Get standardized period key"""
        if self.fiscal_year and self.fiscal_period:
            return f"{self.fiscal_year}-{self.fiscal_period}"
        return ""


@dataclass 
class CompanyInfo:
    """Represents comprehensive company information"""
    
    symbol: str
    name: str
    cik: str
    sic: Optional[str] = None
    business_description: Optional[str] = None
    industry: Optional[str] = None
    sector: Optional[str] = None
    website: Optional[str] = None
    headquarters: Optional[str] = None
    employees: Optional[int] = None
    founded_year: Optional[int] = None
    market_cap: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'cik': self.cik,
            'sic': self.sic,
            'business_description': self.business_description,
            'industry': self.industry,
            'sector': self.sector,
            'website': self.website,
            'headquarters': self.headquarters,
            'employees': self.employees,
            'founded_year': self.founded_year,
            'market_cap': self.market_cap
        }


@dataclass
class TechnicalAnalysisData:
    """Technical analysis data and indicators"""
    
    symbol: str
    period: str
    analysis_date: datetime = field(default_factory=datetime.now)
    
    # Price data
    current_price: Optional[float] = None
    price_change: Optional[float] = None
    price_change_percent: Optional[float] = None
    
    # Technical indicators
    moving_averages: Dict[str, float] = field(default_factory=dict)
    momentum_indicators: Dict[str, float] = field(default_factory=dict)
    volatility_indicators: Dict[str, float] = field(default_factory=dict)
    volume_indicators: Dict[str, float] = field(default_factory=dict)
    
    # AI analysis
    ai_summary: Optional[str] = None
    technical_score: float = 0.0
    trend_direction: Optional[str] = None
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    
    # Metadata
    model_used: Optional[str] = None
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'period': self.period,
            'analysis_date': self.analysis_date.isoformat(),
            'current_price': self.current_price,
            'price_change': self.price_change,
            'price_change_percent': self.price_change_percent,
            'moving_averages': self.moving_averages,
            'momentum_indicators': self.momentum_indicators,
            'volatility_indicators': self.volatility_indicators,
            'volume_indicators': self.volume_indicators,
            'ai_summary': self.ai_summary,
            'technical_score': self.technical_score,
            'trend_direction': self.trend_direction,
            'support_levels': self.support_levels,
            'resistance_levels': self.resistance_levels,
            'model_used': self.model_used,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }


# Backwards compatibility aliases
FinancialMetrics = FundamentalMetrics  # Alias for existing code

# Export all models
__all__ = [
    'FinancialStatementData',
    'QuarterlyData', 
    'FundamentalMetrics',
    'Filing',
    'CompanyInfo',
    'TechnicalAnalysisData',
    'FinancialMetrics'  # Backwards compatibility
]