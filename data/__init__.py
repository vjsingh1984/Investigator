#!/usr/bin/env python3
"""
InvestiGator - Data Package Initialization
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

InvestiGator Data Package
Data processing and retrieval components
"""

# Direct imports to actual implementations (processors and sources were unused wrappers)
from utils.sec_quarterly_processor import SECQuarterlyProcessor
from utils.financial_data_aggregator import FinancialDataAggregator
from utils.sec_frame_api import SECFrameAPI
from utils.ticker_cik_mapper import TickerCIKMapper, ticker_to_cik
from patterns.llm.llm_facade import create_llm_facade

from .models import (
    QuarterlyData, Filing, CompanyInfo, FinancialMetrics,
    FinancialStatementData, FundamentalMetrics, TechnicalAnalysisData
)

__all__ = [
    'SECQuarterlyProcessor', 'FinancialDataAggregator', 'create_llm_facade',
    'SECFrameAPI', 'TickerCIKMapper', 'ticker_to_cik',
    'QuarterlyData', 'Filing', 'CompanyInfo', 'FinancialMetrics',
    'FinancialStatementData', 'FundamentalMetrics', 'TechnicalAnalysisData'
]