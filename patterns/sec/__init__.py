#!/usr/bin/env python3
"""
InvestiGator - SEC Pattern Implementations Initialization
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

SEC Pattern Implementations
SEC data fetching and analysis patterns
"""

from .sec_facade import *
from .sec_strategies import *
from .sec_adapters import *

__all__ = [
    # Facades
    'SECDataFacade', 'FundamentalAnalysisFacadeV2',
    # Strategies
    'CompanyFactsStrategy', 'SubmissionsStrategy', 
    'CachedDataStrategy', 'HybridFetchStrategy',
    # Adapters
    'SECToInternalAdapter', 'InternalToLLMAdapter',
    'FilingContentAdapter', 'CompanyFactsToDetailedAdapter'
]