#!/usr/bin/env python3
"""
InvestiGator - Period Utilities
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Period Utilities - Centralized period standardization functions
Eliminates duplicate period handling logic across the codebase
"""

import re
from typing import Tuple, Optional


def standardize_period(fiscal_year: int, fiscal_period: str) -> str:
    """
    Standardize fiscal year and period into consistent format
    
    Args:
        fiscal_year: Fiscal year (e.g., 2024)
        fiscal_period: Fiscal period (e.g., "Q1", "Q2", "FY", "annual")
        
    Returns:
        Standardized period key (e.g., "2024-Q1", "2024-FY")
    """
    # Normalize fiscal period
    period = str(fiscal_period).upper().strip()
    
    # Handle various quarterly formats
    if period in ['Q1', 'Q1-YTD', 'FIRST QUARTER', '1Q']:
        period = 'Q1'
    elif period in ['Q2', 'Q2-YTD', 'SECOND QUARTER', '2Q']:
        period = 'Q2'
    elif period in ['Q3', 'Q3-YTD', 'THIRD QUARTER', '3Q']:
        period = 'Q3'
    elif period in ['Q4', 'Q4-YTD', 'FOURTH QUARTER', '4Q']:
        period = 'Q4'
    elif period in ['FY', 'ANNUAL', 'YEAR', 'CY']:
        period = 'FY'
    
    return f"{fiscal_year}-{period}"


def parse_period_key(period_key: str) -> Tuple[int, str]:
    """
    Parse a standardized period key back into year and period
    
    Args:
        period_key: Period key (e.g., "2024-Q1", "2024-FY")
        
    Returns:
        Tuple of (fiscal_year, fiscal_period)
    """
    if '-' not in period_key:
        raise ValueError(f"Invalid period key format: {period_key}")
    
    parts = period_key.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid period key format: {period_key}")
    
    try:
        fiscal_year = int(parts[0])
        fiscal_period = parts[1]
        return fiscal_year, fiscal_period
    except ValueError:
        raise ValueError(f"Invalid period key format: {period_key}")


