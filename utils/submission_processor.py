#!/usr/bin/env python3
"""
InvestiGator - Submission Processor Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Submission Processor Module
Handles SEC submission data parsing, filtering, and processing with support for amended filings
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Filing:
    """Represents a single SEC filing"""
    form_type: str
    filing_date: str
    accession_number: str
    primary_document: str
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    is_amended: bool = False
    amendment_number: Optional[int] = None
    report_date: Optional[str] = None
    
    def __post_init__(self):
        """Parse fiscal information from filing data"""
        # Detect amended filings
        if self.form_type.endswith('/A'):
            self.is_amended = True
            # Extract amendment number if available (e.g., 10-K/A2 -> 2)
            if len(self.form_type) > 6 and self.form_type[-2].isdigit():
                self.amendment_number = int(self.form_type[-2])
            else:
                self.amendment_number = 1
        
        # Parse fiscal year from filing date
        if self.filing_date:
            self.fiscal_year = int(self.filing_date.split('-')[0])
            
        # Determine fiscal period based on form type and filing month
        if self.form_type.startswith('10-K'):
            self.fiscal_period = 'FY'
        elif self.form_type.startswith('10-Q'):
            # Estimate quarter based on filing month
            month = int(self.filing_date.split('-')[1])
            if month <= 3:
                self.fiscal_period = 'Q4'
                self.fiscal_year -= 1  # Previous fiscal year
            elif month <= 6:
                self.fiscal_period = 'Q1'
            elif month <= 9:
                self.fiscal_period = 'Q2'
            else:
                self.fiscal_period = 'Q3'
    
    @property
    def period_key(self) -> str:
        """Generate unique period key for this filing"""
        return f"{self.fiscal_year}-{self.fiscal_period}"
    
    @property
    def base_form_type(self) -> str:
        """Get base form type without amendment suffix"""
        if self.is_amended and self.form_type.endswith('/A'):
            return self.form_type[:-2]
        return self.form_type


class SubmissionProcessor:
    """Processes SEC submission data with support for amendments and filtering"""
    
    def __init__(self):
        self.logger = logger
        
    def parse_submissions(self, submissions_data: Dict) -> Dict:
        """
        Parse raw SEC submissions JSON data into structured format
        
        Args:
            submissions_data: Raw submissions data from SEC API
            
        Returns:
            Structured submission data with parsed filings
        """
        try:
            # Extract company information
            parsed_data = {
                'cik': submissions_data.get('cik', ''),
                'entity_type': submissions_data.get('entityType', ''),
                'sic': submissions_data.get('sic', ''),
                'sic_description': submissions_data.get('sicDescription', ''),
                'name': submissions_data.get('name', ''),
                'tickers': submissions_data.get('tickers', []),
                'exchanges': submissions_data.get('exchanges', []),
                'fiscal_year_end': submissions_data.get('fiscalYearEnd', ''),
                'state_of_incorporation': submissions_data.get('stateOfIncorporation', ''),
                'website': submissions_data.get('website', ''),
                'investor_website': submissions_data.get('investorWebsite', ''),
                'category': submissions_data.get('category', ''),
                'description': submissions_data.get('description', ''),
                'addresses': submissions_data.get('addresses', {}),
                'phone': submissions_data.get('phone', ''),
                'flags': submissions_data.get('flags', ''),
                'former_names': submissions_data.get('formerNames', []),
                'filings': self._parse_filings(submissions_data.get('filings', {}))
            }
            
            return parsed_data
            
        except Exception as e:
            self.logger.error(f"Error parsing submissions data: {e}")
            raise
    
    def _parse_filings(self, filings_data: Dict) -> Dict:
        """Parse filings section of submissions data"""
        try:
            recent_filings = filings_data.get('recent', {})
            
            # Extract filing arrays
            form_types = recent_filings.get('form', [])
            filing_dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            primary_documents = recent_filings.get('primaryDocument', [])
            report_dates = recent_filings.get('reportDate', [])
            
            # Parse into Filing objects
            all_filings = []
            for i in range(len(form_types)):
                try:
                    filing = Filing(
                        form_type=form_types[i] if i < len(form_types) else '',
                        filing_date=filing_dates[i] if i < len(filing_dates) else '',
                        accession_number=accession_numbers[i] if i < len(accession_numbers) else '',
                        primary_document=primary_documents[i] if i < len(primary_documents) else '',
                        report_date=report_dates[i] if i < len(report_dates) else None
                    )
                    all_filings.append(filing)
                except Exception as e:
                    self.logger.warning(f"Error parsing filing at index {i}: {e}")
                    continue
            
            return {
                'all': all_filings,
                'count': len(all_filings)
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing filings data: {e}")
            return {'all': [], 'count': 0}
    
    def get_recent_earnings_filings(self, 
                                  submissions_data: Dict, 
                                  limit: int = 8,
                                  include_amendments: bool = True) -> List[Filing]:
        """
        Get recent 10-K and 10-Q filings with amendment handling
        
        Args:
            submissions_data: Parsed submission data
            limit: Maximum number of filings to return
            include_amendments: Whether to include amended filings
            
        Returns:
            List of Filing objects, with amendments taking precedence
        """
        try:
            all_filings = submissions_data.get('filings', {}).get('all', [])
            
            # Filter for earnings filings (10-K, 10-Q and their amendments)
            earnings_filings = []
            for filing in all_filings:
                base_type = filing.base_form_type
                if base_type in ['10-K', '10-Q']:
                    if include_amendments or not filing.is_amended:
                        earnings_filings.append(filing)
            
            # Group by period and resolve amendments
            period_filings = {}
            for filing in earnings_filings:
                period_key = filing.period_key
                
                if period_key not in period_filings:
                    period_filings[period_key] = filing
                else:
                    # Compare with existing filing for this period
                    existing = period_filings[period_key]
                    
                    # Amended filing takes precedence
                    if filing.is_amended and not existing.is_amended:
                        period_filings[period_key] = filing
                    elif filing.is_amended and existing.is_amended:
                        # Higher amendment number takes precedence
                        if (filing.amendment_number or 0) > (existing.amendment_number or 0):
                            period_filings[period_key] = filing
                    elif not filing.is_amended and existing.is_amended:
                        # Keep the amended version
                        pass
                    else:
                        # Both are original, keep the more recent filing date
                        if filing.filing_date > existing.filing_date:
                            period_filings[period_key] = filing
            
            # Generic logic for balanced quarter distribution
            # Priority order: FY (annual), Q4, Q3, Q2, Q1
            period_order = ['FY', 'Q4', 'Q3', 'Q2', 'Q1']
            period_priority = {period: i for i, period in enumerate(period_order)}
            
            # Group filings by period type
            periods_by_type = {}
            for filing in period_filings.values():
                period_type = filing.fiscal_period
                if period_type not in periods_by_type:
                    periods_by_type[period_type] = []
                periods_by_type[period_type].append(filing)
            
            # Sort filings within each period type by fiscal year (descending)
            for period_type in periods_by_type:
                periods_by_type[period_type].sort(
                    key=lambda f: f.fiscal_year,
                    reverse=True
                )
            
            # Calculate slots per period type for balanced distribution
            available_periods = [p for p in period_order if p in periods_by_type]
            num_periods = len(available_periods)
            
            if num_periods == 0:
                sorted_filings = []
            else:
                # Base slots per period (integer division)
                base_slots_per_period = limit // num_periods
                # Extra slots to distribute (remainder)
                extra_slots = limit % num_periods
                
                # Distribute slots: give base slots to all, then distribute extras in priority order
                slots_per_period = {}
                for i, period_type in enumerate(available_periods):
                    slots_per_period[period_type] = base_slots_per_period
                    if i < extra_slots:  # Distribute extra slots to higher priority periods
                        slots_per_period[period_type] += 1
                
                # Select filings based on calculated slots
                selected_filings = []
                for period_type in available_periods:
                    max_slots = slots_per_period[period_type]
                    available_filings = periods_by_type[period_type]
                    selected_count = min(max_slots, len(available_filings))
                    selected_filings.extend(available_filings[:selected_count])
                
                # Sort final selection by fiscal year and period priority (descending)
                sorted_filings = sorted(
                    selected_filings,
                    key=lambda f: (f.fiscal_year, period_priority.get(f.fiscal_period, 5)),
                    reverse=True
                )
            
            # Apply limit
            result = sorted_filings[:limit]
            
            # Log summary with slot distribution
            if num_periods > 0:
                slot_info = ", ".join([f"{p}:{slots_per_period[p]}" for p in available_periods])
                self.logger.info(f"Found {len(result)} unique earnings filings (limit: {limit}, slots: {slot_info})")
            else:
                self.logger.info(f"Found {len(result)} unique earnings filings (limit: {limit})")
            
            for filing in result[:8]:  # Log up to 8 filings
                amend_info = f" (Amendment {filing.amendment_number})" if filing.is_amended else ""
                self.logger.debug(f"  - {filing.form_type} {filing.period_key} filed {filing.filing_date}{amend_info}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting recent earnings filings: {e}")
            return []
    
    def get_filings_by_type(self, 
                           submissions_data: Dict,
                           form_types: List[str],
                           limit: Optional[int] = None) -> List[Filing]:
        """
        Get filings of specific types
        
        Args:
            submissions_data: Parsed submission data
            form_types: List of form types to filter (e.g., ['10-K', '8-K'])
            limit: Maximum number of filings to return
            
        Returns:
            List of Filing objects
        """
        try:
            all_filings = submissions_data.get('filings', {}).get('all', [])
            
            # Filter by form types
            filtered_filings = []
            for filing in all_filings:
                if filing.base_form_type in form_types:
                    filtered_filings.append(filing)
            
            # Sort by filing date (descending)
            filtered_filings.sort(key=lambda f: f.filing_date, reverse=True)
            
            # Apply limit if specified
            if limit:
                filtered_filings = filtered_filings[:limit]
            
            return filtered_filings
            
        except Exception as e:
            self.logger.error(f"Error getting filings by type: {e}")
            return []
    
    def convert_to_cache_format(self, parsed_data: Dict) -> Dict:
        """
        Convert parsed submission data to cache storage format
        
        Args:
            parsed_data: Parsed submission data
            
        Returns:
            Data formatted for cache storage
        """
        try:
            # Convert Filing objects to dictionaries
            filings_dict = parsed_data.get('filings', {})
            all_filings = filings_dict.get('all', [])
            
            # Serialize Filing objects
            serialized_filings = []
            for filing in all_filings:
                if isinstance(filing, Filing):
                    serialized_filings.append({
                        'form_type': filing.form_type,
                        'filing_date': filing.filing_date,
                        'accession_number': filing.accession_number,
                        'primary_document': filing.primary_document,
                        'fiscal_year': filing.fiscal_year,
                        'fiscal_period': filing.fiscal_period,
                        'is_amended': filing.is_amended,
                        'amendment_number': filing.amendment_number,
                        'report_date': filing.report_date,
                        'period_key': filing.period_key,
                        'base_form_type': filing.base_form_type
                    })
                else:
                    serialized_filings.append(filing)
            
            # Create cache format
            cache_data = parsed_data.copy()
            cache_data['filings'] = {
                'all': serialized_filings,
                'count': len(serialized_filings)
            }
            
            return cache_data
            
        except Exception as e:
            self.logger.error(f"Error converting to cache format: {e}")
            raise
    
    def restore_from_cache_format(self, cache_data: Dict) -> Dict:
        """
        Restore parsed submission data from cache format
        
        Args:
            cache_data: Data from cache storage
            
        Returns:
            Parsed submission data with Filing objects
        """
        try:
            # Deep copy to avoid modifying cache data
            restored_data = cache_data.copy()
            
            # Restore Filing objects
            filings_dict = restored_data.get('filings', {})
            serialized_filings = filings_dict.get('all', [])
            
            filing_objects = []
            for filing_data in serialized_filings:
                if isinstance(filing_data, dict):
                    filing = Filing(
                        form_type=filing_data.get('form_type', ''),
                        filing_date=filing_data.get('filing_date', ''),
                        accession_number=filing_data.get('accession_number', ''),
                        primary_document=filing_data.get('primary_document', ''),
                        fiscal_year=filing_data.get('fiscal_year'),
                        fiscal_period=filing_data.get('fiscal_period'),
                        is_amended=filing_data.get('is_amended', False),
                        amendment_number=filing_data.get('amendment_number'),
                        report_date=filing_data.get('report_date')
                    )
                    filing_objects.append(filing)
            
            restored_data['filings'] = {
                'all': filing_objects,
                'count': len(filing_objects)
            }
            
            return restored_data
            
        except Exception as e:
            self.logger.error(f"Error restoring from cache format: {e}")
            raise


# Singleton instance
_processor = None

def get_submission_processor() -> SubmissionProcessor:
    """Get singleton instance of SubmissionProcessor"""
    global _processor
    if _processor is None:
        _processor = SubmissionProcessor()
    return _processor