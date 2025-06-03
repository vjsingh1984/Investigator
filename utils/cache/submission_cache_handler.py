#!/usr/bin/env python3
"""
InvestiGator - Submission Cache Handler
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Unified submission cache handler for consistent disk and RDBMS storage
Integrates with submission_processor for parsing and amendment handling
"""

import os
import json
import gzip
import logging
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
from sqlalchemy import text

from .cache_base import CacheStorageHandler
from .cache_types import CacheType
from ..submission_processor import get_submission_processor

logger = logging.getLogger(__name__)


class SubmissionCacheHandler:
    """
    Unified handler for submission data caching
    Ensures consistency between disk (gzip) and RDBMS (JSONB) storage
    """
    
    def __init__(self, config=None):
        self.config = config
        self.processor = get_submission_processor()
        self.logger = logger
        
        # Cache paths
        self.disk_cache_path = Path("data/sec_cache/submissions")
        self.disk_cache_path.mkdir(parents=True, exist_ok=True)
    
    def get_submission_from_disk(self, ticker: str, cik: str) -> Optional[Dict]:
        """
        Get submission data from disk cache (gzip compressed)
        
        Args:
            ticker: Stock ticker
            cik: Company CIK (padded)
            
        Returns:
            Parsed submission data or None
        """
        try:
            # Try multiple file naming patterns
            file_patterns = [
                f"cik_{cik}_symbol_{ticker}.json.gz",
                f"{ticker}_submissions.json.gz",
                f"{cik}_submissions.json.gz"
            ]
            
            for pattern in file_patterns:
                cache_file = self.disk_cache_path / pattern
                if cache_file.exists():
                    with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    # Check if data needs parsing
                    if 'filings' in cache_data and isinstance(cache_data['filings'], dict):
                        if 'all' in cache_data['filings'] and cache_data['filings']['all']:
                            # Check if already parsed (has Filing objects)
                            first_filing = cache_data['filings']['all'][0]
                            if isinstance(first_filing, dict) and 'period_key' in first_filing:
                                # Already in cache format, restore to parsed format
                                parsed_data = self.processor.restore_from_cache_format(cache_data)
                            else:
                                # Raw SEC data, needs parsing
                                parsed_data = self.processor.parse_submissions(cache_data)
                        else:
                            # Raw SEC data
                            parsed_data = self.processor.parse_submissions(cache_data)
                    else:
                        # Assume it's already parsed
                        parsed_data = cache_data
                    
                    self.logger.debug(f"Loaded submission from disk: {cache_file.name}")
                    return parsed_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error reading submission from disk for {ticker}: {e}")
            return None
    
    def save_submission_to_disk(self, ticker: str, cik: str, data: Dict) -> bool:
        """
        Save submission data to disk cache with maximum gzip compression
        
        Args:
            ticker: Stock ticker
            cik: Company CIK (padded)
            data: Submission data (parsed or raw)
            
        Returns:
            Success status
        """
        try:
            # Convert to cache format if needed
            if 'filings' in data and 'all' in data['filings']:
                cache_data = self.processor.convert_to_cache_format(data)
            else:
                cache_data = data
            
            # Add metadata
            cache_data['_cache_metadata'] = {
                'ticker': ticker,
                'cik': cik,
                'cached_at': datetime.now().isoformat(),
                'cache_version': 2  # Version 2 with submission processor
            }
            
            # Save with maximum compression (level 9)
            cache_file = self.disk_cache_path / f"cik_{cik}_symbol_{ticker}.json.gz"
            with gzip.open(cache_file, 'wt', encoding='utf-8', compresslevel=9) as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"Saved submission to disk: {cache_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving submission to disk for {ticker}: {e}")
            return False
    
    def get_submission_from_rdbms(self, ticker: str, cik: str) -> Optional[Dict]:
        """
        Get submission data from RDBMS (PostgreSQL JSONB)
        
        Args:
            ticker: Stock ticker
            cik: Company CIK (padded)
            
        Returns:
            Parsed submission data or None
        """
        try:
            from utils.db import get_db_manager
            
            db = get_db_manager()
            engine = db.engine
            
            # Query new schema first
            query = """
                SELECT 
                    cik,
                    ticker,
                    company_name,
                    parsed_data,
                    raw_data,
                    updated_at
                FROM sec_submissions_v2
                WHERE cik = :cik OR ticker = :ticker
                ORDER BY updated_at DESC
                LIMIT 1
            """
            
            with engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {'cik': cik, 'ticker': ticker}
                ).fetchone()
                
                if result:
                    # Restore from parsed data
                    parsed_data = self.processor.restore_from_cache_format(result['parsed_data'])
                    self.logger.debug(f"Loaded submission from RDBMS for {ticker}")
                    return parsed_data
                
                # Fallback to old schema
                old_query = """
                    SELECT 
                        cik,
                        ticker,
                        company_name,
                        filings,
                        updated_at
                    FROM sec_submissions
                    WHERE cik = :cik OR ticker = :ticker
                    ORDER BY updated_at DESC
                    LIMIT 1
                """
                
                result = conn.execute(
                    text(old_query),
                    {'cik': cik, 'ticker': ticker}
                ).fetchone()
                
                if result:
                    # Parse raw SEC data
                    parsed_data = self.processor.parse_submissions(result['filings'])
                    self.logger.debug(f"Loaded submission from RDBMS (legacy) for {ticker}")
                    return parsed_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error reading submission from RDBMS for {ticker}: {e}")
            return None
    
    def save_submission_to_rdbms(self, ticker: str, cik: str, data: Dict, raw_data: Optional[Dict] = None) -> bool:
        """
        Save submission data to RDBMS with native PostgreSQL compression
        
        Args:
            ticker: Stock ticker
            cik: Company CIK (padded)
            data: Parsed submission data
            raw_data: Original SEC API response
            
        Returns:
            Success status
        """
        try:
            from utils.db import get_db_manager
            
            db = get_db_manager()
            engine = db.engine
            
            # Convert to cache format for storage
            cache_data = self.processor.convert_to_cache_format(data)
            
            # Extract company info
            company_info = {
                'cik': cik,
                'ticker': ticker,
                'company_name': data.get('name', ''),
                'entity_type': data.get('entity_type', ''),
                'sic': data.get('sic', ''),
                'sic_description': data.get('sic_description', ''),
                'fiscal_year_end': data.get('fiscal_year_end', ''),
                'state_of_incorporation': data.get('state_of_incorporation', ''),
                'website': data.get('website', ''),
                'investor_website': data.get('investor_website', ''),
                'category': data.get('category', ''),
                'description': data.get('description', ''),
                'phone': data.get('phone', '')
            }
            
            # Upsert to sec_submissions_v2
            upsert_query = """
                INSERT INTO sec_submissions_v2 (
                    cik, ticker, company_name,
                    entity_type, sic, sic_description,
                    fiscal_year_end, state_of_incorporation,
                    website, investor_website, category,
                    description, phone,
                    parsed_data, raw_data,
                    updated_at
                ) VALUES (
                    :cik, :ticker, :company_name,
                    :entity_type, :sic, :sic_description,
                    :fiscal_year_end, :state_of_incorporation,
                    :website, :investor_website, :category,
                    :description, :phone,
                    :parsed_data, :raw_data,
                    NOW()
                )
                ON CONFLICT (cik) DO UPDATE SET
                    ticker = EXCLUDED.ticker,
                    company_name = EXCLUDED.company_name,
                    entity_type = EXCLUDED.entity_type,
                    sic = EXCLUDED.sic,
                    sic_description = EXCLUDED.sic_description,
                    fiscal_year_end = EXCLUDED.fiscal_year_end,
                    state_of_incorporation = EXCLUDED.state_of_incorporation,
                    website = EXCLUDED.website,
                    investor_website = EXCLUDED.investor_website,
                    category = EXCLUDED.category,
                    description = EXCLUDED.description,
                    phone = EXCLUDED.phone,
                    parsed_data = EXCLUDED.parsed_data,
                    raw_data = EXCLUDED.raw_data,
                    updated_at = NOW()
            """
            
            with engine.begin() as conn:
                conn.execute(
                    text(upsert_query),
                    {
                        **company_info,
                        'parsed_data': json.dumps(cache_data, ensure_ascii=False),
                        'raw_data': json.dumps(raw_data, ensure_ascii=False) if raw_data else None
                    }
                )
                
                # Also update individual filings table
                self._update_filings_table(conn, ticker, cik, data)
            
            self.logger.debug(f"Saved submission to RDBMS for {ticker}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving submission to RDBMS for {ticker}: {e}")
            return False
    
    def _update_filings_table(self, conn, ticker: str, cik: str, parsed_data: Dict):
        """Update the sec_filings table with individual filing records"""
        try:
            # Get recent earnings filings
            recent_filings = self.processor.get_recent_earnings_filings(parsed_data, limit=20)
            
            for filing in recent_filings:
                # Check if filing exists
                check_query = """
                    SELECT id FROM sec_filings 
                    WHERE accession_number = :accession_number
                """
                existing = conn.execute(
                    check_query,
                    {'accession_number': filing.accession_number}
                ).fetchone()
                
                if not existing:
                    # Insert new filing
                    insert_query = """
                        INSERT INTO sec_filings (
                            cik, ticker, form_type, filing_date,
                            accession_number, primary_document,
                            report_date, fiscal_year, fiscal_period,
                            period_key, is_amended, amendment_number,
                            base_form_type
                        ) VALUES (
                            :cik, :ticker, :form_type, :filing_date,
                            :accession_number, :primary_document,
                            :report_date, :fiscal_year, :fiscal_period,
                            :period_key, :is_amended, :amendment_number,
                            :base_form_type
                        )
                    """
                    
                    conn.execute(
                        insert_query,
                        {
                            'cik': cik,
                            'ticker': ticker,
                            'form_type': filing.form_type,
                            'filing_date': filing.filing_date,
                            'accession_number': filing.accession_number,
                            'primary_document': filing.primary_document,
                            'report_date': filing.report_date,
                            'fiscal_year': filing.fiscal_year,
                            'fiscal_period': filing.fiscal_period,
                            'period_key': filing.period_key,
                            'is_amended': filing.is_amended,
                            'amendment_number': filing.amendment_number,
                            'base_form_type': filing.base_form_type
                        }
                    )
            
        except Exception as e:
            self.logger.warning(f"Error updating filings table: {e}")
    
    def get_submission(self, ticker: str, cik: str, prefer_disk: bool = True) -> Optional[Dict]:
        """
        Get submission data from cache (disk or RDBMS)
        
        Args:
            ticker: Stock ticker
            cik: Company CIK (padded)
            prefer_disk: Whether to prefer disk cache
            
        Returns:
            Parsed submission data or None
        """
        if prefer_disk:
            # Try disk first
            data = self.get_submission_from_disk(ticker, cik)
            if data:
                return data
            # Fallback to RDBMS
            return self.get_submission_from_rdbms(ticker, cik)
        else:
            # Try RDBMS first
            data = self.get_submission_from_rdbms(ticker, cik)
            if data:
                return data
            # Fallback to disk
            return self.get_submission_from_disk(ticker, cik)
    
    def save_submission(self, ticker: str, cik: str, data: Dict, raw_data: Optional[Dict] = None) -> bool:
        """
        Save submission data to both disk and RDBMS caches
        
        Args:
            ticker: Stock ticker
            cik: Company CIK (padded)
            data: Submission data (will be parsed if needed)
            raw_data: Original SEC API response
            
        Returns:
            Success status (True if at least one cache succeeded)
        """
        # Parse data if needed
        if 'filings' in data and not isinstance(data.get('filings', {}).get('all', [{}])[0], dict):
            parsed_data = self.processor.parse_submissions(data)
        else:
            parsed_data = data
        
        # Save to both caches
        disk_success = self.save_submission_to_disk(ticker, cik, parsed_data)
        rdbms_success = self.save_submission_to_rdbms(ticker, cik, parsed_data, raw_data or data)
        
        if disk_success and rdbms_success:
            self.logger.info(f"Saved submission to both disk and RDBMS for {ticker}")
        elif disk_success:
            self.logger.warning(f"Saved submission to disk only for {ticker}")
        elif rdbms_success:
            self.logger.warning(f"Saved submission to RDBMS only for {ticker}")
        else:
            self.logger.error(f"Failed to save submission to any cache for {ticker}")
        
        return disk_success or rdbms_success


# Singleton instance
_handler = None

def get_submission_cache_handler(config=None) -> SubmissionCacheHandler:
    """Get singleton instance of SubmissionCacheHandler"""
    global _handler
    if _handler is None:
        _handler = SubmissionCacheHandler(config)
    return _handler