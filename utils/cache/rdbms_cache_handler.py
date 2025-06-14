#!/usr/bin/env python3
"""
InvestiGator - RDBMS Cache Handler
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

RDBMS based cache storage handler
"""

from typing import Dict, Any, Optional, Union, Tuple
import json
import logging
from datetime import datetime
from sqlalchemy import text

from .cache_base import CacheStorageHandler
from .cache_types import CacheType

# UTF-8 encoding helpers for JSON operations
def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely encode object to JSON with UTF-8 encoding, handling binary characters"""
    return json.dumps(obj, ensure_ascii=False, **kwargs)

def safe_json_loads(json_str: str) -> Any:
    """Safely decode JSON string with UTF-8 encoding"""
    if isinstance(json_str, bytes):
        json_str = json_str.decode('utf-8', errors='replace')
    return json.loads(json_str)

logger = logging.getLogger(__name__)


class RdbmsCacheStorageHandler(CacheStorageHandler):
    """RDBMS based cache storage handler"""
    
    def __init__(self, cache_type: CacheType, priority: int = 0):
        """
        Initialize RDBMS cache handler
        
        Args:
            cache_type: Type of cache
            priority: Priority for lookup
        """
        super().__init__(cache_type, priority)
        
        # Import here to avoid circular dependencies
        from utils.db import (
            get_sec_responses_dao,
            get_llm_responses_dao,
            get_quarterly_metrics_dao,
            get_sec_companyfacts_dao,
            get_sec_submissions_dao,
            DatabaseManager
        )
        
        # Always initialize DatabaseManager for delete operations
        self.db_manager = DatabaseManager()
        
        # Initialize appropriate DAO based on cache type
        if cache_type == CacheType.SEC_RESPONSE:
            self.dao = get_sec_responses_dao()
        elif cache_type == CacheType.LLM_RESPONSE:
            self.dao = get_llm_responses_dao()
        elif cache_type == CacheType.QUARTERLY_METRICS:
            self.dao = get_quarterly_metrics_dao()
        elif cache_type == CacheType.COMPANY_FACTS:
            self.dao = get_sec_companyfacts_dao()
        elif cache_type == CacheType.SUBMISSION_DATA:
            self.dao = get_sec_submissions_dao()
        else:
            raise ValueError(f"Unsupported cache type for RDBMS: {cache_type}")
    
    def get(self, key: Union[Tuple, Dict]) -> Optional[Dict[str, Any]]:
        """Retrieve data from RDBMS cache"""
        if self.priority < 0:
            return None  # Skip lookup for negative priority
            
        try:
            key_dict = self._normalize_key(key)
            
            if self.cache_type == CacheType.SEC_RESPONSE:
                # Fetch from sec_response_store
                symbol = key_dict.get('symbol')
                period = key_dict.get('period')
                category = key_dict.get('category')
                
                if symbol and category and period:
                    result = self.dao.get_response(symbol, category, period)
                    if result:
                        logger.debug(f"Cache hit (RDBMS): SEC response for {symbol} {category} {period}")
                        # Return the full data structure expected by cache manager
                        return {
                            'response': result['response'],
                            'metadata': result['metadata'],
                            'symbol': symbol,
                            'category': category,
                            'period': period,
                            'ts': result['ts']
                        }
                        
            elif self.cache_type == CacheType.LLM_RESPONSE:
                # Fetch from llm_response_store
                symbol = key_dict.get('symbol')
                llm_type = key_dict.get('llm_type')
                form_type = key_dict.get('form_type', 'N/A')
                period = key_dict.get('period', 'N/A')
                
                if symbol and llm_type:
                    result = self.dao.get_llm_response(symbol, form_type, period, llm_type)
                    if result:
                        logger.debug(f"Cache hit (RDBMS): LLM response for {symbol}")
                        return result
                        
            elif self.cache_type == CacheType.SUBMISSION_DATA:
                # Fetch from sec_submissions table (no materialized view)
                symbol = key_dict.get('symbol')
                cik = key_dict.get('cik')
                
                # Only proceed if we have both symbol and CIK
                if symbol and cik:
                    from utils.db import get_sec_submissions_dao
                    dao = get_sec_submissions_dao()
                    
                    # Get submission data directly from sec_submissions table
                    result = dao.get_submission(symbol, cik, max_age_days=7)
                    if result:
                        logger.debug(f"Cache hit (RDBMS): Submission data for {symbol}")
                        return {
                                'symbol': symbol,
                                'cik': cik,
                                'company_name': result['company_name'],
                                'submissions_data': result['submissions_data'],
                                'cached_at': result['updated_at']
                            }
                            
            elif self.cache_type == CacheType.COMPANY_FACTS:
                # Fetch from all_companyfacts_store using DAO
                symbol = key_dict.get('symbol')
                
                if symbol:
                    from utils.db import get_sec_companyfacts_dao
                    dao = get_sec_companyfacts_dao()
                    result = dao.get_company_facts(symbol)
                    
                    if result:
                        logger.debug(f"Cache hit (RDBMS): Company facts for {symbol}")
                        return result
                            
            elif self.cache_type == CacheType.QUARTERLY_METRICS:
                # Fetch from quarterly_metrics
                symbol = key_dict.get('symbol')
                period = key_dict.get('period')
                
                if symbol and period:
                    result = self.dao.get_quarterly_metrics(symbol, period)
                    if result:
                        logger.debug(f"Cache hit (RDBMS): Quarterly metrics for {symbol} {period}")
                        return result
            
            logger.debug(f"Cache miss (RDBMS): {key_dict}")
            return None
            
        except Exception as e:
            logger.error(f"Error reading from RDBMS cache: {e}")
            return None
    
    def set(self, key: Union[Tuple, Dict], value: Dict[str, Any]) -> bool:
        """Store data in RDBMS cache"""
        try:
            key_dict = self._normalize_key(key)
            
            if self.cache_type == CacheType.SEC_RESPONSE:
                # Store in sec_response_store
                return self.dao.save_response(
                    symbol=key_dict.get('symbol'),
                    category=key_dict.get('category'),
                    period=key_dict.get('period'),
                    response=value.get('response', {}),
                    metadata=value.get('metadata', {})
                )
                
            elif self.cache_type == CacheType.LLM_RESPONSE:
                # Store in llm_response_store
                return self.dao.save_llm_response(
                    symbol=key_dict.get('symbol'),
                    form_type=key_dict.get('form_type', 'N/A'),
                    period=key_dict.get('period', 'N/A'),
                    prompt=value.get('prompt', ''),
                    model_info=value.get('model_info', {}),
                    response=value.get('response', {}),
                    metadata=value.get('metadata', {}),
                    llm_type=key_dict.get('llm_type')
                )
                
            elif self.cache_type == CacheType.SUBMISSION_DATA:
                # Store in all_submission_store using DAO
                symbol = key_dict.get('symbol')
                cik = key_dict.get('cik')
                
                if symbol and cik:
                    from utils.db import get_sec_submissions_dao
                    dao = get_sec_submissions_dao()
                    
                    # Extract latest filing date if available
                    latest_filing_date = None
                    submissions_data = value.get('submissions_data', {})
                    if isinstance(submissions_data, dict):
                        filings = submissions_data.get('filings', {}).get('recent', {})
                        filing_dates = filings.get('filingDate', [])
                        if filing_dates:
                            latest_filing_date = filing_dates[0]  # Assuming sorted desc
                    
                    success = dao.save_submission(
                        symbol=symbol,
                        cik=cik,
                        company_name=value.get('company_name', ''),
                        submissions_data=submissions_data,
                        latest_filing_date=latest_filing_date
                    )
                    
                    if success:
                        logger.debug(f"Stored submission data for {symbol}")
                    return success
                        
            elif self.cache_type == CacheType.COMPANY_FACTS:
                # Store in all_companyfacts_store using DAO
                symbol = key_dict.get('symbol')
                
                if symbol:
                    from utils.db import get_sec_companyfacts_dao
                    dao = get_sec_companyfacts_dao()
                    
                    # Handle both direct company facts and wrapped data
                    if 'companyfacts' in value:
                        companyfacts = value['companyfacts']
                        metadata = value.get('metadata', {})
                    else:
                        companyfacts = value
                        metadata = {}
                    
                    # Extract CIK with priority order: metadata > companyfacts > lookup
                    cik = None
                    
                    # 1. Try metadata first (most reliable)
                    if metadata and metadata.get('cik'):
                        cik = metadata['cik']
                    
                    # 2. Try companyfacts data
                    elif companyfacts.get('cik'):
                        cik_val = companyfacts['cik']
                        if isinstance(cik_val, int):
                            cik = f"{cik_val:010d}"  # Pad to 10 digits
                        else:
                            cik = str(cik_val)
                    
                    # 3. Fall back to ticker-CIK lookup
                    if not cik or cik == '0000000000':
                        from utils.ticker_cik_mapper import TickerCIKMapper
                        mapper = TickerCIKMapper()
                        lookup_cik = mapper.get_cik(symbol)
                        if lookup_cik:
                            cik = f"{int(lookup_cik):010d}"
                        else:
                            logger.warning(f"Could not resolve CIK for {symbol}, skipping RDBMS storage")
                            return False
                    
                    company_name = companyfacts.get('entityName', '')
                    
                    success = dao.store_company_facts(
                        symbol=symbol,
                        cik=cik,
                        company_name=company_name,
                        companyfacts=companyfacts,
                        metadata=metadata
                    )
                    
                    if success:
                        logger.debug(f"Stored company facts for {symbol} with CIK {cik}")
                    return success
                        
            elif self.cache_type == CacheType.QUARTERLY_METRICS:
                # Store in quarterly_metrics
                return self.dao.save_quarterly_metrics(
                    symbol=key_dict.get('symbol'),
                    period=key_dict.get('period'),
                    metrics=value.get('metrics', {}),
                    metadata=value.get('metadata', {})
                )
            
            return False
            
        except Exception as e:
            logger.error(f"Error writing to RDBMS cache: {e}")
            return False
    
    def exists(self, key: Union[Tuple, Dict]) -> bool:
        """Check if key exists in RDBMS cache"""
        # Use get method to check existence
        return self.get(key) is not None
    
    def delete(self, key: Union[Tuple, Dict]) -> bool:
        """Delete data from RDBMS cache"""
        try:
            key_dict = self._normalize_key(key)
            symbol = key_dict.get('symbol', '')
            
            if not symbol:
                logger.warning("Cannot delete from RDBMS cache without symbol")
                return False
            
            # Use DAO methods for deletion based on cache type
            if self.cache_type == CacheType.LLM_RESPONSE and self.dao:
                form_type = key_dict.get('form_type')
                period = key_dict.get('period')
                llm_type = key_dict.get('llm_type')
                
                deleted_count = self.dao.delete_llm_responses(
                    symbol=symbol,
                    form_type=form_type,
                    period=period,
                    llm_type=llm_type
                )
                return deleted_count > 0
            else:
                # For other cache types, fall back to basic deletion
                # This would need to be implemented for each cache type
                logger.warning(f"Delete operation not implemented for cache type: {self.cache_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting from RDBMS cache: {e}")
            return False
    
    def delete_by_symbol(self, symbol: str) -> int:
        """
        Optimized symbol-based deletion for RDBMS cache.
        Uses SQL queries with symbol column for efficient deletion.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Number of records deleted
        """
        try:
            deleted_count = 0
            symbol = symbol.upper()  # Normalize to uppercase
            
            # Delete based on cache type using appropriate DAO methods
            if self.cache_type == CacheType.LLM_RESPONSE and self.dao:
                # Delete all LLM responses for this symbol across all form types
                deleted_count = self.dao.delete_llm_responses_by_pattern(
                    symbol_pattern=symbol,
                    form_type_pattern='%'
                )
                logger.info(f"Symbol cleanup [RDBMS-LLM]: Deleted {deleted_count} LLM responses for symbol {symbol}")
                
            elif self.cache_type == CacheType.SUBMISSION_DATA:
                # Delete submission data for this symbol
                from utils.db import get_sec_submissions_dao
                dao = get_sec_submissions_dao()
                if hasattr(dao, 'delete_submissions_by_symbol'):
                    deleted_count = dao.delete_submissions_by_symbol(symbol)
                    logger.info(f"Symbol cleanup [RDBMS-SUB]: Deleted {deleted_count} submissions for symbol {symbol}")
                
            elif self.cache_type == CacheType.COMPANY_FACTS:
                # Delete company facts for this symbol
                from utils.db import get_sec_companyfacts_dao
                dao = get_sec_companyfacts_dao()
                if hasattr(dao, 'delete_companyfacts_by_symbol'):
                    deleted_count = dao.delete_companyfacts_by_symbol(symbol)
                    logger.info(f"Symbol cleanup [RDBMS-CF]: Deleted {deleted_count} company facts for symbol {symbol}")
                
            elif self.cache_type == CacheType.QUARTERLY_METRICS:
                # Delete quarterly metrics for this symbol
                from utils.db import get_quarterly_metrics_dao
                dao = get_quarterly_metrics_dao()
                if hasattr(dao, 'delete_metrics_by_symbol'):
                    deleted_count = dao.delete_metrics_by_symbol(symbol)
                    logger.info(f"Symbol cleanup [RDBMS-QM]: Deleted {deleted_count} quarterly metrics for symbol {symbol}")
                
            else:
                logger.debug(f"Symbol deletion not implemented for cache type: {self.cache_type}")
                return 0
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting by symbol '{symbol}' from RDBMS cache: {e}")
            return 0
    
    def delete_by_pattern(self, pattern: str) -> int:
        """Delete all cache entries matching a pattern (legacy method)"""
        try:
            # Use DAO methods for deletion based on cache type
            if self.cache_type == CacheType.LLM_RESPONSE and self.dao:
                # Convert file pattern to SQL LIKE pattern
                sql_pattern = pattern.replace('*', '%').replace('?', '_')
                
                deleted_count = self.dao.delete_llm_responses_by_pattern(
                    symbol_pattern=sql_pattern,
                    form_type_pattern=sql_pattern
                )
                return deleted_count
            else:
                logger.warning(f"Delete by pattern operation not implemented for cache type: {self.cache_type}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting by pattern from RDBMS cache: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all data from RDBMS cache"""
        try:
            # Use DAO methods for deletion based on cache type
            if self.cache_type == CacheType.LLM_RESPONSE and self.dao:
                # Delete all LLM responses using wildcard pattern
                deleted_count = self.dao.delete_llm_responses_by_pattern(
                    symbol_pattern='%',
                    form_type_pattern='%'
                )
                logger.info(f"Cleared all RDBMS cache data ({deleted_count} entries)")
                return True
            else:
                logger.warning(f"Clear all operation not implemented for cache type: {self.cache_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing RDBMS cache: {e}")
            return False