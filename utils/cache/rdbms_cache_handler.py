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
            get_sec_response_store_dao,
            get_llm_response_store_dao,
            get_quarterly_metrics_dao,
            DatabaseManager
        )
        
        # Always initialize DatabaseManager for delete operations
        self.db_manager = DatabaseManager()
        
        # Initialize appropriate DAO based on cache type
        if cache_type == CacheType.SEC_RESPONSE:
            self.dao = get_sec_response_store_dao()
        elif cache_type == CacheType.LLM_RESPONSE:
            self.dao = get_llm_response_store_dao()
        elif cache_type == CacheType.QUARTERLY_METRICS:
            self.dao = get_quarterly_metrics_dao()
        elif cache_type in [CacheType.SUBMISSION_DATA, CacheType.COMPANY_FACTS]:
            # For submission/company facts, use DatabaseManager directly
            self.dao = None  # Use raw SQL for these operations
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
                # Fetch from all_submission_store or materialized view
                symbol = key_dict.get('symbol')
                cik = key_dict.get('cik')
                
                if symbol:
                    from utils.db import get_all_submission_store_dao
                    dao = get_all_submission_store_dao()
                    
                    # First try to get recent earnings submissions from materialized view
                    recent_submissions = dao.get_recent_earnings_submissions(symbol, limit=4)
                    if recent_submissions:
                        logger.debug(f"Cache hit (RDBMS): Recent submissions for {symbol} from materialized view")
                        return {
                            'symbol': symbol,
                            'cik': recent_submissions[0]['cik'],
                            'company_name': recent_submissions[0]['company_name'],
                            'recent_submissions': recent_submissions
                        }
                    
                    # Fallback to all_submission_store if we have CIK
                    if cik:
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
                    from utils.db import get_all_companyfacts_store_dao
                    dao = get_all_companyfacts_store_dao()
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
                    prompt_context=value.get('prompt_context', {}),
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
                    from utils.db import get_all_submission_store_dao
                    dao = get_all_submission_store_dao()
                    
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
                    from utils.db import get_all_companyfacts_store_dao
                    dao = get_all_companyfacts_store_dao()
                    
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
            category = key_dict.get('category', '')
            
            if not symbol:
                logger.warning("Cannot delete from RDBMS cache without symbol")
                return False
            
            with self.db_manager.get_session() as session:
                # Delete from llm_response_store
                if category:
                    result = session.execute(
                        text("DELETE FROM llm_response_store WHERE symbol = :symbol AND response_type LIKE :category"),
                        {"symbol": symbol, "category": f"%{category}%"}
                    )
                else:
                    result = session.execute(
                        text("DELETE FROM llm_response_store WHERE symbol = :symbol"),
                        {"symbol": symbol}
                    )
                
                deleted_count = result.rowcount
                session.commit()
                
                logger.info(f"Deleted {deleted_count} entries from RDBMS cache for symbol '{symbol}'")
                return deleted_count > 0
                
        except Exception as e:
            logger.error(f"Error deleting from RDBMS cache: {e}")
            return False
    
    def delete_by_pattern(self, pattern: str) -> int:
        """Delete all cache entries matching a pattern"""
        try:
            with self.db_manager.get_session() as session:
                # Convert file pattern to SQL LIKE pattern
                sql_pattern = pattern.replace('*', '%').replace('?', '_')
                
                result = session.execute(
                    text("DELETE FROM llm_response_store WHERE symbol LIKE :pattern OR response_type LIKE :pattern"),
                    {"pattern": sql_pattern}
                )
                
                deleted_count = result.rowcount
                session.commit()
                
                logger.info(f"Deleted {deleted_count} entries from RDBMS cache matching pattern '{pattern}'")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error deleting by pattern from RDBMS cache: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all data from RDBMS cache"""
        try:
            with self.db_manager.get_session() as session:
                # Get count before deletion
                result = session.execute(text("SELECT COUNT(*) FROM llm_response_store"))
                count = result.scalar()
                
                # Clear all LLM response cache data
                session.execute(text("DELETE FROM llm_response_store"))
                session.commit()
                
                logger.info(f"Cleared all RDBMS cache data ({count} entries)")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing RDBMS cache: {e}")
            return False