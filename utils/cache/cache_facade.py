#!/usr/bin/env python3
"""
InvestiGator - Cache Facade with API Fallback
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Provides simplified interface combining cache manager with API fallback capabilities
Implements Chain of Responsibility pattern for cache → API workflow
"""

import logging
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime

from .cache_manager import get_cache_manager
from .cache_types import CacheType
from utils.api_client import SECAPIClient
from config import get_config


class CacheFacade:
    """
    Unified facade for cache operations with automatic API fallback
    Combines robust utils.cache system with chain-of-responsibility fallback
    """
    
    def __init__(self, config=None):
        """Initialize cache facade with configuration"""
        self.config = config or get_config()
        self.cache_manager = get_cache_manager()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize API client for fallback
        self.api_client = SECAPIClient(
            user_agent=self.config.sec.user_agent,
            config=self.config
        )
        
        # Configuration flags
        cache_config = getattr(self.config, 'cache', None) or getattr(self.config, 'cache_control', None)
        self.enable_api_fallback = getattr(cache_config, 'enable_api_fallback', True) if cache_config else True
        self.auto_cache_api_results = getattr(cache_config, 'auto_cache_api_results', True) if cache_config else True
        
        self.logger.info("Cache facade initialized with API fallback support")
    
    def get_company_facts(self, symbol: str, cik: str = None) -> Optional[Dict[str, Any]]:
        """
        Get company facts with automatic cache → API fallback
        
        Args:
            symbol: Stock symbol
            cik: SEC CIK identifier (optional, will be resolved if not provided)
            
        Returns:
            Company facts data or None if not found
        """
        # Resolve CIK upfront - required for both cache and API
        resolved_cik = self._resolve_cik(symbol, cik)
        if not resolved_cik:
            self.logger.warning(f"Could not resolve CIK for symbol: {symbol}")
            return None
        
        # Try cache first
        cache_key = {'symbol': symbol}
        cached_result = self.cache_manager.get(CacheType.COMPANY_FACTS, cache_key)
        
        if cached_result:
            self.logger.debug(f"Cache HIT for company facts: {symbol}")
            return cached_result
        
        # Fallback to API if enabled
        if self.enable_api_fallback:
            return self._api_fallback_company_facts(symbol, resolved_cik, cache_key)
        
        self.logger.debug(f"Cache MISS for company facts: {symbol} (API fallback disabled)")
        return None
    
    def get_submissions(self, symbol: str, cik: str = None, limit: int = 10) -> Optional[Dict[str, Any]]:
        """
        Get submissions with automatic cache → API fallback
        
        Args:
            symbol: Stock symbol
            cik: SEC CIK identifier (optional, will be resolved if not provided)
            limit: Maximum number of submissions to return
            
        Returns:
            Submissions data or None if not found
        """
        # Resolve CIK upfront - required for both cache and API
        resolved_cik = self._resolve_cik(symbol, cik)
        if not resolved_cik:
            self.logger.warning(f"Could not resolve CIK for symbol: {symbol}")
            return None
        
        # Try cache first
        cache_key = (symbol, f"recent_{limit}")
        cached_result = self.cache_manager.get(CacheType.SUBMISSION, cache_key)
        
        if cached_result:
            self.logger.debug(f"Cache HIT for submissions: {symbol}")
            return cached_result
        
        # Fallback to API if enabled
        if self.enable_api_fallback:
            return self._api_fallback_submissions(symbol, resolved_cik, limit, cache_key)
        
        self.logger.debug(f"Cache MISS for submissions: {symbol} (API fallback disabled)")
        return None
    
    def get_with_custom_fallback(
        self, 
        cache_type: CacheType, 
        cache_key: Any, 
        fallback_func: Callable[[], Any],
        cache_result: bool = True
    ) -> Optional[Any]:
        """
        Generic get with custom fallback function
        
        Args:
            cache_type: Type of cache to check
            cache_key: Key to look up in cache
            fallback_func: Function to call if cache miss
            cache_result: Whether to cache the fallback result
            
        Returns:
            Cached data or result from fallback function
        """
        # Try cache first
        cached_result = self.cache_manager.get(cache_type, cache_key)
        
        if cached_result is not None:
            self.logger.debug(f"Cache HIT for {cache_type.value}: {cache_key}")
            return cached_result
        
        # Execute fallback function
        try:
            self.logger.debug(f"Cache MISS for {cache_type.value}: {cache_key}, executing fallback")
            result = fallback_func()
            
            if result is not None and cache_result:
                # Cache the result for future use
                success = self.cache_manager.set(cache_type, cache_key, result)
                if success:
                    self.logger.debug(f"Cached fallback result for {cache_type.value}: {cache_key}")
                else:
                    self.logger.warning(f"Failed to cache fallback result for {cache_type.value}: {cache_key}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fallback function failed for {cache_type.value}: {cache_key} - {e}")
            return None
    
    def get_llm_response(
        self, 
        symbol: str, 
        llm_type: str, 
        form_type: str = "N/A", 
        period: str = "N/A"
    ) -> Optional[Dict[str, Any]]:
        """
        Get LLM response from cache (no API fallback for LLM responses)
        
        Args:
            symbol: Stock symbol
            llm_type: Type of LLM analysis (sec, ta, full)
            form_type: SEC form type
            period: Analysis period
            
        Returns:
            LLM response data or None
        """
        cache_key = {
            'symbol': symbol,
            'form_type': form_type,
            'period': period,
            'llm_type': llm_type
        }
        
        result = self.cache_manager.get(CacheType.LLM_RESPONSE, cache_key)
        
        if result:
            self.logger.debug(f"Cache HIT for LLM response: {symbol}/{llm_type}")
        else:
            self.logger.debug(f"Cache MISS for LLM response: {symbol}/{llm_type}")
        
        return result
    
    def cache_llm_response(
        self,
        symbol: str,
        llm_type: str,
        response_data: Dict[str, Any],
        form_type: str = "N/A",
        period: str = "N/A"
    ) -> bool:
        """
        Cache an LLM response
        
        Args:
            symbol: Stock symbol
            llm_type: Type of LLM analysis
            response_data: Response data to cache
            form_type: SEC form type
            period: Analysis period
            
        Returns:
            True if caching successful
        """
        cache_key = {
            'symbol': symbol,
            'form_type': form_type,
            'period': period,
            'llm_type': llm_type
        }
        
        success = self.cache_manager.set(CacheType.LLM_RESPONSE, cache_key, response_data)
        
        if success:
            self.logger.debug(f"Cached LLM response: {symbol}/{llm_type}")
        else:
            self.logger.warning(f"Failed to cache LLM response: {symbol}/{llm_type}")
        
        return success
    
    def get_quarterly_metrics(self, symbol: str, period: str) -> Optional[Dict[str, Any]]:
        """
        Get quarterly metrics from cache
        
        Args:
            symbol: Stock symbol
            period: Quarter period (e.g., "2024Q1")
            
        Returns:
            Quarterly metrics or None
        """
        cache_key = (symbol, period)
        result = self.cache_manager.get(CacheType.QUARTERLY_METRICS, cache_key)
        
        if result:
            self.logger.debug(f"Cache HIT for quarterly metrics: {symbol}/{period}")
        else:
            self.logger.debug(f"Cache MISS for quarterly metrics: {symbol}/{period}")
        
        return result
    
    def invalidate_symbol_cache(self, symbol: str) -> int:
        """
        Invalidate all cache entries for a specific symbol
        
        Args:
            symbol: Symbol to invalidate
            
        Returns:
            Number of cache entries deleted
        """
        total_deleted = 0
        
        # Invalidate across all cache types
        for cache_type in CacheType:
            try:
                deleted = self.cache_manager.delete_by_pattern(cache_type, f"*{symbol}*")
                total_deleted += deleted
                if deleted > 0:
                    self.logger.info(f"Invalidated {deleted} {cache_type.value} entries for {symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to invalidate {cache_type.value} for {symbol}: {e}")
        
        self.logger.info(f"Total cache entries invalidated for {symbol}: {total_deleted}")
        return total_deleted
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            stats = self.cache_manager.get_stats()
            
            # Add facade-specific information
            stats['facade_info'] = {
                'api_fallback_enabled': self.enable_api_fallback,
                'auto_cache_api_results': self.auto_cache_api_results,
                'has_api_client': self.api_client is not None
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {
                'error': str(e),
                'facade_info': {
                    'api_fallback_enabled': self.enable_api_fallback,
                    'auto_cache_api_results': self.auto_cache_api_results
                }
            }
    
    def _resolve_cik(self, symbol: str, provided_cik: str = None) -> Optional[str]:
        """
        Resolve CIK for a symbol, ensuring it's in proper zero-padded format
        
        Args:
            symbol: Stock symbol
            provided_cik: CIK if already known
            
        Returns:
            Zero-padded CIK string (10 digits) or None if not found
        """
        if provided_cik:
            try:
                # Convert to integer and back to ensure valid CIK
                cik_int = int(provided_cik)
                if cik_int > 0:
                    return f"{cik_int:010d}"
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid CIK format provided: {provided_cik}")
        
        # Look up CIK using ticker-CIK mapper
        try:
            from utils.ticker_cik_mapper import TickerCIKMapper
            mapper = TickerCIKMapper()
            cik = mapper.get_cik(symbol)
            if cik:
                cik_int = int(cik)
                if cik_int > 0:
                    return f"{cik_int:010d}"
        except Exception as e:
            self.logger.error(f"Failed to resolve CIK for {symbol}: {e}")
        
        return None
    
    def _api_fallback_company_facts(self, symbol: str, cik: str, cache_key: Any) -> Optional[Dict[str, Any]]:
        """API fallback for company facts - CIK is already resolved and validated"""
        try:
            # CIK is already resolved and zero-padded, prepare for API call
            cik_for_api = str(int(cik))  # Remove leading zeros for API call
            
            # Fetch from API
            self.logger.info(f"Fetching company facts from API: {symbol} (CIK: {cik_for_api})")
            facts = self.api_client.get_company_facts(cik_for_api)
            
            if facts and self.auto_cache_api_results:
                # Wrap facts with metadata including the resolved CIK
                cache_data = {
                    'symbol': symbol,
                    'cik': cik,  # Already zero-padded
                    'companyfacts': facts,
                    'metadata': {
                        'cik': cik,
                        'symbol': symbol,
                        'api_cik': cik_for_api,
                        'fetched_at': datetime.now().isoformat()
                    }
                }
                
                # Cache the wrapped result
                success = self.cache_manager.set(CacheType.COMPANY_FACTS, cache_key, cache_data)
                if success:
                    self.logger.info(f"Cached API company facts for {symbol}")
                else:
                    self.logger.warning(f"Failed to cache API company facts for {symbol}")
                
                # Return the facts data (unwrapped for backward compatibility)
                return facts
            
            return facts
            
        except Exception as e:
            self.logger.error(f"API fallback failed for company facts {symbol}: {e}")
            return None
    
    def _api_fallback_submissions(self, symbol: str, cik: str, limit: int, cache_key: Any) -> Optional[Dict[str, Any]]:
        """API fallback for submissions - CIK is already resolved and validated"""
        try:
            # CIK is already resolved and zero-padded, prepare for API call
            cik_for_api = str(int(cik))  # Remove leading zeros for API call
            
            # Fetch from API
            self.logger.info(f"Fetching submissions from API: {symbol} (CIK: {cik_for_api})")
            submissions = self.api_client.get_submissions(cik_for_api)
            
            if submissions and self.auto_cache_api_results:
                # Wrap submissions with metadata including the resolved CIK
                cache_data = {
                    'symbol': symbol,
                    'cik': cik,  # Already zero-padded
                    'submissions': submissions,
                    'metadata': {
                        'cik': cik,
                        'symbol': symbol,
                        'api_cik': cik_for_api,
                        'limit': limit,
                        'fetched_at': datetime.now().isoformat()
                    }
                }
                
                # Cache the wrapped result
                success = self.cache_manager.set(CacheType.SUBMISSION, cache_key, cache_data)
                if success:
                    self.logger.info(f"Cached API submissions for {symbol}")
                else:
                    self.logger.warning(f"Failed to cache API submissions for {symbol}")
                
                # Return the submissions data (unwrapped for backward compatibility)
                return submissions
            
            return submissions
            
        except Exception as e:
            self.logger.error(f"API fallback failed for submissions {symbol}: {e}")
            return None


# Global facade instance
_cache_facade = None

def get_cache_facade(config=None) -> CacheFacade:
    """Get global cache facade instance"""
    global _cache_facade
    if _cache_facade is None:
        _cache_facade = CacheFacade(config)
    return _cache_facade


def reset_cache_facade():
    """Reset global cache facade (for testing)"""
    global _cache_facade
    _cache_facade = None