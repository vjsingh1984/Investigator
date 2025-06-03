#!/usr/bin/env python3
"""
InvestiGator - Cache Manager
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Cache manager for coordinating multiple cache handlers
"""

from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
import logging

from .cache_types import CacheType
from .cache_base import CacheStorageHandler
from .file_cache_handler import FileCacheStorageHandler
from .rdbms_cache_handler import RdbmsCacheStorageHandler
from .parquet_cache_handler import ParquetCacheStorageHandler
from .memory_cache_handler import MemoryCacheStorageHandler

logger = logging.getLogger(__name__)


class CacheManager:
    """Manager for coordinating multiple cache handlers"""
    
    def __init__(self, config=None):
        self.handlers: Dict[CacheType, List[CacheStorageHandler]] = {}
        self.config = config
        self._initialize_default_handlers()
    
    def _initialize_default_handlers(self):
        """Initialize default cache handlers based on configuration - uniform across all types"""
        # Try to get config object if available
        config = self.config
        if not config:
            try:
                from config import get_config
                config = get_config()
            except:
                pass
                
        # Check if we should use disk/rdbms/memory caches
        use_disk = True
        use_rdbms = True
        use_memory = True
        
        if config and hasattr(config, 'cache_control'):
            use_disk = config.cache_control.use_disk_cache
            use_rdbms = config.cache_control.use_rdbms_cache
            use_memory = getattr(config.cache_control, 'use_memory_cache', True)
            
        # If no storage configured, no caching at all
        if not use_disk and not use_rdbms and not use_memory:
            logger.info("No cache storage configured - all operations will be on-the-fly")
            return
            
        # Register handlers uniformly for ALL cache types
        for cache_type in CacheType:
            # Memory handler with highest priority (fastest)
            if use_memory:
                memory_ttl = 15  # Default 15 minutes
                if config and hasattr(config, 'cache_control'):
                    memory_ttl = getattr(config.cache_control, 'memory_ttl_minutes', 15)
                
                memory_handler = MemoryCacheStorageHandler(
                    cache_type,
                    priority=30,  # Highest priority
                    ttl_minutes=memory_ttl
                )
                self.register_handler(cache_type, memory_handler)
            # Disk handler with higher priority
            if use_disk:
                # Determine base path based on cache type
                if cache_type in [CacheType.SEC_RESPONSE, CacheType.COMPANY_FACTS, CacheType.QUARTERLY_METRICS]:
                    base_path = Path("data/sec_cache")
                elif cache_type == CacheType.LLM_RESPONSE:
                    base_path = Path("data/llm_cache")
                elif cache_type == CacheType.TECHNICAL_DATA:
                    base_path = Path("data/technical_cache")
                elif cache_type == CacheType.SUBMISSION_DATA:
                    base_path = Path("data/sec_cache/submissions")
                else:
                    base_path = Path("data/cache")
                
                # Use Parquet for technical data (tabular), file handler for others (JSON)
                if cache_type == CacheType.TECHNICAL_DATA:
                    handler = ParquetCacheStorageHandler(
                        cache_type,
                        base_path,
                        priority=10,  # Disk has higher priority
                        config=config
                    )
                else:
                    handler = FileCacheStorageHandler(
                        cache_type,
                        base_path,
                        priority=10,
                        config=config
                    )
                self.register_handler(cache_type, handler)
                
            # RDBMS handler with lower priority
            if use_rdbms:
                # RDBMS doesn't support TECHNICAL_DATA (uses Parquet instead)
                if cache_type != CacheType.TECHNICAL_DATA:
                    try:
                        rdbms_handler = RdbmsCacheStorageHandler(
                            cache_type,
                            priority=5  # RDBMS has lower priority
                        )
                        self.register_handler(cache_type, rdbms_handler)
                    except ValueError as e:
                        logger.debug(f"RDBMS handler not available for {cache_type.value}: {e}")
        
    
    def register_handler(self, cache_type: CacheType, handler: CacheStorageHandler):
        """Register a cache handler"""
        if cache_type not in self.handlers:
            self.handlers[cache_type] = []
        self.handlers[cache_type].append(handler)
        # Sort by priority (highest first)
        self.handlers[cache_type].sort(key=lambda h: h.priority, reverse=True)
    
    def get(self, cache_type: CacheType, key: Union[Tuple, Dict]) -> Optional[Dict[str, Any]]:
        """
        Get data from cache, trying handlers in priority order (highest priority first)
        This ensures disk cache (higher priority) is checked before database (lower priority)
        
        Args:
            cache_type: Type of cache
            key: Cache key
            
        Returns:
            Cached data if found, None otherwise
        """
        # Check if caching is enabled
        if self.config and hasattr(self.config, 'cache_control'):
            if not self.config.cache_control.use_cache or not self.config.cache_control.read_from_cache:
                logger.debug(f"Cache READ disabled for {cache_type.value}")
                return None
                
            # Check specific cache type
            cache_type_map = {
                CacheType.SEC_RESPONSE: 'sec',
                CacheType.LLM_RESPONSE: 'llm',
                CacheType.TECHNICAL_DATA: 'technical',
                CacheType.SUBMISSION_DATA: 'submission',
                CacheType.COMPANY_FACTS: 'company_facts',
                CacheType.QUARTERLY_METRICS: 'quarterly_metrics'
            }
            
            if not self.config.cache_control.is_cache_type_enabled(cache_type.value):
                logger.debug(f"Cache READ disabled for specific type: {cache_type.value}")
                return None
                
            # Check force refresh
            if self.config.cache_control.force_refresh:
                logger.debug(f"Force refresh enabled globally, skipping cache for {cache_type.value}")
                return None
                
            # Check symbol-specific force refresh
            if self.config.cache_control.force_refresh_symbols and isinstance(key, (tuple, dict)):
                # Extract symbol from key
                if isinstance(key, tuple) and len(key) > 0:
                    symbol = key[0]
                elif isinstance(key, dict):
                    symbol = key.get('symbol')
                else:
                    symbol = None
                    
                if symbol and symbol in self.config.cache_control.force_refresh_symbols:
                    logger.debug(f"Force refresh enabled for symbol {symbol}, skipping cache")
                    return None
        
        handlers = self.handlers.get(cache_type, [])
        
        for handler in handlers:
            if handler.priority < 0:
                continue  # Skip handlers with negative priority (audit-only)
                
            try:
                result = handler.get(key)
                if result is not None:
                    logger.debug(f"Cache HIT [{handler.__class__.__name__}]: {cache_type.value} - Priority {handler.priority}")
                    
                    # If we got data from lower priority handler, write to higher priority handlers
                    # This promotes frequently accessed data to faster storage
                    self._promote_to_higher_priority(cache_type, key, result, handler.priority)
                    
                    return result
                else:
                    logger.debug(f"Cache MISS [{handler.__class__.__name__}]: {cache_type.value} - Priority {handler.priority}")
            except Exception as e:
                logger.warning(f"Cache lookup error [{handler.__class__.__name__}]: {e}")
                continue
        
        logger.debug(f"Cache MISS ALL handlers: {cache_type.value}")
        return None
    
    def set(self, cache_type: CacheType, key: Union[Tuple, Dict], value: Dict[str, Any]) -> bool:
        """
        Set data in ALL registered handlers for the cache type
        This ensures data is stored in both disk and database for redundancy
        
        Args:
            cache_type: Type of cache
            key: Cache key
            value: Data to cache
            
        Returns:
            True if at least one handler succeeded
        """
        # Check if caching is enabled
        if self.config and hasattr(self.config, 'cache_control'):
            if not self.config.cache_control.use_cache or not self.config.cache_control.write_to_cache:
                logger.debug(f"Cache WRITE disabled for {cache_type.value}")
                return False
                
            # Check specific cache type
            cache_type_map = {
                CacheType.SEC_RESPONSE: 'sec',
                CacheType.LLM_RESPONSE: 'llm',
                CacheType.TECHNICAL_DATA: 'technical',
                CacheType.SUBMISSION_DATA: 'submission',
                CacheType.COMPANY_FACTS: 'company_facts',
                CacheType.QUARTERLY_METRICS: 'quarterly_metrics'
            }
            
            if not self.config.cache_control.is_cache_type_enabled(cache_type.value):
                logger.debug(f"Cache WRITE disabled for specific type: {cache_type.value}")
                return False
        
        handlers = self.handlers.get(cache_type, [])
        success_count = 0
        total_handlers = len(handlers)
        
        for handler in handlers:
            try:
                if handler.set(key, value):
                    success_count += 1
                    logger.debug(f"Cache WRITE SUCCESS [{handler.__class__.__name__}]: {cache_type.value}")
                else:
                    logger.warning(f"Cache WRITE FAILED [{handler.__class__.__name__}]: {cache_type.value}")
            except Exception as e:
                logger.error(f"Cache write error [{handler.__class__.__name__}]: {e}")
        
        logger.info(f"Cache WRITE: {success_count}/{total_handlers} handlers succeeded for {cache_type.value}")
        return success_count > 0
    
    def _promote_to_higher_priority(self, cache_type: CacheType, key: Union[Tuple, Dict], 
                                   value: Dict[str, Any], found_priority: int):
        """
        Promote data to higher priority storage handlers when found in lower priority storage
        This ensures frequently accessed data migrates to faster storage (disk)
        
        Args:
            cache_type: Type of cache
            key: Cache key
            value: Data to promote
            found_priority: Priority of handler where data was found
        """
        handlers = self.handlers.get(cache_type, [])
        
        for handler in handlers:
            # Only promote to handlers with higher priority than where we found the data
            if handler.priority > found_priority and handler.priority >= 0:
                try:
                    if not handler.exists(key):
                        if handler.set(key, value):
                            logger.debug(f"Cache PROMOTED [{handler.__class__.__name__}]: {cache_type.value} from priority {found_priority} to {handler.priority}")
                except Exception as e:
                    logger.warning(f"Cache promotion error [{handler.__class__.__name__}]: {e}")
    
    def exists(self, cache_type: CacheType, key: Union[Tuple, Dict]) -> bool:
        """Check if key exists in any handler"""
        handlers = self.handlers.get(cache_type, [])
        
        for handler in handlers:
            if handler.priority < 0:
                continue
                
            if handler.exists(key):
                return True
        
        return False
    
    def delete(self, cache_type: CacheType, key: Union[Tuple, Dict]) -> bool:
        """Delete from all handlers"""
        handlers = self.handlers.get(cache_type, [])
        any_deleted = False
        
        for handler in handlers:
            try:
                if handler.delete(key):
                    any_deleted = True
            except Exception as e:
                logger.warning(f"Cache delete error [{handler.__class__.__name__}]: {e}")
        
        return any_deleted
    
    def delete_by_pattern(self, cache_type: CacheType, pattern: str) -> int:
        """Delete all cache entries matching a pattern across all handlers"""
        total_deleted = 0
        handlers = self.handlers.get(cache_type, [])
        
        for handler in handlers:
            try:
                deleted_count = handler.delete_by_pattern(pattern)
                total_deleted += deleted_count
            except Exception as e:
                logger.error(f"Handler {handler.__class__.__name__} failed to delete by pattern: {e}")
        
        logger.info(f"Total deleted by pattern '{pattern}' from {cache_type}: {total_deleted}")
        return total_deleted
    
    def clear_cache_type(self, cache_type: CacheType) -> bool:
        """Clear all data for a specific cache type across all handlers"""
        all_success = True
        handlers = self.handlers.get(cache_type, [])
        
        for handler in handlers:
            try:
                success = handler.clear_all()
                if not success:
                    all_success = False
            except Exception as e:
                logger.error(f"Handler {handler.__class__.__name__} failed to clear: {e}")
                all_success = False
        
        logger.info(f"Cleared cache type {cache_type}: {'success' if all_success else 'partial/failed'}")
        return all_success
    
    def clear_all_caches(self) -> bool:
        """Clear all data from all cache types and handlers"""
        all_success = True
        
        for cache_type in self.handlers.keys():
            success = self.clear_cache_type(cache_type)
            if not success:
                all_success = False
        
        logger.info(f"Cleared all caches: {'success' if all_success else 'partial/failed'}")
        return all_success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'cache_types': {},
            'total_handlers': 0,
            'handler_summary': {}
        }
        
        for cache_type, handlers in self.handlers.items():
            cache_type_stats = {
                'handler_count': len(handlers),
                'handlers': []
            }
            
            for handler in handlers:
                handler_stats = {
                    'handler_type': handler.__class__.__name__,
                    'priority': handler.priority,
                    'cache_type': handler.cache_type.value
                }
                
                # Get handler-specific stats if available
                if hasattr(handler, 'get_stats'):
                    try:
                        handler_stats.update(handler.get_stats())
                    except Exception as e:
                        handler_stats['stats_error'] = str(e)
                
                cache_type_stats['handlers'].append(handler_stats)
                stats['total_handlers'] += 1
                
                # Update handler summary
                handler_name = handler.__class__.__name__
                if handler_name not in stats['handler_summary']:
                    stats['handler_summary'][handler_name] = 0
                stats['handler_summary'][handler_name] += 1
            
            stats['cache_types'][cache_type.value] = cache_type_stats
        
        return stats


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager