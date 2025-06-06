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
import time
from collections import defaultdict, deque
from threading import Lock
from datetime import datetime, timedelta

from .cache_types import CacheType
from .cache_base import CacheStorageHandler
from .file_cache_handler import FileCacheStorageHandler
from .rdbms_cache_handler import RdbmsCacheStorageHandler
from .parquet_cache_handler import ParquetCacheStorageHandler
# Memory cache handler removed - using only disk and RDBMS

logger = logging.getLogger(__name__)


class CacheManager:
    """Manager for coordinating multiple cache handlers with comprehensive performance tracking"""
    
    def __init__(self, config=None):
        self.handlers: Dict[CacheType, List[CacheStorageHandler]] = {}
        self.config = config
        
        # Performance tracking
        self._stats_lock = Lock()
        self._operation_stats = defaultdict(lambda: {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'errors': 0,
            'total_time_ms': 0.0,
            'avg_time_ms': 0.0,
            'handler_performance': defaultdict(lambda: {
                'hits': 0,
                'misses': 0,
                'writes': 0,
                'errors': 0,
                'total_time_ms': 0.0
            })
        })
        
        # Recent operations for debugging (last 100 per cache type)
        self._recent_operations = defaultdict(lambda: deque(maxlen=100))
        
        self._initialize_default_handlers()
    
    def _initialize_default_handlers(self):
        """Initialize cache handlers based on detailed configuration"""
        # Try to get config object if available
        config = self.config
        if not config:
            try:
                from config import get_config
                config = get_config()
            except:
                pass
                
        if not config or not hasattr(config, 'cache_control'):
            logger.warning("No cache configuration found - using minimal defaults")
            return
            
        cache_control = config.cache_control
        
        # If no storage configured, no caching at all
        if not cache_control.use_cache:
            logger.info("Cache disabled - all operations will be on-the-fly")
            return
            
        # Register handlers for each cache type based on configuration
        for cache_type in CacheType:
            cache_type_name = cache_type.value
            cache_config = cache_control.get_cache_config(cache_type_name)
            
            if not cache_config or not cache_config.enabled:
                logger.debug(f"Cache disabled for {cache_type_name}")
                continue
                
            logger.debug(f"Initializing cache handlers for {cache_type_name}")
            
            # Initialize disk handler if enabled
            if cache_config.disk.enabled:
                try:
                    disk_settings = cache_config.disk.settings
                    base_path = Path(disk_settings.get('base_path', f'data/{cache_type_name}_cache'))
                    
                    # Use Parquet for technical data, file handler for others
                    if cache_type == CacheType.TECHNICAL_DATA:
                        handler = ParquetCacheStorageHandler(
                            cache_type,
                            base_path,
                            priority=cache_config.disk.priority,
                            config=config
                        )
                    else:
                        handler = FileCacheStorageHandler(
                            cache_type,
                            base_path,
                            priority=cache_config.disk.priority,
                            config=config
                        )
                    self.register_handler(cache_type, handler)
                    logger.debug(f"Registered disk handler for {cache_type_name} with priority {cache_config.disk.priority}")
                except Exception as e:
                    logger.error(f"Failed to initialize disk handler for {cache_type_name}: {e}")
                
            # Initialize RDBMS handler if enabled
            if cache_config.rdbms.enabled:
                try:
                    rdbms_handler = RdbmsCacheStorageHandler(
                        cache_type,
                        priority=cache_config.rdbms.priority
                    )
                    self.register_handler(cache_type, rdbms_handler)
                    logger.debug(f"Registered RDBMS handler for {cache_type_name} with priority {cache_config.rdbms.priority}")
                except Exception as e:
                    logger.debug(f"RDBMS handler not available for {cache_type_name}: {e}")
        
    
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
        operation_start = time.time()
        key_str = self._format_key_for_logging(key)
        
        # Check if caching is enabled
        if self.config and hasattr(self.config, 'cache_control'):
            if not self.config.cache_control.use_cache or not self.config.cache_control.read_from_cache:
                logger.debug(f"🚫 Cache READ disabled globally for {cache_type.value}")
                return None
                
            if not self.config.cache_control.is_cache_type_enabled(cache_type.value):
                logger.debug(f"🚫 Cache READ disabled for type: {cache_type.value}")
                return None
                
            # Check force refresh
            if self.config.cache_control.force_refresh:
                logger.debug(f"🔄 Force refresh enabled globally, skipping cache for {cache_type.value}")
                return None
                
            # Check symbol-specific force refresh
            if self.config.cache_control.force_refresh_symbols and isinstance(key, (tuple, dict)):
                symbol = self._extract_symbol_from_key(key)
                if symbol and symbol in self.config.cache_control.force_refresh_symbols:
                    logger.debug(f"🔄 Force refresh enabled for symbol {symbol}, skipping cache")
                    return None
        
        handlers = self.handlers.get(cache_type, [])
        if not handlers:
            logger.warning(f"⚠️ No cache handlers configured for {cache_type.value}")
            return None
        
        # Track attempted handlers for logging
        handler_attempts = []
        
        for handler in handlers:
            if handler.priority < 0:
                continue  # Skip handlers with negative priority (audit-only)
                
            handler_start = time.time()
            handler_name = handler.__class__.__name__
            
            try:
                result = handler.get(key)
                handler_time = (time.time() - handler_start) * 1000
                
                handler_attempts.append({
                    'handler': handler_name,
                    'priority': handler.priority,
                    'time_ms': handler_time,
                    'result': 'HIT' if result is not None else 'MISS'
                })
                
                if result is not None:
                    total_time = (time.time() - operation_start) * 1000
                    
                    # Calculate TTL remaining for enhanced logging
                    ttl_remaining = self._calculate_ttl_remaining(result, cache_type)
                    
                    # For LLM responses, add the specific type for better visibility
                    ttl_info = ttl_remaining
                    if cache_type == CacheType.LLM_RESPONSE:
                        llm_type = self._determine_llm_response_type(result)
                        ttl_info = f"{ttl_remaining} ({llm_type})"
                    
                    # Enhanced logging with key info and TTL
                    logger.info(f"✅ Cache HIT [{handler_name}]: {cache_type.value} | Key: {key_str} | Priority: {handler.priority} | Time: {handler_time:.1f}ms | Total: {total_time:.1f}ms | TTL: {ttl_info}")
                    
                    # Update statistics
                    self._update_stats(cache_type, 'hit', total_time, handler_name, handler_time)
                    
                    # Log operation details
                    self._log_operation(cache_type, 'GET_HIT', key_str, handler_name, total_time)
                    
                    # Promote to higher priority if applicable
                    self._promote_to_higher_priority(cache_type, key, result, handler.priority)
                    
                    return result
                else:
                    logger.debug(f"❌ Cache MISS [{handler_name}]: {cache_type.value} | Key: {key_str} | Priority: {handler.priority} | Time: {handler_time:.1f}ms")
                    self._update_handler_stats(cache_type, handler_name, 'miss', handler_time)
                    
            except Exception as e:
                handler_time = (time.time() - handler_start) * 1000
                logger.error(f"💥 Cache ERROR [{handler_name}]: {cache_type.value} | Key: {key_str} | Error: {e} | Time: {handler_time:.1f}ms")
                self._update_stats(cache_type, 'error', handler_time, handler_name, handler_time)
                handler_attempts.append({
                    'handler': handler_name,
                    'priority': handler.priority,
                    'time_ms': handler_time,
                    'result': f'ERROR: {str(e)[:50]}'
                })
                continue
        
        total_time = (time.time() - operation_start) * 1000
        
        # Comprehensive miss logging
        handler_summary = " | ".join([f"{h['handler']}({h['priority']}):{h['result']}" for h in handler_attempts])
        logger.info(f"❌ Cache MISS ALL: {cache_type.value} | Key: {key_str} | Handlers: {handler_summary} | Total: {total_time:.1f}ms")
        
        # Update statistics
        self._update_stats(cache_type, 'miss', total_time)
        self._log_operation(cache_type, 'GET_MISS', key_str, 'ALL_HANDLERS', total_time)
        
        return None
    
    def set(self, cache_type: CacheType, key: Union[Tuple, Dict], value: Dict[str, Any]) -> bool:
        """
        Set data in cache handlers based on priority.
        
        If data exists in higher priority cache, skip writing to lower priority.
        If data doesn't exist or needs updating, write to all handlers.
        File cache will overwrite, RDBMS will upsert.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            value: Data to cache
            
        Returns:
            True if at least one handler succeeded
        """
        operation_start = time.time()
        key_str = self._format_key_for_logging(key)
        
        # Check if caching is enabled
        if self.config and hasattr(self.config, 'cache_control'):
            if not self.config.cache_control.use_cache or not self.config.cache_control.write_to_cache:
                logger.debug(f"🚫 Cache WRITE disabled globally for {cache_type.value}")
                return False
                
            if not self.config.cache_control.is_cache_type_enabled(cache_type.value):
                logger.debug(f"🚫 Cache WRITE disabled for type: {cache_type.value}")
                return False
        
        handlers = self.handlers.get(cache_type, [])
        if not handlers:
            logger.warning(f"⚠️ No cache handlers configured for {cache_type.value}")
            return False
        
        success_count = 0
        total_handlers = len(handlers)
        write_attempts = []
        
        # Check if data already exists in any handler (in priority order)
        existing_priority = -1
        existing_handler = None
        
        for handler in handlers:
            if handler.exists(key):
                existing_priority = handler.priority
                existing_handler = handler.__class__.__name__
                logger.debug(f"📁 Cache EXISTS in [{existing_handler}] with priority {handler.priority}")
                break
        
        # Write to handlers based on existence check
        for handler in handlers:
            handler_name = handler.__class__.__name__
            
            try:
                # Skip writing to lower priority handlers if data exists in higher priority
                if existing_priority > 0 and handler.priority < existing_priority:
                    logger.debug(f"⏩ Cache SKIP WRITE [{handler_name}]: exists in higher priority cache ({existing_handler})")
                    write_attempts.append({
                        'handler': handler_name,
                        'priority': handler.priority,
                        'result': 'SKIPPED',
                        'reason': f'exists_in_{existing_handler}'
                    })
                    continue
                
                # Always write to handlers - file will overwrite, RDBMS will upsert
                handler_start = time.time()
                success = handler.set(key, value)
                handler_time = (time.time() - handler_start) * 1000
                
                write_attempts.append({
                    'handler': handler_name,
                    'priority': handler.priority,
                    'time_ms': handler_time,
                    'result': 'SUCCESS' if success else 'FAILED'
                })
                
                if success:
                    success_count += 1
                    logger.info(f"✅ Cache WRITE SUCCESS [{handler_name}]: {cache_type.value} | Key: {key_str} | Time: {handler_time:.1f}ms")
                    self._update_stats(cache_type, 'write', handler_time, handler_name, handler_time)
                else:
                    logger.warning(f"❌ Cache WRITE FAILED [{handler_name}]: {cache_type.value} | Key: {key_str} | Time: {handler_time:.1f}ms")
                    self._update_handler_stats(cache_type, handler_name, 'error', handler_time)
                    
            except Exception as e:
                handler_time = (time.time() - handler_start) if 'handler_start' in locals() else 0
                handler_time_ms = handler_time * 1000
                logger.error(f"💥 Cache WRITE ERROR [{handler_name}]: {cache_type.value} | Key: {key_str} | Error: {e} | Time: {handler_time_ms:.1f}ms")
                
                write_attempts.append({
                    'handler': handler_name,
                    'priority': handler.priority,
                    'time_ms': handler_time_ms,
                    'result': f'ERROR: {str(e)[:50]}'
                })
                
                self._update_stats(cache_type, 'error', handler_time_ms, handler_name, handler_time_ms)
        
        total_time = (time.time() - operation_start) * 1000
        
        # Comprehensive write result logging
        handler_summary = " | ".join([f"{a['handler']}({a['priority']}):{a['result']}" for a in write_attempts])
        
        if success_count > 0:
            logger.info(f"✅ Cache WRITE COMPLETE: {cache_type.value} | Key: {key_str} | Success: {success_count}/{total_handlers} | Handlers: {handler_summary} | Total: {total_time:.1f}ms")
        else:
            logger.warning(f"❌ Cache WRITE FAILED ALL: {cache_type.value} | Key: {key_str} | Handlers: {handler_summary} | Total: {total_time:.1f}ms")
        
        # Log operation details
        operation_result = 'WRITE_SUCCESS' if success_count > 0 else 'WRITE_FAILED'
        self._log_operation(cache_type, operation_result, key_str, f"{success_count}/{total_handlers}_handlers", total_time)
        
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
        operation_start = time.time()
        key_str = self._format_key_for_logging(key)
        handlers = self.handlers.get(cache_type, [])
        
        if not handlers:
            logger.warning(f"⚠️ No cache handlers configured for {cache_type.value}")
            return False
        
        existence_checks = []
        
        for handler in handlers:
            if handler.priority < 0:
                continue
                
            handler_start = time.time()
            handler_name = handler.__class__.__name__
            
            try:
                exists = handler.exists(key)
                handler_time = (time.time() - handler_start) * 1000
                
                existence_checks.append({
                    'handler': handler_name,
                    'priority': handler.priority,
                    'time_ms': handler_time,
                    'exists': exists
                })
                
                if exists:
                    total_time = (time.time() - operation_start) * 1000
                    logger.info(f"📁 Cache EXISTS [{handler_name}]: {cache_type.value} | Key: {key_str} | Priority: {handler.priority} | Time: {handler_time:.1f}ms | Total: {total_time:.1f}ms")
                    return True
                else:
                    logger.debug(f"📂 Cache NOT EXISTS [{handler_name}]: {cache_type.value} | Key: {key_str} | Priority: {handler.priority} | Time: {handler_time:.1f}ms")
                    
            except Exception as e:
                handler_time = (time.time() - handler_start) * 1000
                logger.error(f"💥 Cache EXISTS ERROR [{handler_name}]: {cache_type.value} | Key: {key_str} | Error: {e} | Time: {handler_time:.1f}ms")
                existence_checks.append({
                    'handler': handler_name,
                    'priority': handler.priority,
                    'time_ms': handler_time,
                    'exists': False,
                    'error': str(e)[:50]
                })
        
        total_time = (time.time() - operation_start) * 1000
        
        # Summary logging
        handler_summary = " | ".join([f"{c['handler']}({c['priority']}):{'EXISTS' if c['exists'] else 'NOT_EXISTS'}" for c in existence_checks])
        logger.info(f"📂 Cache NOT EXISTS ALL: {cache_type.value} | Key: {key_str} | Handlers: {handler_summary} | Total: {total_time:.1f}ms")
        
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
    
    def delete_by_symbol(self, symbol: str, cache_types: Optional[List[CacheType]] = None) -> Dict[str, int]:
        """
        Delete all cache entries for a specific symbol across specified cache types.
        Optimized for symbol-based cleanup using targeted deletion methods.
        
        Args:
            symbol: Stock symbol to delete (e.g., 'AAPL')
            cache_types: List of cache types to clean, or None for all types
            
        Returns:
            Dictionary with cache_type -> deleted_count mapping
        """
        operation_start = time.time()
        symbol = symbol.upper()  # Normalize to uppercase
        
        # Use all cache types if none specified
        target_cache_types = cache_types or list(self.handlers.keys())
        
        deletion_results = {}
        total_deleted = 0
        
        logger.info(f"🧹 Starting symbol-based cleanup for {symbol} across {len(target_cache_types)} cache types")
        
        for cache_type in target_cache_types:
            handlers = self.handlers.get(cache_type, [])
            if not handlers:
                continue
                
            cache_type_deleted = 0
            handler_results = []
            
            for handler in handlers:
                handler_start = time.time()
                handler_name = handler.__class__.__name__
                
                try:
                    # Use optimized symbol-based deletion
                    if hasattr(handler, 'delete_by_symbol'):
                        deleted_count = handler.delete_by_symbol(symbol)
                    else:
                        # Fallback to pattern-based deletion for compatibility
                        deleted_count = handler.delete_by_pattern(f"*{symbol}*")
                    
                    handler_time = (time.time() - handler_start) * 1000
                    cache_type_deleted += deleted_count
                    
                    handler_results.append({
                        'handler': handler_name,
                        'deleted': deleted_count,
                        'time_ms': handler_time
                    })
                    
                    if deleted_count > 0:
                        logger.info(f"🗑️ Symbol cleanup [{handler_name}]: {cache_type.value} | Symbol: {symbol} | Deleted: {deleted_count} | Time: {handler_time:.1f}ms")
                    else:
                        logger.debug(f"🔍 Symbol cleanup [{handler_name}]: {cache_type.value} | Symbol: {symbol} | No entries found | Time: {handler_time:.1f}ms")
                        
                except Exception as e:
                    handler_time = (time.time() - handler_start) * 1000
                    logger.error(f"💥 Symbol cleanup ERROR [{handler_name}]: {cache_type.value} | Symbol: {symbol} | Error: {e} | Time: {handler_time:.1f}ms")
                    handler_results.append({
                        'handler': handler_name,
                        'deleted': 0,
                        'time_ms': handler_time,
                        'error': str(e)
                    })
            
            deletion_results[cache_type.value] = cache_type_deleted
            total_deleted += cache_type_deleted
            
            # Summary logging for this cache type
            if cache_type_deleted > 0:
                handler_summary = " | ".join([f"{r['handler']}:{r['deleted']}" for r in handler_results if r['deleted'] > 0])
                logger.info(f"✅ Symbol cleanup COMPLETE: {cache_type.value} | Symbol: {symbol} | Total deleted: {cache_type_deleted} | Handlers: {handler_summary}")
        
        total_time = (time.time() - operation_start) * 1000
        
        # Final summary
        if total_deleted > 0:
            cache_summary = " | ".join([f"{ct}:{count}" for ct, count in deletion_results.items() if count > 0])
            logger.info(f"🎯 Symbol cleanup SUCCESS: Symbol: {symbol} | Total deleted: {total_deleted} | Cache types: {cache_summary} | Total time: {total_time:.1f}ms")
        else:
            logger.info(f"🔍 Symbol cleanup COMPLETE: Symbol: {symbol} | No entries found across all cache types | Total time: {total_time:.1f}ms")
        
        return deletion_results
    
    def delete_by_pattern(self, cache_type: CacheType, pattern: str) -> int:
        """Delete all cache entries matching a pattern across all handlers (legacy method)"""
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
    
    def _format_key_for_logging(self, key: Union[Tuple, Dict]) -> str:
        """Format cache key for logging purposes"""
        if isinstance(key, dict):
            if 'symbol' in key:
                return f"symbol:{key['symbol']}"
            return f"dict:{len(key)}keys"
        elif isinstance(key, tuple):
            if len(key) > 0:
                return f"tuple:{key[0]}"
            return "tuple:empty"
        else:
            return str(key)[:50]
    
    def _extract_symbol_from_key(self, key: Union[Tuple, Dict]) -> Optional[str]:
        """Extract symbol from cache key for logging and filtering"""
        if isinstance(key, tuple) and len(key) > 0:
            return key[0]
        elif isinstance(key, dict):
            return key.get('symbol')
        return None
    
    def _update_stats(self, cache_type: CacheType, operation: str, total_time: float, 
                     handler_name: str = None, handler_time: float = None):
        """Update performance statistics"""
        with self._stats_lock:
            stats = self._operation_stats[cache_type.value]
            
            if operation == 'hit':
                stats['hits'] += 1
            elif operation == 'miss':
                stats['misses'] += 1
            elif operation == 'write':
                stats['writes'] += 1
            elif operation == 'error':
                stats['errors'] += 1
            
            stats['total_time_ms'] += total_time
            total_ops = stats['hits'] + stats['misses'] + stats['writes']
            if total_ops > 0:
                stats['avg_time_ms'] = stats['total_time_ms'] / total_ops
            
            if handler_name and handler_time is not None:
                self._update_handler_stats(cache_type, handler_name, operation, handler_time)
    
    def _update_handler_stats(self, cache_type: CacheType, handler_name: str, 
                             operation: str, handler_time: float):
        """Update per-handler performance statistics"""
        # NOTE: No lock needed here - always called from within _update_stats which already holds the lock
        handler_stats = self._operation_stats[cache_type.value]['handler_performance'][handler_name]
        
        if operation == 'hit':
            handler_stats['hits'] += 1
        elif operation == 'miss':
            handler_stats['misses'] += 1
        elif operation == 'write':
            handler_stats['writes'] += 1
        elif operation == 'error':
            handler_stats['errors'] += 1
        
        handler_stats['total_time_ms'] += handler_time
    
    def _log_operation(self, cache_type: CacheType, operation: str, key_str: str, 
                      handler: str, time_ms: float):
        """Log detailed operation information for debugging"""
        operation_info = {
            'timestamp': time.time(),
            'operation': operation,
            'key': key_str,
            'handler': handler,
            'time_ms': time_ms
        }
        
        with self._stats_lock:
            self._recent_operations[cache_type.value].append(operation_info)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self._stats_lock:
            stats = {}
            for cache_type, type_stats in self._operation_stats.items():
                # Calculate hit ratio
                hits = type_stats['hits']
                misses = type_stats['misses']
                total_reads = hits + misses
                hit_ratio = (hits / total_reads * 100) if total_reads > 0 else 0
                
                stats[cache_type] = {
                    'operations': {
                        'hits': hits,
                        'misses': misses,
                        'writes': type_stats['writes'],
                        'errors': type_stats['errors'],
                        'total_reads': total_reads
                    },
                    'performance': {
                        'hit_ratio_pct': round(hit_ratio, 2),
                        'avg_time_ms': round(type_stats['avg_time_ms'], 2),
                        'total_time_ms': round(type_stats['total_time_ms'], 2)
                    },
                    'handlers': {}
                }
                
                # Handler-specific stats
                for handler_name, handler_stats in type_stats['handler_performance'].items():
                    handler_total_reads = handler_stats['hits'] + handler_stats['misses']
                    handler_hit_ratio = (handler_stats['hits'] / handler_total_reads * 100) if handler_total_reads > 0 else 0
                    
                    stats[cache_type]['handlers'][handler_name] = {
                        'hits': handler_stats['hits'],
                        'misses': handler_stats['misses'],
                        'writes': handler_stats['writes'],
                        'errors': handler_stats['errors'],
                        'hit_ratio_pct': round(handler_hit_ratio, 2),
                        'total_time_ms': round(handler_stats['total_time_ms'], 2)
                    }
            
            return stats
    
    def get_recent_operations(self, cache_type: CacheType = None, limit: int = 20) -> Dict[str, Any]:
        """Get recent cache operations for debugging"""
        with self._stats_lock:
            if cache_type:
                operations = list(self._recent_operations[cache_type.value])[-limit:]
                return {cache_type.value: operations}
            else:
                result = {}
                for ct, ops in self._recent_operations.items():
                    result[ct] = list(ops)[-limit:]
                return result
    
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
    
    def _calculate_ttl_remaining(self, cached_data: Any, cache_type: CacheType) -> Optional[str]:
        """
        Calculate TTL remaining for cached data based on cache type and metadata.
        
        Args:
            cached_data: The cached data that may contain metadata
            cache_type: Type of cache for TTL rules
            
        Returns:
            String representation of TTL remaining (e.g., "2h 30m", "expired", "no_ttl")
        """
        try:
            # Default TTL rules by cache type (in hours)
            default_ttls = {
                CacheType.SEC_RESPONSE: 6,          # 6 hours for SEC API responses
                CacheType.COMPANY_FACTS: 90 * 24,   # 90 days for company facts
                CacheType.TECHNICAL_DATA: 7 * 24,   # 7 days for technical data
                CacheType.SUBMISSION_DATA: 90 * 24, # 90 days for submission data
                CacheType.QUARTERLY_METRICS: 24,    # 24 hours for quarterly metrics
            }
            
            # Get TTL for this cache type
            ttl_hours = default_ttls.get(cache_type, 24)  # Default to 24 hours
            
            # Special handling for LLM_RESPONSE - determine type from metadata
            if cache_type == CacheType.LLM_RESPONSE:
                llm_type = self._determine_llm_response_type(cached_data)
                if llm_type == 'fundamental':
                    ttl_hours = 30 * 24  # 30 days for fundamental analysis
                elif llm_type == 'technical':
                    ttl_hours = 7 * 24   # 7 days for technical analysis
                elif llm_type == 'synthesis':
                    ttl_hours = 7 * 24   # 7 days for synthesis
                else:
                    ttl_hours = 24       # Default 24 hours for unknown LLM types
            
            # Extract metadata to find cached_at timestamp
            cached_at_str = None
            timestamp_field_names = ['cached_at', 'fetched_at', 'created_at', 'timestamp']
            
            # Check different metadata structures
            if isinstance(cached_data, dict):
                # Check direct metadata
                if 'metadata' in cached_data:
                    metadata = cached_data['metadata']
                    if isinstance(metadata, dict):
                        for field_name in timestamp_field_names:
                            cached_at_str = metadata.get(field_name)
                            if cached_at_str:
                                break
                
                # Check nested response structure if not found yet
                if not cached_at_str and 'response' in cached_data and isinstance(cached_data['response'], dict):
                    response_metadata = cached_data['response'].get('metadata', {})
                    if isinstance(response_metadata, dict):
                        for field_name in timestamp_field_names:
                            cached_at_str = response_metadata.get(field_name)
                            if cached_at_str:
                                break
                
                # Check if timestamp is directly in the data if not found yet
                if not cached_at_str:
                    for field_name in timestamp_field_names:
                        cached_at_str = cached_data.get(field_name)
                        if cached_at_str:
                            break
            
            if not cached_at_str:
                return "no_ttl"
            
            # Parse the cached_at timestamp
            try:
                # Ensure we have a string
                if not isinstance(cached_at_str, str):
                    logger.debug(f"Timestamp is not a string: {type(cached_at_str)} = {cached_at_str}")
                    return "parse_error"
                
                # Handle empty string
                if not cached_at_str.strip():
                    logger.debug("Timestamp is empty string")
                    return "parse_error"
                
                # Handle both with and without microseconds
                if '.' in cached_at_str:
                    cached_at = datetime.fromisoformat(cached_at_str.replace('Z', '+00:00'))
                else:
                    cached_at = datetime.fromisoformat(cached_at_str.replace('Z', ''))
                    
                # If timezone-naive, assume UTC
                if cached_at.tzinfo is None:
                    from datetime import timezone
                    cached_at = cached_at.replace(tzinfo=timezone.utc)
                
                # Calculate age and remaining TTL
                now = datetime.now(cached_at.tzinfo)
                age = now - cached_at
                ttl_delta = timedelta(hours=ttl_hours)
                remaining = ttl_delta - age
                
                if remaining.total_seconds() <= 0:
                    return "expired"
                
                # Format remaining time nicely
                total_seconds = int(remaining.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                
                if hours > 0:
                    if minutes > 0:
                        return f"{hours}h {minutes}m"
                    else:
                        return f"{hours}h"
                elif minutes > 0:
                    return f"{minutes}m"
                else:
                    return f"{total_seconds}s"
                    
            except (ValueError, TypeError) as e:
                logger.debug(f"Error parsing cached_at timestamp '{cached_at_str}': {e}")
                return "parse_error"
                
        except Exception as e:
            logger.debug(f"Error calculating TTL: {e}")
            return "calc_error"
    
    def _determine_llm_response_type(self, cached_data: Any) -> str:
        """
        Determine the type of LLM response (fundamental, technical, synthesis) from cached data.
        
        Args:
            cached_data: The cached LLM response data
            
        Returns:
            String indicating the LLM response type: 'fundamental', 'technical', 'synthesis', or 'unknown'
        """
        try:
            # Check various locations where the LLM type might be stored
            
            # Method 1: Check cache key metadata
            if isinstance(cached_data, dict):
                # Check metadata.cache_key.llm_type
                metadata = cached_data.get('metadata', {})
                if isinstance(metadata, dict):
                    cache_key = metadata.get('cache_key', {})
                    if isinstance(cache_key, dict):
                        llm_type = cache_key.get('llm_type', '')
                        if llm_type == 'sec':
                            return 'fundamental'
                        elif llm_type == 'ta':
                            return 'technical'
                        elif llm_type == 'full':
                            return 'synthesis'
                        elif llm_type:
                            return llm_type  # Return as-is if it's something else
                
                # Method 2: Check response metadata
                response = cached_data.get('response', {})
                if isinstance(response, dict):
                    response_metadata = response.get('metadata', {})
                    if isinstance(response_metadata, dict):
                        # Check task_type
                        task_type = response_metadata.get('task_type', '')
                        if 'fundamental' in task_type.lower() or 'sec' in task_type.lower():
                            return 'fundamental'
                        elif 'technical' in task_type.lower() or 'ta' in task_type.lower():
                            return 'technical'
                        elif 'synthesis' in task_type.lower() or 'investment' in task_type.lower():
                            return 'synthesis'
                        
                        # Check analysis_type
                        analysis_type = response_metadata.get('analysis_type', '')
                        if 'fundamental' in analysis_type.lower():
                            return 'fundamental'
                        elif 'technical' in analysis_type.lower():
                            return 'technical'
                        elif 'synthesis' in analysis_type.lower():
                            return 'synthesis'
                
                # Method 3: Check direct form_type in cache key
                form_type = cached_data.get('form_type', '')
                if form_type == 'COMPREHENSIVE' or form_type.startswith('10-'):
                    return 'fundamental'
                elif form_type == 'TECHNICAL':
                    return 'technical'
                elif form_type == 'SYNTHESIS':
                    return 'synthesis'
                
                # Method 4: Heuristic based on content structure
                if 'financial_health_score' in str(cached_data) or 'quarterly_summary' in str(cached_data):
                    return 'fundamental'
                elif 'technical_score' in str(cached_data) or 'rsi' in str(cached_data).lower():
                    return 'technical'
                elif 'investment_thesis' in str(cached_data) or 'overall_score' in str(cached_data):
                    return 'synthesis'
            
            return 'unknown'
            
        except Exception as e:
            logger.debug(f"Error determining LLM response type: {e}")
            return 'unknown'


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager