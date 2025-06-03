#!/usr/bin/env python3
"""
InvestiGator - Memory Cache Storage Handler with TTL Support
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Provides in-memory caching with time-to-live expiration
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from threading import Lock

from .cache_base import CacheStorageHandler
from .cache_types import CacheType


class MemoryCacheStorageHandler(CacheStorageHandler):
    """In-memory cache storage with TTL support"""
    
    # Class-level cache shared across instances
    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_lock = Lock()
    
    def __init__(self, cache_type: CacheType, priority: int = 30, ttl_minutes: int = 15):
        """
        Initialize memory cache handler
        
        Args:
            cache_type: Type of cache data to handle
            priority: Cache priority (higher = checked first)
            ttl_minutes: Time-to-live for cached items in minutes
        """
        super().__init__(cache_type, priority)
        self.ttl = timedelta(minutes=ttl_minutes)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Cleanup expired entries on initialization
        self._cleanup_expired()
    
    def exists(self, key: Any) -> bool:
        """Check if key exists and is not expired"""
        cache_key = self._build_cache_key(key)
        
        with self._cache_lock:
            if cache_key in self._cache:
                item = self._cache[cache_key]
                if self._is_expired(item):
                    del self._cache[cache_key]
                    return False
                return True
            return False
    
    def get(self, key: Any) -> Optional[Dict[str, Any]]:
        """Get value from memory cache"""
        cache_key = self._build_cache_key(key)
        
        with self._cache_lock:
            if cache_key in self._cache:
                item = self._cache[cache_key]
                
                # Check if expired
                if self._is_expired(item):
                    del self._cache[cache_key]
                    self.logger.debug(f"Expired cache item removed: {cache_key}")
                    return None
                
                self.logger.debug(f"Memory cache HIT: {cache_key}")
                return item['data']
            
            self.logger.debug(f"Memory cache MISS: {cache_key}")
            return None
    
    def set(self, key: Any, value: Dict[str, Any]) -> bool:
        """Set value in memory cache"""
        try:
            cache_key = self._build_cache_key(key)
            
            # Ensure value is JSON serializable for consistency
            try:
                json.dumps(value)
            except (TypeError, ValueError) as e:
                self.logger.warning(f"Value not JSON serializable, skipping memory cache: {e}")
                return False
            
            with self._cache_lock:
                self._cache[cache_key] = {
                    'data': value,
                    'timestamp': datetime.now(),
                    'cache_type': self.cache_type.value,
                    'ttl_minutes': self.ttl.total_seconds() / 60
                }
            
            self.logger.debug(f"Memory cache SET: {cache_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set memory cache: {e}")
            return False
    
    def delete(self, key: Any) -> bool:
        """Delete value from memory cache"""
        cache_key = self._build_cache_key(key)
        
        with self._cache_lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self.logger.debug(f"Memory cache DELETE: {cache_key}")
                return True
            return False
    
    def delete_by_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        import re
        
        deleted_count = 0
        pattern_key = self._build_pattern_key(pattern)
        regex_pattern = pattern_key.replace('*', '.*')
        
        with self._cache_lock:
            keys_to_delete = []
            for cache_key in self._cache.keys():
                if re.match(regex_pattern, cache_key):
                    keys_to_delete.append(cache_key)
            
            for cache_key in keys_to_delete:
                del self._cache[cache_key]
                deleted_count += 1
        
        self.logger.debug(f"Memory cache deleted {deleted_count} items matching pattern: {pattern}")
        return deleted_count
    
    def list_keys(self, limit: Optional[int] = None) -> List[str]:
        """List all non-expired keys"""
        keys = []
        current_time = datetime.now()
        
        with self._cache_lock:
            expired_keys = []
            for cache_key, item in self._cache.items():
                if self._is_expired(item, current_time):
                    expired_keys.append(cache_key)
                else:
                    keys.append(cache_key)
            
            # Clean up expired keys
            for cache_key in expired_keys:
                del self._cache[cache_key]
        
        if limit:
            keys = keys[:limit]
        
        return keys
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            total_items = len(self._cache)
            cache_type_items = sum(1 for item in self._cache.values() 
                                 if item.get('cache_type') == self.cache_type.value)
            
            # Calculate memory usage (rough estimate)
            try:
                cache_size = len(json.dumps(self._cache).encode('utf-8'))
            except:
                cache_size = 0
        
        return {
            'handler_type': 'memory',
            'cache_type': self.cache_type.value,
            'total_items': total_items,
            'cache_type_items': cache_type_items,
            'estimated_size_bytes': cache_size,
            'ttl_minutes': self.ttl.total_seconds() / 60,
            'priority': self.priority
        }
    
    def clear_all(self) -> bool:
        """Clear all cache entries for this cache type"""
        deleted_count = 0
        
        with self._cache_lock:
            keys_to_delete = []
            for cache_key, item in self._cache.items():
                if item.get('cache_type') == self.cache_type.value:
                    keys_to_delete.append(cache_key)
            
            for cache_key in keys_to_delete:
                del self._cache[cache_key]
                deleted_count += 1
        
        self.logger.info(f"Cleared {deleted_count} memory cache entries for {self.cache_type.value}")
        return deleted_count > 0
    
    def clear_cache(self) -> int:
        """Clear all cache entries for this cache type (helper method)"""
        deleted_count = 0
        
        with self._cache_lock:
            keys_to_delete = []
            for cache_key, item in self._cache.items():
                if item.get('cache_type') == self.cache_type.value:
                    keys_to_delete.append(cache_key)
            
            for cache_key in keys_to_delete:
                del self._cache[cache_key]
                deleted_count += 1
        
        return deleted_count
    
    def _build_cache_key(self, key: Any) -> str:
        """Build cache key string"""
        if isinstance(key, dict):
            # Sort keys for consistent hashing
            sorted_items = sorted(key.items())
            key_str = json.dumps(sorted_items, sort_keys=True)
        elif isinstance(key, (list, tuple)):
            key_str = json.dumps(sorted(key) if isinstance(key, list) else key)
        else:
            key_str = str(key)
        
        return f"{self.cache_type.value}:{key_str}"
    
    def _build_pattern_key(self, pattern: str) -> str:
        """Build pattern key for matching"""
        return f"{self.cache_type.value}:{pattern}"
    
    def _is_expired(self, item: Dict[str, Any], current_time: Optional[datetime] = None) -> bool:
        """Check if cache item is expired"""
        if current_time is None:
            current_time = datetime.now()
        
        timestamp = item.get('timestamp', datetime.min)
        return current_time - timestamp > self.ttl
    
    def _cleanup_expired(self) -> int:
        """Clean up expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        with self._cache_lock:
            for cache_key, item in self._cache.items():
                if self._is_expired(item, current_time):
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                del self._cache[cache_key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired memory cache entries")
        
        return len(expired_keys)
    
    @classmethod
    def get_global_stats(cls) -> Dict[str, Any]:
        """Get global memory cache statistics"""
        with cls._cache_lock:
            total_items = len(cls._cache)
            
            try:
                total_size = len(json.dumps(cls._cache).encode('utf-8'))
            except:
                total_size = 0
            
            # Count by cache type
            cache_type_counts = {}
            for item in cls._cache.values():
                cache_type = item.get('cache_type', 'unknown')
                cache_type_counts[cache_type] = cache_type_counts.get(cache_type, 0) + 1
        
        return {
            'total_memory_cache_items': total_items,
            'total_estimated_size_bytes': total_size,
            'cache_type_distribution': cache_type_counts,
            'handler_class': cls.__name__
        }
    
    @classmethod
    def clear_all_memory_cache(cls) -> int:
        """Clear all memory cache entries across all types"""
        with cls._cache_lock:
            item_count = len(cls._cache)
            cls._cache.clear()
        
        logging.getLogger(cls.__name__).info(f"Cleared all {item_count} memory cache entries")
        return item_count