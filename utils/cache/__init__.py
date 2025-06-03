#!/usr/bin/env python3
"""
InvestiGator - Cache Module Initialization
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Cache module for InvestiGator
Provides modular caching infrastructure with memory, disk and RDBMS backends
"""

from .cache_types import CacheType
from .cache_base import CacheStorageHandler
from .file_cache_handler import FileCacheStorageHandler
from .rdbms_cache_handler import RdbmsCacheStorageHandler
from .memory_cache_handler import MemoryCacheStorageHandler
from .cache_manager import CacheManager, get_cache_manager
from .cache_facade import CacheFacade, get_cache_facade

__all__ = [
    'CacheType',
    'CacheStorageHandler',
    'FileCacheStorageHandler',
    'RdbmsCacheStorageHandler',
    'MemoryCacheStorageHandler',
    'CacheManager',
    'get_cache_manager',
    'CacheFacade',
    'get_cache_facade'
]