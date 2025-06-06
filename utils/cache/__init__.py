#!/usr/bin/env python3
"""
InvestiGator - Cache Module Initialization
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Cache module for InvestiGator
Provides modular caching infrastructure with disk and RDBMS backends
"""

from .cache_types import CacheType
from .cache_base import CacheStorageHandler
from .file_cache_handler import FileCacheStorageHandler
from .rdbms_cache_handler import RdbmsCacheStorageHandler
from .cache_manager import CacheManager, get_cache_manager
# Removed cache facade - obsolete wrapper around cache manager

__all__ = [
    'CacheType',
    'CacheStorageHandler',
    'FileCacheStorageHandler',
    'RdbmsCacheStorageHandler',
    'CacheManager',
    'get_cache_manager'
]