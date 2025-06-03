#!/usr/bin/env python3
"""
InvestiGator - File Cache Handler
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

File/Directory based cache storage handler
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import json
import gzip
import logging
from datetime import datetime

from .cache_base import CacheStorageHandler
from .cache_types import CacheType

logger = logging.getLogger(__name__)


class FileCacheStorageHandler(CacheStorageHandler):
    """File/Directory based cache storage handler integrated with existing disk methods"""
    
    def __init__(self, cache_type: CacheType, base_path: Path, priority: int = 0, config=None):
        """
        Initialize file cache handler
        
        Args:
            cache_type: Type of cache
            base_path: Base directory path for cache storage
            priority: Priority for lookup
            config: Configuration object for symbol-specific paths
        """
        super().__init__(cache_type, priority)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.config = config
    
    def _get_file_path(self, key_dict: Dict[str, str]) -> Path:
        """
        Generate file path from key dictionary
        
        Args:
            key_dict: Dictionary containing keys
            
        Returns:
            Path object for the cache file
        """
        symbol = key_dict.get('symbol', 'unknown')
        
        if self.cache_type == CacheType.SEC_RESPONSE:
            # data/sec_cache/{symbol}/{category}_{period}_{form_type}.json
            subdir = self.base_path / symbol
            filename_parts = []
            for k in ['category', 'period', 'form_type']:
                if k in key_dict and key_dict[k] != 'N/A':
                    filename_parts.append(key_dict[k])
            filename = '_'.join(filename_parts) + '.json'
            
        elif self.cache_type == CacheType.LLM_RESPONSE:
            # data/llm_cache/{symbol}/{llm_type}_{period}_{form_type}.json
            subdir = self.base_path / symbol
            filename_parts = []
            for k in ['llm_type', 'period', 'form_type']:
                if k in key_dict and key_dict[k] != 'N/A':
                    filename_parts.append(key_dict[k])
            filename = '_'.join(filename_parts) + '.json'
            
        elif self.cache_type == CacheType.TECHNICAL_DATA:
            # data/technical_cache/{symbol}/{data_type}.json
            subdir = self.base_path / symbol
            data_type = key_dict.get('data_type', 'data')
            filename = f"{data_type}.json"
            
        elif self.cache_type == CacheType.COMPANY_FACTS:
            # data/sec_cache/company_facts/{symbol}.json.gz (compressed)
            subdir = self.base_path / 'company_facts'
            symbol = key_dict.get('symbol', 'unknown')
            filename = f"{symbol}.json.gz"
            
        elif self.cache_type == CacheType.QUARTERLY_METRICS:
            # data/sec_cache/{symbol}/quarterly_metrics_{period}.json
            subdir = self.base_path / symbol
            period = key_dict.get('period', 'unknown')
            filename = f"quarterly_metrics_{period}.json"
            
        else:
            # Default: flat structure
            subdir = self.base_path
            filename = '_'.join(f"{k}_{v}" for k, v in sorted(key_dict.items())) + '.json'
        
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / filename
    
    def get(self, key: Union[Tuple, Dict]) -> Optional[Dict[str, Any]]:
        """Retrieve data from file cache"""
        if self.priority < 0:
            return None  # Skip lookup for negative priority
            
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            
            # Try both compressed and uncompressed versions
            if file_path.exists():
                # Check if it's a gzipped file
                if file_path.suffix == '.gz':
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
            else:
                # Try with .gz extension if original doesn't exist
                gz_path = file_path.with_suffix(file_path.suffix + '.gz')
                if gz_path.exists():
                    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                    file_path = gz_path  # Update for logging
                else:
                    logger.debug(f"Cache miss (file): {file_path}")
                    return None
            
            logger.debug(f"Cache hit (file): {file_path}")
            
            # For backward compatibility, extract data from wrapped format
            if 'data' in data and 'metadata' in data:
                return data['data']
            return data
            
        except Exception as e:
            logger.error(f"Error reading from file cache: {e}")
            return None
    
    def set(self, key: Union[Tuple, Dict], value: Dict[str, Any]) -> bool:
        """Store data in file cache"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            
            # Add metadata for audit
            cache_data = {
                'data': value,
                'metadata': {
                    'cached_at': datetime.utcnow().isoformat(),
                    'cache_key': key_dict,
                    'cache_type': self.cache_type.value
                }
            }
            
            # Use gzip compression for all cache types (uniform compression)
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'wt', encoding='utf-8', compresslevel=9) as f:
                    json.dump(cache_data, f, separators=(',', ':'), default=str)
            else:
                # For backward compatibility, if file doesn't have .gz extension, add it
                gz_path = file_path.with_suffix(file_path.suffix + '.gz')
                with gzip.open(gz_path, 'wt', encoding='utf-8', compresslevel=9) as f:
                    json.dump(cache_data, f, separators=(',', ':'), default=str)
                file_path = gz_path
            
            logger.debug(f"Cached to file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing to file cache: {e}")
            return False
    
    def exists(self, key: Union[Tuple, Dict]) -> bool:
        """Check if key exists in file cache"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            return file_path.exists()
        except Exception as e:
            logger.error(f"Error checking file cache existence: {e}")
            return False
    
    def delete(self, key: Union[Tuple, Dict]) -> bool:
        """Delete data from file cache"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted from file cache: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting from file cache: {e}")
            return False
    
    def delete_by_pattern(self, pattern: str) -> int:
        """Delete all cache entries matching a pattern"""
        try:
            import fnmatch
            deleted_count = 0
            
            # Walk through cache directory and find matching files
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file() and fnmatch.fnmatch(file_path.name, pattern):
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted file matching pattern '{pattern}': {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {e}")
            
            logger.info(f"Deleted {deleted_count} files matching pattern '{pattern}'")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting by pattern '{pattern}': {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all data from file cache"""
        try:
            import shutil
            deleted_count = 0
            
            # Remove all files in cache directory but keep the directory structure
            for item in self.base_path.iterdir():
                if item.is_file():
                    item.unlink()
                    deleted_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_count += 1
            
            logger.info(f"Cleared all file cache data ({deleted_count} items)")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing file cache: {e}")
            return False