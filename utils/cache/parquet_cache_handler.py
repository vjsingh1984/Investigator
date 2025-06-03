#!/usr/bin/env python3
"""
InvestiGator - Parquet Cache Handler
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Parquet cache handler for efficient storage of tabular data with compression
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import hashlib

from .cache_base import CacheStorageHandler
from .cache_types import CacheType

logger = logging.getLogger(__name__)


class ParquetCacheStorageHandler(CacheStorageHandler):
    """Cache handler for storing data in Parquet format with gzip compression"""
    
    def __init__(self, cache_type: CacheType, base_path: Path, priority: int = 10, config=None):
        """
        Initialize Parquet cache handler
        
        Args:
            cache_type: Type of cache (TECHNICAL_DATA, SUBMISSION_DATA, etc.)
            base_path: Base directory for cache storage
            priority: Handler priority (higher = checked first)
            config: Optional Config object with parquet settings
        """
        super().__init__(cache_type, priority)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Get parquet config from Config object or use defaults
        if config and hasattr(config, 'parquet'):
            self.parquet_config = config.parquet
        else:
            # Default configuration - uniform gzip compression
            from config import ParquetConfig
            self.parquet_config = ParquetConfig()
    
    def _normalize_key(self, key: Union[Tuple, Dict]) -> Dict[str, Any]:
        """Normalize cache key to dictionary format"""
        if isinstance(key, tuple):
            # For technical data: (symbol, data_type, timeframe)
            if len(key) >= 3:
                return {
                    'symbol': key[0],
                    'data_type': key[1],
                    'timeframe': key[2]
                }
            elif len(key) >= 2:
                return {
                    'symbol': key[0],
                    'data_type': key[1],
                    'timeframe': 'default'
                }
            else:
                return {'symbol': key[0], 'data_type': 'default', 'timeframe': 'default'}
        return key
    
    def _get_file_path(self, key_dict: Dict[str, Any]) -> Path:
        """Generate file path based on cache key"""
        symbol = key_dict.get('symbol', 'unknown')
        data_type = key_dict.get('data_type', 'data')
        timeframe = key_dict.get('timeframe', 'default')
        
        # Create subdirectory for symbol
        symbol_dir = self.base_path / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"{data_type}_{timeframe}.parquet.gz"
        return symbol_dir / filename
    
    def _get_metadata_path(self, parquet_path: Path) -> Path:
        """Get metadata file path for a parquet file"""
        return parquet_path.with_suffix('.meta.json')
    
    def get(self, key: Union[Tuple, Dict]) -> Optional[Dict[str, Any]]:
        """Retrieve data from parquet cache"""
        if self.priority < 0:
            return None  # Skip lookup for negative priority
            
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            metadata_path = self._get_metadata_path(file_path)
            
            if file_path.exists() and metadata_path.exists():
                # Read metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Read parquet data
                df = pd.read_parquet(file_path, engine='fastparquet')
                
                # Convert DataFrame to dict for compatibility
                data = {
                    'dataframe': df,
                    'data': df.to_dict('records'),
                    'metadata': metadata,
                    'cache_info': {
                        'cached_at': metadata.get('cached_at'),
                        'cache_type': self.cache_type.value,
                        'compression': self.compression,
                        'records': len(df)
                    }
                }
                
                logger.debug(f"Cache hit (parquet): {file_path}")
                return data
            
            logger.debug(f"Cache miss (parquet): {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error reading from parquet cache: {e}")
            return None
    
    def set(self, key: Union[Tuple, Dict], value: Dict[str, Any]) -> bool:
        """Store data in parquet cache"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            metadata_path = self._get_metadata_path(file_path)
            
            # Extract DataFrame or create from data
            if 'dataframe' in value and isinstance(value['dataframe'], pd.DataFrame):
                df = value['dataframe']
            elif 'data' in value:
                # Convert data to DataFrame
                if isinstance(value['data'], list):
                    df = pd.DataFrame(value['data'])
                elif isinstance(value['data'], dict):
                    df = pd.DataFrame([value['data']])
                else:
                    logger.error(f"Unsupported data format for parquet cache: {type(value['data'])}")
                    return False
            else:
                logger.error("No data or dataframe found in value for parquet cache")
                return False
            
            # Ensure datetime columns are properly formatted
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        # Try to convert to datetime
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass  # Keep as is if conversion fails
            
            # Save DataFrame to parquet using configuration
            write_kwargs = self.parquet_config.get_write_kwargs()
            
            # Check if the engine is available
            try:
                df.to_parquet(file_path, **write_kwargs)
            except ImportError as e:
                # Fallback to the other engine if primary fails
                logger.warning(f"Primary engine {self.parquet_config.engine} not available: {e}")
                if self.parquet_config.engine == "fastparquet":
                    logger.info("Falling back to pyarrow engine")
                    from config import ParquetConfig
                    fallback_config = ParquetConfig(engine="pyarrow")
                else:
                    logger.info("Falling back to fastparquet engine")
                    from config import ParquetConfig
                    fallback_config = ParquetConfig(engine="fastparquet")
                
                fallback_kwargs = fallback_config.get_write_kwargs()
                df.to_parquet(file_path, **fallback_kwargs)
            
            # Save metadata
            metadata = {
                'cached_at': datetime.utcnow().isoformat(),
                'cache_key': key_dict,
                'cache_type': self.cache_type.value,
                'engine': self.parquet_config.engine,
                'compression': self.parquet_config.compression if self.parquet_config.engine == 'fastparquet' else self.parquet_config.pyarrow_compression,
                'records': len(df),
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'file_size_bytes': file_path.stat().st_size,
                'original_metadata': value.get('metadata', {})
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Cached to parquet: {file_path} ({len(df)} records, {file_path.stat().st_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error writing to parquet cache: {e}")
            return False
    
    def exists(self, key: Union[Tuple, Dict]) -> bool:
        """Check if key exists in parquet cache"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            metadata_path = self._get_metadata_path(file_path)
            return file_path.exists() and metadata_path.exists()
        except Exception as e:
            logger.error(f"Error checking parquet cache existence: {e}")
            return False
    
    def delete(self, key: Union[Tuple, Dict]) -> bool:
        """Delete data from parquet cache"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            metadata_path = self._get_metadata_path(file_path)
            
            deleted = False
            if file_path.exists():
                file_path.unlink()
                deleted = True
                
            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True
                
            if deleted:
                logger.debug(f"Deleted from parquet cache: {file_path}")
                
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting from parquet cache: {e}")
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
                        # Delete both parquet file and metadata
                        if file_path.suffix in ['.parquet', '.parquet.gz']:
                            metadata_path = self._get_metadata_path(file_path)
                            if metadata_path.exists():
                                metadata_path.unlink()
                            
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted parquet file matching pattern '{pattern}': {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting parquet file {file_path}: {e}")
            
            logger.info(f"Deleted {deleted_count} parquet files matching pattern '{pattern}'")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting by pattern from parquet cache: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all data from parquet cache"""
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
            
            logger.info(f"Cleared all parquet cache data ({deleted_count} items)")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing parquet cache: {e}")
            return False
    
    def get_cache_info(self, key: Union[Tuple, Dict]) -> Optional[Dict[str, Any]]:
        """Get cache metadata without loading the data"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            metadata_path = self._get_metadata_path(file_path)
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Error reading parquet cache metadata: {e}")
            return None