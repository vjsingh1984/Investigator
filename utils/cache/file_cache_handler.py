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
        Generate file path using symbol-specific directory structure
        
        New structure:
        - SEC Cache: data/sec_cache/{cache_type}/{symbol}/filename
        - LLM Cache: data/llm_cache/{symbol}/filename  
        - Technical Cache: data/technical_cache/{symbol}/filename
        
        Args:
            key_dict: Dictionary containing keys
            
        Returns:
            Path object for the cache file
        """
        symbol = key_dict.get('symbol', 'UNKNOWN').upper()
        
        # Check if new symbol directory structure is enabled
        use_symbol_dirs = True
        if (self.config and hasattr(self.config, 'cache_control') and 
            hasattr(self.config.cache_control, 'disk_structure')):
            use_symbol_dirs = self.config.cache_control.disk_structure.get('use_symbol_directories', True)
        
        if use_symbol_dirs:
            # New symbol-directory structure
            if self.cache_type in [CacheType.SUBMISSION_DATA, CacheType.COMPANY_FACTS, CacheType.QUARTERLY_METRICS]:
                # SEC cache with cache_type subdirectory: data/sec_cache/{cache_type}/{symbol}/
                cache_type_name = self._get_cache_type_name()
                symbol_dir = self.base_path / cache_type_name / symbol
            else:
                # Direct symbol directory: data/llm_cache/{symbol}/ or data/technical_cache/{symbol}/
                symbol_dir = self.base_path / symbol
            
            # Generate filename without symbol prefix (since it's in the directory structure)
            filename = self._generate_filename(key_dict, include_symbol=False)
        else:
            # Legacy flat structure
            symbol_dir = self.base_path
            filename = self._generate_filename(key_dict, include_symbol=True)
        
        return symbol_dir / filename
    
    def _get_cache_type_name(self) -> str:
        """Get cache type name for directory structure"""
        cache_type_names = {
            CacheType.SUBMISSION_DATA: 'submissions',
            CacheType.COMPANY_FACTS: 'facts', 
            CacheType.QUARTERLY_METRICS: 'quarterlymetrics',
            CacheType.SEC_RESPONSE: 'responses'
        }
        return cache_type_names.get(self.cache_type, self.cache_type.value)
    
    def _generate_filename(self, key_dict: Dict[str, str], include_symbol: bool = True) -> str:
        """Generate filename based on cache type and key data with universal gzip compression"""
        symbol = key_dict.get('symbol', 'UNKNOWN').upper()
        
        if self.cache_type == CacheType.SEC_RESPONSE:
            fiscal_year = key_dict.get('fiscal_year', '2024')
            fiscal_period = key_dict.get('fiscal_period', 'Q1')
            base_name = f"quarterly_summary_{fiscal_year}-{fiscal_period}"
            
        elif self.cache_type == CacheType.LLM_RESPONSE:
            llm_type = key_dict.get('llm_type', 'unknown')
            form_type = key_dict.get('form_type', '')
            
            # Handle different analysis types with clear, descriptive names
            if form_type == 'COMPREHENSIVE':
                # Comprehensive analysis combining all quarters
                base_name = f"{llm_type}_comprehensive"
            elif llm_type == 'ta' or 'technical' in llm_type.lower():
                # Technical analysis - not tied to specific quarter
                base_name = "technical_analysis"
            elif llm_type == 'synthesis_comprehensive':
                # Comprehensive synthesis mode
                base_name = "investment_synthesis_comprehensive"
            elif llm_type == 'synthesis_quarterly':
                # Quarterly synthesis mode  
                base_name = "investment_synthesis_quarterly"
            elif llm_type == 'full' or 'synthesis' in llm_type.lower():
                # Legacy synthesis - not tied to specific quarter
                base_name = "investment_synthesis"
            elif 'period' in key_dict and key_dict['period'] not in ['UNKNOWN', 'unknown', '']:
                # Individual quarterly analysis with period
                period = key_dict['period']
                # For actual SEC filings, include form_type if available to distinguish 10-Q vs 10-K
                if form_type and form_type != 'COMPREHENSIVE':
                    base_name = f"{llm_type}_{form_type}_{period}"
                else:
                    base_name = f"{llm_type}_{period}"
            elif 'fiscal_year' in key_dict and 'fiscal_period' in key_dict:
                # Fallback to separate fiscal_year and fiscal_period
                fiscal_year = key_dict.get('fiscal_year')
                fiscal_period = key_dict.get('fiscal_period')
                if form_type and form_type != 'COMPREHENSIVE':
                    base_name = f"{llm_type}_{form_type}_{fiscal_year}-{fiscal_period}"
                else:
                    base_name = f"{llm_type}_{fiscal_year}-{fiscal_period}"
            else:
                # Generic fallback using just the llm_type
                base_name = llm_type
                
        elif self.cache_type == CacheType.SUBMISSION_DATA:
            base_name = "submissions"
            
        elif self.cache_type == CacheType.COMPANY_FACTS:
            cik = key_dict.get('cik', 'unknown')
            base_name = f"companyfacts_{cik}" if cik != 'unknown' else "companyfacts"
            
        elif self.cache_type == CacheType.QUARTERLY_METRICS:
            fiscal_year = key_dict.get('fiscal_year', '2024')
            fiscal_period = key_dict.get('fiscal_period', 'Q1')
            base_name = f"metrics_{fiscal_year}-{fiscal_period}"
            
        else:
            base_name = "data"
        
        # Universal gzip compression for all file types
        if include_symbol:
            filename = f"{symbol}_{base_name}.json.gz"
        else:
            filename = f"{base_name}.json.gz"
            
        return filename
    
    def _should_compress_file(self, filename: str) -> bool:
        """Check if file should be compressed based on configuration"""
        if (self.config and hasattr(self.config, 'cache_control') and 
            hasattr(self.config.cache_control, 'disk_structure')):
            compression_config = self.config.cache_control.disk_structure.get('compression', {})
            if compression_config.get('apply_to_all', True):
                return True
            
            # Check specific file extensions
            file_extensions = compression_config.get('file_extensions', [])
            file_ext = Path(filename).suffix
            return file_ext in file_extensions
        
        return True  # Default to compression enabled
    
    def get(self, key: Union[Tuple, Dict]) -> Optional[Dict[str, Any]]:
        """Retrieve data from file cache"""
        if self.priority < 0:
            return None  # Skip lookup for negative priority
            
        try:
            logger.info(f"🔍 FILE CACHE GET: Original key: {key}, type: {type(key)}, cache_type: {self.cache_type}")
            key_dict = self._normalize_key(key)
            logger.info(f"🔍 FILE CACHE GET: Normalized key_dict: {key_dict}")
            file_path = self._get_file_path(key_dict)
            logger.info(f"🔍 FILE CACHE GET: Generated file_path: {file_path}")
            
            # Handle LLM responses with separate prompt and response files
            if self.cache_type == CacheType.LLM_RESPONSE:
                # Get directory and base filename
                file_dir = file_path.parent
                base_name = file_path.stem.replace('.json', '')  # Remove .json from stem if present
                
                # Read prompt file
                prompt_path = file_dir / f"prompt_{base_name}.txt.gz"
                response_path = file_dir / f"llmresponse_{base_name}.json.gz"
                
                if not (prompt_path.exists() and response_path.exists()):
                    logger.info(f"🔍 Cache MISS (LLM): prompt_path={prompt_path} exists={prompt_path.exists()}, response_path={response_path} exists={response_path.exists()}")
                    return None
                
                # Read prompt
                with gzip.open(prompt_path, 'rt', encoding='utf-8') as f:
                    prompt_text = f.read()
                
                # Read response
                with gzip.open(response_path, 'rt', encoding='utf-8') as f:
                    response_data = json.load(f)
                
                # Combine into single structure
                combined_data = {
                    'prompt': prompt_text,
                    'response': response_data.get('response', {}),
                    'model_info': response_data.get('model_info', {}),
                    'metadata': response_data.get('metadata', {})
                }
                
                logger.info(f"🔍 Cache HIT (LLM): prompt_path={prompt_path}, response_path={response_path}")
                return combined_data
            else:
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
                        logger.debug(f"Cache MISS (file): {file_path}")
                        return None
                
                logger.debug(f"Cache HIT (file): {file_path}")
                
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
            
            # Create directory structure (including symbol directories)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle LLM responses with separate prompt and response files
            if self.cache_type == CacheType.LLM_RESPONSE:
                # Get directory and base filename
                file_dir = file_path.parent
                base_name = file_path.stem.replace('.json', '')  # Remove .json from stem if present
                
                # Store prompt as text file
                prompt_path = file_dir / f"prompt_{base_name}.txt.gz"
                prompt_text = value.get('prompt', '')
                with gzip.open(prompt_path, 'wt', encoding='utf-8', compresslevel=9) as f:
                    f.write(prompt_text)
                
                # Store response as JSON file
                response_path = file_dir / f"llmresponse_{base_name}.json.gz"
                response_data = {
                    'response': value.get('response', {}),
                    'model_info': value.get('model_info', {}),
                    'metadata': {
                        'cached_at': datetime.utcnow().isoformat(),
                        'cache_key': key_dict,
                        'cache_type': self.cache_type.value
                    }
                }
                with gzip.open(response_path, 'wt', encoding='utf-8', compresslevel=9) as f:
                    json.dump(response_data, f, separators=(',', ':'), default=str)
                
                logger.debug(f"Cache WRITE (LLM): prompt={prompt_path}, response={response_path}")
                return True
            else:
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
                
                logger.debug(f"Cache WRITE (JSON): {file_path}")
                return True
            
        except Exception as e:
            logger.error(f"Error writing to file cache: {e}")
            return False
    
    def exists(self, key: Union[Tuple, Dict]) -> bool:
        """Check if key exists in file cache"""
        try:
            logger.info(f"🔍 FILE CACHE EXISTS: Starting exists check for key: {key}")
            key_dict = self._normalize_key(key)
            logger.info(f"🔍 FILE CACHE EXISTS: Normalized to: {key_dict}")
            file_path = self._get_file_path(key_dict)
            logger.info(f"🔍 FILE CACHE EXISTS: Got file_path: {file_path}")
            
            # Handle LLM responses with separate files
            if self.cache_type == CacheType.LLM_RESPONSE:
                file_dir = file_path.parent
                base_name = file_path.stem.replace('.json', '')
                prompt_path = file_dir / f"prompt_{base_name}.txt.gz"
                response_path = file_dir / f"llmresponse_{base_name}.json.gz"
                
                exists = prompt_path.exists() and response_path.exists()
                logger.debug(f"Cache EXISTS (LLM): {exists} - prompt={prompt_path.exists()}, response={response_path.exists()}")
                return exists
            else:
                # Check if file exists (try both original and compressed versions)
                if file_path.exists():
                    logger.debug(f"Cache EXISTS: {file_path}")
                    return True
                
                # Try with .gz extension if original doesn't exist
                if not file_path.suffix == '.gz':
                    gz_path = file_path.with_suffix(file_path.suffix + '.gz')
                    if gz_path.exists():
                        logger.debug(f"Cache EXISTS (compressed): {gz_path}")
                        return True
                
                logger.debug(f"Cache NOT EXISTS: {file_path}")
                return False
            
        except Exception as e:
            logger.error(f"Error checking file cache existence: {e}")
            return False
    
    def delete(self, key: Union[Tuple, Dict]) -> bool:
        """Delete data from file cache"""
        try:
            key_dict = self._normalize_key(key)
            file_path = self._get_file_path(key_dict)
            
            deleted = False
            
            # Handle LLM responses with separate files
            if self.cache_type == CacheType.LLM_RESPONSE:
                file_dir = file_path.parent
                base_name = file_path.stem.replace('.json', '')
                prompt_path = file_dir / f"prompt_{base_name}.txt.gz"
                response_path = file_dir / f"llmresponse_{base_name}.json.gz"
                
                # Delete both files
                if prompt_path.exists():
                    prompt_path.unlink()
                    logger.debug(f"Cache DELETE (LLM prompt): {prompt_path}")
                    deleted = True
                    
                if response_path.exists():
                    response_path.unlink()
                    logger.debug(f"Cache DELETE (LLM response): {response_path}")
                    deleted = True
            else:
                # Delete original file if exists
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Cache DELETE: {file_path}")
                    deleted = True
                
                # Also try to delete compressed version
                if not file_path.suffix == '.gz':
                    gz_path = file_path.with_suffix(file_path.suffix + '.gz')
                    if gz_path.exists():
                        gz_path.unlink()
                        logger.debug(f"Cache DELETE (compressed): {gz_path}")
                        deleted = True
            
            if not deleted:
                logger.debug(f"Cache DELETE (not found): {file_path}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting from file cache: {e}")
            return False
    
    def delete_by_symbol(self, symbol: str) -> int:
        """
        Optimized symbol-based deletion using new symbol directory structure.
        Deletes entire symbol directories for clean isolation.
        
        New structure:
        - SEC Cache: data/sec_cache/{cache_type}/{symbol}/
        - LLM Cache: data/llm_cache/{symbol}/
        - Technical Cache: data/technical_cache/{symbol}/
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            symbol = symbol.upper()  # Normalize to uppercase
            
            # Check if new symbol directory structure is enabled
            use_symbol_dirs = True
            if (self.config and hasattr(self.config, 'cache_control') and 
                hasattr(self.config.cache_control, 'disk_structure')):
                use_symbol_dirs = self.config.cache_control.disk_structure.get('use_symbol_directories', True)
            
            if use_symbol_dirs:
                # New symbol-directory structure - delete specific symbol directories
                if self.cache_type in [CacheType.SUBMISSION_DATA, CacheType.COMPANY_FACTS, CacheType.QUARTERLY_METRICS]:
                    # SEC cache with cache_type subdirectory: data/sec_cache/{cache_type}/{symbol}/
                    cache_type_name = self._get_cache_type_name()
                    symbol_dir = self.base_path / cache_type_name / symbol
                else:
                    # Direct symbol directory: data/llm_cache/{symbol}/ or data/technical_cache/{symbol}/
                    symbol_dir = self.base_path / symbol
                
                if symbol_dir.exists() and symbol_dir.is_dir():
                    import shutil
                    try:
                        # Count files before deletion for accurate reporting
                        file_count = sum(1 for _ in symbol_dir.rglob("*") if _.is_file())
                        shutil.rmtree(symbol_dir)
                        deleted_count = file_count
                        logger.info(f"Deleted symbol directory: {symbol_dir}")
                    except Exception as e:
                        logger.error(f"Error deleting symbol directory {symbol_dir}: {e}")
                else:
                    logger.debug(f"Symbol directory does not exist: {symbol_dir}")
            else:
                # Legacy flat structure - search and delete files with symbol in name
                for file_path in self.base_path.rglob("*"):
                    if file_path.is_file():
                        filename = file_path.name.upper()
                        if symbol in filename:
                            try:
                                file_path.unlink()
                                deleted_count += 1
                                logger.debug(f"Deleted symbol file: {file_path}")
                            except Exception as e:
                                logger.error(f"Error deleting file {file_path}: {e}")
            
            logger.info(f"Symbol cleanup [{self.__class__.__name__}]: Deleted {deleted_count} files for symbol {symbol}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting by symbol '{symbol}': {e}")
            return 0
    
    def delete_by_pattern(self, pattern: str) -> int:
        """Delete all cache entries matching a pattern (legacy method)"""
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