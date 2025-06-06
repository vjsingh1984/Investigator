#!/usr/bin/env python3
"""
InvestiGator - Cache Cleanup Utilities
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Cache cleanup utility for InvestiGator
Provides functions to clean disk cache directories and truncate database tables
Enhanced with cache manager integration for better control
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import List, Optional
import psycopg2
from psycopg2 import sql

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from utils.cache.cache_manager import get_cache_manager
from utils.cache.cache_types import CacheType

logger = logging.getLogger(__name__)


class CacheCleanup:
    """Handles cache cleanup operations"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.cache_manager = get_cache_manager()
        
    def clean_disk_cache(self, cache_dirs: Optional[List[str]] = None, symbol: Optional[str] = None):
        """
        Clean disk cache directories
        
        Args:
            cache_dirs: List of cache directories to clean. If None, cleans all
            symbol: If provided, only clean cache for this symbol
        """
        if cache_dirs is None:
            cache_dirs = [
                "data/sec_cache",
                "data/llm_cache", 
                "data/technical_cache",
                "data/price_cache"
            ]
        
        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if not cache_path.exists():
                continue
                
            if symbol:
                # Clean only symbol-specific files/directories
                # Look for symbol subdirectories
                symbol_path = cache_path / symbol
                if symbol_path.exists() and symbol_path.is_dir():
                    logger.info(f"Removing symbol directory: {symbol_path}")
                    shutil.rmtree(symbol_path)
                
                # Look for symbol-prefixed files
                for file_path in cache_path.glob(f"{symbol}*"):
                    if file_path.is_file():
                        logger.info(f"Removing file: {file_path}")
                        file_path.unlink()
                    elif file_path.is_dir():
                        logger.info(f"Removing directory: {file_path}")
                        shutil.rmtree(file_path)
            else:
                # Clean entire cache directory
                logger.info(f"Cleaning cache directory: {cache_path}")
                for item in cache_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                        
    def truncate_cache_tables(self, tables: Optional[List[str]] = None, symbol: Optional[str] = None):
        """
        Truncate cache tables in database
        
        Args:
            tables: List of tables to truncate. If None, truncates all cache tables
            symbol: If provided, only delete records for this symbol
        """
        if tables is None:
            # Only include tables that are actually used by the application
            tables = [
                "sec_responses",             # Used by SEC cache handlers (renamed)
                "llm_responses",             # Used by LLM cache handlers and synthesizer (renamed)
                "sec_submissions",           # Used by SEC quarterly processor and cache (renamed, consolidated)
                "sec_companyfacts",          # Used by SEC quarterly processor and cache (renamed)
                "quarterly_metrics"          # Used by SEC strategies and cache
            ]
            # Note: technical_indicators, stock_analysis, sec_filings removed - using parquet-only storage
        
        try:
            # Connect to database
            conn = psycopg2.connect(self.config.database.url)
            cur = conn.cursor()
            
            for table in tables:
                try:
                    if symbol:
                        # Delete only records for specific symbol
                        # Check which column to use (symbol, ticker, or cik)
                        cur.execute(f"""
                            SELECT column_name 
                            FROM information_schema.columns 
                            WHERE table_name = %s 
                            AND column_name IN ('symbol', 'ticker', 'cik')
                            LIMIT 1
                        """, (table,))
                        
                        result = cur.fetchone()
                        if result:
                            column_name = result[0]
                            query = sql.SQL("DELETE FROM {} WHERE {} = %s").format(
                                sql.Identifier(table),
                                sql.Identifier(column_name)
                            )
                            cur.execute(query, (symbol,))
                            deleted = cur.rowcount
                            logger.info(f"Deleted {deleted} records for {symbol} from {table}")
                    else:
                        # Truncate entire table
                        cur.execute(sql.SQL("TRUNCATE TABLE {} CASCADE").format(
                            sql.Identifier(table)
                        ))
                        logger.info(f"Truncated table: {table}")
                        
                except psycopg2.Error as e:
                    logger.warning(f"Error processing table {table}: {e}")
                    conn.rollback()
                    continue
                    
            conn.commit()
            logger.info("Database cache cleanup completed")
            
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()
                
    def clean_symbol_cache_managed(self, symbol: str) -> bool:
        """Clean all cache data for a specific symbol using optimized cache manager"""
        try:
            logger.info(f"🧹 Starting optimized cache cleanup for symbol: {symbol}")
            
            # Use the new optimized delete_by_symbol method
            deletion_results = self.cache_manager.delete_by_symbol(symbol)
            
            # Calculate total deletions
            total_deleted = sum(deletion_results.values())
            
            # Log results
            if total_deleted > 0:
                cache_summary = " | ".join([f"{ct}:{count}" for ct, count in deletion_results.items() if count > 0])
                logger.info(f"✅ Optimized cleanup SUCCESS for {symbol}: {total_deleted} total deletions | {cache_summary}")
            else:
                logger.info(f"🔍 Optimized cleanup COMPLETE for {symbol}: No cache entries found")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in optimized symbol cache cleanup for {symbol}: {e}")
            return False
    
    def clean_symbols_cache_managed(self, symbols: List[str]) -> bool:
        """Clean cache for multiple symbols using cache manager with parallel processing"""
        try:
            import concurrent.futures
            
            logger.info(f"Cleaning cache for {len(symbols)} symbols in parallel: {symbols}")
            all_success = True
            
            # Use parallel processing for multiple symbols
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(symbols), 4)) as executor:
                # Submit all symbol cleaning tasks
                future_to_symbol = {
                    executor.submit(self.clean_symbol_cache_managed, symbol): symbol 
                    for symbol in symbols
                }
                
                # Wait for all tasks to complete and collect results
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        success = future.result()
                        if not success:
                            all_success = False
                            logger.error(f"Failed to clean cache for symbol: {symbol}")
                        else:
                            logger.info(f"Successfully cleaned cache for symbol: {symbol}")
                    except Exception as e:
                        logger.error(f"Error cleaning cache for symbol {symbol}: {e}")
                        all_success = False
            
            logger.info(f"Parallel symbol cache cleanup completed. Success: {all_success}")
            return all_success
            
        except Exception as e:
            logger.error(f"Error in parallel symbols cache cleanup: {e}")
            return False
    
    def clean_cache_type_managed(self, cache_type: str) -> bool:
        """Clean specific cache type using cache manager"""
        try:
            type_map = {
                'sec': CacheType.SEC_RESPONSE,
                'llm': CacheType.LLM_RESPONSE, 
                'technical': CacheType.TECHNICAL_DATA,
                'submission': CacheType.SUBMISSION_DATA,
                'company_facts': CacheType.COMPANY_FACTS,
                'quarterly_metrics': CacheType.QUARTERLY_METRICS
            }
            
            if cache_type not in type_map:
                logger.error(f"Unknown cache type: {cache_type}")
                return False
            
            return self.cache_manager.clear_cache_type(type_map[cache_type])
            
        except Exception as e:
            logger.error(f"Error cleaning cache type {cache_type}: {e}")
            return False
    
    def clean_all_caches(self, symbol: Optional[str] = None):
        """Clean both disk and database caches"""
        logger.info(f"Starting complete cache cleanup{' for ' + symbol if symbol else ''}")
        if symbol:
            # Use cache manager for symbol-specific cleanup
            self.clean_symbol_cache_managed(symbol)
        else:
            self.clean_disk_cache(symbol=symbol)
            self.truncate_cache_tables(symbol=symbol)
        logger.info("Cache cleanup completed")


def main():
    """Command line interface for cache cleanup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="InvestiGator Cache Cleanup Utility")
    parser.add_argument("--disk", action="store_true", help="Clean disk cache only")
    parser.add_argument("--db", action="store_true", help="Clean database cache only")
    parser.add_argument("--symbol", help="Clean cache for specific symbol only")
    parser.add_argument("--symbols", nargs="+", help="Clean cache for multiple symbols")
    parser.add_argument("--cache-type", choices=['sec', 'llm', 'technical', 'submission', 'company_facts', 'quarterly_metrics'], 
                       help="Clean specific cache type")
    parser.add_argument("--all", action="store_true", help="Clean all caches (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    cleanup = CacheCleanup()
    
    # Handle symbols (single or multiple)
    symbols = []
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = args.symbols
    
    if args.cache_type:
        success = cleanup.clean_cache_type_managed(args.cache_type)
        print(f"Cache type '{args.cache_type}' cleanup: {'SUCCESS' if success else 'FAILED'}")
    elif symbols:
        if len(symbols) == 1:
            success = cleanup.clean_symbol_cache_managed(symbols[0])
            print(f"Symbol {symbols[0]} cache cleanup: {'SUCCESS' if success else 'FAILED'}")
        else:
            success = cleanup.clean_symbols_cache_managed(symbols)
            print(f"Symbols {symbols} cache cleanup: {'SUCCESS' if success else 'FAILED'}")
    elif args.disk:
        cleanup.clean_disk_cache()
        print("Disk cache cleanup: SUCCESS")
    elif args.db:
        cleanup.truncate_cache_tables()
        print("Database cache cleanup: SUCCESS")
    else:
        cleanup.clean_all_caches()
        print("Complete cache cleanup: SUCCESS")


if __name__ == "__main__":
    main()
