#!/usr/bin/env python3
"""
Ticker to CIK (Central Index Key) Mapper

This module provides functionality to map stock tickers to SEC CIK numbers
using the SEC's official ticker mapping file.

Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0
"""

import os
import logging
import requests
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from pathlib import Path
import json
try:
    from config import get_config
except ImportError:
    get_config = None

logger = logging.getLogger(__name__)


class TickerCIKMapper:
    """Maps stock tickers to SEC CIK numbers."""
    
    SEC_TICKER_URL = "https://www.sec.gov/include/ticker.txt"
    DEFAULT_USER_AGENT = "InvestiGator/1.0 (Vijay Singh; singhvjd@gmail.com)"
    CACHE_DURATION = timedelta(hours=24)
    
    def __init__(self, data_dir: str = "data", config=None):
        """
        Initialize the TickerCIKMapper.
        
        Args:
            data_dir: Directory to store ticker mapping file
            config: Configuration object (optional)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.mapping_file = self.data_dir / "ticker_cik_map.txt"
        self.ticker_map: Dict[str, str] = {}
        
        # Get user agent from config or use default
        if config and hasattr(config, 'sec') and hasattr(config.sec, 'user_agent'):
            self.user_agent = config.sec.user_agent
        elif get_config:
            try:
                config_obj = get_config()
                self.user_agent = config_obj.sec.user_agent
            except:
                self.user_agent = self.DEFAULT_USER_AGENT
        else:
            self.user_agent = self.DEFAULT_USER_AGENT
            
        logger.debug(f"Using user agent: {self.user_agent}")
        self._load_mapping()
    
    def _load_mapping(self) -> None:
        """Load ticker mapping from file into memory."""
        if self._should_update_mapping():
            logger.info("Updating ticker mapping from SEC...")
            self._download_mapping()
        
        if self.mapping_file.exists():
            logger.debug(f"Loading ticker mapping from {self.mapping_file}")
            self.ticker_map.clear()
            
            with open(self.mapping_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        ticker, cik = line.split('\t', 1)
                        self.ticker_map[ticker.lower()] = cik.strip()
            
            logger.info(f"Loaded {len(self.ticker_map)} ticker mappings")
        else:
            logger.warning("Ticker mapping file not found")
    
    def _should_update_mapping(self) -> bool:
        """Check if mapping file should be updated."""
        if not self.mapping_file.exists():
            return True
        
        # Check file age
        file_mtime = datetime.fromtimestamp(self.mapping_file.stat().st_mtime)
        if datetime.now() - file_mtime > self.CACHE_DURATION:
            logger.debug("Ticker mapping is outdated, update needed")
            return True
        
        return False
    
    def _download_mapping(self) -> bool:
        """Download ticker mapping from SEC."""
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/plain',
                'Accept-Encoding': 'gzip, deflate'
            }
            
            response = requests.get(
                self.SEC_TICKER_URL,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            # Validate content
            content = response.text
            lines = content.strip().split('\n')
            if len(lines) < 1000:  # SEC has thousands of tickers
                logger.error(f"Downloaded file seems too small: {len(lines)} lines")
                return False
            
            # Write to temp file first
            temp_file = self.mapping_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                f.write(content)
            
            # Move to final location
            temp_file.replace(self.mapping_file)
            logger.info(f"Successfully downloaded {len(lines)} ticker mappings")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download ticker mapping: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading ticker mapping: {e}")
            return False
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK for a given ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            CIK as string, or None if not found
        """
        ticker_lower = ticker.lower().strip()
        
        # Check if we need to reload mapping
        if not self.ticker_map and self.mapping_file.exists():
            self._load_mapping()
        
        cik = self.ticker_map.get(ticker_lower)
        if cik:
            logger.debug(f"Found CIK {cik} for ticker {ticker}")
            return cik
        else:
            logger.warning(f"CIK not found for ticker: {ticker}")
            return None
    
    def get_cik_padded(self, ticker: str) -> Optional[str]:
        """
        Get CIK padded to 10 digits with leading zeros.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            10-digit padded CIK, or None if not found
        """
        cik = self.get_cik(ticker)
        if cik:
            return f"{int(cik):010d}"
        return None
    
    def resolve_cik(self, symbol: str, provided_cik: str = None) -> Optional[str]:
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
                logger.warning(f"Invalid CIK format provided: {provided_cik}")
        
        # Look up CIK using ticker-CIK mapper
        try:
            cik = self.get_cik(symbol)
            if cik:
                cik_int = int(cik)
                if cik_int > 0:
                    return f"{cik_int:010d}"
        except Exception as e:
            logger.warning(f"Failed to resolve CIK for {symbol}: {e}")
        
        return None
    
    def get_multiple_ciks(self, tickers: List[str]) -> Dict[str, Optional[str]]:
        """
        Get CIKs for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping tickers to CIKs
        """
        return {ticker: self.get_cik(ticker) for ticker in tickers}
    
    def refresh_mapping(self) -> bool:
        """Force refresh of ticker mapping."""
        logger.info("Forcing ticker mapping refresh...")
        return self._download_mapping()
    
    def search_by_cik(self, cik: str) -> Optional[str]:
        """
        Reverse lookup: find ticker by CIK.
        
        Args:
            cik: CIK number
            
        Returns:
            Ticker symbol, or None if not found
        """
        cik_str = str(cik).lstrip('0')  # Remove leading zeros
        
        for ticker, mapped_cik in self.ticker_map.items():
            if mapped_cik.lstrip('0') == cik_str:
                return ticker.upper()
        
        return None
    
    def get_all_tickers(self) -> List[str]:
        """Get list of all available tickers."""
        return sorted([t.upper() for t in self.ticker_map.keys()])
    
    def ticker_exists(self, ticker: str) -> bool:
        """Check if a ticker exists in the mapping."""
        return ticker.lower() in self.ticker_map
    
    def save_cache(self, cache_file: str = "ticker_cache.json") -> None:
        """Save current mapping to a JSON cache file."""
        cache_path = self.data_dir / cache_file
        with open(cache_path, 'w') as f:
            json.dump({
                'last_updated': datetime.now().isoformat(),
                'ticker_count': len(self.ticker_map),
                'mappings': {k.upper(): v for k, v in self.ticker_map.items()}
            }, f, indent=2)
        logger.info(f"Saved ticker cache to {cache_path}")


# Singleton instance
_mapper_instance: Optional[TickerCIKMapper] = None


def get_ticker_mapper(data_dir: str = "data") -> TickerCIKMapper:
    """Get or create singleton TickerCIKMapper instance."""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = TickerCIKMapper(data_dir)
    return _mapper_instance


# Convenience functions
def ticker_to_cik(ticker: str) -> Optional[str]:
    """Convert ticker to CIK using default mapper."""
    return get_ticker_mapper().get_cik(ticker)


def ticker_to_cik_padded(ticker: str) -> Optional[str]:
    """Convert ticker to 10-digit padded CIK."""
    return get_ticker_mapper().get_cik_padded(ticker)


if __name__ == "__main__":
    # Test the mapper
    logging.basicConfig(level=logging.INFO)
    
    mapper = TickerCIKMapper()
    
    # Test some common tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"]
    
    print("\nTicker to CIK Mapping Test:")
    print("-" * 40)
    for ticker in test_tickers:
        cik = mapper.get_cik(ticker)
        cik_padded = mapper.get_cik_padded(ticker)
        print(f"{ticker}: CIK={cik}, Padded={cik_padded}")
    
    # Test reverse lookup
    print("\nReverse Lookup Test:")
    print("-" * 40)
    test_cik = "320193"  # Apple's CIK
    ticker = mapper.search_by_cik(test_cik)
    print(f"CIK {test_cik} -> Ticker: {ticker}")
    
    # Save cache
    mapper.save_cache()
    print("\nCache saved successfully")
