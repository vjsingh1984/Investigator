#!/usr/bin/env python3
"""
InvestiGator - SEC Frame API Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

SEC Frame API Module
Handles interactions with SEC EDGAR Frame API for consolidated financial data
"""

import logging
import requests
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from config import get_config
from utils.cache.cache_manager import get_cache_manager
from utils.cache.cache_types import CacheType

logger = logging.getLogger(__name__)

@dataclass
class FrameAPIRequest:
    """Represents a Frame API request"""
    concept: str
    unit: str
    year: int
    period: Optional[str] = None  # None for annual, Q1/Q2/Q3/Q4 for quarterly
    
    def get_url_path(self) -> str:
        """Get URL path for this request"""
        if self.period:
            return f"/api/xbrl/frames/{self.concept}/{self.unit}/CY{self.year}{self.period}.json"
        else:
            return f"/api/xbrl/frames/{self.concept}/{self.unit}/CY{self.year}.json"

class SECFrameAPI:
    """
    Handles SEC EDGAR Frame API interactions.
    
    The Frame API provides consolidated XBRL data across all companies for specific
    concepts, units, and time periods.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.cache_manager = get_cache_manager()
        self.session = requests.Session()
        
        # Set up headers
        self.session.headers.update({
            'User-Agent': self.config.sec.user_agent,
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0 / self.config.sec.rate_limit  # seconds
        
        self.main_logger = self.config.get_main_logger('sec_frame_api')
    
    def get_frame_data(self, concept: str, unit: str, year: int, 
                      period: Optional[str] = None, use_cache: bool = True) -> Optional[Dict]:
        """
        Get frame data for a specific concept/unit/period.
        
        Args:
            concept: XBRL concept (e.g., 'us-gaap:Revenues')
            unit: Unit of measurement (e.g., 'USD')
            year: Calendar year
            period: Quarter (Q1, Q2, Q3, Q4) or None for annual
            use_cache: Whether to use cached data
            
        Returns:
            Frame data dictionary or None if not found
        """
        try:
            request = FrameAPIRequest(concept, unit, year, period)
            
            # Check cache first
            if use_cache:
                cached_data = self._get_cached_frame_data(request)
                if cached_data:
                    return cached_data
            
            # Fetch from API
            response_data = self._fetch_frame_data(request)
            
            # Cache the response
            if response_data and use_cache:
                self._cache_frame_data(request, response_data)
            
            return response_data
            
        except Exception as e:
            self.main_logger.error(f"Error getting frame data for {concept}: {e}")
            return None
    
    def get_company_frame_data(self, cik: str, concept: str, unit: str, 
                             year: int, period: Optional[str] = None) -> Optional[Dict]:
        """
        Get frame data for a specific company.
        
        Args:
            cik: Company CIK
            concept: XBRL concept
            unit: Unit of measurement
            year: Calendar year
            period: Quarter or None for annual
            
        Returns:
            Company-specific data from frame or None
        """
        try:
            frame_data = self.get_frame_data(concept, unit, year, period)
            if not frame_data:
                return None
            
            # Find data for this CIK
            data_entries = frame_data.get('data', [])
            
            # Normalize CIK (remove leading zeros for comparison)
            normalized_cik = str(int(cik))
            
            for entry in data_entries:
                if str(entry.get('cik', 0)) == normalized_cik:
                    return entry
            
            return None
            
        except Exception as e:
            self.main_logger.error(f"Error getting company frame data: {e}")
            return None
    
    def get_multiple_concepts(self, concepts: List[str], unit: str, year: int,
                            period: Optional[str] = None) -> Dict[str, Dict]:
        """
        Get frame data for multiple concepts efficiently.
        
        Args:
            concepts: List of XBRL concepts
            unit: Unit of measurement
            year: Calendar year
            period: Quarter or None for annual
            
        Returns:
            Dictionary mapping concept to frame data
        """
        results = {}
        
        for concept in concepts:
            try:
                # Add rate limiting between requests
                self._rate_limit()
                
                frame_data = self.get_frame_data(concept, unit, year, period)
                if frame_data:
                    results[concept] = frame_data
                    
            except Exception as e:
                self.main_logger.error(f"Error fetching concept {concept}: {e}")
                continue
        
        return results
    
    def search_company_concepts(self, cik: str, concepts: List[str], 
                              unit: str, year: int, period: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for multiple concepts for a specific company.
        
        Args:
            cik: Company CIK
            concepts: List of XBRL concepts to search
            unit: Unit of measurement
            year: Calendar year
            period: Quarter or None for annual
            
        Returns:
            Dictionary of found concept data
        """
        found_data = {}
        
        for concept in concepts:
            try:
                self._rate_limit()
                
                company_data = self.get_company_frame_data(cik, concept, unit, year, period)
                if company_data:
                    found_data[concept] = company_data
                    
            except Exception as e:
                self.main_logger.error(f"Error searching concept {concept} for CIK {cik}: {e}")
                continue
        
        return found_data
    
    def _fetch_frame_data(self, request: FrameAPIRequest) -> Optional[Dict]:
        """Fetch frame data from SEC API"""
        try:
            self._rate_limit()
            
            url = f"{self.config.sec.base_url}{request.get_url_path()}"
            
            self.main_logger.debug(f"Fetching frame data: {url}")
            
            response = self.session.get(url, timeout=self.config.sec.timeout)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                self.main_logger.debug(f"Frame data not found: {request.concept} {request.year}")
                return None
            else:
                self.main_logger.warning(f"Frame API request failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.main_logger.error(f"Frame API request error: {e}")
            return None
        except Exception as e:
            self.main_logger.error(f"Unexpected error fetching frame data: {e}")
            return None
    
    def _get_cached_frame_data(self, request: FrameAPIRequest) -> Optional[Dict]:
        """Get cached frame data"""
        try:
            cache_key = f"frame_{request.concept}_{request.unit}_{request.year}"
            if request.period:
                cache_key += f"_{request.period}"
            
            return self.cache_manager.get(
                CacheType.SEC_RESPONSE,
                cache_key
            )
            
        except Exception as e:
            self.main_logger.error(f"Error getting cached frame data: {e}")
            return None
    
    def _cache_frame_data(self, request: FrameAPIRequest, data: Dict):
        """Cache frame data"""
        try:
            cache_key = f"frame_{request.concept}_{request.unit}_{request.year}"
            if request.period:
                cache_key += f"_{request.period}"
            
            metadata = {
                'concept': request.concept,
                'unit': request.unit,
                'year': request.year,
                'period': request.period,
                'api_url': request.get_url_path(),
                'data_count': len(data.get('data', [])),
                'fetched_at': datetime.utcnow().isoformat()
            }
            
            self.cache_manager.set(
                CacheType.SEC_RESPONSE,
                cache_key,
                {'data': data, 'metadata': metadata}
            )
            
        except Exception as e:
            self.main_logger.error(f"Error caching frame data: {e}")
    
    def _rate_limit(self):
        """Implement rate limiting for SEC API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            self.main_logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_available_periods(self, concept: str, unit: str, year: int) -> List[str]:
        """
        Get available periods for a concept/unit/year.
        
        Args:
            concept: XBRL concept
            unit: Unit of measurement  
            year: Calendar year
            
        Returns:
            List of available periods (may include 'annual', 'Q1', 'Q2', 'Q3', 'Q4')
        """
        available_periods = []
        
        # Check annual data
        annual_data = self.get_frame_data(concept, unit, year, None)
        if annual_data and annual_data.get('data'):
            available_periods.append('annual')
        
        # Check quarterly data
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            quarterly_data = self.get_frame_data(concept, unit, year, quarter)
            if quarterly_data and quarterly_data.get('data'):
                available_periods.append(quarter)
        
        return available_periods
    
    def get_concept_taxonomy(self, concept: str) -> Optional[Dict]:
        """
        Get taxonomy information for a concept.
        
        Args:
            concept: XBRL concept
            
        Returns:
            Taxonomy information if available
        """
        try:
            # This would fetch taxonomy data from SEC
            # For now, return basic info from config
            xbrl_abbreviations = self.config.sec.xbrl_tag_abbreviations
            
            # Remove namespace prefix for lookup
            clean_concept = concept.replace('us-gaap:', '')
            
            if clean_concept in xbrl_abbreviations:
                return {
                    'concept': concept,
                    'abbreviation': xbrl_abbreviations[clean_concept],
                    'namespace': 'us-gaap'
                }
            
            return None
            
        except Exception as e:
            self.main_logger.error(f"Error getting concept taxonomy: {e}")
            return None

def get_frame_api() -> SECFrameAPI:
    """Get SEC Frame API instance"""
    return SECFrameAPI()