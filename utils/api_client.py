#!/usr/bin/env python3
"""
InvestiGator - API Client Utilities
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

API Client Utilities - Centralized HTTP client patterns
Eliminates duplicate HTTP session management, rate limiting, and retry logic
"""

import time
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


def rate_limit(delay: float = 0.1):
    """
    Decorator for rate limiting API calls
    
    Args:
        delay: Minimum delay between calls in seconds
    """
    def decorator(func: Callable) -> Callable:
        func._last_call_time = 0.0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            time_since_last_call = current_time - func._last_call_time
            
            if time_since_last_call < delay:
                sleep_time = delay - time_since_last_call
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            result = func(*args, **kwargs)
            func._last_call_time = time.time()
            return result
        
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator for retrying failed API calls with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise
                    
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
            
        return wrapper
    return decorator


class BaseAPIClient(ABC):
    """
    Base class for API clients with common functionality
    """
    
    # Class-level rate limiting tracker for different hosts
    _rate_limit_tracker: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, base_url: str, user_agent: str, rate_limit_delay: float = 0.1, timeout: Optional[int] = None):
        """
        Initialize API client
        
        Args:
            base_url: Base URL for API
            user_agent: User agent string for requests
            rate_limit_delay: Delay between requests in seconds
            timeout: Default timeout for requests in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout or 30  # Default 30 seconds if not specified
        
        # Extract host for rate limiting tracking
        parsed_url = urlparse(self.base_url)
        self.host = parsed_url.netloc or parsed_url.path  # Handle cases like localhost:11434
        
        # Initialize rate limiting tracker for this host if not exists
        if self.host not in BaseAPIClient._rate_limit_tracker:
            BaseAPIClient._rate_limit_tracker[self.host] = {
                'last_request_time': 0.0,
                'request_count': 0,
                'created_at': datetime.now().isoformat()
            }
        
        # Initialize session with common headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests with host-based tracking"""
        current_time = time.time()
        tracker = BaseAPIClient._rate_limit_tracker[self.host]
        
        time_since_last = current_time - tracker['last_request_time']
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting for {self.host}: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)
        
        # Update tracker
        tracker['last_request_time'] = time.time()
        tracker['request_count'] += 1
        
        # Log stats every 10 requests
        if tracker['request_count'] % 10 == 0:
            logger.info(f"API Stats for {self.host}: {tracker['request_count']} requests, "
                       f"last request at {datetime.fromtimestamp(tracker['last_request_time']).isoformat()}")
    
    @retry_on_failure(max_retries=3, backoff_factor=0.5)
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make HTTP request with rate limiting and error handling
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
        """
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Use timeout from kwargs or default
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
        
        try:
            logger.debug(f"Making {method} request to {url} (timeout: {kwargs['timeout']}s)")
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {method} {url} - {e}")
            raise
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> requests.Response:
        """Make GET request"""
        return self._make_request('GET', endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict] = None, json: Optional[Dict] = None) -> requests.Response:
        """Make POST request"""
        return self._make_request('POST', endpoint, data=data, json=json)
    
    def get_json(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request and return JSON response"""
        response = self.get(endpoint, params=params)
        return response.json()
    
    def post_json(self, endpoint: str, data: Optional[Dict] = None, json: Optional[Dict] = None) -> Dict[str, Any]:
        """Make POST request and return JSON response"""
        response = self.post(endpoint, data=data, json=json)
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this API client"""
        tracker = BaseAPIClient._rate_limit_tracker.get(self.host, {})
        return {
            'host': self.host,
            'request_count': tracker.get('request_count', 0),
            'last_request_time': datetime.fromtimestamp(tracker.get('last_request_time', 0)).isoformat() if tracker.get('last_request_time', 0) > 0 else 'Never',
            'created_at': tracker.get('created_at', 'Unknown'),
            'rate_limit_delay': self.rate_limit_delay
        }
    
    @staticmethod
    def get_all_stats() -> Dict[str, Dict[str, Any]]:
        """Get statistics for all API clients"""
        stats = {}
        for host, tracker in BaseAPIClient._rate_limit_tracker.items():
            stats[host] = {
                'request_count': tracker.get('request_count', 0),
                'last_request_time': datetime.fromtimestamp(tracker.get('last_request_time', 0)).isoformat() if tracker.get('last_request_time', 0) > 0 else 'Never',
                'created_at': tracker.get('created_at', 'Unknown')
            }
        return stats
    
    def close(self) -> None:
        """Close the HTTP session"""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class SECAPIClient(BaseAPIClient):
    """Specialized client for SEC EDGAR API"""
    
    def __init__(self, user_agent: str, config=None):
        # Get timeout from config if available
        timeout = 30  # Default
        if config and hasattr(config, 'sec') and hasattr(config.sec, 'timeout'):
            timeout = config.sec.timeout
        
        # Get rate limit from config if available
        rate_limit_delay = 0.1  # Default 10 requests/second
        if config and hasattr(config, 'sec') and hasattr(config.sec, 'rate_limit'):
            rate_limit_delay = 1.0 / config.sec.rate_limit
        
        # SEC allows 10 requests per second
        super().__init__(
            base_url="https://data.sec.gov",
            user_agent=user_agent,
            rate_limit_delay=rate_limit_delay,
            timeout=timeout
        )
    
    def get_company_facts(self, cik: str) -> Dict[str, Any]:
        """Get company facts for CIK"""
        return self.get_json(f"/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json")
    
    def get_submissions(self, cik: str) -> Dict[str, Any]:
        """Get submissions for CIK"""
        return self.get_json(f"/submissions/CIK{cik.zfill(10)}.json")
    
    def get_frame_data(self, concept: str, unit: str, year: int) -> Dict[str, Any]:
        """Get frame data for concept"""
        return self.get_json(f"/api/xbrl/frames/{concept}/{unit}/CY{year}.json")


class OllamaAPIClient(BaseAPIClient):
    """Specialized client for Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", config=None):
        # Get timeout from config if available
        timeout = 300  # Default 5 minutes for LLM
        if config and hasattr(config, 'ollama') and hasattr(config.ollama, 'timeout'):
            timeout = config.ollama.timeout
        
        # Get rate limit from config if available
        rate_limit_delay = 0.01  # Default 100 requests/second
        if config and hasattr(config, 'ollama') and hasattr(config.ollama, 'rate_limit_delay'):
            rate_limit_delay = config.ollama.rate_limit_delay
        
        super().__init__(
            base_url=base_url,
            user_agent="InvestiGator/1.0",
            rate_limit_delay=rate_limit_delay,
            timeout=timeout
        )
    
    def generate(self, model: str, prompt: str, system: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate text using Ollama model"""
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False
        }
        
        if system:
            payload['system'] = system
            
        # Add options if provided
        options = {}
        for key in ['temperature', 'top_p', 'num_ctx', 'num_predict']:
            if key in kwargs:
                options[key] = kwargs.pop(key)
        
        if options:
            payload['options'] = options
            
        # Add any remaining kwargs to payload
        payload.update(kwargs)
        
        return self.post_json('/api/generate', json=payload)
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        return self.get_json('/api/tags')
    
    def show_model(self, model: str) -> Dict[str, Any]:
        """Show model information"""
        return self.post_json('/api/show', json={'name': model})