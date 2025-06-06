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
    
    def post(self, endpoint: str, data: Optional[Dict] = None, json: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Make POST request"""
        return self._make_request('POST', endpoint, data=data, json=json, **kwargs)
    
    def get_json(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request and return JSON response"""
        response = self.get(endpoint, params=params)
        return response.json()
    
    def post_json(self, endpoint: str, data: Optional[Dict] = None, json: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Make POST request and return JSON response"""
        response = self.post(endpoint, data=data, json=json, **kwargs)
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
    
    def get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """
        Get model capabilities including context size
        
        Returns:
            Dict containing context_size, parameter_count, etc.
        """
        try:
            model_info = self.show_model(model)
            
            # Extract context size from model info
            # Use model-specific defaults based on known model capabilities
            context_size = self._get_default_context_size(model)
            parameter_size = 0
            
            # First, check the modelfile for PARAMETER num_ctx
            if 'modelfile' in model_info:
                modelfile = model_info['modelfile']
                if isinstance(modelfile, str):
                    # Parse PARAMETER num_ctx from modelfile
                    import re
                    num_ctx_match = re.search(r'PARAMETER\s+num_ctx\s+(\d+)', modelfile)
                    if num_ctx_match:
                        context_size = int(num_ctx_match.group(1))
                        logger.debug(f"Found num_ctx={context_size} in modelfile for {model}")
            
            # Check modelinfo for context window as backup
            if context_size == 4096 and 'modelinfo' in model_info:
                modelinfo = model_info['modelinfo']
                
                # Parse context length from various possible fields
                if isinstance(modelinfo, dict):
                    # Common fields where context size might be stored
                    context_fields = [
                        'context_length',
                        'max_position_embeddings',
                        'n_ctx',
                        'max_seq_len',
                        'context_window'
                    ]
                    
                    for field in context_fields:
                        if field in modelinfo and isinstance(modelinfo[field], (int, float)):
                            context_size = int(modelinfo[field])
                            logger.debug(f"Found {field}={context_size} in modelinfo for {model}")
                            break
                    
                    # Extract parameter count
                    param_fields = ['num_parameters', 'parameter_count', 'params']
                    for field in param_fields:
                        if field in modelinfo and isinstance(modelinfo[field], (int, float, str)):
                            parameter_size = modelinfo[field]
                            break
            
            # Check model file details for additional info
            if 'details' in model_info:
                details = model_info['details']
                if isinstance(details, dict):
                    if 'parameter_size' in details:
                        parameter_size = details['parameter_size']
                    
                    # Some models store context in details
                    if 'context_length' in details:
                        detected_context = int(details['context_length'])
                        if detected_context > context_size:  # Use the larger value
                            context_size = detected_context
                            logger.debug(f"Found context_length={context_size} in details for {model}")
            
            # Extract parameter size from modelfile if available
            if parameter_size == 0 and 'modelfile' in model_info:
                modelfile = model_info['modelfile']
                if isinstance(modelfile, str):
                    # Look for patterns like "30B", "70B", etc.
                    import re
                    param_match = re.search(r'(\d+(?:\.\d+)?)B', model)
                    if param_match:
                        parameter_size = f"{param_match.group(1)}B"
            
            # Calculate memory requirements if we have parameter size
            memory_requirements = self._estimate_memory_requirements(parameter_size)
            
            return {
                'model_name': model,
                'context_size': context_size,
                'parameter_size': parameter_size,
                'memory_requirements': memory_requirements,
                'available': True,
                'raw_info': model_info
            }
            
        except Exception as e:
            logger.warning(f"Failed to get capabilities for model {model}: {e}")
            return {
                'model_name': model,
                'context_size': 4096,  # Conservative default
                'parameter_size': 0,
                'memory_requirements': {},
                'available': False,
                'error': str(e)
            }
    
    def _get_default_context_size(self, model: str) -> int:
        """Get default context size for known models when Ollama API doesn't provide it"""
        # Known model context sizes based on model specifications
        model_contexts = {
            # Llama 3.1 models have 128K context
            'llama3.1': 131072,
            'llama-3.1': 131072,
            
            # Llama 3.3 models 
            'llama3.3': 131072,
            'llama-3.3': 131072,
            
            # Qwen models
            'qwen2.5:32b': 32768,
            'qwen3-30b-40k': 40960,
            'qwen3:30b': 40960,
            
            # Phi models
            'phi4-reasoning:plus': 32768,
            'phi-4': 16384,
            'phi3': 4096,
            
            # Mixtral models
            'mixtral-8x7b': 32768,
            'mixtral:8x7b': 32768,
            
            # Mistral models
            'mistral:7b': 8192,
            'mistral:latest': 8192,
        }
        
        # Try to match model name with known contexts
        for model_key, context_size in model_contexts.items():
            if model_key in model.lower():
                logger.debug(f"Using known context size {context_size} for model {model}")
                return context_size
        
        # Default fallback
        logger.warning(f"Unknown model {model}, using conservative 4096 context size")
        return 4096
    
    def _estimate_memory_requirements(self, parameter_size) -> Dict[str, Any]:
        """Estimate memory requirements for a model"""
        if not parameter_size:
            return {}
        
        try:
            # Extract numeric value from parameter size string
            import re
            if isinstance(parameter_size, str):
                match = re.search(r'(\d+(?:\.\d+)?)', str(parameter_size))
                if match:
                    params_b = float(match.group(1))
                else:
                    return {}
            else:
                params_b = float(parameter_size) / 1e9  # Convert to billions
            
            # Estimate memory requirements (rough calculations)
            # Q4_K quantization uses ~4 bits per parameter on average
            # Plus overhead for KV cache, activations, etc.
            model_memory_gb = params_b * 0.5  # ~4 bits per param in GB
            kv_cache_gb = 2.0  # Estimate for KV cache
            overhead_gb = 2.0  # OS and other overhead
            
            total_estimated_gb = model_memory_gb + kv_cache_gb + overhead_gb
            
            # Get system memory
            try:
                import subprocess
                result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
                if result.returncode == 0:
                    system_memory_bytes = int(result.stdout.split(': ')[1])
                    system_memory_gb = system_memory_bytes / (1024**3)
                else:
                    system_memory_gb = 64  # Default assumption
            except:
                system_memory_gb = 64  # Default assumption
            
            memory_sufficient = total_estimated_gb <= (system_memory_gb * 0.8)  # Leave 20% buffer
            
            return {
                'model_memory_gb': round(model_memory_gb, 1),
                'kv_cache_gb': kv_cache_gb,
                'total_estimated_gb': round(total_estimated_gb, 1),
                'system_memory_gb': round(system_memory_gb, 1),
                'memory_sufficient': memory_sufficient,
                'utilization_percent': round((total_estimated_gb / system_memory_gb) * 100, 1)
            }
            
        except Exception as e:
            logger.debug(f"Error estimating memory requirements: {e}")
            return {}