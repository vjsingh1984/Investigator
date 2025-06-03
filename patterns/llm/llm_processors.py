#!/usr/bin/env python3
"""
InvestiGator - LLM Processor Pattern Implementations
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

LLM Processor Pattern Implementations
Chain of Responsibility, Template Method, and Queue-based processors
"""

import logging
import queue
import threading
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import Future
import json

from .llm_interfaces import (
    ILLMHandler, ILLMProcessor, ILLMSubject, ILLMObserver, ILLMAnalysisTemplate,
    LLMRequest, LLMResponse, LLMTaskType, LLMPriority
)
from .llm_strategies import ILLMStrategy, ILLMCacheStrategy
from utils.api_client import OllamaAPIClient

logger = logging.getLogger(__name__)

# ============================================================================
# Chain of Responsibility Handlers
# ============================================================================

class LLMCacheHandler(ILLMHandler):
    """First handler in chain - checks cache for existing responses"""
    
    def __init__(self, cache_manager, cache_strategy: ILLMCacheStrategy):
        super().__init__()
        self.cache_manager = cache_manager
        self.cache_strategy = cache_strategy
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def handle(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Check cache first, pass to next handler if miss"""
        if not self.cache_manager:
            return self._handle_next(request)
        
        try:
            # Generate cache key
            cache_key = self.cache_strategy.get_cache_key(request)
            
            # Try to get from cache
            cached_response = self.cache_manager.get(cache_key, 'llm_response')
            
            if cached_response:
                self.logger.info(f"Cache HIT for request {request.request_id}")
                
                # Reconstruct LLMResponse from cached data
                return LLMResponse(
                    content=cached_response['content'],
                    model=cached_response['model'],
                    processing_time_ms=0,  # Instant from cache
                    tokens_used=cached_response.get('tokens_used'),
                    metadata=cached_response.get('metadata', {}),
                    request_id=request.request_id,
                    timestamp=datetime.utcnow()
                )
            
            self.logger.debug(f"Cache MISS for request {request.request_id}")
            
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")
        
        # Cache miss or error - pass to next handler
        return self._handle_next(request)

class LLMValidationHandler(ILLMHandler):
    """Second handler - validates request parameters"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def handle(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Validate request before processing"""
        try:
            # Validate required fields
            if not request.model:
                return LLMResponse(
                    content="",
                    model="unknown",
                    processing_time_ms=0,
                    error="Model name is required",
                    request_id=request.request_id,
                    timestamp=datetime.utcnow()
                )
            
            if not request.prompt:
                return LLMResponse(
                    content="",
                    model=request.model,
                    processing_time_ms=0,
                    error="Prompt is required",
                    request_id=request.request_id,
                    timestamp=datetime.utcnow()
                )
            
            # Validate prompt length (prevent extremely long prompts)
            if len(request.prompt) > 100000:  # 100k chars
                return LLMResponse(
                    content="",
                    model=request.model,
                    processing_time_ms=0,
                    error="Prompt too long (max 100k characters)",
                    request_id=request.request_id,
                    timestamp=datetime.utcnow()
                )
            
            # Validation passed - continue to next handler
            return self._handle_next(request)
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return LLMResponse(
                content="",
                model=request.model or "unknown",
                processing_time_ms=0,
                error=f"Validation failed: {str(e)}",
                request_id=request.request_id,
                timestamp=datetime.utcnow()
            )

class LLMExecutionHandler(ILLMHandler):
    """Final handler - executes the LLM request"""
    
    def __init__(self, config, cache_manager=None, cache_strategy: ILLMCacheStrategy = None):
        super().__init__()
        self.config = config
        self.cache_manager = cache_manager
        self.cache_strategy = cache_strategy
        self.api_client = OllamaAPIClient(config=config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def handle(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Execute LLM request via API"""
        start_time = time.time()
        
        try:
            # Prepare API request
            api_request = {
                'model': request.model,
                'prompt': request.prompt,
                'options': {
                    'temperature': request.temperature,
                    'top_p': request.top_p
                }
            }
            
            if request.system_prompt:
                api_request['system'] = request.system_prompt
            
            if request.num_ctx:
                api_request['options']['num_ctx'] = request.num_ctx
            
            if request.num_predict:
                api_request['options']['num_predict'] = request.num_predict
            
            # Execute request
            response_data = self.api_client.post('/api/generate', api_request, timeout=request.timeout)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Create response object
            response = LLMResponse(
                content=response_data.get('response', ''),
                model=request.model,
                processing_time_ms=processing_time,
                tokens_used=response_data.get('tokens', 0),
                metadata=request.metadata,
                request_id=request.request_id,
                timestamp=datetime.utcnow()
            )
            
            # Cache the response if strategy allows
            self._cache_response(request, response)
            
            return response
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"LLM execution failed: {e}")
            
            return LLMResponse(
                content="",
                model=request.model,
                processing_time_ms=processing_time,
                error=str(e),
                request_id=request.request_id,
                timestamp=datetime.utcnow()
            )
    
    def _cache_response(self, request: LLMRequest, response: LLMResponse):
        """Cache response if caching is enabled and appropriate"""
        if not self.cache_manager or not self.cache_strategy:
            return
        
        try:
            if self.cache_strategy.should_cache(request, response):
                cache_key = self.cache_strategy.get_cache_key(request)
                
                # Get TTL based on task type
                task_type = None
                if request.metadata and 'task_type' in request.metadata:
                    task_type = LLMTaskType(request.metadata['task_type'])
                
                ttl = self.cache_strategy.get_ttl(task_type) if task_type else 86400
                
                # Cache response data
                cache_data = {
                    'content': response.content,
                    'model': response.model,
                    'processing_time_ms': response.processing_time_ms,
                    'tokens_used': response.tokens_used,
                    'metadata': response.metadata,
                    'timestamp': response.timestamp.isoformat() if response.timestamp else None
                }
                
                self.cache_manager.set(cache_key, cache_data, 'llm_response', ttl=ttl)
                self.logger.debug(f"Cached LLM response with key {cache_key[:16]}...")
                
        except Exception as e:
            self.logger.warning(f"Failed to cache LLM response: {e}")

# ============================================================================
# Queue-Based Processor with Observer Pattern
# ============================================================================

class QueuedLLMProcessor(ILLMProcessor, ILLMSubject):
    """Queue-based LLM processor with observer notifications"""
    
    def __init__(self, config, num_threads: int = 1, cache_manager=None, cache_strategy=None):
        self.config = config
        self.num_threads = num_threads
        self.request_queue = queue.PriorityQueue()
        self.processing_threads = []
        self.stop_event = threading.Event()
        self.observers: List[ILLMObserver] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Create handler chain
        self.handler_chain = self._create_handler_chain(cache_manager, cache_strategy)
        
        # Start processing threads
        self.start()
    
    def _create_handler_chain(self, cache_manager, cache_strategy) -> ILLMHandler:
        """Create processing handler chain"""
        # Create handlers
        cache_handler = LLMCacheHandler(cache_manager, cache_strategy) if cache_manager else None
        validation_handler = LLMValidationHandler()
        execution_handler = LLMExecutionHandler(self.config, cache_manager, cache_strategy)
        
        # Chain handlers
        if cache_handler:
            cache_handler.set_next(validation_handler).set_next(execution_handler)
            return cache_handler
        else:
            validation_handler.set_next(execution_handler)
            return validation_handler
    
    def start(self):
        """Start processing threads"""
        self.stop_event.clear()
        self.processing_threads = []
        
        for i in range(self.num_threads):
            thread = threading.Thread(
                target=self._process_queue,
                daemon=True,
                name=f"LLMProcessor-{i}"
            )
            thread.start()
            self.processing_threads.append(thread)
        
        self.logger.info(f"Started LLM processor with {self.num_threads} threads")
    
    def stop(self):
        """Stop all processing threads"""
        self.stop_event.set()
        
        for thread in self.processing_threads:
            if thread and thread.is_alive():
                thread.join(timeout=10)
        
        self.processing_threads = []
        self.logger.info("Stopped LLM processor")
    
    def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process single request synchronously"""
        future = self._add_request_to_queue(request)
        return future.result()
    
    def process_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Process multiple requests"""
        futures = [self._add_request_to_queue(req) for req in requests]
        return [future.result() for future in futures]
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.request_queue.qsize()
    
    def _add_request_to_queue(self, request: LLMRequest) -> Future:
        """Add request to processing queue"""
        future = Future()
        
        # Use priority for queue ordering (lower number = higher priority)
        priority = request.priority if request.priority else LLMPriority.NORMAL.value
        
        # Add timestamp as tiebreaker
        timestamp = time.time()
        
        self.request_queue.put((priority, timestamp, request, future))
        
        # Notify observers
        self.notify_queued(request)
        
        return future
    
    def _process_queue(self):
        """Background thread processing requests"""
        while not self.stop_event.is_set():
            try:
                # Get request with timeout
                priority, timestamp, request, future = self.request_queue.get(timeout=1.0)
                
                # Notify observers
                self.notify_started(request)
                
                try:
                    # Process through handler chain
                    response = self.handler_chain.handle(request)
                    
                    # Set result
                    future.set_result(response)
                    
                    # Notify observers
                    self.notify_completed(request, response)
                    
                except Exception as e:
                    self.logger.error(f"Error processing request {request.request_id}: {e}")
                    
                    # Create error response
                    error_response = LLMResponse(
                        content="",
                        model=request.model,
                        processing_time_ms=0,
                        error=str(e),
                        request_id=request.request_id,
                        timestamp=datetime.utcnow()
                    )
                    
                    future.set_result(error_response)
                    
                    # Notify observers
                    self.notify_error(request, e)
                
                finally:
                    self.request_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Queue processing error: {e}")
    
    # Observer pattern implementation
    def attach(self, observer: ILLMObserver) -> None:
        """Attach observer"""
        self.observers.append(observer)
    
    def detach(self, observer: ILLMObserver) -> None:
        """Detach observer"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_queued(self, request: LLMRequest) -> None:
        """Notify observers of queued request"""
        for observer in self.observers:
            try:
                observer.on_request_queued(request)
            except Exception as e:
                self.logger.warning(f"Observer notification failed: {e}")
    
    def notify_started(self, request: LLMRequest) -> None:
        """Notify observers of started processing"""
        for observer in self.observers:
            try:
                observer.on_processing_started(request)
            except Exception as e:
                self.logger.warning(f"Observer notification failed: {e}")
    
    def notify_completed(self, request: LLMRequest, response: LLMResponse) -> None:
        """Notify observers of completed processing"""
        for observer in self.observers:
            try:
                observer.on_processing_completed(request, response)
            except Exception as e:
                self.logger.warning(f"Observer notification failed: {e}")
    
    def notify_error(self, request: LLMRequest, error: Exception) -> None:
        """Notify observers of processing error"""
        for observer in self.observers:
            try:
                observer.on_processing_error(request, error)
            except Exception as e:
                self.logger.warning(f"Observer notification failed: {e}")

# ============================================================================
# Template Method Implementation
# ============================================================================

class StandardLLMAnalysisTemplate(ILLMAnalysisTemplate):
    """Standard template for LLM analysis workflows"""
    
    def __init__(self, processor: ILLMProcessor, strategy: ILLMStrategy):
        self.processor = processor
        self.strategy = strategy
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_input(self, symbol: str, data: Dict[str, Any], task_type: LLMTaskType) -> bool:
        """Validate input parameters"""
        if not symbol or len(symbol) > 10:
            return False
        
        if not data:
            return False
        
        if not isinstance(task_type, LLMTaskType):
            return False
        
        return True
    
    def prepare_analysis_request(self, symbol: str, data: Dict[str, Any], task_type: LLMTaskType) -> LLMRequest:
        """Prepare analysis request using strategy"""
        return self.strategy.prepare_request(task_type, {**data, 'symbol': symbol})
    
    def execute_analysis(self, request: LLMRequest) -> LLMResponse:
        """Execute analysis using processor"""
        return self.processor.process_request(request)
    
    def process_analysis_results(self, response: LLMResponse, task_type: LLMTaskType) -> Dict[str, Any]:
        """Process results using strategy"""
        return self.strategy.process_response(response, task_type)
    
    def create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'error': error_message,
            'timestamp': datetime.utcnow().isoformat(),
            'success': False
        }