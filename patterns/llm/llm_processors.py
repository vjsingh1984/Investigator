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
from .llm_model_config import get_model_config_manager
from utils.api_client import OllamaAPIClient
from utils.llm_response_processor import get_llm_response_processor

logger = logging.getLogger(__name__)

# ============================================================================
# Chain of Responsibility Handlers
# ============================================================================

class LLMCacheHandler(ILLMHandler):
    """First handler in chain - checks cache for existing responses"""
    
    def __init__(self, cache_manager, cache_strategy: ILLMCacheStrategy, config=None):
        super().__init__()
        self.cache_manager = cache_manager
        self.cache_strategy = cache_strategy
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _log_to_both(self, symbol: str, message: str, level: str = 'info'):
        """Log to both symbol-specific and main loggers"""
        # Log to main logger
        getattr(self.logger, level)(message)
        
        # Log to symbol logger if config is available
        if self.config and symbol and symbol != 'UNKNOWN':
            try:
                symbol_logger = self.config.get_symbol_logger(symbol, 'llm_cache')
                getattr(symbol_logger, level)(message)
            except Exception as e:
                self.logger.debug(f"Failed to log to symbol logger for {symbol}: {e}")
    
    def _generate_cache_key_dict(self, request: LLMRequest) -> Dict[str, str]:
        """Generate dictionary-based cache key consistent with file cache handler"""
        if not request.metadata:
            return {}
        
        symbol = request.metadata.get('symbol', 'UNKNOWN')
        task_type = request.metadata.get('task_type', 'unknown')
        
        # Map task types to the format expected by file cache handler
        if task_type in ['quarterly_summary', 'comprehensive_analysis']:
            # For SEC fundamental analysis
            llm_type = 'sec'
            
            # Try to get form_type and period from metadata
            form_type = request.metadata.get('form_type', '10-Q')
            fiscal_year = request.metadata.get('fiscal_year', '')
            fiscal_period = request.metadata.get('fiscal_period', '')
            
            if fiscal_year and fiscal_period:
                period = f"{fiscal_year}-{fiscal_period}"
            else:
                period = request.metadata.get('period', '')
            
            # Handle comprehensive analysis
            if task_type == 'comprehensive_analysis':
                form_type = 'COMPREHENSIVE'
                if not period:
                    period = f"{fiscal_year}-FY" if fiscal_year else "2025-FY"
            
            return {
                'symbol': symbol,
                'form_type': form_type,
                'period': period,
                'llm_type': llm_type
            }
        elif task_type == 'technical_analysis':
            return {
                'symbol': symbol,
                'llm_type': 'ta'
            }
        elif task_type == 'synthesis':
            return {
                'symbol': symbol,
                'form_type': 'SYNTHESIS',
                'period': f"{request.metadata.get('fiscal_year', '2025')}-{request.metadata.get('fiscal_period', 'Q1')}",
                'fiscal_year': request.metadata.get('fiscal_year', 2025),
                'fiscal_period': request.metadata.get('fiscal_period', 'Q1'),
                'llm_type': 'full'
            }
        else:
            # Generic fallback
            return {
                'symbol': symbol,
                'llm_type': task_type
            }
    
    def handle(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Check cache first, pass to next handler if miss"""
        if not self.cache_manager:
            return self._handle_next(request)
        
        try:
            # Generate dictionary-based cache key (consistent with file handler)
            cache_key = self._generate_cache_key_dict(request)
            
            # Try to get from cache
            from utils.cache.cache_types import CacheType
            cached_response = self.cache_manager.get(CacheType.LLM_RESPONSE, cache_key)
            
            if cached_response:
                symbol = request.metadata.get('symbol', 'UNKNOWN') if request.metadata else 'UNKNOWN'
                task_type = request.metadata.get('task_type', 'unknown') if request.metadata else 'unknown'
                
                # Log comprehensive cache hit details
                cached_tokens = cached_response.get('tokens_used', 0)
                cached_content_length = len(cached_response.get('content', ''))
                cache_metadata = cached_response.get('metadata', {})
                
                cache_hit_msg = (f"🎯 CACHE HIT - {symbol} {task_type} | "
                               f"Model: {cached_response.get('model', 'unknown')} | "
                               f"Tokens: {cached_tokens} | "
                               f"Response length: {cached_content_length} chars | "
                               f"Request: {request.request_id[:8]}")
                self._log_to_both(symbol, cache_hit_msg)
                
                # Log cache metadata if available
                if cache_metadata.get('fiscal_period'):
                    cache_details_msg = (f"🎯 CACHE HIT DETAILS - {symbol} | "
                                       f"Fiscal: {cache_metadata.get('fiscal_period')} | "
                                       f"Cached at: {cache_metadata.get('created_at', 'unknown')}")
                    self._log_to_both(symbol, cache_details_msg)
                
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
            
            symbol = request.metadata.get('symbol', 'UNKNOWN') if request.metadata else 'UNKNOWN'
            task_type = request.metadata.get('task_type', 'unknown') if request.metadata else 'unknown'
            prompt_length = len(request.prompt) + len(request.system_prompt or '')
            
            cache_miss_msg = (f"❌ CACHE MISS - {symbol} {task_type} | "
                           f"Model: {request.model} | "
                           f"Prompt length: {prompt_length} chars | "
                           f"Request: {request.request_id[:8]} | "
                           f"Cache key: {cache_key[:32]}...")
            self._log_to_both(symbol, cache_miss_msg)
            
            # Log additional request details for troubleshooting
            if request.metadata:
                fiscal_info = request.metadata.get('fiscal_period', 'N/A')
                miss_details_msg = (f"❌ CACHE MISS DETAILS - {symbol} | "
                                  f"Fiscal: {fiscal_info} | "
                                  f"Task priority: {request.priority.value if hasattr(request.priority, 'value') else 'unknown'}")
                self._log_to_both(symbol, miss_details_msg)
            
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")
        
        # Cache miss or error - pass to next handler
        return self._handle_next(request)
    
    def store_response(self, request: LLMRequest, response: LLMResponse):
        """Store LLM response in cache"""
        if not self.cache_manager or not self.cache_strategy:
            return
        
        try:
            if self.cache_strategy.should_cache(request, response):
                # Generate dictionary-based cache key 
                cache_key = self._generate_cache_key_dict(request)
                
                # Get TTL based on task type
                task_type = None
                if request.metadata and 'task_type' in request.metadata:
                    from .llm_interfaces import LLMTaskType
                    task_type = LLMTaskType(request.metadata['task_type'])
                
                ttl = self.cache_strategy.get_ttl(task_type) if task_type else 86400
                
                # Cache response data
                cache_data = {
                    'prompt': request.prompt,  # Include prompt for file cache storage
                    'response': response.content,  # Store actual response content
                    'content': response.content,  # Keep for backward compatibility
                    'model': response.model,
                    'processing_time_ms': response.processing_time_ms,
                    'tokens_used': response.tokens_used,
                    'metadata': response.metadata,
                    'timestamp': response.timestamp.isoformat() if response.timestamp else None,
                    'model_info': {  # Add model info for comprehensive caching
                        'model': response.model,
                        'temperature': getattr(request, 'temperature', 0.3),
                        'top_p': getattr(request, 'top_p', 0.9)
                    }
                }
                
                from utils.cache.cache_types import CacheType
                success = self.cache_manager.set(CacheType.LLM_RESPONSE, cache_key, cache_data)
                symbol = request.metadata.get('symbol', 'UNKNOWN') if request.metadata else 'UNKNOWN'
                task_type_str = request.metadata.get('task_type', 'unknown') if request.metadata else 'unknown'
                
                if success:
                    cache_write_msg = (f"💾 CACHE STORE - {symbol} {task_type_str} | "
                                     f"Size: {len(response.content)} chars | "
                                     f"Tokens: {response.tokens_used} | "
                                     f"TTL: {ttl//86400}d")
                    self._log_to_both(symbol, cache_write_msg)
                else:
                    cache_fail_msg = (f"❌ CACHE STORE FAILED - {symbol} {task_type_str}")
                    self._log_to_both(symbol, cache_fail_msg, 'warning')
                
        except Exception as e:
            self.logger.warning(f"Failed to store LLM response in cache: {e}")

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
    
    def __init__(self, config, cache_handler=None):
        super().__init__()
        self.config = config
        self.cache_handler = cache_handler  # Delegate all cache operations to this
        self.api_client = OllamaAPIClient(config=config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model_capabilities_cache = {}  # Cache model info to avoid repeated API calls
        self.model_config_manager = get_model_config_manager()
        self.response_processor = get_llm_response_processor()
    
    def _log_to_both(self, symbol: str, message: str, level: str = 'info'):
        """Log to both symbol-specific and main loggers"""
        # Log to main logger
        getattr(self.logger, level)(message)
        
        # Log to symbol logger if config is available
        if self.config and symbol and symbol != 'UNKNOWN':
            try:
                symbol_logger = self.config.get_symbol_logger(symbol, 'llm_execution')
                getattr(symbol_logger, level)(message)
            except Exception as e:
                self.logger.debug(f"Failed to log to symbol logger for {symbol}: {e}")
    
    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """Get and cache model capabilities from Ollama API"""
        if model_name in self.model_capabilities_cache:
            return self.model_capabilities_cache[model_name]
        
        try:
            capabilities = self.api_client.get_model_capabilities(model_name)
            self.model_capabilities_cache[model_name] = capabilities
            
            self.logger.info(f"🔍 MODEL INFO - {model_name}: context_size={capabilities['context_size']}, available={capabilities['available']}")
            if capabilities.get('parameter_size'):
                self.logger.info(f"🔍 MODEL INFO - {model_name}: parameter_size={capabilities['parameter_size']}")
            
            # Log memory requirements if available
            memory_req = capabilities.get('memory_requirements', {})
            if memory_req:
                self.logger.info(f"🔍 MEMORY INFO - {model_name}: estimated={memory_req.get('total_estimated_gb', 0)}GB, "
                               f"system={memory_req.get('system_memory_gb', 0)}GB, "
                               f"sufficient={memory_req.get('memory_sufficient', True)}")
                
                if not memory_req.get('memory_sufficient', True):
                    self.logger.warning(f"⚠️ MEMORY WARNING - {model_name} may require {memory_req.get('total_estimated_gb', 0)}GB "
                                      f"but system only has {memory_req.get('system_memory_gb', 0)}GB available")
            
            return capabilities
        except Exception as e:
            self.logger.error(f"Failed to get model capabilities for {model_name}: {e}")
            # Return conservative defaults
            fallback = {
                'model_name': model_name,
                'context_size': 4096,
                'parameter_size': 0,
                'available': True,  # Assume available for fallback
                'error': str(e)
            }
            self.model_capabilities_cache[model_name] = fallback
            return fallback
    
    def calculate_dynamic_context_size(self, request: LLMRequest) -> Dict[str, int]:
        """Calculate appropriate context size based on model capabilities and prompt length"""
        # First try our known model configurations
        task_type = request.metadata.get('task_type', 'general')
        if hasattr(task_type, 'value'):
            task_type = task_type.value
        
        # Map LLMTaskType values to our config task types
        task_type_map = {
            'comprehensive_analysis': 'sec',
            'quarterly_summary': 'sec',
            'technical_analysis': 'ta',
            'synthesis': 'synthesis'
        }
        config_task_type = task_type_map.get(task_type, 'general')
        
        # Use enhanced model configuration
        context_params = self.model_config_manager.get_optimal_context_size(
            model_name=request.model,
            prompt_length=len(request.prompt) + len(request.system_prompt or ''),
            desired_output=request.num_predict,
            task_type=config_task_type
        )
        
        # Also get capabilities from Ollama API for comparison
        model_caps = self.get_model_capabilities(request.model)
        api_context = model_caps.get('context_size', 4096)
        
        # Calculate actual prompt size (including system prompt)
        prompt_tokens = len(request.prompt) // 4  # Rough estimation: 4 chars per token
        system_tokens = len(request.system_prompt or '') // 4
        total_input_tokens = prompt_tokens + system_tokens
        
        # Log the comparison
        self.logger.info(
            f"🔍 CONTEXT CONFIG - Model: {request.model}, Task: {config_task_type}, "
            f"Config context: {context_params['num_ctx']}, API context: {api_context}, "
            f"Num predict: {context_params['num_predict']}"
        )
        
        self.logger.info(
            f"🔍 PROMPT SIZE - Input tokens: ~{total_input_tokens}, "
            f"Output tokens: {context_params['num_predict']}, "
            f"Total: ~{total_input_tokens + context_params['num_predict']}"
        )
        
        # Use the actual model context size from API
        context_params['num_ctx'] = api_context
        
        # Validate that the total required context fits within model limits
        total_required = total_input_tokens + context_params['num_predict']
        if total_required > api_context:
            self.logger.warning(
                f"⚠️ CONTEXT WARNING - Total required tokens (~{total_required}) exceeds model context ({api_context}). "
                f"Prompt may be truncated by Ollama."
            )
            
            # Adjust num_predict to fit within context if possible
            max_output = api_context - total_input_tokens - 100  # Leave small buffer
            if max_output > 0:
                context_params['num_predict'] = min(context_params['num_predict'], max_output)
                self.logger.info(f"🔧 CONTEXT ADJUSTMENT - Reduced num_predict to {context_params['num_predict']} to fit context")
            else:
                self.logger.error(f"❌ CONTEXT ERROR - Input prompt (~{total_input_tokens} tokens) exceeds model context ({api_context})")
        
        return context_params
    
    def handle(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Execute LLM request via API"""
        start_time = time.time()
        
        try:
            # Calculate dynamic context size
            context_params = self.calculate_dynamic_context_size(request)
            
            # Prepare API request
            api_request = {
                'model': request.model,
                'prompt': request.prompt,
                'stream': False,
                'options': {
                    'temperature': request.temperature,
                    'top_p': request.top_p,
                    'num_ctx': request.num_ctx or context_params['num_ctx'],  # Use dynamic context if not specified
                    'num_predict': request.num_predict or context_params['num_predict']  # Use optimized prediction size
                }
            }
            
            if request.system_prompt:
                api_request['system'] = request.system_prompt
            
            # Log comprehensive request details being sent to Ollama
            import json
            symbol = request.metadata.get('symbol', 'UNKNOWN') if request.metadata else 'UNKNOWN'
            task_type = request.metadata.get('task_type', 'unknown') if request.metadata else 'unknown'
            prompt_length = len(api_request.get('prompt', ''))
            system_length = len(api_request.get('system', ''))
            
            execute_msg = (f"🚀 LLM EXECUTE - {symbol} {task_type} | "
                           f"Model: {request.model} | "
                           f"Prompt: {prompt_length} chars | "
                           f"System: {system_length} chars | "
                           f"Total input: {prompt_length + system_length} chars | "
                           f"Request: {request.request_id[:8]}")
            self._log_to_both(symbol, execute_msg)
            
            context_msg = (f"🚀 LLM CONTEXT - {symbol} | "
                           f"Context size: {api_request['options']['num_ctx']} | "
                           f"Max output: {api_request['options']['num_predict']} | "
                           f"Temperature: {api_request['options']['temperature']} | "
                           f"Top-p: {api_request['options']['top_p']}")
            self._log_to_both(symbol, context_msg)
            
            self.logger.debug(f"🔍 OLLAMA API DEBUG - Request options: {json.dumps(api_request['options'], indent=2)}")
            
            # Execute request
            response_data = self.api_client.post_json('/api/generate', json=api_request, timeout=request.timeout)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Extract token and timing information from Ollama response
            prompt_eval_count = response_data.get('prompt_eval_count', 0)
            eval_count = response_data.get('eval_count', 0)
            total_tokens = prompt_eval_count + eval_count
            response_content = response_data.get('response', '')
            response_length = len(response_content)
            
            # Calculate timing details
            prompt_eval_duration = response_data.get('prompt_eval_duration', 0) / 1_000_000  # Convert to ms
            eval_duration = response_data.get('eval_duration', 0) / 1_000_000  # Convert to ms
            total_duration = response_data.get('total_duration', 0) / 1_000_000  # Convert to ms
            load_duration = response_data.get('load_duration', 0) / 1_000_000  # Convert to ms
            
            # Log comprehensive response details
            complete_msg = (f"✅ LLM COMPLETE - {symbol} {task_type} | "
                           f"Processing: {processing_time}ms | "
                           f"Input tokens: {prompt_eval_count} | "
                           f"Output tokens: {eval_count} | "
                           f"Total tokens: {total_tokens} | "
                           f"Response: {response_length} chars")
            self._log_to_both(symbol, complete_msg)
            
            timing_msg = (f"✅ LLM TIMING - {symbol} | "
                         f"Load: {load_duration:.1f}ms | "
                         f"Prompt eval: {prompt_eval_duration:.1f}ms | "
                         f"Generation: {eval_duration:.1f}ms | "
                         f"Total: {total_duration:.1f}ms")
            self._log_to_both(symbol, timing_msg)
            
            # Calculate and log efficiency metrics
            if eval_count > 0 and eval_duration > 0:
                tokens_per_second = eval_count / (eval_duration / 1000)
                efficiency_msg = (f"✅ LLM EFFICIENCY - {symbol} | "
                                f"Generation speed: {tokens_per_second:.1f} tokens/sec | "
                                f"Context utilization: {total_tokens}/{api_request['options']['num_ctx']} ({(total_tokens/api_request['options']['num_ctx']*100):.1f}%)")
                self._log_to_both(symbol, efficiency_msg)
            
            # Analyze response structure for JSON responses
            response_keys = []
            try:
                if response_content.strip().startswith('{'):
                    import json
                    parsed_response = json.loads(response_content)
                    response_keys = list(parsed_response.keys()) if isinstance(parsed_response, dict) else []
                    json_keys_msg = (f"✅ LLM JSON KEYS - {symbol} | "
                                   f"Response keys: {response_keys[:10]} | "  # Limit to first 10 keys
                                   f"Total keys: {len(response_keys)}")
                    self._log_to_both(symbol, json_keys_msg)
            except:
                # Not JSON or parsing failed - that's fine
                pass
            
            self.logger.debug(f"🔍 OLLAMA API DEBUG - Full response keys: {list(response_data.keys()) if response_data else 'None'}")
            self.logger.debug(f"🔍 OLLAMA API DEBUG - Response preview: '{response_content[:200]}...' " if len(response_content) > 200 else f"'{response_content}'")
            
            # Add token details to metadata
            enhanced_metadata = request.metadata or {}
            enhanced_metadata['tokens'] = {
                'input': prompt_eval_count,
                'output': eval_count,
                'total': total_tokens
            }
            enhanced_metadata['timings'] = {
                'prompt_eval_duration': response_data.get('prompt_eval_duration', 0),
                'eval_duration': response_data.get('eval_duration', 0),
                'total_duration': response_data.get('total_duration', 0),
                'load_duration': response_data.get('load_duration', 0)
            }
            
            # Create response object
            response = LLMResponse(
                content=response_data.get('response', ''),
                model=request.model,
                processing_time_ms=processing_time,
                tokens_used=total_tokens,
                metadata=enhanced_metadata,
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
        """Delegate cache storage to cache handler"""
        if self.cache_handler:
            self.cache_handler.store_response(request, response)

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
        cache_handler = LLMCacheHandler(cache_manager, cache_strategy, self.config) if cache_manager else None
        validation_handler = LLMValidationHandler()
        execution_handler = LLMExecutionHandler(self.config, cache_handler)
        
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
