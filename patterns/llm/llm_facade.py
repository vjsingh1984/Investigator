#!/usr/bin/env python3
"""
InvestiGator - LLM Facade Pattern Implementation
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

LLM Facade - Simplified interface for pattern-based LLM operations
Provides a clean API for all LLM processing needs
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .llm_interfaces import (
    LLMRequest, LLMResponse, LLMTaskType, LLMPriority,
    ILLMStrategy, ILLMProcessor, ILLMObserver, ILLMAnalysisTemplate
)
from .llm_strategies import (
    ComprehensiveLLMStrategy, QuickLLMStrategy,
    LLMCacheStrategy
)
from .llm_processors import QueuedLLMProcessor, StandardLLMAnalysisTemplate

logger = logging.getLogger(__name__)

class LLMAnalysisObserver(ILLMObserver):
    """Default observer for LLM processing events"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def on_request_queued(self, request: LLMRequest) -> None:
        """Called when request is added to queue"""
        self.logger.debug(f"LLM request queued: {request.request_id} for {request.metadata.get('symbol', 'UNKNOWN') if request.metadata else 'UNKNOWN'}")
    
    def on_processing_started(self, request: LLMRequest) -> None:
        """Called when processing starts"""
        symbol = request.metadata.get('symbol', 'UNKNOWN') if request.metadata else 'UNKNOWN'
        task_type = request.metadata.get('task_type', 'unknown') if request.metadata else 'unknown'
        self.logger.info(f"Starting {task_type} analysis for {symbol}")
    
    def on_processing_completed(self, request: LLMRequest, response: LLMResponse) -> None:
        """Called when processing completes"""
        symbol = request.metadata.get('symbol', 'UNKNOWN') if request.metadata else 'UNKNOWN'
        task_type = request.metadata.get('task_type', 'unknown') if request.metadata else 'unknown'
        self.logger.info(f"Completed {task_type} analysis for {symbol} in {response.processing_time_ms}ms")
    
    def on_processing_error(self, request: LLMRequest, error: Exception) -> None:
        """Called when processing fails"""
        symbol = request.metadata.get('symbol', 'UNKNOWN') if request.metadata else 'UNKNOWN'
        task_type = request.metadata.get('task_type', 'unknown') if request.metadata else 'unknown'
        self.logger.error(f"Failed {task_type} analysis for {symbol}: {error}")

class LLMFacade:
    """
    Facade for simplified LLM operations using design patterns
    Replaces the functionality of ollama_interface.py
    """
    
    def __init__(self, config, cache_manager=None, strategy_type: str = "comprehensive"):
        """
        Initialize LLM facade with configuration
        
        Args:
            config: Configuration object
            cache_manager: Optional cache manager for response caching
            strategy_type: Type of LLM strategy ("comprehensive" or "quick")
        """
        self.config = config
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Create strategy and cache strategy
        self.strategy = self._create_strategy(strategy_type)
        self.cache_strategy = LLMCacheStrategy(config) if cache_manager else None
        
        # Create processor with observer
        self.processor = QueuedLLMProcessor(
            config=config,
            num_threads=config.ollama.num_llm_threads,  # Use configured thread count
            cache_manager=cache_manager,
            cache_strategy=self.cache_strategy
        )
        
        # Add observer for logging
        self.observer = LLMAnalysisObserver()
        self.processor.attach(self.observer)
        
        # Create analysis template
        self.analysis_template = StandardLLMAnalysisTemplate(self.processor, self.strategy)
        
        self.logger.info(f"LLM Facade initialized with {strategy_type} strategy")
    
    def _create_strategy(self, strategy_type: str) -> ILLMStrategy:
        """Create appropriate strategy based on type"""
        if strategy_type == "comprehensive":
            return ComprehensiveLLMStrategy(self.config)
        elif strategy_type == "quick":
            return QuickLLMStrategy(self.config)
        else:
            self.logger.warning(f"Unknown strategy type '{strategy_type}', using comprehensive")
            return ComprehensiveLLMStrategy(self.config)
    
    # ============================================================================
    # High-Level Analysis Methods (Template Method Pattern)
    # ============================================================================
    
    def analyze_fundamental(self, symbol: str, quarterly_data: List[Dict], filing_data: Dict = None) -> Dict[str, Any]:
        """
        Perform fundamental analysis using template method pattern
        
        Args:
            symbol: Stock symbol
            quarterly_data: List of quarterly financial data
            filing_data: Optional filing information
            
        Returns:
            Structured analysis result
        """
        try:
            data = {
                'symbol': symbol,
                'quarterly_data': quarterly_data,
                'filing_data': filing_data or {}
            }
            
            return self.analysis_template.analyze(symbol, data, LLMTaskType.FUNDAMENTAL_ANALYSIS)
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'task_type': 'fundamental_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def analyze_technical(self, symbol: str, price_data: Dict, indicators: Dict = None) -> Dict[str, Any]:
        """
        Perform technical analysis using template method pattern
        
        Args:
            symbol: Stock symbol
            price_data: Price and volume data
            indicators: Technical indicators
            
        Returns:
            Structured analysis result
        """
        try:
            data = {
                'symbol': symbol,
                'price_data': price_data,
                'indicators': indicators or {}
            }
            
            return self.analysis_template.analyze(symbol, data, LLMTaskType.TECHNICAL_ANALYSIS)
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'task_type': 'technical_analysis',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def synthesize_analysis(self, symbol: str, fundamental_result: Dict, technical_result: Dict) -> Dict[str, Any]:
        """
        Synthesize fundamental and technical analysis
        
        Args:
            symbol: Stock symbol
            fundamental_result: Fundamental analysis result
            technical_result: Technical analysis result
            
        Returns:
            Synthesized analysis result
        """
        try:
            data = {
                'symbol': symbol,
                'fundamental_analysis': fundamental_result,
                'technical_analysis': technical_result
            }
            
            return self.analysis_template.analyze(symbol, data, LLMTaskType.SYNTHESIS)
            
        except Exception as e:
            self.logger.error(f"Error in synthesis: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'task_type': 'synthesis',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def analyze_quarterly_summary(self, symbol: str, quarter_data: Dict) -> Dict[str, Any]:
        """
        Create quarterly performance summary
        
        Args:
            symbol: Stock symbol
            quarter_data: Single quarter financial data
            
        Returns:
            Quarterly summary result
        """
        try:
            data = {
                'symbol': symbol,
                'quarter_data': quarter_data
            }
            
            return self.analysis_template.analyze(symbol, data, LLMTaskType.QUARTERLY_SUMMARY)
            
        except Exception as e:
            self.logger.error(f"Error in quarterly summary: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'task_type': 'quarterly_summary',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def assess_risks(self, symbol: str, all_data: Dict) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment
        
        Args:
            symbol: Stock symbol
            all_data: Combined analysis data
            
        Returns:
            Risk assessment result
        """
        try:
            data = dict(all_data)
            data['symbol'] = symbol
            
            return self.analysis_template.analyze(symbol, data, LLMTaskType.RISK_ASSESSMENT)
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'task_type': 'risk_assessment',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    # ============================================================================
    # Direct LLM Methods (Strategy Pattern)
    # ============================================================================
    
    def generate_response(self, task_type: LLMTaskType, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate LLM response using strategy pattern
        
        Args:
            task_type: Type of analysis task
            data: Input data for analysis
            
        Returns:
            Processed LLM response
        """
        try:
            # Prepare request using strategy
            request = self.strategy.prepare_request(task_type, data)
            
            # Process request through processor
            response = self.processor.process_request(request)
            
            # Process response using strategy
            return self.strategy.process_response(response, task_type)
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                'error': str(e),
                'task_type': task_type.value if isinstance(task_type, LLMTaskType) else str(task_type),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def generate_batch_responses(self, requests: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate multiple LLM responses in batch
        
        Args:
            requests: List of request dictionaries with 'task_type' and 'data' keys
            
        Returns:
            List of processed responses
        """
        try:
            llm_requests = []
            
            for req in requests:
                task_type = LLMTaskType(req['task_type'])
                data = req['data']
                llm_request = self.strategy.prepare_request(task_type, data)
                llm_requests.append((llm_request, task_type))
            
            # Process batch
            llm_responses = self.processor.process_batch([req for req, _ in llm_requests])
            
            # Process responses
            results = []
            for i, response in enumerate(llm_responses):
                _, task_type = llm_requests[i]
                result = self.strategy.process_response(response, task_type)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            return [{'error': str(e), 'timestamp': datetime.utcnow().isoformat()}] * len(requests)
    
    # ============================================================================
    # Legacy Compatibility Methods
    # ============================================================================
    
    def query_ollama(self, model: str, prompt: str, system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        Legacy compatibility method for direct Ollama queries
        Maintains backward compatibility with existing code
        
        Args:
            model: Model name
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        try:
            # Create direct LLM request
            request = LLMRequest(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=kwargs.get('temperature', 0.3),
                top_p=kwargs.get('top_p', 0.9),
                num_ctx=kwargs.get('num_ctx'),
                num_predict=kwargs.get('num_predict'),
                timeout=kwargs.get('timeout'),
                metadata=kwargs.get('metadata', {}),
                priority=kwargs.get('priority', LLMPriority.NORMAL.value),
                request_id=kwargs.get('request_id')
            )
            
            # Process request
            response = self.processor.process_request(request)
            
            # Return in legacy format
            return {
                'response': response.content,
                'model': response.model,
                'processing_time_ms': response.processing_time_ms,
                'tokens_used': response.tokens_used,
                'error': response.error,
                'metadata': response.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error in legacy query: {e}")
            return {
                'response': '',
                'model': model,
                'processing_time_ms': 0,
                'tokens_used': 0,
                'error': str(e),
                'metadata': {
                    'error': str(e),
                    'model': model,
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def generate(self, model: str, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        Legacy compatibility method that returns just the response content
        Maintains backward compatibility with existing code that expects a string response
        
        Args:
            model: Model name
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Response content as string
        """
        try:
            result = self.query_ollama(model, prompt, system_prompt, **kwargs)
            return result.get('response', '')
            
        except Exception as e:
            self.logger.error(f"Error in legacy query: {e}")
            return ''
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current processing queue status"""
        return {
            'queue_size': self.processor.get_queue_size(),
            'strategy': self.strategy.get_strategy_name(),
            'cache_enabled': self.cache_manager is not None,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def shutdown(self):
        """Shutdown the LLM facade and cleanup resources"""
        try:
            self.processor.detach(self.observer)
            self.processor.stop()
            self.logger.info("LLM Facade shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()

# ============================================================================
# Factory Functions for Easy Usage
# ============================================================================

def create_llm_facade(config, cache_manager=None, strategy_type: str = "comprehensive") -> LLMFacade:
    """
    Factory function to create LLM facade
    
    Args:
        config: Configuration object
        cache_manager: Optional cache manager
        strategy_type: Strategy type ("comprehensive" or "quick")
        
    Returns:
        Configured LLM facade instance
    """
    return LLMFacade(config, cache_manager, strategy_type)

def create_quick_llm_facade(config, cache_manager=None) -> LLMFacade:
    """
    Factory function for quick analysis LLM facade
    
    Args:
        config: Configuration object
        cache_manager: Optional cache manager
        
    Returns:
        Quick analysis LLM facade
    """
    return LLMFacade(config, cache_manager, "quick")

def create_comprehensive_llm_facade(config, cache_manager=None) -> LLMFacade:
    """
    Factory function for comprehensive analysis LLM facade
    
    Args:
        config: Configuration object
        cache_manager: Optional cache manager
        
    Returns:
        Comprehensive analysis LLM facade
    """
    return LLMFacade(config, cache_manager, "comprehensive")