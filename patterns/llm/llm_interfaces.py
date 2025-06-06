#!/usr/bin/env python3
"""
InvestiGator - LLM Processing Interfaces
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

LLM Processing Interfaces and Abstract Classes
Defines contracts for pattern-based LLM operations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

# ============================================================================
# LLM Data Models
# ============================================================================

@dataclass
class LLMRequest:
    """LLM request with metadata"""
    model: str
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.3
    top_p: float = 0.9
    num_ctx: Optional[int] = None
    num_predict: Optional[int] = None
    timeout: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    priority: int = 5  # 1=highest, 10=lowest
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class LLMResponse:
    """LLM response with processing info"""
    content: str
    model: str
    processing_time_ms: int
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None

class LLMTaskType(Enum):
    """Types of LLM analysis tasks"""
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    SYNTHESIS = "synthesis"
    QUARTERLY_SUMMARY = "quarterly_summary"
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    
class LLMPriority(Enum):
    """LLM request priorities"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 5
    LOW = 8
    BACKGROUND = 10

# ============================================================================
# Strategy Pattern Interfaces
# ============================================================================

class ILLMStrategy(ABC):
    """Strategy interface for different LLM analysis approaches"""
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy identifier"""
        pass
    
    @abstractmethod
    def prepare_request(self, task_type: LLMTaskType, data: Dict[str, Any]) -> LLMRequest:
        """Prepare LLM request for specific task"""
        pass
    
    @abstractmethod
    def process_response(self, response: LLMResponse, task_type: LLMTaskType) -> Dict[str, Any]:
        """Process LLM response into structured data"""
        pass
    
    @abstractmethod
    def get_model_for_task(self, task_type: LLMTaskType) -> str:
        """Get appropriate model for task type"""
        pass

class ILLMProcessor(ABC):
    """Processor interface for executing LLM requests"""
    
    @abstractmethod
    def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process a single LLM request"""
        pass
    
    @abstractmethod
    def process_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Process multiple LLM requests"""
        pass
    
    @abstractmethod
    def get_queue_size(self) -> int:
        """Get current queue size"""
        pass

# ============================================================================
# Chain of Responsibility Interfaces
# ============================================================================

class ILLMHandler(ABC):
    """Handler interface for LLM processing chain"""
    
    def __init__(self):
        self._next_handler: Optional[ILLMHandler] = None
    
    def set_next(self, handler: 'ILLMHandler') -> 'ILLMHandler':
        """Set next handler in chain"""
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Handle LLM request or pass to next handler"""
        pass
    
    def _handle_next(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Pass request to next handler if exists"""
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

# ============================================================================
# Observer Pattern Interfaces  
# ============================================================================

class ILLMObserver(ABC):
    """Observer interface for LLM processing events"""
    
    @abstractmethod
    def on_request_queued(self, request: LLMRequest) -> None:
        """Called when request is added to queue"""
        pass
    
    @abstractmethod
    def on_processing_started(self, request: LLMRequest) -> None:
        """Called when processing starts"""
        pass
    
    @abstractmethod
    def on_processing_completed(self, request: LLMRequest, response: LLMResponse) -> None:
        """Called when processing completes"""
        pass
    
    @abstractmethod
    def on_processing_error(self, request: LLMRequest, error: Exception) -> None:
        """Called when processing fails"""
        pass

class ILLMSubject(ABC):
    """Subject interface for LLM processing events"""
    
    @abstractmethod
    def attach(self, observer: ILLMObserver) -> None:
        """Attach observer"""
        pass
    
    @abstractmethod
    def detach(self, observer: ILLMObserver) -> None:
        """Detach observer"""
        pass
    
    @abstractmethod
    def notify_queued(self, request: LLMRequest) -> None:
        """Notify observers of queued request"""
        pass
    
    @abstractmethod
    def notify_started(self, request: LLMRequest) -> None:
        """Notify observers of started processing"""
        pass
    
    @abstractmethod
    def notify_completed(self, request: LLMRequest, response: LLMResponse) -> None:
        """Notify observers of completed processing"""
        pass
    
    @abstractmethod
    def notify_error(self, request: LLMRequest, error: Exception) -> None:
        """Notify observers of processing error"""
        pass

# ============================================================================
# Template Method Interfaces
# ============================================================================

class ILLMAnalysisTemplate(ABC):
    """Template method interface for standardized analysis workflows"""
    
    def analyze(self, symbol: str, data: Dict[str, Any], task_type: LLMTaskType) -> Dict[str, Any]:
        """Template method for analysis workflow"""
        # Validate input
        if not self.validate_input(symbol, data, task_type):
            return self.create_error_result("Invalid input data")
        
        # Prepare request
        request = self.prepare_analysis_request(symbol, data, task_type)
        
        # Execute analysis
        response = self.execute_analysis(request)
        
        # Process results
        return self.process_analysis_results(response, task_type)
    
    @abstractmethod
    def validate_input(self, symbol: str, data: Dict[str, Any], task_type: LLMTaskType) -> bool:
        """Validate input parameters"""
        pass
    
    @abstractmethod
    def prepare_analysis_request(self, symbol: str, data: Dict[str, Any], task_type: LLMTaskType) -> LLMRequest:
        """Prepare LLM request for analysis"""
        pass
    
    @abstractmethod
    def execute_analysis(self, request: LLMRequest) -> LLMResponse:
        """Execute the analysis request"""
        pass
    
    @abstractmethod
    def process_analysis_results(self, response: LLMResponse, task_type: LLMTaskType) -> Dict[str, Any]:
        """Process and format analysis results"""
        pass
    
    @abstractmethod
    def create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        pass

# ============================================================================
# Factory Interfaces
# ============================================================================

class ILLMFactory(ABC):
    """Factory interface for creating LLM components"""
    
    @abstractmethod
    def create_strategy(self, strategy_type: str, config: Any) -> ILLMStrategy:
        """Create LLM strategy instance"""
        pass
    
    @abstractmethod
    def create_processor(self, processor_type: str, config: Any) -> ILLMProcessor:
        """Create LLM processor instance"""
        pass
    
    @abstractmethod
    def create_handler_chain(self, config: Any) -> ILLMHandler:
        """Create handler chain for processing"""
        pass
    
    @abstractmethod
    def create_observer(self, observer_type: str, config: Any) -> ILLMObserver:
        """Create LLM observer instance"""
        pass

# ============================================================================
# Cache Integration Interfaces
# ============================================================================

class ILLMCacheStrategy(ABC):
    """Cache strategy interface for LLM responses"""
    
    @abstractmethod
    def get_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request"""
        pass
    
    @abstractmethod
    def should_cache(self, request: LLMRequest, response: LLMResponse) -> bool:
        """Determine if response should be cached"""
        pass
    
    @abstractmethod
    def get_ttl(self, task_type: LLMTaskType) -> int:
        """Get cache TTL for task type"""
        pass
    
    @abstractmethod
    def is_cacheable_task(self, task_type: LLMTaskType) -> bool:
        """Check if task type is cacheable"""
        pass