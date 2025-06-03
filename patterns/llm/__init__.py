#!/usr/bin/env python3
"""
InvestiGator - LLM Pattern Implementations Initialization
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

LLM Pattern Implementations
LLM processing and interaction patterns
"""

from .llm_facade import *
from .llm_interfaces import *
from .llm_strategies import *
from .llm_processors import *

__all__ = [
    # Facade and factories
    'LLMFacade', 'create_llm_facade', 'create_comprehensive_llm_facade', 'create_quick_llm_facade',
    # Interfaces
    'LLMRequest', 'LLMResponse', 'LLMTaskType', 'LLMPriority',
    'ILLMStrategy', 'ILLMProcessor', 'ILLMHandler', 'ILLMObserver',
    'ILLMSubject', 'ILLMAnalysisTemplate', 'ILLMFactory', 'ILLMCacheStrategy',
    # Strategies
    'ComprehensiveLLMStrategy', 'QuickLLMStrategy',
    'StandardLLMCacheStrategy', 'AggressiveLLMCacheStrategy',
    # Processors
    'LLMCacheHandler', 'LLMValidationHandler', 'LLMExecutionHandler',
    'QueuedLLMProcessor', 'StandardLLMAnalysisTemplate',
    # Observer
    'LLMAnalysisObserver'
]