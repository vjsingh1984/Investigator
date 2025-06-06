#!/usr/bin/env python3
"""
InvestiGator - Prompt Optimizer
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Smart prompt optimization for LLM context window management
Automatically truncates and prioritizes data based on model capabilities
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class PromptOptimizationConfig:
    """Configuration for prompt optimization"""
    max_context_size: int = 4096
    reserved_output_tokens: int = 1024
    min_prompt_tokens: int = 512
    truncation_strategy: str = "intelligent"  # "intelligent", "tail", "head"
    preserve_sections: List[str] = None  # Sections to always preserve
    
    def __post_init__(self):
        if self.preserve_sections is None:
            self.preserve_sections = ["current_price", "recommendation", "score"]

class PromptOptimizer:
    """Optimizes prompts for LLM context window constraints"""
    
    def __init__(self, config: PromptOptimizationConfig = None):
        self.config = config or PromptOptimizationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def optimize_technical_analysis_prompt(self, prompt_data: Dict[str, Any], 
                                         max_context: int) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize technical analysis prompt for model context window
        
        Args:
            prompt_data: Dict containing all prompt components
            max_context: Maximum context size for the model
        
        Returns:
            Tuple of (optimized_prompt, optimization_metadata)
        """
        available_tokens = max_context - self.config.reserved_output_tokens
        
        # Extract key components
        system_prompt = prompt_data.get('system_prompt', '')
        base_prompt = prompt_data.get('base_prompt', '')
        technical_data = prompt_data.get('technical_data', {})
        
        # Estimate current sizes
        system_tokens = self.estimate_tokens(system_prompt)
        base_tokens = self.estimate_tokens(base_prompt)
        
        # Calculate remaining space for technical data
        remaining_tokens = available_tokens - system_tokens - base_tokens
        
        if remaining_tokens < self.config.min_prompt_tokens:
            self.logger.warning(f"Very limited context space: {remaining_tokens} tokens remaining")
        
        # Optimize technical data
        optimized_data, data_metadata = self._optimize_technical_data(
            technical_data, remaining_tokens
        )
        
        # Reconstruct prompt
        optimized_prompt = self._reconstruct_prompt(
            system_prompt, base_prompt, optimized_data
        )
        
        optimization_metadata = {
            'original_tokens': self.estimate_tokens(str(prompt_data)),
            'optimized_tokens': self.estimate_tokens(optimized_prompt),
            'context_utilization': self.estimate_tokens(optimized_prompt) / max_context,
            'data_optimization': data_metadata,
            'truncation_applied': data_metadata.get('truncated', False)
        }
        
        self.logger.info(f"🔍 PROMPT OPTIMIZATION - Original: {optimization_metadata['original_tokens']} tokens, "
                        f"Optimized: {optimization_metadata['optimized_tokens']} tokens, "
                        f"Utilization: {optimization_metadata['context_utilization']:.1%}")
        
        return optimized_prompt, optimization_metadata
    
    def _optimize_technical_data(self, technical_data: Dict[str, Any], 
                                max_tokens: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimize technical data to fit within token limit"""
        optimized_data = {}
        metadata = {'truncated': False, 'sections_truncated': [], 'original_sections': 0}
        
        if not technical_data:
            return optimized_data, metadata
        
        # Priority order for technical analysis sections
        priority_sections = [
            'current_price',
            'price_change_1d',
            'volume',
            'rsi',
            'macd',
            'sma_20',
            'sma_50',
            'bollinger_bands',
            'recent_highs_lows',
            'support_resistance',
            'historical_data'  # Lowest priority - can be heavily truncated
        ]
        
        metadata['original_sections'] = len(technical_data)
        used_tokens = 0
        
        # First pass: Add high-priority sections
        for section in priority_sections:
            if section in technical_data and used_tokens < max_tokens:
                section_data = technical_data[section]
                section_text = json.dumps(section_data) if isinstance(section_data, (dict, list)) else str(section_data)
                section_tokens = self.estimate_tokens(section_text)
                
                if used_tokens + section_tokens <= max_tokens:
                    optimized_data[section] = section_data
                    used_tokens += section_tokens
                else:
                    # Try to truncate this section
                    truncated_data = self._truncate_section_data(section_data, max_tokens - used_tokens)
                    if truncated_data:
                        optimized_data[section] = truncated_data
                        used_tokens += self.estimate_tokens(str(truncated_data))
                        metadata['sections_truncated'].append(section)
                        metadata['truncated'] = True
                    break
        
        # Second pass: Add remaining sections if space allows
        for section, data in technical_data.items():
            if section not in optimized_data and used_tokens < max_tokens:
                section_text = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
                section_tokens = self.estimate_tokens(section_text)
                
                if used_tokens + section_tokens <= max_tokens:
                    optimized_data[section] = data
                    used_tokens += section_tokens
                else:
                    metadata['truncated'] = True
                    break
        
        return optimized_data, metadata
    
    def _truncate_section_data(self, data: Any, max_tokens: int) -> Any:
        """Intelligently truncate section data"""
        if max_tokens < 50:  # Not enough space for meaningful data
            return None
        
        if isinstance(data, dict):
            # For dict data, keep the most important keys
            important_keys = ['value', 'current', 'signal', 'trend', 'score']
            truncated = {}
            used_tokens = 0
            
            # Add important keys first
            for key in important_keys:
                if key in data and used_tokens < max_tokens:
                    value = data[key]
                    value_tokens = self.estimate_tokens(str(value))
                    if used_tokens + value_tokens <= max_tokens:
                        truncated[key] = value
                        used_tokens += value_tokens
            
            # Add other keys if space allows
            for key, value in data.items():
                if key not in truncated and used_tokens < max_tokens:
                    value_tokens = self.estimate_tokens(str(value))
                    if used_tokens + value_tokens <= max_tokens:
                        truncated[key] = value
                        used_tokens += value_tokens
            
            return truncated if truncated else None
        
        elif isinstance(data, list):
            # For list data, keep recent entries
            if not data:
                return None
            
            # Start with most recent entries (assume list is chronological)
            truncated = []
            used_tokens = 0
            
            for item in reversed(data):
                item_tokens = self.estimate_tokens(str(item))
                if used_tokens + item_tokens <= max_tokens:
                    truncated.insert(0, item)  # Maintain original order
                    used_tokens += item_tokens
                else:
                    break
            
            return truncated if truncated else [data[-1]]  # At least keep the most recent
        
        else:
            # For scalar data, truncate string representation if too long
            text = str(data)
            if self.estimate_tokens(text) <= max_tokens:
                return data
            else:
                # Truncate to fit
                max_chars = max_tokens * 4  # Rough conversion back to chars
                return text[:max_chars] + "..." if len(text) > max_chars else text
    
    def _reconstruct_prompt(self, system_prompt: str, base_prompt: str, 
                           optimized_data: Dict[str, Any]) -> str:
        """Reconstruct the optimized prompt"""
        data_section = json.dumps(optimized_data, indent=2) if optimized_data else "{}"
        
        # Combine all parts
        full_prompt = f"{base_prompt}\n\nTechnical Data:\n{data_section}"
        
        return full_prompt.strip()

def get_prompt_optimizer() -> PromptOptimizer:
    """Get global prompt optimizer instance"""
    return PromptOptimizer()