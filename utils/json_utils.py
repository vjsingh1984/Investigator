#!/usr/bin/env python3
"""
InvestiGator - JSON Utilities
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

JSON Utilities - Centralized JSON handling functions
Eliminates duplicate safe JSON encoding/decoding across the codebase
"""

import json
from typing import Any


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely encode object to JSON with UTF-8 encoding, handling binary characters"""
    return json.dumps(obj, ensure_ascii=False, **kwargs)


def safe_json_loads(json_str: str) -> Any:
    """Safely decode JSON string with UTF-8 encoding"""
    if isinstance(json_str, bytes):
        json_str = json_str.decode('utf-8', errors='replace')
    return json.loads(json_str)


def extract_json_from_text(text: str) -> dict:
    """
    Extract JSON from text that may contain additional content before/after JSON.
    Uses a more robust approach than simple string searching.
    """
    import re
    
    # First, try to parse the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON between code blocks
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Find the first { and use a stack to find the matching }
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("No JSON object found in text")
    
    # Use a simple parser to find matching braces
    stack = []
    in_string = False
    escape_next = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                stack.append(char)
            elif char == '}':
                stack.pop()
                if not stack:
                    # Found matching closing brace
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        # Continue searching for next valid JSON
                        pass
    
    raise ValueError("No valid JSON object found in text")


