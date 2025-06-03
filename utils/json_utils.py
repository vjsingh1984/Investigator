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


