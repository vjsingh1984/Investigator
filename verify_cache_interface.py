#!/usr/bin/env python3
"""
Verification script to test cache manager interface for submission data
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from utils.cache.cache_manager import get_cache_manager
from utils.cache.cache_types import CacheType

def test_submission_cache_interface():
    """Test that submission data can be accessed via cache manager interface"""
    
    print("🧪 Testing Cache Manager Interface for Submission Data")
    print("=" * 60)
    
    # Initialize cache manager
    config = get_config()
    cache_manager = get_cache_manager()
    
    # Test cache key structure for submission data
    test_symbol = "AAPL"
    test_cik = "0000320193"  # Apple's CIK
    
    cache_key = {
        'symbol': test_symbol,
        'cik': test_cik
    }
    
    print(f"📋 Testing cache key: {cache_key}")
    
    # Test exists method
    print("\\n1️⃣ Testing cache_manager.exists()...")
    try:
        exists = cache_manager.exists(CacheType.SUBMISSION_DATA, cache_key)
        print(f"   ✅ cache_manager.exists() returned: {exists}")
    except Exception as e:
        print(f"   ❌ cache_manager.exists() failed: {e}")
        return False
    
    # Test get method
    print("\\n2️⃣ Testing cache_manager.get()...")
    try:
        result = cache_manager.get(CacheType.SUBMISSION_DATA, cache_key)
        if result:
            print(f"   ✅ cache_manager.get() returned data with keys: {list(result.keys())}")
            
            # Check expected fields
            expected_fields = ['symbol', 'cik', 'submissions_data']
            for field in expected_fields:
                if field in result:
                    print(f"      ✅ Found expected field: {field}")
                else:
                    print(f"      ⚠️  Missing expected field: {field}")
                    
            # Check if submissions_data has filings
            submissions_data = result.get('submissions_data', {})
            if isinstance(submissions_data, dict):
                filings = submissions_data.get('filings', {})
                recent = filings.get('recent', {})
                if recent and 'form' in recent:
                    form_count = len(recent['form'])
                    print(f"      ✅ Found {form_count} filings in submissions_data")
                else:
                    print(f"      ⚠️  No filings found in submissions_data")
            else:
                print(f"      ⚠️  submissions_data is not a dict: {type(submissions_data)}")
                
        else:
            print(f"   ℹ️  cache_manager.get() returned None (no cached data)")
            
    except Exception as e:
        print(f"   ❌ cache_manager.get() failed: {e}")
        return False
    
    # Test cache hierarchy (disk -> RDBMS)
    print("\\n3️⃣ Testing cache hierarchy...")
    try:
        handlers = cache_manager.handlers.get(CacheType.SUBMISSION_DATA, [])
        if handlers:
            print(f"   ✅ Found {len(handlers)} handlers for SUBMISSION_DATA")
            for i, handler in enumerate(handlers):
                handler_name = handler.__class__.__name__
                priority = handler.priority
                print(f"      {i+1}. {handler_name} (priority: {priority})")
        else:
            print(f"   ⚠️  No handlers found for SUBMISSION_DATA")
            
    except Exception as e:
        print(f"   ❌ Cache hierarchy check failed: {e}")
        return False
    
    print("\\n✅ Cache manager interface verification completed!")
    return True

def test_submission_processor_integration():
    """Test submission processor integration with cache manager"""
    
    print("\\n🔗 Testing Submission Processor Integration")
    print("=" * 60)
    
    from utils.sec_quarterly_processor import SECQuarterlyProcessor
    
    try:
        processor = SECQuarterlyProcessor()
        test_symbol = "AAPL"
        test_cik = "0000320193"
        
        print(f"📋 Testing with {test_symbol} (CIK: {test_cik})")
        
        # Test _check_submissions_store
        print("\\n1️⃣ Testing _check_submissions_store()...")
        exists = processor._check_submissions_store(test_symbol, test_cik)
        print(f"   ✅ _check_submissions_store() returned: {exists}")
        
        if exists:
            # Test _extract_recent_periods
            print("\\n2️⃣ Testing _extract_recent_periods()...")
            quarterly_data = processor._extract_recent_periods(test_symbol, test_cik, max_periods=4)
            print(f"   ✅ _extract_recent_periods() returned {len(quarterly_data)} periods")
            
            for i, qd in enumerate(quarterly_data[:2]):  # Show first 2
                print(f"      {i+1}. {qd.fiscal_year}-{qd.fiscal_period} ({qd.form_type}) - {qd.filing_date}")
        else:
            print("   ℹ️  No submission data found, skipping _extract_recent_periods test")
            
    except Exception as e:
        print(f"   ❌ Submission processor integration test failed: {e}")
        return False
        
    print("\\n✅ Submission processor integration test completed!")
    return True

if __name__ == "__main__":
    print("🚀 InvestiGator Cache Interface Verification")
    print("=" * 70)
    
    success = True
    
    # Test cache manager interface
    if not test_submission_cache_interface():
        success = False
    
    # Test processor integration
    if not test_submission_processor_integration():
        success = False
    
    print("\\n" + "=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED! Cache manager interface is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Check the output above for details.")
    
    sys.exit(0 if success else 1)