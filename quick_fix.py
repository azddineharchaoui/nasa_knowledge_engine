#!/usr/bin/env python3
"""
Quick fix for NASA Knowledge Engine hanging issues.
This script addresses:
1. Summarization parameter conflicts (max_length vs max_new_tokens)
2. API timeout issues causing app to hang
3. Provides development mode with immediate sample data
"""

import os
import sys
import re

def fix_summarizer_config():
    """Fix summarizer configuration conflicts."""
    print("🔧 Fixing summarizer configuration...")
    
    summarizer_file = "summarizer.py"
    
    try:
        with open(summarizer_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix 1: Remove max_length when max_new_tokens is used
        content = re.sub(
            r'max_length=max_len,\s*min_length=min_len,',
            'max_new_tokens=max_new_tokens, min_length=min_len,',
            content
        )
        
        # Fix 2: Update variable names
        content = re.sub(
            r'max_len = max\(20, input_length - 5\)',
            'max_new_tokens = max(15, input_length // 2)',
            content
        )
        
        content = re.sub(
            r'max_len = max\(50, input_length // 2\)',
            'max_new_tokens = max(30, input_length // 3)',
            content
        )
        
        content = re.sub(
            r'max_len = min\(120, max\(60, input_length // 3\)\)',
            'max_new_tokens = min(80, max(40, input_length // 4))',
            content
        )
        
        # Fix 3: Update log message
        content = re.sub(
            r'max_len=\{max_len\}',
            'max_new_tokens={max_new_tokens}',
            content
        )
        
        with open(summarizer_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Summarizer configuration fixed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix summarizer: {e}")
        return False

def fix_api_timeouts():
    """Reduce API timeouts to prevent hanging."""
    print("🔧 Fixing API timeout issues...")
    
    data_fetch_file = "data_fetch.py"
    
    try:
        with open(data_fetch_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix 1: Reduce GeneLab API timeout
        content = re.sub(
            r'timeout=45\)',
            'timeout=10)',
            content
        )
        
        # Fix 2: Reduce NTRS timeout
        content = re.sub(
            r'timeout = 60  # Increased from 30s to 60s',
            'timeout = 15  # Reduced to prevent hanging',
            content
        )
        
        # Fix 3: Reduce retry attempts
        content = re.sub(
            r'max_retries = 3',
            'max_retries = 2',
            content
        )
        
        with open(data_fetch_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ API timeouts reduced")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix API timeouts: {e}")
        return False

def create_development_mode():
    """Create development mode configuration."""
    print("🔧 Creating development mode...")
    
    app_file = "app.py"
    
    try:
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add development mode toggle at the top
        dev_config = '''
# DEVELOPMENT MODE CONFIGURATION
DEVELOPMENT_MODE = True  # Set to False for production
QUICK_LOAD = True        # Use sample data first to avoid API delays
'''
        
        # Insert after imports
        import_end = content.find('from utils import')
        if import_end != -1:
            next_line = content.find('\n', import_end) + 1
            content = content[:next_line] + dev_config + content[next_line:]
        
        # Modify data loading to use development mode
        old_fetch = 'publications = fetch_publications(query, limit)'
        new_fetch = '''publications = fetch_publications(
            query, 
            limit, 
            use_sample_first=DEVELOPMENT_MODE and QUICK_LOAD
        )'''
        
        content = content.replace(old_fetch, new_fetch)
        
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Development mode configured")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create development mode: {e}")
        return False

def add_fallback_function():
    """Add immediate fallback for stuck loading."""
    print("🔧 Adding immediate fallback function...")
    
    data_fetch_file = "data_fetch.py"
    
    try:
        with open(data_fetch_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add use_sample_first parameter to function signature
        old_signature = 'def fetch_publications(query: str, limit: int = 10) -> List[Dict]:'
        new_signature = 'def fetch_publications(query: str, limit: int = 10, use_sample_first: bool = False) -> List[Dict]:'
        
        content = content.replace(old_signature, new_signature)
        
        # Add immediate sample data fallback at the beginning of the function
        fallback_code = '''    
    # Quick development mode - use sample data first
    if use_sample_first:
        try:
            log("🚀 Development mode: Using sample data first...")
            sample_publications = load_sample_publications()
            if sample_publications:
                filtered_samples = _filter_sample_data(sample_publications, query)
                log(f"✅ Sample data loaded: {len(filtered_samples)} publications")
                return filtered_samples[:limit]
        except Exception as e:
            log_error(f"Sample data failed: {str(e)}")
            # Continue to API methods
    
'''
        
        # Insert after the docstring
        docstring_end = content.find('"""', content.find('"""') + 3) + 3
        next_line = content.find('\n', docstring_end) + 1
        content = content[:next_line] + fallback_code + content[next_line:]
        
        with open(data_fetch_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Immediate fallback function added")
        return True
        
    except Exception as e:
        print(f"❌ Failed to add fallback function: {e}")
        return False

def main():
    """Apply all quick fixes."""
    print("🚀 NASA Knowledge Engine - Quick Fix Tool")
    print("=" * 50)
    
    fixes_applied = 0
    total_fixes = 4
    
    if fix_summarizer_config():
        fixes_applied += 1
    
    if fix_api_timeouts():
        fixes_applied += 1
        
    if add_fallback_function():
        fixes_applied += 1
        
    if create_development_mode():
        fixes_applied += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Applied {fixes_applied}/{total_fixes} fixes")
    
    if fixes_applied == total_fixes:
        print("🎉 All fixes applied successfully!")
        print("\n📋 Next Steps:")
        print("1. Run: streamlit run nasa_knowledge_engine/app.py")
        print("2. App should load much faster with sample data")
        print("3. To use live APIs, set DEVELOPMENT_MODE = False")
        print("4. Check browser console for any remaining errors")
    else:
        print("⚠️ Some fixes failed - manual intervention may be needed")
    
    return fixes_applied == total_fixes

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)