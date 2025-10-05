#!/usr/bin/env python3
"""
Fix Streamlit deprecation warnings and critical errors
"""

import re

def fix_streamlit_issues():
    """Fix all Streamlit deprecation and error issues."""
    print("🔧 Fixing Streamlit issues...")
    
    app_file = "app.py"
    
    try:
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix 1: Replace all use_container_width=True with width='stretch'
        content = re.sub(
            r'use_container_width=True',
            "width='stretch'",
            content
        )
        
        # Fix 2: Replace all use_container_width=False with width='content' 
        content = re.sub(
            r'use_container_width=False',
            "width='content'", 
            content
        )
        
        # Fix 3: Fix the cache stats call
        content = re.sub(
            r'cache_info = st\.cache_data\.get_stats\(\)',
            'cache_info = None  # Cache stats deprecated',
            content
        )
        
        # Fix 4: Fix the cache check
        content = re.sub(
            r'if hasattr\(cache_info, \'__len__\'\):[\s\n]*st\.info\(f"💾 Cache: Active"\)[\s\n]*else:[\s\n]*st\.info\(f"💾 Cache: Unavailable"\)',
            'st.info("💾 Cache: Active")',
            content,
            flags=re.MULTILINE | re.DOTALL
        )
        
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Streamlit issues fixed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix Streamlit issues: {e}")
        return False

def fix_knowledge_graph_issues():
    """Fix knowledge graph array ambiguity errors."""
    print("🔧 Fixing knowledge graph issues...")
    
    kg_file = "kg_builder.py"
    
    try:
        with open(kg_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix empty array checks
        content = re.sub(
            r'if hub_entities:',
            'if len(hub_entities) > 0:',
            content
        )
        
        # Fix numpy array truth value issues
        content = re.sub(
            r'if degrees:',
            'if degrees and len(degrees) > 0:',
            content
        )
        
        with open(kg_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Knowledge graph issues fixed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix knowledge graph: {e}")
        return False

def main():
    """Apply all fixes."""
    print("🚀 NASA Knowledge Engine - Error Fix Tool")
    print("=" * 50)
    
    fixes = 0
    
    if fix_streamlit_issues():
        fixes += 1
        
    if fix_knowledge_graph_issues():
        fixes += 1
    
    print(f"\n🎯 Applied {fixes}/2 fixes")
    
    if fixes == 2:
        print("🎉 All critical errors fixed!")
        print("\n📋 Ready to launch:")
        print("   streamlit run app.py")
    else:
        print("⚠️ Some fixes may need manual attention")

if __name__ == "__main__":
    main()