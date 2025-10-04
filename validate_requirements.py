#!/usr/bin/env python3
"""
Requirements Validation Script for NASA Space Biology Knowledge Engine

This script validates that all required dependencies are installed and working correctly.
Run this after installing requirements to ensure everything is set up properly.
"""

import sys
import importlib
import subprocess
from typing import Dict, List, Tuple

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    min_version = (3, 8)
    
    if version >= min_version:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires >= {min_version[0]}.{min_version[1]})")
        return False

def check_package_imports() -> Dict[str, bool]:
    """Check if all required packages can be imported."""
    
    # Core packages mapping (import_name -> package_name)
    core_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas', 
        'requests': 'requests',
        'bs4': 'beautifulsoup4',
        'nltk': 'nltk',
        'spacy': 'spacy',
        'networkx': 'networkx',
        'plotly': 'plotly',
        'numpy': 'numpy',
        'scipy': 'scipy'
    }
    
    # AI/ML packages
    ml_packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'sklearn': 'scikit-learn'
    }
    
    # Optional packages
    optional_packages = {
        'psutil': 'psutil',
        'pytest': 'pytest',
        'IPython': 'ipython'
    }
    
    all_packages = {**core_packages, **ml_packages, **optional_packages}
    results = {}
    
    print("🔍 Checking package imports...")
    print("-" * 40)
    
    for import_name, package_name in all_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"✅ {package_name}")
            results[package_name] = True
        except ImportError as e:
            print(f"❌ {package_name}: {e}")
            results[package_name] = False
    
    return results

def check_spacy_model() -> bool:
    """Check if spaCy language model is available."""
    print("\n🔍 Checking spaCy model...")
    print("-" * 30)
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        print(f"✅ spaCy model 'en_core_web_sm' loaded successfully")
        print(f"   Model version: {nlp.meta.get('version', 'unknown')}")
        
        # Test basic functionality
        doc = nlp("NASA space biology research")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"   Test extraction: {len(entities)} entities found")
        
        return True
        
    except OSError:
        print("❌ spaCy model 'en_core_web_sm' not found")
        print("   Install with: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"❌ spaCy model error: {e}")
        return False

def check_nltk_data() -> bool:
    """Check if NLTK data is available."""
    print("\n🔍 Checking NLTK data...")
    print("-" * 25)
    
    try:
        import nltk
        
        # Check for punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
            print("✅ NLTK punkt tokenizer available")
            
            # Test basic functionality
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize("This is a test sentence.")
            print(f"   Test tokenization: {len(tokens)} tokens")
            
            return True
            
        except LookupError:
            print("❌ NLTK punkt tokenizer not found")
            print("   Download with: python -c \"import nltk; nltk.download('punkt')\"")
            return False
            
    except ImportError:
        print("❌ NLTK not available")
        return False

def check_ai_models() -> bool:
    """Check if AI models can be loaded."""
    print("\n🤖 Checking AI model functionality...")
    print("-" * 35)
    
    try:
        from transformers import pipeline
        
        # Test with a lightweight model
        print("📥 Testing transformers pipeline...")
        classifier = pipeline(
            'sentiment-analysis', 
            model='distilbert-base-uncased-finetuned-sst-2-english',
            return_all_scores=False
        )
        
        result = classifier("This is a positive test.")
        print(f"✅ Transformers pipeline working")
        print(f"   Test result: {result[0]['label']} ({result[0]['score']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"⚠️ AI model test failed: {e}")
        print("   Models will be downloaded on first use")
        return False

def check_gpu_availability() -> Dict[str, any]:
    """Check GPU/CUDA availability for AI models."""
    print("\n🎮 Checking GPU availability...")
    print("-" * 30)
    
    gpu_info = {
        'cuda_available': False,
        'device_count': 0,
        'device_name': None,
        'recommended_device': 'cpu'
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            gpu_info['recommended_device'] = 'cuda'
            
            print(f"✅ CUDA available")
            print(f"   Device count: {gpu_info['device_count']}")
            print(f"   Primary device: {gpu_info['device_name']}")
            print(f"   Recommended: GPU acceleration")
        else:
            print("💻 CUDA not available, using CPU")
            print("   Recommended: CPU processing (slower but functional)")
            
    except ImportError:
        print("❌ PyTorch not available")
    
    return gpu_info

def check_file_permissions() -> bool:
    """Check if necessary directories can be created."""
    print("\n📁 Checking file system permissions...")
    print("-" * 40)
    
    import os
    from pathlib import Path
    
    directories = ['data', 'logs', 'cache', '__pycache__']
    success = True
    
    for dir_name in directories:
        try:
            dir_path = Path(dir_name)
            dir_path.mkdir(exist_ok=True)
            
            # Test write access
            test_file = dir_path / 'test_write.tmp'
            test_file.write_text("test")
            test_file.unlink()
            
            print(f"✅ {dir_name}/ - read/write access")
            
        except Exception as e:
            print(f"❌ {dir_name}/ - {e}")
            success = False
    
    return success

def generate_report(results: Dict) -> None:
    """Generate a summary report."""
    print("\n" + "="*60)
    print("📊 INSTALLATION VALIDATION REPORT")
    print("="*60)
    
    # Count successes
    package_results = results['packages']
    successful_packages = sum(1 for success in package_results.values() if success)
    total_packages = len(package_results)
    
    print(f"📦 Packages: {successful_packages}/{total_packages} successful")
    print(f"🔤 spaCy Model: {'✅ Ready' if results['spacy_model'] else '❌ Missing'}")
    print(f"📝 NLTK Data: {'✅ Ready' if results['nltk_data'] else '❌ Missing'}")
    print(f"🤖 AI Models: {'✅ Functional' if results['ai_models'] else '⚠️ Limited'}")
    print(f"🎮 GPU Support: {'✅ Available' if results['gpu']['cuda_available'] else '💻 CPU Only'}")
    print(f"📁 File Access: {'✅ Ready' if results['file_permissions'] else '❌ Issues'}")
    
    # Overall status
    critical_checks = [
        results['python_version'],
        successful_packages >= total_packages * 0.9,  # 90% packages
        results['spacy_model'],
        results['file_permissions']
    ]
    
    if all(critical_checks):
        print(f"\n🎉 OVERALL STATUS: ✅ READY FOR DEPLOYMENT")
        print("   All critical components are functional")
    else:
        print(f"\n⚠️ OVERALL STATUS: ❌ NEEDS ATTENTION")
        print("   Some critical components need to be fixed")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if not results['spacy_model']:
        print("   • Install spaCy model: python -m spacy download en_core_web_sm")
    if not results['nltk_data']:
        print("   • Download NLTK data: python -c \"import nltk; nltk.download('punkt')\"")
    if not results['ai_models']:
        print("   • AI models will download automatically on first use")
    if not results['gpu']['cuda_available']:
        print("   • Consider GPU setup for faster AI processing (optional)")
    
    failed_packages = [pkg for pkg, success in package_results.items() if not success]
    if failed_packages:
        print(f"   • Install missing packages: pip install {' '.join(failed_packages)}")

def main():
    """Run complete validation suite."""
    print("🚀 NASA Space Biology Knowledge Engine")
    print("📋 Requirements Validation Suite")
    print("=" * 50)
    
    # Run all checks
    results = {
        'python_version': check_python_version(),
        'packages': check_package_imports(),
        'spacy_model': check_spacy_model(),
        'nltk_data': check_nltk_data(),
        'ai_models': check_ai_models(),
        'gpu': check_gpu_availability(),
        'file_permissions': check_file_permissions()
    }
    
    # Generate report
    generate_report(results)
    
    # Exit code for automation
    critical_success = all([
        results['python_version'],
        sum(results['packages'].values()) >= len(results['packages']) * 0.9,
        results['spacy_model'],
        results['file_permissions']
    ])
    
    sys.exit(0 if critical_success else 1)

if __name__ == "__main__":
    main()