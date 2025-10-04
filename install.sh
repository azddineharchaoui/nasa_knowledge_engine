#!/bin/bash
# NASA Space Biology Knowledge Engine - Complete Installation Script
# This script installs all dependencies and sets up the environment

echo "🚀 NASA Space Biology Knowledge Engine Installation"
echo "=================================================="

# Check Python version
python_version=$(python --version 2>&1)
echo "📋 Detected Python version: $python_version"

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📦 Installing Python packages from requirements.txt..."
pip install -r requirements.txt

# Download spaCy language model (CRITICAL)
echo "📥 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "📥 Downloading NLTK data..."
python -c "
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    print('✅ NLTK punkt tokenizer already available')
except LookupError:
    print('📥 Downloading NLTK punkt tokenizer...')
    nltk.download('punkt')
    print('✅ NLTK punkt tokenizer downloaded')
"

# Verify spaCy installation
echo "🔍 Verifying spaCy installation..."
python -c "
import spacy
try:
    nlp = spacy.load('en_core_web_sm')
    print('✅ spaCy model loaded successfully')
    print(f'   Model info: {nlp.meta[\"name\"]} v{nlp.meta[\"version\"]}')
except OSError:
    print('❌ spaCy model failed to load')
    print('   Run: python -m spacy download en_core_web_sm')
    exit(1)
"

# Verify core imports
echo "🔍 Verifying core package imports..."
python -c "
import sys

packages_to_test = [
    'streamlit', 'pandas', 'requests', 'beautifulsoup4', 'nltk', 
    'spacy', 'networkx', 'plotly', 'torch', 'transformers', 
    'numpy', 'scipy', 'sklearn', 'psutil', 'pytest'
]

failed_imports = []
for package in packages_to_test:
    try:
        if package == 'beautifulsoup4':
            import bs4
        elif package == 'sklearn':
            import sklearn
        else:
            __import__(package)
        print(f'✅ {package}')
    except ImportError as e:
        print(f'❌ {package}: {e}')
        failed_imports.append(package)

if failed_imports:
    print(f'\\n⚠️ Failed to import: {failed_imports}')
    print('Please install missing packages manually.')
    sys.exit(1)
else:
    print('\\n🎉 All required packages imported successfully!')
"

# Test transformers availability
echo "🤖 Testing AI model availability..."
python -c "
try:
    from transformers import pipeline
    # Test with a small model for quick verification
    classifier = pipeline('sentiment-analysis', 
                         model='distilbert-base-uncased-finetuned-sst-2-english',
                         return_all_scores=False)
    result = classifier('This is a test.')
    print('✅ Transformers pipeline working')
    print(f'   Test result: {result}')
except Exception as e:
    print(f'⚠️ Transformers test failed: {e}')
    print('   AI summarization may use fallback methods')
"

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data
mkdir -p logs
mkdir -p cache
echo "✅ Directories created: data/, logs/, cache/"

# Check available models
echo "📋 Checking available AI models..."
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

models_to_check = [
    'facebook/bart-large-cnn',
    'facebook/bart-base', 
    'sshleifer/distilbart-cnn-12-6',
    't5-small'
]

print('Available models:')
for model_name in models_to_check:
    try:
        # Just check if model config is accessible
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f'✅ {model_name}')
    except Exception as e:
        print(f'⚠️ {model_name}: May need download on first use')

print(f'\\nPyTorch CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch device: {\"cuda\" if torch.cuda.is_available() else \"cpu\"}')
"

echo ""
echo "🎉 Installation Complete!"
echo "======================="
echo ""
echo "🚀 To run the application:"
echo "   streamlit run app.py"
echo ""
echo "🧪 To run tests:"
echo "   pytest tests/ -v"
echo ""
echo "📖 To validate the complete pipeline:"
echo "   python test_final_integration.py"
echo ""
echo "💡 First run may be slow as AI models download automatically"
echo ""