@echo off
REM NASA Space Biology Knowledge Engine - Complete Installation Script (Windows)
REM This script installs all dependencies and sets up the environment

echo 🚀 NASA Space Biology Knowledge Engine Installation
echo ==================================================

REM Check Python version
echo 📋 Detecting Python version...
python --version

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing Python packages from requirements.txt...
pip install -r requirements.txt

REM Download spaCy language model (CRITICAL)
echo 📥 Downloading spaCy language model...
python -m spacy download en_core_web_sm

REM Download NLTK data
echo 📥 Downloading NLTK data...
python -c "import nltk; nltk.download('punkt')"

REM Verify spaCy installation
echo 🔍 Verifying spaCy installation...
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ spaCy model loaded successfully')"

REM Verify core imports
echo 🔍 Verifying core package imports...
python -c "packages=['streamlit','pandas','requests','bs4','nltk','spacy','networkx','plotly','torch','transformers','numpy','scipy','sklearn','psutil','pytest']; [__import__(p) for p in packages]; print('✅ All packages imported successfully')"

REM Test AI models
echo 🤖 Testing AI model availability...
python -c "from transformers import pipeline; print('✅ Transformers available')"

REM Create directories
echo 📁 Creating data directories...
if not exist "data" mkdir data
if not exist "logs" mkdir logs  
if not exist "cache" mkdir cache
echo ✅ Directories created

echo.
echo 🎉 Installation Complete!
echo =======================
echo.
echo 🚀 To run the application:
echo    streamlit run app.py
echo.
echo 🧪 To run tests:
echo    pytest tests/ -v
echo.
echo 📖 To validate the pipeline:
echo    python test_final_integration.py
echo.
pause