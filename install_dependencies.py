#!/usr/bin/env python3
"""
NASA Space Biology Knowledge Engine - Enhanced Installation Script
Handles all dependencies with proper error checking and fallbacks.
"""

import subprocess
import sys
import os
import importlib
from typing import List, Tuple, Optional

class DependencyInstaller:
    def __init__(self):
        self.installed_packages = []
        self.failed_packages = []
        self.optional_packages = []
        
    def run_command(self, command: List[str], description: str) -> Tuple[bool, str]:
        """Run a command with error handling."""
        try:
            print(f"\n🔄 {description}...")
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True
            )
            print(f"✅ {description} completed successfully")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed: {e.stderr}")
            return False, e.stderr
        except Exception as e:
            print(f"❌ Unexpected error during {description}: {str(e)}")
            return False, str(e)
    
    def check_package(self, package_name: str, import_name: Optional[str] = None) -> bool:
        """Check if a package is already installed."""
        try:
            if import_name:
                importlib.import_module(import_name)
            else:
                importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    def install_core_packages(self):
        """Install core packages first."""
        print("\n🚀 Installing Core Packages...")
        
        core_packages = [
            ("pip", "pip>=24.0"),
            ("setuptools", "setuptools>=68.0.0"),
            ("wheel", "wheel>=0.42.0"),
        ]
        
        for package_name, package_spec in core_packages:
            if not self.check_package(package_name):
                success, output = self.run_command(
                    [sys.executable, "-m", "pip", "install", "--upgrade", package_spec],
                    f"Installing {package_name}"
                )
                if success:
                    self.installed_packages.append(package_name)
                else:
                    self.failed_packages.append(package_name)
    
    def install_framework_packages(self):
        """Install framework packages."""
        print("\n📦 Installing Framework Packages...")
        
        framework_packages = [
            "streamlit>=1.38.0",
            "pandas>=2.2.0", 
            "requests>=2.31.0",
            "click>=8.1.0",
            "altair>=5.0.0",
            "protobuf>=4.25.0",
            "toml>=0.10.0",
            "pyyaml>=6.0.0",
            "markdown>=3.6.0",
        ]
        
        for package in framework_packages:
            package_name = package.split(">=")[0].split("==")[0]
            if not self.check_package(package_name):
                success, output = self.run_command(
                    [sys.executable, "-m", "pip", "install", package],
                    f"Installing {package_name}"
                )
                if success:
                    self.installed_packages.append(package_name)
                else:
                    self.failed_packages.append(package_name)
    
    def install_ml_packages(self):
        """Install ML packages with fallbacks for compatibility."""
        print("\n🤖 Installing ML/AI Packages...")
        
        # Try to install PyTorch with CPU support first
        torch_packages = [
            ("torch>=2.6.0", "torch"),
            ("torchvision>=0.21.0", "torchvision"), 
            ("torchaudio>=2.6.0", "torchaudio")
        ]
        
        for package_spec, import_name in torch_packages:
            if not self.check_package(import_name):
                # Try CPU version first
                success, output = self.run_command(
                    [sys.executable, "-m", "pip", "install", package_spec, "--index-url", "https://download.pytorch.org/whl/cpu"],
                    f"Installing {import_name} (CPU version)"
                )
                
                if not success:
                    # Fallback to regular installation
                    success, output = self.run_command(
                        [sys.executable, "-m", "pip", "install", package_spec],
                        f"Installing {import_name} (fallback)"
                    )
                
                if success:
                    self.installed_packages.append(import_name)
                else:
                    self.failed_packages.append(import_name)
        
        # Install other ML packages
        ml_packages = [
            ("transformers>=4.44.0", "transformers"),
            ("sentence-transformers>=2.7.0", "sentence_transformers"),
            ("scikit-learn>=1.4.0", "sklearn"),
            ("numpy>=1.26.0", "numpy"),
            ("scipy>=1.12.0", "scipy"),
            ("joblib>=1.3.0", "joblib"),
        ]
        
        for package_spec, import_name in ml_packages:
            if not self.check_package(import_name):
                success, output = self.run_command(
                    [sys.executable, "-m", "pip", "install", package_spec],
                    f"Installing {import_name}"
                )
                if success:
                    self.installed_packages.append(import_name)
                else:
                    self.failed_packages.append(import_name)
    
    def install_nlp_packages(self):
        """Install NLP packages."""
        print("\n📝 Installing NLP Packages...")
        
        nlp_packages = [
            ("nltk>=3.8.1", "nltk"),
            ("spacy>=3.7.6", "spacy"),
            ("beautifulsoup4>=4.12.0", "bs4"),
            ("lxml>=4.9.0", "lxml"),
            ("tokenizers>=0.20.0", "tokenizers"),
            ("regex>=2023.12.0", "regex"),
        ]
        
        for package_spec, import_name in nlp_packages:
            if not self.check_package(import_name):
                success, output = self.run_command(
                    [sys.executable, "-m", "pip", "install", package_spec],
                    f"Installing {import_name}"
                )
                if success:
                    self.installed_packages.append(import_name)
                else:
                    self.failed_packages.append(import_name)
    
    def install_visualization_packages(self):
        """Install visualization packages."""
        print("\n📊 Installing Visualization Packages...")
        
        viz_packages = [
            ("plotly>=5.15.0", "plotly"),
            ("networkx>=3.3", "networkx"),
            ("matplotlib>=3.8.0", "matplotlib"),
        ]
        
        for package_spec, import_name in viz_packages:
            if not self.check_package(import_name):
                success, output = self.run_command(
                    [sys.executable, "-m", "pip", "install", package_spec],
                    f"Installing {import_name}"
                )
                if success:
                    self.installed_packages.append(import_name)
                else:
                    self.failed_packages.append(import_name)
    
    def install_utility_packages(self):
        """Install utility packages."""
        print("\n🛠️ Installing Utility Packages...")
        
        util_packages = [
            ("psutil>=6.0.0", "psutil"),
            ("tqdm>=4.66.0", "tqdm"),
            ("werkzeug>=3.0.0", "werkzeug"),
            ("pytest>=7.4.0", "pytest"),
        ]
        
        for package_spec, import_name in util_packages:
            if not self.check_package(import_name):
                success, output = self.run_command(
                    [sys.executable, "-m", "pip", "install", package_spec],
                    f"Installing {import_name}"
                )
                if success:
                    self.installed_packages.append(import_name)
                else:
                    self.failed_packages.append(import_name)
    
    def download_nlp_models(self):
        """Download required NLP models."""
        print("\n🧠 Downloading NLP Models...")
        
        # Download spaCy model
        if self.check_package("spacy"):
            success, output = self.run_command(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                "Downloading spaCy English model"
            )
            if not success:
                print("⚠️ spaCy model download failed - will use fallback NLP processing")
        
        # Download NLTK data
        if self.check_package("nltk"):
            success, output = self.run_command(
                [sys.executable, "-c", "import nltk; nltk.download('punkt'); nltk.download('stopwords')"],
                "Downloading NLTK data"
            )
            if not success:
                print("⚠️ NLTK data download failed - will use basic tokenization")
    
    def validate_installation(self):
        """Validate that key packages are working."""
        print("\n✅ Validating Installation...")
        
        critical_imports = [
            ("streamlit", "Streamlit framework"),
            ("pandas", "Data processing"),
            ("requests", "HTTP requests"),
            ("plotly", "Visualizations"),
            ("networkx", "Graph processing"),
        ]
        
        working_imports = []
        failed_imports = []
        
        for import_name, description in critical_imports:
            try:
                importlib.import_module(import_name)
                working_imports.append(f"✅ {description} ({import_name})")
            except ImportError:
                failed_imports.append(f"❌ {description} ({import_name})")
        
        print("\n📋 Installation Summary:")
        print("=" * 50)
        
        if working_imports:
            print("✅ Working Dependencies:")
            for item in working_imports:
                print(f"   {item}")
        
        if failed_imports:
            print("\n❌ Failed Dependencies:")
            for item in failed_imports:
                print(f"   {item}")
        
        if self.installed_packages:
            print(f"\n📦 Successfully installed {len(self.installed_packages)} packages")
        
        if self.failed_packages:
            print(f"\n⚠️  Failed to install {len(self.failed_packages)} packages: {', '.join(self.failed_packages)}")
        
        # Test core functionality
        try:
            print("\n🔍 Testing Core Functionality...")
            import pandas as pd
            import networkx as nx
            import plotly.graph_objects as go
            
            # Create test data
            df = pd.DataFrame({'test': [1, 2, 3]})
            G = nx.Graph()
            G.add_edge(1, 2)
            fig = go.Figure()
            
            print("✅ Core functionality test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Core functionality test failed: {e}")
            return False
    
    def run_installation(self):
        """Run the complete installation process."""
        print("🚀 NASA Space Biology Knowledge Engine - Dependency Installation")
        print("=" * 60)
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        
        try:
            self.install_core_packages()
            self.install_framework_packages() 
            self.install_nlp_packages()
            self.install_visualization_packages()
            self.install_ml_packages()
            self.install_utility_packages()
            self.download_nlp_models()
            
            success = self.validate_installation()
            
            if success:
                print("\n🎉 Installation completed successfully!")
                print("\nNext steps:")
                print("1. Run: streamlit run nasa_knowledge_engine/app.py")
                print("2. Or test: python -c 'import nasa_knowledge_engine.app; print(\"Success!\")'")
            else:
                print("\n⚠️  Installation completed with some issues.")
                print("The application may still work with reduced functionality.")
            
            return success
            
        except KeyboardInterrupt:
            print("\n❌ Installation interrupted by user.")
            return False
        except Exception as e:
            print(f"\n❌ Installation failed with error: {e}")
            return False

def main():
    installer = DependencyInstaller()
    success = installer.run_installation()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())