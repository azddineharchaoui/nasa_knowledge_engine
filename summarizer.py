"""
AI Summarization module for NASA Space Biology hackathon prototype.

This module provides BART-based text summarization for research abstracts,
with GPU acceleration support and intelligent caching for performance.
"""

import pandas as pd
import time
import re
import math
from typing import List, Dict, Optional, Union, Tuple
from collections import Counter, defaultdict
from utils import log, log_error

# Optional imports with fallbacks for installation flexibility
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    log("✓ Transformers library loaded successfully")
except ImportError:
    log_error("Transformers/PyTorch not available - summarization will use fallback methods")
    TRANSFORMERS_AVAILABLE = False
    torch = None
    pipeline = None

# Advanced text analysis imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
    log("✓ Scikit-learn available for advanced text analysis")
except ImportError:
    log_error("Scikit-learn not available - using basic text analysis")
    SKLEARN_AVAILABLE = False
    np = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    log_error("NLTK not available - using basic tokenization")
    NLTK_AVAILABLE = False

# Global cache for the summarization pipeline
_summarizer_cache = None

# Space Biology Domain-Specific Keywords for Impact Analysis
SPACE_BIOLOGY_IMPACTS = {
    'health': {
        'bone': ['bone loss', 'bone density', 'osteoporosis', 'demineralization', 'bone remodeling'],
        'muscle': ['muscle atrophy', 'muscle mass', 'muscle weakness', 'sarcopenia', 'muscle wasting'],
        'cardiovascular': ['cardiovascular deconditioning', 'cardiac', 'blood pressure', 'orthostatic intolerance'],
        'immune': ['immune suppression', 'immune function', 'T-cell', 'cytokine', 'inflammation'],
        'vision': ['vision changes', 'SANS', 'optic disc swelling', 'visual acuity'],
        'kidney': ['kidney stones', 'renal', 'calcium metabolism', 'hypercalciuria'],
        'radiation': ['radiation exposure', 'DNA damage', 'cancer risk', 'cosmic rays']
    },
    'biological': {
        'gene_expression': ['gene expression', 'RNA', 'transcription', 'upregulation', 'downregulation'],
        'protein': ['protein synthesis', 'proteomics', 'enzyme activity', 'protein levels'],
        'cellular': ['cell division', 'apoptosis', 'cell cycle', 'cellular stress', 'oxidative stress'],
        'metabolism': ['metabolism', 'glucose', 'insulin', 'metabolic', 'energy'],
        'circadian': ['circadian rhythm', 'sleep', 'melatonin', 'sleep-wake cycle']
    },
    'operational': {
        'performance': ['cognitive performance', 'motor skills', 'reaction time', 'coordination'],
        'countermeasures': ['exercise', 'COLRES', 'treadmill', 'resistance training', 'nutrition'],
        'mission': ['ISS', 'EVA', 'spacewalk', 'mission duration', 'long-duration']
    }
}

QUANTITATIVE_PATTERNS = [
    r'(\d+(?:\.\d+)?%)\s*(?:increase|decrease|reduction|change)',
    r'(\d+(?:\.\d+)?-fold)\s*(?:increase|decrease|higher|lower)',
    r'p\s*[<>=]\s*(\d+(?:\.\d+)?)',
    r'(\d+(?:\.\d+)?)\s*(?:mg|kg|ml|cm|mm|μm|nm)',
    r'(\d+)\s*(?:days?|weeks?|months?|years?)\s*(?:of|in|during)',
    r'n\s*=\s*(\d+)',
    r'(\d+(?:\.\d+)?)\s*(?:±|\+/-|SD|SE)\s*(\d+(?:\.\d+)?)',
]

ORGANISM_PATTERNS = {
    'mouse': ['mouse', 'mice', 'mus musculus', 'murine'],
    'rat': ['rat', 'rats', 'rattus norvegicus'],
    'human': ['human', 'astronaut', 'crew', 'subject', 'participant'],
    'plant': ['arabidopsis', 'plant', 'seedling', 'growth'],
    'cell': ['cell', 'culture', 'fibroblast', 'osteoblast', 'myocyte']
}


def load_summarizer() -> Optional[object]:
    """
    Load and cache summarization pipeline with fallback model chain.
    
    Implements intelligent model selection with fallbacks:
    1. facebook/bart-large-cnn (primary)
    2. facebook/bart-base (medium)
    3. sshleifer/distilbart-cnn-12-6 (lightweight)
    4. t5-small (minimal)
    
    Features:
    - GPU/CPU auto-detection
    - Memory-optimized loading
    - Progress indicators
    - Robust error handling
    - Model warming and validation
    
    Returns:
        Transformers pipeline object or None if all models fail
    """
    global _summarizer_cache
    
    # Return cached pipeline if available
    if _summarizer_cache is not None:
        log("✓ Using cached summarizer")
        return _summarizer_cache
    
    if not TRANSFORMERS_AVAILABLE:
        log_error("❌ Transformers library not available - install with: pip install transformers torch")
        return None
    
    # Model fallback chain - ordered by capability
    model_chain = [
        ('facebook/bart-large-cnn', 'BART Large (Best Quality)'),
        ('facebook/bart-base', 'BART Base (Balanced)'), 
        ('sshleifer/distilbart-cnn-12-6', 'DistilBART (Lightweight)'),
        ('t5-small', 'T5 Small (Minimal)')
    ]
    
    # Intelligent device selection with memory management
    device_info = _get_optimal_device()
    device = device_info['device']
    
    log(f"🚀 Loading summarization model on {device_info['description']}...")
    
    for model_name, model_desc in model_chain:
        try:
            log(f"📦 Attempting to load {model_desc}: {model_name}")
            
            # Memory check before loading large models
            if not _check_memory_requirements(model_name):
                log(f"⚠️ Insufficient memory for {model_name}, trying lighter model...")
                continue
            
            # Load model with progress indication
            _summarizer_cache = _load_model_with_progress(
                model_name=model_name,
                device=device,
                device_info=device_info
            )
            
            # Validate model functionality
            if _validate_summarizer(_summarizer_cache):
                log(f"✅ Successfully loaded and validated {model_desc}")
                return _summarizer_cache
            else:
                log(f"❌ Model validation failed for {model_name}")
                _summarizer_cache = None
                continue
                
        except Exception as e:
            log_error(f"❌ Failed to load {model_name}: {str(e)}")
            _summarizer_cache = None
            
            # Check if it's a memory error
            if 'memory' in str(e).lower() or 'cuda out of memory' in str(e).lower():
                log("💾 Memory error detected - trying lighter model...")
                continue
            elif 'connection' in str(e).lower() or 'timeout' in str(e).lower():
                log("🌐 Network error detected - check internet connection")
                break  # Don't try other models if network is the issue
            else:
                log(f"🔄 Trying next model in fallback chain...")
                continue
    
    log_error("❌ All summarization models failed to load - using fallback methods")
    return None


def _get_optimal_device() -> Dict[str, Union[str, int]]:
    """Detect optimal device configuration with memory info."""
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            return {
                'device': 0,
                'type': 'cuda', 
                'description': f'GPU (CUDA) - {gpu_memory:.1f}GB VRAM',
                'memory_gb': gpu_memory
            }
        else:
            import psutil
            cpu_memory = psutil.virtual_memory().total / 1e9  # GB
            return {
                'device': -1,
                'type': 'cpu',
                'description': f'CPU - {cpu_memory:.1f}GB RAM',
                'memory_gb': cpu_memory
            }
    except Exception as e:
        log_error(f"Device detection error: {str(e)}")
        return {
            'device': -1,
            'type': 'cpu',
            'description': 'CPU (default)',
            'memory_gb': 8.0  # Conservative estimate
        }


def _check_memory_requirements(model_name: str) -> bool:
    """Check if system has sufficient memory for model."""
    try:
        # Rough memory requirements (GB)
        memory_requirements = {
            'facebook/bart-large-cnn': 2.0,
            'facebook/bart-base': 1.0,
            'sshleifer/distilbart-cnn-12-6': 0.5,
            't5-small': 0.3
        }
        
        required_gb = memory_requirements.get(model_name, 1.0)
        
        if torch.cuda.is_available():
            # Check GPU memory
            free_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            available_memory = free_memory * 0.8  # Leave 20% buffer
        else:
            # Check system memory
            import psutil
            available_memory = psutil.virtual_memory().available / 1e9
            available_memory *= 0.5  # Conservative for CPU inference
        
        has_sufficient = available_memory >= required_gb
        log(f"💾 Memory check: {available_memory:.1f}GB available, {required_gb:.1f}GB required - {'✅ OK' if has_sufficient else '❌ Insufficient'}")
        
        return has_sufficient
        
    except Exception as e:
        log_error(f"Memory check failed: {str(e)}")
        return True  # Assume OK if check fails


def _load_model_with_progress(model_name: str, device: int, device_info: Dict) -> object:
    """Load model with progress indication and optimized settings."""
    try:
        log(f"⏳ Downloading/loading model components...")
        
        # Optimized model configuration based on device
        model_kwargs = {
            'low_cpu_mem_usage': True,
            'torch_dtype': torch.float16 if device_info['type'] == 'cuda' else torch.float32
        }
        
        # Add device-specific optimizations
        if device_info['type'] == 'cuda':
            model_kwargs.update({
                'device_map': 'auto',
                'use_safetensors': True
            })
        
        # Load pipeline with timeout handling
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Model loading timed out")
        
        # Cross-platform model loading with timeout
        pipeline_obj = None
        
        if hasattr(signal, 'SIGALRM'):
            # Unix/Linux timeout using signals
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5 minutes
            
            try:
                pipeline_obj = pipeline(
                    task='summarization',
                    model=model_name,
                    device=device,
                    model_kwargs=model_kwargs,
                    clean_up_tokenization_spaces=True
                )
            finally:
                signal.alarm(0)  # Clear timeout
                
        else:
            # Windows timeout using threading
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def load_model_thread():
                try:
                    model_obj = pipeline(
                        task='summarization',
                        model=model_name,
                        device=device,
                        model_kwargs=model_kwargs,
                        clean_up_tokenization_spaces=True
                    )
                    result_queue.put(model_obj)
                except Exception as e:
                    exception_queue.put(e)
            
            thread = threading.Thread(target=load_model_thread)
            thread.daemon = True
            thread.start()
            thread.join(timeout=300)  # 5 minutes
            
            if thread.is_alive():
                raise TimeoutError("Model loading timed out after 5 minutes")
            
            if not exception_queue.empty():
                raise exception_queue.get()
            
            if not result_queue.empty():
                pipeline_obj = result_queue.get()
            else:
                raise Exception("Model loading failed without specific error")
        
        log(f"✅ Model loaded successfully")
        return pipeline_obj
            
    except TimeoutError:
        raise Exception(f"Model loading timed out after 5 minutes")
    except Exception as e:
        raise Exception(f"Model loading failed: {str(e)}")


def _validate_summarizer(summarizer: object) -> bool:
    """Validate summarizer functionality with test input."""
    if summarizer is None:
        return False
    
    try:
        log("🧪 Validating summarizer with test input...")
        
        test_text = (
            "Space biology research investigates the effects of microgravity on living organisms. "
            "Studies have shown that astronauts experience bone loss, muscle atrophy, and "
            "cardiovascular deconditioning during long-duration spaceflight missions. "
            "Understanding these biological changes is crucial for future Mars exploration missions."
        )
        
        # Test summarization with conservative parameters
        result = summarizer(
            test_text,
            max_length=50,
            min_length=20,
            do_sample=False,
            truncation=True
        )
        
        # Validate result structure
        if (isinstance(result, list) and len(result) > 0 and 
            isinstance(result[0], dict) and 'summary_text' in result[0]):
            
            summary = result[0]['summary_text']
            
            # Basic quality checks
            if (summary and len(summary.strip()) >= 10 and 
                len(summary) < len(test_text) and 
                'space' in summary.lower() or 'biology' in summary.lower()):
                
                log(f"✅ Validation successful - generated {len(summary)} char summary")
                return True
            else:
                log(f"❌ Validation failed - poor quality summary: '{summary[:100]}...'")
                return False
        else:
            log(f"❌ Validation failed - invalid result structure: {type(result)}")
            return False
            
    except Exception as e:
        log_error(f"❌ Summarizer validation failed: {str(e)}")
        return False


def _create_space_biology_prompt(text: str) -> str:
    """Create impact-focused prompt for space biology research summarization."""
    space_bio_context = (
        "Summarize the space biology research impacts and key findings: "
        "Focus on health effects, biological changes, mission implications, "
        "quantitative results, statistical significance, countermeasures, "
        "and organism-specific experimental conditions. "
    )
    return f"{space_bio_context}\n\n{text}"


def _validate_space_biology_summary_quality(summary: str, original_text: str) -> bool:
    """Validate summary quality with space biology-specific criteria."""
    if not summary or len(summary.strip()) < 20:
        return False
    
    # Length validation (should be 40-120 words for optimal impact communication)
    word_count = len(summary.split())
    if word_count < 30 or word_count > 150:
        return False
    
    # Compression ratio validation (should compress meaningfully)
    compression_ratio = len(summary) / len(original_text)
    if compression_ratio > 0.8 or compression_ratio < 0.1:
        return False
    
    # Space biology relevance validation
    space_bio_terms = {
        'microgravity', 'space', 'astronaut', 'spaceflight', 'iss', 'mars',
        'bone', 'muscle', 'cardiovascular', 'radiation', 'immune',
        'health', 'effects', 'changes', 'mission', 'countermeasures',
        'biological', 'physiological', 'organism', 'experimental'
    }
    
    summary_lower = summary.lower()
    original_lower = original_text.lower()
    
    # Check for space biology terminology
    space_terms_in_summary = sum(1 for term in space_bio_terms if term in summary_lower)
    space_terms_in_original = sum(1 for term in space_bio_terms if term in original_lower)
    
    # Summary should retain key space biology terms
    if space_terms_in_original > 0 and space_terms_in_summary == 0:
        return False
    
    # Content coherence validation
    if summary.endswith('...') or summary.count('.') == 0:
        return False
    
    # Check for quantitative information preservation
    import re
    numbers_in_original = len(re.findall(r'\d+\.?\d*%?', original_text))
    numbers_in_summary = len(re.findall(r'\d+\.?\d*%?', summary))
    
    # Should preserve some quantitative data if present
    if numbers_in_original >= 2 and numbers_in_summary == 0:
        return False
    
    # Word overlap validation (meaningful content retention)
    original_words = set(original_lower.split())
    summary_words = set(summary_lower.split())
    
    if len(original_words) > 0:
        overlap = len(summary_words.intersection(original_words)) / len(original_words)
        if overlap < 0.15:  # At least 15% overlap for relevance
            return False
    
    return True


def _calculate_summary_relevance_score(summary: str, original_text: str) -> float:
    """Calculate relevance score for space biology summaries (0-1 scale)."""
    if not summary or not original_text:
        return 0.0
    
    score = 0.0
    
    # Impact terminology score (0.3 weight)
    impact_terms = {
        'effects', 'impacts', 'changes', 'findings', 'results',
        'significant', 'increased', 'decreased', 'reduced', 'enhanced'
    }
    summary_lower = summary.lower()
    impact_score = sum(0.05 for term in impact_terms if term in summary_lower)
    score += min(impact_score, 0.3)
    
    # Space biology domain score (0.25 weight)
    domain_terms = {
        'microgravity', 'space', 'astronaut', 'spaceflight', 'radiation',
        'bone', 'muscle', 'cardiovascular', 'immune', 'health'
    }
    domain_score = sum(0.05 for term in domain_terms if term in summary_lower)
    score += min(domain_score, 0.25)
    
    # Quantitative information score (0.2 weight)
    import re
    quantitative_patterns = [
        r'\d+\.?\d*%',  # Percentages
        r'\d+\.?\d*\s*fold',  # Fold changes
        r'\d+\.?\d*\s*times',  # Multiples
        r'p\s*[<>=]\s*0\.\d+',  # P-values
        r'\d+\.?\d*\s*months?',  # Time periods
        r'\d+\.?\d*\s*days?'  # Time periods
    ]
    quant_matches = sum(len(re.findall(pattern, summary, re.IGNORECASE)) 
                       for pattern in quantitative_patterns)
    score += min(quant_matches * 0.05, 0.2)
    
    # Length appropriateness score (0.15 weight)
    word_count = len(summary.split())
    if 50 <= word_count <= 90:  # Optimal range for impact summaries
        score += 0.15
    elif 40 <= word_count <= 120:  # Acceptable range
        score += 0.10
    
    # Coherence and structure score (0.1 weight)
    sentence_count = summary.count('.') + summary.count('!') + summary.count('?')
    if 2 <= sentence_count <= 4:  # Well-structured summary
        score += 0.1
    elif sentence_count >= 1:  # At least one complete sentence
        score += 0.05
    
    return min(score, 1.0)


def summarize_text(
    text: str,
    min_length: int = 40,
    max_length: int = 120,
    do_sample: bool = False,
    num_beams: int = 4,
    temperature: float = 1.0,
    timeout_seconds: int = 30
) -> Optional[str]:
    """
    Summarize single text with enhanced AI model and robust error handling.
    
    Features:
    - Optimized parameters for space biology content
    - Timeout protection for slow inference
    - Quality validation of generated summaries
    - Automatic fallback on failure
    - Impact-focused prompting
    
    Args:
        text: Input text to summarize
        min_length: Minimum summary length in tokens (default: 40)
        max_length: Maximum summary length in tokens (default: 120)
        do_sample: Use sampling vs beam search (default: False)
        num_beams: Number of beams for beam search (default: 4)
        temperature: Sampling temperature (default: 1.0)
        timeout_seconds: Maximum inference time (default: 30s)
        
    Returns:
        High-quality summary text or None if all methods fail
        
    Example:
        >>> summary = summarize_text(
        ...     "Microgravity causes bone loss in astronauts...", 
        ...     max_length=80,
        ...     num_beams=4
        ... )
        >>> print(f"Impact Summary: {summary}")
    """
    # Input validation
    if not text or not isinstance(text, str):
        log("❌ Invalid input: text must be non-empty string")
        return None
        
    clean_text = text.strip()
    if len(clean_text) < 50:
        log(f"⚠️ Text too short for summarization ({len(clean_text)} chars)")
        return clean_text if len(clean_text) > 0 else None
    
    # Parameter validation
    min_length = max(10, min(min_length, max_length - 10))
    max_length = max(min_length + 10, max_length)
    
    summarizer = load_summarizer()
    if summarizer is None:
        log("🔄 Using fallback summarization method")
        return _enhanced_fallback_summarization(clean_text, max_length)
    
    try:
        log(f"🤖 AI summarizing text ({len(clean_text)} chars) with {num_beams} beams...")
        
        # Create enhanced space biology impact-focused prompt
        impact_prompt = _create_space_biology_prompt(clean_text)
        

        
        # Execute with cross-platform timeout protection
        def _execute_summarization():
            return summarizer(
                impact_prompt,
                min_length=min_length,
                max_length=max_length,
                do_sample=do_sample,
                num_beams=num_beams if not do_sample else 1,
                temperature=temperature if do_sample else 0.7,
                truncation=True,
                # Enhanced quality parameters for space biology
                no_repeat_ngram_size=3,
                early_stopping=True,
                length_penalty=1.1,      # Slightly prefer shorter summaries
                repetition_penalty=1.3,  # Stronger penalty for repetition
                diversity_penalty=0.2,   # Encourage diverse content
                num_return_sequences=1
            )
        
        # Cross-platform timeout handling
        import signal
        result = None
        if hasattr(signal, 'SIGALRM'):
            # Unix/Linux timeout
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Summarization timed out after {timeout_seconds}s")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = _execute_summarization()
            finally:
                signal.alarm(0)  # Clear timeout
        else:
            # Windows timeout using threading
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def summarization_thread():
                try:
                    res = _execute_summarization()
                    result_queue.put(res)
                except Exception as e:
                    exception_queue.put(e)
            
            thread = threading.Thread(target=summarization_thread)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Summarization timed out after {timeout_seconds}s")
            
            if not exception_queue.empty():
                raise exception_queue.get()
            
            if not result_queue.empty():
                result = result_queue.get()
            else:
                raise Exception("Summarization failed without specific error")
        
        # Extract and validate summary
        if isinstance(result, list) and len(result) > 0:
            raw_summary = result[0].get('summary_text', '')
            
            # Clean up prompt artifacts
            summary = _clean_summary_text(raw_summary, impact_prompt)
            
            # Enhanced space biology quality validation
            if _validate_space_biology_summary_quality(summary, clean_text):
                relevance_score = _calculate_summary_relevance_score(summary, clean_text)
                word_count = len(summary.split())
                log(f"✅ Summary passed quality validation (relevance: {relevance_score:.2f})")
                log(f"✅ High-quality summary generated ({len(summary)} chars, {word_count} words)")
                return summary
            else:
                log(f"⚠️ AI summary failed space biology quality check, using fallback")
                return _enhanced_fallback_summarization(clean_text, max_length)
        else:
            log(f"❌ Invalid AI response format: {type(result)}")
            return _enhanced_fallback_summarization(clean_text, max_length)
            
    except TimeoutError as e:
        log_error(f"⏱️ Summarization timeout: {str(e)}")
        return _enhanced_fallback_summarization(clean_text, max_length)
    except Exception as e:
        log_error(f"🤖 AI summarization failed: {str(e)}")
        return _enhanced_fallback_summarization(clean_text, max_length)


def _clean_summary_text(raw_summary: str, original_prompt: str) -> str:
    """Clean AI-generated summary from prompt artifacts and improve readability."""
    if not raw_summary:
        return ""
    
    summary = raw_summary.strip()
    
    # Remove prompt prefix if it appears in summary
    prompt_prefix = "Summarize the space biology research impacts and key findings:"
    if summary.startswith(prompt_prefix):
        summary = summary[len(prompt_prefix):].strip()
    
    # Remove common AI artifacts
    artifacts = [
        "Here is a summary:", "Summary:", "In summary,", "To summarize,",
        "The text describes", "This text discusses", "The research shows"
    ]
    
    for artifact in artifacts:
        if summary.lower().startswith(artifact.lower()):
            summary = summary[len(artifact):].strip()
    
    # Clean up punctuation and spacing
    summary = ' '.join(summary.split())  # Normalize whitespace
    
    # Ensure proper sentence ending
    if summary and not summary[-1] in '.!?':
        summary += '.'
    
    return summary


def _validate_summary_quality(summary: str, original_text: str) -> bool:
    """Validate that generated summary meets quality standards."""
    if not summary or not original_text:
        return False
    
    try:
        # Length validation
        if len(summary) >= len(original_text):
            log(f"❌ Summary too long: {len(summary)} vs {len(original_text)} chars")
            return False
        
        if len(summary) < 20:
            log(f"❌ Summary too short: {len(summary)} chars")
            return False
        
        # Content validation
        summary_lower = summary.lower()
        original_lower = original_text.lower()
        
        # Check for key space biology terms
        space_bio_terms = [
            'space', 'microgravity', 'astronaut', 'mission', 'radiation', 
            'bone', 'muscle', 'health', 'biological', 'effect', 'study',
            'research', 'iss', 'mars', 'flight'
        ]
        
        terms_in_summary = sum(1 for term in space_bio_terms if term in summary_lower)
        terms_in_original = sum(1 for term in space_bio_terms if term in original_lower)
        
        # Summary should retain key domain terms
        if terms_in_original > 0 and terms_in_summary == 0:
            log(f"❌ Summary missing space biology context")
            return False
        
        # Check for nonsensical repetition
        words = summary_lower.split()
        if len(set(words)) < len(words) * 0.6:  # Too many repeated words
            log(f"❌ Summary has excessive repetition")
            return False
        
        # Basic coherence check
        if summary.count('.') == 0 and len(summary) > 50:
            log(f"❌ Summary lacks proper sentence structure")
            return False
        
        log(f"✅ Summary passed quality validation")
        return True
        
    except Exception as e:
        log_error(f"Summary validation error: {str(e)}")
        return False  # Fail safe


def summarize_batch(
    texts: List[str],
    min_length: int = 40,
    max_length: int = 120,
    do_sample: bool = False,
    num_beams: int = 4,
    batch_size: int = 4,
    show_progress: bool = True,
    timeout_per_batch: int = 120,
    relevance_threshold: float = 0.5
) -> List[Optional[str]]:
    """
    High-performance batch summarization with progress tracking and memory management.
    
    Features:
    - Intelligent batch processing with memory optimization
    - Progress tracking and ETA calculation
    - Robust error handling with per-batch fallbacks
    - Memory cleanup between batches
    - Quality validation for each summary
    - Impact-focused prompting for space biology content
    
    Args:
        texts: List of texts to summarize
        min_length: Minimum summary length (default: 40)
        max_length: Maximum summary length (default: 120)
        do_sample: Use sampling vs beam search (default: False)
        num_beams: Beam search width (default: 4)
        batch_size: Texts per batch - adjusted for memory (default: 4)
        show_progress: Show progress indicators (default: True)
        timeout_per_batch: Timeout per batch in seconds (default: 120s)
        
    Returns:
        List of summaries with same length as input texts
        
    Example:
        >>> abstracts = ["Space research abstract 1...", "Biology study 2..."]
        >>> summaries = summarize_batch(abstracts, batch_size=6, num_beams=4)
        >>> print(f"Processed {len(summaries)} abstracts")
    """
    if not texts or len(texts) == 0:
        log("⚠️ Empty text list provided")
        return []
    
    total_texts = len(texts)
    log(f"🚀 Starting batch summarization of {total_texts} texts...")
    
    # Dynamic batch size adjustment based on available memory
    optimal_batch_size = _calculate_optimal_batch_size(batch_size, total_texts)
    
    summarizer = load_summarizer()
    if summarizer is None:
        log("🔄 AI model unavailable - using enhanced fallback for all texts")
        return [_enhanced_fallback_summarization(text, max_length) if text else None for text in texts]
    
    summaries = []
    successful_count = 0
    start_time = time.time() if show_progress else None
    
    try:
        # Process in optimized batches
        total_batches = (total_texts + optimal_batch_size - 1) // optimal_batch_size
        
        for batch_idx in range(0, total_texts, optimal_batch_size):
            batch_num = batch_idx // optimal_batch_size + 1
            batch_end = min(batch_idx + optimal_batch_size, total_texts)
            batch = texts[batch_idx:batch_end]
            
            if show_progress:
                progress = (batch_num / total_batches) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / batch_num) * (total_batches - batch_num) if batch_num > 0 else 0
                log(f"📦 Processing batch {batch_num}/{total_batches} ({progress:.1f}%) - ETA: {eta:.1f}s")
            
            # Process current batch with timeout protection
            try:
                batch_summaries = _process_single_batch(
                    batch=batch,
                    summarizer=summarizer,
                    min_length=min_length,
                    max_length=max_length,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    timeout=timeout_per_batch,
                    relevance_threshold=relevance_threshold
                )
                
                summaries.extend(batch_summaries)
                successful_count += sum(1 for s in batch_summaries if s and s != "No summary available")
                
                # Memory cleanup after each batch
                _cleanup_batch_memory()
                
            except Exception as batch_error:
                log_error(f"❌ Batch {batch_num} failed: {str(batch_error)}")
                # Fallback processing for failed batch
                fallback_summaries = [
                    _enhanced_fallback_summarization(text, max_length) if text else "No summary available" 
                    for text in batch
                ]
                summaries.extend(fallback_summaries)
                successful_count += sum(1 for s in fallback_summaries if s and s != "No summary available")
        
        # Final statistics
        total_time = time.time() - start_time if start_time else 0
        success_rate = (successful_count / total_texts) * 100 if total_texts > 0 else 0
        
        log(f"✅ Batch summarization completed: {successful_count}/{total_texts} successful ({success_rate:.1f}%) in {total_time:.1f}s")
        return summaries
        
    except Exception as e:
        log_error(f"🚨 Critical batch processing error: {str(e)}")
        # Complete fallback for all remaining texts
        remaining_texts = texts[len(summaries):]
        fallback_summaries = [
            _enhanced_fallback_summarization(text, max_length) if text else "No summary available" 
            for text in remaining_texts
        ]
        summaries.extend(fallback_summaries)
        return summaries


def _calculate_optimal_batch_size(requested_batch_size: int, total_texts: int) -> int:
    """Calculate optimal batch size based on available memory and text count."""
    try:
        # Get available memory
        device_info = _get_optimal_device()
        available_memory_gb = device_info.get('memory_gb', 8.0)
        
        # Memory-based batch size calculation
        # Rough estimate: 0.5GB per text for BART processing
        memory_based_batch = max(1, int(available_memory_gb * 0.4 / 0.5))
        
        # Choose conservative batch size
        optimal_batch = min(requested_batch_size, memory_based_batch, 8)  # Max 8 per batch
        
        if optimal_batch != requested_batch_size:
            log(f"💾 Adjusted batch size from {requested_batch_size} to {optimal_batch} for memory optimization")
        
        return optimal_batch
        
    except Exception as e:
        log_error(f"Batch size calculation error: {str(e)}")
        return min(requested_batch_size, 4)  # Safe fallback


def _process_single_batch(
    batch: List[str], 
    summarizer: object,
    min_length: int,
    max_length: int, 
    do_sample: bool,
    num_beams: int,
    timeout: int,
    relevance_threshold: float = 0.5
) -> List[str]:
    """Process a single batch with cross-platform timeout protection and quality validation."""
    import signal
    
    # Prepare texts with impact prompting
    processed_texts = []
    text_indices = []  # Track which original indices have valid texts
    
    for i, text in enumerate(batch):
        if text and len(text.strip()) >= 50:
            impact_prompt = f"Summarize the space biology research impacts and key findings: {text.strip()}"
            processed_texts.append(impact_prompt)
            text_indices.append(i)
    
    # Initialize results for the batch
    batch_results = ["No summary available"] * len(batch)
    
    if not processed_texts:
        return batch_results
    
    # Cross-platform timeout processing
    def _execute_batch_summarization():
        return summarizer(
            processed_texts,
            min_length=min_length,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams if not do_sample else 1,
            truncation=True,
            no_repeat_ngram_size=3,
            early_stopping=True,
            length_penalty=1.0,
            repetition_penalty=1.2
        )
    
    import signal
    results = None
    try:
        if hasattr(signal, 'SIGALRM'):
            # Unix/Linux timeout
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Batch processing timed out after {timeout}s")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                results = _execute_batch_summarization()
            finally:
                signal.alarm(0)  # Clear timeout
        else:
            # Windows timeout using threading
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def batch_thread():
                try:
                    res = _execute_batch_summarization()
                    result_queue.put(res)
                except Exception as e:
                    exception_queue.put(e)
            
            thread = threading.Thread(target=batch_thread)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout)
            
            if thread.is_alive():
                raise TimeoutError(f"Batch processing timed out after {timeout}s")
            
            if not exception_queue.empty():
                raise exception_queue.get()
            
            if not result_queue.empty():
                results = result_queue.get()
            else:
                raise Exception("Batch processing failed")
        
        # Extract and validate summaries
        if results:
            for i, result in enumerate(results):
                if i < len(text_indices):
                    original_idx = text_indices[i]
                    original_text = batch[original_idx]
                    
                    if isinstance(result, dict) and 'summary_text' in result:
                        raw_summary = result['summary_text']
                        clean_summary = _clean_summary_text(raw_summary, processed_texts[i])
                        
                        # Enhanced quality validation with relevance scoring
                        if _validate_space_biology_summary_quality(clean_summary, original_text):
                            relevance_score = _calculate_summary_relevance_score(clean_summary, original_text)
                            if relevance_score >= relevance_threshold:
                                batch_results[original_idx] = clean_summary
                                log(f"✅ Summary passed quality validation")
                            else:
                                log(f"⚠️ Summary relevance too low ({relevance_score:.2f}), using fallback")
                                batch_results[original_idx] = _enhanced_fallback_summarization(original_text, max_length)
                        else:
                            # Use fallback for poor quality AI summary
                            log(f"⚠️ Summary failed quality validation, using fallback")
                            batch_results[original_idx] = _enhanced_fallback_summarization(original_text, max_length)
        
        return batch_results
        
    except TimeoutError:
        log_error(f"⏱️ Batch timed out, using fallback for all texts")
        # Fallback for timeout
        for i in text_indices:
            batch_results[i] = _enhanced_fallback_summarization(batch[i], max_length)
        return batch_results


def _cleanup_batch_memory():
    """Clean up memory between batches to prevent accumulation."""
    try:
        import gc
        gc.collect()
        
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        # Memory cleanup is not critical, just log and continue
        pass


def _fallback_summarization(text: str, max_length: int = 100) -> Optional[str]:
    """
    Fallback summarization using simple text extraction methods.
    
    Used when BART model is unavailable or fails.
    
    Args:
        text: Input text to summarize
        max_length: Maximum length of summary
        
    Returns:
        Simple extractive summary or None
    """
    if not text or len(text.strip()) < 50:
        return None
    
    try:
        # Simple extractive summarization
        sentences = text.split('. ')
        
        if len(sentences) <= 1:
            # Single sentence or no periods - truncate
            return text[:max_length].strip() + "..." if len(text) > max_length else text.strip()
        
        # Take first 2-3 sentences up to max_length
        summary = ""
        for sentence in sentences[:3]:
            candidate = (summary + " " + sentence).strip()
            if len(candidate) > max_length:
                break
            summary = candidate
        
        if not summary:
            # If no sentences fit, truncate first sentence
            summary = sentences[0][:max_length-3] + "..."
        
        return summary.strip()
        
    except Exception as e:
        log_error(f"Fallback summarization failed: {str(e)}")
        return text[:max_length].strip() + "..." if len(text) > max_length else text.strip()


def get_summarizer_info() -> Dict[str, Union[str, bool]]:
    """
    Get information about the current summarizer configuration.
    
    Returns:
        Dictionary with summarizer status and configuration
    """
    info = {
        'transformers_available': TRANSFORMERS_AVAILABLE,
        'cuda_available': torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False,
        'model_loaded': _summarizer_cache is not None,
        'model_name': 'facebook/bart-large-cnn' if TRANSFORMERS_AVAILABLE else 'fallback',
        'device': 'cuda' if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) else 'cpu'
    }
    
    return info


def summarize_abstracts(df: pd.DataFrame, batch_size: int = 6, show_progress: bool = True) -> pd.DataFrame:
    """
    Process DataFrame abstracts with advanced AI summarization and comprehensive error handling.
    
    Features:
    - High-performance batch processing optimized for hackathon demos
    - Impact-focused prompting for space biology research
    - Robust fallback chain for reliability
    - Progress tracking with ETA calculation
    - Memory optimization for large datasets
    - Quality validation and metrics
    - Graceful handling of missing/invalid data
    
    Args:
        df: DataFrame with 'abstract' column (required)
        batch_size: Texts per processing batch (default: 6, auto-adjusted)
        show_progress: Display progress indicators (default: True)
        
    Returns:
        DataFrame with new 'summary' column containing 50-100 word impact summaries
        
    Example:
        >>> research_df = pd.DataFrame({
        ...     'title': ['Bone Loss Study', 'Muscle Atrophy Research'],
        ...     'abstract': ['Microgravity causes...', 'Space environments lead...']
        ... })
        >>> summarized_df = summarize_abstracts(research_df)
        >>> print(summarized_df['summary'].iloc[0])
    """
    import time
    start_time = time.time()
    
    # Input validation and logging
    log(f"🚀 Starting abstract summarization for {len(df)} publications...")
    
    if df is None or df.empty:
        log("⚠️ Empty or None DataFrame provided")
        empty_df = pd.DataFrame() if df is None else df.copy()
        empty_df['summary'] = []
        return empty_df
    
    # Column validation
    if 'abstract' not in df.columns:
        log_error("❌ Missing 'abstract' column in DataFrame")
        df_copy = df.copy()
        df_copy['summary'] = ['No abstract available'] * len(df)
        return df_copy
    
    # Prepare abstracts with data cleaning
    abstracts = df['abstract'].fillna('').astype(str)
    valid_count = sum(1 for abstract in abstracts if len(abstract.strip()) >= 50)
    
    log(f"📊 Dataset analysis: {valid_count}/{len(abstracts)} abstracts suitable for AI summarization")
    
    if valid_count == 0:
        log("⚠️ No valid abstracts found (all too short)")
        df_copy = df.copy()
        df_copy['summary'] = ['Abstract too short for summarization'] * len(df)
        return df_copy
    
    # Initialize summarization system
    summarizer_info = get_summarizer_info()
    log(f"🤖 Summarization config: {summarizer_info['model_name']} on {summarizer_info['device']}")
    
    try:
        # High-performance batch processing
        log(f"⏳ Processing {len(abstracts)} abstracts with batch_size={batch_size}...")
        
        summaries = summarize_batch(
            texts=abstracts.tolist(),
            min_length=40,           # Ensure sufficient detail for impacts
            max_length=120,          # Allow comprehensive impact descriptions
            do_sample=False,         # Use beam search for consistency
            num_beams=4,             # Enhanced beam search for quality
            batch_size=batch_size,
            show_progress=show_progress,
            timeout_per_batch=150,   # Extended time for quality processing
            relevance_threshold=0.5  # Ensure high-quality space biology summaries
        )
        
        # Add summaries to DataFrame
        df_result = df.copy()
        df_result['summary'] = summaries
        
        # Calculate and log comprehensive statistics
        stats = _calculate_summarization_stats(abstracts.tolist(), summaries, start_time)
        _log_summarization_results(stats)
        
        # Optional: Add metadata columns for analysis
        if show_progress:
            df_result['summary_length'] = [len(s) if s else 0 for s in summaries]
            df_result['summarization_method'] = [
                'AI' if s and s != 'No summary available' and 'fallback' not in s.lower() else 'Fallback'
                for s in summaries
            ]
        
        log(f"✅ Abstract summarization pipeline completed successfully in {time.time() - start_time:.1f}s")
        return df_result
        
    except Exception as e:
        log_error(f"🚨 Critical error in abstract summarization: {str(e)}")
        
        # Emergency fallback: process with simple method
        log("🆘 Activating emergency fallback summarization...")
        
        try:
            emergency_summaries = [
                _enhanced_fallback_summarization(abstract, 100) if abstract and len(abstract.strip()) >= 50 
                else 'No summary available'
                for abstract in abstracts
            ]
            
            df_result = df.copy()
            df_result['summary'] = emergency_summaries
            
            emergency_success = sum(1 for s in emergency_summaries if s != 'No summary available')
            log(f"🆘 Emergency fallback completed: {emergency_success}/{len(abstracts)} summaries generated")
            
            return df_result
            
        except Exception as emergency_error:
            log_error(f"🚨 Emergency fallback also failed: {str(emergency_error)}")
            
            # Final failsafe
            df_result = df.copy()
            df_result['summary'] = ['Summarization system unavailable'] * len(df)
            return df_result


def _calculate_summarization_stats(abstracts: List[str], summaries: List[str], start_time: float) -> Dict:
    """Calculate comprehensive statistics for summarization performance."""
    total_time = time.time() - start_time
    
    # Count statistics
    total_abstracts = len(abstracts)
    valid_abstracts = sum(1 for a in abstracts if a and len(a.strip()) >= 50)
    successful_summaries = sum(1 for s in summaries if s and s != 'No summary available' and len(s.strip()) > 20)
    ai_summaries = sum(1 for s in summaries if s and 'fallback' not in s.lower() and s != 'No summary available')
    
    # Length statistics
    abstract_lengths = [len(a) for a in abstracts if a]
    summary_lengths = [len(s) for s in summaries if s and s != 'No summary available']
    
    # Compression statistics
    compression_ratios = []
    for abstract, summary in zip(abstracts, summaries):
        if abstract and summary and summary != 'No summary available' and len(abstract.strip()) > 0:
            ratio = len(summary) / len(abstract)
            compression_ratios.append(ratio)
    
    return {
        'total_abstracts': total_abstracts,
        'valid_abstracts': valid_abstracts, 
        'successful_summaries': successful_summaries,
        'ai_summaries': ai_summaries,
        'fallback_summaries': successful_summaries - ai_summaries,
        'success_rate': (successful_summaries / total_abstracts * 100) if total_abstracts > 0 else 0,
        'ai_success_rate': (ai_summaries / valid_abstracts * 100) if valid_abstracts > 0 else 0,
        'avg_abstract_length': sum(abstract_lengths) / len(abstract_lengths) if abstract_lengths else 0,
        'avg_summary_length': sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0,
        'avg_compression_ratio': sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0,
        'processing_time': total_time,
        'throughput': total_abstracts / total_time if total_time > 0 else 0
    }


def _log_summarization_results(stats: Dict):
    """Log comprehensive summarization results and performance metrics."""
    log("📈 SUMMARIZATION RESULTS:")
    log(f"   • Total processed: {stats['total_abstracts']} abstracts")
    log(f"   • Successful summaries: {stats['successful_summaries']} ({stats['success_rate']:.1f}%)")
    log(f"   • AI summaries: {stats['ai_summaries']} ({stats['ai_success_rate']:.1f}% of valid)")
    log(f"   • Fallback summaries: {stats['fallback_summaries']}")
    log(f"   • Average lengths: {stats['avg_abstract_length']:.0f} → {stats['avg_summary_length']:.0f} chars")
    log(f"   • Compression ratio: {stats['avg_compression_ratio']:.2f}x")
    log(f"   • Processing time: {stats['processing_time']:.1f}s ({stats['throughput']:.1f} abstracts/sec)")


def _batch_process_abstracts_with_bart(
    prompted_abstracts: List[str], 
    summarizer, 
    batch_size: int = 4,
    max_length: int = 100
) -> List[str]:
    """
    Process abstracts using BART model in batches for optimal performance.
    
    Args:
        prompted_abstracts: List of impact-prompted abstracts
        summarizer: Loaded BART pipeline
        batch_size: Number of abstracts to process simultaneously
        max_length: Maximum summary length
        
    Returns:
        List of summaries
    """
    summaries = []
    
    try:
        log(f"BART batch processing {len(prompted_abstracts)} abstracts (batch_size={batch_size})...")
        
        for i in range(0, len(prompted_abstracts), batch_size):
            batch = prompted_abstracts[i:i + batch_size]
            
            # Separate valid and invalid texts in batch
            batch_results = []
            valid_texts = []
            valid_indices = []
            
            for j, text in enumerate(batch):
                if text and len(text.strip()) >= 50:
                    valid_texts.append(text)
                    valid_indices.append(j)
            
            # Process valid texts if any
            if valid_texts:
                try:
                    log(f"Processing batch {i//batch_size + 1}: {len(valid_texts)} valid abstracts")
                    
                    # BART summarization with impact focus
                    bart_results = summarizer(
                        valid_texts,
                        max_length=max_length,
                        min_length=30,
                        do_sample=False,
                        truncation=True,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )
                    
                    # Extract summary texts
                    valid_summaries = []
                    for result in bart_results:
                        if isinstance(result, dict) and 'summary_text' in result:
                            summary = result['summary_text']
                            # Clean up the prompt prefix if it appears in summary
                            if summary.startswith('Summarize the space biology impacts'):
                                # Find the actual summary after the prompt
                                parts = summary.split(':', 1)
                                if len(parts) > 1:
                                    summary = parts[1].strip()
                            valid_summaries.append(summary)
                        else:
                            valid_summaries.append("Summary generation failed")
                    
                    # Map results back to batch positions
                    valid_idx = 0
                    for j in range(len(batch)):
                        if j in valid_indices:
                            batch_results.append(valid_summaries[valid_idx])
                            valid_idx += 1
                        else:
                            batch_results.append("No summary available")
                            
                except Exception as batch_error:
                    log_error(f"BART batch {i//batch_size + 1} failed: {str(batch_error)}")
                    # Fallback for failed batch
                    batch_results = [
                        _fallback_summarization(text.replace('Summarize the space biology impacts and key findings: ', ''), max_length) 
                        if text else "No summary available" 
                        for text in batch
                    ]
            else:
                # No valid texts in batch
                batch_results = ["No summary available"] * len(batch)
            
            summaries.extend(batch_results)
            
        log(f"✓ BART batch processing completed")
        return summaries
        
    except Exception as e:
        log_error(f"BART batch processing failed: {str(e)}")
        # Complete fallback
        return [
            _fallback_summarization(text.replace('Summarize the space biology impacts and key findings: ', ''), max_length) 
            if text else "No summary available" 
            for text in prompted_abstracts
        ]


def _batch_process_abstracts_fallback(
    abstracts: List[str], 
    max_length: int = 100
) -> List[str]:
    """
    Fallback batch processing using rule-based summarization.
    
    Args:
        abstracts: List of original abstracts
        max_length: Maximum summary length
        
    Returns:
        List of summaries using extractive methods
    """
    log(f"Fallback batch processing {len(abstracts)} abstracts...")
    
    summaries = []
    for abstract in abstracts:
        if not abstract or len(abstract.strip()) < 50:
            summaries.append("No summary available")
        else:
            # Enhanced fallback with impact focus
            summary = _enhanced_fallback_summarization(abstract, max_length)
            summaries.append(summary)
    
    return summaries


def _enhanced_fallback_summarization(text: str, max_length: int = 100) -> str:
    """
    Enhanced fallback summarization with space biology impact focus.
    
    Args:
        text: Input abstract text
        max_length: Maximum summary length
        
    Returns:
        Impact-focused extractive summary
    """
    if not text or len(text.strip()) < 50:
        return "No summary available"
    
    try:
        # Space biology impact keywords for sentence prioritization
        impact_keywords = {
            'effects', 'impacts', 'results', 'findings', 'shows', 'demonstrates',
            'microgravity', 'radiation', 'bone loss', 'muscle atrophy', 'astronaut',
            'spaceflight', 'iss', 'mars', 'space', 'biological', 'health'
        }
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 1:
            return text[:max_length-3] + "..." if len(text) > max_length else text
        
        # Score sentences by impact keyword presence
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for keyword in impact_keywords if keyword.lower() in sentence.lower())
            scored_sentences.append((sentence, score))
        
        # Sort by score (impact relevance) and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Build summary from highest scoring sentences
        summary = ""
        for sentence, score in scored_sentences:
            candidate = (summary + " " + sentence + ".").strip()
            if len(candidate) > max_length:
                break
            summary = candidate
            
            # Stop if we have enough content
            if len(summary) >= max_length * 0.7:  # 70% of max length
                break
        
        if not summary:
            # Fallback to first sentence
            summary = sentences[0]
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
        
        return summary
        
    except Exception as e:
        log_error(f"Enhanced fallback summarization failed: {str(e)}")
        return text[:max_length-3] + "..." if len(text) > max_length else text


def clear_summarizer_cache():
    """Clear the cached summarizer to free memory."""
    global _summarizer_cache
    if _summarizer_cache is not None:
        del _summarizer_cache
        _summarizer_cache = None
        log("✓ Summarizer cache cleared")


# Test and demonstration functionality
if __name__ == '__main__':
    """Test the summarization functionality."""
    log("Testing summarizer.py functionality...")
    
    # Test data - NASA space biology abstracts
    test_abstracts = [
        """
        Bone loss in space environments represents a critical challenge for long-duration spaceflight. 
        This study examines the effects of microgravity on bone mineral density in laboratory mice over 
        a 30-day period aboard the International Space Station. Results indicate significant decreases 
        in trabecular bone volume and cortical thickness, with implications for astronaut health during 
        extended missions to Mars and beyond. The research provides insights into countermeasure 
        development for maintaining skeletal integrity in space.
        """,
        
        """
        Space radiation poses significant challenges to biological systems during interplanetary travel. 
        This investigation studies the effects of galactic cosmic radiation on Arabidopsis thaliana 
        seedlings grown in simulated space conditions. Findings reveal altered gene expression patterns 
        in DNA repair pathways and oxidative stress responses, providing insights for future deep space 
        missions. The work contributes to understanding radiation protection requirements for Mars exploration.
        """,
        
        """
        Muscle atrophy represents one of the most significant physiological challenges facing astronauts 
        during long-duration missions. This comprehensive study analyzes molecular mechanisms underlying 
        muscle wasting in space through proteomics analysis of tissue samples from space-flown rodents. 
        Results identify key pathways involved in protein degradation and provide targets for 
        countermeasure development to maintain crew fitness during extended spaceflight operations.
        """
    ]
    
    try:
        # Test summarizer info
        info = get_summarizer_info()
        log(f"Summarizer configuration: {info}")
        
        # Test DataFrame summarization (main feature)
        log("\nTesting DataFrame abstract summarization...")
        
        # Try to load sample data from preprocess module
        try:
            from preprocess import load_and_preprocess
            log("Loading sample data from preprocessing pipeline...")
            sample_df = load_and_preprocess('data/sample_publications.json')
            
            if not sample_df.empty:
                # Use real sample data
                log(f"✓ Loaded {len(sample_df)} sample publications")
                result_df = summarize_abstracts(sample_df)
                
                # Critical assertion: summaries should be shorter than abstracts
                if 'summary' in result_df.columns and 'abstract' in result_df.columns:
                    for i, row in result_df.iterrows():
                        if row['summary'] and row['summary'] != 'No summary available':
                            assert len(row['summary']) < len(row['abstract']), f"Summary longer than abstract for row {i}"
                    log("✓ All summaries are shorter than original abstracts")
                else:
                    log("⚠ Missing required columns for length comparison")
                
            else:
                log("⚠ Sample data empty, using test data instead")
                raise ImportError("Empty sample data")
                
        except (ImportError, FileNotFoundError, Exception) as e:
            log(f"Could not load sample data: {str(e)}, using test data instead")
            
            # Fallback to test DataFrame
            test_df = pd.DataFrame({
                'title': [
                    'Microgravity Effects on Bone Density',
                    'Space Radiation Impact on Plants',
                    'Muscle Atrophy in Space'
                ],
                'abstract': test_abstracts,
                'experiment_id': ['TEST_001', 'TEST_002', 'TEST_003']
            })
            
            result_df = summarize_abstracts(test_df)
            
            # Critical assertion for test data
            if 'summary' in result_df.columns:
                for i, row in result_df.iterrows():
                    if row['summary'] and row['summary'] != 'No summary available':
                        assert len(row['summary']) < len(row['abstract']), f"Summary longer than abstract for row {i}"
                log("✓ All test summaries are shorter than original abstracts")
        
        if 'summary' in result_df.columns:
            log("✓ DataFrame summarization successful")
            
            print("\n" + "="*80)
            print("DATAFRAME SUMMARIZATION TEST:")
            print("="*80)
            
            for i, row in result_df.iterrows():
                print(f"\nPublication {i+1}:")
                print(f"Title: {row['title']}")
                print(f"Original abstract length: {len(row['abstract'])} chars")
                print(f"Summary: {row['summary']}")
                print(f"Summary length: {len(row['summary'])} chars")
                print("-" * 60)
            
            print("="*80)
        else:
            log("⚠ DataFrame summarization failed - no summary column added")
        
        # Test single summarization
        log("\nTesting single text summarization...")
        summary = summarize_text(
            test_abstracts[0],
            min_length=30,
            max_length=100,
            do_sample=False
        )
        
        if summary:
            log(f"✓ Single summarization successful")
            print("\n" + "="*60)
            print("SINGLE SUMMARIZATION TEST:")
            print("="*60)
            print(f"Original length: {len(test_abstracts[0])} chars")
            print(f"Summary length: {len(summary)} chars")
            print(f"Summary: {summary}")
            print("="*60)
        else:
            log("⚠ Single summarization returned None")
        
        # Test batch summarization
        log("\nTesting batch summarization...")
        summaries = summarize_batch(
            test_abstracts,
            min_length=30,
            max_length=80,
            batch_size=2
        )
        
        successful_summaries = sum(1 for s in summaries if s)
        log(f"✓ Batch summarization: {successful_summaries}/{len(test_abstracts)} successful")
        
        print(f"\nBATCH SUMMARIZATION RESULTS:")
        print("="*60)
        for i, (original, summary) in enumerate(zip(test_abstracts, summaries)):
            print(f"\nAbstract {i+1}:")
            print(f"  Original: {len(original)} chars")
            if summary:
                print(f"  Summary: {summary}")
                print(f"  Length: {len(summary)} chars")
            else:
                print(f"  Summary: [Failed]")
        print("="*60)
        
        # Test empty DataFrame handling
        log("\nTesting empty DataFrame handling...")
        empty_df = pd.DataFrame({'abstract': []})
        empty_result = summarize_abstracts(empty_df)
        if 'summary' in empty_result.columns:
            log("✓ Empty DataFrame handled correctly")
        
        # Test missing abstract column
        log("\nTesting missing abstract column...")
        no_abstract_df = pd.DataFrame({'title': ['Test'], 'other': ['data']})
        no_abstract_result = summarize_abstracts(no_abstract_df)
        if 'summary' in no_abstract_result.columns:
            log("✓ Missing abstract column handled correctly")
        
        # Performance info
        log(f"\n🎉 Summarization testing completed!")
        log(f"📊 Configuration: {info['model_name']} on {info['device']}")
        log(f"📈 Batch processing: 4 abstracts per batch for optimal performance")
        log(f"🎯 Impact-focused prompting: 'Summarize space biology impacts...'")
        
    except Exception as e:
        log_error(f"Testing failed: {str(e)}")
        log("This may be expected if transformers/torch are not installed")
        
        # Test fallback functionality with DataFrame
        log("\nTesting fallback DataFrame summarization...")
        try:
            test_df = pd.DataFrame({
                'abstract': [test_abstracts[0][:200]]  # Shorter for fallback
            })
            fallback_result = summarize_abstracts(test_df)
            if 'summary' in fallback_result.columns:
                log("✓ Fallback DataFrame summarization working")
                print(f"Fallback summary: {fallback_result['summary'].iloc[0]}")
            else:
                log("✗ Fallback DataFrame summarization failed")
        except Exception as fallback_error:
            log_error(f"Fallback testing failed: {str(fallback_error)}")
    
    finally:
        # Cleanup
        clear_summarizer_cache()


# Advanced Summarization Functions

def _ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    if not NLTK_AVAILABLE:
        return False
    
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        return True
    except LookupError:
        try:
            log("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            return True
        except Exception as e:
            log_error(f"Failed to download NLTK data: {e}")
            return False


def _extract_sentences(text: str) -> List[str]:
    """Extract sentences from text using NLTK or basic splitting."""
    if NLTK_AVAILABLE and _ensure_nltk_data():
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    
    # Fallback to basic sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def _calculate_sentence_tfidf_scores(sentences: List[str], query_keywords: Optional[List[str]] = None) -> List[float]:
    """Calculate TF-IDF scores for sentences."""
    if not SKLEARN_AVAILABLE or len(sentences) < 2:
        # Fallback to simple word frequency scoring
        return _calculate_simple_sentence_scores(sentences, query_keywords)
    
    try:
        # Prepare documents (sentences)
        documents = [' '.join(word_tokenize(sent.lower())) if NLTK_AVAILABLE else sent.lower() 
                    for sent in sentences]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english' if NLTK_AVAILABLE else None,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate sentence scores
        scores = []
        for i, sentence in enumerate(sentences):
            # Base TF-IDF score (sum of all term scores)
            base_score = float(tfidf_matrix[i].sum())
            
            # Position weighting (earlier sentences get higher scores)
            position_weight = 1.0 - (i / len(sentences)) * 0.3
            
            # Length normalization (prefer medium-length sentences)
            length_score = min(1.0, len(sentence.split()) / 20)
            
            # Query keyword boost
            keyword_boost = 1.0
            if query_keywords:
                sentence_lower = sentence.lower()
                keyword_matches = sum(1 for kw in query_keywords if kw.lower() in sentence_lower)
                keyword_boost = 1.0 + (keyword_matches * 0.5)
            
            final_score = base_score * position_weight * length_score * keyword_boost
            scores.append(final_score)
        
        return scores
        
    except Exception as e:
        log_error(f"TF-IDF calculation failed: {e}")
        return _calculate_simple_sentence_scores(sentences, query_keywords)


def _calculate_simple_sentence_scores(sentences: List[str], query_keywords: Optional[List[str]] = None) -> List[float]:
    """Fallback sentence scoring using word frequency."""
    scores = []
    
    # Get all words for frequency calculation
    all_words = []
    for sent in sentences:
        words = word_tokenize(sent.lower()) if NLTK_AVAILABLE else sent.lower().split()
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    max_freq = max(word_freq.values()) if word_freq else 1
    
    for i, sentence in enumerate(sentences):
        words = word_tokenize(sentence.lower()) if NLTK_AVAILABLE else sentence.lower().split()
        
        # Word frequency score
        freq_score = sum(word_freq[word] for word in words) / max_freq
        
        # Position weight
        position_weight = 1.0 - (i / len(sentences)) * 0.3
        
        # Query keyword boost
        keyword_boost = 1.0
        if query_keywords:
            sentence_lower = sentence.lower()
            keyword_matches = sum(1 for kw in query_keywords if kw.lower() in sentence_lower)
            keyword_boost = 1.0 + (keyword_matches * 0.5)
        
        scores.append(freq_score * position_weight * keyword_boost)
    
    return scores


def _extract_named_entities_frequency(text: str) -> Dict[str, int]:
    """Extract and count named entities for sentence importance."""
    entities = defaultdict(int)
    
    # Space biology specific entities
    text_lower = text.lower()
    
    # Count organism mentions
    for organism, patterns in ORGANISM_PATTERNS.items():
        for pattern in patterns:
            entities[f"organism_{organism}"] += len(re.findall(rf'\b{re.escape(pattern)}\b', text_lower))
    
    # Count impact mentions
    for category, impacts in SPACE_BIOLOGY_IMPACTS.items():
        for impact_type, terms in impacts.items():
            for term in terms:
                count = len(re.findall(rf'\b{re.escape(term)}\b', text_lower))
                if count > 0:
                    entities[f"{category}_{impact_type}"] += count
    
    return dict(entities)


def extract_extractive_summary(text: str, num_sentences: int = 3, 
                              query_keywords: Optional[List[str]] = None) -> Dict[str, any]:
    """Extract most informative sentences using TF-IDF and entity analysis."""
    start_time = time.time()
    
    try:
        # Extract sentences
        sentences = _extract_sentences(text)
        
        if len(sentences) <= num_sentences:
            return {
                'sentences': sentences,
                'scores': [1.0] * len(sentences),
                'summary': ' '.join(sentences),
                'metadata': {
                    'method': 'all_sentences',
                    'processing_time': time.time() - start_time,
                    'sentence_count': len(sentences)
                }
            }
        
        # Calculate TF-IDF scores
        tfidf_scores = _calculate_sentence_tfidf_scores(sentences, query_keywords)
        
        # Calculate named entity frequency scores
        entity_freq = _extract_named_entities_frequency(text)
        
        # Enhanced sentence scoring
        enhanced_scores = []
        for i, (sentence, tfidf_score) in enumerate(zip(sentences, tfidf_scores)):
            # Entity density score for this sentence
            sent_entities = _extract_named_entities_frequency(sentence)
            entity_score = sum(sent_entities.values()) / max(1, len(sentence.split()))
            
            # Quantitative content score
            quant_matches = sum(1 for pattern in QUANTITATIVE_PATTERNS 
                              if re.search(pattern, sentence, re.IGNORECASE))
            quant_score = min(1.0, quant_matches * 0.3)
            
            # Combined score
            final_score = tfidf_score + (entity_score * 0.3) + (quant_score * 0.2)
            enhanced_scores.append(final_score)
        
        # Select top sentences
        sentence_pairs = list(zip(sentences, enhanced_scores, range(len(sentences))))
        sentence_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_sentences = sentence_pairs[:num_sentences]
        # Sort by original order for coherent summary
        top_sentences.sort(key=lambda x: x[2])
        
        selected_sentences = [s[0] for s in top_sentences]
        selected_scores = [s[1] for s in top_sentences]
        
        summary = ' '.join(selected_sentences)
        
        return {
            'sentences': selected_sentences,
            'scores': selected_scores,
            'summary': summary,
            'metadata': {
                'method': 'tfidf_enhanced',
                'processing_time': time.time() - start_time,
                'original_sentence_count': len(sentences),
                'selected_count': len(selected_sentences),
                'avg_score': sum(selected_scores) / len(selected_scores),
                'entity_frequency': entity_freq,
                'compression_ratio': len(summary) / len(text)
            }
        }
        
    except Exception as e:
        log_error(f"Extractive summary extraction failed: {e}")
        # Fallback to first N sentences
        sentences = _extract_sentences(text)[:num_sentences]
        return {
            'sentences': sentences,
            'scores': [0.5] * len(sentences),
            'summary': ' '.join(sentences),
            'metadata': {
                'method': 'fallback_first_sentences',
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
        }


def extract_impact_focused_analysis(text: str, impact_type: str = 'all') -> Dict[str, any]:
    """Extract impact-focused information from research abstracts."""
    start_time = time.time()
    
    analysis = {
        'impact_type': impact_type,
        'quantitative_findings': [],
        'methodologies': [],
        'sample_info': {},
        'timeline_info': {},
        'organism_effects': {},
        'system_effects': {},
        'statistical_significance': [],
        'metadata': {}
    }
    
    try:
        text_lower = text.lower()
        
        # Extract quantitative findings
        for pattern in QUANTITATIVE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end].strip()
                
                analysis['quantitative_findings'].append({
                    'value': match.group(),
                    'context': context,
                    'position': match.start()
                })
        
        # Extract experimental methodologies
        methodology_patterns = [
            r'(RNA-seq|RNA sequencing|transcriptome|transcriptomic)',
            r'(proteomics?|protein analysis|western blot)',
            r'(micro-CT|microCT|computed tomography)',
            r'(qPCR|RT-PCR|quantitative PCR)',
            r'(immunofluorescence|immunohistochemistry|IHC)',
            r'(flow cytometry|FACS)',
            r'(hindlimb unloading|HLU|suspension)',
            r'(spaceflight|space flight|ISS|international space station)',
        ]
        
        for pattern in methodology_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                analysis['methodologies'].append({
                    'method': match.group(),
                    'position': match.start()
                })
        
        # Extract sample size information
        sample_patterns = [
            r'n\s*=\s*(\d+)',
            r'(\d+)\s*(?:mice|rats|subjects|animals|participants)',
            r'sample size\s*(?:of\s*)?(\d+)',
        ]
        
        for pattern in sample_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    sample_size = int(re.search(r'\d+', match.group()).group())
                    analysis['sample_info']['sample_size'] = sample_size
                    break
                except (ValueError, AttributeError):
                    continue
        
        # Extract timeline information
        timeline_patterns = [
            r'(\d+)\s*(?:day|week|month|year)s?\s*(?:of|in|during|for)',
            r'(?:for|during|over)\s*(\d+)\s*(?:day|week|month|year)s?',
            r'(\d+)[–-](\d+)\s*(?:day|week|month|year)s?',
        ]
        
        for pattern in timeline_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                analysis['timeline_info'][match.group()] = {
                    'duration': match.group(),
                    'context': text[max(0, match.start()-30):match.end()+30]
                }
        
        # Extract organism-specific effects
        if impact_type == 'all' or impact_type == 'biological':
            for organism, patterns in ORGANISM_PATTERNS.items():
                for pattern in patterns:
                    if pattern in text_lower:
                        # Find effects mentioned near organism
                        organism_positions = [m.start() for m in re.finditer(rf'\b{re.escape(pattern)}\b', text_lower)]
                        
                        for pos in organism_positions:
                            context_window = text[max(0, pos-100):pos+200]
                            
                            # Look for effects in the context
                            effects = []
                            for category, impacts in SPACE_BIOLOGY_IMPACTS.items():
                                for impact_type_key, terms in impacts.items():
                                    for term in terms:
                                        if term in context_window.lower():
                                            effects.append(f"{category}_{impact_type_key}")
                            
                            if effects:
                                analysis['organism_effects'][organism] = {
                                    'effects': list(set(effects)),
                                    'context': context_window.strip()
                                }
        
        # Extract system-specific effects
        system_keywords = {
            'musculoskeletal': ['bone', 'muscle', 'skeletal', 'osteo', 'myo'],
            'cardiovascular': ['heart', 'cardiac', 'blood', 'vascular', 'circulation'],
            'nervous': ['brain', 'neural', 'neuron', 'cognitive', 'motor'],
            'immune': ['immune', 'immunological', 'T-cell', 'B-cell', 'cytokine'],
            'endocrine': ['hormone', 'endocrine', 'insulin', 'cortisol', 'thyroid'],
        }
        
        for system, keywords in system_keywords.items():
            system_mentions = sum(1 for kw in keywords if kw in text_lower)
            if system_mentions > 0:
                # Find specific effects for this system
                effects = []
                if system in ['musculoskeletal']:
                    for term in SPACE_BIOLOGY_IMPACTS['health']['bone'] + SPACE_BIOLOGY_IMPACTS['health']['muscle']:
                        if term in text_lower:
                            effects.append(term)
                elif system == 'cardiovascular':
                    for term in SPACE_BIOLOGY_IMPACTS['health']['cardiovascular']:
                        if term in text_lower:
                            effects.append(term)
                
                analysis['system_effects'][system] = {
                    'mentions': system_mentions,
                    'effects': effects
                }
        
        # Extract statistical significance
        sig_patterns = [
            r'p\s*[<>=]\s*(\d+(?:\.\d+)?)',
            r'significance|significant(?:ly)?',
            r'p-value|p value',
            r'confidence interval|CI',
        ]
        
        for pattern in sig_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context = text[max(0, match.start()-30):match.end()+30]
                analysis['statistical_significance'].append({
                    'match': match.group(),
                    'context': context.strip()
                })
        
        # Add processing metadata
        analysis['metadata'] = {
            'processing_time': time.time() - start_time,
            'text_length': len(text),
            'quantitative_count': len(analysis['quantitative_findings']),
            'methodology_count': len(analysis['methodologies']),
            'organism_count': len(analysis['organism_effects']),
            'system_count': len(analysis['system_effects']),
            'statistical_mentions': len(analysis['statistical_significance'])
        }
        
        return analysis
        
    except Exception as e:
        log_error(f"Impact-focused analysis failed: {e}")
        analysis['metadata']['error'] = str(e)
        analysis['metadata']['processing_time'] = time.time() - start_time
        return analysis


def _calculate_coverage_metrics(summary: str, original_text: str) -> Dict[str, float]:
    """Calculate comprehensive coverage metrics for summary quality."""
    metrics = {
        'compression_ratio': len(summary) / len(original_text) if original_text else 0.0,
        'word_coverage': 0.0,
        'entity_coverage': 0.0,
        'keyword_density': 0.0,
        'readability_score': 0.0,
        'technical_complexity': 0.0
    }
    
    try:
        # Word coverage - percentage of unique words preserved
        if NLTK_AVAILABLE:
            original_words = set(word_tokenize(original_text.lower()))
            summary_words = set(word_tokenize(summary.lower()))
        else:
            original_words = set(original_text.lower().split())
            summary_words = set(summary.lower().split())
        
        if original_words:
            metrics['word_coverage'] = len(summary_words & original_words) / len(original_words)
        
        # Entity coverage - preservation of important entities
        original_entities = _extract_named_entities_frequency(original_text)
        summary_entities = _extract_named_entities_frequency(summary)
        
        if original_entities:
            entity_overlap = sum(min(original_entities.get(entity, 0), summary_entities.get(entity, 0))
                               for entity in original_entities)
            total_entities = sum(original_entities.values())
            metrics['entity_coverage'] = entity_overlap / total_entities if total_entities > 0 else 0.0
        
        # Keyword density - space biology terms per 100 words
        summary_words_count = len(summary.split())
        if summary_words_count > 0:
            space_bio_terms = 0
            summary_lower = summary.lower()
            
            for category, impacts in SPACE_BIOLOGY_IMPACTS.items():
                for impact_type, terms in impacts.items():
                    for term in terms:
                        space_bio_terms += len(re.findall(rf'\b{re.escape(term)}\b', summary_lower))
            
            metrics['keyword_density'] = (space_bio_terms / summary_words_count) * 100
        
        # Readability score (simplified)
        sentences = _extract_sentences(summary)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            # Simple readability: prefer 15-25 words per sentence
            readability = 1.0 - abs(avg_sentence_length - 20) / 20
            metrics['readability_score'] = max(0.0, min(1.0, readability))
        
        # Technical complexity - presence of technical terms
        technical_patterns = [
            r'\b\d+(?:\.\d+)?\s*(?:mg|kg|ml|cm|mm|μm|nm|%)',  # Units
            r'\bp\s*[<>=]\s*\d+(?:\.\d+)?',  # P-values
            r'\b(?:RNA|DNA|protein|gene|enzyme)\b',  # Molecular terms
            r'\b(?:microgravity|spaceflight|ISS)\b',  # Space terms
        ]
        
        technical_matches = sum(len(re.findall(pattern, summary, re.IGNORECASE)) 
                              for pattern in technical_patterns)
        metrics['technical_complexity'] = min(1.0, technical_matches / max(1, summary_words_count) * 10)
        
    except Exception as e:
        log_error(f"Coverage metrics calculation failed: {e}")
    
    return metrics


def _calculate_summary_confidence(summary: str, original_text: str, method: str) -> float:
    """Calculate confidence score for summary quality."""
    try:
        # Base confidence by method
        method_confidence = {
            'abstractive_ai': 0.8,
            'extractive_tfidf': 0.7,
            'hybrid': 0.85,
            'fallback': 0.4
        }
        
        base_confidence = method_confidence.get(method, 0.5)
        
        # Adjust based on length appropriateness
        length_ratio = len(summary) / len(original_text) if original_text else 0
        length_score = 1.0
        
        if length_ratio < 0.1:  # Too short
            length_score = 0.6
        elif length_ratio > 0.8:  # Too long
            length_score = 0.7
        elif 0.15 <= length_ratio <= 0.4:  # Good compression
            length_score = 1.0
        
        # Adjust based on content quality
        content_score = 1.0
        
        # Check for coherence (basic)
        if len(summary.split('.')) < 2:  # Very short
            content_score *= 0.8
        
        # Check for space biology relevance
        summary_lower = summary.lower()
        space_terms = sum(1 for category, impacts in SPACE_BIOLOGY_IMPACTS.items()
                         for impact_type, terms in impacts.items()
                         for term in terms if term in summary_lower)
        
        if space_terms == 0:
            content_score *= 0.7  # Low domain relevance
        elif space_terms >= 3:
            content_score *= 1.1  # High domain relevance
        
        final_confidence = base_confidence * length_score * content_score
        return min(1.0, max(0.0, final_confidence))
        
    except Exception as e:
        log_error(f"Confidence calculation failed: {e}")
        return 0.5


def generate_comprehensive_summary(text: str, keywords: Optional[List[str]] = None,
                                 summary_types: Optional[List[str]] = None) -> Dict[str, any]:
    """Generate comprehensive multi-modal summary with all enhancements."""
    start_time = time.time()
    
    if summary_types is None:
        summary_types = ['general', 'health', 'biological']
    
    if keywords is None:
        keywords = []
    
    comprehensive_result = {
        'summaries': {},
        'quality_metrics': {},
        'processing_metadata': {},
        'recommendations': []
    }
    
    try:
        # Generate extractive summary
        log("Generating extractive summary...")
        extractive_result = extract_extractive_summary(text, num_sentences=3, query_keywords=keywords)
        comprehensive_result['summaries']['extractive'] = extractive_result
        
        # Generate abstractive summary if available
        abstractive_summary = None
        abstractive_metadata = {}
        
        if TRANSFORMERS_AVAILABLE and _summarizer_cache:
            try:
                log("Generating abstractive summary...")
                abstractive_result = _summarize_single_text(text, min_length=50, max_length=120)
                if abstractive_result and 'summary' in abstractive_result:
                    abstractive_summary = abstractive_result['summary']
                    abstractive_metadata = abstractive_result.get('metadata', {})
                    comprehensive_result['summaries']['abstractive'] = {
                        'summary': abstractive_summary,
                        'metadata': abstractive_metadata
                    }
            except Exception as e:
                log_error(f"Abstractive summarization failed: {e}")
        
        # Generate impact-focused analysis for each type
        for summary_type in summary_types:
            log(f"Generating {summary_type} impact analysis...")
            impact_result = extract_impact_focused_analysis(text, impact_type=summary_type)
            comprehensive_result['summaries'][f'impact_{summary_type}'] = impact_result
        
        # Calculate comprehensive quality metrics
        log("Calculating quality metrics...")
        for summary_name, summary_data in comprehensive_result['summaries'].items():
            if isinstance(summary_data, dict) and 'summary' in summary_data:
                summary_text = summary_data['summary']
            elif isinstance(summary_data, dict) and summary_name == 'extractive':
                summary_text = summary_data.get('summary', '')
            elif isinstance(summary_data, str):
                summary_text = summary_data
            else:
                continue
            
            if summary_text:
                coverage = _calculate_coverage_metrics(summary_text, text)
                confidence = _calculate_summary_confidence(
                    summary_text, text, 
                    'abstractive_ai' if 'abstractive' in summary_name else 'extractive_tfidf'
                )
                
                comprehensive_result['quality_metrics'][summary_name] = {
                    'coverage_metrics': coverage,
                    'confidence_score': confidence,
                    'length': len(summary_text),
                    'word_count': len(summary_text.split())
                }
        
        # Add processing metadata
        comprehensive_result['processing_metadata'] = {
            'total_processing_time': time.time() - start_time,
            'original_text_length': len(text),
            'original_word_count': len(text.split()),
            'keywords_used': keywords,
            'summary_types_generated': list(comprehensive_result['summaries'].keys()),
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE,
            'nltk_available': NLTK_AVAILABLE
        }
        
        return comprehensive_result
        
    except Exception as e:
        log_error(f"Comprehensive summarization failed: {e}")
        comprehensive_result['error'] = str(e)
        comprehensive_result['processing_metadata']['processing_time'] = time.time() - start_time
        return comprehensive_result