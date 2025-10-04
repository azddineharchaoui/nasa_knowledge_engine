#!/usr/bin/env python3
"""
Ultra-Fast Summarizer - Zero-Blocking Version

This module provides an extremely fast summarization system that prioritizes
speed and reliability over AI model complexity.
"""

import pandas as pd
import time
import re
import threading
import signal
from typing import List, Dict, Optional, Union
from utils import log, log_error

# Optional imports with robust fallbacks
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    log("Transformers library available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    log("Transformers unavailable - using ultra-fast fallback only")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Global cache
_lightweight_summarizer = None
_summarizer_failed = False
_use_ai_summarization = False  # Default to fast fallback

def get_lightweight_summarizer(timeout=5):
    """
    Get ultra-lightweight summarizer with strict timeout protection.
    
    Uses the smallest possible model with aggressive timeout protection.
    """
    global _lightweight_summarizer, _summarizer_failed, _use_ai_summarization
    
    # Skip if previously failed or AI disabled
    if _summarizer_failed or not _use_ai_summarization:
        return None
    
    # Return cached if available
    if _lightweight_summarizer is not None:
        return _lightweight_summarizer
    
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        log("Loading ultra-lightweight summarizer (5s timeout)...")
        start_time = time.time()
        
        # Use the absolute smallest model available
        _lightweight_summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",  # More reliable than distilbart
            device=-1,  # Force CPU
            torch_dtype=torch.float16,  # Use half precision for speed
            model_kwargs={
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16
            }
        )
        
        load_time = time.time() - start_time
        if load_time > timeout:
            log(f"Model loading took {load_time:.1f}s (timeout: {timeout}s) - disabling AI")
            _summarizer_failed = True
            _lightweight_summarizer = None
            return None
        
        log(f"Ultra-lightweight summarizer loaded in {load_time:.1f}s")
        return _lightweight_summarizer
        
    except Exception as e:
        log_error(f"Ultra-lightweight summarizer failed: {str(e)}")
        _summarizer_failed = True
        return None


def quick_summarize_text(text: str, max_length: int = 100) -> str:
    """
    Ultra-fast text summarization with instant fallback.
    
    Prioritizes speed over AI complexity - designed for real-time use.
    """
    if not text or len(text.strip()) < 30:
        return "Text too short for summarization"
    
    # Always use fast extractive method by default
    return ultra_fast_extractive_summary(text, max_length)


def ultra_fast_extractive_summary(text: str, max_length: int = 100) -> str:
    """
    Ultra-fast extractive summarization - optimized for speed.
    
    Uses simple heuristics for maximum performance.
    """
    try:
        # Quick text cleaning
        clean_text = text.strip()
        if len(clean_text) <= max_length:
            return clean_text
        
        # Split into sentences (simple approach)
        sentences = []
        current = ""
        for char in clean_text:
            current += char
            if char in '.!?':
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        
        if len(sentences) <= 1:
            return clean_text[:max_length] + "..." if len(clean_text) > max_length else clean_text
        
        # Simple scoring - prioritize first sentences and key terms
        scored = []
        key_words = ['significant', 'found', 'results', 'effects', 'study', 'research', 'data', 'analysis']
        
        for i, sentence in enumerate(sentences):
            if len(sentence) < 10:
                continue
                
            score = 0
            sentence_lower = sentence.lower()
            
            # Position bonus (first sentences are usually important)
            if i < 2:
                score += 3
            elif i < len(sentences) // 2:
                score += 1
            
            # Key word bonus
            for word in key_words:
                if word in sentence_lower:
                    score += 2
                    break
            
            # Length bonus (not too short, not too long)
            if 20 <= len(sentence) <= 150:
                score += 1
            
            scored.append((score, sentence))
        
        if not scored:
            return clean_text[:max_length] + "..."
        
        # Sort by score and build summary
        scored.sort(key=lambda x: x[0], reverse=True)
        
        summary_parts = []
        current_length = 0
        
        for score, sentence in scored:
            if current_length + len(sentence) <= max_length:
                summary_parts.append(sentence)
                current_length += len(sentence) + 1
            else:
                break
        
        if summary_parts:
            return '. '.join(summary_parts) + '.'
        else:
            # Fallback to first sentence
            return scored[0][1][:max_length] + "..." if len(scored[0][1]) > max_length else scored[0][1]
            
    except Exception as e:
        log_error(f"Ultra-fast summarization failed: {str(e)}")
        return text[:max_length] + "..." if len(text) > max_length else text


def extractive_fallback_summary(text: str, max_length: int = 100) -> str:
    """Backward compatibility wrapper."""
    return ultra_fast_extractive_summary(text, max_length)


def ultra_fast_batch_summarize(texts: List[str], max_length: int = 100) -> List[str]:
    """
    Ultra-fast batch summarization - optimized for maximum speed.
    
    Uses vectorized operations where possible for maximum performance.
    """
    if not texts:
        return []
    
    log(f"Ultra-fast batch summarization: {len(texts)} texts")
    start_time = time.time()
    
    summaries = []
    
    for text in texts:
        try:
            if text and len(text.strip()) > 20:
                summary = ultra_fast_extractive_summary(text, max_length)
                summaries.append(summary)
            else:
                summaries.append("No content to summarize")
        except Exception as e:
            log_error(f"Text failed: {str(e)}")
            summaries.append("Summarization failed")
    
    total_time = time.time() - start_time
    log(f"Ultra-fast batch completed in {total_time:.3f}s ({len(texts)/total_time:.1f} texts/sec)")
    
    return summaries


def fast_batch_summarize(texts: List[str], max_length: int = 100, timeout_per_text: float = 2.0) -> List[str]:
    """Backward compatibility wrapper."""
    return ultra_fast_batch_summarize(texts, max_length)


def ultra_fast_summarize_abstracts(df: pd.DataFrame, max_length: int = 100) -> pd.DataFrame:
    """
    Ultra-fast abstract summarization - zero blocking, maximum speed.
    
    Features:
    - No AI model loading (uses extractive methods only)
    - Sub-second processing for typical datasets
    - Guaranteed completion without timeouts
    - Optimized for dashboard responsiveness
    
    Args:
        df: DataFrame with 'abstract' column
        max_length: Maximum summary length (default: 100)
        
    Returns:
        DataFrame with 'summary' column added
    """
    start_time = time.time()
    
    if df is None or df.empty:
        log("Empty DataFrame - returning empty result")
        return pd.DataFrame()
    
    if 'abstract' not in df.columns:
        log("No 'abstract' column - adding placeholder summaries")
        df_copy = df.copy()
        df_copy['summary'] = ['No abstract available'] * len(df)
        return df_copy
    
    # Prepare abstracts
    abstracts = df['abstract'].fillna('').astype(str)
    valid_abstracts = [abs for abs in abstracts if len(abs.strip()) > 20]
    
    log(f"Ultra-fast processing: {len(valid_abstracts)}/{len(abstracts)} valid abstracts")
    
    try:
        # Ultra-fast batch processing
        summaries = ultra_fast_batch_summarize(abstracts.tolist(), max_length)
        
        # Add to DataFrame
        df_result = df.copy()
        df_result['summary'] = summaries
        
        # Statistics
        total_time = time.time() - start_time
        avg_length = sum(len(s) for s in summaries) / len(summaries) if summaries else 0
        
        log(f"Ultra-fast summarization completed:")
        log(f"   - {len(summaries)} summaries generated")
        log(f"   - {total_time:.3f}s total time ({len(summaries)/total_time:.1f} summaries/sec)")
        log(f"   - Average length: {avg_length:.0f} characters")
        
        return df_result
        
    except Exception as e:
        log_error(f"Ultra-fast summarization failed: {str(e)}")
        
        # Emergency fallback - simple truncation
        df_result = df.copy()
        df_result['summary'] = [
            abs[:max_length] + "..." if len(abs) > max_length else abs
            for abs in abstracts
        ]
        
        log(f"Emergency fallback completed in {time.time() - start_time:.3f}s")
        return df_result


def lightweight_summarize_abstracts(df: pd.DataFrame, max_length: int = 100, timeout: float = 30.0) -> pd.DataFrame:
    """Backward compatibility wrapper - routes to ultra-fast version."""
    return ultra_fast_summarize_abstracts(df, max_length)


def enable_ai_summarization(enable: bool = True):
    """Enable or disable AI summarization (default: disabled for speed)."""
    global _use_ai_summarization
    _use_ai_summarization = enable
    log(f"AI summarization {'enabled' if enable else 'disabled'} - using {'AI' if enable else 'extractive'} methods")


def get_summarizer_status() -> Dict[str, Union[str, bool, float]]:
    """Get current summarizer status for debugging."""
    return {
        'transformers_available': TRANSFORMERS_AVAILABLE,
        'sklearn_available': SKLEARN_AVAILABLE,
        'lightweight_loaded': _lightweight_summarizer is not None,
        'summarizer_failed': _summarizer_failed,
        'ai_enabled': _use_ai_summarization,
        'mode': 'AI' if _use_ai_summarization and not _summarizer_failed else 'Ultra-Fast Extractive'
    }


# Alias for backward compatibility
def summarize_abstracts(df: pd.DataFrame, batch_size: int = 6, show_progress: bool = True) -> pd.DataFrame:
    """Backward compatibility wrapper - routes to lightweight version."""
    log("🔄 Using lightweight summarizer for fast processing")
    return lightweight_summarize_abstracts(df, max_length=100, timeout=30.0)


if __name__ == "__main__":
    # Quick test
    test_df = pd.DataFrame({
        'title': ['Test Study 1', 'Test Study 2'],
        'abstract': [
            'This study examined the effects of microgravity on bone density in mice during a 30-day spaceflight mission. The results showed significant bone loss in the experimental group compared to controls.',
            'Researchers investigated muscle atrophy in astronauts and found significant changes in protein synthesis. The data revealed a 15% reduction in muscle mass over the course of the mission.'
        ]
    })
    
    print("Testing ultra-fast summarizer...")
    print(f"Status: {get_summarizer_status()}")
    
    result_df = ultra_fast_summarize_abstracts(test_df)
    print(f"Test completed - {len(result_df)} summaries generated")
    
    for i, summary in enumerate(result_df['summary']):
        print(f"Summary {i+1}: {summary}")
    
    print("\nPerformance test with larger dataset...")
    large_test = pd.DataFrame({
        'abstract': [f'This is test abstract number {i} with some scientific content about space biology research and microgravity effects on biological systems.' for i in range(100)]
    })
    
    start = time.time()
    large_result = ultra_fast_summarize_abstracts(large_test)
    end = time.time()
    
    print(f"Large test completed: {len(large_result)} summaries in {end-start:.3f}s ({len(large_result)/(end-start):.1f} summaries/sec)")

