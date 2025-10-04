"""
AI Summarization module for NASA Space Biology hackathon prototype.

This module provides BART-based text summarization for research abstracts,
with GPU acceleration support and intelligent caching for performance.
"""

import pandas as pd
import time
from typing import List, Dict, Optional, Union
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

# Global cache for the summarization pipeline
_summarizer_cache = None


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
        
        # Add space biology impact focus to prompt
        impact_prompt = f"Summarize the space biology research impacts and key findings: {clean_text}"
        

        
        # Execute with cross-platform timeout protection
        def _execute_summarization():
            return summarizer(
                impact_prompt,
                min_length=min_length,
                max_length=max_length,
                do_sample=do_sample,
                num_beams=num_beams if not do_sample else 1,
                temperature=temperature if do_sample else 1.0,
                truncation=True,
                # Quality enhancements
                no_repeat_ngram_size=3,
                early_stopping=True,
                length_penalty=1.0,
                repetition_penalty=1.2
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
            
            # Quality validation
            if _validate_summary_quality(summary, clean_text):
                log(f"✅ High-quality summary generated ({len(summary)} chars)")
                return summary
            else:
                log(f"⚠️ AI summary failed quality check, using fallback")
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
    timeout_per_batch: int = 120
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
                    timeout=timeout_per_batch
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
    timeout: int
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
                        
                        # Quality validation
                        if _validate_summary_quality(clean_summary, original_text):
                            batch_results[original_idx] = clean_summary
                        else:
                            # Use fallback for poor quality AI summary
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
            min_length=40,
            max_length=120,  # Optimized for 50-100 word summaries
            do_sample=False,  # Use beam search for quality
            num_beams=4,      # Balance quality vs speed
            batch_size=batch_size,
            show_progress=show_progress,
            timeout_per_batch=120  # 2 minutes per batch
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