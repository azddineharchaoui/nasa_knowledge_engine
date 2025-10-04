"""
AI Summarization module for NASA Space Biology hackathon prototype.

This module provides BART-based text summarization for research abstracts,
with GPU acceleration support and intelligent caching for performance.
"""

import pandas as pd
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
    Load and cache BART summarization pipeline with optimal configuration.
    
    Uses facebook/bart-large-cnn model with GPU acceleration if available.
    Implements intelligent device selection and caching for performance.
    
    Returns:
        Transformers pipeline object or None if unavailable
        
    Example:
        >>> summarizer = load_summarizer()
        >>> if summarizer:
        ...     summary = summarizer("Long text...", min_length=30, max_length=100)
    """
    global _summarizer_cache
    
    # Return cached pipeline if available
    if _summarizer_cache is not None:
        log("Using cached BART summarizer")
        return _summarizer_cache
    
    if not TRANSFORMERS_AVAILABLE:
        log_error("Transformers library not available - cannot load summarizer")
        return None
    
    try:
        log("Loading BART summarization model...")
        
        # Intelligent device selection
        if torch.cuda.is_available():
            device = 0  # Use first GPU
            log(f"✓ CUDA available - using GPU device {device}")
        else:
            device = -1  # Use CPU
            log("Using CPU for summarization")
        
        # Load BART model with optimal configuration
        _summarizer_cache = pipeline(
            task='summarization',
            model='facebook/bart-large-cnn',
            device=device,
            # Performance optimizations
            model_kwargs={
                'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                'low_cpu_mem_usage': True
            }
        )
        
        log("✓ BART summarizer loaded and cached successfully")
        return _summarizer_cache
        
    except Exception as e:
        log_error(f"Failed to load BART summarizer: {str(e)}")
        log("This may be due to model download issues or insufficient memory")
        return None


def summarize_text(
    text: str,
    min_length: int = 30,
    max_length: int = 100,
    do_sample: bool = False
) -> Optional[str]:
    """
    Summarize a single text using BART model.
    
    Args:
        text: Input text to summarize
        min_length: Minimum length of summary (default: 30)
        max_length: Maximum length of summary (default: 100)
        do_sample: Whether to use sampling for generation (default: False)
        
    Returns:
        Generated summary text or None if summarization fails
        
    Example:
        >>> summary = summarize_text("Long research abstract...", max_length=80)
        >>> print(summary)
    """
    if not text or len(text.strip()) < 50:
        log("Text too short for summarization")
        return None
    
    summarizer = load_summarizer()
    if summarizer is None:
        return _fallback_summarization(text, max_length)
    
    try:
        log(f"Summarizing text ({len(text)} chars)...")
        
        # Generate summary with optimized parameters
        result = summarizer(
            text,
            min_length=min_length,
            max_length=max_length,
            do_sample=do_sample,
            truncation=True,
            # Additional optimization parameters
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        summary = result[0]['summary_text']
        log(f"✓ Summary generated ({len(summary)} chars)")
        return summary
        
    except Exception as e:
        log_error(f"BART summarization failed: {str(e)}")
        return _fallback_summarization(text, max_length)


def summarize_batch(
    texts: List[str],
    min_length: int = 30,
    max_length: int = 100,
    do_sample: bool = False,
    batch_size: int = 4
) -> List[Optional[str]]:
    """
    Summarize multiple texts in batches for efficiency.
    
    Args:
        texts: List of texts to summarize
        min_length: Minimum length of summaries
        max_length: Maximum length of summaries
        do_sample: Whether to use sampling
        batch_size: Number of texts to process simultaneously
        
    Returns:
        List of summaries (None for failed summarizations)
        
    Example:
        >>> abstracts = ["Abstract 1...", "Abstract 2..."]
        >>> summaries = summarize_batch(abstracts, max_length=80)
    """
    if not texts:
        return []
    
    summarizer = load_summarizer()
    if summarizer is None:
        log("Using fallback summarization for batch processing")
        return [_fallback_summarization(text, max_length) for text in texts]
    
    summaries = []
    
    try:
        log(f"Batch summarizing {len(texts)} texts...")
        
        # Process in batches for memory efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Filter valid texts
            valid_batch = [text for text in batch if text and len(text.strip()) >= 50]
            
            if not valid_batch:
                # Add None for each invalid text in batch
                summaries.extend([None] * len(batch))
                continue
            
            try:
                # Batch summarization
                results = summarizer(
                    valid_batch,
                    min_length=min_length,
                    max_length=max_length,
                    do_sample=do_sample,
                    truncation=True,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                
                # Extract summary texts
                batch_summaries = [result['summary_text'] for result in results]
                
                # Map back to original batch (handling skipped texts)
                valid_idx = 0
                for text in batch:
                    if text and len(text.strip()) >= 50:
                        summaries.append(batch_summaries[valid_idx])
                        valid_idx += 1
                    else:
                        summaries.append(None)
                        
            except Exception as batch_error:
                log_error(f"Batch summarization failed: {str(batch_error)}")
                # Fallback to individual processing for this batch
                for text in batch:
                    summaries.append(_fallback_summarization(text, max_length))
        
        log(f"✓ Batch summarization completed: {sum(1 for s in summaries if s)} successful")
        return summaries
        
    except Exception as e:
        log_error(f"Batch summarization error: {str(e)}")
        return [_fallback_summarization(text, max_length) for text in texts]


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


def summarize_abstracts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary column to DataFrame by batch processing abstracts with impact focus.
    
    Processes df['abstract'] column using BART summarizer with space biology impact
    prompting. Handles empty/missing abstracts gracefully and uses batch processing
    for optimal performance during hackathon demos.
    
    Args:
        df: DataFrame containing 'abstract' column
        
    Returns:
        DataFrame with added 'summary' column containing impact-focused summaries
        
    Example:
        >>> df = pd.DataFrame({'abstract': ['Long abstract about bone loss...']})
        >>> df_with_summaries = summarize_abstracts(df)
        >>> print(df_with_summaries['summary'].iloc[0])
    """
    log(f"Summarizing abstracts for {len(df)} publications...")
    
    if df.empty:
        log("Empty DataFrame - returning with empty summary column")
        df['summary'] = []
        return df
    
    # Ensure abstract column exists
    if 'abstract' not in df.columns:
        log_error("No 'abstract' column found in DataFrame")
        df['summary'] = ['No abstract available'] * len(df)
        return df
    
    # Get summarizer
    summarizer = load_summarizer()
    
    # Prepare abstracts for processing
    abstracts = df['abstract'].fillna('').astype(str)
    
    # Create impact-focused prompts for better summarization
    prompted_abstracts = []
    for abstract in abstracts:
        if not abstract or len(abstract.strip()) < 50:
            prompted_abstracts.append('')
        else:
            # Add impact-focused prompt to guide summarization
            impact_prompt = f"Summarize the space biology impacts and key findings: {abstract}"
            prompted_abstracts.append(impact_prompt)
    
    if summarizer is not None:
        # Use BART model with batch processing
        summaries = _batch_process_abstracts_with_bart(
            prompted_abstracts, 
            summarizer,
            batch_size=4,
            max_length=100
        )
    else:
        # Use fallback method
        log("Using fallback summarization for abstracts")
        summaries = _batch_process_abstracts_fallback(
            abstracts,  # Use original abstracts for fallback
            max_length=100
        )
    
    # Add summaries to DataFrame
    df = df.copy()
    df['summary'] = summaries
    
    # Log statistics
    successful_summaries = sum(1 for s in summaries if s and s.strip() and s != 'No summary available')
    log(f"✓ Abstract summarization completed: {successful_summaries}/{len(df)} successful")
    
    return df


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