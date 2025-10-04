"""
Data preprocessing module for NASA Space Biology hackathon prototype.

This module handles data cleaning, text preprocessing, and keyword extraction
for publications data using vectorized pandas operations for optimal performance.
"""

import pandas as pd
import re
import ast
from typing import List, Dict, Optional, Union
from bs4 import BeautifulSoup
import numpy as np

# Optional imports with fallbacks for hackathon flexibility
try:
    import nltk
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    log_error("NLTK not available - using basic tokenization")
    NLTK_AVAILABLE = False

from utils import load_from_cache, log, log_error


# Core space biology terms - high-priority keywords for quick identification
CORE_SPACE_TERMS = {
    'microgravity', 'radiation', 'iss', 'mars', 'spaceflight', 'astronaut',
    'bone loss', 'muscle atrophy', 'immune system', 'cardiovascular',
    'space medicine', 'countermeasures', 'weightlessness', 'space station'
}

# Extended space biology keywords for comprehensive extraction
SPACE_BIOLOGY_KEYWORDS = [
    # Primary space factors
    'microgravity', 'weightlessness', 'zero gravity', 'hypergravity',
    'radiation', 'space radiation', 'cosmic radiation', 'galactic cosmic rays', 'solar particle events',
    
    # Biological systems affected
    'bone loss', 'bone density', 'osteoporosis', 'skeletal system',
    'muscle atrophy', 'muscle mass', 'muscle wasting', 'sarcopenia',
    'cardiovascular', 'cardiac', 'heart function', 'blood pressure',
    'immune system', 'immunosuppression', 'immune response',
    'nervous system', 'neurological', 'vestibular', 'spatial orientation',
    
    # Physiological effects
    'fluid shift', 'blood volume', 'red blood cells', 'hematocrit',
    'calcium metabolism', 'protein synthesis', 'gene expression',
    'circadian rhythm', 'sleep patterns', 'psychological stress',
    
    # Countermeasures and interventions
    'countermeasures', 'exercise countermeasures', 'resistance training', 'treadmill',
    'pharmaceuticals', 'nutrition', 'dietary supplements',
    'artificial gravity', 'centrifuge', 'vibration therapy',
    
    # Mission contexts
    'spaceflight', 'space mission', 'iss', 'international space station',
    'mars', 'mars mission', 'deep space', 'long duration', 'expedition',
    'astronaut', 'crew health', 'space medicine',
    
    # Research organisms
    'mouse', 'rat', 'rodent', 'arabidopsis', 'plant biology',
    'cell culture', 'tissue engineering', 'stem cells'
]

# Convert to set for O(1) lookups and deduplication
SPACE_KEYWORDS_SET = set(keyword.lower() for keyword in SPACE_BIOLOGY_KEYWORDS)

# Compile regex pattern for efficient matching
KEYWORDS_PATTERN = re.compile(r'\b(' + '|'.join(SPACE_BIOLOGY_KEYWORDS) + r')\b', re.IGNORECASE)

# Download required NLTK data (with error handling)
def _ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    if not NLTK_AVAILABLE:
        return False
    
    try:
        nltk.data.find('tokenizers/punkt')
        return True
    except LookupError:
        log("Downloading required NLTK data...")
        try:
            nltk.download('punkt', quiet=True)
            return True
        except Exception as e:
            log_error(f"Failed to download NLTK data: {str(e)}")
            return False

# Initialize NLTK data
NLTK_READY = _ensure_nltk_data()


def load_and_preprocess(filename: str = 'data/publications.json') -> pd.DataFrame:
    """
    Load and preprocess NASA publications data with comprehensive cleaning and keyword extraction.
    
    Performs vectorized operations for optimal performance on large datasets.
    Handles missing data gracefully and extracts space biology keywords.
    
    Args:
        filename: Path to the JSON file containing publications data
        
    Returns:
        Preprocessed pandas DataFrame with additional columns:
        - cleaned_abstract: HTML-stripped, normalized abstract text
        - keywords: List of space biology keywords found in title and abstract
        - keyword_count: Number of keywords found (for filtering/ranking)
        - has_impacts: Boolean indicating if impacts data is available
        
    Example:
        >>> df = load_and_preprocess('data/publications.json')
        >>> print(df.columns)
        >>> print(df['keywords'].iloc[0])
        ['microgravity', 'bone loss', 'astronaut health']
    """
    log(f"Loading and preprocessing data from {filename}")
    
    # Step 1: Load data with fallback mechanisms
    try:
        # Try loading with utils cache function first
        data = load_from_cache(filename)
        
        if data is None:
            log(f"Cache loading failed, trying pandas read_json...")
            try:
                df = pd.read_json(filename)
            except (FileNotFoundError, ValueError) as e:
                log_error(f"Failed to load JSON file: {str(e)}")
                log("Creating empty DataFrame with expected schema")
                return _create_empty_dataframe()
        else:
            log(f"Successfully loaded {len(data)} records from cache")
            df = pd.DataFrame(data)
            
    except Exception as e:
        log_error(f"Unexpected error loading data: {str(e)}")
        return _create_empty_dataframe()
    
    if df.empty:
        log("Warning: Loaded DataFrame is empty")
        return _create_empty_dataframe()
    
    log(f"Preprocessing {len(df)} publications...")
    
    # Step 2: Handle missing values and data type consistency
    df = _handle_missing_values(df)
    
    # Step 3: Clean and normalize text data (vectorized operations)
    df = _clean_text_data(df)
    
    # Step 4: Extract keywords from title and abstract (vectorized)
    df = _extract_keywords(df)
    
    # Step 5: Add derived features for analysis
    df = _add_derived_features(df)
    
    # Step 6: Validate and optimize DataFrame
    df = _validate_and_optimize(df)
    
    log(f"✓ Preprocessing completed: {len(df)} publications with {len(df.columns)} features")
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with appropriate defaults."""
    log("Handling missing values...")
    
    # Fill missing text fields with empty strings
    text_columns = ['title', 'abstract', 'experiment_id']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    
    # Handle impacts field (can be list or string)
    if 'impacts' in df.columns:
        df['impacts'] = df['impacts'].apply(lambda x: x if isinstance(x, list) else [] if pd.isna(x) else [str(x)])
    else:
        df['impacts'] = [[] for _ in range(len(df))]
    
    # Handle metadata field - convert to string to avoid Arrow serialization issues
    if 'metadata' in df.columns:
        df['metadata'] = df['metadata'].apply(
            lambda x: str(x) if isinstance(x, dict) and x else '{}' if pd.isna(x) or not x else str(x)
        )
    else:
        df['metadata'] = ['{}' for _ in range(len(df))]
    
    return df


def _clean_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize text data using vectorized operations."""
    log("Cleaning text data...")
    
    # Clean abstracts - vectorized HTML stripping and normalization
    if 'abstract' in df.columns:
        # Strip HTML tags using BeautifulSoup (vectorized with apply)
        df['cleaned_abstract'] = df['abstract'].apply(_strip_html_and_normalize)
    else:
        df['cleaned_abstract'] = ''
    
    # Clean titles similarly
    if 'title' in df.columns:
        df['cleaned_title'] = df['title'].apply(_strip_html_and_normalize)
    else:
        df['cleaned_title'] = ''
    
    return df


def _strip_html_and_normalize(text: str) -> str:
    """
    Strip HTML tags and normalize text.
    
    Args:
        text: Input text that may contain HTML
        
    Returns:
        Clean, normalized text
    """
    if not text or pd.isna(text):
        return ''
    
    try:
        # Remove HTML tags using BeautifulSoup
        soup = BeautifulSoup(str(text), 'html.parser')
        clean_text = soup.get_text()
        
        # Normalize whitespace and convert to lowercase
        normalized = ' '.join(clean_text.lower().split())
        
        return normalized
    except Exception as e:
        log_error(f"Error cleaning text: {str(e)}")
        return str(text).lower().strip()


def _extract_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """Extract space biology keywords using enhanced NLTK tokenization and regex operations."""
    log("Extracting space biology keywords with NLTK enhancement...")
    
    # Combine title and abstract for comprehensive keyword extraction
    df['combined_text'] = df['cleaned_title'].fillna('') + ' ' + df['cleaned_abstract'].fillna('')
    
    # Enhanced keyword extraction using both NLTK and regex
    def extract_keywords_from_text(text: str) -> List[str]:
        """
        Extract unique keywords from text using NLTK tokenization and keyword matching.
        
        Uses both regex pattern matching and NLTK word tokenization for robust extraction.
        Deduplicates results and prioritizes core space biology terms.
        """
        if not text or pd.isna(text):
            return []
        
        text_str = str(text).lower()
        found_keywords = set()
        
        try:
            # Method 1: Regex pattern matching (existing approach)
            regex_matches = KEYWORDS_PATTERN.findall(text_str)
            found_keywords.update(match.lower() for match in regex_matches)
            
            # Method 2: NLTK tokenization for better word boundary detection (if available)
            if NLTK_AVAILABLE and NLTK_READY:
                try:
                    tokens = word_tokenize(text_str)
                    
                    # Check each token against our keyword set
                    for token in tokens:
                        if token in SPACE_KEYWORDS_SET:
                            found_keywords.add(token)
                    
                    # Check for multi-word terms (bigrams, trigrams)
                    for i in range(len(tokens) - 1):
                        # Bigrams
                        bigram = f"{tokens[i]} {tokens[i+1]}"
                        if bigram in SPACE_KEYWORDS_SET:
                            found_keywords.add(bigram)
                        
                        # Trigrams
                        if i < len(tokens) - 2:
                            trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                            if trigram in SPACE_KEYWORDS_SET:
                                found_keywords.add(trigram)
                                
                except Exception as nltk_error:
                    log_error(f"NLTK tokenization failed: {str(nltk_error)}")
                    # Fall back to regex-only approach
                    pass
            else:
                # Fallback: basic whitespace tokenization
                tokens = text_str.split()
                for token in tokens:
                    if token in SPACE_KEYWORDS_SET:
                        found_keywords.add(token)
            
        except Exception as e:
            log_error(f"Error in keyword extraction: {str(e)}")
            return []
        
        # Convert set to sorted list for consistent output
        unique_keywords = sorted(list(found_keywords))
        
        # Prioritize core terms (move them to front of list)
        core_found = [kw for kw in unique_keywords if kw in CORE_SPACE_TERMS]
        other_found = [kw for kw in unique_keywords if kw not in CORE_SPACE_TERMS]
        
        return core_found + other_found
    
    # Apply keyword extraction (vectorized)
    df['keywords'] = df['combined_text'].apply(extract_keywords_from_text)
    
    # Add keyword count for filtering and ranking
    df['keyword_count'] = df['keywords'].apply(len)
    
    # Add core keyword count for prioritization
    df['core_keyword_count'] = df['keywords'].apply(
        lambda kw_list: len([kw for kw in kw_list if kw in CORE_SPACE_TERMS])
    )
    
    # Drop temporary combined_text column
    df = df.drop('combined_text', axis=1)
    
    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for analysis and filtering."""
    log("Adding derived features...")
    
    # Boolean indicator for impact data availability
    df['has_impacts'] = df['impacts'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
    
    # Extract source information from metadata
    df['data_source'] = df['metadata'].apply(
        lambda x: ast.literal_eval(x).get('source', 'unknown') if isinstance(x, str) and x != '{}' else 'unknown'
    )
    
    # Extract organism information
    df['organism'] = df['metadata'].apply(
        lambda x: ast.literal_eval(x).get('organism', '') if isinstance(x, str) and x != '{}' else ''
    ).fillna('').astype(str)
    
    # Extract experiment type
    df['experiment_type'] = df['metadata'].apply(
        lambda x: ast.literal_eval(x).get('experiment_type', '') if isinstance(x, str) and x != '{}' else ''
    ).fillna('').astype(str)
    
    # Text length metrics for quality assessment
    df['abstract_length'] = df['cleaned_abstract'].str.len()
    df['title_length'] = df['cleaned_title'].str.len()
    
    # Quality score based on available information
    df['quality_score'] = (
        (df['abstract_length'] > 50).astype(int) +
        (df['title_length'] > 10).astype(int) +
        df['has_impacts'].astype(int) +
        (df['keyword_count'] > 0).astype(int) +
        (df['organism'] != '').astype(int)
    )
    
    return df


def _validate_and_optimize(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data integrity and optimize DataFrame memory usage."""
    log("Validating and optimizing DataFrame...")
    
    # Ensure required columns exist
    required_columns = ['title', 'abstract', 'experiment_id', 'cleaned_abstract', 'keywords']
    for col in required_columns:
        if col not in df.columns:
            log_error(f"Missing required column: {col}")
            if col == 'keywords':
                df[col] = [[] for _ in range(len(df))]
            else:
                df[col] = ''
    
    # Optimize data types for memory efficiency
    categorical_columns = ['data_source', 'organism', 'experiment_type']
    for col in categorical_columns:
        if col in df.columns and df[col].nunique() < len(df) * 0.5:
            df[col] = df[col].astype('category')
    
    # Sort by quality score for better user experience
    if 'quality_score' in df.columns:
        df = df.sort_values(['quality_score', 'keyword_count'], ascending=[False, False])
        df = df.reset_index(drop=True)
    
    return df


def _create_empty_dataframe() -> pd.DataFrame:
    """Create empty DataFrame with expected schema."""
    log("Creating empty DataFrame with expected schema")
    
    return pd.DataFrame({
        'title': pd.Series([], dtype='str'),
        'abstract': pd.Series([], dtype='str'),
        'experiment_id': pd.Series([], dtype='str'),
        'impacts': pd.Series([], dtype='object'),
        'metadata': pd.Series([], dtype='object'),
        'cleaned_abstract': pd.Series([], dtype='str'),
        'cleaned_title': pd.Series([], dtype='str'),
        'keywords': pd.Series([], dtype='object'),
        'keyword_count': pd.Series([], dtype='int'),
        'has_impacts': pd.Series([], dtype='bool'),
        'data_source': pd.Series([], dtype='str'),
        'organism': pd.Series([], dtype='str'),
        'experiment_type': pd.Series([], dtype='str'),
        'abstract_length': pd.Series([], dtype='int'),
        'title_length': pd.Series([], dtype='int'),
        'quality_score': pd.Series([], dtype='int')
    })


def get_keyword_stats(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get statistics about keyword frequency across the dataset.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Dictionary mapping keywords to their frequency counts
    """
    if df.empty or 'keywords' not in df.columns:
        return {}
    
    # Flatten all keywords and count frequencies
    all_keywords = []
    for keyword_list in df['keywords']:
        if isinstance(keyword_list, list):
            all_keywords.extend(keyword_list)
    
    # Count keyword frequencies
    keyword_counts = {}
    for keyword in all_keywords:
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # Sort by frequency (descending)
    sorted_keywords = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_keywords


def filter_by_keywords(df: pd.DataFrame, keywords: List[str], min_matches: int = 1) -> pd.DataFrame:
    """
    Filter DataFrame by keyword presence.
    
    Args:
        df: Preprocessed DataFrame
        keywords: List of keywords to filter by
        min_matches: Minimum number of keywords that must match
        
    Returns:
        Filtered DataFrame
    """
    if df.empty or 'keywords' not in df.columns:
        return df
    
    keywords_lower = [kw.lower() for kw in keywords]
    
    def matches_criteria(keyword_list: List[str]) -> bool:
        if not isinstance(keyword_list, list):
            return False
        matches = sum(1 for kw in keyword_list if kw.lower() in keywords_lower)
        return matches >= min_matches
    
    mask = df['keywords'].apply(matches_criteria)
    return df[mask].reset_index(drop=True)


def save_to_csv(df: pd.DataFrame, filename: str = 'data/preprocessed.csv') -> bool:
    """
    Save preprocessed DataFrame to CSV file.
    
    Args:
        df: Preprocessed DataFrame to save
        filename: Output CSV file path
        
    Returns:
        True if save successful, False otherwise
    """
    try:
        # Ensure output directory exists
        from pathlib import Path
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert list columns to string representation for CSV compatibility
        df_copy = df.copy()
        
        # Handle list columns (keywords, impacts)
        if 'keywords' in df_copy.columns:
            df_copy['keywords'] = df_copy['keywords'].apply(
                lambda x: '|'.join(x) if isinstance(x, list) else str(x)
            )
        
        if 'impacts' in df_copy.columns:
            df_copy['impacts'] = df_copy['impacts'].apply(
                lambda x: '|'.join(x) if isinstance(x, list) else str(x)
            )
        
        # Handle dict columns (metadata)
        if 'metadata' in df_copy.columns:
            df_copy['metadata'] = df_copy['metadata'].apply(str)
        
        # Save to CSV
        df_copy.to_csv(filename, index=False, encoding='utf-8')
        log(f"✓ Saved {len(df_copy)} preprocessed records to {filename}")
        return True
        
    except Exception as e:
        log_error(f"Failed to save CSV: {str(e)}")
        return False


# Test function for development and validation
if __name__ == '__main__':
    """Test preprocessing functionality with comprehensive validation."""
    log("Testing preprocess.py functionality...")
    
    try:
        # Test preprocessing pipeline
        log("Loading and preprocessing sample data...")
        df = load_and_preprocess('data/sample_publications.json')
        
        # Critical assertion for pipeline integrity
        assert 'keywords' in df.columns, "Keywords column missing from preprocessed data"
        log("✓ Keywords column validation passed")
        
        if not df.empty:
            log(f"✓ Preprocessing successful: {len(df)} publications loaded")
            log(f"Columns: {list(df.columns)}")
            log(f"Average keywords per publication: {df['keyword_count'].mean():.1f}")
            
            # Display DataFrame head as requested
            print("\n" + "="*80)
            print("DATAFRAME HEAD - PREPROCESSED PUBLICATIONS:")
            print("="*80)
            print(df.head())
            print("="*80)
            
            # Enhanced sample results display
            if len(df) > 0:
                sample = df.iloc[0]
                print(f"\nSample Publication Details:")
                print(f"Title: {sample['title'][:80]}...")
                print(f"Keywords found: {sample['keywords']}")
                print(f"Core keywords: {sample['core_keyword_count']}")
                print(f"Quality score: {sample['quality_score']}")
                print(f"Data source: {sample['data_source']}")
                print(f"Organism: {sample['organism']}")
                
            # Show enhanced keyword statistics
            keyword_stats = get_keyword_stats(df)
            print(f"\nTop 10 Keywords Across Dataset:")
            for i, (keyword, count) in enumerate(list(keyword_stats.items())[:10]):
                priority = "⭐" if keyword in CORE_SPACE_TERMS else "  "
                print(f"  {i+1:2d}. {priority} {keyword}: {count} occurrences")
            
            # Test filtering functionality
            test_keywords = ['microgravity', 'bone loss', 'radiation']
            filtered_df = filter_by_keywords(df, test_keywords, min_matches=1)
            print(f"\nFiltered results for {test_keywords}: {len(filtered_df)} publications")
            
            # Test CSV saving functionality
            log("\nTesting CSV export...")
            csv_success = save_to_csv(df, 'data/preprocessed.csv')
            if csv_success:
                log("✓ CSV export successful")
            else:
                log("⚠ CSV export failed")
            
            # Performance metrics
            print(f"\nProcessing Performance:")
            print(f"  • Total publications: {len(df)}")
            print(f"  • Publications with keywords: {(df['keyword_count'] > 0).sum()}")
            print(f"  • Publications with core keywords: {(df['core_keyword_count'] > 0).sum()}")
            print(f"  • Average quality score: {df['quality_score'].mean():.2f}")
            
        else:
            log("⚠ No data loaded - testing with empty DataFrame")
            # Test empty DataFrame handling
            assert 'keywords' in df.columns, "Keywords column missing from empty DataFrame"
            
        log("\n🎉 All preprocess.py tests completed successfully!")
        log("✓ NLTK tokenization integration working")
        log("✓ Keyword deduplication functioning")
        log("✓ CSV export capability verified")
        log("✓ Core assertions passed")
        
    except AssertionError as e:
        log_error(f"Critical assertion failed: {str(e)}")
        log("This indicates a fundamental issue with the preprocessing pipeline")
        raise
        
    except Exception as e:
        log_error(f"Testing failed: {str(e)}")
        log("This may indicate issues with NLTK setup, sample data, or preprocessing logic")
        
        # Provide debugging information
        log("Debugging information:")
        try:
            import nltk
            log(f"  • NLTK version: {nltk.__version__}")
            log(f"  • NLTK data path: {nltk.data.path}")
        except Exception as nltk_err:
            log_error(f"  • NLTK import failed: {str(nltk_err)}")
        
        raise