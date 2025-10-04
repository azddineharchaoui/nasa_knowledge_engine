"""
Utility functions and constants for NASA Space Biology hackathon prototype.

This module provides helper functions for data fetching, caching, and configuration
for the NASA research data analysis and knowledge graph system.
"""

import requests
import pandas as pd
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# API Endpoints and Configuration Constants
GENELAB_SEARCH_URL = 'https://genelab-data.ndc.nasa.gov/genelab/data/search?term={query}'
NTRS_SEARCH_URL = 'https://ntrs.nasa.gov/search?q={query}'

# Default cache directory
DEFAULT_CACHE_DIR = Path('data')


def cache_to_json(data: Union[Dict, List, Any], filename: str = 'data/publications.json') -> bool:
    """
    Cache data to JSON file with automatic directory creation.
    
    This function is idempotent and optimized for fast execution. It creates
    the necessary directory structure if it doesn't exist and safely writes
    the data to the specified JSON file.
    
    Args:
        data: The data to cache (dict, list, or any JSON-serializable object)
        filename: Path to the output JSON file (default: 'data/publications.json')
        
    Returns:
        bool: True if caching was successful, False otherwise
        
    Example:
        >>> sample_data = {'experiments': [{'id': 1, 'title': 'Test'}]}
        >>> cache_to_json(sample_data, 'data/experiments.json')
        True
    """
    try:
        # Convert string path to Path object for consistent handling
        file_path = Path(filename)
        
        # Create parent directory if it doesn't exist (idempotent operation)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write data to JSON file with proper formatting
        with file_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"✓ Data successfully cached to {file_path}")
        return True
        
    except (IOError, TypeError, ValueError) as e:
        print(f"✗ Error caching data to {filename}: {str(e)}")
        return False


def load_from_json(filename: str = 'data/publications.json') -> Optional[Union[Dict, List]]:
    """
    Load data from JSON cache file.
    
    Args:
        filename: Path to the JSON file to load
        
    Returns:
        The loaded data if successful, None if file doesn't exist or error occurs
        
    Example:
        >>> data = load_from_json('data/experiments.json')
        >>> if data:
        ...     print(f"Loaded {len(data)} items")
    """
    try:
        file_path = Path(filename)
        
        if not file_path.exists():
            print(f"Cache file {filename} not found")
            return None
            
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"✓ Data loaded from {file_path}")
        return data
        
    except (IOError, json.JSONDecodeError) as e:
        print(f"✗ Error loading data from {filename}: {str(e)}")
        return None


def ensure_cache_dir(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR) -> Path:
    """
    Ensure cache directory exists and return Path object.
    
    Args:
        cache_dir: Directory path for caching (default: 'data')
        
    Returns:
        Path object for the cache directory
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def get_api_urls() -> Dict[str, str]:
    """
    Get dictionary of all configured API URLs.
    
    Returns:
        Dictionary mapping API names to their URLs
        
    Example:
        >>> urls = get_api_urls()
        >>> print(urls['genelab'])
    """
    return {
        'genelab': GENELAB_SEARCH_URL,
        'ntrs': NTRS_SEARCH_URL
    }


def format_search_url(api_name: str, query: str) -> Optional[str]:
    """
    Format search URL for the specified API with the given query.
    
    Args:
        api_name: Name of the API ('genelab', 'ntrs')
        query: Search query term
        
    Returns:
        Formatted URL string or None if API name is invalid
        
    Example:
        >>> url = format_search_url('genelab', 'microgravity')
        >>> print(url)
        https://genelab-data.ndc.nasa.gov/genelab/data/search?term=microgravity
    """
    urls = get_api_urls()
    
    if api_name not in urls:
        print(f"✗ Unknown API name: {api_name}. Available APIs: {list(urls.keys())}")
        return None
    
    try:
        return urls[api_name].format(query=query)
    except KeyError as e:
        print(f"✗ Error formatting URL for {api_name}: {str(e)}")
        return None


# Configuration for logging and debugging
VERBOSE_LOGGING = True

def load_from_cache(filename: str) -> Optional[Union[Dict, List]]:
    """
    Load data from cache file, return None if not found.
    
    Simple wrapper around load_from_json for consistent API naming.
    Handles missing files gracefully without verbose error messages.
    
    Args:
        filename: Path to the cache file to load
        
    Returns:
        Loaded data or None if file doesn't exist or can't be read
        
    Example:
        >>> data = load_from_cache('data/experiments.json')
        >>> if data is not None:
        ...     print("Cache hit!")
    """
    try:
        file_path = Path(filename)
        if not file_path.exists():
            return None
            
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
            
    except (IOError, json.JSONDecodeError):
        return None


def log(msg: str) -> None:
    """
    Simple logger for prototype development.
    
    Args:
        msg: Message to log
        
    Example:
        >>> log("Processing data batch 1/5")
        [INFO] Processing data batch 1/5
    """
    print(f'[INFO] {msg}')


def log_info(message: str) -> None:
    """Simple logging function for development and debugging."""
    if VERBOSE_LOGGING:
        print(f"[INFO] {message}")


def log_error(message: str) -> None:
    """Simple error logging function."""
    print(f"[ERROR] {message}")


# Test and validation functions for prototype reliability
if __name__ == '__main__':
    """
    Test suite for utils.py functionality.
    Ensures reliability for repeated prototyping runs.
    """
    log("Starting utils.py reliability tests...")
    
    # Test 1: Cache and load cycle
    test_data = {'test': 1, 'items': ['a', 'b', 'c'], 'nested': {'key': 'value'}}
    test_filename = 'data/test_cache.json'
    
    log("Test 1: Cache and load functionality")
    
    # Cache the test data
    cache_success = cache_to_json(test_data, test_filename)
    assert cache_success, "Failed to cache test data"
    
    # Load using both functions
    loaded_data_1 = load_from_json(test_filename)
    loaded_data_2 = load_from_cache(test_filename)
    
    # Verify equality
    assert loaded_data_1 == test_data, f"load_from_json failed: {loaded_data_1} != {test_data}"
    assert loaded_data_2 == test_data, f"load_from_cache failed: {loaded_data_2} != {test_data}"
    assert loaded_data_1 == loaded_data_2, "Functions returned different results"
    
    log("✓ Cache/load cycle test passed")
    
    # Test 2: Missing file handling
    log("Test 2: Missing file handling")
    missing_data = load_from_cache('data/nonexistent.json')
    assert missing_data is None, "Should return None for missing files"
    log("✓ Missing file test passed")
    
    # Test 3: URL formatting
    log("Test 3: URL formatting")
    genelab_url = format_search_url('genelab', 'microgravity')
    expected_url = 'https://genelab-data.ndc.nasa.gov/genelab/data/search?term=microgravity'
    assert genelab_url == expected_url, f"URL formatting failed: {genelab_url}"
    log("✓ URL formatting test passed")
    
    # Test 4: Repeated runs (idempotent operations)
    log("Test 4: Idempotent operations")
    for i in range(3):
        cache_success = cache_to_json(test_data, test_filename)
        loaded_data = load_from_cache(test_filename)
        assert cache_success and loaded_data == test_data, f"Failed on iteration {i+1}"
    log("✓ Repeated operations test passed")
    
    # Cleanup
    test_file_path = Path(test_filename)
    if test_file_path.exists():
        test_file_path.unlink()
        log("✓ Test cleanup completed")
    
    log("🎉 All reliability tests passed! utils.py is ready for prototyping.")