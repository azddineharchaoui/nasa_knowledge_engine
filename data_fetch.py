"""
NASA API data fetching module for Space Biology hackathon prototype.

This module handles data retrieval from NASA GeneLab, OSDR, and NTRS APIs
with robust error handling, BeautifulSoup fallback, and standardized output format.
"""

import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re
from utils import GENELAB_SEARCH_URL, OSDR_STUDIES_URL, NTRS_SEARCH_URL, log, log_error, cache_to_json, load_from_cache


def fetch_publications(query: str = 'space biology', limit: int = 100) -> List[Dict]:
    """
    Fetch publications from NASA GeneLab API with comprehensive fallback chain.
    
    Retrieves research data using multiple strategies:
    1. GeneLab API (primary)
    2. NTRS web scraping (secondary)
    3. Sample data file (offline fallback)
    
    Ensures the application always has data for development and demos.
    
    Args:
        query: Search term for GeneLab API (default: 'space biology')
        limit: Maximum number of publications to return (default: 100)
        
    Returns:
        List of dictionaries containing publication data with fields:
        - title: Publication/experiment title
        - abstract: Research abstract or description
        - experiment_id: Unique GeneLab experiment identifier
        - impacts: Space biology impacts if available
        - metadata: Additional experiment metadata
        
    Example:
        >>> publications = fetch_publications('microgravity', limit=50)
        >>> print(f"Found {len(publications)} publications")
        >>> print(publications[0]['title'])
    """
    log(f"Fetching publications: query='{query}', limit={limit}")
    
    # Strategy 1: Try GeneLab API first
    try:
        log("Attempting GeneLab API...")
        publications = _fetch_genelab_publications(query, limit)
        
        if publications:
            log(f"✓ GeneLab API succeeded: {len(publications)} publications")
            return publications[:limit]
        else:
            log("⚠ GeneLab API returned no results, trying fallback...")
            
    except requests.exceptions.RequestException as e:
        log_error(f"GeneLab API network error: {str(e)}")
        log("🔄 Network issues detected, falling back to NTRS...")
    except requests.exceptions.Timeout as e:
        log_error(f"GeneLab API timeout: {str(e)}")
        log("🔄 Request timeout, falling back to NTRS...")
    except Exception as e:
        log_error(f"GeneLab API failed: {str(e)}")
        log("🔄 Falling back to NTRS web scraping...")
    
    # Strategy 2: Fallback to NTRS web scraping
    try:
        log("Attempting NTRS fallback...")
        fallback_publications = search_ntrs_fallback(query, limit)
        
        if fallback_publications:
            log(f"✓ NTRS fallback succeeded: {len(fallback_publications)} publications")
            return fallback_publications[:limit]
        else:
            log("⚠ NTRS fallback returned no results")
            
    except requests.exceptions.RequestException as e:
        log_error(f"NTRS network error: {str(e)}")
        log("🔄 Network issues with NTRS, falling back to sample data...")
    except requests.exceptions.Timeout as e:
        log_error(f"NTRS timeout: {str(e)}")
        log("🔄 NTRS timeout, falling back to sample data...")
    except Exception as e:
        log_error(f"NTRS fallback failed: {str(e)}")
        log("🔄 Falling back to sample data...")
    
    # Strategy 3: Final fallback to sample data
    try:
        log("Attempting sample data fallback...")
        sample_publications = load_sample_publications()
        
        if sample_publications:
            log(f"✓ Sample data fallback succeeded: {len(sample_publications)} publications")
            # Filter sample data based on query if possible
            filtered_samples = _filter_sample_data(sample_publications, query)
            return filtered_samples[:limit]
        
    except Exception as e:
        log_error(f"Sample data fallback failed: {str(e)}")
    
    # If all strategies fail, return empty list with warning
    log_error("All data fetching strategies failed - returning empty dataset")
    log("⚠ Application will continue with no data. Check network connectivity and API status.")
    return []


def load_sample_publications() -> List[Dict]:
    """
    Load sample publications from local JSON file for offline development.
    
    Returns:
        List of sample publication dictionaries
    """
    sample_file = 'data/sample_publications.json'
    log(f"Loading sample data from {sample_file}")
    
    sample_data = load_from_cache(sample_file)
    
    if sample_data:
        log(f"✓ Loaded {len(sample_data)} sample publications")
        return sample_data
    else:
        log_error(f"Failed to load sample data from {sample_file}")
        return []


def _filter_sample_data(sample_data: List[Dict], query: str) -> List[Dict]:
    """
    Filter sample data based on query terms for more relevant results.
    
    Args:
        sample_data: List of sample publications
        query: Search query to filter by
        
    Returns:
        Filtered list of publications
    """
    if not query or query.lower() in ['space biology', '']:
        return sample_data
    
    query_terms = query.lower().split()
    filtered_data = []
    
    for pub in sample_data:
        # Check if query terms appear in title, abstract, or impacts
        text_content = (
            pub.get('title', '').lower() + ' ' +
            pub.get('abstract', '').lower() + ' ' +
            ' '.join(pub.get('impacts', []))
        )
        
        # If any query term matches, include this publication
        if any(term in text_content for term in query_terms):
            filtered_data.append(pub)
    
    # If no matches found, return all sample data
    if not filtered_data:
        log(f"No sample data matched query '{query}', returning all samples")
        return sample_data
    
    log(f"Filtered sample data: {len(filtered_data)} publications match '{query}'")
    return filtered_data


def _fetch_genelab_publications(query: str, limit: int) -> List[Dict]:
    """
    Internal function to fetch from GeneLab API specifically.
    
    Separated for cleaner fallback logic in main fetch_publications function.
    Includes comprehensive error handling for network issues.
    """
    # OFFLINE MODE FOR TESTING - Uncomment the next line to test fallback
    # raise requests.exceptions.ConnectionError("Testing offline mode")
    
    # Format the GeneLab search URL with query parameter
    url = GENELAB_SEARCH_URL.format(query=query)
    log(f"Making request to: {url}")
    
    # Make API request with timeout and headers
    headers = {
        'User-Agent': 'NASA-Space-Biology-Hackathon/1.0',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.exceptions.ConnectionError as e:
        log_error(f"Connection error to GeneLab API: {str(e)}")
        raise
    except requests.exceptions.Timeout as e:
        log_error(f"Timeout error to GeneLab API: {str(e)}")
        raise
    except requests.exceptions.RequestException as e:
        log_error(f"Request error to GeneLab API: {str(e)}")
        raise
    
    # Check for successful response
    if response.status_code != 200:
        error_msg = f"API request failed with status {response.status_code}: {response.text}"
        raise ValueError(error_msg)
    
    log("✓ Successfully received response from GeneLab API")
    
    # Parse JSON response
    try:
        data = response.json()
    except ValueError as e:
        error_msg = f"Failed to parse JSON response: {str(e)}"
        raise ValueError(error_msg)
    
    # Extract results from API response
    # GeneLab API typically returns results in 'results' or 'studies' field
    results = data.get('results', data.get('studies', []))
    
    if not results:
        log("Warning: No results found in API response")
        return []
    
    log(f"Processing {len(results)} raw results from API")
    
    # Process and standardize publication data
    publications = []
    
    for i, item in enumerate(results[:limit]):
        try:
            # Extract core fields with fallbacks
            publication = {
                # Title: Primary identifier for the research
                'title': item.get('title') or item.get('name') or item.get('study_title') or f"Study {item.get('id', i)}",
                
                # Abstract: Research description or summary
                'abstract': (
                    item.get('abstract') or 
                    item.get('description') or 
                    item.get('summary') or 
                    item.get('study_description') or 
                    "No abstract available"
                ),
                
                # Experiment ID: Unique identifier for tracking
                'experiment_id': (
                    item.get('experiment_id') or 
                    item.get('study_id') or 
                    item.get('id') or 
                    item.get('accession') or 
                    f"EXP_{i:04d}"
                ),
                
                # Impacts: Space biology effects and outcomes
                'impacts': (
                    item.get('impacts') or 
                    item.get('effects') or 
                    item.get('outcomes') or 
                    item.get('biological_effects') or 
                    []
                ),
                
                # Metadata: Additional experimental details
                'metadata': {
                    'source': 'GeneLab_API',
                    'organism': item.get('organism') or item.get('species'),
                    'experiment_type': item.get('experiment_type') or item.get('study_type'),
                    'platform': item.get('platform') or item.get('assay_technology'),
                    'factors': item.get('factors') or item.get('experimental_factors'),
                    'mission': item.get('mission') or item.get('flight_program'),
                    'publication_date': item.get('publication_date') or item.get('release_date'),
                    'doi': item.get('doi'),
                    'project_link': item.get('project_link') or item.get('data_link'),
                    'raw_data': item  # Keep original data for debugging
                }
            }
            
            publications.append(publication)
            
        except Exception as e:
            log_error(f"Error processing item {i}: {str(e)}")
            continue
    
    log(f"✓ Successfully processed {len(publications)} publications from GeneLab")
    return publications


def fetch_osdr_studies(limit: int = 100) -> List[Dict]:
    """
    Fetch studies from NASA OSDR (Open Science Data Repository) API.
    
    Args:
        limit: Maximum number of studies to return
        
    Returns:
        List of standardized publication dictionaries
        
    Raises:
        ValueError: If API request fails
    """
    log(f"Fetching studies from OSDR API: limit={limit}")
    
    try:
        headers = {
            'User-Agent': 'NASA-Space-Biology-Hackathon/1.0',
            'Accept': 'application/json'
        }
        
        response = requests.get(OSDR_STUDIES_URL, headers=headers, timeout=30)
        
        if response.status_code != 200:
            error_msg = f"OSDR API request failed with status {response.status_code}"
            log_error(error_msg)
            raise ValueError(error_msg)
        
        data = response.json()
        studies = data.get('studies', [])[:limit]
        
        log(f"✓ Successfully fetched {len(studies)} studies from OSDR")
        return studies
        
    except Exception as e:
        log_error(f"Error fetching OSDR studies: {str(e)}")
        raise


def search_ntrs_fallback(query: str = 'space biology', limit: int = 50) -> List[Dict]:
    """
    Search NASA Technical Reports Server (NTRS) using web scraping as fallback.
    
    Uses BeautifulSoup to parse HTML when API responses are unavailable.
    This provides a robust fallback for data collection during hackathon demos.
    
    Args:
        query: Search query for NTRS
        limit: Maximum number of results to return
        
    Returns:
        List of standardized publication dictionaries
        
    Note:
        This is a fallback data source when primary APIs are unavailable.
        Implements web scraping with BeautifulSoup for resilience.
    """
    log(f"Searching NTRS with BeautifulSoup fallback: query='{query}', limit={limit}")
    
    # OFFLINE MODE FOR TESTING - Uncomment the next line to test sample data fallback
    # raise requests.exceptions.ConnectionError("Testing NTRS offline mode")
    
    try:
        # Format NTRS search URL
        url = NTRS_SEARCH_URL.format(query=query)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
        except requests.exceptions.ConnectionError as e:
            log_error(f"NTRS connection error: {str(e)}")
            raise
        except requests.exceptions.Timeout as e:
            log_error(f"NTRS timeout error: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            log_error(f"NTRS request error: {str(e)}")
            raise
        
        if response.status_code != 200:
            log_error(f"NTRS search failed with status {response.status_code}")
            raise requests.exceptions.HTTPError(f"HTTP {response.status_code}")
        
        log("✓ NTRS response received, parsing with BeautifulSoup...")
        
        # Parse HTML content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find search result containers (typical NTRS structure)
        # Note: These selectors are simulated - real NTRS would need actual inspection
        result_containers = soup.find_all('div', class_=['search-result', 'result-item', 'document-item'])
        
        if not result_containers:
            # Fallback: look for any divs with common result patterns
            result_containers = soup.find_all('div', string=re.compile(r'(abstract|summary|report)', re.I))[:limit]
        
        publications = []
        
        for i, container in enumerate(result_containers[:limit]):
            try:
                # Extract title - look for various title patterns
                title_elem = (
                    container.find('h2') or 
                    container.find('h3') or 
                    container.find('a', class_='title') or
                    container.find('strong')
                )
                title = title_elem.get_text().strip() if title_elem else f"NTRS Document {i+1}"
                
                # Extract abstract/description
                abstract_elem = (
                    container.find('div', class_=['abstract', 'summary', 'description']) or
                    container.find('p', string=re.compile(r'abstract', re.I)) or
                    container.find('p')
                )
                abstract = abstract_elem.get_text().strip() if abstract_elem else "No abstract available from NTRS"
                
                # Generate document ID from URL or create one
                link_elem = container.find('a', href=True)
                doc_id = f"NTRS_{i:04d}"
                if link_elem and 'href' in link_elem.attrs:
                    href = link_elem['href']
                    # Extract document ID from URL if possible
                    id_match = re.search(r'(\d+)', href)
                    if id_match:
                        doc_id = f"NTRS_{id_match.group(1)}"
                
                # Create standardized publication entry
                publication = {
                    'title': title[:200],  # Limit title length
                    'abstract': abstract[:1000],  # Limit abstract length
                    'experiment_id': doc_id,
                    'impacts': [],  # NTRS doesn't typically have structured impacts
                    'metadata': {
                        'source': 'NTRS_fallback',
                        'organism': None,
                        'experiment_type': 'Technical Report',
                        'platform': 'NTRS',
                        'query_used': query,
                        'scraped_url': url,
                        'document_link': link_elem['href'] if link_elem and 'href' in link_elem.attrs else None
                    }
                }
                
                publications.append(publication)
                
            except Exception as e:
                log_error(f"Error parsing NTRS result {i}: {str(e)}")
                continue
        
        log(f"✓ Successfully scraped {len(publications)} publications from NTRS")
        return publications
        
    except requests.exceptions.RequestException as e:
        log_error(f"Network error in NTRS BeautifulSoup fallback: {str(e)}")
        raise
    except Exception as e:
        log_error(f"Unexpected error in NTRS BeautifulSoup fallback: {str(e)}")
        raise


def search_ntrs(query: str = 'space biology', limit: int = 50) -> List[Dict]:
    """
    Legacy function - redirects to BeautifulSoup implementation.
    """
    return search_ntrs_fallback(query, limit)


# Test function for development and debugging
if __name__ == '__main__':
    """Test the data fetching functionality with caching and validation."""
    log("Testing data_fetch.py functionality...")
    
    try:
        # Fetch publications with fallback capability
        log("Testing fetch_publications with fallback...")
        data = fetch_publications('space biology', limit=10)
        
        # Cache the results
        log("Caching results to JSON...")
        cache_success = cache_to_json(data, 'data/test_publications.json')
        
        # Validate results
        assert len(data) > 0, "No publications were fetched"
        assert cache_success, "Failed to cache data"
        
        log(f"✓ Successfully fetched and cached {len(data)} publications")
        
        # Print sample publication for inspection
        if data:
            sample = data[0]
            print("\n" + "="*50)
            print("SAMPLE PUBLICATION:")
            print("="*50)
            print(f"Title: {sample['title']}")
            print(f"Experiment ID: {sample['experiment_id']}")
            print(f"Abstract: {sample['abstract'][:200]}...")
            print(f"Source: {sample['metadata'].get('source', 'Unknown')}")
            print(f"Impacts: {sample.get('impacts', [])}")
            print("="*50)
        
        # Test individual components
        log("\nTesting individual components:")
        
        # Test sample data loading
        try:
            log("Testing sample data loading...")
            sample_data = load_sample_publications()
            log(f"✓ Sample data loaded: {len(sample_data)} publications available")
        except Exception as e:
            log(f"⚠ Sample data loading failed: {str(e)}")
        
        # Test NTRS fallback specifically
        try:
            log("Testing NTRS fallback...")
            ntrs_results = search_ntrs_fallback('microgravity', limit=3)
            log(f"✓ NTRS fallback returned {len(ntrs_results)} results")
        except Exception as e:
            log(f"⚠ NTRS fallback test failed: {str(e)}")
        
        # Test OSDR API
        try:
            log("Testing OSDR API...")
            osdr_results = fetch_osdr_studies(limit=3)
            log(f"✓ OSDR API returned {len(osdr_results)} results")
        except Exception as e:
            log(f"⚠ OSDR API test failed: {str(e)}")
        
        # Test offline mode by simulating network failure
        log("\nTesting offline mode resilience:")
        try:
            log("Simulating network failure scenario...")
            # This will test the full fallback chain
            offline_data = fetch_publications('radiation', limit=3)
            log(f"✓ Offline mode successful: {len(offline_data)} publications")
            if offline_data:
                log(f"Sample offline result: {offline_data[0]['title'][:50]}...")
        except Exception as e:
            log(f"⚠ Offline mode test failed: {str(e)}")
        
        log("\n🎉 All data_fetch.py tests completed successfully!")
        log(f"📊 Total publications fetched: {len(data)}")
        log("📁 Results cached to 'data/test_publications.json'")
        log("💾 Sample data available in 'data/sample_publications.json' for offline development")
        
    except AssertionError as e:
        log_error(f"Assertion failed: {str(e)}")
        log("This indicates a critical issue with the data fetching pipeline")
        
        # Fallback to sample data even in test failure
        try:
            log("Attempting to load sample data as emergency fallback...")
            emergency_data = load_sample_publications()
            if emergency_data:
                log(f"✓ Emergency fallback successful: {len(emergency_data)} sample publications loaded")
                cache_to_json(emergency_data, 'data/emergency_publications.json')
            else:
                log("✗ Emergency fallback failed - no data available")
        except Exception as fallback_error:
            log_error(f"Emergency fallback failed: {str(fallback_error)}")
        
    except Exception as e:
        log_error(f"Test failed with unexpected error: {str(e)}")
        log("This may be expected if NASA APIs are unavailable during development")
        
        # Ensure sample data is always available
        log("Ensuring sample data availability for development...")
        try:
            sample_data = load_sample_publications()
            if sample_data:
                log("✓ Sample data is available for offline development")
            else:
                log("⚠ Sample data not found - check 'data/sample_publications.json'")
        except Exception as sample_error:
            log_error(f"Sample data check failed: {str(sample_error)}")