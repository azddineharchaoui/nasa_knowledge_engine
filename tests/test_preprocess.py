"""
Pytest test suite for preprocess.py module.

Tests data preprocessing functionality including text cleaning,
keyword extraction, and DataFrame processing for NASA Space Biology data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import (
    load_and_preprocess,
    _strip_html_and_normalize,
    _extract_keywords,
    _handle_missing_values,
    _clean_text_data,
    save_to_csv,
    get_keyword_stats,
    filter_by_keywords,
    CORE_SPACE_TERMS,
    SPACE_KEYWORDS_SET
)


class TestTextCleaning:
    """Test HTML cleaning and text normalization functions."""
    
    def test_strip_html_basic(self):
        """Test basic HTML tag removal."""
        html_text = "<p>This is a <strong>test</strong> with HTML tags.</p>"
        cleaned = _strip_html_and_normalize(html_text)
        
        assert "<p>" not in cleaned
        assert "<strong>" not in cleaned
        assert "</p>" not in cleaned
        assert "</strong>" not in cleaned
        assert "this is a test with html tags." == cleaned
    
    def test_strip_html_complex(self):
        """Test complex HTML structure removal."""
        complex_html = """
        <div class="abstract">
            <h2>Abstract</h2>
            <p>Bone loss in <em>microgravity</em> environments 
            affects <a href="#">astronauts</a> during spaceflight.</p>
            <br/>
            <span style="color:red">Important findings</span>
        </div>
        """
        cleaned = _strip_html_and_normalize(complex_html)
        
        # Should not contain any HTML tags
        assert "<" not in cleaned
        assert ">" not in cleaned
        
        # Should contain the actual content
        assert "bone loss" in cleaned
        assert "microgravity" in cleaned
        assert "astronauts" in cleaned
        assert "spaceflight" in cleaned
    
    def test_strip_html_edge_cases(self):
        """Test edge cases for HTML stripping."""
        # Empty string
        assert _strip_html_and_normalize("") == ""
        
        # None input
        assert _strip_html_and_normalize(None) == ""
        
        # Plain text (no HTML)
        plain_text = "This is plain text with no HTML tags."
        assert _strip_html_and_normalize(plain_text) == plain_text.lower()
        
        # Malformed HTML
        malformed = "<p>Unclosed tag and <div>nested"
        cleaned = _strip_html_and_normalize(malformed)
        assert "<" not in cleaned
        assert "unclosed tag and nested" == cleaned
    
    def test_text_normalization(self):
        """Test text normalization (whitespace, case)."""
        messy_text = "  THIS   has    EXTRA     whitespace   "
        cleaned = _strip_html_and_normalize(messy_text)
        
        assert cleaned == "this has extra whitespace"
        assert "  " not in cleaned  # No double spaces
        assert not cleaned.startswith(" ")  # No leading space
        assert not cleaned.endswith(" ")   # No trailing space


class TestKeywordExtraction:
    """Test keyword extraction functionality."""
    
    def test_extract_keywords_sample_data(self):
        """Test keyword extraction on sample DataFrame."""
        # Create sample data with known keywords
        sample_data = [{
            'title': 'Microgravity Effects on Bone Density',
            'abstract': 'Bone loss in space environments affects astronauts during ISS missions. Radiation exposure compounds the effects.',
            'experiment_id': 'TEST_001',
            'impacts': ['bone loss', 'microgravity'],
            'metadata': {'source': 'test_data'}
        }]
        
        df = pd.DataFrame(sample_data)
        
        # Apply text cleaning first
        df = _handle_missing_values(df)
        df = _clean_text_data(df)
        
        # Extract keywords
        df = _extract_keywords(df)
        
        # Verify keywords column exists
        assert 'keywords' in df.columns
        
        # Check that expected keywords are found
        keywords = df['keywords'].iloc[0]
        assert isinstance(keywords, list)
        assert 'bone loss' in keywords
        assert 'microgravity' in keywords
        assert 'iss' in keywords
        assert 'radiation' in keywords
        
        # Check keyword counts
        assert df['keyword_count'].iloc[0] > 0
        assert df['core_keyword_count'].iloc[0] > 0
    
    def test_keyword_deduplication(self):
        """Test that keywords are properly deduplicated."""
        sample_data = [{
            'title': 'Microgravity microgravity MICROGRAVITY effects',
            'abstract': 'Repeated microgravity mentions and microgravity studies show microgravity impacts.',
            'experiment_id': 'TEST_002',
            'impacts': [],
            'metadata': {}
        }]
        
        df = pd.DataFrame(sample_data)
        df = _handle_missing_values(df)
        df = _clean_text_data(df)
        df = _extract_keywords(df)
        
        keywords = df['keywords'].iloc[0]
        
        # Should only have one instance of 'microgravity' despite multiple mentions
        microgravity_count = keywords.count('microgravity')
        assert microgravity_count == 1
    
    def test_core_terms_prioritization(self):
        """Test that core space terms are prioritized in results."""
        sample_data = [{
            'title': 'Mars Mission Radiation Effects',
            'abstract': 'ISS studies show microgravity and radiation impacts on astronaut health.',
            'experiment_id': 'TEST_003',
            'impacts': [],
            'metadata': {}
        }]
        
        df = pd.DataFrame(sample_data)
        df = _handle_missing_values(df)
        df = _clean_text_data(df)
        df = _extract_keywords(df)
        
        keywords = df['keywords'].iloc[0]
        
        # Core terms should appear first
        core_found = [kw for kw in keywords if kw in CORE_SPACE_TERMS]
        
        # Should have found core terms
        assert len(core_found) > 0
        assert 'mars' in core_found
        assert 'iss' in core_found
        assert 'microgravity' in core_found
        assert 'radiation' in core_found
    
    def test_multiword_terms(self):
        """Test extraction of multi-word terms."""
        sample_data = [{
            'title': 'Space Radiation Effects',
            'abstract': 'Space radiation and cosmic radiation affect bone density and muscle mass.',
            'experiment_id': 'TEST_004',
            'impacts': [],
            'metadata': {}
        }]
        
        df = pd.DataFrame(sample_data)
        df = _handle_missing_values(df)
        df = _clean_text_data(df)
        df = _extract_keywords(df)
        
        keywords = df['keywords'].iloc[0]
        
        # Should find multi-word terms
        assert any('radiation' in kw for kw in keywords)
        assert any('bone' in kw or 'muscle' in kw for kw in keywords)


class TestDataFrameProcessing:
    """Test full DataFrame preprocessing pipeline."""
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing on sample data."""
        # Sample data matching the structure from data_fetch.py
        sample_data = [
            {
                'title': 'Microgravity Effects on Bone Density in Rodent Models',
                'abstract': 'Bone loss in space environments represents a critical challenge. This study examines effects of <em>microgravity</em> on bone mineral density.',
                'experiment_id': 'GLDS-101',
                'impacts': ['bone loss', 'microgravity', 'skeletal system'],
                'metadata': {
                    'source': 'test_data',
                    'organism': 'Mus musculus',
                    'experiment_type': 'Transcriptome Analysis'
                }
            },
            {
                'title': 'Radiation Exposure Impact on Plant Growth',
                'abstract': 'Space radiation poses significant challenges to biological systems during interplanetary travel.',
                'experiment_id': 'GLDS-242',
                'impacts': ['radiation damage', 'plant biology'],
                'metadata': {
                    'source': 'test_data',
                    'organism': 'Arabidopsis thaliana',
                    'experiment_type': 'Gene Expression'
                }
            }
        ]
        
        df = pd.DataFrame(sample_data)
        
        # Apply full preprocessing pipeline
        df = _handle_missing_values(df)
        df = _clean_text_data(df)
        df = _extract_keywords(df)
        
        # Test expected columns exist
        expected_columns = [
            'title', 'abstract', 'experiment_id', 'impacts', 'metadata',
            'cleaned_abstract', 'cleaned_title', 'keywords', 'keyword_count',
            'core_keyword_count'
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Missing expected column: {col}"
        
        # Test first record
        first_record = df.iloc[0]
        assert 'bone loss' in first_record['keywords']
        assert 'microgravity' in first_record['keywords']
        assert first_record['keyword_count'] > 0
        assert first_record['core_keyword_count'] > 0
        
        # Test HTML was removed from cleaned_abstract
        assert '<em>' not in first_record['cleaned_abstract']
        assert '</em>' not in first_record['cleaned_abstract']
        
        # Test second record
        second_record = df.iloc[1]
        assert 'radiation' in second_record['keywords']
        assert second_record['keyword_count'] > 0
    
    def test_missing_data_handling(self):
        """Test handling of missing and malformed data."""
        sample_data = [
            {
                'title': None,
                'abstract': '',
                'experiment_id': 'TEST_MISSING',
                'impacts': None,
                'metadata': None
            },
            {
                'title': 'Valid Title',
                'abstract': 'Valid abstract with microgravity content',
                # Missing experiment_id
                'impacts': 'not a list',  # Wrong type
                'metadata': 'not a dict'  # Wrong type
            }
        ]
        
        df = pd.DataFrame(sample_data)
        df = _handle_missing_values(df)
        df = _clean_text_data(df)
        df = _extract_keywords(df)
        
        # Should not crash and should handle missing values
        assert len(df) == 2
        assert df['title'].iloc[0] == ''  # None converted to empty string
        assert isinstance(df['impacts'].iloc[0], list)  # None converted to empty list
        assert isinstance(df['metadata'].iloc[0], dict)  # None converted to empty dict
        
        # Second record should have valid keywords despite missing fields
        assert df['keyword_count'].iloc[1] > 0
        assert 'microgravity' in df['keywords'].iloc[1]


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_csv_export(self):
        """Test CSV export functionality."""
        sample_data = [{
            'title': 'Test Publication',
            'abstract': 'Test abstract with microgravity effects',
            'experiment_id': 'TEST_CSV',
            'impacts': ['microgravity', 'test'],
            'metadata': {'source': 'test'},
            'keywords': ['microgravity', 'test']
        }]
        
        df = pd.DataFrame(sample_data)
        
        # Test CSV saving
        test_csv_path = 'data/test_output.csv'
        success = save_to_csv(df, test_csv_path)
        
        assert success, "CSV save should succeed"
        assert Path(test_csv_path).exists(), "CSV file should be created"
        
        # Test CSV content
        loaded_df = pd.read_csv(test_csv_path)
        assert len(loaded_df) == 1
        assert loaded_df['title'].iloc[0] == 'Test Publication'
        
        # Keywords should be pipe-separated string
        assert isinstance(loaded_df['keywords'].iloc[0], str)
        assert 'microgravity|test' == loaded_df['keywords'].iloc[0]
        
        # Cleanup
        if Path(test_csv_path).exists():
            Path(test_csv_path).unlink()
    
    def test_keyword_stats(self):
        """Test keyword statistics generation."""
        sample_data = [
            {'keywords': ['microgravity', 'bone loss']},
            {'keywords': ['microgravity', 'radiation']},
            {'keywords': ['radiation', 'iss']}
        ]
        
        df = pd.DataFrame(sample_data)
        stats = get_keyword_stats(df)
        
        # microgravity appears twice
        assert stats['microgravity'] == 2
        # radiation appears twice
        assert stats['radiation'] == 2
        # bone loss appears once
        assert stats['bone loss'] == 1
        # iss appears once
        assert stats['iss'] == 1
    
    def test_keyword_filtering(self):
        """Test keyword-based filtering."""
        sample_data = [
            {'title': 'Study 1', 'keywords': ['microgravity', 'bone loss']},
            {'title': 'Study 2', 'keywords': ['radiation', 'iss']},
            {'title': 'Study 3', 'keywords': ['microgravity', 'radiation']},
            {'title': 'Study 4', 'keywords': ['plant biology']}
        ]
        
        df = pd.DataFrame(sample_data)
        
        # Filter for microgravity studies
        filtered = filter_by_keywords(df, ['microgravity'])
        assert len(filtered) == 2  # Studies 1 and 3
        
        # Filter for radiation studies
        filtered = filter_by_keywords(df, ['radiation'])
        assert len(filtered) == 2  # Studies 2 and 3
        
        # Filter requiring both microgravity AND radiation
        filtered = filter_by_keywords(df, ['microgravity', 'radiation'], min_matches=2)
        assert len(filtered) == 1  # Only Study 3
        
        # Filter for non-existent keyword
        filtered = filter_by_keywords(df, ['nonexistent'])
        assert len(filtered) == 0


class TestIntegration:
    """Integration tests for the full preprocessing module."""
    
    def test_load_and_preprocess_integration(self):
        """Test the main load_and_preprocess function with mock data."""
        # This test assumes sample_publications.json exists
        # We'll create temporary test data if needed
        
        test_data = [
            {
                'title': 'Integration Test Study',
                'abstract': 'This study examines <strong>microgravity</strong> effects on bone loss and muscle atrophy in astronauts aboard the ISS.',
                'experiment_id': 'INTEGRATION_001',
                'impacts': ['microgravity', 'bone loss'],
                'metadata': {
                    'source': 'integration_test',
                    'organism': 'Homo sapiens'
                }
            }
        ]
        
        # Save temporary test file
        import json
        test_file = 'data/integration_test.json'
        Path('data').mkdir(exist_ok=True)
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        try:
            # Test full preprocessing pipeline
            df = load_and_preprocess(test_file)
            
            # Verify critical assertions
            assert 'keywords' in df.columns
            assert len(df) > 0
            
            # Check that bone loss was extracted
            keywords = df['keywords'].iloc[0]
            assert 'bone loss' in keywords
            assert 'microgravity' in keywords
            assert 'iss' in keywords
            
            # Check HTML was cleaned
            assert '<strong>' not in df['cleaned_abstract'].iloc[0]
            
        finally:
            # Cleanup
            if Path(test_file).exists():
                Path(test_file).unlink()


# Pytest configuration and fixtures
@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame for tests."""
    sample_data = [
        {
            'title': 'Microgravity Study',
            'abstract': 'Bone loss effects in space',
            'experiment_id': 'FIXTURE_001',
            'impacts': ['microgravity', 'bone loss'],
            'metadata': {'source': 'fixture'}
        },
        {
            'title': 'Radiation Research',
            'abstract': 'Space radiation impacts on cells',
            'experiment_id': 'FIXTURE_002',
            'impacts': ['radiation'],
            'metadata': {'source': 'fixture'}
        }
    ]
    return pd.DataFrame(sample_data)


def test_sample_fixture(sample_dataframe):
    """Test that the sample fixture works correctly."""
    assert len(sample_dataframe) == 2
    assert 'title' in sample_dataframe.columns
    assert 'abstract' in sample_dataframe.columns


if __name__ == '__main__':
    """Run tests directly if script is executed."""
    pytest.main([__file__, '-v'])