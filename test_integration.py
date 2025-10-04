#!/usr/bin/env python3
"""
Test script for the integration pipeline.
"""

from integrate_core import run_pipeline

def test_integration():
    """Test the complete integration pipeline."""
    print('Testing fixed integration pipeline...')
    
    # Run pipeline with radiation query
    df, graph = run_pipeline('radiation', limit=5)
    
    print(f'Results: df={len(df) if df is not None else 0} rows, graph={graph.number_of_nodes() if graph is not None else 0} nodes')
    
    if graph is not None:
        from kg_builder import query_kg
        result = query_kg(graph, 'radiation')
        print(f'Radiation query: {len(result["matches"])} matches')
        
        # Show some details
        if result['summaries']:
            print('Related experiments:')
            for summary in result['summaries'][:2]:  # Show first 2
                print(f'  - {summary["title"]}')
    
    print('Integration test completed!')

if __name__ == '__main__':
    test_integration()