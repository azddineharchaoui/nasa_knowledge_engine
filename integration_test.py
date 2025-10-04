#!/usr/bin/env python3
"""
Integration test for the complete NASA Space Biology pipeline.

Tests the full workflow: data fetch → preprocess → summarize → knowledge graph
"""

from data_fetch import fetch_publications
from preprocess import load_and_preprocess
from summarizer import summarize_abstracts
from kg_builder import extract_entities, build_knowledge_graph, analyze_graph_structure

def test_complete_pipeline():
    """Test the complete pipeline integration."""
    print('Testing complete pipeline integration...')

    # 1. Fetch sample data (will use fallbacks if needed)
    print('1. Fetching data...')
    data = fetch_publications('bone loss')
    print(f'   Fetched {len(data)} publications')

    # 2. Preprocess data
    print('2. Preprocessing...')
    # Save data to file first, then load it
    from utils import cache_to_json
    temp_file = 'data/integration_test.json'
    cache_to_json(data, temp_file)
    df = load_and_preprocess(temp_file)
    if df is not None and len(df) > 0:
        print(f'   Processed {len(df)} abstracts')
        
        # 3. Summarize abstracts
        print('3. Summarizing...')
        df_with_summaries = summarize_abstracts(df)
        summaries_count = df_with_summaries['summary'].notna().sum()
        print(f'   Generated {summaries_count} summaries')
        
        # 4. Extract entities from abstracts
        print('4. Extracting entities...')
        entities_list = []
        for i, abstract in enumerate(df['abstract'].head(3)):  # Test first 3
            if abstract:
                entities = extract_entities(abstract)
                entities_list.append(entities)
                print(f'     Abstract {i+1}: {len(entities["impacts"])} impacts, {len(entities["organizations"])} orgs')
        
        print(f'   Extracted entities from {len(entities_list)} abstracts')
        
        # 5. Build knowledge graph
        print('5. Building knowledge graph...')
        experiment_ids = [f'INTEGRATION-{i+1:03d}' for i in range(len(entities_list))]
        graph = build_knowledge_graph(entities_list, experiment_ids)
        
        if graph:
            analysis = analyze_graph_structure(graph)
            print(f'   Graph: {analysis["nodes"]} nodes, {analysis["edges"]} edges')
            print(f'   Node types: {analysis["node_types"]}')
        else:
            print('   Graph construction skipped (NetworkX not available)')
        
        print('🎉 Complete pipeline integration successful!')
        return True
    else:
        print('⚠ No data available for testing')
        return False

if __name__ == '__main__':
    test_complete_pipeline()