#!/usr/bin/env python3
"""
Core Integration Pipeline for NASA Space Biology hackathon prototype.

This module integrates all components of the pipeline:
Data Fetch → Preprocess → Summarize → Knowledge Graph

Provides a complete end-to-end workflow for processing NASA research data
and building knowledge graphs for analysis and visualization.
"""

import pandas as pd
from typing import Tuple, Optional, Any
from pathlib import Path

# Import all pipeline modules
try:
    import data_fetch
    DATA_FETCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: data_fetch module not available - {e}")
    DATA_FETCH_AVAILABLE = False

try:
    import preprocess
    PREPROCESS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: preprocess module not available - {e}")
    PREPROCESS_AVAILABLE = False

try:
    import summarizer
    SUMMARIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: summarizer module not available - {e}")
    SUMMARIZER_AVAILABLE = False

try:
    import kg_builder
    KG_BUILDER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: kg_builder module not available - {e}")
    KG_BUILDER_AVAILABLE = False

# Import utilities
from utils import log, log_error, cache_to_json


def run_pipeline(query: str = 'space biology', limit: int = 50) -> Tuple[Optional[pd.DataFrame], Optional[Any]]:
    """
    Execute the complete NASA Space Biology research pipeline.
    
    Orchestrates the full workflow:
    1. Fetch publications from NASA APIs (with fallbacks)
    2. Preprocess text data and extract keywords
    3. Summarize abstracts using AI/fallback methods
    4. Build knowledge graph with entities and relationships
    
    Args:
        query: Search query for publications (default: 'space biology')
        limit: Maximum number of publications to fetch (default: 50)
        
    Returns:
        Tuple of (processed_dataframe, knowledge_graph)
        Returns (None, None) if pipeline fails at any stage
        
    Example:
        >>> df, graph = run_pipeline('bone loss', limit=20)
        >>> if df is not None:
        ...     print(f"Processed {len(df)} publications")
        >>> if graph is not None:
        ...     print(f"Knowledge graph: {graph.number_of_nodes()} nodes")
    """
    log(f"🚀 Starting NASA Space Biology pipeline...")
    log(f"Query: '{query}', Limit: {limit}")
    
    # Step 1: Fetch Publications
    log("📡 Step 1: Fetching publications...")
    if not DATA_FETCH_AVAILABLE:
        log_error("Data fetch module not available")
        return None, None
    
    try:
        data = data_fetch.fetch_publications(query, limit)
        if not data:
            log_error("No data retrieved from fetch step")
            return None, None
        
        log(f"✓ Fetched {len(data)} publications")
    except Exception as e:
        log_error(f"Error in data fetch step: {str(e)}")
        return None, None
    
    # Step 2: Preprocess Data
    log("🔧 Step 2: Preprocessing data...")
    if not PREPROCESS_AVAILABLE:
        log_error("Preprocess module not available")
        return None, None
    
    try:
        # Save data temporarily for preprocessing
        temp_file = 'data/pipeline_temp.json'
        cache_to_json(data, temp_file)
        
        df = preprocess.load_and_preprocess(temp_file)
        if df is None or len(df) == 0:
            log_error("Preprocessing returned empty DataFrame")
            return None, None
        
        log(f"✓ Preprocessed {len(df)} records with {len(df.columns)} features")
    except Exception as e:
        log_error(f"Error in preprocessing step: {str(e)}")
        return None, None
    
    # Step 3: Summarize Abstracts  
    log("🤖 Step 3: Summarizing abstracts...")
    if not SUMMARIZER_AVAILABLE:
        log_error("Summarizer module not available")
        return df, None  # Return df even if summarization fails
    
    try:
        df = summarizer.summarize_abstracts(df)
        summaries_count = df['summary'].notna().sum() if 'summary' in df.columns else 0
        log(f"✓ Generated {summaries_count} summaries")
    except Exception as e:
        log_error(f"Error in summarization step: {str(e)}")
        # Continue pipeline without summaries
    
    # Step 4: Build Knowledge Graph
    log("🕸️ Step 4: Building knowledge graph...")
    if not KG_BUILDER_AVAILABLE:
        log_error("Knowledge graph builder not available")
        return df, None
    
    try:
        # Ensure DataFrame has required 'id' column for knowledge graph
        if 'id' not in df.columns and 'experiment_id' in df.columns:
            df = df.rename(columns={'experiment_id': 'id'})
            log("Mapped 'experiment_id' to 'id' column for knowledge graph")
        elif 'id' not in df.columns:
            # Create id column if missing
            df['id'] = [f'EXP-{i+1:03d}' for i in range(len(df))]
            log("Created 'id' column for knowledge graph")
        
        G = kg_builder.build_kg(df)
        if G is None:
            log_error("Knowledge graph construction returned None")
            return df, None
        
        log(f"✓ Built knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        log_error(f"Error in knowledge graph step: {str(e)}")
        return df, None
    
    # Pipeline Complete
    log("🎉 Pipeline completed successfully!")
    log(f"Final results: {len(df)} publications, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return df, G


def run_query_analysis(df: pd.DataFrame, graph: Any, query_terms: list = None) -> dict:
    """
    Run analysis queries on the processed data and knowledge graph.
    
    Args:
        df: Processed DataFrame from pipeline
        graph: Knowledge graph from pipeline
        query_terms: List of terms to query (default: ['radiation', 'bone loss', 'muscle'])
        
    Returns:
        Dictionary with query results and analysis
    """
    if query_terms is None:
        query_terms = ['radiation', 'bone loss', 'muscle', 'microgravity']
    
    results = {
        'dataframe_stats': {
            'total_publications': len(df) if df is not None else 0,
            'columns': list(df.columns) if df is not None else [],
            'summaries_available': df['summary'].notna().sum() if df is not None and 'summary' in df.columns else 0
        },
        'graph_stats': {
            'total_nodes': graph.number_of_nodes() if graph is not None else 0,
            'total_edges': graph.number_of_edges() if graph is not None else 0
        },
        'query_results': {}
    }
    
    # Run queries on knowledge graph
    if graph is not None and KG_BUILDER_AVAILABLE:
        for term in query_terms:
            try:
                query_result = kg_builder.query_kg(graph, term)
                results['query_results'][term] = {
                    'matches': len(query_result.get('matches', [])),
                    'neighbors': len(query_result.get('neighbors', [])),
                    'summaries': len(query_result.get('summaries', []))
                }
            except Exception as e:
                results['query_results'][term] = {'error': str(e)}
    
    return results


def get_pipeline_status() -> dict:
    """
    Get status of all pipeline components.
    
    Returns:
        Dictionary with availability status of each module
    """
    return {
        'data_fetch': DATA_FETCH_AVAILABLE,
        'preprocess': PREPROCESS_AVAILABLE, 
        'summarizer': SUMMARIZER_AVAILABLE,
        'kg_builder': KG_BUILDER_AVAILABLE,
        'pipeline_ready': all([
            DATA_FETCH_AVAILABLE,
            PREPROCESS_AVAILABLE,
            SUMMARIZER_AVAILABLE,
            KG_BUILDER_AVAILABLE
        ])
    }


def cleanup_temp_files():
    """Clean up temporary files created during pipeline execution."""
    temp_files = [
        'data/pipeline_temp.json',
        'data/integration_test.json'
    ]
    
    for temp_file in temp_files:
        try:
            temp_path = Path(temp_file)
            if temp_path.exists():
                temp_path.unlink()
                log(f"Cleaned up {temp_file}")
        except Exception as e:
            log_error(f"Error cleaning {temp_file}: {str(e)}")


if __name__ == '__main__':
    """
    Main execution: Run pipeline with 'radiation' query and print results.
    """
    print("NASA Space Biology Pipeline Integration")
    print("=" * 50)
    
    # Check pipeline status
    status = get_pipeline_status()
    print("\n📊 Pipeline Component Status:")
    for component, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {component}: {available}")
    
    if not status['pipeline_ready']:
        print("\n⚠️  Warning: Not all pipeline components available")
        print("   Pipeline will run with available components only")
    
    # Run pipeline with 'radiation' query as requested
    print(f"\n🚀 Running pipeline with query: 'radiation'")
    print("-" * 40)
    
    try:
        df, G = run_pipeline('radiation', limit=30)
        
        # Print results
        print(f"\n📋 Pipeline Results:")
        print(f"=" * 30)
        
        if df is not None:
            print(f"📄 DataFrame: {len(df)} publications processed")
            print(f"   Columns: {len(df.columns)} features")
            if 'summary' in df.columns:
                summaries = df['summary'].notna().sum()
                print(f"   Summaries: {summaries} generated")
            if 'keywords' in df.columns:
                total_keywords = df['keywords'].apply(len).sum() if df['keywords'].dtype == 'object' else 0
                print(f"   Keywords: {total_keywords} total extracted")
        else:
            print("❌ DataFrame: Not generated")
        
        if G is not None:
            print(f"🕸️  Knowledge Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Query the graph for radiation-related content
            if KG_BUILDER_AVAILABLE:
                radiation_query = kg_builder.query_kg(G, 'radiation')
                print(f"   Radiation matches: {len(radiation_query.get('matches', []))}")
                print(f"   Related experiments: {len(radiation_query.get('summaries', []))}")
        else:
            print("❌ Knowledge Graph: Not generated")
        
        # Run additional analysis
        print(f"\n🔍 Query Analysis:")
        print(f"-" * 20)
        analysis = run_query_analysis(df, G)
        
        for term, results in analysis.get('query_results', {}).items():
            if 'error' not in results:
                print(f"   '{term}': {results['matches']} matches, {results['neighbors']} neighbors")
            else:
                print(f"   '{term}': Error - {results['error']}")
        
        print(f"\n✅ Pipeline execution completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        log_error(f"Main pipeline execution error: {str(e)}")
    
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary files...")
        cleanup_temp_files()
        print(f"🎯 Integration complete!")