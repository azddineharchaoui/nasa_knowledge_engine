"""
NASA Space Biology Knowledge Engine - Streamlit Dashboard

Interactive web application for exploring space biology research impacts
through AI-powered knowledge graphs and summarization.


identified that the Streamlit app's slowness was due to full integration pipeline
execution and expensive graph layout computations. 
Implemented a fast mode that skips heavy AI summarization and reduces 
the number of publications loaded, along with reducing graph layout iterations.
These changes significantly improved app startup time and rendering speed
"""


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from integrate_core import run_pipeline
    from kg_builder import query_kg
    import utils
    PIPELINE_AVAILABLE = True
except ImportError as e:
    st.error(f"Pipeline modules not available: {str(e)}")
    PIPELINE_AVAILABLE = False


@st.cache_resource
def load_data(fast_mode: bool = True, limit: int = 20):
    """Load and cache the complete pipeline results."""
    if not PIPELINE_AVAILABLE:
        # Return sample data for demo purposes
        sample_data = pd.DataFrame({
            'title': ['Microgravity Effects on Bone Loss', 'Radiation Impact on DNA', 'ISS Plant Growth Study'],
            'abstract': ['Study of bone density...', 'Cosmic radiation analysis...', 'Plant growth in space...'],
            'summary': ['Bone loss observed', 'DNA damage detected', 'Growth enhanced'],
            'keywords': [['microgravity', 'bone'], ['radiation', 'DNA'], ['plant', 'growth']]
        })
        return sample_data, None
    
    try:
        utils.log("Loading Space Biology data pipeline...")
        # Propagate fast mode via env for downstream components that may read it
        os.environ['NKE_FAST_MODE'] = '1' if fast_mode else '0'
        df, G = run_pipeline(query='space biology', limit=limit, fast_mode=fast_mode)
        utils.log(f"Loaded {len(df)} publications with {G.number_of_nodes() if G else 0} knowledge graph nodes")
        return df, G
    except Exception as e:
        st.error(f"Pipeline execution failed: {str(e)}")
        return pd.DataFrame(), None


def create_network_plot(G, query_term=None):
    """Create enhanced interactive network plot using NetworkX and Plotly with custom layouts."""
    if G is None or G.number_of_nodes() == 0:
        return go.Figure().add_annotation(text="No graph data available", 
                                         xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Use NetworkX for better layout algorithms - force-directed spring layout
    try:
        # Multiple layout options for better visualization
        if G.number_of_nodes() > 100:
            # For large graphs, use faster layout
            pos = nx.spring_layout(G, k=2/np.sqrt(G.number_of_nodes()), iterations=20, seed=42)
        elif G.number_of_nodes() > 20:
            # Medium graphs - balanced quality/speed
            pos = nx.spring_layout(G, k=1.5, iterations=30, seed=42)
        else:
            # Small graphs - quicker layout for responsiveness
            pos = nx.spring_layout(G, k=1, iterations=25, seed=42)
            
        # Alternative layouts for comparison (can be toggled)
        if query_term and len(query_term) > 0:
            # For search results, try focused layout for better visibility
            pos = nx.spring_layout(G, k=1.2, iterations=20, seed=42)
            
    except Exception:
        # Fallback to simple random layout
        pos = nx.random_layout(G)
    
    # Enhanced edge extraction with varying weights
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Get edge weight if available
        edge_data = G.get_edge_data(edge[0], edge[1])
        weight = edge_data.get('weight', 1) if edge_data else 1
        edge_weights.extend([weight, weight, None])
    
    # Enhanced node extraction with proper typing and coloring
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_labels = []
    
    # Define color scheme by node type as requested
    color_map = {
        'Experiment': '#1f77b4',  # Blue for experiments
        'Impact': '#d62728',      # Red for impacts  
        'Result': '#2ca02c',      # Green for results
        'Organism': '#ff7f0e',    # Orange for organisms
        'Location': '#9467bd',    # Purple for locations
        'Default': '#7f7f7f'      # Gray for unknown
    }
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Enhanced node type detection and coloring
        node_str = str(node)
        node_type = 'Default'
        
        if ':' in node_str:
            node_type = node_str.split(':')[0]
        elif 'experiment' in node_str.lower():
            node_type = 'Experiment'
        elif 'impact' in node_str.lower():
            node_type = 'Impact'
        elif 'result' in node_str.lower():
            node_type = 'Result'
        
        # Apply color mapping
        color = color_map.get(node_type, color_map['Default'])
        node_color.append(color)
        
        # Node size based on degree (connections)
        degree = G.degree(node)
        size = max(8, min(25, 8 + degree * 2))  # Size 8-25 based on connections
        node_size.append(size)
        
        # Enhanced hover text with more details
        node_attrs = G.nodes[node]
        hover_text = f"<b>{node_type}: {node}</b><br>"
        hover_text += f"Connections: {degree}<br>"
        
        if 'title' in node_attrs:
            title = node_attrs['title'][:60] + '...' if len(node_attrs['title']) > 60 else node_attrs['title']
            hover_text += f"Title: {title}<br>"
            
        if 'summary' in node_attrs:
            summary = node_attrs['summary'][:120] + '...' if len(node_attrs['summary']) > 120 else node_attrs['summary']
            hover_text += f"Summary: {summary}<br>"
            
        if 'experiment_type' in node_attrs:
            hover_text += f"Type: {node_attrs['experiment_type']}<br>"
            
        # Highlight if matches search term
        if query_term and query_term.lower() in node_str.lower():
            hover_text += f"<b>🎯 Matches search: {query_term}</b>"
        
        node_text.append(hover_text)
        
        # Node labels for key nodes
        if degree > 2 or (query_term and query_term.lower() in node_str.lower()):
            label = node_str.split(':')[-1][:15] + '...' if len(node_str.split(':')[-1]) > 15 else node_str.split(':')[-1]
        else:
            label = ''
        node_labels.append(label)
    
    # Create enhanced figure with multiple traces
    fig = go.Figure()
    
    # Add edges with varying opacity based on weights
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(136,136,136,0.6)'),
        hoverinfo='none',
        mode='lines',
        name='Connections',
        showlegend=False
    ))
    
    # Add nodes with enhanced styling
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition="middle center",
        textfont=dict(size=8, color='white'),
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        name='Nodes',
        showlegend=False
    ))
    
    # Add legend for node types
    legend_traces = []
    for node_type, color in color_map.items():
        if node_type != 'Default':
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=f'{node_type}s',
                showlegend=True
            ))
    
    # Enhanced layout with better interactivity
    fig.update_layout(
        title=dict(
            text=f'Space Biology Knowledge Graph{f" - Search: {query_term}" if query_term else ""}',
            font=dict(size=16),
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        annotations=[
            dict(
                text="💡 Hover over nodes for details • Click and drag to explore • Legend shows node types",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                xanchor='center',
                font=dict(size=10, color='gray')
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        # Enhanced interactivity
        dragmode='pan'
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="NASA Space Biology Knowledge Engine",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for interactive features
    if 'selected_node' not in st.session_state:
        st.session_state.selected_node = None
    if 'show_node_details' not in st.session_state:
        st.session_state.show_node_details = False
    
    # Main title
    st.title('🚀 Space Biology Knowledge Engine')
    st.markdown("*AI-powered exploration of space biology research impacts*")
    
    # Sidebar controls
    st.sidebar.header("🔍 Search & Filters")

    # Performance options
    st.sidebar.subheader("Performance")
    fast_mode = st.sidebar.checkbox("Fast Mode (skip AI summarization)", value=True,
                                    help="Uses lightweight summaries and speeds up loading. Turn off for full AI summaries.")

    # Data limit control depending on mode
    default_limit = 20 if fast_mode else 50
    limit = st.sidebar.slider("Publications to load", min_value=5, max_value=100 if not fast_mode else 50,
                              value=default_limit, step=5)

    # Load data
    with st.spinner('Loading Space Biology pipeline...'):
        df, G = load_data(fast_mode=fast_mode, limit=limit)
    
    # Sidebar controls continued
    st.sidebar.header("🔍 Search & Filters")
    
    # Search input as requested
    search = st.sidebar.text_input(
        'Search impacts (e.g. radiation)', 
        placeholder='Enter keywords like: radiation, microgravity, bone loss'
    )
    
    # Additional filters
    st.sidebar.subheader("Filters")
    
    if not df.empty:
        # Limit results with proper bounds checking
        df_len = len(df)
        if df_len > 5:
            max_results = st.sidebar.slider("Max Results", 5, df_len, min(20, df_len))
        else:
            # If we have 5 or fewer rows, just show all and display info
            max_results = df_len
            st.sidebar.info(f"Showing all {df_len} available results")
        
        # Extract unique keywords from DataFrame
        unique_keywords = []
        if 'keywords' in df.columns:
            all_keywords = []
            for keywords in df['keywords']:
                if isinstance(keywords, list):
                    all_keywords.extend([kw.strip().lower() for kw in keywords if kw.strip()])
                elif isinstance(keywords, str) and keywords.strip():
                    # Handle comma-separated keywords as strings
                    all_keywords.extend([kw.strip().lower() for kw in keywords.split(',') if kw.strip()])
            
            # Get unique keywords sorted by frequency (most common first)
            keyword_counts = pd.Series(all_keywords).value_counts()
            unique_keywords = keyword_counts.head(25).index.tolist()  # Top 25 keywords
        
        # Filter by keywords using multiselect as requested
        selected_keywords = []
        if unique_keywords:
            selected_keywords = st.sidebar.multiselect(
                'Filter by keywords', 
                options=unique_keywords,
                default=[],
                help="Select one or more keywords to filter publications"
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Research Publications")
        
        if df.empty:
            st.warning("No data available. Please check the pipeline configuration.")
        else:
            # Apply filters to DataFrame before query
            filtered_df = df.copy()
            
            # Apply keyword filter first
            if selected_keywords:
                def keyword_match(keywords_cell):
                    if isinstance(keywords_cell, list):
                        cell_keywords = [kw.strip().lower() for kw in keywords_cell]
                    elif isinstance(keywords_cell, str):
                        cell_keywords = [kw.strip().lower() for kw in keywords_cell.split(',')]
                    else:
                        cell_keywords = []
                    return any(kw in selected_keywords for kw in cell_keywords)
                
                keyword_mask = filtered_df['keywords'].apply(keyword_match)
                filtered_df = filtered_df[keyword_mask]
                st.info(f"Filtered to {len(filtered_df)} publications with selected keywords")
            
            # Apply search filter
            if search:
                search_mask = (
                    filtered_df['title'].str.contains(search, case=False, na=False) |
                    filtered_df['abstract'].str.contains(search, case=False, na=False)
                )
                if 'summary' in filtered_df.columns:
                    search_mask |= filtered_df['summary'].str.contains(search, case=False, na=False)
                
                filtered_df = filtered_df[search_mask]
                st.info(f"Found {len(filtered_df)} publications matching '{search}'")
            
            # Limit results
            if 'max_results' in locals():
                filtered_df = filtered_df.head(max_results)
            
            # Show dataframe as requested
            st.dataframe(filtered_df.head(), use_container_width=True)
            
            # Publication details
            if len(filtered_df) > 0:
                st.subheader("📄 Publication Details")
                
                # Select publication to view
                pub_titles = [f"{i}: {title[:60]}..." for i, title in enumerate(filtered_df['title'].head(10))]
                selected_pub = st.selectbox("Select publication to view", pub_titles)
                
                if selected_pub:
                    pub_idx = int(selected_pub.split(':')[0])
                    pub_data = filtered_df.iloc[pub_idx]
                    
                    st.write(f"**Title:** {pub_data['title']}")
                    st.write(f"**Abstract:** {pub_data['abstract'][:500]}...")
                    
                    if 'summary' in pub_data and pd.notna(pub_data['summary']):
                        st.write(f"**AI Summary:** {pub_data['summary']}")
                    
                    if 'keywords' in pub_data and isinstance(pub_data['keywords'], list):
                        st.write(f"**Keywords:** {', '.join(pub_data['keywords'])}")
    
    with col2:
        st.header("🕸️ Knowledge Graph")
        
        # Show graph statistics
        if G is not None:
            st.metric("Nodes", G.number_of_nodes())
            st.metric("Edges", G.number_of_edges())
            
            # Enhanced knowledge graph search with results display
            if search and G.number_of_nodes() > 0:
                try:
                    results = query_kg(G, search)
                    if results and 'nodes' in results and len(results['nodes']) > 0:
                        st.success(f"Found {len(results['nodes'])} related nodes")
                        
                        # Display search results as requested
                        st.subheader('🔍 Results')
                        
                        # Convert nodes to DataFrame for display
                        nodes_data = []
                        for node in results['nodes']:
                            node_info = {
                                'Node': str(node),
                                'Type': str(node).split(':')[0] if ':' in str(node) else 'Unknown'
                            }
                            
                            # Add node attributes if available
                            if hasattr(G, 'nodes') and node in G.nodes():
                                attrs = G.nodes[node]
                                if 'title' in attrs:
                                    node_info['Title'] = attrs['title'][:50] + '...' if len(attrs['title']) > 50 else attrs['title']
                                if 'summary' in attrs:
                                    node_info['Summary'] = attrs['summary'][:100] + '...' if len(attrs['summary']) > 100 else attrs['summary']
                            
                            nodes_data.append(node_info)
                        
                        # Display results DataFrame with interactive selection
                        results_df = pd.DataFrame(nodes_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Add node selection for detailed view
                        if len(nodes_data) > 0:
                            node_options = [f"{node['Node']} ({node['Type']})" for node in nodes_data]
                            selected_node_display = st.selectbox(
                                "Select node for details:",
                                options=["None"] + node_options,
                                key="node_selector"
                            )
                            
                            if selected_node_display != "None":
                                # Extract actual node from display string
                                selected_node = selected_node_display.split(' (')[0]
                                st.session_state.selected_node = selected_node
                                st.session_state.show_node_details = True
                        
                        # Show node details in expander using session state
                        if st.session_state.show_node_details and st.session_state.selected_node:
                            with st.expander(f"📋 Node Details: {st.session_state.selected_node}", expanded=True):
                                node_name = st.session_state.selected_node
                                
                                if node_name in G.nodes():
                                    node_attrs = G.nodes[node_name]
                                    
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.write(f"**Node ID:** {node_name}")
                                        st.write(f"**Type:** {node_name.split(':')[0] if ':' in node_name else 'Unknown'}")
                                        
                                        # Show node degree (connections)
                                        degree = G.degree(node_name)
                                        st.write(f"**Connections:** {degree}")
                                        
                                        # Show neighbors
                                        neighbors = list(G.neighbors(node_name))
                                        if neighbors:
                                            st.write(f"**Connected to:** {len(neighbors)} nodes")
                                            neighbor_names = [str(n)[:30] + '...' if len(str(n)) > 30 else str(n) for n in neighbors[:5]]
                                            st.write(f"*Sample connections:* {', '.join(neighbor_names)}")
                                    
                                    with col2:
                                        if 'title' in node_attrs:
                                            st.write(f"**Title:** {node_attrs['title']}")
                                        
                                        if 'summary' in node_attrs:
                                            st.write(f"**Summary:** {node_attrs['summary']}")
                                        
                                        if 'experiment_type' in node_attrs:
                                            st.write(f"**Experiment Type:** {node_attrs['experiment_type']}")
                                        
                                        if 'impacts' in node_attrs:
                                            st.write(f"**Impacts:** {node_attrs['impacts']}")
                                    
                                    # Clear selection button
                                    if st.button("Clear Selection", key="clear_node_selection"):
                                        st.session_state.selected_node = None
                                        st.session_state.show_node_details = False
                                        st.experimental_rerun()
                                else:
                                    st.error(f"Node '{node_name}' not found in knowledge graph.")
                        
                        # Create subgraph visualization
                        if 'edges' in results and len(results['edges']) > 0:
                            # Create subgraph for visualization
                            subgraph = nx.Graph()
                            subgraph.add_nodes_from(results['nodes'])
                            subgraph.add_edges_from(results['edges'])
                            
                            # Add node attributes from original graph
                            for node in results['nodes']:
                                if node in G.nodes():
                                    subgraph.nodes[node].update(G.nodes[node])
                            
                            # Create enhanced graph visualization
                            fig = create_network_plot(subgraph, search)
                            fig.update_layout(title=f'Knowledge Graph Results for "{search}"')
                            st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.info(f"No related nodes found for '{search}'")
                        
                except Exception as e:
                    st.error(f"Graph query failed: {str(e)}")
        else:
            st.info("Knowledge graph not available")
    
    # Network visualization
    st.header("🌐 Interactive Knowledge Graph")
    
    if G is not None and G.number_of_nodes() > 0:
        # Create and display network plot
        network_fig = create_network_plot(G, search)
        st.plotly_chart(network_fig, use_container_width=True)
    else:
        st.info("Knowledge graph visualization not available")
    
    # Summary statistics
    st.header("📈 Summary Statistics")
    
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Publications", len(df))
        
        with col2:
            if 'summary' in df.columns:
                summaries_count = df['summary'].notna().sum()
                st.metric("AI Summaries", summaries_count)
            else:
                st.metric("AI Summaries", "N/A")
        
        with col3:
            if 'keywords' in df.columns:
                total_keywords = sum(len(kw) if isinstance(kw, list) else 0 for kw in df['keywords'])
                st.metric("Total Keywords", total_keywords)
            else:
                st.metric("Total Keywords", "N/A")
        
        with col4:
            if G is not None:
                st.metric("Graph Nodes", G.number_of_nodes())
            else:
                st.metric("Graph Nodes", "N/A")
        
        # Keyword frequency chart
        if 'keywords' in df.columns:
            st.subheader("🏷️ Most Frequent Keywords")
            
            all_keywords = []
            for keywords in df['keywords']:
                if isinstance(keywords, list):
                    all_keywords.extend(keywords)
            
            if all_keywords:
                keyword_counts = pd.Series(all_keywords).value_counts().head(10)
                fig_bar = px.bar(
                    x=keyword_counts.values, 
                    y=keyword_counts.index, 
                    orientation='h',
                    title="Top 10 Keywords",
                    labels={'x': 'Frequency', 'y': 'Keywords'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("*NASA Space Biology Knowledge Engine - Hackathon 2025*")


if __name__ == "__main__":
    main()