"""
Knowledge Graph Builder for NASA Space Biology hackathon prototype.

This module provides functionality to extract entities and relationships from research
abstracts and build knowledge graphs using spaCy NER and NetworkX graph structures.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import pandas as pd

# Optional imports with fallbacks for missing dependencies
try:
    import spacy
    from spacy.tokens import Doc, Token
    SPACY_AVAILABLE = True
except ImportError:
    print("Warning: spaCy not available. Knowledge graph functionality will use fallbacks.")
    SPACY_AVAILABLE = False
    spacy = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    print("Warning: NetworkX not available. Graph visualization will be limited.")
    NETWORKX_AVAILABLE = False
    nx = None

# Import local utilities
from utils import log_info, log_error, cache_to_json, load_from_cache


# Global NLP model variable
nlp = None

def load_nlp_model(model_name: str = 'en_core_web_sm') -> bool:
    """
    Load spaCy NLP model for entity extraction.
    
    Args:
        model_name: Name of the spaCy model to load
        
    Returns:
        bool: True if model loaded successfully, False otherwise
        
    Example:
        >>> success = load_nlp_model()
        >>> if success:
        ...     print("NLP model ready for processing")
    """
    global nlp
    
    if not SPACY_AVAILABLE:
        log_error("spaCy not available. Install with: pip install spacy")
        return False
    
    try:
        nlp = spacy.load(model_name)
        log_info(f"✓ Loaded spaCy model: {model_name}")
        return True
        
    except OSError:
        log_error(f"Model '{model_name}' not found. Download with: python -m spacy download {model_name}")
        return False
    
    except Exception as e:
        log_error(f"Error loading spaCy model: {str(e)}")
        return False


def extract_entities(text: str) -> Dict:
    """
    Extract named entities and custom impact relationships from text.
    
    Uses spaCy NER to identify entities and custom rules to find health-related
    impacts like 'bone loss', 'muscle atrophy', etc.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing:
        - entities: List of (text, label) tuples from spaCy NER
        - impacts: List of health impact terms found using custom rules
        - locations: Extracted location entities (GPE, LOC)
        - organizations: Organization entities (ORG)
        - custom_relations: Rule-based relationship extraction
        
    Example:
        >>> result = extract_entities("Bone loss affects astronauts during ISS missions")
        >>> print(result['impacts'])  # ['bone loss']
        >>> print(result['entities'])  # [('ISS', 'ORG'), ...]
    """
    if not text or not text.strip():
        return {
            'entities': [],
            'impacts': [],
            'locations': [],
            'organizations': [],
            'custom_relations': []
        }
    
    # Initialize result structure
    result = {
        'entities': [],
        'impacts': [],
        'locations': [],
        'organizations': [],
        'custom_relations': []
    }
    
    # Fallback processing if spaCy not available
    if not SPACY_AVAILABLE or nlp is None:
        return _fallback_entity_extraction(text)
    
    try:
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract standard named entities
        for ent in doc.ents:
            result['entities'].append((ent.text, ent.label_))
            
            # Categorize specific entity types
            if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entity, Location
                result['locations'].append(ent.text)
            elif ent.label_ == 'ORG':  # Organization
                result['organizations'].append(ent.text)
        
        # Custom rule-based impact extraction
        result['impacts'] = _extract_health_impacts(text, doc)
        
        # Extract custom relationships using dependency parsing
        result['custom_relations'] = _extract_custom_relations(doc)
        
        log_info(f"Extracted {len(result['entities'])} entities, {len(result['impacts'])} impacts")
        return result
        
    except Exception as e:
        log_error(f"Error in entity extraction: {str(e)}")
        return _fallback_entity_extraction(text)


def _extract_health_impacts(text: str, doc: Optional[object] = None) -> List[str]:
    """
    Extract health impact terms using custom rules.
    
    Looks for patterns like:
    - "bone loss"
    - "muscle atrophy" 
    - "radiation exposure"
    - "cardiovascular deconditioning"
    
    Args:
        text: Input text
        doc: spaCy Doc object (optional)
        
    Returns:
        List of identified health impact terms
    """
    impacts = []
    
    # Health impact patterns - terms that indicate biological effects
    impact_patterns = [
        # Bone-related impacts
        r'bone\s+(?:loss|density|deterioration|demineralization|weakening)',
        r'(?:decreased|reduced|loss\s+of)\s+bone\s+(?:density|mineral|mass)',
        r'osteo(?:porosis|penia)',
        
        # Muscle-related impacts  
        r'muscle\s+(?:atrophy|wasting|weakness|deterioration|loss)',
        r'(?:decreased|reduced|loss\s+of)\s+muscle\s+(?:mass|strength|function)',
        r'sarcopenia',
        
        # Cardiovascular impacts
        r'cardiovascular\s+(?:deconditioning|changes|adaptation)',
        r'heart\s+(?:rate|function|performance)\s+(?:changes|alterations)',
        r'blood\s+pressure\s+(?:changes|regulation)',
        
        # Radiation impacts
        r'radiation\s+(?:exposure|damage|effects)',
        r'DNA\s+(?:damage|repair|mutation)',
        r'cancer\s+risk',
        
        # General physiological impacts
        r'immune\s+(?:system|function|response)\s+(?:suppression|changes|impairment)',
        r'sleep\s+(?:disturbances|disorders|disruption)',
        r'vision\s+(?:changes|impairment|problems)',
        r'spatial\s+(?:disorientation|orientation)\s+(?:problems|issues)'
    ]
    
    # Find all matching patterns
    text_lower = text.lower()
    for pattern in impact_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            impact_term = match.group().strip()
            if impact_term not in impacts:
                impacts.append(impact_term)
    
    # Custom rule: look for 'loss' near 'bone' as requested
    loss_bone_pattern = r'(?:bone\s+\w*\s*){0,3}loss|loss\s+(?:\w+\s+){0,3}bone'
    bone_loss_matches = re.finditer(loss_bone_pattern, text_lower)
    for match in bone_loss_matches:
        context = match.group().strip()
        if 'bone' in context and 'loss' in context:
            impacts.append('bone loss')
            break
    
    return list(set(impacts))  # Remove duplicates


def _extract_custom_relations(doc: object) -> List[Dict]:
    """
    Extract relationships using dependency parsing.
    
    Looks for patterns like:
    - "X causes Y"
    - "X affects Y" 
    - "Y studied in Z"
    
    Args:
        doc: spaCy Doc object
        
    Returns:
        List of relationship dictionaries
    """
    relations = []
    
    if not doc:
        return relations
    
    try:
        # Look for causal relationships
        for token in doc:
            # Pattern: "X causes Y"
            if token.lemma_ in ['cause', 'lead', 'result']:
                # Find subject (what causes)
                subjects = [child for child in token.children if child.dep_ in ['nsubj', 'nsubjpass']]
                # Find object (what is caused)
                objects = [child for child in token.children if child.dep_ in ['dobj', 'pobj']]
                
                for subj in subjects:
                    for obj in objects:
                        relations.append({
                            'subject': subj.text,
                            'relation': 'causes',
                            'object': obj.text,
                            'confidence': 0.8
                        })
            
            # Pattern: "X affects Y"
            elif token.lemma_ in ['affect', 'impact', 'influence']:
                subjects = [child for child in token.children if child.dep_ in ['nsubj', 'nsubjpass']]
                objects = [child for child in token.children if child.dep_ in ['dobj', 'pobj']]
                
                for subj in subjects:
                    for obj in objects:
                        relations.append({
                            'subject': subj.text,
                            'relation': 'affects',
                            'object': obj.text,
                            'confidence': 0.7
                        })
            
            # Pattern: "studied in X"
            elif token.lemma_ == 'study' and token.dep_ == 'ROOT':
                # Find location/context
                prep_phrases = [child for child in token.children if child.dep_ == 'prep']
                for prep in prep_phrases:
                    if prep.text in ['in', 'during', 'aboard']:
                        objects = [child for child in prep.children if child.dep_ == 'pobj']
                        for obj in objects:
                            relations.append({
                                'subject': 'research',
                                'relation': 'studied_in',
                                'object': obj.text,
                                'confidence': 0.6
                            })
        
        return relations
        
    except Exception as e:
        log_error(f"Error extracting custom relations: {str(e)}")
        return []


def _fallback_entity_extraction(text: str) -> Dict:
    """
    Fallback entity extraction when spaCy is unavailable.
    
    Uses simple regex patterns to identify common entities and impacts.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with extracted information using simple patterns
    """
    result = {
        'entities': [],
        'impacts': [],
        'locations': [],
        'organizations': [],
        'custom_relations': []
    }
    
    # Simple patterns for common space biology entities
    org_patterns = [
        r'\bNASA\b', r'\bISS\b', r'\bInternational Space Station\b',
        r'\bSpaceX\b', r'\bESA\b', r'\bJAXA\b'
    ]
    
    location_patterns = [
        r'\bMars\b', r'\bMoon\b', r'\bEarth\b', r'\bspace\b',
        r'\bmicrogravity\b', r'\borbit\b'
    ]
    
    # Find organizations
    for pattern in org_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity_text = match.group()
            result['entities'].append((entity_text, 'ORG'))
            result['organizations'].append(entity_text)
    
    # Find locations
    for pattern in location_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity_text = match.group()
            result['entities'].append((entity_text, 'LOC'))
            result['locations'].append(entity_text)
    
    # Extract health impacts using the same function
    result['impacts'] = _extract_health_impacts(text)
    
    return result


def build_knowledge_graph(entities_list: List[Dict], experiment_ids: Optional[List[str]] = None) -> Optional[object]:
    """
    Build NetworkX knowledge graph from extracted entities and relationships.
    
    Creates a graph with nodes representing:
    - Experiments (Experiment:<id>)
    - Health impacts (Impact:<term>)
    - Results/findings (Result:<term>)
    - Locations (Location:<name>)
    
    And edges representing relationships like 'causes', 'affects', 'studied_in'.
    
    Args:
        entities_list: List of entity dictionaries from extract_entities()
        experiment_ids: Optional list of experiment IDs to associate with entities
        
    Returns:
        NetworkX graph object or None if NetworkX unavailable
        
    Example:
        >>> entities = [extract_entities(abstract1), extract_entities(abstract2)]
        >>> graph = build_knowledge_graph(entities, ['GLDS-001', 'GLDS-002'])
        >>> print(f"Graph has {graph.number_of_nodes()} nodes")
    """
    if not NETWORKX_AVAILABLE:
        log_error("NetworkX not available. Install with: pip install networkx")
        return None
    
    try:
        # Create directed graph
        G = nx.DiGraph()
        
        # Process each set of extracted entities
        for i, entities in enumerate(entities_list):
            # Create experiment node if ID provided
            exp_id = experiment_ids[i] if experiment_ids and i < len(experiment_ids) else f"EXP-{i+1}"
            exp_node = f"Experiment:{exp_id}"
            G.add_node(exp_node, type='experiment', id=exp_id)
            
            # Add impact nodes and relationships
            for impact in entities.get('impacts', []):
                impact_node = f"Impact:{impact}"
                G.add_node(impact_node, type='impact', term=impact)
                G.add_edge(exp_node, impact_node, relation='studies')
            
            # Add organization nodes
            for org in entities.get('organizations', []):
                org_node = f"Organization:{org}"
                G.add_node(org_node, type='organization', name=org)
                G.add_edge(exp_node, org_node, relation='conducted_by')
            
            # Add location nodes
            for location in entities.get('locations', []):
                loc_node = f"Location:{location}"
                G.add_node(loc_node, type='location', name=location)
                G.add_edge(exp_node, loc_node, relation='conducted_in')
            
            # Add custom relationships
            for relation in entities.get('custom_relations', []):
                subj_node = f"Entity:{relation['subject']}"
                obj_node = f"Entity:{relation['object']}"
                
                G.add_node(subj_node, type='entity', name=relation['subject'])
                G.add_node(obj_node, type='entity', name=relation['object'])
                G.add_edge(subj_node, obj_node, 
                          relation=relation['relation'],
                          confidence=relation.get('confidence', 0.5))
        
        log_info(f"✓ Built knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
        
    except Exception as e:
        log_error(f"Error building knowledge graph: {str(e)}")
        return None


def analyze_graph_structure(graph: object) -> Dict:
    """
    Analyze knowledge graph structure and extract insights.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        Dictionary with graph analysis results
    """
    if not graph or not NETWORKX_AVAILABLE:
        return {'error': 'Graph or NetworkX not available'}
    
    try:
        analysis = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'node_types': {},
            'top_connected_nodes': [],
            'isolated_nodes': list(nx.isolates(graph)),
            'density': nx.density(graph)
        }
        
        # Count node types
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            analysis['node_types'][node_type] = analysis['node_types'].get(node_type, 0) + 1
        
        # Find most connected nodes
        degrees = dict(graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        analysis['top_connected_nodes'] = top_nodes
        
        return analysis
        
    except Exception as e:
        log_error(f"Error analyzing graph: {str(e)}")
        return {'error': str(e)}


def build_kg(df: pd.DataFrame) -> Optional[object]:
    """
    Build knowledge graph from DataFrame with experiments, summaries, and entity relationships.
    
    Creates a comprehensive knowledge graph where:
    - Each row becomes an Experiment node with title and summary attributes
    - Entities extracted from summaries become Impact/Entity nodes
    - Simple causal relationships detected using 'causes' keyword
    - Graph serialized to 'data/graph.pkl' for persistence
    
    Args:
        df: DataFrame with columns 'id', 'title', 'summary' (and optionally 'abstract')
        
    Returns:
        NetworkX Graph object or None if NetworkX unavailable
        
    Example:
        >>> df = pd.DataFrame({
        ...     'id': ['GLDS-001', 'GLDS-002'], 
        ...     'title': ['Bone Study', 'Muscle Research'],
        ...     'summary': ['Microgravity causes bone loss', 'Exercise prevents muscle atrophy']
        ... })
        >>> graph = build_kg(df)
        >>> print(f"Graph saved with {graph.number_of_nodes()} nodes")
    """
    if not NETWORKX_AVAILABLE:
        log_error("NetworkX not available. Install with: pip install networkx")
        return None
    
    if df is None or df.empty:
        log_error("Cannot build KG from empty DataFrame")
        return None
    
    try:
        # Create directed graph for causal relationships
        G = nx.DiGraph()
        
        # Ensure required columns exist
        required_cols = ['id', 'title']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            log_error(f"Missing required columns: {missing_cols}")
            return None
        
        # Use summary if available, otherwise fallback to abstract
        text_col = 'summary' if 'summary' in df.columns else 'abstract'
        if text_col not in df.columns:
            log_error("DataFrame must contain 'summary' or 'abstract' column")
            return None
        
        log_info(f"Building knowledge graph from {len(df)} experiments using '{text_col}' column")
        
        # Process each experiment (DataFrame row)
        for idx, row in df.iterrows():
            # Create experiment node
            exp_id = row['id']
            exp_node = f"Experiment:{exp_id}"
            
            # Node attributes - include available fields
            node_attrs = {
                'type': 'experiment',
                'id': exp_id,
                'title': row['title']
            }
            
            # Add summary if available
            if text_col in df.columns and pd.notna(row[text_col]):
                node_attrs['summary'] = str(row[text_col])[:500]  # Limit length for storage
            
            # Add other metadata if available
            for col in ['abstract', 'metadata', 'impacts']:
                if col in df.columns and pd.notna(row[col]):
                    node_attrs[col] = str(row[col])[:200]  # Limit length
            
            G.add_node(exp_node, **node_attrs)
            
            # Extract entities from the text content
            text_content = row[text_col] if pd.notna(row[text_col]) else ""
            if text_content:
                entities = extract_entities(text_content)
                
                # Add Impact nodes from extracted health impacts
                for impact in entities.get('impacts', []):
                    impact_node = f"Impact:{impact}"
                    G.add_node(impact_node, 
                              type='impact', 
                              term=impact,
                              description=f"Health impact: {impact}")
                    
                    # Connect experiment to impact
                    G.add_edge(exp_node, impact_node, 
                              relation='studies',
                              confidence=0.8)
                
                # Add Organization nodes
                for org in entities.get('organizations', []):
                    org_node = f"Organization:{org}"
                    G.add_node(org_node,
                              type='organization',
                              name=org)
                    G.add_edge(exp_node, org_node,
                              relation='conducted_by',
                              confidence=0.7)
                
                # Add Location nodes
                for location in entities.get('locations', []):
                    loc_node = f"Location:{location}"
                    G.add_node(loc_node,
                              type='location', 
                              name=location)
                    G.add_edge(exp_node, loc_node,
                              relation='conducted_in',
                              confidence=0.6)
                
                # Simple causal relationship detection using 'causes' keyword
                text_lower = text_content.lower()
                sentences = text_content.split('.')
                
                for sentence in sentences:
                    sentence_lower = sentence.lower().strip()
                    if 'causes' in sentence_lower:
                        # Extract simple cause-effect relationships
                        cause_effect = _extract_simple_causes(sentence, entities)
                        for cause, effect in cause_effect:
                            cause_node = f"Entity:{cause}"
                            effect_node = f"Entity:{effect}"
                            
                            # Add cause and effect nodes if not already present
                            if not G.has_node(cause_node):
                                G.add_node(cause_node, type='entity', name=cause)
                            if not G.has_node(effect_node):
                                G.add_node(effect_node, type='entity', name=effect)
                            
                            # Add causal edge
                            G.add_edge(cause_node, effect_node, 
                                      relation='causes',
                                      confidence=0.9,
                                      source_experiment=exp_id,
                                      evidence=sentence.strip()[:100])
        
        # Serialize graph to pickle file
        output_file = 'data/graph.pkl'
        try:
            from pathlib import Path
            import pickle
            Path('data').mkdir(exist_ok=True)  # Ensure directory exists
            
            # Try NetworkX method first, fallback to pickle
            try:
                if hasattr(nx, 'write_gpickle'):
                    nx.write_gpickle(G, output_file)
                else:
                    # Fallback for newer NetworkX versions
                    with open(output_file, 'wb') as f:
                        pickle.dump(G, f)
                log_info(f"✓ Knowledge graph serialized to {output_file}")
            except Exception as nx_e:
                # Fallback to standard pickle
                with open(output_file, 'wb') as f:
                    pickle.dump(G, f)
                log_info(f"✓ Knowledge graph serialized to {output_file} (using pickle)")
        except Exception as e:
            log_error(f"Failed to serialize graph: {str(e)}")
        
        log_info(f"✓ Built knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Print graph statistics
        node_types = {}
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        log_info(f"Node distribution: {node_types}")
        
        return G
        
    except Exception as e:
        log_error(f"Error building knowledge graph: {str(e)}")
        return None


def _extract_simple_causes(sentence: str, entities: Dict) -> List[Tuple[str, str]]:
    """
    Extract simple cause-effect relationships from sentences containing 'causes'.
    
    Uses basic pattern matching to identify what causes what in a sentence.
    
    Args:
        sentence: Sentence containing 'causes'
        entities: Previously extracted entities for context
        
    Returns:
        List of (cause, effect) tuples
    """
    cause_effect_pairs = []
    
    try:
        sentence_lower = sentence.lower().strip()
        
        # Simple pattern: "X causes Y"
        if ' causes ' in sentence_lower:
            parts = sentence_lower.split(' causes ')
            if len(parts) >= 2:
                cause_part = parts[0].strip()
                effect_part = parts[1].strip()
                
                # Extract the main terms (simple noun extraction)
                cause_words = cause_part.split()[-3:]  # Last few words before 'causes'
                effect_words = effect_part.split()[:3]   # First few words after 'causes'
                
                cause = ' '.join([w for w in cause_words if w.isalpha()]).strip()
                effect = ' '.join([w for w in effect_words if w.isalpha()]).strip()
                
                # Clean up common stop words and punctuation
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
                
                cause_clean = ' '.join([w for w in cause.split() if w not in stop_words])
                effect_clean = ' '.join([w for w in effect.split() if w not in stop_words])
                
                if cause_clean and effect_clean:
                    cause_effect_pairs.append((cause_clean, effect_clean))
        
        # Also check for impact terms from entities
        impacts = entities.get('impacts', [])
        for impact in impacts:
            if impact.lower() in sentence_lower and ' causes ' in sentence_lower:
                # Find what causes this impact
                cause_idx = sentence_lower.find(' causes ')
                impact_idx = sentence_lower.find(impact.lower())
                
                if cause_idx < impact_idx:  # "X causes [impact]"
                    cause_part = sentence_lower[:cause_idx].strip()
                    cause_words = cause_part.split()[-2:]  # Last couple words
                    cause = ' '.join([w for w in cause_words if w.isalpha()])
                    if cause:
                        cause_effect_pairs.append((cause, impact))
                elif cause_idx > impact_idx:  # "[impact] causes Y" 
                    effect_part = sentence_lower[cause_idx + 8:].strip()  # After ' causes '
                    effect_words = effect_part.split()[:2]  # First couple words
                    effect = ' '.join([w for w in effect_words if w.isalpha()])
                    if effect:
                        cause_effect_pairs.append((impact, effect))
        
        return cause_effect_pairs
        
    except Exception as e:
        log_error(f"Error extracting causes from sentence: {str(e)}")
        return []


def load_kg(filepath: str = 'data/graph.pkl') -> Optional[object]:
    """
    Load serialized knowledge graph from pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        NetworkX graph object or None if loading fails
    """
    if not NETWORKX_AVAILABLE:
        log_error("NetworkX not available")
        return None
    
    try:
        from pathlib import Path
        import pickle
        
        if not Path(filepath).exists():
            log_error(f"Graph file {filepath} not found")
            return None
        
        # Try NetworkX method first, fallback to pickle
        try:
            if hasattr(nx, 'read_gpickle'):
                G = nx.read_gpickle(filepath)
            else:
                # Fallback for newer NetworkX versions
                with open(filepath, 'rb') as f:
                    G = pickle.load(f)
        except Exception:
            # Fallback to standard pickle
            with open(filepath, 'rb') as f:
                G = pickle.load(f)
        
        log_info(f"✓ Loaded knowledge graph from {filepath}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
        
    except Exception as e:
        log_error(f"Error loading knowledge graph: {str(e)}")
        return None


def query_kg(G: object, keyword: str) -> Dict:
    """
    Query knowledge graph for nodes containing keyword and return relevant subgraph.
    
    Searches for nodes containing the keyword (case-insensitive), finds their neighbors,
    and returns a subgraph dictionary with nodes, edges, and associated summaries.
    
    Args:
        G: NetworkX graph object
        keyword: Search keyword (case-insensitive)
        
    Returns:
        Dictionary containing:
        - matches: List of nodes that match the keyword
        - neighbors: All neighbor nodes of matches
        - subgraph_nodes: Combined set of matching nodes and their neighbors
        - edges: Edges within the subgraph
        - summaries: Experiment summaries for matched experiments
        - node_details: Detailed attributes for all subgraph nodes
        - statistics: Query result statistics
        
    Example:
        >>> result = query_kg(graph, "bone loss")
        >>> print(f"Found {len(result['matches'])} matching nodes")
        >>> for summary in result['summaries']:
        ...     print(f"Related experiment: {summary}")
    """
    if not G or not NETWORKX_AVAILABLE:
        return {
            'matches': [],
            'neighbors': [],
            'subgraph_nodes': [],
            'edges': [],
            'summaries': [],
            'node_details': {},
            'statistics': {'total_matches': 0, 'total_neighbors': 0, 'subgraph_size': 0},
            'error': 'Graph not available or NetworkX missing'
        }
    
    if not keyword or not keyword.strip():
        return {
            'matches': [],
            'neighbors': [],
            'subgraph_nodes': [],
            'edges': [],
            'summaries': [],
            'node_details': {},
            'statistics': {'total_matches': 0, 'total_neighbors': 0, 'subgraph_size': 0},
            'error': 'Empty keyword provided'
        }
    
    try:
        keyword_lower = keyword.lower().strip()
        
        # Find all nodes containing the keyword (case-insensitive)
        matching_nodes = []
        
        for node in G.nodes():
            node_str = str(node).lower()
            node_attrs = G.nodes.get(node, {})
            
            # Check node name/ID
            if keyword_lower in node_str:
                matching_nodes.append(node)
                continue
            
            # Check node attributes for keyword matches
            attrs_to_check = ['title', 'summary', 'abstract', 'name', 'term', 'description']
            for attr in attrs_to_check:
                if attr in node_attrs and node_attrs[attr]:
                    attr_value = str(node_attrs[attr]).lower()
                    if keyword_lower in attr_value:
                        matching_nodes.append(node)
                        break
        
        # Handle no matches case
        if not matching_nodes:
            log_info(f"No nodes found matching keyword: '{keyword}'")
            return {
                'matches': [],
                'neighbors': [],
                'subgraph_nodes': [],
                'edges': [],
                'summaries': [],
                'node_details': {},
                'statistics': {'total_matches': 0, 'total_neighbors': 0, 'subgraph_size': 0},
                'message': f"No matches found for keyword: '{keyword}'"
            }
        
        # Find all neighbors of matching nodes
        all_neighbors = set()
        for node in matching_nodes:
            neighbors = set(G.neighbors(node))
            all_neighbors.update(neighbors)
        
        # Remove matching nodes from neighbors (avoid duplication)
        neighbors_only = all_neighbors - set(matching_nodes)
        
        # Create subgraph with matches and neighbors
        subgraph_nodes = set(matching_nodes) | all_neighbors
        subgraph = G.subgraph(subgraph_nodes)
        
        # Extract edges within the subgraph
        subgraph_edges = []
        for edge in subgraph.edges(data=True):
            source, target, attrs = edge
            edge_info = {
                'source': source,
                'target': target,
                'relation': attrs.get('relation', 'unknown'),
                'confidence': attrs.get('confidence', 0.5)
            }
            # Add additional edge attributes
            for key, value in attrs.items():
                if key not in ['relation', 'confidence']:
                    edge_info[key] = value
            
            subgraph_edges.append(edge_info)
        
        # Extract summaries from experiment nodes in the subgraph
        summaries = []
        for node in subgraph_nodes:
            node_attrs = G.nodes.get(node, {})
            if node_attrs.get('type') == 'experiment':
                summary_info = {
                    'experiment_id': node_attrs.get('id', node),
                    'title': node_attrs.get('title', 'No title'),
                    'summary': node_attrs.get('summary', 'No summary available')
                }
                summaries.append(summary_info)
        
        # Collect detailed node information
        node_details = {}
        for node in subgraph_nodes:
            node_attrs = G.nodes.get(node, {})
            node_details[node] = {
                'type': node_attrs.get('type', 'unknown'),
                'attributes': node_attrs,
                'is_match': node in matching_nodes,
                'is_neighbor': node in neighbors_only,
                'degree': G.degree(node)
            }
        
        # Calculate statistics
        statistics = {
            'total_matches': len(matching_nodes),
            'total_neighbors': len(neighbors_only),
            'subgraph_size': len(subgraph_nodes),
            'subgraph_edges': len(subgraph_edges),
            'experiment_summaries': len(summaries),
            'keyword_searched': keyword
        }
        
        log_info(f"Query '{keyword}' found {len(matching_nodes)} matches, {len(neighbors_only)} neighbors")
        
        return {
            'matches': list(matching_nodes),
            'neighbors': list(neighbors_only),
            'subgraph_nodes': list(subgraph_nodes),
            'edges': subgraph_edges,
            'summaries': summaries,
            'node_details': node_details,
            'statistics': statistics,
            'subgraph': subgraph  # Include actual NetworkX subgraph for further analysis
        }
        
    except Exception as e:
        log_error(f"Error querying knowledge graph: {str(e)}")
        return {
            'matches': [],
            'neighbors': [],
            'subgraph_nodes': [],
            'edges': [],
            'summaries': [],
            'node_details': {},
            'statistics': {'total_matches': 0, 'total_neighbors': 0, 'subgraph_size': 0},
            'error': f"Query failed: {str(e)}"
        }


def get_kg_info() -> Dict:
    """
    Get information about knowledge graph capabilities.
    
    Returns:
        Dictionary with system capabilities and status
    """
    return {
        'spacy_available': SPACY_AVAILABLE,
        'networkx_available': NETWORKX_AVAILABLE,
        'nlp_model_loaded': nlp is not None,
        'supported_entities': ['ORG', 'GPE', 'LOC', 'PERSON', 'MISC'],
        'supported_relations': ['causes', 'affects', 'studied_in', 'conducted_by', 'conducted_in'],
        'health_impact_patterns': [
            'bone loss', 'muscle atrophy', 'radiation exposure',
            'cardiovascular deconditioning', 'immune suppression'
        ],
        'graph_serialization': 'nx.write_gpickle supported',
        'causal_detection': 'Simple "causes" keyword matching',
        'graph_querying': 'Keyword-based subgraph extraction with neighbors'
    }


# Main execution for testing and validation
if __name__ == '__main__':
    """
    Test knowledge graph builder functionality.
    """
    print("Testing knowledge graph builder...")
    
    # Test 1: Load NLP model
    print("\n1. Testing NLP model loading...")
    model_loaded = load_nlp_model()
    print(f"✓ NLP model loaded: {model_loaded}")
    
    # Test 2: Entity extraction
    print("\n2. Testing entity extraction...")
    test_text = """
    This study examines bone loss in astronauts during ISS missions. 
    Microgravity causes significant muscle atrophy and cardiovascular deconditioning.
    NASA research conducted aboard the International Space Station reveals 
    critical health impacts for future Mars exploration.
    """
    
    entities = extract_entities(test_text)
    print(f"Extracted entities: {len(entities['entities'])}")
    print(f"Health impacts: {entities['impacts']}")
    print(f"Organizations: {entities['organizations']}")
    print(f"Locations: {entities['locations']}")
    
    # Test 3: Knowledge graph building (original method)
    print("\n3. Testing original knowledge graph construction...")
    if NETWORKX_AVAILABLE:
        graph = build_knowledge_graph([entities], ['TEST-001'])
        if graph:
            analysis = analyze_graph_structure(graph)
            print(f"Original graph analysis: {analysis}")
    else:
        print("NetworkX not available - skipping graph construction test")
    
    # Test 4: DataFrame-based knowledge graph building (new method)
    print("\n4. Testing DataFrame-based knowledge graph (build_kg)...")
    if NETWORKX_AVAILABLE:
        # Create test DataFrame
        test_df = pd.DataFrame({
            'id': ['GLDS-001', 'GLDS-002', 'GLDS-003'],
            'title': [
                'Bone Loss in Microgravity',
                'Muscle Atrophy Study', 
                'Radiation Effects Research'
            ],
            'summary': [
                'Microgravity causes bone loss in astronauts during spaceflight missions.',
                'Prolonged weightlessness causes muscle atrophy and strength reduction.',
                'Space radiation causes DNA damage and increases cancer risk for crew members.'
            ]
        })
        
        kg_graph = build_kg(test_df)
        if kg_graph:
            kg_analysis = analyze_graph_structure(kg_graph)
            print(f"DataFrame KG analysis: {kg_analysis}")
            
            # Test loading the serialized graph
            loaded_graph = load_kg()
            if loaded_graph:
                print(f"✓ Successfully loaded serialized graph: {loaded_graph.number_of_nodes()} nodes")
                
                # Test querying the knowledge graph
                print("\n   Testing knowledge graph querying...")
                query_result = query_kg(kg_graph, "bone loss")
                print(f"   Query 'bone loss': {query_result['statistics']['total_matches']} matches, {query_result['statistics']['total_neighbors']} neighbors")
                
                # Test another query
                muscle_result = query_kg(kg_graph, "muscle")
                print(f"   Query 'muscle': {muscle_result['statistics']['total_matches']} matches, {muscle_result['statistics']['total_neighbors']} neighbors")
                
                # Test no matches case
                no_match_result = query_kg(kg_graph, "nonexistent")
                print(f"   Query 'nonexistent': {no_match_result.get('message', 'No message')}")
                
        else:
            print("DataFrame knowledge graph construction failed")
    else:
        print("NetworkX not available - skipping DataFrame KG test")
    
    # Test 5: System info
    print("\n5. System capabilities...")
    info = get_kg_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n🎉 Knowledge graph builder tests completed!")