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
    Enhanced entity extraction for space biology research abstracts.
    
    Extracts comprehensive entities including organisms, experimental conditions,
    measurements, space-specific terms, and multiple relationship types.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing:
        - entities: List of (text, label) tuples from spaCy NER and custom patterns
        - impacts: Health and biological impact terms
        - locations: Spatial and environmental locations
        - organizations: Research organizations and agencies
        - organisms: Scientific organism names and common names
        - experimental_conditions: Spaceflight, ground control, radiation conditions
        - measurements: Gene expression, protein levels, physiological measurements
        - space_terms: Microgravity, ISS, spacecraft-specific terminology
        - causal_relations: "causes", "leads to", "results in" relationships
        - experimental_relations: "compared to", "versus" relationships
        - temporal_relations: "during", "after", "before" relationships
        - location_relations: "in space", "on ISS" relationships
        - mitigation_strategies: Exercise, nutrition, medication countermeasures
        
    Example:
        >>> result = extract_entities("Microgravity causes bone loss in Mus musculus during ISS missions")
        >>> print(result['impacts'])  # ['bone loss']
        >>> print(result['organisms'])  # ['Mus musculus']
        >>> print(result['space_terms'])  # ['microgravity', 'ISS']
    """
    if not text or not text.strip():
        return {
            'entities': [], 'impacts': [], 'locations': [], 'organizations': [],
            'organisms': [], 'experimental_conditions': [], 'measurements': [],
            'space_terms': [], 'causal_relations': [], 'experimental_relations': [],
            'temporal_relations': [], 'location_relations': [], 'mitigation_strategies': []
        }
    
    # Initialize comprehensive result structure
    result = {
        'entities': [], 'impacts': [], 'locations': [], 'organizations': [],
        'organisms': [], 'experimental_conditions': [], 'measurements': [],
        'space_terms': [], 'causal_relations': [], 'experimental_relations': [],
        'temporal_relations': [], 'location_relations': [], 'mitigation_strategies': []
    }
    
    # Fallback processing if spaCy not available
    if not SPACY_AVAILABLE or nlp is None:
        return _enhanced_fallback_extraction(text)
    
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
        
        # Enhanced custom entity extraction
        result['impacts'] = _extract_enhanced_health_impacts(text, doc)
        result['organisms'] = _extract_organisms(text, doc)
        result['experimental_conditions'] = _extract_experimental_conditions(text, doc)
        result['measurements'] = _extract_measurements(text, doc)
        result['space_terms'] = _extract_space_terms(text, doc)
        result['mitigation_strategies'] = _extract_mitigation_strategies(text, doc)
        
        # Enhanced relationship extraction
        result['causal_relations'] = _extract_causal_relations(text, doc)
        result['experimental_relations'] = _extract_experimental_relations(text, doc)
        result['temporal_relations'] = _extract_temporal_relations(text, doc)
        result['location_relations'] = _extract_location_relations(text, doc)
        
        # Add custom patterns to locations and organizations
        result['locations'].extend(_extract_space_locations(text))
        result['organizations'].extend(_extract_space_organizations(text))
        
        # Remove duplicates from string lists (not relationship dictionaries)
        string_categories = ['impacts', 'locations', 'organizations', 'organisms', 
                           'experimental_conditions', 'measurements', 'space_terms', 'mitigation_strategies']
        for key in string_categories:
            if key in result and isinstance(result[key], list):
                result[key] = list(set(result[key]))
        
        # Remove duplicate tuples from entities
        if 'entities' in result:
            result['entities'] = list(set(result['entities']))
        
        total_entities = sum(len(v) for v in result.values() if isinstance(v, list))
        log_info(f"Enhanced extraction: {total_entities} total entities across {len(result)} categories")
        return result
        
    except Exception as e:
        log_error(f"Error in entity extraction: {str(e)}")
        return _fallback_entity_extraction(text)


def _extract_enhanced_health_impacts(text: str, doc: Optional[object] = None) -> List[str]:
    """
    Extract comprehensive health and biological impact terms.
    
    Args:
        text: Input text
        doc: spaCy Doc object (optional)
        
    Returns:
        List of identified health impact terms
    """
    impacts = []
    text_lower = text.lower()
    
    # Comprehensive health impact patterns
    impact_patterns = [
        # Bone and skeletal impacts
        r'bone\s+(?:loss|density|deterioration|demineralization|weakening|resorption)',
        r'(?:decreased|reduced|loss\s+of)\s+bone\s+(?:density|mineral|mass|formation)',
        r'osteo(?:porosis|penia|blast|clast)',
        r'skeletal\s+(?:deconditioning|changes|adaptations)',
        r'trabecular\s+(?:bone|thinning|loss)',
        r'cortical\s+(?:bone|thinning|porosity)',
        
        # Muscle and strength impacts
        r'muscle\s+(?:atrophy|wasting|weakness|deterioration|loss|deconditioning)',
        r'(?:decreased|reduced|loss\s+of)\s+muscle\s+(?:mass|strength|function|volume)',
        r'sarcopenia|myofiber\s+(?:atrophy|changes)',
        r'strength\s+(?:loss|reduction|decline)',
        r'exercise\s+(?:capacity|tolerance)\s+(?:reduction|decline)',
        
        # Cardiovascular impacts
        r'cardiovascular\s+(?:deconditioning|changes|adaptation|dysfunction)',
        r'heart\s+(?:rate|function|performance|muscle)\s+(?:changes|alterations|deconditioning)',
        r'blood\s+pressure\s+(?:changes|regulation|orthostatic)',
        r'cardiac\s+(?:output|function|deconditioning)\s+(?:reduction|changes)',
        r'orthostatic\s+(?:intolerance|hypotension)',
        
        # Radiation and DNA impacts
        r'radiation\s+(?:exposure|damage|effects|induced)',
        r'DNA\s+(?:damage|repair|mutation|strand\s+breaks)',
        r'cancer\s+(?:risk|incidence|development)',
        r'chromosomal\s+(?:aberrations|damage)',
        r'oxidative\s+(?:stress|damage)',
        
        # Neurological and vision impacts
        r'vision\s+(?:changes|impairment|problems|loss)',
        r'visual\s+(?:acuity|field)\s+(?:changes|loss)',
        r'optic\s+(?:disc|nerve)\s+(?:swelling|changes)',
        r'spatial\s+(?:disorientation|orientation)\s+(?:problems|issues)',
        r'neuro(?:cognitive|behavioral)\s+(?:changes|impairment)',
        r'brain\s+(?:volume|changes|structure)\s+(?:loss|alterations)',
        
        # Immune system impacts
        r'immune\s+(?:system|function|response)\s+(?:suppression|changes|impairment|dysfunction)',
        r'immunosuppression|immune\s+deficiency',
        r'T-cell\s+(?:function|count|response)\s+(?:reduction|impairment)',
        r'cytokine\s+(?:production|response)\s+(?:changes|alterations)',
        
        # Sleep and circadian impacts
        r'sleep\s+(?:disturbances|disorders|disruption|quality)\s+(?:changes|reduction)',
        r'circadian\s+(?:rhythm|cycle)\s+(?:disruption|changes)',
        r'sleep-wake\s+cycle\s+(?:disruption|alterations)',
        
        # Kidney and fluid impacts
        r'kidney\s+(?:stone|function)\s+(?:formation|changes|impairment)',
        r'renal\s+(?:function|calculi)\s+(?:changes|formation)',
        r'fluid\s+(?:shift|redistribution)',
        r'dehydration|fluid\s+loss',
        
        # Gene expression and protein impacts
        r'gene\s+expression\s+(?:changes|alterations|upregulation|downregulation)',
        r'protein\s+(?:levels|expression|modifications)\s+(?:changes|alterations)',
        r'transcriptional\s+(?:changes|regulation|activity)',
        r'enzymatic\s+activity\s+(?:changes|reduction|enhancement)'
    ]
    
    # Find all matching patterns
    for pattern in impact_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            impact_term = match.group().strip()
            if impact_term and impact_term not in impacts:
                impacts.append(impact_term)
    
    return impacts


def _extract_organisms(text: str, doc: Optional[object] = None) -> List[str]:
    """Extract organism names from text using scientific and common names."""
    organisms = []
    text_lower = text.lower()
    
    # Scientific organism name patterns
    organism_patterns = [
        # Common lab organisms
        r'mus\s+musculus',
        r'rattus\s+norvegicus', 
        r'arabidopsis\s+thaliana',
        r'drosophila\s+melanogaster',
        r'caenorhabditis\s+elegans',
        r'saccharomyces\s+cerevisiae',
        r'escherichia\s+coli',
        r'homo\s+sapiens',
        
        # Common names
        r'\bmice?\b|\bmouse\b',
        r'\brats?\b',
        r'\bflies\b|\bfly\b',
        r'\bworms?\b',
        r'\byeast\b',
        r'\bbacteria\b',
        r'\bhumans?\b|\bpeople\b',
        r'\bastronauts?\b|\bcrewmembers?\b',
        
        # Plant terms
        r'seedlings?\b',
        r'plants?\b',
        r'arabidopsis\b'
    ]
    
    for pattern in organism_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            organism = match.group().strip()
            if organism and organism not in organisms:
                organisms.append(organism)
    
    return organisms


def _extract_experimental_conditions(text: str, doc: Optional[object] = None) -> List[str]:
    """Extract experimental conditions and treatments."""
    conditions = []
    text_lower = text.lower()
    
    condition_patterns = [
        # Spaceflight conditions
        r'spaceflight\b|space\s+flight\b',
        r'microgravity\b|μg\b|μ-g\b',
        r'hypergravity\b|hyperg\b',
        r'ground\s+control\b',
        r'flight\s+(?:group|animals|samples)',
        r'control\s+(?:group|animals|samples)',
        r'vivarium\s+control',
        r'habitat\s+control',
        
        # Radiation conditions
        r'radiation\s+exposure\b',
        r'cosmic\s+(?:radiation|rays?)',
        r'galactic\s+cosmic\s+rays?',
        r'solar\s+particle\s+events?',
        r'proton\s+(?:radiation|exposure)',
        r'gamma\s+(?:radiation|rays?)',
        r'heavy\s+ion\s+(?:radiation|exposure)',
        r'iron\s+ion\s+(?:radiation|exposure)',
        
        # Environmental conditions
        r'simulated\s+(?:microgravity|spaceflight)',
        r'clinostat\b|clinorotation\b',
        r'random\s+positioning\s+machine',
        r'hindlimb\s+(?:unloading|suspension)',
        r'head-down\s+tilt\b',
        r'bed\s+rest\b',
        
        # Duration terms
        r'\d+\s+(?:days?|weeks?|months?)\s+(?:of|in|during)',
        r'(?:short|long)-(?:term|duration)',
        r'acute\s+exposure\b',
        r'chronic\s+exposure\b'
    ]
    
    for pattern in condition_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            condition = match.group().strip()
            if condition and condition not in conditions:
                conditions.append(condition)
    
    return conditions


def _extract_measurements(text: str, doc: Optional[object] = None) -> List[str]:
    """Extract measurement types and methodologies.""" 
    measurements = []
    text_lower = text.lower()
    
    measurement_patterns = [
        # Gene expression measurements
        r'gene\s+expression\b',
        r'mRNA\s+(?:levels?|expression)',
        r'RNA-seq\b|RNA\s+sequencing',
        r'microarray\s+analysis',
        r'qPCR\b|quantitative\s+PCR',
        r'RT-PCR\b',
        r'transcriptome\s+analysis',
        
        # Protein measurements
        r'protein\s+(?:levels?|expression|content)',
        r'proteomics?\b|proteomic\s+analysis',
        r'mass\s+spectrometry',
        r'western\s+blot(?:ting)?',
        r'immunoblot(?:ting)?',
        r'ELISA\b',
        r'immunofluorescence\b',
        
        # Physiological measurements
        r'bone\s+(?:density|mineral\s+density)',
        r'(?:micro-?)?CT\s+(?:scan|analysis)',
        r'DXA\s+scan\b|DEXA\b',
        r'muscle\s+(?:mass|volume|cross-sectional\s+area)',
        r'grip\s+strength\b',
        r'(?:maximum|peak)\s+(?:force|torque)',
        
        # Cardiovascular measurements
        r'heart\s+rate\b|HR\b',
        r'blood\s+pressure\b|BP\b',
        r'cardiac\s+output\b',
        r'stroke\s+volume\b',
        r'echocardiography\b',
        r'ECG\b|electrocardiogram',
        
        # Blood/biochemical measurements
        r'blood\s+(?:glucose|pressure|flow)',
        r'serum\s+(?:levels?|biomarkers?)',
        r'plasma\s+(?:levels?|concentration)',
        r'biomarkers?\b',
        r'hormones?\s+levels?',
        
        # Behavioral measurements
        r'behavioral\s+(?:tests?|analysis)',
        r'cognitive\s+(?:function|performance)',
        r'motor\s+(?:function|activity)',
        r'locomotor\s+activity\b'
    ]
    
    for pattern in measurement_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            measurement = match.group().strip()
            if measurement and measurement not in measurements:
                measurements.append(measurement)
    
    return measurements


def _extract_space_terms(text: str, doc: Optional[object] = None) -> List[str]:
    """Extract space-specific terminology."""
    space_terms = []
    text_lower = text.lower()
    
    space_patterns = [
        # Space vehicles and locations
        r'\bISS\b|International\s+Space\s+Station',
        r'space\s+shuttle\b',
        r'SpaceX\b|Dragon\b',
        r'Soyuz\b',
        r'Mir\s+(?:space\s+)?station',
        
        # Space environments
        r'microgravity\b|μg\b',
        r'weightlessness\b',
        r'zero\s+(?:gravity|g)',
        r'space\s+environment',
        r'orbital\s+(?:environment|flight)',
        
        # Mission terms
        r'spaceflight\b|space\s+flight',
        r'EVA\b|extravehicular\s+activity',
        r'spacewalk\b|space\s+walk',
        r'mission\s+duration',
        r'expedition\s+\d+',
        r'crew\s+(?:member|rotation)',
        
        # Planetary terms
        r'\bMars\b|Martian\b',
        r'\bMoon\b|lunar\b',
        r'deep\s+space\b',
        r'interplanetary\b',
        
        # Research platforms
        r'Rodent\s+Research\b',
        r'Advanced\s+Plant\s+Habitat',
        r'Vegetable\s+Production\s+System',
        r'Cell\s+Culture\s+Module'
    ]
    
    for pattern in space_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            term = match.group().strip()
            if term and term not in space_terms:
                space_terms.append(term)
    
    return space_terms


def _extract_mitigation_strategies(text: str, doc: Optional[object] = None) -> List[str]:
    """Extract countermeasures and mitigation strategies."""
    strategies = []
    text_lower = text.lower()
    
    strategy_patterns = [
        # Exercise countermeasures
        r'exercise\s+(?:countermeasures?|protocols?|training)',
        r'resistance\s+(?:training|exercise)',
        r'aerobic\s+(?:training|exercise)',
        r'treadmill\s+(?:exercise|training)',
        r'cycle\s+ergometer\b',
        r'ARED\b|Advanced\s+Resistive\s+Exercise\s+Device',
        
        # Nutritional interventions
        r'nutritional\s+(?:supplements?|interventions?)',
        r'dietary\s+(?:supplements?|modifications?)',
        r'vitamin\s+D\s+(?:supplementation|therapy)',
        r'calcium\s+(?:supplementation|intake)',
        r'protein\s+(?:supplementation|intake)',
        r'bisphosphonates?\b',
        
        # Pharmacological interventions
        r'drug\s+(?:therapy|treatment|intervention)',
        r'medication\b|pharmaceuticals?',
        r'therapeutic\s+(?:agents?|interventions?)',
        
        # Physical interventions  
        r'artificial\s+gravity\b',
        r'centrifuge\s+(?:training|exposure)',
        r'lower\s+body\s+negative\s+pressure',
        r'compression\s+(?:garments?|suits?)',
        
        # Preventive measures
        r'preventive\s+measures?',
        r'countermeasures?\b',
        r'mitigation\s+strategies?',
        r'protective\s+(?:measures?|equipment)'
    ]
    
    for pattern in strategy_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            strategy = match.group().strip()
            if strategy and strategy not in strategies:
                strategies.append(strategy)
    
    return strategies


def _extract_space_locations(text: str) -> List[str]:
    """Extract space-specific locations."""
    locations = []
    text_lower = text.lower()
    
    location_patterns = [
        r'\bISS\b|International\s+Space\s+Station',
        r'\bMars\b|Martian\s+(?:surface|environment)',
        r'\bMoon\b|lunar\s+(?:surface|environment)', 
        r'space\s+(?:station|environment|habitat)',
        r'orbital\s+(?:laboratory|platform)',
        r'deep\s+space\b',
        r'low\s+Earth\s+orbit\b|LEO\b',
        r'microgravity\s+environment'
    ]
    
    for pattern in location_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            location = match.group().strip()
            if location and location not in locations:
                locations.append(location)
    
    return locations


def _extract_space_organizations(text: str) -> List[str]:
    """Extract space-related organizations."""
    organizations = []
    text_lower = text.lower()
    
    org_patterns = [
        r'\bNASA\b',
        r'\bESA\b|European\s+Space\s+Agency',
        r'\bJAXA\b|Japan\s+Aerospace\s+Exploration\s+Agency',
        r'SpaceX\b',
        r'Blue\s+Origin\b',
        r'Boeing\b',
        r'Roscosmos\b',
        r'CSA\b|Canadian\s+Space\s+Agency'
    ]
    
    for pattern in org_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            org = match.group().strip()
            if org and org not in organizations:
                organizations.append(org)
    
    return organizations


def _extract_causal_relations(text: str, doc: Optional[object] = None) -> List[Dict]:
    """Extract causal relationships from text."""
    relations = []
    text_lower = text.lower()
    
    # Causal relationship patterns  
    causal_patterns = [
        (r'(.{1,50})\s+causes?\s+(.{1,50})', 'causes'),
        (r'(.{1,50})\s+leads?\s+to\s+(.{1,50})', 'leads_to'),
        (r'(.{1,50})\s+results?\s+in\s+(.{1,50})', 'results_in'),
        (r'(.{1,50})\s+induces?\s+(.{1,50})', 'induces'),
        (r'(.{1,50})\s+triggers?\s+(.{1,50})', 'triggers'),
        (r'(.{1,50})\s+produces?\s+(.{1,50})', 'produces'),
        (r'due\s+to\s+(.{1,50}),?\s*(.{1,50})', 'due_to'),
        (r'(.{1,50})\s+is\s+caused\s+by\s+(.{1,50})', 'caused_by')
    ]
    
    for pattern, relation_type in causal_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            try:
                cause = match.group(1).strip()
                effect = match.group(2).strip()
                if cause and effect and len(cause) > 3 and len(effect) > 3:
                    relations.append({
                        'subject': cause,
                        'relation': relation_type,
                        'object': effect,
                        'confidence': 0.8,
                        'type': 'causal'
                    })
            except (IndexError, AttributeError):
                continue
    
    return relations


def _extract_experimental_relations(text: str, doc: Optional[object] = None) -> List[Dict]:
    """Extract experimental comparison relationships."""
    relations = []
    text_lower = text.lower()
    
    experimental_patterns = [
        (r'(.{1,50})\s+compared\s+to\s+(.{1,50})', 'compared_to'),
        (r'(.{1,50})\s+versus\s+(.{1,50})', 'versus'),
        (r'(.{1,50})\s+vs\.?\s+(.{1,50})', 'versus'),
        (r'(.{1,50})\s+relative\s+to\s+(.{1,50})', 'relative_to'),
        (r'(.{1,50})\s+in\s+contrast\s+to\s+(.{1,50})', 'contrasts_with'),
        (r'(.{1,50})\s+differs?\s+from\s+(.{1,50})', 'differs_from'),
        (r'(.{1,50})\s+(?:group|condition)\s+and\s+(.{1,50})\s+(?:group|condition)', 'experimental_groups')
    ]
    
    for pattern, relation_type in experimental_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            try:
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()
                if entity1 and entity2 and len(entity1) > 3 and len(entity2) > 3:
                    relations.append({
                        'subject': entity1,
                        'relation': relation_type,
                        'object': entity2,
                        'confidence': 0.7,
                        'type': 'experimental'
                    })
            except (IndexError, AttributeError):
                continue
    
    return relations


def _extract_temporal_relations(text: str, doc: Optional[object] = None) -> List[Dict]:
    """Extract temporal relationships."""
    relations = []
    text_lower = text.lower()
    
    temporal_patterns = [
        (r'during\s+(.{1,50}),?\s*(.{1,50})', 'during'),
        (r'after\s+(.{1,50}),?\s*(.{1,50})', 'after'),  
        (r'before\s+(.{1,50}),?\s*(.{1,50})', 'before'),
        (r'following\s+(.{1,50}),?\s*(.{1,50})', 'following'),
        (r'prior\s+to\s+(.{1,50}),?\s*(.{1,50})', 'prior_to'),
        (r'at\s+(\d+\s+(?:days?|weeks?|months?)),?\s*(.{1,50})', 'at_timepoint'),
        (r'(\w+)\s+occurs?\s+during\s+(.{1,50})', 'occurs_during')
    ]
    
    for pattern, relation_type in temporal_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            try:
                time_ref = match.group(1).strip()
                event = match.group(2).strip()
                if time_ref and event and len(time_ref) > 2 and len(event) > 3:
                    relations.append({
                        'subject': event,
                        'relation': relation_type,
                        'object': time_ref,
                        'confidence': 0.6,
                        'type': 'temporal'
                    })
            except (IndexError, AttributeError):
                continue
    
    return relations


def _extract_location_relations(text: str, doc: Optional[object] = None) -> List[Dict]:
    """Extract location-based relationships."""
    relations = []
    text_lower = text.lower()
    
    location_patterns = [
        (r'(.{1,50})\s+in\s+space\b', 'in_space'),
        (r'(.{1,50})\s+on\s+(?:the\s+)?ISS\b', 'on_ISS'),
        (r'(.{1,50})\s+during\s+(?:space)?flight\b', 'during_flight'), 
        (r'(.{1,50})\s+in\s+microgravity\b', 'in_microgravity'),
        (r'(.{1,50})\s+on\s+(?:the\s+)?(?:space\s+)?station\b', 'on_station'),
        (r'(.{1,50})\s+in\s+orbit\b', 'in_orbit'),
        (r'(.{1,50})\s+aboard\s+(.{1,50})', 'aboard'),
        (r'(.{1,50})\s+on\s+Mars\b', 'on_Mars')
    ]
    
    for pattern, relation_type in location_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            try:
                if relation_type == 'aboard':
                    entity = match.group(1).strip()
                    location = match.group(2).strip()
                else:
                    entity = match.group(1).strip()
                    location = relation_type.replace('_', ' ')
                    
                if entity and len(entity) > 3:
                    relations.append({
                        'subject': entity,
                        'relation': relation_type,
                        'object': location,
                        'confidence': 0.7,
                        'type': 'location'
                    })
            except (IndexError, AttributeError):
                continue
    
    return relations


def _enhanced_fallback_extraction(text: str) -> Dict:
    """Enhanced fallback entity extraction when spaCy is unavailable."""
    result = {
        'entities': [], 'impacts': [], 'locations': [], 'organizations': [],
        'organisms': [], 'experimental_conditions': [], 'measurements': [],
        'space_terms': [], 'causal_relations': [], 'experimental_relations': [],
        'temporal_relations': [], 'location_relations': [], 'mitigation_strategies': []
    }
    
    # Use all the new extraction functions
    result['impacts'] = _extract_enhanced_health_impacts(text)
    result['organisms'] = _extract_organisms(text)
    result['experimental_conditions'] = _extract_experimental_conditions(text)
    result['measurements'] = _extract_measurements(text)
    result['space_terms'] = _extract_space_terms(text)
    result['mitigation_strategies'] = _extract_mitigation_strategies(text)
    result['locations'] = _extract_space_locations(text)
    result['organizations'] = _extract_space_organizations(text)
    
    # Relationships
    result['causal_relations'] = _extract_causal_relations(text)
    result['experimental_relations'] = _extract_experimental_relations(text)
    result['temporal_relations'] = _extract_temporal_relations(text)
    result['location_relations'] = _extract_location_relations(text)
    
    # Combine into entities list
    for category in ['organisms', 'space_terms', 'measurements']:
        for item in result[category]:
            result['entities'].append((item, category.upper()))
    
    return result


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
    result['impacts'] = _extract_enhanced_health_impacts(text)
    
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
                try:
                    text_lower = text_content.lower()
                    sentences = [s.strip() for s in text_content.split('.') if s.strip()]
                except (AttributeError, TypeError) as e:
                    log_error(f"Error processing text for causal analysis: {str(e)}")
                    sentences = []
                
                for sentence in sentences:
                    try:
                        if not sentence or not isinstance(sentence, str):
                            continue
                            
                        sentence_lower = sentence.lower().strip()
                        if len(sentence_lower) > 0 and 'causes' in sentence_lower:
                            # Extract simple cause-effect relationships with error handling
                            try:
                                cause_effect = _extract_simple_causes(sentence, entities)
                                if not isinstance(cause_effect, list):
                                    continue
                            except Exception as e:
                                log_error(f"Error extracting causes from sentence: {str(e)}")
                                continue
                                
                            for cause_effect_pair in cause_effect:
                                try:
                                    if not isinstance(cause_effect_pair, (tuple, list)) or len(cause_effect_pair) != 2:
                                        continue
                                    cause, effect = cause_effect_pair
                                    
                                    if not cause or not effect:  # Skip if either is empty
                                        continue
                                        
                                    cause_node = f"Entity:{str(cause).strip()}"
                                    effect_node = f"Entity:{str(effect).strip()}"
                                    
                                    # Add cause and effect nodes if not already present
                                    if not G.has_node(cause_node):
                                        G.add_node(cause_node, type='entity', name=str(cause).strip())
                                    if not G.has_node(effect_node):
                                        G.add_node(effect_node, type='entity', name=str(effect).strip())
                                    
                                    # Add causal edge with safe string handling
                                    evidence_text = str(sentence).strip()[:100] if sentence else ""
                                    G.add_edge(cause_node, effect_node, 
                                              relation='causes',
                                              confidence=0.9,
                                              source_experiment=exp_id,
                                              evidence=evidence_text)
                                              
                                except (ValueError, TypeError, Exception) as e:
                                    log_error(f"Error adding causal relationship: {str(e)}")
                                    continue
                                    
                    except Exception as e:
                        log_error(f"Error processing sentence for causal relationships: {str(e)}")
                        continue
        
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
        
        # Handle no matches case - use len() to avoid array comparison issues
        if len(matching_nodes) == 0:
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