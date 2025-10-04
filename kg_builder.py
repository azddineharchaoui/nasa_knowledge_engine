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


def _get_organism_taxonomy(organism_name: str) -> str:
    """Get basic taxonomy information for known organisms."""
    taxonomy_map = {
        'mus musculus': 'Mammalia > Rodentia > Muridae',
        'mice': 'Mammalia > Rodentia > Muridae',
        'mouse': 'Mammalia > Rodentia > Muridae',
        'rattus norvegicus': 'Mammalia > Rodentia > Muridae',
        'rat': 'Mammalia > Rodentia > Muridae',
        'rats': 'Mammalia > Rodentia > Muridae',
        'arabidopsis thaliana': 'Plantae > Brassicales > Brassicaceae',
        'arabidopsis': 'Plantae > Brassicales > Brassicaceae',
        'plants': 'Plantae',
        'plant': 'Plantae',
        'human': 'Mammalia > Primates > Hominidae',
        'humans': 'Mammalia > Primates > Hominidae'
    }
    return taxonomy_map.get(organism_name.lower(), 'Unknown taxonomy')


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
    Comprehensive analysis of knowledge graph structure with advanced metrics.
    
    Provides detailed analytics including:
    - Basic graph metrics (nodes, edges, density)
    - Centrality analysis (degree, betweenness, closeness)
    - Hub identification and ranking
    - Community/cluster detection
    - Connectivity analysis
    - Research domain analysis
    - Biological pathway insights
    
    Args:
        graph: NetworkX graph object with rich node/edge attributes
        
    Returns:
        Dictionary with comprehensive graph analysis results including:
        - Basic metrics, centrality measures, hub analysis
        - Community structure, pathway analysis, research insights
    """
    if not graph or not NETWORKX_AVAILABLE:
        return {'error': 'Graph or NetworkX not available'}
    
    try:
        # Basic graph metrics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph)
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        
        analysis = {
            'basic_metrics': {
                'nodes': num_nodes,
                'edges': num_edges,
                'density': density,
                'average_degree': avg_degree,
                'is_connected': nx.is_connected(graph.to_undirected()) if num_nodes > 0 else False
            },
            'node_analysis': {},
            'centrality_analysis': {},
            'hub_analysis': {},
            'community_analysis': {},
            'research_insights': {},
            'pathway_analysis': {}
        }
        
        # Node type distribution and hierarchy analysis
        node_types = {}
        node_levels = {}
        hub_nodes = []
        
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            node_level = attrs.get('node_level', 0)
            hub_score = attrs.get('hub_score', 0)
            
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            if node_level in node_levels:
                node_levels[node_level] += 1
            else:
                node_levels[node_level] = 1
            
            if attrs.get('is_hub', False) or hub_score > 2:
                hub_nodes.append({
                    'node': node,
                    'type': node_type,
                    'hub_score': hub_score,
                    'degree': graph.degree(node),
                    'frequency': attrs.get('frequency', 1)
                })
        
        analysis['node_analysis'] = {
            'node_types': node_types,
            'hierarchy_levels': node_levels,
            'isolated_nodes': list(nx.isolates(graph))
        }
        
        # Centrality analysis
        try:
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph)
            
            # Get top nodes by centrality
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            analysis['centrality_analysis'] = {
                'top_degree_centrality': top_degree,
                'top_betweenness_centrality': top_betweenness,
                'avg_degree_centrality': sum(degree_centrality.values()) / len(degree_centrality) if degree_centrality else 0,
                'avg_betweenness_centrality': sum(betweenness_centrality.values()) / len(betweenness_centrality) if betweenness_centrality else 0
            }
        except Exception as e:
            log_error(f"Error calculating centrality: {str(e)}")
            analysis['centrality_analysis'] = {'error': str(e)}
        
        # Hub analysis
        hub_nodes.sort(key=lambda x: x['hub_score'], reverse=True)
        analysis['hub_analysis'] = {
            'total_hubs': len(hub_nodes),
            'top_hubs': hub_nodes[:10],
            'hub_types': {}
        }
        
        for hub in hub_nodes:
            hub_type = hub['type']
            if hub_type in analysis['hub_analysis']['hub_types']:
                analysis['hub_analysis']['hub_types'][hub_type] += 1
            else:
                analysis['hub_analysis']['hub_types'][hub_type] = 1
        
        # Community/cluster analysis
        try:
            if hasattr(nx, 'community') and num_nodes > 2:
                G_undirected = graph.to_undirected()
                communities = nx.community.greedy_modularity_communities(G_undirected)
                
                # Analyze community composition
                community_info = []
                for i, community in enumerate(communities):
                    community_types = {}
                    for node in community:
                        if graph.has_node(node):
                            node_type = graph.nodes[node].get('type', 'unknown')
                            community_types[node_type] = community_types.get(node_type, 0) + 1
                    
                    community_info.append({
                        'community_id': i,
                        'size': len(community),
                        'node_types': community_types,
                        'nodes': list(community)[:5]  # Sample nodes
                    })
                
                analysis['community_analysis'] = {
                    'num_communities': len(communities),
                    'communities': community_info,
                    'modularity': nx.community.modularity(G_undirected, communities) if communities else 0
                }
        except Exception as e:
            log_error(f"Error in community analysis: {str(e)}")
            analysis['community_analysis'] = {'error': str(e)}
        
        # Research insights analysis
        experiments = [n for n in graph.nodes() if n.startswith('Experiment:')]
        impacts = [n for n in graph.nodes() if n.startswith('Impact:')]
        organisms = [n for n in graph.nodes() if n.startswith('Organism:')]
        conditions = [n for n in graph.nodes() if n.startswith('Condition:')]
        
        # Find most studied combinations
        organism_impact_pairs = []
        for org in organisms[:5]:  # Limit for performance
            for impact in impacts[:5]:
                # Check if both are studied in same experiments
                org_experiments = set(graph.predecessors(org))
                impact_experiments = set(graph.predecessors(impact))
                shared_exps = org_experiments & impact_experiments
                if shared_exps:
                    organism_impact_pairs.append({
                        'organism': org.split(':', 1)[1],
                        'impact': impact.split(':', 1)[1],
                        'shared_experiments': len(shared_exps),
                        'experiments': list(shared_exps)[:3]
                    })
        
        organism_impact_pairs.sort(key=lambda x: x['shared_experiments'], reverse=True)
        
        analysis['research_insights'] = {
            'total_experiments': len(experiments),
            'total_impacts': len(impacts),
            'total_organisms': len(organisms),
            'total_conditions': len(conditions),
            'most_studied_combinations': organism_impact_pairs[:5],
            'research_coverage': {
                'organism_diversity': len(organisms),
                'impact_diversity': len(impacts),
                'condition_diversity': len(conditions)
            }
        }
        
        # Pathway analysis
        causal_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('edge_type') == 'causal']
        biological_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('edge_type') == 'biological_pathway']
        intervention_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('edge_type') == 'intervention']
        
        analysis['pathway_analysis'] = {
            'causal_relationships': len(causal_edges),
            'biological_pathways': len(biological_edges),
            'interventions': len(intervention_edges),
            'pathway_complexity': len(causal_edges) + len(biological_edges),
            'sample_causal_paths': causal_edges[:5],
            'sample_interventions': intervention_edges[:3]
        }
        
        # Performance assessment
        analysis['performance_assessment'] = {
            'connectivity_score': min(avg_degree / 2.0, 1.0),  # Target: avg degree ≥ 2
            'size_score': min((num_nodes * num_edges) / (50 * 40), 1.0),  # Target: 50+ nodes, 40+ edges
            'richness_score': len(node_types) / 8.0,  # Diversity of node types
            'hub_score': len(hub_nodes) / max(num_nodes * 0.1, 1),  # Hub density
            'overall_score': 0.0
        }
        
        # Calculate overall score
        perf = analysis['performance_assessment']
        perf['overall_score'] = (perf['connectivity_score'] * 0.3 + 
                               perf['size_score'] * 0.3 + 
                               perf['richness_score'] * 0.2 + 
                               perf['hub_score'] * 0.2)
        
        return analysis
        
    except Exception as e:
        log_error(f"Error in comprehensive graph analysis: {str(e)}")
        return {'error': str(e)}


def build_kg(df: pd.DataFrame) -> Optional[object]:
    """
    Build rich, hierarchical knowledge graph from DataFrame with comprehensive entity relationships.
    
    Creates an advanced knowledge graph with:
    - Typed node hierarchy: Study → Experiment → Impact/Organism/Condition
    - Weighted edges based on relationship confidence and frequency
    - Hub nodes for major concepts (microgravity, ISS, bone loss)
    - Temporal connections between related studies
    - Cross-experiment connectivity through shared entities
    - Advanced analytics (centrality, clustering, communities)
    
    Args:
        df: DataFrame with columns 'id', 'title', 'summary' (and optionally 'abstract', 'publication_date')
        
    Returns:
        NetworkX Graph object with rich connectivity and analytics, or None if unavailable
        
    Example:
        >>> df = pd.DataFrame({
        ...     'id': ['GLDS-001', 'GLDS-002'], 
        ...     'title': ['Bone Study', 'Muscle Research'],
        ...     'summary': ['Microgravity causes bone loss in mice', 'Exercise prevents muscle atrophy']
        ... })
        >>> graph = build_kg(df)
        >>> print(f"Rich graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    """
    if not NETWORKX_AVAILABLE:
        log_error("NetworkX not available. Install with: pip install networkx")
        return None
    
    if df is None or df.empty:
        log_error("Cannot build KG from empty DataFrame")
        return None
    
    try:
        # Create directed graph for rich relationships
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
        
        log_info(f"Building rich knowledge graph from {len(df)} experiments using '{text_col}' column")
        
        # Initialize tracking for entity frequencies and hub identification
        entity_frequencies = {}
        organism_experiments = {}
        impact_experiments = {}
        condition_experiments = {}
        all_entities = {}
        
        # First pass: Extract all entities and track frequencies
        for idx, row in df.iterrows():
            exp_id = row['id']
            text_content = row[text_col] if pd.notna(row[text_col]) else ""
            
            if text_content:
                entities = extract_entities(text_content)
                all_entities[exp_id] = entities
                
                # Track entity frequencies across experiments
                for category in ['impacts', 'organisms', 'experimental_conditions', 'measurements', 'space_terms']:
                    for entity in entities.get(category, []):
                        entity_key = f"{category}:{entity}"
                        entity_frequencies[entity_key] = entity_frequencies.get(entity_key, 0) + 1
                        
                        # Track which experiments study what
                        if category == 'organisms':
                            if entity not in organism_experiments:
                                organism_experiments[entity] = []
                            organism_experiments[entity].append(exp_id)
                        elif category == 'impacts':
                            if entity not in impact_experiments:
                                impact_experiments[entity] = []
                            impact_experiments[entity].append(exp_id)
                        elif category == 'experimental_conditions':
                            if entity not in condition_experiments:
                                condition_experiments[entity] = []
                            condition_experiments[entity].append(exp_id)
        
        # Identify hub entities (appearing in 3+ experiments or high frequency)
        hub_entities = {k: v for k, v in entity_frequencies.items() if v >= min(3, len(df) * 0.3)}
        log_info(f"Identified {len(hub_entities)} hub entities: {list(hub_entities.keys())[:5]}...")
        
        # Second pass: Create hierarchical nodes and rich connectivity
        for idx, row in df.iterrows():
            # Create experiment node with comprehensive metadata
            exp_id = row['id']
            exp_node = f"Experiment:{exp_id}"
            
            # Rich node attributes including publication metadata
            node_attrs = {
                'type': 'experiment',
                'node_level': 2,  # Hierarchy level
                'id': exp_id,
                'title': row['title'],
                'centrality_degree': 0,  # Will be calculated later
                'centrality_betweenness': 0.0,
                'hub_score': 0.0
            }
            
            # Add comprehensive metadata
            if text_col in df.columns and pd.notna(row[text_col]):
                node_attrs['summary'] = str(row[text_col])[:500]
            
            # Add publication date for temporal analysis
            if 'publication_date' in df.columns and pd.notna(row['publication_date']):
                node_attrs['publication_date'] = str(row['publication_date'])
            
            # Add other metadata fields
            for col in ['abstract', 'metadata', 'impacts', 'authors', 'doi']:
                if col in df.columns and pd.notna(row[col]):
                    node_attrs[col] = str(row[col])[:200]
            
            G.add_node(exp_node, **node_attrs)
            
            # Extract and process entities for this experiment
            text_content = row[text_col] if pd.notna(row[text_col]) else ""
            if text_content:
                entities = all_entities.get(exp_id, {})
                
                # Create hierarchical Impact nodes with rich attributes
                for impact in entities.get('impacts', []):
                    impact_node = f"Impact:{impact}"
                    is_hub = f"impacts:{impact}" in hub_entities
                    frequency = entity_frequencies.get(f"impacts:{impact}", 1)
                    
                    if not G.has_node(impact_node):
                        G.add_node(impact_node, 
                                  type='impact',
                                  node_level=3,  # Lower in hierarchy
                                  term=impact,
                                  description=f"Health/biological impact: {impact}",
                                  frequency=frequency,
                                  is_hub=is_hub,
                                  severity_score=min(frequency * 0.2, 1.0),
                                  experiments_studying=len(impact_experiments.get(impact, [exp_id])))
                    
                    # Weighted edge based on confidence and frequency
                    weight = 0.8 + (frequency * 0.1)
                    G.add_edge(exp_node, impact_node, 
                              relation='studies',
                              confidence=0.8,
                              weight=weight,
                              evidence_strength=frequency)
                
                # Create Organism nodes with taxonomy information
                for organism in entities.get('organisms', []):
                    org_node = f"Organism:{organism}"
                    is_hub = f"organisms:{organism}" in hub_entities
                    frequency = entity_frequencies.get(f"organisms:{organism}", 1)
                    
                    if not G.has_node(org_node):
                        G.add_node(org_node,
                                  type='organism',
                                  node_level=3,
                                  name=organism,
                                  frequency=frequency,
                                  is_hub=is_hub,
                                  taxonomy_info=_get_organism_taxonomy(organism),
                                  experiments_count=len(organism_experiments.get(organism, [exp_id])))
                    
                    # Create "studies_organism" relationship
                    weight = 0.9 + (frequency * 0.05)
                    G.add_edge(exp_node, org_node,
                              relation='studies_organism',
                              confidence=0.9,
                              weight=weight)
                
                # Create Experimental Condition nodes
                for condition in entities.get('experimental_conditions', []):
                    cond_node = f"Condition:{condition}"
                    frequency = entity_frequencies.get(f"experimental_conditions:{condition}", 1)
                    is_hub = f"experimental_conditions:{condition}" in hub_entities
                    
                    if not G.has_node(cond_node):
                        G.add_node(cond_node,
                                  type='experimental_condition',
                                  node_level=3,
                                  name=condition,
                                  frequency=frequency,
                                  is_hub=is_hub,
                                  experiments_count=len(condition_experiments.get(condition, [exp_id])))
                    
                    weight = 0.7 + (frequency * 0.1)
                    G.add_edge(exp_node, cond_node,
                              relation='uses_condition',
                              confidence=0.7,
                              weight=weight)
                
                # Add Measurement and Space Term nodes
                for measurement in entities.get('measurements', []):
                    meas_node = f"Measurement:{measurement}"
                    if not G.has_node(meas_node):
                        G.add_node(meas_node,
                                  type='measurement',
                                  node_level=4,
                                  name=measurement)
                    G.add_edge(exp_node, meas_node,
                              relation='uses_measurement',
                              confidence=0.6,
                              weight=0.6)
                
                for space_term in entities.get('space_terms', []):
                    space_node = f"SpaceTerm:{space_term}"
                    frequency = entity_frequencies.get(f"space_terms:{space_term}", 1)
                    is_hub = f"space_terms:{space_term}" in hub_entities
                    
                    if not G.has_node(space_node):
                        G.add_node(space_node,
                                  type='space_term',
                                  node_level=2,  # Important space concepts
                                  name=space_term,
                                  frequency=frequency,
                                  is_hub=is_hub)
                    
                    weight = 0.8 + (frequency * 0.1)
                    G.add_edge(exp_node, space_node,
                              relation='involves',
                              confidence=0.8,
                              weight=weight)
                
                # Add Organization and Location nodes
                for org in entities.get('organizations', []):
                    org_node = f"Organization:{org}"
                    if not G.has_node(org_node):
                        G.add_node(org_node,
                                  type='organization',
                                  node_level=1,  # Top level
                                  name=org)
                    G.add_edge(exp_node, org_node,
                              relation='conducted_by',
                              confidence=0.7,
                              weight=0.7)
                
                for location in entities.get('locations', []):
                    loc_node = f"Location:{location}"
                    if not G.has_node(loc_node):
                        G.add_node(loc_node,
                              type='location',
                              node_level=2,
                              name=location)
                    G.add_edge(exp_node, loc_node,
                              relation='conducted_in',
                              confidence=0.6,
                              weight=0.6)
                
                # Extract comprehensive relationships from all entity types
                causal_relations = entities.get('causal_relations', [])
                experimental_relations = entities.get('experimental_relations', [])
                temporal_relations = entities.get('temporal_relations', [])
                location_relations = entities.get('location_relations', [])
                
                # Process causal relationships
                for relation in causal_relations:
                    try:
                        if not isinstance(relation, dict):
                            continue
                        
                        subj = relation.get('subject', '').strip()
                        obj = relation.get('object', '').strip()
                        relation_type = relation.get('relation', 'causes')
                        confidence = relation.get('confidence', 0.8)
                        
                        if subj and obj:
                            subj_node = f"Entity:{subj}"
                            obj_node = f"Entity:{obj}"
                            
                            # Add nodes if not present
                            if not G.has_node(subj_node):
                                G.add_node(subj_node, type='entity', name=subj, node_level=4)
                            if not G.has_node(obj_node):
                                G.add_node(obj_node, type='entity', name=obj, node_level=4)
                            
                            # Add weighted causal edge
                            G.add_edge(subj_node, obj_node,
                                      relation=relation_type,
                                      confidence=confidence,
                                      weight=confidence,
                                      source_experiment=exp_id,
                                      edge_type='causal')
                    except Exception as e:
                        log_error(f"Error processing causal relation: {str(e)}")
                        continue
                
                # Process experimental relationships (comparisons)
                for relation in experimental_relations:
                    try:
                        if not isinstance(relation, dict):
                            continue
                        
                        subj = relation.get('subject', '').strip()
                        obj = relation.get('object', '').strip()
                        relation_type = relation.get('relation', 'compared_to')
                        confidence = relation.get('confidence', 0.7)
                        
                        if subj and obj:
                            subj_node = f"Entity:{subj}"
                            obj_node = f"Entity:{obj}"
                            
                            if not G.has_node(subj_node):
                                G.add_node(subj_node, type='entity', name=subj, node_level=4)
                            if not G.has_node(obj_node):
                                G.add_node(obj_node, type='entity', name=obj, node_level=4)
                            
                            G.add_edge(subj_node, obj_node,
                                      relation=relation_type,
                                      confidence=confidence,
                                      weight=confidence * 0.8,
                                      source_experiment=exp_id,
                                      edge_type='experimental')
                    except Exception as e:
                        log_error(f"Error processing experimental relation: {str(e)}")
                        continue
                
                # Process temporal relationships
                for relation in temporal_relations:
                    try:
                        if not isinstance(relation, dict):
                            continue
                        
                        subj = relation.get('subject', '').strip()
                        obj = relation.get('object', '').strip()
                        relation_type = relation.get('relation', 'temporal')
                        confidence = relation.get('confidence', 0.6)
                        
                        if subj and obj:
                            subj_node = f"Entity:{subj}"
                            obj_node = f"Entity:{obj}"
                            
                            if not G.has_node(subj_node):
                                G.add_node(subj_node, type='entity', name=subj, node_level=4)
                            if not G.has_node(obj_node):
                                G.add_node(obj_node, type='entity', name=obj, node_level=4)
                            
                            G.add_edge(subj_node, obj_node,
                                      relation=relation_type,
                                      confidence=confidence,
                                      weight=confidence * 0.6,
                                      source_experiment=exp_id,
                                      edge_type='temporal')
                    except Exception as e:
                        log_error(f"Error processing temporal relation: {str(e)}")
                        continue
        
        # PHASE 3: Create cross-experiment connectivity
        log_info("Creating cross-experiment connections...")
        
        # Connect experiments studying same organisms
        for organism, exp_list in organism_experiments.items():
            if len(exp_list) > 1:
                for i in range(len(exp_list)):
                    for j in range(i + 1, len(exp_list)):
                        exp1_node = f"Experiment:{exp_list[i]}"
                        exp2_node = f"Experiment:{exp_list[j]}"
                        if G.has_node(exp1_node) and G.has_node(exp2_node):
                            G.add_edge(exp1_node, exp2_node,
                                      relation='studies_same_organism',
                                      shared_organism=organism,
                                      confidence=0.8,
                                      weight=0.8,
                                      edge_type='cross_experiment')
        
        # Connect experiments with shared impacts
        for impact, exp_list in impact_experiments.items():
            if len(exp_list) > 1:
                for i in range(len(exp_list)):
                    for j in range(i + 1, len(exp_list)):
                        exp1_node = f"Experiment:{exp_list[i]}"
                        exp2_node = f"Experiment:{exp_list[j]}"
                        if G.has_node(exp1_node) and G.has_node(exp2_node):
                            G.add_edge(exp1_node, exp2_node,
                                      relation='studies_same_impact',
                                      shared_impact=impact,
                                      confidence=0.7,
                                      weight=0.7,
                                      edge_type='cross_experiment')
        
        # Connect experiments with shared conditions
        for condition, exp_list in condition_experiments.items():
            if len(exp_list) > 1:
                for i in range(len(exp_list)):
                    for j in range(i + 1, len(exp_list)):
                        exp1_node = f"Experiment:{exp_list[i]}"
                        exp2_node = f"Experiment:{exp_list[j]}"
                        if G.has_node(exp1_node) and G.has_node(exp2_node):
                            G.add_edge(exp1_node, exp2_node,
                                      relation='uses_same_condition',
                                      shared_condition=condition,
                                      confidence=0.6,
                                      weight=0.6,
                                      edge_type='cross_experiment')
        
        # Create "studied_in" relationships (Impact → Organism → Experiment)
        for impact_node in [n for n in G.nodes() if n.startswith('Impact:')]:
            impact_name = impact_node.split(':', 1)[1]
            # Find organisms that have this impact
            for organism_node in [n for n in G.nodes() if n.startswith('Organism:')]:
                # Check if any experiment studies both this impact and organism
                impact_experiments_set = set()
                organism_experiments_set = set()
                
                for pred in G.predecessors(impact_node):
                    if pred.startswith('Experiment:'):
                        impact_experiments_set.add(pred)
                
                for pred in G.predecessors(organism_node):
                    if pred.startswith('Experiment:'):
                        organism_experiments_set.add(pred)
                
                # If same experiments study both, create connection
                if impact_experiments_set & organism_experiments_set:
                    G.add_edge(impact_node, organism_node,
                              relation='affects_organism',
                              confidence=0.75,
                              weight=0.75,
                              edge_type='biological_pathway')
        
        # Add mitigation relationships
        for exp_id in all_entities:
            entities = all_entities[exp_id]
            exp_node = f"Experiment:{exp_id}"
            
            for mitigation in entities.get('mitigation_strategies', []):
                mitigation_node = f"Mitigation:{mitigation}"
                if not G.has_node(mitigation_node):
                    G.add_node(mitigation_node,
                              type='mitigation',
                              node_level=3,
                              name=mitigation,
                              strategy_type='countermeasure')
                
                G.add_edge(mitigation_node, exp_node,
                          relation='mitigates',
                          confidence=0.8,
                          weight=0.8,
                          edge_type='intervention')
                
                # Connect mitigations to impacts they address
                for impact in entities.get('impacts', []):
                    impact_node = f"Impact:{impact}"
                    if G.has_node(impact_node):
                        G.add_edge(mitigation_node, impact_node,
                                  relation='mitigated_by',
                                  confidence=0.7,
                                  weight=0.7,
                                  edge_type='therapeutic')
        
        # PHASE 4: Calculate graph analytics
        log_info("Calculating graph analytics...")
        
        # Calculate centrality measures
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Update node attributes with centrality scores
            for node in G.nodes():
                G.nodes[node]['centrality_degree'] = degree_centrality.get(node, 0.0)
                G.nodes[node]['centrality_betweenness'] = betweenness_centrality.get(node, 0.0)
                
                # Calculate hub score (combination of degree and frequency)
                frequency = G.nodes[node].get('frequency', 1)
                degree = G.degree(node)
                hub_score = (degree * 0.7) + (frequency * 0.3)
                G.nodes[node]['hub_score'] = hub_score
            
            # Identify communities/clusters
            if hasattr(nx, 'community') and hasattr(nx.community, 'greedy_modularity_communities'):
                try:
                    # Convert to undirected for community detection
                    G_undirected = G.to_undirected()
                    communities = nx.community.greedy_modularity_communities(G_undirected)
                    
                    # Add community information to nodes
                    for i, community in enumerate(communities):
                        for node in community:
                            if G.has_node(node):
                                G.nodes[node]['community'] = i
                    
                    log_info(f"Detected {len(communities)} research communities")
                except Exception as e:
                    log_error(f"Community detection failed: {str(e)}")
            
        except Exception as e:
            log_error(f"Error calculating centrality measures: {str(e)}")
        
        # Add temporal connections if publication dates available
        if 'publication_date' in df.columns:
            try:
                # Sort experiments by date
                exp_dates = {}
                for idx, row in df.iterrows():
                    if pd.notna(row.get('publication_date')):
                        exp_dates[row['id']] = row['publication_date']
                
                sorted_exps = sorted(exp_dates.items(), key=lambda x: x[1])
                
                # Connect chronologically adjacent experiments
                for i in range(len(sorted_exps) - 1):
                    current_exp = f"Experiment:{sorted_exps[i][0]}"
                    next_exp = f"Experiment:{sorted_exps[i + 1][0]}"
                    
                    if G.has_node(current_exp) and G.has_node(next_exp):
                        G.add_edge(current_exp, next_exp,
                                  relation='precedes_chronologically',
                                  confidence=0.5,
                                  weight=0.3,
                                  edge_type='temporal_sequence')
            except Exception as e:
                log_error(f"Error creating temporal connections: {str(e)}")
        
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
        
        # Calculate final graph metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G) if num_nodes > 0 else 0
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        
        log_info(f"✓ Built rich knowledge graph: {num_nodes} nodes, {num_edges} edges")
        log_info(f"  Graph density: {density:.3f}, Average degree: {avg_degree:.2f}")
        
        # Print comprehensive graph statistics
        node_types = {}
        edge_types = {}
        hub_nodes = []
        
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Identify hub nodes
            if attrs.get('is_hub', False) or attrs.get('hub_score', 0) > 3:
                hub_nodes.append((node, attrs.get('hub_score', 0)))
        
        for u, v, attrs in G.edges(data=True):
            edge_type = attrs.get('edge_type', attrs.get('relation', 'unknown'))
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # Sort hub nodes by score
        hub_nodes.sort(key=lambda x: x[1], reverse=True)
        top_hubs = [node for node, score in hub_nodes[:5]]
        
        log_info(f"Node distribution: {node_types}")
        log_info(f"Edge distribution: {edge_types}")
        log_info(f"Top hub nodes: {top_hubs}")
        log_info(f"Hub entities identified: {len(hub_entities)}")
        
        # Validate connectivity requirements
        if avg_degree >= 2.0:
            log_info("✅ Graph meets connectivity requirement (avg degree ≥ 2)")
        else:
            log_info(f"⚠️ Graph connectivity below target (avg degree: {avg_degree:.2f})")
        
        if num_nodes >= 15 and num_edges >= 10:
            log_info("✅ Graph meets size requirements")
        else:
            log_info(f"⚠️ Graph size below target (nodes: {num_nodes}, edges: {num_edges})")
        
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