"""
Advanced entity extraction for MOSDAC knowledge graph construction.
"""
import re
import spacy
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import json
from datetime import datetime
import hashlib

from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

@dataclass
class Entity:
    """Represents an extracted entity with metadata."""
    id: str
    text: str
    label: str
    confidence: float
    start_pos: int
    end_pos: int
    source_document: str
    attributes: Dict[str, Any]
    synonyms: List[str]
    canonical_form: str

class EntityExtractor:
    """
    Advanced entity extractor for MOSDAC content.
    Combines NLP models with domain-specific patterns and rules.
    """
    
    def __init__(self):
        self.nlp = None
        self._load_spacy_model()
        
        # MOSDAC-specific entity types and patterns
        self.entity_types = config.get('knowledge_graph.entity_types', [
            'Satellite', 'Mission', 'Product', 'Service', 'Location', 
            'Date', 'Organization', 'Technology'
        ])
        
        # Define comprehensive patterns for each entity type
        self.entity_patterns = {
            'Satellite': {
                'patterns': [
                    r'\b(INSAT-?\w*)\b',
                    r'\b(IRS-?\w*)\b',
                    r'\b(Oceansat-?\w*)\b', 
                    r'\b(ResourceSat-?\w*)\b',
                    r'\b(CartoSat-?\w*)\b',
                    r'\b(RISAT-?\w*)\b',
                    r'\b(SARAL)\b',
                    r'\b(Astrosat)\b',
                    r'\b(Chandrayaan-?\w*)\b',
                    r'\b(Mangalyaan)\b',
                    r'\b(Landsat-?\w*)\b',
                    r'\b(MODIS)\b',
                    r'\b(Sentinel-?\w*)\b',
                    r'\b(SPOT-?\w*)\b'
                ],
                'keywords': ['satellite', 'spacecraft', 'platform']
            },
            'Mission': {
                'patterns': [
                    r'\b(Mission-?\w*)\b',
                    r'\b(\w+\s+Mission)\b',
                    r'\b(Operation\s+\w+)\b'
                ],
                'keywords': ['mission', 'operation', 'program', 'project']
            },
            'Product': {
                'patterns': [
                    r'\b(L1[A-Z])\b',
                    r'\b(L2[A-Z])\b',
                    r'\b(L3[A-Z])\b',
                    r'\b(L4[A-Z])\b',
                    r'\b(\w+\s+Product)\b',
                    r'\b(Standard\s+Data\s+Product)\b',
                    r'\b(SDP)\b'
                ],
                'keywords': ['product', 'dataset', 'data', 'imagery', 'level']
            },
            'Service': {
                'patterns': [
                    r'\b(\w+\s+Service)\b',
                    r'\b(Web\s+Service)\b',
                    r'\b(API)\b',
                    r'\b(Portal)\b'
                ],
                'keywords': ['service', 'api', 'portal', 'interface', 'access']
            },
            'Location': {
                'patterns': [
                    # Indian states and cities
                    r'\b(Andhra Pradesh|Arunachal Pradesh|Assam|Bihar|Chhattisgarh|Goa|Gujarat|Haryana|Himachal Pradesh|Jammu and Kashmir|Jharkhand|Karnataka|Kerala|Madhya Pradesh|Maharashtra|Manipur|Meghalaya|Mizoram|Nagaland|Odisha|Punjab|Rajasthan|Sikkim|Tamil Nadu|Telangana|Tripura|Uttar Pradesh|Uttarakhand|West Bengal)\b',
                    r'\b(Delhi|Mumbai|Kolkata|Chennai|Bangalore|Hyderabad|Ahmedabad|Pune|Surat|Jaipur|Lucknow|Kanpur|Nagpur|Indore|Bhopal|Visakhapatnam)\b',
                    # Geographic features
                    r'\b(Arabian Sea|Bay of Bengal|Indian Ocean|Himalaya|Western Ghats|Eastern Ghats|Deccan Plateau|Indo-Gangetic Plain)\b',
                    # Coordinates
                    r'\b\d+(?:\.\d+)?°?[NS]\s*,?\s*\d+(?:\.\d+)?°?[EW]\b'
                ],
                'keywords': ['location', 'region', 'area', 'latitude', 'longitude', 'coordinate']
            },
            'Organization': {
                'patterns': [
                    r'\b(ISRO|Indian Space Research Organisation)\b',
                    r'\b(NASA|National Aeronautics and Space Administration)\b',
                    r'\b(ESA|European Space Agency)\b',
                    r'\b(NOAA)\b',
                    r'\b(NRSC|National Remote Sensing Centre)\b',
                    r'\b(SAC|Space Applications Centre)\b',
                    r'\b(MOSDAC|Meteorological and Oceanographic Satellite Data Archival Centre)\b'
                ],
                'keywords': ['organization', 'agency', 'centre', 'institute']
            },
            'Technology': {
                'patterns': [
                    r'\b(LISS|Linear Imaging Self Scanner)\b',
                    r'\b(WiFS|Wide Field Sensor)\b',
                    r'\b(AWiFS|Advanced Wide Field Sensor)\b',
                    r'\b(PAN|Panchromatic)\b',
                    r'\b(SWIR|Short Wave Infrared)\b',
                    r'\b(TIR|Thermal Infrared)\b',
                    r'\b(SAR|Synthetic Aperture Radar)\b',
                    r'\b(INSAT CCD|Charge Coupled Device)\b'
                ],
                'keywords': ['sensor', 'instrument', 'detector', 'camera', 'radar']
            },
            'Date': {
                'patterns': [
                    r'\b\d{4}-\d{2}-\d{2}\b',
                    r'\b\d{2}/\d{2}/\d{4}\b',
                    r'\b\d{2}\.\d{2}\.\d{4}\b',
                    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                    r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
                ],
                'keywords': ['date', 'time', 'year', 'month', 'day']
            }
        }
        
        # Load domain-specific gazetteers/dictionaries
        self.entity_dictionaries = self._load_entity_dictionaries()
        
        # Synonym mappings for normalization
        self.synonym_mappings = {
            'ISRO': ['Indian Space Research Organisation', 'Indian Space Research Organization'],
            'MOSDAC': ['Meteorological and Oceanographic Satellite Data Archival Centre'],
            'SAR': ['Synthetic Aperture Radar'],
            'IRS': ['Indian Remote Sensing'],
            'INSAT': ['Indian National Satellite']
        }
    
    def _load_spacy_model(self):
        """Load spaCy model for NER."""
        try:
            model_name = config.get('models.ner_model', 'en_core_web_sm')
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            self.nlp = None
    
    def _load_entity_dictionaries(self) -> Dict[str, Set[str]]:
        """Load pre-defined dictionaries of known entities."""
        dictionaries = {}
        
        # Satellite names
        dictionaries['Satellite'] = {
            'INSAT-1A', 'INSAT-1B', 'INSAT-1C', 'INSAT-1D',
            'INSAT-2A', 'INSAT-2B', 'INSAT-2C', 'INSAT-2D', 'INSAT-2E',
            'INSAT-3A', 'INSAT-3B', 'INSAT-3C', 'INSAT-3D', 'INSAT-3DR',
            'INSAT-3E', 'INSAT-3F', 'INSAT-3G',
            'IRS-1A', 'IRS-1B', 'IRS-1C', 'IRS-1D', 'IRS-P2', 'IRS-P3', 'IRS-P4',
            'IRS-P5', 'IRS-P6', 'ResourceSat-1', 'ResourceSat-2', 'ResourceSat-2A',
            'CartoSat-1', 'CartoSat-2', 'CartoSat-2A', 'CartoSat-2B', 'CartoSat-3',
            'Oceansat-1', 'Oceansat-2', 'Oceansat-3',
            'RISAT-1', 'RISAT-2', 'RISAT-2B',
            'SARAL', 'Astrosat', 'Chandrayaan-1', 'Chandrayaan-2', 'Mangalyaan'
        }
        
        # Organizations
        dictionaries['Organization'] = {
            'ISRO', 'Indian Space Research Organisation', 'NASA', 'ESA', 'NOAA',
            'NRSC', 'National Remote Sensing Centre', 'SAC', 'Space Applications Centre',
            'MOSDAC', 'Meteorological and Oceanographic Satellite Data Archival Centre',
            'ANTRIX', 'DOS', 'Department of Space'
        }
        
        # Technologies/Sensors
        dictionaries['Technology'] = {
            'LISS', 'Linear Imaging Self Scanner', 'LISS-I', 'LISS-II', 'LISS-III', 'LISS-IV',
            'WiFS', 'Wide Field Sensor', 'AWiFS', 'Advanced Wide Field Sensor',
            'PAN', 'Panchromatic', 'SWIR', 'Short Wave Infrared',
            'TIR', 'Thermal Infrared', 'MIR', 'Middle Infrared',
            'SAR', 'Synthetic Aperture Radar', 'CCD', 'Charge Coupled Device',
            'MODIS', 'AVHRR', 'VIIRS', 'OLI', 'TIRS', 'ETM+'
        }
        
        return dictionaries
    
    def _generate_entity_id(self, text: str, label: str, document_id: str) -> str:
        """Generate unique entity ID."""
        content = f"{text}_{label}_{document_id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _normalize_entity_text(self, text: str, label: str) -> Tuple[str, List[str]]:
        """Normalize entity text and find synonyms."""
        normalized_text = text.strip()
        synonyms = []
        
        # Handle known synonyms
        for canonical, syns in self.synonym_mappings.items():
            if text.upper() in [s.upper() for s in syns] or text.upper() == canonical.upper():
                normalized_text = canonical
                synonyms = syns
                break
        
        # Additional normalization rules
        if label == 'Satellite':
            # Standardize satellite naming
            normalized_text = re.sub(r'\s+', '-', normalized_text)
            normalized_text = normalized_text.upper()
        
        elif label == 'Date':
            # Standardize date format
            try:
                # Try different date formats and convert to ISO
                from dateutil import parser
                parsed_date = parser.parse(text)
                normalized_text = parsed_date.strftime('%Y-%m-%d')
            except:
                pass
        
        elif label == 'Location':
            # Standardize location names
            normalized_text = normalized_text.title()
        
        return normalized_text, synonyms
    
    def _extract_spacy_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities using spaCy NER."""
        entities = []
        
        if not self.nlp:
            return entities
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            # Map spaCy labels to our taxonomy
            label_mapping = {
                'GPE': 'Location',  # Geopolitical entity
                'LOC': 'Location',  # Location
                'ORG': 'Organization',  # Organization
                'DATE': 'Date',  # Date
                'PERSON': 'Organization',  # Map persons to organizations for MOSDAC context
                'PRODUCT': 'Product',  # Product
                'FACILITY': 'Service'  # Facility as service
            }
            
            mapped_label = label_mapping.get(ent.label_, ent.label_)
            
            # Only include entities that match our taxonomy
            if mapped_label in self.entity_types:
                normalized_text, synonyms = self._normalize_entity_text(ent.text, mapped_label)
                
                entity = Entity(
                    id=self._generate_entity_id(ent.text, mapped_label, document_id),
                    text=ent.text,
                    label=mapped_label,
                    confidence=0.8,  # spaCy confidence
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    source_document=document_id,
                    attributes={
                        'spacy_label': ent.label_,
                        'lemma': ent.lemma_ if hasattr(ent, 'lemma_') else ent.text
                    },
                    synonyms=synonyms,
                    canonical_form=normalized_text
                )
                entities.append(entity)
        
        return entities
    
    def _extract_pattern_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities using domain-specific patterns."""
        entities = []
        
        for entity_type, config_data in self.entity_patterns.items():
            patterns = config_data.get('patterns', [])
            keywords = config_data.get('keywords', [])
            
            # Pattern-based extraction
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group()
                    normalized_text, synonyms = self._normalize_entity_text(matched_text, entity_type)
                    
                    entity = Entity(
                        id=self._generate_entity_id(matched_text, entity_type, document_id),
                        text=matched_text,
                        label=entity_type,
                        confidence=0.9,  # High confidence for pattern matches
                        start_pos=match.start(),
                        end_pos=match.end(),
                        source_document=document_id,
                        attributes={
                            'extraction_method': 'pattern',
                            'pattern': pattern
                        },
                        synonyms=synonyms,
                        canonical_form=normalized_text
                    )
                    entities.append(entity)
            
            # Keyword-based extraction
            for keyword in keywords:
                # Look for phrases containing the keyword
                keyword_pattern = rf'\b\w*{re.escape(keyword)}\w*\b'
                matches = re.finditer(keyword_pattern, text, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group()
                    
                    # Skip if already extracted by patterns
                    if any(e.text.lower() == matched_text.lower() for e in entities):
                        continue
                    
                    normalized_text, synonyms = self._normalize_entity_text(matched_text, entity_type)
                    
                    entity = Entity(
                        id=self._generate_entity_id(matched_text, entity_type, document_id),
                        text=matched_text,
                        label=entity_type,
                        confidence=0.6,  # Lower confidence for keyword matches
                        start_pos=match.start(),
                        end_pos=match.end(),
                        source_document=document_id,
                        attributes={
                            'extraction_method': 'keyword',
                            'keyword': keyword
                        },
                        synonyms=synonyms,
                        canonical_form=normalized_text
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_dictionary_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities using pre-defined dictionaries."""
        entities = []
        
        for entity_type, dictionary in self.entity_dictionaries.items():
            for known_entity in dictionary:
                # Case-insensitive search
                pattern = rf'\b{re.escape(known_entity)}\b'
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    matched_text = match.group()
                    normalized_text, synonyms = self._normalize_entity_text(matched_text, entity_type)
                    
                    entity = Entity(
                        id=self._generate_entity_id(matched_text, entity_type, document_id),
                        text=matched_text,
                        label=entity_type,
                        confidence=0.95,  # Very high confidence for dictionary matches
                        start_pos=match.start(),
                        end_pos=match.end(),
                        source_document=document_id,
                        attributes={
                            'extraction_method': 'dictionary',
                            'dictionary_entry': known_entity
                        },
                        synonyms=synonyms,
                        canonical_form=normalized_text
                    )
                    entities.append(entity)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on text overlap and similarity."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_pos)
        
        deduplicated = []
        
        for entity in entities:
            is_duplicate = False
            
            for existing in deduplicated:
                # Check for exact text match
                if entity.canonical_form.lower() == existing.canonical_form.lower():
                    # Keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
                
                # Check for significant overlap
                overlap_start = max(entity.start_pos, existing.start_pos)
                overlap_end = min(entity.end_pos, existing.end_pos)
                overlap_length = max(0, overlap_end - overlap_start)
                
                entity_length = entity.end_pos - entity.start_pos
                existing_length = existing.end_pos - existing.start_pos
                
                # If more than 70% overlap, consider as duplicate
                if overlap_length > 0.7 * min(entity_length, existing_length):
                    # Keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(entity)
        
        return deduplicated
    
    def extract_entities(self, text: str, document_id: str) -> List[Entity]:
        """
        Extract all entities from text using multiple methods.
        
        Args:
            text: Input text to process
            document_id: Unique identifier for source document
            
        Returns:
            List of extracted entities
        """
        all_entities = []
        
        # Extract using different methods
        spacy_entities = self._extract_spacy_entities(text, document_id)
        pattern_entities = self._extract_pattern_entities(text, document_id)
        dictionary_entities = self._extract_dictionary_entities(text, document_id)
        
        # Combine all entities
        all_entities.extend(spacy_entities)
        all_entities.extend(pattern_entities)
        all_entities.extend(dictionary_entities)
        
        # Deduplicate and return
        final_entities = self._deduplicate_entities(all_entities)
        
        logger.info(f"Extracted {len(final_entities)} entities from document {document_id}")
        return final_entities
    
    def batch_extract_entities(self, documents: List[Dict]) -> Dict[str, List[Entity]]:
        """
        Extract entities from multiple documents.
        
        Args:
            documents: List of document dictionaries with 'content' and 'document_id'
            
        Returns:
            Dictionary mapping document IDs to their extracted entities
        """
        results = {}
        
        for doc in documents:
            document_id = doc.get('document_id', 'unknown')
            content = doc.get('content', '')
            
            if content:
                entities = self.extract_entities(content, document_id)
                results[document_id] = entities
            else:
                results[document_id] = []
        
        total_entities = sum(len(entities) for entities in results.values())
        logger.info(f"Extracted {total_entities} entities from {len(documents)} documents")
        
        return results
    
    def get_entity_statistics(self, entities: List[Entity]) -> Dict[str, Any]:
        """Generate statistics about extracted entities."""
        if not entities:
            return {}
        
        stats = {
            'total_entities': len(entities),
            'entity_types': {},
            'confidence_distribution': {
                'high_confidence': 0,  # > 0.8
                'medium_confidence': 0,  # 0.5 - 0.8
                'low_confidence': 0  # < 0.5
            },
            'extraction_methods': {},
            'avg_confidence': sum(e.confidence for e in entities) / len(entities)
        }
        
        for entity in entities:
            # Count by type
            entity_type = entity.label
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
            
            # Count by confidence
            if entity.confidence > 0.8:
                stats['confidence_distribution']['high_confidence'] += 1
            elif entity.confidence > 0.5:
                stats['confidence_distribution']['medium_confidence'] += 1
            else:
                stats['confidence_distribution']['low_confidence'] += 1
            
            # Count by extraction method
            method = entity.attributes.get('extraction_method', 'spacy')
            stats['extraction_methods'][method] = stats['extraction_methods'].get(method, 0) + 1
        
        return stats
    
    def save_entities(self, entities: List[Entity], output_file: str):
        """Save extracted entities to JSON file."""
        entity_dicts = []
        for entity in entities:
            entity_dict = {
                'id': entity.id,
                'text': entity.text,
                'label': entity.label,
                'confidence': entity.confidence,
                'start_pos': entity.start_pos,
                'end_pos': entity.end_pos,
                'source_document': entity.source_document,
                'attributes': entity.attributes,
                'synonyms': entity.synonyms,
                'canonical_form': entity.canonical_form
            }
            entity_dicts.append(entity_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entity_dicts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(entities)} entities to {output_file}")