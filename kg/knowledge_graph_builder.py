#!/usr/bin/env python3
"""Knowledge Graph Builder for MOSDAC Data.

This module processes crawler output to extract entities and relationships,
then builds a comprehensive knowledge graph using Neo4j.
"""

import json
import logging
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

import spacy
from spacy.matcher import Matcher
from neo4j import GraphDatabase
import pandas as pd


@dataclass
class Entity:
    """Represents an extracted entity with metadata."""
    id: str
    name: str
    type: str
    aliases: List[str]
    properties: Dict[str, Any]
    confidence: float
    source_urls: List[str]
    mentions: int = 0


@dataclass
class Relationship:
    """Represents a relationship between two entities."""
    id: str
    source_entity: str
    target_entity: str
    relationship_type: str
    properties: Dict[str, Any]
    confidence: float
    evidence: List[str]
    source_urls: List[str]


class MOSDACKnowledgeGraphBuilder:
    """Builds knowledge graph from MOSDAC crawler data."""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        """Initialize the knowledge graph builder.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize spaCy NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.error("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            raise
        
        # Initialize Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Entity and relationship storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'entities_extracted': 0,
            'relationships_created': 0,
            'errors': 0
        }
        
        # Initialize patterns and matchers
        self._setup_patterns()
        self._setup_entity_matchers()
    
    def _setup_patterns(self):
        """Setup domain-specific patterns for MOSDAC entities."""
        
        # Satellite missions
        self.mission_patterns = {
            'INSAT-3D': ['insat-3d', 'insat3d', 'INSAT 3D', 'insat 3d'],
            'OCEANSAT-2': ['oceansat-2', 'oceansat2', 'OCEANSAT 2', 'oceansat 2', 'OCM'],
            'SCATSAT-1': ['scatsat-1', 'scatsat1', 'SCATSAT 1', 'scatsat 1', 'OSCAT'],
            'MEGHA-TROPIQUES': ['megha-tropiques', 'meghatropiques', 'MT1', 'megha tropiques'],
            'CARTOSAT-2': ['cartosat-2', 'cartosat2', 'CARTOSAT 2', 'cartosat 2'],
            'RESOURCESAT-2': ['resourcesat-2', 'resourcesat2', 'RESOURCESAT 2', 'resourcesat 2'],
            'RISAT-1': ['risat-1', 'risat1', 'RISAT 1', 'risat 1'],
            'ASTROSAT': ['astrosat', 'ASTROSAT', 'astro sat']
        }
        
        # Instruments
        self.instrument_patterns = {
            'Ocean Color Monitor': ['OCM', 'ocean color monitor', 'ocean colour monitor'],
            'Thermal Infrared Imaging Sensor': ['TIIS', 'thermal infrared imaging sensor'],
            'Scatterometer': ['OSCAT', 'scatterometer', 'scat'],
            'Imager': ['imager', 'imaging sensor'],
            'Sounder': ['sounder', 'atmospheric sounder'],
            'MADRAS': ['MADRAS', 'microwave analysis and detection of rain'],
            'SAPHIR': ['SAPHIR', 'sondeur atmospherique'],
            'ScaRaB': ['ScaRaB', 'scanner for radiation budget']
        }
        
        # Data products
        self.product_patterns = {
            'Ocean Color': ['ocean color', 'ocean colour', 'chlorophyll', 'chl-a'],
            'Sea Surface Temperature': ['SST', 'sea surface temperature', 'temperature'],
            'Wind Speed': ['wind speed', 'wind velocity', 'surface wind'],
            'Wind Direction': ['wind direction', 'wind vector'],
            'Precipitation': ['precipitation', 'rainfall', 'rain rate'],
            'Water Vapor': ['water vapor', 'water vapour', 'humidity'],
            'Cloud': ['cloud', 'cloud cover', 'cloud mask'],
            'Vegetation Index': ['NDVI', 'vegetation index', 'vegetation'],
            'Aerosol Optical Depth': ['AOD', 'aerosol optical depth', 'aerosol'],
            'Total Ozone': ['ozone', 'total ozone', 'O3']
        }
        
        # Locations and regions
        self.location_patterns = {
            'Indian Ocean': ['indian ocean', 'IO'],
            'Arabian Sea': ['arabian sea', 'AS'],
            'Bay of Bengal': ['bay of bengal', 'BoB'],
            'Indian Subcontinent': ['indian subcontinent', 'india'],
            'Global': ['global', 'worldwide', 'earth'],
            'Tropical': ['tropical', 'tropics'],
            'Polar': ['polar', 'arctic', 'antarctic']
        }
        
        # Technical specifications
        self.resolution_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre)(?:\s+resolution)?',
            r'(\d+(?:\.\d+)?)\s*(?:km|kilometer|kilometre)(?:\s+resolution)?',
            r'resolution[:\s]+(\d+(?:\.\d+)?)\s*(?:m|km|meter|kilometre)',
            r'spatial resolution[:\s]+(\d+(?:\.\d+)?)\s*(?:m|km)'
        ]
        
        self.frequency_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:GHz|MHz|Hz)',
            r'frequency[:\s]+(\d+(?:\.\d+)?)\s*(?:GHz|MHz|Hz)',
            r'(\d+(?:\.\d+)?)\s*(?:gigahertz|megahertz|hertz)'
        ]
    
    def _setup_entity_matchers(self):
        """Setup spaCy matchers for entity extraction."""
        self.matcher = Matcher(self.nlp.vocab)
        
        # Mission patterns
        for mission, patterns in self.mission_patterns.items():
            for pattern in patterns:
                pattern_tokens = [{"LOWER": token.lower()} for token in pattern.split()]
                self.matcher.add(f"MISSION_{mission}", [pattern_tokens])
        
        # Instrument patterns
        for instrument, patterns in self.instrument_patterns.items():
            for pattern in patterns:
                pattern_tokens = [{"LOWER": token.lower()} for token in pattern.split()]
                self.matcher.add(f"INSTRUMENT_{instrument}", [pattern_tokens])
        
        # Product patterns
        for product, patterns in self.product_patterns.items():
            for pattern in patterns:
                pattern_tokens = [{"LOWER": token.lower()} for token in pattern.split()]
                self.matcher.add(f"PRODUCT_{product}", [pattern_tokens])
        
        # Location patterns
        for location, patterns in self.location_patterns.items():
            for pattern in patterns:
                pattern_tokens = [{"LOWER": token.lower()} for token in pattern.split()]
                self.matcher.add(f"LOCATION_{location}", [pattern_tokens])
    
    def process_crawler_output(self, input_file: str) -> Dict[str, Any]:
        """Process crawler output and build knowledge graph.
        
        Args:
            input_file: Path to crawler output JSON file
            
        Returns:
            Dictionary with processing results and statistics
        """
        self.logger.info(f"Processing crawler output: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                crawler_data = json.load(f)
            
            # Handle both single object and array formats
            if isinstance(crawler_data, dict):
                documents = [crawler_data]
            else:
                documents = crawler_data
            
            # Process each document
            for doc in documents:
                try:
                    self._process_document(doc)
                    self.stats['documents_processed'] += 1
                except Exception as e:
                    self.logger.error(f"Error processing document {doc.get('url', 'unknown')}: {e}")
                    self.stats['errors'] += 1
            
            # Extract relationships
            self._extract_relationships()
            
            # Build graph in Neo4j
            self._build_neo4j_graph()
            
            # Generate statistics
            results = self._generate_results()
            
            self.logger.info(f"Processing completed. Entities: {len(self.entities)}, "
                           f"Relationships: {len(self.relationships)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing crawler output: {e}")
            raise
    
    def _process_document(self, doc: Dict[str, Any]):
        """Process a single document from crawler output.
        
        Args:
            doc: Document dictionary from crawler
        """
        url = doc.get('url', '')
        content = doc.get('content', '')
        
        if not content:
            return
        
        # Extract entities from content
        self._extract_entities_from_text(content, url)
        
        # Extract entities from structured data
        if 'mission_info' in doc:
            self._extract_mission_entities(doc['mission_info'], url)
        
        if 'product_info' in doc:
            self._extract_product_entities(doc['product_info'], url)
        
        if 'technical_specs' in doc:
            self._extract_technical_entities(doc['technical_specs'], url)
        
        if 'coverage_info' in doc:
            self._extract_coverage_entities(doc['coverage_info'], url)
    
    def _extract_entities_from_text(self, text: str, url: str):
        """Extract entities from text using NLP.
        
        Args:
            text: Text content to process
            url: Source URL for provenance
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'LOC']:
                entity_id = self._create_entity_id(ent.text, ent.label_)
                
                if entity_id not in self.entities:
                    self.entities[entity_id] = Entity(
                        id=entity_id,
                        name=ent.text,
                        type=ent.label_,
                        aliases=[],
                        properties={'spacy_label': ent.label_},
                        confidence=0.7,
                        source_urls=[url]
                    )
                else:
                    self.entities[entity_id].mentions += 1
                    if url not in self.entities[entity_id].source_urls:
                        self.entities[entity_id].source_urls.append(url)
        
        # Extract pattern-based entities
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id]
            
            # Parse label to get type and name
            if '_' in label:
                entity_type, entity_name = label.split('_', 1)
                entity_id = self._create_entity_id(entity_name, entity_type)
                
                if entity_id not in self.entities:
                    self.entities[entity_id] = Entity(
                        id=entity_id,
                        name=entity_name,
                        type=entity_type,
                        aliases=[span.text],
                        properties={'matched_text': span.text},
                        confidence=0.9,
                        source_urls=[url]
                    )
                else:
                    self.entities[entity_id].mentions += 1
                    if span.text not in self.entities[entity_id].aliases:
                        self.entities[entity_id].aliases.append(span.text)
                    if url not in self.entities[entity_id].source_urls:
                        self.entities[entity_id].source_urls.append(url)
        
        # Extract technical specifications
        self._extract_resolution_entities(text, url)
        self._extract_frequency_entities(text, url)
    
    def _extract_mission_entities(self, mission_info: Dict[str, Any], url: str):
        """Extract mission-related entities.
        
        Args:
            mission_info: Mission information dictionary
            url: Source URL
        """
        if 'mission' in mission_info:
            mission_name = mission_info['mission']
            entity_id = self._create_entity_id(mission_name, 'MISSION')
            
            properties = {}
            if 'launch_info' in mission_info:
                properties['launch_info'] = mission_info['launch_info']
            if 'orbit_info' in mission_info:
                properties['orbit_info'] = mission_info['orbit_info']
            if 'objectives' in mission_info:
                properties['objectives'] = mission_info['objectives']
            
            if entity_id not in self.entities:
                self.entities[entity_id] = Entity(
                    id=entity_id,
                    name=mission_name,
                    type='MISSION',
                    aliases=[],
                    properties=properties,
                    confidence=1.0,
                    source_urls=[url]
                )
            else:
                self.entities[entity_id].properties.update(properties)
                if url not in self.entities[entity_id].source_urls:
                    self.entities[entity_id].source_urls.append(url)
    
    def _extract_product_entities(self, product_info: Dict[str, Any], url: str):
        """Extract product-related entities.
        
        Args:
            product_info: Product information dictionary
            url: Source URL
        """
        # Extract products from products list
        if 'products' in product_info:
            for product in product_info['products']:
                entity_id = self._create_entity_id(product, 'PRODUCT')
                
                if entity_id not in self.entities:
                    self.entities[entity_id] = Entity(
                        id=entity_id,
                        name=product,
                        type='PRODUCT',
                        aliases=[],
                        properties={},
                        confidence=0.9,
                        source_urls=[url]
                    )
        
        # Extract products from products table
        if 'products_table' in product_info:
            for product_entry in product_info['products_table']:
                if isinstance(product_entry, dict) and 'name' in product_entry:
                    product_name = product_entry['name']
                    entity_id = self._create_entity_id(product_name, 'PRODUCT')
                    
                    properties = {}
                    if 'description' in product_entry:
                        properties['description'] = product_entry['description']
                    
                    if entity_id not in self.entities:
                        self.entities[entity_id] = Entity(
                            id=entity_id,
                            name=product_name,
                            type='PRODUCT',
                            aliases=[],
                            properties=properties,
                            confidence=0.95,
                            source_urls=[url]
                        )
                    else:
                        self.entities[entity_id].properties.update(properties)
        
        # Extract data formats
        if 'data_formats' in product_info:
            for format_name in product_info['data_formats']:
                entity_id = self._create_entity_id(format_name, 'FORMAT')
                
                if entity_id not in self.entities:
                    self.entities[entity_id] = Entity(
                        id=entity_id,
                        name=format_name,
                        type='FORMAT',
                        aliases=[],
                        properties={'category': 'data_format'},
                        confidence=0.8,
                        source_urls=[url]
                    )
    
    def _extract_technical_entities(self, tech_specs: Dict[str, Any], url: str):
        """Extract technical specification entities.
        
        Args:
            tech_specs: Technical specifications dictionary
            url: Source URL
        """
        if 'resolution' in tech_specs:
            resolution = tech_specs['resolution']
            entity_id = self._create_entity_id(resolution, 'RESOLUTION')
            
            if entity_id not in self.entities:
                self.entities[entity_id] = Entity(
                    id=entity_id,
                    name=resolution,
                    type='RESOLUTION',
                    aliases=[],
                    properties={'category': 'spatial_resolution'},
                    confidence=0.9,
                    source_urls=[url]
                )
        
        if 'spectral_bands' in tech_specs:
            bands = tech_specs['spectral_bands']
            entity_id = self._create_entity_id(f"{bands}_bands", 'SPECIFICATION')
            
            if entity_id not in self.entities:
                self.entities[entity_id] = Entity(
                    id=entity_id,
                    name=f"{bands} spectral bands",
                    type='SPECIFICATION',
                    aliases=[],
                    properties={'category': 'spectral_bands', 'count': bands},
                    confidence=0.9,
                    source_urls=[url]
                )
        
        if 'swath_width' in tech_specs:
            swath = tech_specs['swath_width']
            entity_id = self._create_entity_id(swath, 'SPECIFICATION')
            
            if entity_id not in self.entities:
                self.entities[entity_id] = Entity(
                    id=entity_id,
                    name=f"Swath width: {swath}",
                    type='SPECIFICATION',
                    aliases=[],
                    properties={'category': 'swath_width', 'value': swath},
                    confidence=0.9,
                    source_urls=[url]
                )
    
    def _extract_coverage_entities(self, coverage_info: Dict[str, Any], url: str):
        """Extract coverage and location entities.
        
        Args:
            coverage_info: Coverage information dictionary
            url: Source URL
        """
        if 'regions' in coverage_info:
            for region in coverage_info['regions']:
                entity_id = self._create_entity_id(region, 'LOCATION')
                
                if entity_id not in self.entities:
                    self.entities[entity_id] = Entity(
                        id=entity_id,
                        name=region,
                        type='LOCATION',
                        aliases=[],
                        properties={'category': 'geographic_region'},
                        confidence=0.9,
                        source_urls=[url]
                    )
        
        if 'coordinate_ranges' in coverage_info:
            for coord_range in coverage_info['coordinate_ranges']:
                if isinstance(coord_range, list) and len(coord_range) == 2:
                    range_name = f"{coord_range[0]} to {coord_range[1]}"
                    entity_id = self._create_entity_id(range_name, 'COORDINATE_RANGE')
                    
                    if entity_id not in self.entities:
                        self.entities[entity_id] = Entity(
                            id=entity_id,
                            name=range_name,
                            type='COORDINATE_RANGE',
                            aliases=[],
                            properties={
                                'start': coord_range[0],
                                'end': coord_range[1],
                                'category': 'coordinate_range'
                            },
                            confidence=0.8,
                            source_urls=[url]
                        )
    
    def _extract_resolution_entities(self, text: str, url: str):
        """Extract resolution specifications from text.
        
        Args:
            text: Text to search
            url: Source URL
        """
        for pattern in self.resolution_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                resolution_text = match.group(0)
                entity_id = self._create_entity_id(resolution_text, 'RESOLUTION')
                
                if entity_id not in self.entities:
                    self.entities[entity_id] = Entity(
                        id=entity_id,
                        name=resolution_text,
                        type='RESOLUTION',
                        aliases=[],
                        properties={
                            'value': match.group(1),
                            'full_match': resolution_text,
                            'category': 'spatial_resolution'
                        },
                        confidence=0.8,
                        source_urls=[url]
                    )
    
    def _extract_frequency_entities(self, text: str, url: str):
        """Extract frequency specifications from text.
        
        Args:
            text: Text to search
            url: Source URL
        """
        for pattern in self.frequency_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                frequency_text = match.group(0)
                entity_id = self._create_entity_id(frequency_text, 'FREQUENCY')
                
                if entity_id not in self.entities:
                    self.entities[entity_id] = Entity(
                        id=entity_id,
                        name=frequency_text,
                        type='FREQUENCY',
                        aliases=[],
                        properties={
                            'value': match.group(1),
                            'full_match': frequency_text,
                            'category': 'frequency'
                        },
                        confidence=0.8,
                        source_urls=[url]
                    )
    
    def _extract_relationships(self):
        """Extract relationships between entities."""
        self.logger.info("Extracting relationships between entities...")
        
        # Mission-Instrument relationships
        self._extract_mission_instrument_relationships()
        
        # Mission-Product relationships  
        self._extract_mission_product_relationships()
        
        # Instrument-Product relationships
        self._extract_instrument_product_relationships()
        
        # Mission-Location relationships
        self._extract_mission_location_relationships()
        
        # Product-Format relationships
        self._extract_product_format_relationships()
        
        # Mission-Specification relationships
        self._extract_mission_specification_relationships()
        
        # Co-occurrence relationships
        self._extract_cooccurrence_relationships()
    
    def _extract_mission_instrument_relationships(self):
        """Extract CARRIES relationships between missions and instruments."""
        missions = [e for e in self.entities.values() if e.type == 'MISSION']
        instruments = [e for e in self.entities.values() if e.type == 'INSTRUMENT']
        
        for mission in missions:
            for instrument in instruments:
                # Check if they appear in the same sources
                common_urls = set(mission.source_urls) & set(instrument.source_urls)
                if common_urls:
                    confidence = len(common_urls) / max(len(mission.source_urls), len(instrument.source_urls))
                    
                    if confidence > 0.3:  # Threshold for relationship confidence
                        rel_id = f"{mission.id}_CARRIES_{instrument.id}"
                        relationship = Relationship(
                            id=rel_id,
                            source_entity=mission.id,
                            target_entity=instrument.id,
                            relationship_type='CARRIES',
                            properties={'type': 'mission_instrument'},
                            confidence=confidence,
                            evidence=[f"Co-occurrence in {len(common_urls)} sources"],
                            source_urls=list(common_urls)
                        )
                        self.relationships.append(relationship)
    
    def _extract_mission_product_relationships(self):
        """Extract PRODUCES relationships between missions and products."""
        missions = [e for e in self.entities.values() if e.type == 'MISSION']
        products = [e for e in self.entities.values() if e.type == 'PRODUCT']
        
        for mission in missions:
            for product in products:
                common_urls = set(mission.source_urls) & set(product.source_urls)
                if common_urls:
                    confidence = len(common_urls) / max(len(mission.source_urls), len(product.source_urls))
                    
                    if confidence > 0.2:
                        rel_id = f"{mission.id}_PRODUCES_{product.id}"
                        relationship = Relationship(
                            id=rel_id,
                            source_entity=mission.id,
                            target_entity=product.id,
                            relationship_type='PRODUCES',
                            properties={'type': 'mission_product'},
                            confidence=confidence,
                            evidence=[f"Co-occurrence in {len(common_urls)} sources"],
                            source_urls=list(common_urls)
                        )
                        self.relationships.append(relationship)
    
    def _extract_instrument_product_relationships(self):
        """Extract MEASURES relationships between instruments and products."""
        instruments = [e for e in self.entities.values() if e.type == 'INSTRUMENT']
        products = [e for e in self.entities.values() if e.type == 'PRODUCT']
        
        # Specific instrument-product mappings
        instrument_product_mappings = {
            'Ocean Color Monitor': ['Ocean Color', 'Chlorophyll'],
            'Thermal Infrared Imaging Sensor': ['Sea Surface Temperature'],
            'Scatterometer': ['Wind Speed', 'Wind Direction'],
            'Imager': ['Cloud', 'Water Vapor'],
            'Sounder': ['Temperature', 'Humidity'],
            'MADRAS': ['Precipitation'],
            'SAPHIR': ['Water Vapor', 'Temperature'],
            'ScaRaB': ['Radiation']
        }
        
        for instrument in instruments:
            for product in products:
                confidence = 0.0
                evidence = []
                
                # Check specific mappings
                if instrument.name in instrument_product_mappings:
                    if any(prod.lower() in product.name.lower() 
                          for prod in instrument_product_mappings[instrument.name]):
                        confidence = 0.9
                        evidence.append("Domain knowledge mapping")
                
                # Check co-occurrence
                common_urls = set(instrument.source_urls) & set(product.source_urls)
                if common_urls:
                    co_occurrence_confidence = len(common_urls) / max(len(instrument.source_urls), len(product.source_urls))
                    confidence = max(confidence, co_occurrence_confidence * 0.7)
                    evidence.append(f"Co-occurrence in {len(common_urls)} sources")
                
                if confidence > 0.3:
                    rel_id = f"{instrument.id}_MEASURES_{product.id}"
                    relationship = Relationship(
                        id=rel_id,
                        source_entity=instrument.id,
                        target_entity=product.id,
                        relationship_type='MEASURES',
                        properties={'type': 'instrument_product'},
                        confidence=confidence,
                        evidence=evidence,
                        source_urls=list(common_urls) if common_urls else instrument.source_urls[:1]
                    )
                    self.relationships.append(relationship)
    
    def _extract_mission_location_relationships(self):
        """Extract OPERATES_IN relationships between missions and locations."""
        missions = [e for e in self.entities.values() if e.type == 'MISSION']
        locations = [e for e in self.entities.values() if e.type == 'LOCATION']
        
        for mission in missions:
            for location in locations:
                common_urls = set(mission.source_urls) & set(location.source_urls)
                if common_urls:
                    confidence = len(common_urls) / max(len(mission.source_urls), len(location.source_urls))
                    
                    if confidence > 0.2:
                        rel_id = f"{mission.id}_OPERATES_IN_{location.id}"
                        relationship = Relationship(
                            id=rel_id,
                            source_entity=mission.id,
                            target_entity=location.id,
                            relationship_type='OPERATES_IN',
                            properties={'type': 'mission_location'},
                            confidence=confidence,
                            evidence=[f"Co-occurrence in {len(common_urls)} sources"],
                            source_urls=list(common_urls)
                        )
                        self.relationships.append(relationship)
    
    def _extract_product_format_relationships(self):
        """Extract AVAILABLE_IN relationships between products and formats."""
        products = [e for e in self.entities.values() if e.type == 'PRODUCT']
        formats = [e for e in self.entities.values() if e.type == 'FORMAT']
        
        for product in products:
            for format_entity in formats:
                common_urls = set(product.source_urls) & set(format_entity.source_urls)
                if common_urls:
                    confidence = len(common_urls) / max(len(product.source_urls), len(format_entity.source_urls))
                    
                    if confidence > 0.2:
                        rel_id = f"{product.id}_AVAILABLE_IN_{format_entity.id}"
                        relationship = Relationship(
                            id=rel_id,
                            source_entity=product.id,
                            target_entity=format_entity.id,
                            relationship_type='AVAILABLE_IN',
                            properties={'type': 'product_format'},
                            confidence=confidence,
                            evidence=[f"Co-occurrence in {len(common_urls)} sources"],
                            source_urls=list(common_urls)
                        )
                        self.relationships.append(relationship)
    
    def _extract_mission_specification_relationships(self):
        """Extract HAS_SPECIFICATION relationships between missions and specs."""
        missions = [e for e in self.entities.values() if e.type == 'MISSION']
        specifications = [e for e in self.entities.values() 
                         if e.type in ['RESOLUTION', 'SPECIFICATION', 'FREQUENCY']]
        
        for mission in missions:
            for spec in specifications:
                common_urls = set(mission.source_urls) & set(spec.source_urls)
                if common_urls:
                    confidence = len(common_urls) / max(len(mission.source_urls), len(spec.source_urls))
                    
                    if confidence > 0.3:
                        rel_id = f"{mission.id}_HAS_SPECIFICATION_{spec.id}"
                        relationship = Relationship(
                            id=rel_id,
                            source_entity=mission.id,
                            target_entity=spec.id,
                            relationship_type='HAS_SPECIFICATION',
                            properties={'type': 'mission_specification', 'spec_type': spec.type},
                            confidence=confidence,
                            evidence=[f"Co-occurrence in {len(common_urls)} sources"],
                            source_urls=list(common_urls)
                        )
                        self.relationships.append(relationship)
    
    def _extract_cooccurrence_relationships(self):
        """Extract general RELATED_TO relationships based on co-occurrence."""
        entities_list = list(self.entities.values())
        
        for i, entity1 in enumerate(entities_list):
            for entity2 in entities_list[i+1:]:
                # Skip if already have a specific relationship
                existing_rels = [r for r in self.relationships 
                               if (r.source_entity == entity1.id and r.target_entity == entity2.id) or
                                  (r.source_entity == entity2.id and r.target_entity == entity1.id)]
                
                if existing_rels:
                    continue
                
                # Check co-occurrence
                common_urls = set(entity1.source_urls) & set(entity2.source_urls)
                if len(common_urls) >= 2:  # Require at least 2 common sources
                    confidence = len(common_urls) / max(len(entity1.source_urls), len(entity2.source_urls))
                    
                    if confidence > 0.4:  # Higher threshold for general relationships
                        rel_id = f"{entity1.id}_RELATED_TO_{entity2.id}"
                        relationship = Relationship(
                            id=rel_id,
                            source_entity=entity1.id,
                            target_entity=entity2.id,
                            relationship_type='RELATED_TO',
                            properties={'type': 'co_occurrence'},
                            confidence=confidence,
                            evidence=[f"Strong co-occurrence in {len(common_urls)} sources"],
                            source_urls=list(common_urls)
                        )
                        self.relationships.append(relationship)
    
    def _build_neo4j_graph(self):
        """Build the knowledge graph in Neo4j."""
        self.logger.info("Building knowledge graph in Neo4j...")
        
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create nodes
            for entity in self.entities.values():
                self._create_node(session, entity)
            
            # Create relationships
            for relationship in self.relationships:
                self._create_relationship(session, relationship)
            
            # Create indexes
            self._create_indexes(session)
    
    def _create_node(self, session, entity: Entity):
        """Create a node in Neo4j for an entity."""
        # Convert properties to Neo4j-compatible format
        properties = {
            'name': entity.name,
            'confidence': entity.confidence,
            'mentions': entity.mentions,
            'aliases': entity.aliases,
            'source_urls': entity.source_urls
        }
        
        # Add entity-specific properties
        for key, value in entity.properties.items():
            if isinstance(value, (str, int, float, bool)):
                properties[key] = value
            else:
                properties[key] = str(value)
        
        # Create node with label based on entity type
        query = f"""
        CREATE (n:{entity.type} {{id: $id}})
        SET n += $properties
        """
        
        session.run(query, id=entity.id, properties=properties)
        self.stats['entities_extracted'] += 1
    
    def _create_relationship(self, session, relationship: Relationship):
        """Create a relationship in Neo4j."""
        properties = {
            'confidence': relationship.confidence,
            'evidence': relationship.evidence,
            'source_urls': relationship.source_urls
        }
        
        # Add relationship-specific properties
        for key, value in relationship.properties.items():
            if isinstance(value, (str, int, float, bool)):
                properties[key] = value
            else:
                properties[key] = str(value)
        
        query = f"""
        MATCH (a {{id: $source_id}})
        MATCH (b {{id: $target_id}})
        CREATE (a)-[r:{relationship.relationship_type}]->(b)
        SET r += $properties
        """
        
        session.run(query, 
                   source_id=relationship.source_entity,
                   target_id=relationship.target_entity,
                   properties=properties)
        self.stats['relationships_created'] += 1
    
    def _create_indexes(self, session):
        """Create indexes for better query performance."""
        indexes = [
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (n) ON (n.id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (n) ON (n.name)",
            "CREATE INDEX mission_index IF NOT EXISTS FOR (n:MISSION) ON (n.name)",
            "CREATE INDEX instrument_index IF NOT EXISTS FOR (n:INSTRUMENT) ON (n.name)",
            "CREATE INDEX product_index IF NOT EXISTS FOR (n:PRODUCT) ON (n.name)",
            "CREATE INDEX location_index IF NOT EXISTS FOR (n:LOCATION) ON (n.name)"
        ]
        
        for index_query in indexes:
            try:
                session.run(index_query)
            except Exception as e:
                self.logger.warning(f"Could not create index: {e}")
    
    def _create_entity_id(self, name: str, entity_type: str) -> str:
        """Create a unique entity ID.
        
        Args:
            name: Entity name
            entity_type: Entity type
            
        Returns:
            Unique entity ID
        """
        # Normalize name for ID
        normalized_name = re.sub(r'[^\w\s-]', '', name).strip()
        normalized_name = re.sub(r'\s+', '_', normalized_name)
        return f"{entity_type}_{normalized_name}".upper()
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate processing results and statistics.
        
        Returns:
            Dictionary with results and statistics
        """
        # Count entities by type
        entity_counts = Counter(entity.type for entity in self.entities.values())
        
        # Count relationships by type
        relationship_counts = Counter(rel.relationship_type for rel in self.relationships)
        
        # Calculate confidence statistics
        entity_confidences = [entity.confidence for entity in self.entities.values()]
        relationship_confidences = [rel.confidence for rel in self.relationships]
        
        results = {
            'processing_timestamp': datetime.utcnow().isoformat(),
            'statistics': {
                'documents_processed': self.stats['documents_processed'],
                'total_entities': len(self.entities),
                'total_relationships': len(self.relationships),
                'errors': self.stats['errors']
            },
            'entity_breakdown': dict(entity_counts),
            'relationship_breakdown': dict(relationship_counts),
            'confidence_stats': {
                'entity_confidence_avg': sum(entity_confidences) / len(entity_confidences) if entity_confidences else 0,
                'relationship_confidence_avg': sum(relationship_confidences) / len(relationship_confidences) if relationship_confidences else 0,
                'high_confidence_entities': len([c for c in entity_confidences if c > 0.8]),
                'high_confidence_relationships': len([c for c in relationship_confidences if c > 0.8])
            },
            'sample_entities': [asdict(entity) for entity in list(self.entities.values())[:5]],
            'sample_relationships': [asdict(rel) for rel in self.relationships[:5]]
        }
        
        return results
    
    def export_graph_data(self, output_dir: str = "output"):
        """Export graph data to files.
        
        Args:
            output_dir: Output directory for exported files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export entities
        entities_data = [asdict(entity) for entity in self.entities.values()]
        with open(output_path / "entities.json", 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)
        
        # Export relationships
        relationships_data = [asdict(rel) for rel in self.relationships]
        with open(output_path / "relationships.json", 'w', encoding='utf-8') as f:
            json.dump(relationships_data, f, indent=2, ensure_ascii=False)
        
        # Export to CSV for analysis
        entities_df = pd.DataFrame(entities_data)
        entities_df.to_csv(output_path / "entities.csv", index=False)
        
        relationships_df = pd.DataFrame(relationships_data)
        relationships_df.to_csv(output_path / "relationships.csv", index=False)
        
        self.logger.info(f"Graph data exported to {output_dir}/")
    
    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query on the graph.
        
        Args:
            query: Cypher query string
            
        Returns:
            Query results
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]
    
    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    with MOSDACKnowledgeGraphBuilder() as kg_builder:
        results = kg_builder.process_crawler_output("crawler_output.json")
        print("Knowledge graph built successfully!")
        print(f"Entities: {results['statistics']['total_entities']}")
        print(f"Relationships: {results['statistics']['total_relationships']}")
        
        # Export data
        kg_builder.export_graph_data()
        
        # Example queries
        missions = kg_builder.query_graph("MATCH (m:MISSION) RETURN m.name as mission")
        print(f"Missions found: {[m['mission'] for m in missions]}")