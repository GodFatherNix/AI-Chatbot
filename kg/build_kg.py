#!/usr/bin/env python3
"""Knowledge Graph Builder for MOSDAC content.

Extracts entities and relationships from processed content to build
a knowledge graph for enhanced RAG retrieval and relationship discovery.
"""

import os
import json
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import re

# NLP and entity extraction
import spacy
from spacy.tokens import Doc, Span
import en_core_web_sm

# Graph database
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
from py2neo.bulk import create_nodes, create_relationships

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Relation:
    """Represents a relationship between entities."""
    source: Entity
    target: Entity
    relation_type: str
    confidence: float = 1.0
    context: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MOSDACEntityExtractor:
    """Extract domain-specific entities from MOSDAC content."""
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, downloading...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # MOSDAC-specific entity patterns
        self.mission_patterns = {
            'INSAT-3D', 'OCEANSAT-2', 'SCATSAT-1', 'MEGHA-TROPIQUES',
            'CARTOSAT', 'RESOURCESAT', 'RISAT', 'ASTROSAT', 'CHANDRAYAAN',
            'MARS ORBITER MISSION', 'ADITYA', 'GAGAN'
        }
        
        self.instrument_patterns = {
            'Ocean Color Monitor', 'OCM', 'Ku-band Scatterometer', 'OSCAT',
            'MADRAS', 'SAPHIR', 'ScaRaB', 'ROSA', 'LISS', 'AWiFS', 'PAN',
            'CCD', 'LISS-3', 'LISS-4', 'HySI', 'VHRR', 'CPS', 'SEM'
        }
        
        self.parameter_patterns = {
            'Ocean Color', 'Sea Surface Temperature', 'SST', 'Chlorophyll',
            'Suspended Sediment', 'Wind Speed', 'Wind Direction', 'Wind Vector',
            'Temperature', 'Humidity', 'Water Vapor', 'Precipitation',
            'Cloud Cover', 'Cloud Properties', 'Outgoing Longwave Radiation',
            'Vegetation Index', 'NDVI', 'Land Surface Temperature'
        }
        
        self.location_patterns = {
            'Arabian Sea', 'Bay of Bengal', 'Indian Ocean', 'Tropical Pacific',
            'Indian Subcontinent', 'Monsoon Region', 'Equatorial Pacific'
        }
        
        # Custom entity ruler
        self._add_custom_patterns()
    
    def _add_custom_patterns(self):
        """Add custom entity patterns to spaCy pipeline."""
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self.nlp.get_pipe("entity_ruler")
        
        patterns = []
        
        # Mission patterns
        for mission in self.mission_patterns:
            patterns.append({"label": "MISSION", "pattern": mission})
            patterns.append({"label": "MISSION", "pattern": mission.lower()})
        
        # Instrument patterns
        for instrument in self.instrument_patterns:
            patterns.append({"label": "INSTRUMENT", "pattern": instrument})
        
        # Parameter patterns
        for param in self.parameter_patterns:
            patterns.append({"label": "PARAMETER", "pattern": param})
        
        # Location patterns
        for location in self.location_patterns:
            patterns.append({"label": "LOCATION", "pattern": location})
        
        # Add resolution patterns
        patterns.extend([
            {"label": "RESOLUTION", "pattern": [{"TEXT": {"REGEX": r"\d+"}}, {"LOWER": "m"}]},
            {"label": "RESOLUTION", "pattern": [{"TEXT": {"REGEX": r"\d+"}}, {"LOWER": "km"}]},
            {"label": "RESOLUTION", "pattern": [{"TEXT": {"REGEX": r"\d+"}}, {"LOWER": "meter"}]},
        ])
        
        # Add date patterns
        patterns.extend([
            {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{4}"}}]},
            {"label": "DATE_RANGE", "pattern": [
                {"TEXT": {"REGEX": r"\d{4}"}}, 
                {"TEXT": "-"}, 
                {"TEXT": {"REGEX": r"\d{4}"}}
            ]},
        ])
        
        ruler.add_patterns(patterns)
    
    def extract_entities(self, text: str, source_url: str = "") -> List[Entity]:
        """Extract entities from text."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,
                metadata={
                    'source_url': source_url,
                    'extraction_method': 'spacy_ner'
                }
            )
            entities.append(entity)
        
        # Additional pattern-based extraction
        entities.extend(self._extract_additional_patterns(text, source_url))
        
        return self._deduplicate_entities(entities)
    
    def _extract_additional_patterns(self, text: str, source_url: str) -> List[Entity]:
        """Extract additional domain-specific patterns."""
        entities = []
        
        # Extract coordinate patterns
        coord_pattern = r'(\d+\.?\d*)[°\s]*([NS])[,\s]+(\d+\.?\d*)[°\s]*([EW])'
        for match in re.finditer(coord_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(0),
                label='COORDINATES',
                start=match.start(),
                end=match.end(),
                metadata={'source_url': source_url, 'type': 'coordinates'}
            ))
        
        # Extract frequency patterns
        freq_pattern = r'(\d+\.?\d*)\s*(GHz|MHz|Hz)'
        for match in re.finditer(freq_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(0),
                label='FREQUENCY',
                start=match.start(),
                end=match.end(),
                metadata={'source_url': source_url, 'type': 'frequency'}
            ))
        
        # Extract data format patterns
        format_pattern = r'\b(HDF|NetCDF|GeoTIFF|JPEG|PNG|Binary|ASCII)\b'
        for match in re.finditer(format_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(0),
                label='DATA_FORMAT',
                start=match.start(),
                end=match.end(),
                metadata={'source_url': source_url, 'type': 'data_format'}
            ))
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities with overlapping spans."""
        entities.sort(key=lambda x: (x.start, x.end))
        deduplicated = []
        
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if (entity.start < existing.end and entity.end > existing.start):
                    # Overlapping entities - keep the longer one
                    if len(entity.text) > len(existing.text):
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated


class RelationshipExtractor:
    """Extract relationships between entities."""
    
    def __init__(self):
        self.relation_patterns = {
            'CARRIES': [
                r'(\w+)\s+carries?\s+(\w+)',
                r'(\w+)\s+equipped\s+with\s+(\w+)',
                r'(\w+)\s+has\s+(\w+)'
            ],
            'MEASURES': [
                r'(\w+)\s+measures?\s+(\w+)',
                r'(\w+)\s+monitors?\s+(\w+)',
                r'(\w+)\s+observes?\s+(\w+)'
            ],
            'OPERATES_IN': [
                r'(\w+)\s+operates?\s+in\s+(\w+)',
                r'(\w+)\s+covers?\s+(\w+)',
                r'(\w+)\s+over\s+(\w+)'
            ],
            'LAUNCHED_IN': [
                r'(\w+)\s+launched\s+in\s+(\d{4})',
                r'(\w+)\s+launch\s*:\s*(\d{4})'
            ],
            'HAS_RESOLUTION': [
                r'(\w+).*?resolution\s*:\s*(\d+\s*(?:m|km))',
                r'(\w+).*?(\d+\s*(?:m|km))\s+resolution'
            ]
        }
    
    def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relationships between entities."""
        relationships = []
        
        # Pattern-based relationship extraction
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                relationships.extend(
                    self._extract_pattern_relations(text, entities, pattern, relation_type)
                )
        
        # Co-occurrence based relationships
        relationships.extend(self._extract_cooccurrence_relations(text, entities))
        
        return relationships
    
    def _extract_pattern_relations(self, text: str, entities: List[Entity], 
                                 pattern: str, relation_type: str) -> List[Relation]:
        """Extract relationships using regex patterns."""
        relations = []
        entity_map = {e.text.lower(): e for e in entities}
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            if len(groups) >= 2:
                source_text = groups[0].lower().strip()
                target_text = groups[1].lower().strip()
                
                source_entity = entity_map.get(source_text)
                target_entity = entity_map.get(target_text)
                
                if source_entity and target_entity:
                    relations.append(Relation(
                        source=source_entity,
                        target=target_entity,
                        relation_type=relation_type,
                        context=match.group(0),
                        confidence=0.8,
                        metadata={'extraction_method': 'pattern_based'}
                    ))
        
        return relations
    
    def _extract_cooccurrence_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relationships based on entity co-occurrence."""
        relations = []
        
        # Find entities that appear in the same sentence
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_entities = [
                e for e in entities 
                if e.start >= 0 and sentence.lower().find(e.text.lower()) >= 0
            ]
            
            # Create co-occurrence relationships
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    if entity1.label != entity2.label:  # Different entity types
                        relation_type = self._infer_relation_type(entity1, entity2)
                        
                        relations.append(Relation(
                            source=entity1,
                            target=entity2,
                            relation_type=relation_type,
                            context=sentence.strip(),
                            confidence=0.6,
                            metadata={'extraction_method': 'cooccurrence'}
                        ))
        
        return relations
    
    def _infer_relation_type(self, entity1: Entity, entity2: Entity) -> str:
        """Infer relationship type based on entity labels."""
        label_pairs = {
            ('MISSION', 'INSTRUMENT'): 'CARRIES',
            ('MISSION', 'PARAMETER'): 'MEASURES',
            ('MISSION', 'LOCATION'): 'OPERATES_IN',
            ('INSTRUMENT', 'PARAMETER'): 'MEASURES',
            ('MISSION', 'DATE'): 'LAUNCHED_IN',
            ('MISSION', 'RESOLUTION'): 'HAS_RESOLUTION',
        }
        
        pair = (entity1.label, entity2.label)
        reverse_pair = (entity2.label, entity1.label)
        
        return label_pairs.get(pair, label_pairs.get(reverse_pair, 'RELATED_TO'))


class KnowledgeGraphBuilder:
    """Build and manage the MOSDAC knowledge graph."""
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password"):
        
        # Initialize Neo4j connection
        try:
            self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        
        # Initialize extractors
        self.entity_extractor = MOSDACEntityExtractor()
        self.relation_extractor = RelationshipExtractor()
        
        # Statistics
        self.stats = {
            'processed_documents': 0,
            'extracted_entities': 0,
            'extracted_relations': 0,
            'created_nodes': 0,
            'created_relationships': 0
        }
    
    def build_kg_from_documents(self, data_file: str) -> Dict[str, int]:
        """Build knowledge graph from processed documents."""
        logger.info(f"Building knowledge graph from: {data_file}")
        
        # Load processed data
        with open(data_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        if not isinstance(documents, list):
            documents = [documents]
        
        logger.info(f"Processing {len(documents)} documents for KG extraction")
        
        # Process each document
        all_entities = []
        all_relations = []
        
        for doc in documents:
            try:
                entities, relations = self._process_document(doc)
                all_entities.extend(entities)
                all_relations.extend(relations)
                self.stats['processed_documents'] += 1
                
            except Exception as e:
                logger.error(f"Error processing document {doc.get('url', 'unknown')}: {e}")
                continue
        
        # Create nodes and relationships in Neo4j
        self._create_graph_nodes(all_entities)
        self._create_graph_relationships(all_relations)
        
        logger.info("Knowledge graph construction completed:")
        for key, value in self.stats.items():
            logger.info(f"  {key}: {value}")
        
        return self.stats
    
    def _process_document(self, doc: Dict[str, Any]) -> Tuple[List[Entity], List[Relation]]:
        """Process a single document for entity and relation extraction."""
        text = doc.get('content', '')
        url = doc.get('url', '')
        
        if not text or len(text.strip()) < 50:
            return [], []
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(text, url)
        self.stats['extracted_entities'] += len(entities)
        
        # Extract relationships
        relations = self.relation_extractor.extract_relationships(text, entities)
        self.stats['extracted_relations'] += len(relations)
        
        return entities, relations
    
    def _create_graph_nodes(self, entities: List[Entity]):
        """Create nodes in Neo4j from extracted entities."""
        if not entities:
            return
        
        # Group entities by label for efficient creation
        entities_by_label = {}
        for entity in entities:
            label = entity.label
            if label not in entities_by_label:
                entities_by_label[label] = []
            entities_by_label[label].append(entity)
        
        # Create nodes for each entity type
        for label, entity_list in entities_by_label.items():
            node_data = []
            for entity in entity_list:
                node_data.append({
                    'name': entity.text,
                    'label': entity.label,
                    'confidence': entity.confidence,
                    'source_url': entity.metadata.get('source_url', ''),
                    'created_at': datetime.utcnow().isoformat()
                })
            
            # Batch create nodes
            try:
                create_nodes(
                    self.graph.auto(),
                    node_data,
                    labels={label, "Entity"}
                )
                self.stats['created_nodes'] += len(node_data)
                logger.debug(f"Created {len(node_data)} {label} nodes")
                
            except Exception as e:
                logger.error(f"Error creating {label} nodes: {e}")
    
    def _create_graph_relationships(self, relations: List[Relation]):
        """Create relationships in Neo4j."""
        if not relations:
            return
        
        relationship_data = []
        for relation in relations:
            relationship_data.append({
                'source_name': relation.source.text,
                'source_label': relation.source.label,
                'target_name': relation.target.text,
                'target_label': relation.target.label,
                'relation_type': relation.relation_type,
                'confidence': relation.confidence,
                'context': relation.context,
                'created_at': datetime.utcnow().isoformat()
            })
        
        # Create relationships using Cypher queries
        try:
            for rel_data in relationship_data:
                cypher = """
                MATCH (source:Entity {name: $source_name})
                MATCH (target:Entity {name: $target_name})
                MERGE (source)-[r:`{relation_type}`]->(target)
                SET r.confidence = $confidence,
                    r.context = $context,
                    r.created_at = $created_at
                """.format(relation_type=rel_data['relation_type'])
                
                self.graph.run(cypher, rel_data)
            
            self.stats['created_relationships'] += len(relationship_data)
            logger.debug(f"Created {len(relationship_data)} relationships")
            
        except Exception as e:
            logger.error(f"Error creating relationships: {e}")
    
    def query_knowledge_graph(self, entity_name: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Query the knowledge graph for entity relationships."""
        try:
            cypher = """
            MATCH (e:Entity {name: $entity_name})-[r*1..{max_depth}]-(connected)
            RETURN e.name as source, 
                   type(r[0]) as relationship, 
                   connected.name as target,
                   connected.label as target_type,
                   r[0].confidence as confidence
            LIMIT 50
            """.format(max_depth=max_depth)
            
            results = self.graph.run(cypher, entity_name=entity_name).data()
            return results
            
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return []
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        try:
            stats = {}
            
            # Count nodes by type
            node_counts = self.graph.run("""
                MATCH (n:Entity)
                RETURN n.label as entity_type, count(n) as count
                ORDER BY count DESC
            """).data()
            
            stats['node_counts'] = {item['entity_type']: item['count'] for item in node_counts}
            
            # Count relationships
            rel_counts = self.graph.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                ORDER BY count DESC
            """).data()
            
            stats['relationship_counts'] = {item['relationship_type']: item['count'] for item in rel_counts}
            
            # Total counts
            total_nodes = self.graph.run("MATCH (n:Entity) RETURN count(n) as count").data()[0]['count']
            total_rels = self.graph.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']
            
            stats['total_nodes'] = total_nodes
            stats['total_relationships'] = total_rels
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {}


def main():
    """Main function for testing the KG builder."""
    
    # Example usage
    kg_builder = KnowledgeGraphBuilder()
    
    # Create sample data
    sample_data = [
        {
            "url": "https://www.mosdac.gov.in/test",
            "title": "OCEANSAT-2 Mission",
            "content": """
            OCEANSAT-2 is an Indian satellite designed for ocean studies. 
            The satellite carries Ocean Color Monitor (OCM) and Ku-band 
            scatterometer for wind vector retrieval over oceans.
            
            OCEANSAT-2 was launched in 2009 and operates in sun-synchronous orbit.
            The OCM has 360m resolution and measures ocean color, chlorophyll, 
            and suspended sediment concentration. The satellite covers global oceans
            including Arabian Sea and Bay of Bengal.
            
            Data products include Ocean Color, Sea Surface Temperature (SST), 
            and Chlorophyll concentration in HDF and NetCDF formats.
            """,
            "content_type": "webpage"
        }
    ]
    
    # Save sample data
    with open('sample_kg_data.json', 'w') as f:
        json.dump(sample_data, f)
    
    # Build knowledge graph
    stats = kg_builder.build_kg_from_documents('sample_kg_data.json')
    print("Knowledge graph construction completed!")
    print(f"Statistics: {stats}")
    
    # Test queries
    results = kg_builder.query_knowledge_graph("OCEANSAT-2")
    print(f"\nRelationships for OCEANSAT-2:")
    for result in results:
        print(f"  {result['source']} --{result['relationship']}--> {result['target']}")
    
    # Get graph stats
    graph_stats = kg_builder.get_graph_stats()
    print(f"\nGraph statistics: {graph_stats}")
    
    # Clean up
    os.remove('sample_kg_data.json')


if __name__ == "__main__":
    main()