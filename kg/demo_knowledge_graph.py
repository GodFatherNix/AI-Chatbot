#!/usr/bin/env python3
"""Simplified Knowledge Graph Demo for MOSDAC.

This demonstrates the knowledge graph pipeline without requiring 
external dependencies like spaCy or Neo4j.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Any


@dataclass
class Entity:
    """Represents an extracted entity."""
    id: str
    name: str
    type: str
    confidence: float
    properties: Dict[str, Any]
    source_urls: List[str]


@dataclass 
class Relationship:
    """Represents a relationship between entities."""
    id: str
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    evidence: List[str]


class SimplifiedKnowledgeGraphDemo:
    """Simplified knowledge graph demo without external dependencies."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.logger = logging.getLogger(__name__)
    
    def create_sample_data(self):
        """Create sample crawler output data."""
        return [
            {
                "url": "https://www.mosdac.gov.in/missions/oceansat2",
                "title": "OCEANSAT-2 Mission Details",
                "content": "OCEANSAT-2 satellite for ocean color monitoring with OCM instrument",
                "mission_info": {
                    "mission": "OCEANSAT-2",
                    "launch_info": "September 23, 2009",
                    "orbit_info": "Sun-synchronous polar orbit"
                },
                "product_info": {
                    "products_table": [
                        {"name": "Ocean Color", "description": "Chlorophyll-a concentration"},
                        {"name": "Sea Surface Temperature", "description": "SST measurements"}
                    ],
                    "data_formats": ["HDF", "NetCDF"]
                },
                "technical_specs": {
                    "resolution": "360m",
                    "spectral_bands": "8"
                },
                "coverage_info": {
                    "regions": ["Indian Ocean", "Arabian Sea"]
                }
            },
            {
                "url": "https://www.mosdac.gov.in/missions/insat3d",
                "title": "INSAT-3D Weather Satellite",
                "content": "INSAT-3D geostationary satellite with Imager and Sounder instruments",
                "mission_info": {
                    "mission": "INSAT-3D",
                    "launch_info": "July 26, 2013",
                    "orbit_info": "Geostationary orbit"
                },
                "product_info": {
                    "products": ["Temperature", "Humidity", "Cloud"],
                    "data_formats": ["HDF", "NetCDF"]
                },
                "coverage_info": {
                    "regions": ["Indian Ocean", "Indian Subcontinent"]
                }
            },
            {
                "url": "https://www.mosdac.gov.in/missions/scatsat1",
                "title": "SCATSAT-1 Wind Monitoring",
                "content": "SCATSAT-1 with OSCAT scatterometer for wind measurements",
                "mission_info": {
                    "mission": "SCATSAT-1",
                    "launch_info": "September 26, 2016",
                    "orbit_info": "Sun-synchronous orbit"
                },
                "product_info": {
                    "products": ["Wind Speed", "Wind Direction"],
                    "data_formats": ["HDF", "Binary"]
                },
                "technical_specs": {
                    "resolution": "25 km",
                    "frequency": "13.515 GHz"
                },
                "coverage_info": {
                    "regions": ["Global"]
                }
            }
        ]
    
    def extract_entities(self, documents):
        """Extract entities from documents."""
        self.logger.info("Extracting entities from documents...")
        
        for doc in documents:
            url = doc.get('url', '')
            
            # Extract mission entities
            if 'mission_info' in doc and 'mission' in doc['mission_info']:
                mission_name = doc['mission_info']['mission']
                entity_id = f"MISSION_{mission_name.replace('-', '_')}"
                
                self.entities[entity_id] = Entity(
                    id=entity_id,
                    name=mission_name,
                    type='MISSION',
                    confidence=1.0,
                    properties=doc['mission_info'],
                    source_urls=[url]
                )
            
            # Extract product entities
            if 'product_info' in doc:
                # From products list
                if 'products' in doc['product_info']:
                    for product in doc['product_info']['products']:
                        entity_id = f"PRODUCT_{product.replace(' ', '_').upper()}"
                        
                        if entity_id not in self.entities:
                            self.entities[entity_id] = Entity(
                                id=entity_id,
                                name=product,
                                type='PRODUCT',
                                confidence=0.9,
                                properties={},
                                source_urls=[url]
                            )
                
                # From products table
                if 'products_table' in doc['product_info']:
                    for product_entry in doc['product_info']['products_table']:
                        product_name = product_entry['name']
                        entity_id = f"PRODUCT_{product_name.replace(' ', '_').upper()}"
                        
                        if entity_id not in self.entities:
                            self.entities[entity_id] = Entity(
                                id=entity_id,
                                name=product_name,
                                type='PRODUCT',
                                confidence=0.95,
                                properties=product_entry,
                                source_urls=[url]
                            )
                
                # Extract format entities
                if 'data_formats' in doc['product_info']:
                    for format_name in doc['product_info']['data_formats']:
                        entity_id = f"FORMAT_{format_name}"
                        
                        if entity_id not in self.entities:
                            self.entities[entity_id] = Entity(
                                id=entity_id,
                                name=format_name,
                                type='FORMAT',
                                confidence=0.8,
                                properties={'category': 'data_format'},
                                source_urls=[url]
                            )
            
            # Extract location entities
            if 'coverage_info' in doc and 'regions' in doc['coverage_info']:
                for region in doc['coverage_info']['regions']:
                    entity_id = f"LOCATION_{region.replace(' ', '_').upper()}"
                    
                    if entity_id not in self.entities:
                        self.entities[entity_id] = Entity(
                            id=entity_id,
                            name=region,
                            type='LOCATION',
                            confidence=0.9,
                            properties={'category': 'geographic_region'},
                            source_urls=[url]
                        )
            
            # Extract technical specification entities
            if 'technical_specs' in doc:
                if 'resolution' in doc['technical_specs']:
                    resolution = doc['technical_specs']['resolution']
                    entity_id = f"RESOLUTION_{resolution.replace(' ', '_')}"
                    
                    if entity_id not in self.entities:
                        self.entities[entity_id] = Entity(
                            id=entity_id,
                            name=resolution,
                            type='RESOLUTION',
                            confidence=0.9,
                            properties={'category': 'spatial_resolution'},
                            source_urls=[url]
                        )
        
        self.logger.info(f"Extracted {len(self.entities)} entities")
    
    def extract_relationships(self):
        """Extract relationships between entities."""
        self.logger.info("Extracting relationships...")
        
        # Get entities by type
        missions = [e for e in self.entities.values() if e.type == 'MISSION']
        products = [e for e in self.entities.values() if e.type == 'PRODUCT']
        locations = [e for e in self.entities.values() if e.type == 'LOCATION']
        formats = [e for e in self.entities.values() if e.type == 'FORMAT']
        
        # Mission-Product relationships (PRODUCES)
        for mission in missions:
            for product in products:
                # Check if they appear in same sources
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
                            confidence=confidence,
                            evidence=[f"Co-occurrence in {len(common_urls)} sources"]
                        )
                        self.relationships.append(relationship)
        
        # Mission-Location relationships (OPERATES_IN)
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
                            confidence=confidence,
                            evidence=[f"Co-occurrence in {len(common_urls)} sources"]
                        )
                        self.relationships.append(relationship)
        
        # Product-Format relationships (AVAILABLE_IN)
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
                            confidence=confidence,
                            evidence=[f"Co-occurrence in {len(common_urls)} sources"]
                        )
                        self.relationships.append(relationship)
        
        # Add domain-specific instrument relationships
        self.add_instrument_relationships()
        
        self.logger.info(f"Extracted {len(self.relationships)} relationships")
    
    def add_instrument_relationships(self):
        """Add instrument entities and relationships based on domain knowledge."""
        # Known mission-instrument mappings
        instrument_mappings = {
            'OCEANSAT-2': 'Ocean Color Monitor',
            'INSAT-3D': 'Imager',
            'SCATSAT-1': 'Scatterometer'
        }
        
        # Known instrument-product mappings
        product_mappings = {
            'Ocean Color Monitor': ['Ocean Color', 'Sea Surface Temperature'],
            'Imager': ['Temperature', 'Humidity', 'Cloud'],
            'Scatterometer': ['Wind Speed', 'Wind Direction']
        }
        
        # Add instrument entities
        for mission_name, instrument_name in instrument_mappings.items():
            entity_id = f"INSTRUMENT_{instrument_name.replace(' ', '_').upper()}"
            
            if entity_id not in self.entities:
                # Find mission entity to get source URL
                mission_entity = None
                for entity in self.entities.values():
                    if entity.type == 'MISSION' and entity.name == mission_name:
                        mission_entity = entity
                        break
                
                if mission_entity:
                    self.entities[entity_id] = Entity(
                        id=entity_id,
                        name=instrument_name,
                        type='INSTRUMENT',
                        confidence=0.95,
                        properties={'derived_from': 'domain_knowledge'},
                        source_urls=mission_entity.source_urls
                    )
        
        # Add CARRIES relationships (Mission -> Instrument)
        for mission_name, instrument_name in instrument_mappings.items():
            mission_id = f"MISSION_{mission_name.replace('-', '_')}"
            instrument_id = f"INSTRUMENT_{instrument_name.replace(' ', '_').upper()}"
            
            if mission_id in self.entities and instrument_id in self.entities:
                rel_id = f"{mission_id}_CARRIES_{instrument_id}"
                relationship = Relationship(
                    id=rel_id,
                    source_entity=mission_id,
                    target_entity=instrument_id,
                    relationship_type='CARRIES',
                    confidence=0.95,
                    evidence=['Domain knowledge']
                )
                self.relationships.append(relationship)
        
        # Add MEASURES relationships (Instrument -> Product)
        for instrument_name, product_names in product_mappings.items():
            instrument_id = f"INSTRUMENT_{instrument_name.replace(' ', '_').upper()}"
            
            for product_name in product_names:
                product_id = f"PRODUCT_{product_name.replace(' ', '_').upper()}"
                
                if instrument_id in self.entities and product_id in self.entities:
                    rel_id = f"{instrument_id}_MEASURES_{product_id}"
                    relationship = Relationship(
                        id=rel_id,
                        source_entity=instrument_id,
                        target_entity=product_id,
                        relationship_type='MEASURES',
                        confidence=0.9,
                        evidence=['Domain knowledge']
                    )
                    self.relationships.append(relationship)
    
    def generate_statistics(self):
        """Generate statistics about the knowledge graph."""
        # Count entities by type
        entity_counts = Counter(entity.type for entity in self.entities.values())
        
        # Count relationships by type
        relationship_counts = Counter(rel.relationship_type for rel in self.relationships)
        
        # Calculate confidence statistics
        entity_confidences = [entity.confidence for entity in self.entities.values()]
        relationship_confidences = [rel.confidence for rel in self.relationships]
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'statistics': {
                'total_entities': len(self.entities),
                'total_relationships': len(self.relationships),
            },
            'entity_breakdown': dict(entity_counts),
            'relationship_breakdown': dict(relationship_counts),
            'confidence_stats': {
                'entity_confidence_avg': sum(entity_confidences) / len(entity_confidences) if entity_confidences else 0,
                'relationship_confidence_avg': sum(relationship_confidences) / len(relationship_confidences) if relationship_confidences else 0,
                'high_confidence_entities': len([c for c in entity_confidences if c > 0.8]),
                'high_confidence_relationships': len([c for c in relationship_confidences if c > 0.8])
            }
        }
    
    def export_results(self, output_dir="kg_demo_output"):
        """Export results to files."""
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
        
        # Export statistics
        stats = self.generate_statistics()
        with open(output_path / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Export visualization data
        viz_data = self.generate_visualization_data()
        with open(output_path / "visualization.json", 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def generate_visualization_data(self):
        """Generate data for graph visualization."""
        nodes = []
        for entity in self.entities.values():
            nodes.append({
                'id': entity.id,
                'name': entity.name,
                'type': entity.type,
                'confidence': entity.confidence,
                'color': self.get_node_color(entity.type)
            })
        
        edges = []
        for rel in self.relationships:
            edges.append({
                'source': rel.source_entity,
                'target': rel.target_entity,
                'relationship': rel.relationship_type,
                'confidence': rel.confidence,
                'color': self.get_edge_color(rel.relationship_type)
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'summary': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'node_types': len(set(node['type'] for node in nodes)),
                'relationship_types': len(set(edge['relationship'] for edge in edges))
            }
        }
    
    def get_node_color(self, entity_type):
        """Get color for node visualization."""
        colors = {
            'MISSION': '#FF6B6B',
            'INSTRUMENT': '#4ECDC4',
            'PRODUCT': '#45B7D1', 
            'LOCATION': '#96CEB4',
            'FORMAT': '#FFEAA7',
            'RESOLUTION': '#DDA0DD'
        }
        return colors.get(entity_type, '#BDC3C7')
    
    def get_edge_color(self, relationship_type):
        """Get color for edge visualization."""
        colors = {
            'CARRIES': '#E74C3C',
            'MEASURES': '#3498DB',
            'PRODUCES': '#2ECC71',
            'OPERATES_IN': '#F39C12',
            'AVAILABLE_IN': '#9B59B6'
        }
        return colors.get(relationship_type, '#95A5A6')
    
    def print_summary(self):
        """Print a summary of the knowledge graph."""
        stats = self.generate_statistics()
        
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH SUMMARY")
        print("="*60)
        
        print(f"Total Entities: {stats['statistics']['total_entities']}")
        print(f"Total Relationships: {stats['statistics']['total_relationships']}")
        
        print(f"\nEntity Breakdown:")
        for entity_type, count in stats['entity_breakdown'].items():
            print(f"  {entity_type}: {count}")
        
        print(f"\nRelationship Breakdown:")
        for rel_type, count in stats['relationship_breakdown'].items():
            print(f"  {rel_type}: {count}")
        
        print(f"\nConfidence Statistics:")
        print(f"  Average Entity Confidence: {stats['confidence_stats']['entity_confidence_avg']:.2f}")
        print(f"  Average Relationship Confidence: {stats['confidence_stats']['relationship_confidence_avg']:.2f}")
        print(f"  High Confidence Entities: {stats['confidence_stats']['high_confidence_entities']}")
        print(f"  High Confidence Relationships: {stats['confidence_stats']['high_confidence_relationships']}")
        
        print(f"\nSample Entities:")
        for i, entity in enumerate(list(self.entities.values())[:5]):
            print(f"  {entity.name} ({entity.type}) - confidence: {entity.confidence:.2f}")
        
        print(f"\nSample Relationships:")
        for i, rel in enumerate(self.relationships[:5]):
            source_name = self.entities[rel.source_entity].name
            target_name = self.entities[rel.target_entity].name
            print(f"  {source_name} --{rel.relationship_type}--> {target_name} (confidence: {rel.confidence:.2f})")


def main():
    """Run the knowledge graph demo."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("MOSDAC Knowledge Graph Demo")
    print("This demonstrates knowledge graph creation from satellite data")
    print("without requiring external dependencies like spaCy or Neo4j.\n")
    
    # Initialize demo
    kg_demo = SimplifiedKnowledgeGraphDemo()
    
    # Create sample data
    logger.info("Creating sample crawler data...")
    sample_data = kg_demo.create_sample_data()
    print(f"✓ Created sample data with {len(sample_data)} documents")
    
    # Extract entities
    logger.info("Extracting entities...")
    kg_demo.extract_entities(sample_data)
    print(f"✓ Extracted {len(kg_demo.entities)} entities")
    
    # Extract relationships
    logger.info("Extracting relationships...")
    kg_demo.extract_relationships()
    print(f"✓ Extracted {len(kg_demo.relationships)} relationships")
    
    # Export results
    logger.info("Exporting results...")
    output_path = kg_demo.export_results()
    print(f"✓ Results exported to {output_path}/")
    
    # Print summary
    kg_demo.print_summary()
    
    print(f"\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Files created in {output_path}/:")
    print("  - entities.json (entity data)")
    print("  - relationships.json (relationship data)")
    print("  - statistics.json (graph statistics)")
    print("  - visualization.json (visualization data)")
    
    print(f"\nThis demonstrates the core knowledge graph functionality.")
    print(f"The full implementation includes:")
    print(f"  - spaCy NLP for advanced entity extraction")
    print(f"  - Neo4j integration for graph database storage")
    print(f"  - Advanced relationship inference algorithms")
    print(f"  - Integration with the web crawler pipeline")


if __name__ == "__main__":
    main()