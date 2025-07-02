#!/usr/bin/env python3
"""MOSDAC Knowledge Graph Pipeline Runner.

This script orchestrates the complete knowledge graph creation pipeline:
1. Load crawler output data
2. Extract entities and relationships  
3. Build Neo4j knowledge graph
4. Export results and generate reports
5. Provide query interface

Usage:
    python run_knowledge_graph_pipeline.py --input crawler_output.json
    python run_knowledge_graph_pipeline.py --input data/ --batch
    python run_knowledge_graph_pipeline.py --demo
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from knowledge_graph_builder import MOSDACKnowledgeGraphBuilder


def setup_logging(log_level='INFO', log_file=None):
    """Setup comprehensive logging."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def check_dependencies(logger):
    """Check if required dependencies are available."""
    logger.info("Checking dependencies...")
    
    missing_deps = []
    warnings = []
    
    # Check spaCy
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("✓ spaCy with en_core_web_sm model available")
        except OSError:
            warnings.append("spaCy en_core_web_sm model not found. Install with: python -m spacy download en_core_web_sm")
    except ImportError:
        missing_deps.append("spacy")
    
    # Check Neo4j driver
    try:
        import neo4j
        logger.info("✓ Neo4j driver available")
    except ImportError:
        missing_deps.append("neo4j")
    
    # Check pandas
    try:
        import pandas
        logger.info("✓ Pandas available")
    except ImportError:
        missing_deps.append("pandas")
    
    if missing_deps:
        logger.error(f"Missing required dependencies: {missing_deps}")
        logger.error("Install with: pip install " + " ".join(missing_deps))
        return False
    
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    
    logger.info("✓ All required dependencies available")
    return True


def test_neo4j_connection(neo4j_uri, neo4j_user, neo4j_password, logger):
    """Test Neo4j database connection."""
    logger.info(f"Testing Neo4j connection to {neo4j_uri}...")
    
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            result = session.run("RETURN 'Hello Neo4j' as message")
            record = result.single()
            if record and record['message'] == 'Hello Neo4j':
                logger.info("✓ Neo4j connection successful")
                driver.close()
                return True
            else:
                logger.error("✗ Neo4j connection test failed")
                driver.close()
                return False
                
    except Exception as e:
        logger.error(f"✗ Neo4j connection failed: {e}")
        return False


def load_crawler_data(input_path, logger):
    """Load crawler data from file or directory."""
    logger.info(f"Loading crawler data from {input_path}...")
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Ensure data is a list
            if isinstance(data, dict):
                data = [data]
            
            logger.info(f"✓ Loaded {len(data)} documents from {input_path}")
            return data
            
        except Exception as e:
            logger.error(f"✗ Failed to load {input_path}: {e}")
            return None
    
    elif input_path.is_dir():
        # Directory with multiple files
        all_data = []
        json_files = list(input_path.glob("*.json"))
        
        if not json_files:
            logger.error(f"✗ No JSON files found in {input_path}")
            return None
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                # Ensure file data is a list
                if isinstance(file_data, dict):
                    file_data = [file_data]
                
                all_data.extend(file_data)
                logger.info(f"  Loaded {len(file_data)} documents from {json_file.name}")
                
            except Exception as e:
                logger.warning(f"  Failed to load {json_file}: {e}")
        
        logger.info(f"✓ Loaded total {len(all_data)} documents from {len(json_files)} files")
        return all_data
    
    else:
        logger.error(f"✗ Input path {input_path} does not exist")
        return None


def create_demo_data():
    """Create demonstration data for testing."""
    demo_data = [
        {
            "url": "https://www.mosdac.gov.in/missions/oceansat2",
            "title": "OCEANSAT-2 Mission Details",
            "content": """OCEANSAT-2 is an Indian satellite designed for ocean color monitoring and 
            sea surface temperature (SST) measurements. The satellite was launched on 
            September 23, 2009 from Satish Dhawan Space Centre. It carries the Ocean Color 
            Monitor (OCM) instrument with 8 spectral bands and 360m spatial resolution.""",
            "content_type": "webpage",
            "crawled_at": "2024-01-15T10:30:00Z",
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
            "content": """INSAT-3D is a multipurpose geostationary satellite for weather monitoring. 
            It carries an Imager and Sounder for atmospheric observations providing temperature 
            and humidity profiles over the Indian Ocean region.""",
            "content_type": "webpage", 
            "crawled_at": "2024-01-15T10:35:00Z",
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
            "content": """SCATSAT-1 carries the OSCAT scatterometer for measuring ocean surface 
            wind speed and direction. It provides global coverage with 25 km spatial resolution 
            supporting weather forecasting and cyclone monitoring.""",
            "content_type": "webpage",
            "crawled_at": "2024-01-15T10:40:00Z",
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
    
    return demo_data


def run_knowledge_graph_pipeline(
    input_data: List[Dict[str, Any]],
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j", 
    neo4j_password: str = "password",
    output_dir: str = "kg_output",
    logger=None
):
    """Run the complete knowledge graph pipeline.
    
    Args:
        input_data: List of crawler output documents
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username  
        neo4j_password: Neo4j password
        output_dir: Output directory for results
        logger: Logger instance
        
    Returns:
        Dictionary with pipeline results
    """
    if not logger:
        logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("STARTING KNOWLEDGE GRAPH PIPELINE")
    logger.info("="*60)
    
    pipeline_start_time = datetime.utcnow()
    
    try:
        # Initialize knowledge graph builder
        logger.info("Initializing knowledge graph builder...")
        
        with MOSDACKnowledgeGraphBuilder(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user, 
            neo4j_password=neo4j_password
        ) as kg_builder:
            
            # Save input data temporarily
            temp_input_file = Path(output_dir) / "temp_input.json"
            temp_input_file.parent.mkdir(exist_ok=True)
            
            with open(temp_input_file, 'w', encoding='utf-8') as f:
                json.dump(input_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(input_data)} documents to temporary file")
            
            # Process crawler output
            logger.info("Processing crawler output...")
            results = kg_builder.process_crawler_output(str(temp_input_file))
            
            # Export graph data
            logger.info("Exporting graph data...")
            kg_builder.export_graph_data(output_dir)
            
            # Generate detailed statistics
            logger.info("Generating detailed statistics...")
            detailed_stats = generate_detailed_statistics(kg_builder, results, logger)
            
            # Save detailed results
            with open(Path(output_dir) / "pipeline_results.json", 'w', encoding='utf-8') as f:
                json.dump(detailed_stats, f, indent=2, ensure_ascii=False)
            
            # Generate sample queries
            logger.info("Generating sample queries...")
            sample_queries = generate_sample_queries(kg_builder, logger)
            
            with open(Path(output_dir) / "sample_queries.json", 'w', encoding='utf-8') as f:
                json.dump(sample_queries, f, indent=2, ensure_ascii=False)
            
            # Generate visualization data
            logger.info("Generating visualization data...")
            viz_data = generate_visualization_data(kg_builder, logger)
            
            with open(Path(output_dir) / "visualization_data.json", 'w', encoding='utf-8') as f:
                json.dump(viz_data, f, indent=2, ensure_ascii=False)
            
            # Clean up temporary file
            temp_input_file.unlink()
            
            pipeline_end_time = datetime.utcnow()
            pipeline_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
            
            logger.info("="*60)
            logger.info("KNOWLEDGE GRAPH PIPELINE COMPLETED")
            logger.info("="*60)
            logger.info(f"Duration: {pipeline_duration:.2f} seconds")
            logger.info(f"Entities created: {detailed_stats['statistics']['total_entities']}")
            logger.info(f"Relationships created: {detailed_stats['statistics']['total_relationships']}")
            logger.info(f"Results saved to: {output_dir}/")
            
            return {
                'status': 'success',
                'duration_seconds': pipeline_duration,
                'results': detailed_stats,
                'output_directory': output_dir
            }
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return {
            'status': 'failed',
            'error': str(e),
            'output_directory': output_dir
        }


def generate_detailed_statistics(kg_builder, results, logger):
    """Generate detailed statistics about the knowledge graph."""
    logger.info("Analyzing knowledge graph structure...")
    
    # Enhanced entity analysis
    entities_by_type = {}
    entities_by_confidence = {'high': 0, 'medium': 0, 'low': 0}
    
    for entity in kg_builder.entities.values():
        # Group by type
        if entity.type not in entities_by_type:
            entities_by_type[entity.type] = {
                'count': 0,
                'avg_confidence': 0,
                'examples': []
            }
        
        entities_by_type[entity.type]['count'] += 1
        entities_by_type[entity.type]['avg_confidence'] += entity.confidence
        
        if len(entities_by_type[entity.type]['examples']) < 3:
            entities_by_type[entity.type]['examples'].append(entity.name)
        
        # Group by confidence
        if entity.confidence >= 0.8:
            entities_by_confidence['high'] += 1
        elif entity.confidence >= 0.5:
            entities_by_confidence['medium'] += 1
        else:
            entities_by_confidence['low'] += 1
    
    # Calculate average confidences
    for entity_type in entities_by_type:
        count = entities_by_type[entity_type]['count']
        entities_by_type[entity_type]['avg_confidence'] /= count
    
    # Enhanced relationship analysis
    relationships_by_type = {}
    relationship_confidence_stats = {}
    
    for rel in kg_builder.relationships:
        # Group by type
        if rel.relationship_type not in relationships_by_type:
            relationships_by_type[rel.relationship_type] = {
                'count': 0,
                'avg_confidence': 0,
                'examples': []
            }
        
        relationships_by_type[rel.relationship_type]['count'] += 1
        relationships_by_type[rel.relationship_type]['avg_confidence'] += rel.confidence
        
        if len(relationships_by_type[rel.relationship_type]['examples']) < 3:
            source_name = kg_builder.entities[rel.source_entity].name
            target_name = kg_builder.entities[rel.target_entity].name
            relationships_by_type[rel.relationship_type]['examples'].append(
                f"{source_name} → {target_name}"
            )
    
    # Calculate average confidences
    for rel_type in relationships_by_type:
        count = relationships_by_type[rel_type]['count']
        relationships_by_type[rel_type]['avg_confidence'] /= count
    
    # Network analysis
    source_entities = set(rel.source_entity for rel in kg_builder.relationships)
    target_entities = set(rel.target_entity for rel in kg_builder.relationships) 
    connected_entities = source_entities | target_entities
    isolated_entities = set(kg_builder.entities.keys()) - connected_entities
    
    network_stats = {
        'total_nodes': len(kg_builder.entities),
        'total_edges': len(kg_builder.relationships),
        'connected_nodes': len(connected_entities),
        'isolated_nodes': len(isolated_entities),
        'graph_density': len(kg_builder.relationships) / (len(kg_builder.entities) * (len(kg_builder.entities) - 1)) if len(kg_builder.entities) > 1 else 0
    }
    
    # Combine with original results
    detailed_stats = results.copy()
    detailed_stats.update({
        'detailed_entity_analysis': entities_by_type,
        'entity_confidence_distribution': entities_by_confidence,
        'detailed_relationship_analysis': relationships_by_type,
        'network_analysis': network_stats,
        'top_entities_by_mentions': [
            {'name': entity.name, 'type': entity.type, 'mentions': entity.mentions}
            for entity in sorted(kg_builder.entities.values(), key=lambda x: x.mentions, reverse=True)[:10]
        ]
    })
    
    return detailed_stats


def generate_sample_queries(kg_builder, logger):
    """Generate sample Cypher queries for the knowledge graph."""
    logger.info("Generating sample queries...")
    
    sample_queries = {
        'basic_queries': {
            'all_missions': {
                'description': 'Find all satellite missions',
                'cypher': 'MATCH (m:MISSION) RETURN m.name as mission, m.launch_info as launch_info ORDER BY m.name',
                'explanation': 'Returns all mission nodes with their names and launch information'
            },
            'all_instruments': {
                'description': 'Find all instruments',
                'cypher': 'MATCH (i:INSTRUMENT) RETURN i.name as instrument ORDER BY i.name',
                'explanation': 'Returns all instrument nodes'
            },
            'all_products': {
                'description': 'Find all data products',
                'cypher': 'MATCH (p:PRODUCT) RETURN p.name as product, p.description as description ORDER BY p.name',
                'explanation': 'Returns all product nodes with descriptions'
            }
        },
        'relationship_queries': {
            'mission_instruments': {
                'description': 'Find instruments carried by each mission',
                'cypher': 'MATCH (m:MISSION)-[r:CARRIES]->(i:INSTRUMENT) RETURN m.name as mission, i.name as instrument, r.confidence as confidence ORDER BY m.name',
                'explanation': 'Shows which instruments are carried by which missions'
            },
            'instrument_products': {
                'description': 'Find products measured by each instrument',
                'cypher': 'MATCH (i:INSTRUMENT)-[r:MEASURES]->(p:PRODUCT) RETURN i.name as instrument, p.name as product, r.confidence as confidence ORDER BY i.name',
                'explanation': 'Shows which products are measured by which instruments'
            },
            'mission_locations': {
                'description': 'Find operating regions for each mission',
                'cypher': 'MATCH (m:MISSION)-[r:OPERATES_IN]->(l:LOCATION) RETURN m.name as mission, l.name as location, r.confidence as confidence ORDER BY m.name',
                'explanation': 'Shows where each mission operates'
            }
        },
        'complex_queries': {
            'mission_product_chain': {
                'description': 'Find complete mission-instrument-product chains',
                'cypher': '''
                MATCH (m:MISSION)-[r1:CARRIES]->(i:INSTRUMENT)-[r2:MEASURES]->(p:PRODUCT)
                RETURN m.name as mission, i.name as instrument, p.name as product,
                       r1.confidence as carries_confidence, r2.confidence as measures_confidence
                ORDER BY m.name, i.name
                ''',
                'explanation': 'Shows complete chains from missions through instruments to products'
            },
            'high_confidence_relationships': {
                'description': 'Find high-confidence relationships (>0.8)',
                'cypher': '''
                MATCH (a)-[r]->(b)
                WHERE r.confidence > 0.8
                RETURN labels(a)[0] as source_type, a.name as source,
                       type(r) as relationship, r.confidence as confidence,
                       labels(b)[0] as target_type, b.name as target
                ORDER BY r.confidence DESC
                ''',
                'explanation': 'Returns only high-confidence relationships'
            },
            'ocean_related_entities': {
                'description': 'Find entities related to ocean monitoring',
                'cypher': '''
                MATCH (n)
                WHERE n.name =~ '(?i).*ocean.*' OR n.name =~ '(?i).*sea.*' OR n.name =~ '(?i).*marine.*'
                RETURN labels(n)[0] as type, n.name as name, n.confidence as confidence
                ORDER BY n.confidence DESC
                ''',
                'explanation': 'Finds entities with ocean-related names'
            }
        },
        'analytical_queries': {
            'entity_statistics': {
                'description': 'Get statistics about entity types',
                'cypher': '''
                MATCH (n)
                RETURN labels(n)[0] as entity_type, count(n) as count,
                       avg(n.confidence) as avg_confidence, max(n.confidence) as max_confidence
                ORDER BY count DESC
                ''',
                'explanation': 'Provides statistics about different entity types'
            },
            'relationship_statistics': {
                'description': 'Get statistics about relationship types',
                'cypher': '''
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count,
                       avg(r.confidence) as avg_confidence
                ORDER BY count DESC
                ''',
                'explanation': 'Provides statistics about different relationship types'
            },
            'connected_components': {
                'description': 'Find nodes with most connections',
                'cypher': '''
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-()
                RETURN labels(n)[0] as type, n.name as name, count(r) as connections
                ORDER BY connections DESC
                LIMIT 10
                ''',
                'explanation': 'Shows nodes with the most connections (hubs)'
            }
        }
    }
    
    # Execute sample queries if possible
    try:
        for category in sample_queries:
            for query_name, query_info in sample_queries[category].items():
                try:
                    results = kg_builder.query_graph(query_info['cypher'])
                    query_info['sample_results'] = results[:5]  # First 5 results
                    query_info['result_count'] = len(results)
                except Exception as e:
                    query_info['error'] = str(e)
    except Exception as e:
        logger.warning(f"Could not execute sample queries: {e}")
    
    return sample_queries


def generate_visualization_data(kg_builder, logger):
    """Generate data for graph visualization."""
    logger.info("Generating visualization data...")
    
    # Node data
    nodes = []
    for entity in kg_builder.entities.values():
        nodes.append({
            'id': entity.id,
            'name': entity.name,
            'type': entity.type,
            'confidence': entity.confidence,
            'mentions': entity.mentions,
            'size': min(max(entity.mentions * 5, 10), 50),  # Scale node size
            'color': get_node_color(entity.type)
        })
    
    # Edge data
    edges = []
    for rel in kg_builder.relationships:
        edges.append({
            'source': rel.source_entity,
            'target': rel.target_entity,
            'relationship': rel.relationship_type,
            'confidence': rel.confidence,
            'width': max(rel.confidence * 5, 1),  # Scale edge width
            'color': get_edge_color(rel.relationship_type)
        })
    
    # Network statistics for layout
    node_degrees = {}
    for edge in edges:
        node_degrees[edge['source']] = node_degrees.get(edge['source'], 0) + 1
        node_degrees[edge['target']] = node_degrees.get(edge['target'], 0) + 1
    
    # Add degree information to nodes
    for node in nodes:
        node['degree'] = node_degrees.get(node['id'], 0)
    
    visualization_data = {
        'nodes': nodes,
        'edges': edges,
        'statistics': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'node_types': len(set(node['type'] for node in nodes)),
            'relationship_types': len(set(edge['relationship'] for edge in edges))
        },
        'layout_suggestions': {
            'force_directed': 'Good for general overview',
            'hierarchical': 'Good for mission -> instrument -> product chains',
            'circular': 'Good for showing all entity types'
        }
    }
    
    return visualization_data


def get_node_color(entity_type):
    """Get color for node based on entity type."""
    colors = {
        'MISSION': '#FF6B6B',
        'INSTRUMENT': '#4ECDC4', 
        'PRODUCT': '#45B7D1',
        'LOCATION': '#96CEB4',
        'FORMAT': '#FFEAA7',
        'RESOLUTION': '#DDA0DD',
        'SPECIFICATION': '#F0E68C',
        'FREQUENCY': '#FFB347'
    }
    return colors.get(entity_type, '#BDC3C7')


def get_edge_color(relationship_type):
    """Get color for edge based on relationship type."""
    colors = {
        'CARRIES': '#E74C3C',
        'MEASURES': '#3498DB',
        'PRODUCES': '#2ECC71',
        'OPERATES_IN': '#F39C12',
        'AVAILABLE_IN': '#9B59B6',
        'HAS_SPECIFICATION': '#1ABC9C',
        'RELATED_TO': '#95A5A6'
    }
    return colors.get(relationship_type, '#BDC3C7')


def interactive_query_mode(kg_builder, logger):
    """Interactive mode for querying the knowledge graph."""
    logger.info("\n" + "="*60)
    logger.info("ENTERING INTERACTIVE QUERY MODE")
    logger.info("="*60)
    logger.info("Type 'help' for sample queries, 'quit' to exit")
    
    sample_queries = {
        '1': 'MATCH (m:MISSION) RETURN m.name ORDER BY m.name',
        '2': 'MATCH (m:MISSION)-[:CARRIES]->(i:INSTRUMENT) RETURN m.name, i.name',
        '3': 'MATCH (i:INSTRUMENT)-[:MEASURES]->(p:PRODUCT) RETURN i.name, p.name',
        '4': 'MATCH (m:MISSION)-[:OPERATES_IN]->(l:LOCATION) RETURN m.name, l.name',
        '5': 'MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC'
    }
    
    while True:
        try:
            user_input = input("\nCypher> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting interactive mode")
                break
            
            elif user_input.lower() == 'help':
                print("\nSample queries:")
                print("1. List all missions")
                print("2. Show mission-instrument relationships") 
                print("3. Show instrument-product relationships")
                print("4. Show mission-location relationships")
                print("5. Count entities by type")
                print("\nOr enter any Cypher query directly")
                continue
            
            elif user_input in sample_queries:
                query = sample_queries[user_input]
                print(f"Executing: {query}")
            else:
                query = user_input
            
            if query:
                results = kg_builder.query_graph(query)
                
                if results:
                    print(f"\nResults ({len(results)} rows):")
                    print("-" * 50)
                    
                    for i, result in enumerate(results):
                        if i >= 10:  # Limit display to 10 results
                            print(f"... and {len(results) - 10} more results")
                            break
                        print(result)
                else:
                    print("No results returned")
        
        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run the knowledge graph pipeline."""
    parser = argparse.ArgumentParser(
        description='MOSDAC Knowledge Graph Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Input crawler data file or directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='kg_output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--neo4j-uri',
        default='bolt://localhost:7687',
        help='Neo4j database URI'
    )
    
    parser.add_argument(
        '--neo4j-user',
        default='neo4j',
        help='Neo4j username'
    )
    
    parser.add_argument(
        '--neo4j-password',
        default='password',
        help='Neo4j password'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run with demonstration data'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enter interactive query mode after processing'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-file',
        help='Log file path'
    )
    
    parser.add_argument(
        '--skip-neo4j-test',
        action='store_true',
        help='Skip Neo4j connection test'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or f"{args.output}/pipeline.log"
    logger = setup_logging(args.log_level, log_file)
    
    logger.info("MOSDAC Knowledge Graph Pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check dependencies
    if not check_dependencies(logger):
        return 1
    
    # Test Neo4j connection
    if not args.skip_neo4j_test:
        if not test_neo4j_connection(args.neo4j_uri, args.neo4j_user, args.neo4j_password, logger):
            logger.error("Neo4j connection failed. Use --skip-neo4j-test to bypass this check.")
            return 1
    
    # Load input data
    if args.demo:
        logger.info("Using demonstration data")
        input_data = create_demo_data()
    elif args.input:
        input_data = load_crawler_data(args.input, logger)
        if input_data is None:
            return 1
    else:
        logger.error("No input specified. Use --input <file/dir> or --demo")
        return 1
    
    # Run pipeline
    results = run_knowledge_graph_pipeline(
        input_data=input_data,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        output_dir=args.output,
        logger=logger
    )
    
    if results['status'] == 'failed':
        logger.error("Pipeline failed")
        return 1
    
    # Interactive mode
    if args.interactive:
        try:
            with MOSDACKnowledgeGraphBuilder(
                neo4j_uri=args.neo4j_uri,
                neo4j_user=args.neo4j_user,
                neo4j_password=args.neo4j_password
            ) as kg_builder:
                interactive_query_mode(kg_builder, logger)
        except Exception as e:
            logger.error(f"Interactive mode failed: {e}")
    
    logger.info("Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())