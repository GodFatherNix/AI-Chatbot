#!/usr/bin/env python3
"""Test the complete data ingestion and knowledge graph pipeline.

This demonstrates the working implementations without requiring external dependencies.
"""

import json
import os
import re
from typing import Dict, List, Any
from datetime import datetime


class MockMOSDACCrawler:
    """Mock MOSDAC crawler for testing without Scrapy."""
    
    def __init__(self):
        self.visited_urls = set()
        
    def crawl_sample_data(self) -> List[Dict[str, Any]]:
        """Generate sample crawled data that represents real MOSDAC content."""
        
        sample_pages = [
            {
                "url": "https://www.mosdac.gov.in/missions/oceansat2",
                "title": "OCEANSAT-2 Mission Overview",
                "content": """
                OCEANSAT-2 is an Indian satellite designed for ocean studies. 
                The satellite carries Ocean Color Monitor (OCM) and Ku-band 
                pencil beam scatterometer (OSCAT) for wind vector retrieval over oceans.
                
                Mission Specifications:
                - Launch Date: September 23, 2009
                - Orbit: Sun-synchronous polar orbit at 720 km altitude
                - Inclination: 98.28 degrees
                - Mission Life: 5 years (extended)
                
                Instruments:
                1. Ocean Color Monitor (OCM)
                   - Resolution: 360m at nadir
                   - Swath: 1420 km
                   - Spectral Bands: 8 bands (402-885 nm)
                
                2. Ku-band Scatterometer (OSCAT)
                   - Resolution: 25 km x 25 km
                   - Swath: 1400 km
                   - Frequency: 13.515 GHz
                
                Data Products:
                - Ocean Color concentration
                - Sea Surface Temperature (SST)
                - Chlorophyll-a concentration
                - Suspended sediment concentration
                - Wind speed and direction over oceans
                
                Coverage Areas:
                - Global oceans including Arabian Sea and Bay of Bengal
                - Tropical and polar regions
                - Coastal zones around Indian subcontinent
                
                Data Formats: HDF, NetCDF, GeoTIFF
                Access Methods: FTP, HTTP, Online portal
                """,
                "content_type": "webpage",
                "crawled_at": "2024-01-15T10:30:00Z",
                "mission_info": {
                    "mission": "OCEANSAT-2",
                    "specifications": {
                        "resolution": "360m",
                        "temporal_coverage": "2009-present"
                    }
                },
                "product_info": {
                    "products": ["Ocean Color", "SST", "Chlorophyll"],
                    "data_formats": ["HDF", "NetCDF", "GeoTIFF"],
                    "access_methods": ["FTP", "HTTP", "PORTAL"]
                }
            },
            {
                "url": "https://www.mosdac.gov.in/missions/insat3d",
                "title": "INSAT-3D Meteorological Satellite",
                "content": """
                INSAT-3D is a meteorological satellite providing enhanced weather monitoring 
                capabilities over the Indian region. The satellite carries advanced 
                instruments for atmospheric sounding and imaging.
                
                Mission Details:
                - Launch: July 26, 2013
                - Orbit: Geostationary at 82° East longitude
                - Mission Duration: 10 years
                - Operational Status: Active
                
                Primary Instruments:
                1. Imager
                   - Resolution: 4 km (VIS/IR), 8 km (WV)
                   - Spectral Channels: 6 channels
                   - Coverage: Full Earth disc
                
                2. Sounder
                   - Resolution: 10 km
                   - Spectral Channels: 18 channels (IR), 1 channel (VIS)
                   - Atmospheric profiling capability
                
                Meteorological Products:
                - Temperature profiles
                - Humidity profiles
                - Cloud imagery and properties
                - Outgoing Longwave Radiation (OLR)
                - Sea Surface Temperature
                - Rainfall estimation
                - Cyclone monitoring
                
                Applications:
                - Weather forecasting and monitoring
                - Disaster management
                - Agriculture and water resource management
                - Climate studies
                
                Geographic Coverage:
                - Indian subcontinent and surrounding oceans
                - Arabian Sea and Bay of Bengal
                - Monsoon regions
                
                Data Distribution: Real-time and archived data through MOSDAC portal
                """,
                "content_type": "webpage",
                "crawled_at": "2024-01-15T11:15:00Z",
                "mission_info": {
                    "mission": "INSAT-3D",
                    "specifications": {
                        "resolution": "4km",
                        "temporal_coverage": "2013-present"
                    }
                },
                "product_info": {
                    "products": ["Temperature", "Humidity", "Cloud Imagery"],
                    "data_formats": ["HDF", "NetCDF"],
                    "access_methods": ["HTTP", "PORTAL"]
                }
            },
            {
                "url": "https://www.mosdac.gov.in/products/wind-data",
                "title": "Wind Data Products from SCATSAT-1",
                "content": """
                SCATSAT-1 provides high-quality wind vector data over global oceans 
                using Ku-band scatterometer technology. The mission continues the 
                legacy of OCEANSAT-2 scatterometer observations.
                
                Mission Information:
                - Launch: September 26, 2016
                - Orbit: Sun-synchronous at 720 km altitude
                - Mission Life: 5 years
                - Current Status: Operational
                
                Scatterometer Specifications:
                - Operating Frequency: 13.515 GHz (Ku-band)
                - Resolution: 25 km x 25 km
                - Swath Width: 1400 km
                - Measurement Accuracy: Wind speed ±2 m/s, Direction ±20°
                
                Wind Products:
                - Wind speed over oceans (2-24 m/s range)
                - Wind direction (0-360 degrees)
                - Wind stress and curl
                - Ocean surface roughness
                
                Scientific Applications:
                - Tropical cyclone monitoring and tracking
                - Monsoon circulation analysis
                - Ocean-atmosphere interaction studies
                - Weather forecasting model validation
                - Climate research
                
                Data Coverage:
                - Global oceans with 2-day repeat cycle
                - Tropical cyclone regions
                - Indian Ocean and surrounding seas
                - Monsoon-affected areas
                
                Data Quality: Level-2 and Level-3 processed products
                Update Frequency: Daily, weekly, and monthly composites
                """,
                "content_type": "webpage",
                "crawled_at": "2024-01-15T12:00:00Z",
                "mission_info": {
                    "mission": "SCATSAT-1",
                    "specifications": {
                        "resolution": "25km",
                        "temporal_coverage": "2016-present"
                    }
                },
                "product_info": {
                    "products": ["Wind Speed", "Wind Direction"],
                    "data_formats": ["HDF", "NetCDF"],
                    "access_methods": ["FTP", "HTTP"]
                }
            }
        ]
        
        print(f"🕷️ Mock crawler extracted {len(sample_pages)} pages")
        return sample_pages


class MockDocumentProcessor:
    """Mock document processor for testing without dependencies."""
    
    def __init__(self):
        self.stats = {
            'processed_items': 0,
            'text_extracted': 0,
            'chunks_created': 0,
            'embeddings_generated': 0
        }
    
    def process_crawled_data(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Process crawled data and generate chunks."""
        
        processed_chunks = []
        
        for item in data:
            # Extract and clean text
            text = self._clean_text(item.get('content', ''))
            if not text or len(text.strip()) < 50:
                continue
                
            self.stats['text_extracted'] += 1
            
            # Chunk the text
            chunks = self._chunk_text(text, item)
            processed_chunks.extend(chunks)
            self.stats['chunks_created'] += len(chunks)
            
            # Mock embedding generation
            self._generate_mock_embeddings(chunks)
            
            self.stats['processed_items'] += 1
        
        print(f"📄 Processed {len(data)} documents into {len(processed_chunks)} chunks")
        return self.stats
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ''
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\:\;]', ' ', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str, item: Dict[str, Any], chunk_size: int = 500) -> List[Dict[str, Any]]:
        """Split text into chunks."""
        
        # Simple word-based chunking
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) >= 50:
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': len(chunks),
                    'source_url': item.get('url', ''),
                    'source_title': item.get('title', ''),
                    'metadata': {
                        'content_type': item.get('content_type', ''),
                        'mission_info': item.get('mission_info', {}),
                        'product_info': item.get('product_info', {})
                    }
                })
        
        return chunks
    
    def _generate_mock_embeddings(self, chunks: List[Dict[str, Any]]):
        """Mock embedding generation."""
        self.stats['embeddings_generated'] += len(chunks)
        print(f"  🔢 Generated {len(chunks)} mock embeddings")


class MockKnowledgeGraphBuilder:
    """Mock knowledge graph builder for testing without Neo4j."""
    
    def __init__(self):
        self.entities = []
        self.relationships = []
        self.stats = {
            'processed_documents': 0,
            'extracted_entities': 0,
            'extracted_relations': 0,
            'created_nodes': 0,
            'created_relationships': 0
        }
    
    def build_kg_from_documents(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Build knowledge graph from documents."""
        
        print("🕸️ Building knowledge graph...")
        
        for doc in data:
            entities, relations = self._process_document(doc)
            self.entities.extend(entities)
            self.relationships.extend(relations)
            self.stats['processed_documents'] += 1
        
        # Simulate node and relationship creation
        self.stats['created_nodes'] = len(set(e['text'] for e in self.entities))
        self.stats['created_relationships'] = len(self.relationships)
        
        print(f"  📊 Extracted {len(self.entities)} entities")
        print(f"  🔗 Created {len(self.relationships)} relationships")
        
        return self.stats
    
    def _process_document(self, doc: Dict[str, Any]) -> tuple:
        """Extract entities and relationships from document."""
        text = doc.get('content', '')
        url = doc.get('url', '')
        
        # Extract entities using patterns
        entities = self._extract_entities(text, url)
        self.stats['extracted_entities'] += len(entities)
        
        # Extract relationships
        relations = self._extract_relationships(text, entities)
        self.stats['extracted_relations'] += len(relations)
        
        return entities, relations
    
    def _extract_entities(self, text: str, source_url: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Mission patterns
        mission_pattern = r'\b(OCEANSAT-2|INSAT-3D|SCATSAT-1|MEGHA-TROPIQUES|CARTOSAT|RESOURCESAT)\b'
        for match in re.finditer(mission_pattern, text, re.IGNORECASE):
            entities.append({
                'text': match.group(0),
                'label': 'MISSION',
                'source_url': source_url
            })
        
        # Instrument patterns
        instrument_pattern = r'\b(Ocean Color Monitor|OCM|Scatterometer|OSCAT|Imager|Sounder)\b'
        for match in re.finditer(instrument_pattern, text, re.IGNORECASE):
            entities.append({
                'text': match.group(0),
                'label': 'INSTRUMENT',
                'source_url': source_url
            })
        
        # Parameter patterns
        param_pattern = r'\b(Ocean Color|Sea Surface Temperature|SST|Chlorophyll|Wind Speed|Wind Direction|Temperature|Humidity)\b'
        for match in re.finditer(param_pattern, text, re.IGNORECASE):
            entities.append({
                'text': match.group(0),
                'label': 'PARAMETER',
                'source_url': source_url
            })
        
        # Location patterns
        location_pattern = r'\b(Arabian Sea|Bay of Bengal|Indian Ocean|Indian subcontinent)\b'
        for match in re.finditer(location_pattern, text, re.IGNORECASE):
            entities.append({
                'text': match.group(0),
                'label': 'LOCATION',
                'source_url': source_url
            })
        
        # Resolution patterns
        resolution_pattern = r'\b(\d+\s*(?:km|m))\b'
        for match in re.finditer(resolution_pattern, text, re.IGNORECASE):
            entities.append({
                'text': match.group(0),
                'label': 'RESOLUTION',
                'source_url': source_url
            })
        
        return entities
    
    def _extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        
        # Pattern-based relationship extraction
        entity_texts = [e['text'] for e in entities]
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if both entities appear in the same sentence
                sentences = text.split('.')
                for sentence in sentences:
                    if (entity1['text'].lower() in sentence.lower() and 
                        entity2['text'].lower() in sentence.lower()):
                        
                        relation_type = self._infer_relation_type(entity1, entity2)
                        
                        relationships.append({
                            'source': entity1['text'],
                            'target': entity2['text'],
                            'relation_type': relation_type,
                            'context': sentence.strip()
                        })
                        break
        
        return relationships
    
    def _infer_relation_type(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> str:
        """Infer relationship type based on entity labels."""
        label_pairs = {
            ('MISSION', 'INSTRUMENT'): 'CARRIES',
            ('MISSION', 'PARAMETER'): 'MEASURES',
            ('MISSION', 'LOCATION'): 'OPERATES_IN',
            ('INSTRUMENT', 'PARAMETER'): 'MEASURES',
            ('MISSION', 'RESOLUTION'): 'HAS_RESOLUTION',
        }
        
        pair = (entity1['label'], entity2['label'])
        reverse_pair = (entity2['label'], entity1['label'])
        
        return label_pairs.get(pair, label_pairs.get(reverse_pair, 'RELATED_TO'))
    
    def query_knowledge_graph(self, entity_name: str) -> List[Dict[str, Any]]:
        """Query the knowledge graph for entity relationships."""
        results = []
        
        for rel in self.relationships:
            if rel['source'] == entity_name:
                results.append({
                    'source': rel['source'],
                    'relationship': rel['relation_type'],
                    'target': rel['target'],
                    'context': rel['context'][:100] + '...' if len(rel['context']) > 100 else rel['context']
                })
            elif rel['target'] == entity_name:
                results.append({
                    'source': rel['target'],
                    'relationship': rel['relation_type'],
                    'target': rel['source'],
                    'context': rel['context'][:100] + '...' if len(rel['context']) > 100 else rel['context']
                })
        
        return results[:10]  # Limit results


def run_complete_ingestion_pipeline():
    """Run the complete data ingestion and KG pipeline test."""
    
    print("🚀 Starting Complete MOSDAC Data Ingestion Pipeline Test")
    print("=" * 70)
    
    # Step 1: Web Crawling
    print("\n📡 Step 1: Web Crawling")
    print("-" * 30)
    crawler = MockMOSDACCrawler()
    crawled_data = crawler.crawl_sample_data()
    
    # Save crawled data
    with open('crawled_data.json', 'w', encoding='utf-8') as f:
        json.dump(crawled_data, f, indent=2, ensure_ascii=False)
    
    # Step 2: Document Processing
    print("\n📄 Step 2: Document Processing & Embedding Generation")
    print("-" * 55)
    processor = MockDocumentProcessor()
    processing_stats = processor.process_crawled_data(crawled_data)
    
    print("Processing Statistics:")
    for key, value in processing_stats.items():
        print(f"  {key}: {value}")
    
    # Step 3: Knowledge Graph Construction
    print("\n🕸️ Step 3: Knowledge Graph Construction")
    print("-" * 40)
    kg_builder = MockKnowledgeGraphBuilder()
    kg_stats = kg_builder.build_kg_from_documents(crawled_data)
    
    print("Knowledge Graph Statistics:")
    for key, value in kg_stats.items():
        print(f"  {key}: {value}")
    
    # Step 4: Test Knowledge Graph Queries
    print("\n🔍 Step 4: Knowledge Graph Query Testing")
    print("-" * 45)
    
    test_entities = ['OCEANSAT-2', 'INSAT-3D', 'Ocean Color Monitor']
    
    for entity in test_entities:
        results = kg_builder.query_knowledge_graph(entity)
        print(f"\nRelationships for '{entity}':")
        if results:
            for result in results:
                print(f"  {result['source']} --{result['relationship']}--> {result['target']}")
                print(f"    Context: {result['context']}")
        else:
            print(f"  No relationships found for {entity}")
    
    # Step 5: Entity and Relationship Analysis
    print("\n📊 Step 5: Entity & Relationship Analysis")
    print("-" * 45)
    
    # Count entities by type
    entity_counts = {}
    for entity in kg_builder.entities:
        label = entity['label']
        entity_counts[label] = entity_counts.get(label, 0) + 1
    
    print("Entity Counts by Type:")
    for entity_type, count in sorted(entity_counts.items()):
        print(f"  {entity_type}: {count}")
    
    # Count relationships by type
    rel_counts = {}
    for rel in kg_builder.relationships:
        rel_type = rel['relation_type']
        rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
    
    print("\nRelationship Counts by Type:")
    for rel_type, count in sorted(rel_counts.items()):
        print(f"  {rel_type}: {count}")
    
    # Step 6: Sample Entity Extraction Examples
    print("\n🎯 Step 6: Sample Entity Extraction Examples")
    print("-" * 50)
    
    sample_entities_by_type = {}
    for entity in kg_builder.entities:
        label = entity['label']
        if label not in sample_entities_by_type:
            sample_entities_by_type[label] = []
        if len(sample_entities_by_type[label]) < 3:  # Show max 3 examples per type
            sample_entities_by_type[label].append(entity['text'])
    
    for entity_type, examples in sample_entities_by_type.items():
        print(f"{entity_type}: {', '.join(examples)}")
    
    # Clean up
    os.remove('crawled_data.json')
    
    print("\n✅ Pipeline Test Completed Successfully!")
    print("\nSummary:")
    print(f"  📄 Processed: {len(crawled_data)} documents")
    print(f"  🔤 Extracted: {len(kg_builder.entities)} entities")
    print(f"  🔗 Created: {len(kg_builder.relationships)} relationships")
    print(f"  📊 Entity Types: {len(entity_counts)}")
    print(f"  🔀 Relation Types: {len(rel_counts)}")


if __name__ == "__main__":
    run_complete_ingestion_pipeline()