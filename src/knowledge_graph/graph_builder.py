"""
Knowledge graph builder for MOSDAC content.
"""
import networkx as nx
import json
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
from pathlib import Path

from .entity_extractor import Entity
from .relation_extractor import Relation
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    entity_type: str
    properties: Dict[str, Any]
    confidence: float
    source_documents: List[str]

@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]
    confidence: float
    context: str

class GraphBuilder:
    """
    Builds and manages the MOSDAC knowledge graph.
    Supports multiple graph representations and query operations.
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed graph allowing multiple edges
        self.nodes_dict = {}  # Node ID -> GraphNode mapping
        self.edges_dict = {}  # Edge ID -> GraphEdge mapping
        
        # Statistics
        self.stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'node_types': {},
            'edge_types': {},
            'creation_timestamp': None,
            'last_updated': None
        }
        
        # Entity type hierarchy for reasoning
        self.entity_hierarchy = {
            'Satellite': {'Technology', 'Mission', 'Product'},
            'Organization': {'Satellite', 'Mission'},
            'Technology': {'Product'},
            'Mission': {'Product'},
            'Location': {'Product', 'Service'},
            'Date': {'Satellite', 'Mission', 'Product'}
        }
    
    def add_entity_as_node(self, entity: Entity) -> GraphNode:
        """
        Add an entity as a node to the knowledge graph.
        
        Args:
            entity: Entity to add as node
            
        Returns:
            GraphNode representation
        """
        # Check if node already exists
        if entity.id in self.nodes_dict:
            existing_node = self.nodes_dict[entity.id]
            # Update source documents
            if entity.source_document not in existing_node.source_documents:
                existing_node.source_documents.append(entity.source_document)
            return existing_node
        
        # Create new node
        properties = {
            'text': entity.text,
            'canonical_form': entity.canonical_form,
            'synonyms': entity.synonyms,
            'start_pos': entity.start_pos,
            'end_pos': entity.end_pos,
            'attributes': entity.attributes
        }
        
        node = GraphNode(
            id=entity.id,
            label=entity.canonical_form,
            entity_type=entity.label,
            properties=properties,
            confidence=entity.confidence,
            source_documents=[entity.source_document]
        )
        
        # Add to NetworkX graph
        self.graph.add_node(
            entity.id,
            label=entity.canonical_form,
            entity_type=entity.label,
            confidence=entity.confidence,
            **properties
        )
        
        # Store in our dictionary
        self.nodes_dict[entity.id] = node
        
        # Update statistics
        self.stats['total_nodes'] += 1
        entity_type = entity.label
        self.stats['node_types'][entity_type] = self.stats['node_types'].get(entity_type, 0) + 1
        
        return node
    
    def add_relation_as_edge(self, relation: Relation) -> GraphEdge:
        """
        Add a relation as an edge to the knowledge graph.
        
        Args:
            relation: Relation to add as edge
            
        Returns:
            GraphEdge representation
        """
        # Ensure source and target nodes exist
        source_node = self.add_entity_as_node(relation.subject_entity)
        target_node = self.add_entity_as_node(relation.object_entity)
        
        # Check if edge already exists
        if relation.id in self.edges_dict:
            return self.edges_dict[relation.id]
        
        # Create new edge
        properties = {
            'source_sentence': relation.source_sentence,
            'attributes': relation.attributes
        }
        
        edge = GraphEdge(
            id=relation.id,
            source_id=relation.subject_entity.id,
            target_id=relation.object_entity.id,
            relation_type=relation.predicate,
            properties=properties,
            confidence=relation.confidence,
            context=relation.context
        )
        
        # Add to NetworkX graph
        self.graph.add_edge(
            relation.subject_entity.id,
            relation.object_entity.id,
            key=relation.id,
            relation_type=relation.predicate,
            confidence=relation.confidence,
            context=relation.context,
            **properties
        )
        
        # Store in our dictionary
        self.edges_dict[relation.id] = edge
        
        # Update statistics
        self.stats['total_edges'] += 1
        relation_type = relation.predicate
        self.stats['edge_types'][relation_type] = self.stats['edge_types'].get(relation_type, 0) + 1
        
        return edge
    
    def build_graph_from_extractions(self, 
                                   entities_by_doc: Dict[str, List[Entity]], 
                                   relations_by_doc: Dict[str, List[Relation]]) -> None:
        """
        Build knowledge graph from extracted entities and relations.
        
        Args:
            entities_by_doc: Dictionary mapping document IDs to entity lists
            relations_by_doc: Dictionary mapping document IDs to relation lists
        """
        logger.info("Building knowledge graph from extractions...")
        
        # Add all entities as nodes
        total_entities = 0
        for doc_id, entities in entities_by_doc.items():
            for entity in entities:
                self.add_entity_as_node(entity)
                total_entities += 1
        
        # Add all relations as edges
        total_relations = 0
        for doc_id, relations in relations_by_doc.items():
            for relation in relations:
                self.add_relation_as_edge(relation)
                total_relations += 1
        
        # Update timestamps
        self.stats['creation_timestamp'] = datetime.now().isoformat()
        self.stats['last_updated'] = datetime.now().isoformat()
        
        logger.info(f"Built knowledge graph with {total_entities} entities and {total_relations} relations")
        logger.info(f"Graph contains {self.stats['total_nodes']} nodes and {self.stats['total_edges']} edges")
    
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        return self.nodes_dict.get(node_id)
    
    def get_edge_by_id(self, edge_id: str) -> Optional[GraphEdge]:
        """Get edge by ID."""
        return self.edges_dict.get(edge_id)
    
    def get_nodes_by_type(self, entity_type: str) -> List[GraphNode]:
        """Get all nodes of a specific entity type."""
        return [node for node in self.nodes_dict.values() if node.entity_type == entity_type]
    
    def get_edges_by_type(self, relation_type: str) -> List[GraphEdge]:
        """Get all edges of a specific relation type."""
        return [edge for edge in self.edges_dict.values() if edge.relation_type == relation_type]
    
    def get_neighbors(self, node_id: str, direction: str = 'both') -> List[GraphNode]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: ID of the central node
            direction: 'in', 'out', or 'both'
            
        Returns:
            List of neighboring nodes
        """
        if node_id not in self.graph:
            return []
        
        neighbors = set()
        
        if direction in ['out', 'both']:
            neighbors.update(self.graph.successors(node_id))
        
        if direction in ['in', 'both']:
            neighbors.update(self.graph.predecessors(node_id))
        
        return [self.nodes_dict[neighbor_id] for neighbor_id in neighbors if neighbor_id in self.nodes_dict]
    
    def find_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        try:
            # Convert to undirected for path finding
            undirected_graph = self.graph.to_undirected()
            path = nx.shortest_path(undirected_graph, source_id, target_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def find_connected_components(self) -> List[List[str]]:
        """Find connected components in the graph."""
        undirected_graph = self.graph.to_undirected()
        components = list(nx.connected_components(undirected_graph))
        return [list(component) for component in components]
    
    def get_node_degree(self, node_id: str) -> Dict[str, int]:
        """Get degree information for a node."""
        if node_id not in self.graph:
            return {'in_degree': 0, 'out_degree': 0, 'total_degree': 0}
        
        in_degree = self.graph.in_degree(node_id)
        out_degree = self.graph.out_degree(node_id)
        
        return {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'total_degree': in_degree + out_degree
        }
    
    def find_nodes_by_text(self, search_text: str, fuzzy: bool = False) -> List[GraphNode]:
        """
        Find nodes by text content.
        
        Args:
            search_text: Text to search for
            fuzzy: Whether to use fuzzy matching
            
        Returns:
            List of matching nodes
        """
        matching_nodes = []
        search_lower = search_text.lower()
        
        for node in self.nodes_dict.values():
            if fuzzy:
                # Simple fuzzy matching - check if search text is contained
                if (search_lower in node.label.lower() or 
                    search_lower in node.properties.get('text', '').lower() or
                    any(search_lower in syn.lower() for syn in node.properties.get('synonyms', []))):
                    matching_nodes.append(node)
            else:
                # Exact matching
                if (search_lower == node.label.lower() or 
                    search_lower == node.properties.get('text', '').lower() or
                    search_lower in [syn.lower() for syn in node.properties.get('synonyms', [])]):
                    matching_nodes.append(node)
        
        return matching_nodes
    
    def get_subgraph(self, node_ids: List[str], include_neighbors: bool = False) -> 'GraphBuilder':
        """
        Extract a subgraph containing specified nodes.
        
        Args:
            node_ids: List of node IDs to include
            include_neighbors: Whether to include direct neighbors
            
        Returns:
            New GraphBuilder instance with subgraph
        """
        subgraph_nodes = set(node_ids)
        
        if include_neighbors:
            for node_id in node_ids:
                neighbors = self.get_neighbors(node_id)
                subgraph_nodes.update(neighbor.id for neighbor in neighbors)
        
        # Create new graph builder
        subgraph_builder = GraphBuilder()
        
        # Add nodes
        for node_id in subgraph_nodes:
            if node_id in self.nodes_dict:
                node = self.nodes_dict[node_id]
                subgraph_builder.nodes_dict[node_id] = node
                subgraph_builder.graph.add_node(node_id, **node.properties)
        
        # Add edges between included nodes
        for edge in self.edges_dict.values():
            if edge.source_id in subgraph_nodes and edge.target_id in subgraph_nodes:
                subgraph_builder.edges_dict[edge.id] = edge
                subgraph_builder.graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    key=edge.id,
                    **edge.properties
                )
        
        # Update statistics
        subgraph_builder.stats['total_nodes'] = len(subgraph_builder.nodes_dict)
        subgraph_builder.stats['total_edges'] = len(subgraph_builder.edges_dict)
        
        return subgraph_builder
    
    def calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures for nodes."""
        centrality_measures = {}
        
        try:
            # Convert to undirected for some measures
            undirected_graph = self.graph.to_undirected()
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(undirected_graph)
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(undirected_graph)
            
            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(undirected_graph)
            
            # PageRank (for directed graph)
            pagerank = nx.pagerank(self.graph)
            
            # Combine measures
            for node_id in self.nodes_dict.keys():
                centrality_measures[node_id] = {
                    'degree_centrality': degree_centrality.get(node_id, 0),
                    'betweenness_centrality': betweenness_centrality.get(node_id, 0),
                    'closeness_centrality': closeness_centrality.get(node_id, 0),
                    'pagerank': pagerank.get(node_id, 0)
                }
        
        except Exception as e:
            logger.error(f"Error calculating centrality measures: {e}")
            return {}
        
        return centrality_measures
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        stats = self.stats.copy()
        
        if self.graph.number_of_nodes() > 0:
            # Graph structure metrics
            stats['graph_metrics'] = {
                'density': nx.density(self.graph),
                'number_of_strongly_connected_components': nx.number_strongly_connected_components(self.graph),
                'number_of_weakly_connected_components': nx.number_weakly_connected_components(self.graph),
                'is_directed_acyclic': nx.is_directed_acyclic_graph(self.graph)
            }
            
            # Degree statistics
            degrees = [d for n, d in self.graph.degree()]
            if degrees:
                stats['degree_statistics'] = {
                    'average_degree': sum(degrees) / len(degrees),
                    'max_degree': max(degrees),
                    'min_degree': min(degrees)
                }
        
        return stats
    
    def save_graph(self, output_dir: str = "data/knowledge_graph"):
        """Save the knowledge graph to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save NetworkX graph
        nx_file = output_path / f"knowledge_graph_{timestamp}.gexf"
        nx.write_gexf(self.graph, nx_file)
        
        # Save nodes
        nodes_file = output_path / f"nodes_{timestamp}.json"
        nodes_data = [asdict(node) for node in self.nodes_dict.values()]
        with open(nodes_file, 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, indent=2, ensure_ascii=False)
        
        # Save edges
        edges_file = output_path / f"edges_{timestamp}.json"
        edges_data = [asdict(edge) for edge in self.edges_dict.values()]
        with open(edges_file, 'w', encoding='utf-8') as f:
            json.dump(edges_data, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_file = output_path / f"graph_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_graph_statistics(), f, indent=2, ensure_ascii=False)
        
        # Save as pickle for quick loading
        pickle_file = output_path / f"graph_builder_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Saved knowledge graph to {output_path}")
        return {
            'networkx_file': str(nx_file),
            'nodes_file': str(nodes_file),
            'edges_file': str(edges_file),
            'stats_file': str(stats_file),
            'pickle_file': str(pickle_file)
        }
    
    def load_graph(self, pickle_file: str):
        """Load knowledge graph from pickle file."""
        try:
            with open(pickle_file, 'rb') as f:
                loaded_builder = pickle.load(f)
            
            # Copy data to current instance
            self.graph = loaded_builder.graph
            self.nodes_dict = loaded_builder.nodes_dict
            self.edges_dict = loaded_builder.edges_dict
            self.stats = loaded_builder.stats
            
            logger.info(f"Loaded knowledge graph from {pickle_file}")
            logger.info(f"Graph contains {self.stats['total_nodes']} nodes and {self.stats['total_edges']} edges")
            
        except Exception as e:
            logger.error(f"Error loading graph from {pickle_file}: {e}")
            raise
    
    def query_graph(self, query_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute various types of queries on the knowledge graph.
        
        Args:
            query_type: Type of query ('find_entity', 'find_relations', 'path_query', etc.)
            **kwargs: Query parameters
            
        Returns:
            List of query results
        """
        results = []
        
        if query_type == 'find_entity':
            entity_name = kwargs.get('entity_name', '')
            entity_type = kwargs.get('entity_type', None)
            
            nodes = self.find_nodes_by_text(entity_name, fuzzy=True)
            if entity_type:
                nodes = [node for node in nodes if node.entity_type == entity_type]
            
            results = [asdict(node) for node in nodes]
        
        elif query_type == 'find_relations':
            subject_id = kwargs.get('subject_id')
            object_id = kwargs.get('object_id')
            relation_type = kwargs.get('relation_type')
            
            edges = list(self.edges_dict.values())
            
            if subject_id:
                edges = [edge for edge in edges if edge.source_id == subject_id]
            if object_id:
                edges = [edge for edge in edges if edge.target_id == object_id]
            if relation_type:
                edges = [edge for edge in edges if edge.relation_type == relation_type]
            
            results = [asdict(edge) for edge in edges]
        
        elif query_type == 'path_query':
            source_name = kwargs.get('source_name', '')
            target_name = kwargs.get('target_name', '')
            
            source_nodes = self.find_nodes_by_text(source_name)
            target_nodes = self.find_nodes_by_text(target_name)
            
            for source_node in source_nodes:
                for target_node in target_nodes:
                    path = self.find_shortest_path(source_node.id, target_node.id)
                    if path:
                        path_info = {
                            'source': asdict(source_node),
                            'target': asdict(target_node),
                            'path_nodes': [self.nodes_dict[node_id].label for node_id in path],
                            'path_length': len(path) - 1
                        }
                        results.append(path_info)
        
        elif query_type == 'neighborhood_query':
            entity_name = kwargs.get('entity_name', '')
            max_hops = kwargs.get('max_hops', 1)
            
            nodes = self.find_nodes_by_text(entity_name)
            for node in nodes:
                neighbors = self.get_neighbors(node.id)
                neighborhood_info = {
                    'central_node': asdict(node),
                    'neighbors': [asdict(neighbor) for neighbor in neighbors],
                    'neighbor_count': len(neighbors)
                }
                results.append(neighborhood_info)
        
        return results