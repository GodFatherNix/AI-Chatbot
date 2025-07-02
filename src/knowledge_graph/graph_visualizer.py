"""
Knowledge graph visualization for MOSDAC content.
"""
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
import colorsys

from .graph_builder import GraphBuilder, GraphNode, GraphEdge
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

class GraphVisualizer:
    """
    Visualizes knowledge graphs using various plotting libraries.
    Supports static and interactive visualizations.
    """
    
    def __init__(self, graph_builder: GraphBuilder):
        self.graph_builder = graph_builder
        self.graph = graph_builder.graph
        
        # Color schemes for different entity types
        self.entity_colors = {
            'Satellite': '#FF6B6B',      # Red
            'Organization': '#4ECDC4',    # Teal
            'Technology': '#45B7D1',      # Blue
            'Mission': '#96CEB4',         # Green
            'Product': '#FECA57',         # Yellow
            'Service': '#FF9FF3',         # Pink
            'Location': '#54A0FF',        # Light Blue
            'Date': '#5F27CD'             # Purple
        }
        
        # Edge colors for different relation types
        self.relation_colors = {
            'operates': '#FF6B6B',
            'carries': '#4ECDC4',
            'produces': '#45B7D1',
            'observes': '#96CEB4',
            'supports': '#FECA57',
            'related_to': '#C7ECEE',
            'located_at': '#54A0FF',
            'launched_on': '#5F27CD'
        }
        
        # Layout algorithms
        self.layout_algorithms = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'shell': nx.shell_layout,
            'spectral': nx.spectral_layout
        }
    
    def _generate_colors(self, n_colors: int) -> List[str]:
        """Generate n distinct colors."""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            colors.append(hex_color)
        return colors
    
    def _get_node_color(self, node_id: str) -> str:
        """Get color for a node based on its entity type."""
        node = self.graph_builder.get_node_by_id(node_id)
        if node:
            return self.entity_colors.get(node.entity_type, '#CCCCCC')
        return '#CCCCCC'
    
    def _get_edge_color(self, edge_data: Dict) -> str:
        """Get color for an edge based on its relation type."""
        relation_type = edge_data.get('relation_type', 'unknown')
        return self.relation_colors.get(relation_type, '#999999')
    
    def _calculate_node_sizes(self, metric: str = 'degree') -> Dict[str, float]:
        """Calculate node sizes based on various metrics."""
        sizes = {}
        
        if metric == 'degree':
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            for node_id, degree in degrees.items():
                sizes[node_id] = 300 + (degree / max_degree) * 1000
        
        elif metric == 'confidence':
            for node_id in self.graph.nodes():
                node = self.graph_builder.get_node_by_id(node_id)
                if node:
                    sizes[node_id] = 300 + node.confidence * 700
                else:
                    sizes[node_id] = 300
        
        elif metric == 'centrality':
            try:
                centrality = nx.degree_centrality(self.graph)
                for node_id, cent in centrality.items():
                    sizes[node_id] = 300 + cent * 1000
            except:
                # Fallback to uniform size
                for node_id in self.graph.nodes():
                    sizes[node_id] = 500
        
        else:
            # Uniform size
            for node_id in self.graph.nodes():
                sizes[node_id] = 500
        
        return sizes
    
    def plot_static_graph(self, 
                         layout: str = 'spring',
                         node_size_metric: str = 'degree',
                         show_labels: bool = True,
                         figsize: Tuple[int, int] = (15, 10),
                         output_file: Optional[str] = None) -> None:
        """
        Create a static visualization of the knowledge graph.
        
        Args:
            layout: Layout algorithm to use
            node_size_metric: Metric for determining node sizes
            show_labels: Whether to show node labels
            figsize: Figure size
            output_file: Path to save the plot
        """
        if self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, cannot create visualization")
            return
        
        # Calculate layout
        if layout in self.layout_algorithms:
            try:
                pos = self.layout_algorithms[layout](self.graph, k=3, iterations=50)
            except:
                logger.warning(f"Layout {layout} failed, using spring layout")
                pos = nx.spring_layout(self.graph, k=3, iterations=50)
        else:
            pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Calculate node properties
        node_colors = [self._get_node_color(node) for node in self.graph.nodes()]
        node_sizes = self._calculate_node_sizes(node_size_metric)
        node_size_list = [node_sizes.get(node, 500) for node in self.graph.nodes()]
        
        # Calculate edge properties
        edge_colors = [self._get_edge_color(edge_data) for _, _, edge_data in self.graph.edges(data=True)]
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color=edge_colors,
            alpha=0.6,
            width=1.5,
            arrows=True,
            arrowsize=20,
            arrowstyle='->'
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=node_size_list,
            alpha=0.8,
            linewidths=2,
            edgecolors='black'
        )
        
        # Add labels if requested
        if show_labels:
            labels = {}
            for node_id in self.graph.nodes():
                node = self.graph_builder.get_node_by_id(node_id)
                if node:
                    # Truncate long labels
                    label = node.label[:20] + "..." if len(node.label) > 20 else node.label
                    labels[node_id] = label
                else:
                    labels[node_id] = node_id
            
            nx.draw_networkx_labels(
                self.graph, pos,
                labels=labels,
                font_size=8,
                font_weight='bold'
            )
        
        plt.title("MOSDAC Knowledge Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Add legend for entity types
        unique_types = list(set(node.entity_type for node in self.graph_builder.nodes_dict.values()))
        legend_elements = []
        for entity_type in unique_types:
            color = self.entity_colors.get(entity_type, '#CCCCCC')
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, label=entity_type))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved static graph visualization to {output_file}")
        
        plt.show()
    
    def plot_interactive_graph(self, 
                              layout: str = 'spring',
                              node_size_metric: str = 'degree',
                              output_file: Optional[str] = None) -> go.Figure:
        """
        Create an interactive visualization of the knowledge graph.
        
        Args:
            layout: Layout algorithm to use
            node_size_metric: Metric for determining node sizes
            output_file: Path to save the HTML file
            
        Returns:
            Plotly figure object
        """
        if self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, cannot create visualization")
            return go.Figure()
        
        # Calculate layout
        if layout in self.layout_algorithms:
            try:
                pos = self.layout_algorithms[layout](self.graph, k=3, iterations=50)
            except:
                logger.warning(f"Layout {layout} failed, using spring layout")
                pos = nx.spring_layout(self.graph, k=3, iterations=50)
        else:
            pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        node_info = []
        
        node_size_dict = self._calculate_node_sizes(node_size_metric)
        
        for node_id in self.graph.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            node = self.graph_builder.get_node_by_id(node_id)
            if node:
                node_text.append(node.label)
                node_colors.append(self._get_node_color(node_id))
                node_sizes.append(node_size_dict.get(node_id, 500) / 20)  # Scale for Plotly
                
                # Hover information
                hover_text = f"<b>{node.label}</b><br>"
                hover_text += f"Type: {node.entity_type}<br>"
                hover_text += f"Confidence: {node.confidence:.2f}<br>"
                hover_text += f"Sources: {len(node.source_documents)}"
                node_info.append(hover_text)
            else:
                node_text.append(node_id)
                node_colors.append('#CCCCCC')
                node_sizes.append(25)
                node_info.append(f"Node: {node_id}")
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black'),
                opacity=0.8
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='MOSDAC Knowledge Graph (Interactive)',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive Knowledge Graph - Hover over nodes for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="black", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=1200,
                           height=800
                       ))
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved interactive graph visualization to {output_file}")
        
        return fig
    
    def plot_subgraph(self, 
                     entity_name: str,
                     max_hops: int = 2,
                     layout: str = 'spring',
                     output_file: Optional[str] = None) -> go.Figure:
        """
        Plot a subgraph centered around a specific entity.
        
        Args:
            entity_name: Name of the central entity
            max_hops: Maximum number of hops from central entity
            layout: Layout algorithm
            output_file: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        # Find the entity
        nodes = self.graph_builder.find_nodes_by_text(entity_name, fuzzy=True)
        if not nodes:
            logger.warning(f"Entity '{entity_name}' not found")
            return go.Figure()
        
        central_node = nodes[0]  # Take the first match
        
        # Get subgraph nodes within max_hops
        subgraph_nodes = {central_node.id}
        current_level = {central_node.id}
        
        for hop in range(max_hops):
            next_level = set()
            for node_id in current_level:
                neighbors = self.graph_builder.get_neighbors(node_id)
                next_level.update(neighbor.id for neighbor in neighbors)
            
            subgraph_nodes.update(next_level)
            current_level = next_level
        
        # Create subgraph
        subgraph_builder = self.graph_builder.get_subgraph(list(subgraph_nodes))
        
        # Create visualizer for subgraph
        subgraph_visualizer = GraphVisualizer(subgraph_builder)
        
        # Plot the subgraph
        fig = subgraph_visualizer.plot_interactive_graph(layout=layout, output_file=output_file)
        fig.update_layout(title=f'Subgraph around "{central_node.label}"')
        
        return fig
    
    def plot_entity_type_distribution(self, output_file: Optional[str] = None) -> go.Figure:
        """
        Plot distribution of entity types.
        
        Args:
            output_file: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Count entity types
        type_counts = {}
        for node in self.graph_builder.nodes_dict.values():
            type_counts[node.entity_type] = type_counts.get(node.entity_type, 0) + 1
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                marker_color=[self.entity_colors.get(t, '#CCCCCC') for t in type_counts.keys()]
            )
        ])
        
        fig.update_layout(
            title='Distribution of Entity Types',
            xaxis_title='Entity Type',
            yaxis_title='Count',
            showlegend=False
        )
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved entity type distribution to {output_file}")
        
        return fig
    
    def plot_relation_type_distribution(self, output_file: Optional[str] = None) -> go.Figure:
        """
        Plot distribution of relation types.
        
        Args:
            output_file: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Count relation types
        type_counts = {}
        for edge in self.graph_builder.edges_dict.values():
            type_counts[edge.relation_type] = type_counts.get(edge.relation_type, 0) + 1
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                marker_color=[self.relation_colors.get(t, '#999999') for t in type_counts.keys()]
            )
        ])
        
        fig.update_layout(
            title='Distribution of Relation Types',
            xaxis_title='Relation Type',
            yaxis_title='Count',
            showlegend=False
        )
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved relation type distribution to {output_file}")
        
        return fig
    
    def create_dashboard(self, output_file: str = "graph_dashboard.html") -> None:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            output_file: Path to save the dashboard HTML
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Entity Type Distribution', 'Relation Type Distribution', 
                          'Node Degree Distribution', 'Graph Statistics'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "table"}]]
        )
        
        # Entity type distribution
        type_counts = {}
        for node in self.graph_builder.nodes_dict.values():
            type_counts[node.entity_type] = type_counts.get(node.entity_type, 0) + 1
        
        fig.add_trace(
            go.Bar(x=list(type_counts.keys()), y=list(type_counts.values()), name="Entity Types"),
            row=1, col=1
        )
        
        # Relation type distribution
        rel_counts = {}
        for edge in self.graph_builder.edges_dict.values():
            rel_counts[edge.relation_type] = rel_counts.get(edge.relation_type, 0) + 1
        
        fig.add_trace(
            go.Bar(x=list(rel_counts.keys()), y=list(rel_counts.values()), name="Relation Types"),
            row=1, col=2
        )
        
        # Node degree distribution
        degrees = [d for n, d in self.graph.degree()]
        fig.add_trace(
            go.Histogram(x=degrees, name="Degree Distribution"),
            row=2, col=1
        )
        
        # Graph statistics table
        stats = self.graph_builder.get_graph_statistics()
        table_data = [
            ["Total Nodes", stats['total_nodes']],
            ["Total Edges", stats['total_edges']],
            ["Graph Density", f"{stats.get('graph_metrics', {}).get('density', 0):.3f}"],
            ["Connected Components", stats.get('graph_metrics', {}).get('number_of_weakly_connected_components', 0)]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*table_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="MOSDAC Knowledge Graph Dashboard",
            showlegend=False,
            height=800
        )
        
        fig.write_html(output_file)
        logger.info(f"Saved dashboard to {output_file}")
    
    def export_for_gephi(self, output_file: str) -> None:
        """
        Export graph in Gephi-compatible format.
        
        Args:
            output_file: Path to save the GEXF file
        """
        # Add node attributes for Gephi
        for node_id in self.graph.nodes():
            node = self.graph_builder.get_node_by_id(node_id)
            if node:
                self.graph.nodes[node_id]['label'] = node.label
                self.graph.nodes[node_id]['entity_type'] = node.entity_type
                self.graph.nodes[node_id]['confidence'] = node.confidence
        
        # Add edge attributes for Gephi
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge = self.graph_builder.get_edge_by_id(key)
            if edge:
                data['relation_type'] = edge.relation_type
                data['confidence'] = edge.confidence
        
        # Write GEXF file
        nx.write_gexf(self.graph, output_file)
        logger.info(f"Exported graph for Gephi to {output_file}")