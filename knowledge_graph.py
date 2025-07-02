import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple
import re
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentExtractor:
    """Extract and clean content from web pages"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_page_content(self, url: str) -> Optional[Dict]:
        """Extract structured content from a web page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            content = {
                'url': url,
                'title': self._extract_title(soup),
                'headings': self._extract_headings(soup),
                'paragraphs': self._extract_paragraphs(soup),
                'links': self._extract_links(soup, url),
                'metadata': self._extract_metadata(soup),
                'content_hash': hash(response.text)
            }
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else ""
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all headings with their levels"""
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                headings.append({
                    'level': level,
                    'text': heading.get_text().strip(),
                    'id': heading.get('id', '')
                })
        return headings
    
    def _extract_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """Extract paragraph content"""
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if len(text) > 20:  # Filter out very short paragraphs
                paragraphs.append(text)
        return paragraphs
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract internal and external links"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            links.append({
                'text': link.get_text().strip(),
                'url': absolute_url,
                'is_internal': self._is_internal_link(absolute_url, base_url)
            })
        return links
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract meta tags and other metadata"""
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        return metadata
    
    def _is_internal_link(self, url: str, base_url: str) -> bool:
        """Check if a link is internal to the same domain"""
        try:
            return urlparse(url).netloc == urlparse(base_url).netloc
        except:
            return False

class KnowledgeGraphBuilder:
    """Build and manage knowledge graphs from extracted content"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.content_extractor = ContentExtractor()
        self.processed_urls: Set[str] = set()
    
    def build_from_urls(self, start_urls: List[str], max_depth: int = 2, max_pages: int = 100) -> nx.DiGraph:
        """Build knowledge graph by crawling from start URLs"""
        logger.info(f"Building knowledge graph from {len(start_urls)} start URLs")
        
        to_process = [(url, 0) for url in start_urls]
        processed_count = 0
        
        while to_process and processed_count < max_pages:
            url, depth = to_process.pop(0)
            
            if url in self.processed_urls or depth > max_depth:
                continue
            
            logger.info(f"Processing URL: {url} (depth: {depth})")
            content = self.content_extractor.extract_page_content(url)
            
            if content:
                self._add_content_to_graph(content)
                self.processed_urls.add(url)
                processed_count += 1
                
                # Add internal links for further processing
                if depth < max_depth:
                    for link in content['links']:
                        if link['is_internal'] and link['url'] not in self.processed_urls:
                            to_process.append((link['url'], depth + 1))
            
            # Respect rate limiting
            time.sleep(config.WEB_SCRAPING_DELAY)
        
        logger.info(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _add_content_to_graph(self, content: Dict):
        """Add extracted content to the knowledge graph"""
        url = content['url']
        
        # Add main page node
        self.graph.add_node(url, **{
            'type': 'page',
            'title': content['title'],
            'content_hash': content['content_hash']
        })
        
        # Add heading nodes and relationships
        for heading in content['headings']:
            heading_id = f"{url}#heading_{hash(heading['text'])}"
            self.graph.add_node(heading_id, **{
                'type': 'heading',
                'level': heading['level'],
                'text': heading['text']
            })
            self.graph.add_edge(url, heading_id, relation='contains_heading')
        
        # Add paragraph nodes
        for i, paragraph in enumerate(content['paragraphs']):
            if len(paragraph) > config.MAX_CONTENT_LENGTH:
                paragraph = paragraph[:config.MAX_CONTENT_LENGTH] + "..."
            
            para_id = f"{url}#paragraph_{i}"
            self.graph.add_node(para_id, **{
                'type': 'paragraph',
                'text': paragraph,
                'position': i
            })
            self.graph.add_edge(url, para_id, relation='contains_paragraph')
        
        # Add link relationships
        for link in content['links']:
            if link['is_internal']:
                self.graph.add_edge(url, link['url'], **{
                    'relation': 'links_to',
                    'link_text': link['text']
                })
    
    def get_content_for_embedding(self) -> List[Dict]:
        """Get structured content for embedding generation"""
        content_items = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data['type'] in ['heading', 'paragraph']:
                content_items.append({
                    'id': node_id,
                    'text': node_data['text'],
                    'type': node_data['type'],
                    'metadata': {k: v for k, v in node_data.items() if k != 'text'}
                })
        
        return content_items
    
    def save_to_file(self, filepath: str):
        """Save knowledge graph to JSON file"""
        graph_data = {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True))
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge graph saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load knowledge graph from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            self.graph = nx.DiGraph()
            
            # Add nodes
            for node_id, node_data in graph_data['nodes'].items():
                self.graph.add_node(node_id, **node_data)
            
            # Add edges
            for edge in graph_data['edges']:
                self.graph.add_edge(edge[0], edge[1], **edge[2])
            
            logger.info(f"Knowledge graph loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
    
    def get_related_content(self, node_id: str, max_hops: int = 2) -> List[Dict]:
        """Get content related to a specific node within max_hops"""
        if node_id not in self.graph:
            return []
        
        related_nodes = set()
        current_level = {node_id}
        
        for hop in range(max_hops):
            next_level = set()
            for node in current_level:
                # Get neighbors (both incoming and outgoing)
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                next_level.update(neighbors)
            
            related_nodes.update(next_level)
            current_level = next_level
        
        # Get content for related nodes
        related_content = []
        for node in related_nodes:
            node_data = self.graph.nodes[node]
            if node_data['type'] in ['heading', 'paragraph']:
                related_content.append({
                    'id': node,
                    'text': node_data['text'],
                    'type': node_data['type'],
                    'metadata': {k: v for k, v in node_data.items() if k != 'text'}
                })
        
        return related_content