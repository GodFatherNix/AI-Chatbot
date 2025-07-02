#!/usr/bin/env python3
"""Document processing pipeline for MOSDAC content.

Processes crawled content by:
1. Extracting text from HTML and documents
2. Chunking content into manageable pieces
3. Generating embeddings using sentence-transformers
4. Storing in Qdrant vector database
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Text processing
import re
from bs4 import BeautifulSoup
import tiktoken

# Document processing
import PyPDF2
from docx import Document as DocxDocument
import pandas as pd

# Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# Vector database
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, CollectionInfo
from qdrant_client.http.exceptions import ResponseHandlingException

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process MOSDAC documents for RAG pipeline."""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L12-v2",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "mosdac_embeddings"):
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Initialize vector database
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        
        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Processing statistics
        self.stats = {
            'processed_items': 0,
            'text_extracted': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'errors': 0
        }
        
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embed_dim,
                        distance=Distance.COSINE
                    )
                )
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to setup Qdrant collection: {e}")
            raise
    
    def process_crawled_data(self, data_file: str) -> Dict[str, int]:
        """Process crawled data from JSON file."""
        logger.info(f"Processing crawled data from: {data_file}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load crawled data
        with open(data_file, 'r', encoding='utf-8') as f:
            items = json.load(f)
        
        if not isinstance(items, list):
            items = [items]  # Handle single item
        
        logger.info(f"Loaded {len(items)} items for processing")
        
        # Process each item
        for item in items:
            try:
                self._process_single_item(item)
                self.stats['processed_items'] += 1
                
                if self.stats['processed_items'] % 10 == 0:
                    logger.info(f"Processed {self.stats['processed_items']} items...")
                    
            except Exception as e:
                logger.error(f"Error processing item {item.get('url', 'unknown')}: {e}")
                self.stats['errors'] += 1
                continue
        
        logger.info("Processing completed. Statistics:")
        for key, value in self.stats.items():
            logger.info(f"  {key}: {value}")
        
        return self.stats
    
    def _process_single_item(self, item: Dict[str, Any]):
        """Process a single crawled item."""
        url = item.get('url', '')
        content_type = item.get('content_type', 'unknown')
        
        # Extract text content
        if content_type == 'webpage':
            text = self._extract_text_from_html(item)
        elif content_type == 'document':
            text = self._extract_text_from_document(item)
        else:
            logger.warning(f"Unknown content type: {content_type} for {url}")
            return
        
        if not text or len(text.strip()) < 50:
            logger.warning(f"Insufficient text content for {url}")
            return
        
        self.stats['text_extracted'] += 1
        
        # Chunk the text
        chunks = self._chunk_text(text, item)
        self.stats['chunks_created'] += len(chunks)
        
        # Generate embeddings and store
        self._generate_and_store_embeddings(chunks, item)
    
    def _extract_text_from_html(self, item: Dict[str, Any]) -> str:
        """Extract clean text from HTML content."""
        try:
            content = item.get('content', '')
            if not content:
                return ''
            
            # If content is already clean text, return it
            if not content.strip().startswith('<'):
                return self._clean_text(content)
            
            # Parse HTML and extract text
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Extract text
            text = soup.get_text()
            
            return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ''
    
    def _extract_text_from_document(self, item: Dict[str, Any]) -> str:
        """Extract text from document files (PDF, DOCX, etc.)."""
        try:
            file_type = item.get('file_type', '').lower()
            content = item.get('content', b'')
            
            if isinstance(content, str):
                # Content is already text
                return self._clean_text(content)
            
            if file_type == 'pdf':
                return self._extract_pdf_text(content)
            elif file_type in ['doc', 'docx']:
                return self._extract_docx_text(content)
            elif file_type in ['xls', 'xlsx']:
                return self._extract_excel_text(content)
            else:
                # Try to decode as text
                try:
                    text = content.decode('utf-8', errors='ignore')
                    return self._clean_text(text)
                except:
                    logger.warning(f"Cannot extract text from file type: {file_type}")
                    return ''
                    
        except Exception as e:
            logger.error(f"Error extracting text from document: {e}")
            return ''
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            import io
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ''
    
    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX content."""
        try:
            import io
            doc_file = io.BytesIO(content)
            doc = DocxDocument(doc_file)
            
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ''
    
    def _extract_excel_text(self, content: bytes) -> str:
        """Extract text from Excel content."""
        try:
            import io
            excel_file = io.BytesIO(content)
            
            # Read all sheets
            excel_data = pd.read_excel(excel_file, sheet_name=None)
            
            text = ""
            for sheet_name, df in excel_data.items():
                text += f"Sheet: {sheet_name}\n"
                text += df.to_string(index=False) + "\n\n"
            
            return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting Excel text: {e}")
            return ''
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ''
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\:\;]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str, item: Dict[str, Any], 
                   chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces for embedding."""
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= chunk_size:
            # Text is small enough, return as single chunk
            return [{
                'text': text,
                'chunk_id': 0,
                'total_chunks': 1,
                'metadata': self._create_chunk_metadata(item, 0, 1)
            }]
        
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Skip very short chunks
            if len(chunk_text.strip()) < 50:
                continue
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': chunk_id,
                'total_chunks': 0,  # Will be updated after loop
                'metadata': self._create_chunk_metadata(item, chunk_id, 0)
            })
            
            chunk_id += 1
        
        # Update total chunk count
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
    
    def _create_chunk_metadata(self, item: Dict[str, Any], 
                              chunk_id: int, total_chunks: int) -> Dict[str, Any]:
        """Create metadata for a text chunk."""
        return {
            'source_url': item.get('url', ''),
            'source_title': item.get('title', ''),
            'content_type': item.get('content_type', ''),
            'file_type': item.get('file_type', ''),
            'chunk_id': chunk_id,
            'total_chunks': total_chunks,
            'crawled_at': item.get('crawled_at', ''),
            'processed_at': datetime.utcnow().isoformat(),
            'mission_info': item.get('mission_info', {}),
            'product_info': item.get('product_info', {})
        }
    
    def _generate_and_store_embeddings(self, chunks: List[Dict[str, Any]], 
                                     source_item: Dict[str, Any]):
        """Generate embeddings for chunks and store in Qdrant."""
        
        if not chunks:
            return
        
        try:
            # Generate embeddings for all chunks
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedder.encode(texts, normalize_embeddings=True)
            
            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = self._generate_point_id(source_item['url'], chunk['chunk_id'])
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        'text': chunk['text'],
                        'metadata': chunk['metadata']
                    }
                )
                points.append(point)
            
            # Store in Qdrant
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            self.stats['embeddings_generated'] += len(points)
            logger.debug(f"Stored {len(points)} embeddings for {source_item.get('url', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error generating/storing embeddings: {e}")
            raise
    
    def _generate_point_id(self, url: str, chunk_id: int) -> str:
        """Generate unique point ID for Qdrant."""
        # Create hash from URL and chunk ID
        content = f"{url}_{chunk_id}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(query, normalize_embeddings=True)
            
            # Search in Qdrant
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.id,
                    'score': result.score,
                    'text': result.payload['text'],
                    'metadata': result.payload['metadata']
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection_info = self.qdrant.get_collection(self.collection_name)
            return {
                'total_points': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.value
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}


def main():
    """Main function for testing the document processor."""
    
    # Example usage
    processor = DocumentProcessor()
    
    # Create sample data for testing
    sample_data = [
        {
            "url": "https://www.mosdac.gov.in/test",
            "title": "OCEANSAT-2 Mission Overview",
            "content": """
            OCEANSAT-2 is an Indian satellite designed for ocean studies. 
            The satellite carries Ocean Color Monitor (OCM) and Ku-band 
            pencil beam scatterometer for wind vector retrieval over oceans.
            
            Key specifications:
            - Launch: September 23, 2009
            - Orbit: Sun-synchronous polar orbit
            - Resolution: 360m for OCM
            - Coverage: Global oceans
            
            Products include:
            - Ocean Color
            - Sea Surface Temperature (SST)
            - Chlorophyll concentration
            - Suspended sediment concentration
            """,
            "content_type": "webpage",
            "mission_info": {"mission": "OCEANSAT-2"},
            "product_info": {"products": ["Ocean Color", "SST", "Chlorophyll"]}
        }
    ]
    
    # Save sample data
    with open('sample_data.json', 'w') as f:
        json.dump(sample_data, f)
    
    # Process the data
    stats = processor.process_crawled_data('sample_data.json')
    print("Processing completed!")
    print(f"Statistics: {stats}")
    
    # Test search
    results = processor.search_similar("ocean color satellite data", limit=3)
    print(f"\nSearch results for 'ocean color satellite data':")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Source: {result['metadata']['source_title']}")
    
    # Clean up
    os.remove('sample_data.json')


if __name__ == "__main__":
    main()