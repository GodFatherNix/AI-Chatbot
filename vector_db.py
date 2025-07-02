import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import logging
from config import config
import os

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Vector database interface for semantic search using ChromaDB"""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector database with embeddings"""
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector database")
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []
        
        for doc in documents:
            # Generate embedding for the text
            text = doc['text']
            embedding = self.embedding_model.encode(text).tolist()
            
            ids.append(doc['id'])
            embeddings.append(embedding)
            documents_text.append(text)
            metadatas.append({
                'type': doc['type'],
                **doc.get('metadata', {})
            })
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents_text,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents using semantic similarity"""
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Prepare where clause for filtering
        where_clause = filter_metadata if filter_metadata else None
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )
        
        # Format results
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                search_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        logger.info(f"Found {len(search_results)} relevant documents")
        return search_results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a specific document by its ID"""
        try:
            result = self.collection.get(ids=[doc_id])
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'text': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
        return None
    
    def update_document(self, doc_id: str, text: str, metadata: Dict):
        """Update an existing document"""
        try:
            embedding = self.embedding_model.encode(text).tolist()
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )
            logger.info(f"Updated document: {doc_id}")
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
    
    def delete_document(self, doc_id: str):
        """Delete a document from the vector database"""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'embedding_model': config.EMBEDDING_MODEL
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def hybrid_search(self, query: str, keywords: List[str] = None, top_k: int = 10) -> List[Dict]:
        """Perform hybrid search combining semantic and keyword matching"""
        # Start with semantic search
        semantic_results = self.search(query, top_k * 2)  # Get more results initially
        
        if not keywords:
            return semantic_results[:top_k]
        
        # Apply keyword filtering
        keyword_filtered = []
        for result in semantic_results:
            text_lower = result['text'].lower()
            keyword_match_count = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            
            if keyword_match_count > 0:
                result['keyword_matches'] = keyword_match_count
                keyword_filtered.append(result)
        
        # Sort by combination of semantic similarity and keyword matches
        if keyword_filtered:
            keyword_filtered.sort(
                key=lambda x: (x['keyword_matches'], -x['distance']), 
                reverse=True
            )
            return keyword_filtered[:top_k]
        
        # Fallback to semantic results if no keyword matches
        return semantic_results[:top_k]

class EmbeddingManager:
    """Manage embeddings and vector operations"""
    
    def __init__(self):
        self.vector_db = VectorDatabase()
    
    def build_embeddings_from_knowledge_graph(self, knowledge_graph_builder):
        """Build vector embeddings from knowledge graph content"""
        logger.info("Building embeddings from knowledge graph")
        
        # Clear existing embeddings
        self.vector_db.clear_collection()
        
        # Get content from knowledge graph
        content_items = knowledge_graph_builder.get_content_for_embedding()
        
        if not content_items:
            logger.warning("No content found in knowledge graph")
            return
        
        # Chunk large content if necessary
        chunked_content = self._chunk_content(content_items)
        
        # Add to vector database
        self.vector_db.add_documents(chunked_content)
        
        logger.info(f"Built embeddings for {len(chunked_content)} content chunks")
    
    def _chunk_content(self, content_items: List[Dict]) -> List[Dict]:
        """Chunk large content into smaller pieces for better embedding quality"""
        chunked_items = []
        
        for item in content_items:
            text = item['text']
            
            if len(text) <= config.CHUNK_SIZE:
                chunked_items.append(item)
            else:
                # Split into chunks
                chunks = self._split_text_into_chunks(text)
                for i, chunk in enumerate(chunks):
                    chunked_item = {
                        'id': f"{item['id']}_chunk_{i}",
                        'text': chunk,
                        'type': item['type'],
                        'metadata': {
                            **item['metadata'],
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'original_id': item['id']
                        }
                    }
                    chunked_items.append(chunked_item)
        
        return chunked_items
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + config.CHUNK_SIZE
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to find a good breaking point (sentence end, paragraph, etc.)
            chunk_text = text[start:end]
            
            # Look for sentence boundaries
            for delimiter in ['. ', '.\n', '!\n', '?\n']:
                last_delim = chunk_text.rfind(delimiter)
                if last_delim > config.CHUNK_SIZE * 0.7:  # At least 70% of chunk size
                    end = start + last_delim + len(delimiter)
                    break
            
            chunks.append(text[start:end])
            start = end - config.CHUNK_OVERLAP
        
        return chunks